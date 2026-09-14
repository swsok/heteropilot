"""Instance -> island attribution (uncertainty work order §2.4.1 rev 2).

An accuracy domain is measured on one card, so a lookup needs PER-INSTANCE load.
`instance_island_ids` reproduces the order `compile_to_sim_config` emits
instances in; this pins the two against each other so they cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.plan import CandidateConfig, IslandAssignment, Role, ServingArch
from planner.predictor.llmservingsim import compile_to_sim_config, instance_island_ids
from planner.spec import load_service_spec
from planner.topology import TopologyGraph

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def world():
    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    cluster = load_cluster_spec(ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml")
    profiles = load_profiles_for(cluster, ROOT)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    return spec, cluster, profiles, islands


def _compiled_order(candidate, cluster, profiles, islands) -> list[dict]:
    config, _reduction = compile_to_sim_config(
        candidate, cluster, islands, profiles, topology=TopologyGraph(cluster),
    )
    return [inst for node in config["nodes"] for inst in node["instances"]]


def test_the_helper_matches_what_the_compiler_actually_emits(world) -> None:
    spec, cluster, profiles, islands = world
    a40s = [i for i in islands if i.startswith("cuda-a40")]
    candidate = CandidateConfig(
        id="c", model=spec.model, dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=a40s[0], tp_size=1, dp_replicas=3),
            IslandAssignment(island_id=a40s[1], tp_size=2, dp_replicas=2),
        ],
    )
    mapped = instance_island_ids(candidate, cluster, islands)
    emitted = _compiled_order(candidate, cluster, profiles, islands)

    assert len(mapped) == len(emitted) == 5, "3 replicas + 2 replicas"
    # Same hardware and device count, instance by instance.
    for island_id, instance in zip(mapped, emitted, strict=True):
        expected_devices = next(
            a.devices_per_replica for a in candidate.assignments
            if a.island_id == island_id
        )
        assert instance["num_npus"] == expected_devices
        assert instance["hardware"] == profiles[
            islands[island_id].accelerator_model
        ].sim_hardware


def test_a_pd_candidates_two_roles_map_to_their_own_islands(world) -> None:
    spec, cluster, profiles, islands = world
    rngd = next(i for i in islands if i.startswith("furiosa-rngd-node_rngd0"))
    a40 = next(i for i in islands if i.startswith("cuda-a40-node_a40a"))
    candidate = CandidateConfig(
        id="pd", model=spec.model, dtype="bfloat16",
        serving_arch=ServingArch.PD_SPLIT,
        assignments=[
            IslandAssignment(island_id=rngd, role=Role.PREFILL, tp_size=4),
            IslandAssignment(island_id=a40, role=Role.DECODE, tp_size=4),
        ],
    )
    mapped = instance_island_ids(candidate, cluster, islands)
    emitted = _compiled_order(candidate, cluster, profiles, islands)
    assert len(mapped) == 2
    roles = {island_id: inst["pd_type"] for island_id, inst in zip(mapped, emitted,
                                                                  strict=True)}
    assert roles[rngd] == "prefill"
    assert roles[a40] == "decode"


def test_a_replicated_candidate_is_judged_per_card_not_per_deployment() -> None:
    """The reason this exists: a 4-replica candidate is not running at 4x.

    Charging the whole deployment's in-flight count to each card would push an
    ordinary candidate outside every measured envelope - which is exactly what
    E-A1 saw before the rule changed. The margin policy reads the run's
    per-hardware operating point (`planner/util/operating_point.py`, busiest
    instance of each hardware kind), so the per-card figure is what it judges.
    """
    from planner.optimizer.margin import AccuracyDomainMargin
    from planner.plan import PredictedMetrics
    from planner.predictor import SimOutcome, SimResult
    from planner.predictor.calibration import AccuracyDomain

    domain = AccuracyDomain(
        fitted_at_concurrency=10.0,
        points=[{"conc": 10.0, "tpot_err_pct": -1.0}, {"conc": 60.0, "tpot_err_pct": -10.0}],
    )
    candidate = CandidateConfig(
        id="c", model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[IslandAssignment(island_id="i0", tp_size=1, dp_replicas=4)],
    )
    metrics = PredictedMetrics(
        p50_ttft_ms=1.0, p95_ttft_ms=1.0, p99_ttft_ms=1.0,
        p50_tpot_ms=1.0, p95_tpot_ms=1.0, p99_tpot_ms=1.0,
        throughput_tps=1.0, slo_goodput_rps=1.0, slo_attainment=1.0,
        completed_requests=1, completed_tokens=1,
        # 160 in flight across the deployment, 40 on the busiest card.
        served_concurrency=40.0,
        served_concurrency_per_island={"i0": 40.0},
    )
    sim = SimResult(
        "c", SimOutcome.OK, metrics=metrics,
        operating_point={"A40": {"concurrency": 40.0, "phase": "total",
                                 "requests": 160, "wall_s": 10.0}},
    )
    decision = AccuracyDomainMargin({"A40": domain}).decide(candidate, sim, metrics, {"i0": "A40"})
    assert decision.status == "in_domain", "40 per card is inside [10, 60]"
    # error -(1.0 + 30/50 * 9.0) = -6.4 %, margin -e/(1+e) = 6.4/93.6 (D70)
    error = 1.0 + 30 / 50 * 9.0
    assert decision.tpot_percent == pytest.approx(error / (100.0 - error) * 100.0)
