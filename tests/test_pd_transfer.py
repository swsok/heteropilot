"""Planner-side P/D KV-transfer cost (Phase 5 increment 2, docs/phase5_plan.md).

The simulator models the prefill->decode handoff as free, so HeteroPilot prices
it in `apply_pd_transfer_cost` (planner/optimizer/exhaustive.py), applied inside
`evaluate_candidates` after prediction and before feasibility. These tests pin:

- the adjustment is exact (TTFT += xfer_ms, energy += per-request energy, TPOT and
  non-P/D candidates untouched);
- oracle-agreement still holds with a *non-zero* transfer in the space;
- the transfer is network-sensitive at the planner level (TTFT rises as the
  inter-island bandwidth drops) - the increment-4 sweep, reproducible planner-side;
- the transfer can flip the recommendation (§5.9 adoption crossing);
- a disconnected island pair is charged zero and records the assumption;
- the honesty caveat travels with any P/D output and is absent otherwise.

Every synthetic cluster here is source=placeholder - no hardware was measured
(absolute rule 3).
"""

from __future__ import annotations

import itertools

import pytest

from planner.inventory import (
    Accelerator,
    AcceleratorProfile,
    AcceleratorType,
    ClusterSpecV2,
    Link,
    LinkType,
    Node,
    NodePower,
    SupportedModel,
    detect_islands,
)
from planner.optimizer import exhaustive, pareto
from planner.optimizer.exhaustive import apply_pd_transfer_cost
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    PredictedMetrics,
    Role,
    ServingArch,
)
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.spec import ServiceSpec
from planner.topology import TopologyGraph
from planner.util.kv_transfer import kv_transfer_cost

from .conftest import MockPredictor

MODEL = "meta-llama/Llama-3.1-8B"


def _connected_pd_cluster(
    bw_gbps: float,
    *,
    latency_ns: float = 1000.0,
    energy_per_bit_pj: float | None = 5.0,
    powered: bool = True,
) -> tuple[ClusterSpecV2, list, dict[str, AcceleratorProfile]]:
    """Two single-GPU islands joined by one tunable inter-island link.

    Both GPUs sit at the island's representative endpoint (`node/gpu0`), so the
    prefill->decode path is exactly the one declared link, which lets a test dial
    the transfer cost by bandwidth alone.
    """
    def _node(node_id: str) -> Node:
        return Node(
            id=node_id,
            accelerators=[
                Accelerator(
                    id="gpu0", type=AcceleratorType.GPU, vendor="synthetic",
                    model="PD-GPU", backend="cuda", memory_gb=80.0,
                )
            ],
            power=NodePower() if powered else None,
        )

    cluster = ClusterSpecV2(
        cluster_id="pd-transfer-synthetic",
        nodes=[_node("nodea"), _node("nodeb")],
        links=[
            Link(
                id="fabric", src="nodea/gpu0", dst="nodeb/gpu0",
                type=LinkType.ETHERNET, bandwidth_gbps=bw_gbps,
                latency_ns=latency_ns, energy_per_bit_pj=energy_per_bit_pj,
            )
        ],
    )
    supported = [SupportedModel(pattern="*", dtypes=["bfloat16"])]
    profiles = {
        "PD-GPU": AcceleratorProfile(
            profile_id="pd-gpu", vendor="synthetic", model="PD-GPU", backend="cuda",
            memory_gb=80.0, memory_bandwidth_gbps=3000.0, sim_hardware="PDGPU",
            supported_models=supported, max_tp_size=1,
        ),
    }
    islands = detect_islands(cluster, profiles)
    return cluster, islands, profiles


def _pd_candidate(islands: list) -> CandidateConfig:
    by_node = {i.node_id: i for i in islands}
    a, b = by_node["nodea"], by_node["nodeb"]
    return CandidateConfig(
        id="pd-a-b", model=MODEL, dtype="bfloat16",
        serving_arch=ServingArch.PD_SPLIT,
        assignments=[
            IslandAssignment(island_id=a.id, role=Role.PREFILL, tp_size=1),
            IslandAssignment(island_id=b.id, role=Role.DECODE, tp_size=1),
        ],
    )


def _agg_candidate(islands: list) -> CandidateConfig:
    a = {i.node_id: i for i in islands}["nodea"]
    return CandidateConfig(
        id="agg-a", model=MODEL, dtype="bfloat16",
        serving_arch=ServingArch.AGGREGATED,
        assignments=[IslandAssignment(island_id=a.id, role=Role.AGGREGATED, tp_size=1)],
    )


def _base_metrics(**over) -> PredictedMetrics:
    base: dict = {
        "p50_ttft_ms": 100.0, "p95_ttft_ms": 150.0, "p99_ttft_ms": 200.0,
        "p50_tpot_ms": 8.0, "p95_tpot_ms": 9.0, "p99_tpot_ms": 10.0,
        "throughput_tps": 1000.0, "slo_goodput_rps": 10.0, "slo_attainment": 1.0,
        "completed_requests": 100, "completed_tokens": 20_000,
        "total_energy_j": 5000.0, "average_power_w": 500.0, "peak_power_w": 600.0,
        "tokens_per_joule": 4.0, "sim_wall_seconds": 0.0,
    }
    base.update(over)
    return PredictedMetrics(**base)


# --- exact adjustment -----------------------------------------------------


def _xfer(spec, path, prompt_tokens):
    ms, energy = kv_transfer_cost(
        spec.model, spec.service.dtype, prompt_tokens, path,
        kv_cache_dtype=spec.service.kv_cache_dtype,
    )
    return ms, energy


def test_pd_ttft_rises_by_the_matching_percentile_transfer_ms(spec) -> None:
    cluster, islands, _ = _connected_pd_cluster(100.0)
    by_id = {i.id: i for i in islands}
    topo = TopologyGraph(cluster)
    cand = _pd_candidate(islands)
    raw = _base_metrics()

    adjusted, info = apply_pd_transfer_cost(cand, raw, spec, cluster, by_id, topo)

    path = topo.path("nodea/gpu0", "nodeb/gpu0")
    tok = spec.traffic.input_tokens
    xfer_p50, energy_j = _xfer(spec, path, tok.p50)  # energy uses the p50 prompt
    xfer_p95, _ = _xfer(spec, path, tok.p95 if tok.p95 is not None else tok.p50)
    xfer_p99, _ = _xfer(spec, path, tok.p99 if tok.p99 is not None else tok.p50)
    assert xfer_p50 > 0.0
    # Each percentile shifts by the transfer time of its own prompt length.
    assert adjusted.p50_ttft_ms == pytest.approx(raw.p50_ttft_ms + xfer_p50)
    assert adjusted.p95_ttft_ms == pytest.approx(raw.p95_ttft_ms + xfer_p95)
    assert adjusted.p99_ttft_ms == pytest.approx(raw.p99_ttft_ms + xfer_p99)
    # TPOT untouched: the transfer is one-time, pre-decode.
    assert adjusted.p50_tpot_ms == raw.p50_tpot_ms
    assert adjusted.p99_tpot_ms == raw.p99_tpot_ms
    # Energy: per-request transfer energy (p50/median prompt) times the request count.
    expected_energy = raw.total_energy_j + energy_j * raw.completed_requests
    assert adjusted.total_energy_j == pytest.approx(expected_energy)
    assert adjusted.tokens_per_joule == pytest.approx(raw.completed_tokens / expected_energy)
    assert info["xfer_ms_p50"] == pytest.approx(xfer_p50)
    assert info["xfer_ms_p95"] == pytest.approx(xfer_p95)
    assert info["xfer_ms_p99"] == pytest.approx(xfer_p99)
    assert info["energy_j_per_req"] == pytest.approx(energy_j)
    assert info["prompt_tokens"]["p50"] == tok.p50


def test_heavy_tailed_prompt_makes_p99_offset_exceed_p50(spec) -> None:
    """A skewed prompt distribution (p99 >> p50) must give the p99 TTFT a strictly
    larger transfer offset than p50 - the whole point of per-percentile pricing."""
    raw = spec.model_dump()
    raw["traffic"]["input_tokens"] = {"p50": 100, "p95": 2000, "p99": 8000}
    skewed = ServiceSpec.model_validate(raw)
    cluster, islands, _ = _connected_pd_cluster(10.0)
    by_id = {i.id: i for i in islands}
    _, info = apply_pd_transfer_cost(
        _pd_candidate(islands), _base_metrics(), skewed, cluster, by_id,
        TopologyGraph(cluster),
    )
    assert info["xfer_ms_p50"] < info["xfer_ms_p95"] < info["xfer_ms_p99"]


def test_non_pd_candidate_is_byte_identical(spec) -> None:
    cluster, islands, _ = _connected_pd_cluster(100.0)
    by_id = {i.id: i for i in islands}
    topo = TopologyGraph(cluster)
    raw = _base_metrics()
    adjusted, info = apply_pd_transfer_cost(
        _agg_candidate(islands), raw, spec, cluster, by_id, topo
    )
    assert adjusted is raw
    assert info == {}


def test_energyless_metrics_leave_energy_none(spec) -> None:
    """A plan whose sim produced no energy stays energy-free after the adjustment."""
    cluster, islands, _ = _connected_pd_cluster(100.0)
    by_id = {i.id: i for i in islands}
    topo = TopologyGraph(cluster)
    raw = _base_metrics(total_energy_j=None, average_power_w=None,
                        peak_power_w=None, tokens_per_joule=None)
    adjusted, _ = apply_pd_transfer_cost(
        _pd_candidate(islands), raw, spec, cluster, by_id, topo
    )
    assert adjusted.total_energy_j is None
    assert adjusted.tokens_per_joule is None
    assert adjusted.p99_ttft_ms > raw.p99_ttft_ms  # TTFT still shifts


# --- network sweep (increment-4 headline, planner-side) -------------------


def test_pd_ttft_rises_monotonically_as_bandwidth_drops(spec) -> None:
    raw = _base_metrics()
    ttfts = []
    for bw in (1000.0, 100.0, 10.0, 1.0):
        cluster, islands, _ = _connected_pd_cluster(bw)
        by_id = {i.id: i for i in islands}
        adjusted, _ = apply_pd_transfer_cost(
            _pd_candidate(islands), raw, spec, cluster, by_id, TopologyGraph(cluster)
        )
        ttfts.append(adjusted.p99_ttft_ms)
    # Strictly increasing as the fabric slows: the transfer term dominates.
    assert all(lo < hi for lo, hi in itertools.pairwise(ttfts)), ttfts
    assert ttfts[-1] > ttfts[0]


# --- adoption crossing (§5.9) ---------------------------------------------


class _StubPredictor(Predictor):
    """Hand-set raw metrics per serving arch, so the ranking flip is controlled.

    The MockPredictor's TTFT never feeds the objective, only feasibility; the
    crossing is therefore a feasibility flip. This stub gives the P/D candidate
    strictly lower energy (so it wins under minimize_energy *when feasible*) but a
    raw TTFT that the transfer cost can push over the SLO at low bandwidth.
    """

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        if candidate.serving_arch is ServingArch.PD_SPLIT:
            m = _base_metrics(p50_ttft_ms=60.0, p95_ttft_ms=80.0, p99_ttft_ms=100.0,
                              total_energy_j=3000.0, tokens_per_joule=20_000 / 3000.0)
        else:
            m = _base_metrics(p50_ttft_ms=120.0, p95_ttft_ms=170.0, p99_ttft_ms=200.0,
                              total_energy_j=5000.0, tokens_per_joule=20_000 / 5000.0)
        return SimResult(candidate.id, SimOutcome.OK, metrics=m)


def _tight_ttft_spec(spec: ServiceSpec) -> ServiceSpec:
    raw = spec.model_dump()
    raw["slo"]["ttft"] = {"percentile": 99, "max_ms": 300.0}
    raw["objective"] = {"primary": "minimize_energy", "secondary": "minimize_active_accelerators"}
    return ServiceSpec.model_validate(raw)


def _recommended_arch(bw_gbps: float, spec: ServiceSpec) -> ServingArch | None:
    cluster, islands, profiles = _connected_pd_cluster(bw_gbps)
    by_id = {i.id: i for i in islands}
    cands = [_pd_candidate(islands), _agg_candidate(islands)]
    res = exhaustive.evaluate_candidates(
        cands, spec, cluster, by_id, profiles, _StubPredictor()
    )
    if not res.feasible_plans:
        return None
    ranked = pareto.rank(
        res.feasible_plans, spec.objective.primary, spec.objective.secondary
    )
    return ranked[0].plan.candidate.serving_arch


def test_transfer_cost_flips_the_recommendation(spec) -> None:
    """High fabric bandwidth: the cheaper P/D plan wins. Low bandwidth: the
    transfer pushes P/D's TTFT past the SLO, so the aggregated plan wins."""
    tight = _tight_ttft_spec(spec)
    assert _recommended_arch(1000.0, tight) is ServingArch.PD_SPLIT
    assert _recommended_arch(0.1, tight) is ServingArch.AGGREGATED


def test_pd_becomes_infeasible_when_transfer_blows_the_ttft_slo(spec) -> None:
    """The transfer must move P/D from feasible to infeasible, via evaluate."""
    tight = _tight_ttft_spec(spec)
    cluster_hi, islands_hi, profiles = _connected_pd_cluster(1000.0)
    cluster_lo, islands_lo, _ = _connected_pd_cluster(0.1)

    def pd_feasible(cluster, islands) -> bool:
        by_id = {i.id: i for i in islands}
        res = exhaustive.evaluate_candidates(
            [_pd_candidate(islands)], tight, cluster, by_id, profiles, _StubPredictor()
        )
        return any(
            p.candidate.serving_arch is ServingArch.PD_SPLIT for p in res.feasible_plans
        )

    assert pd_feasible(cluster_hi, islands_hi)
    assert not pd_feasible(cluster_lo, islands_lo)


# --- oracle agreement with a non-zero transfer ----------------------------


def test_oracle_agreement_holds_with_nonzero_transfer(spec) -> None:
    """§9 with a P/D transfer that actually costs something (the default
    heterogeneous-lab cluster has no fabric link to the A5000s, so its transfer is
    zero; this cluster's is not). The adjustment is applied identically in oracle
    and pruned modes, so the optimum must still match."""
    cluster, islands, profiles = _connected_pd_cluster(50.0)
    pruned = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(),
        enable_pd=True, enable_bound_pruning=True,
    )
    oracle = exhaustive.oracle(
        spec, cluster, islands, profiles, MockPredictor(), enable_pd=True
    )
    assert pruned.feasible == oracle.feasible
    assert pruned.recommended is not None and oracle.recommended is not None
    assert pruned.recommended.plan.candidate.id == oracle.recommended.plan.candidate.id
    assert pruned.recommended.value == oracle.recommended.value


# --- no-path handling -----------------------------------------------------


def test_disconnected_islands_are_charged_class_default_not_zero(spec) -> None:
    """A40x8-style: two islands with no declared link. The transfer is charged the
    interconnect CLASS DEFAULT (consistent with the sim's compiler), never an
    optimistic 0, and the class-default assumption is recorded."""
    from planner.topology import CLASS_DEFAULT_GBPS, LinkType

    cluster, islands, _ = _connected_pd_cluster(100.0)
    # Drop the fabric link so the two islands are disconnected.
    cluster = cluster.model_copy(update={"links": []})
    by_id = {i.id: i for i in islands}
    topo = TopologyGraph(cluster)
    raw = _base_metrics()
    adjusted, info = apply_pd_transfer_cost(
        _pd_candidate(islands), raw, spec, cluster, by_id, topo
    )
    # Non-zero: the transfer is priced on the class-default bandwidth.
    assert adjusted.p99_ttft_ms > raw.p99_ttft_ms
    assert info["xfer_ms_p99"] > 0.0
    assert info["class_default"] is True
    # No priced link carries energy, so the energy term is 0 (recorded as such).
    assert info["energy_j_per_req"] == 0.0
    assert adjusted.total_energy_j == raw.total_energy_j
    # Consistent with what reduce_for_simulator would compile for this pair.
    assert info["path_bw_gbps"] == CLASS_DEFAULT_GBPS[LinkType.INFINIBAND]
    reduction = topo.reduce_for_simulator(islands)
    assert reduction.link_bw_gbps == info["path_bw_gbps"]
    assert info["assumptions"]
    assert "not connected" in info["assumptions"][0]


def test_no_path_does_not_crash_full_search(spec) -> None:
    """On the default cluster the A5000 island has no fabric link, so cross-island
    P/D hits the no-path branch. The whole search must still complete."""
    cluster, islands, _ = _connected_pd_cluster(100.0)
    cluster = cluster.model_copy(update={"links": []})
    out = exhaustive.search(
        spec, cluster, islands, {"PD-GPU": _connected_pd_cluster(100.0)[2]["PD-GPU"]},
        MockPredictor(), enable_pd=True,
    )
    assert out is not None  # completed without raising


# --- honesty caveat -------------------------------------------------------


def test_pd_output_carries_the_transfer_caveat(spec) -> None:
    cluster, islands, profiles = _connected_pd_cluster(50.0)
    out = exhaustive.search(spec, cluster, islands, profiles, MockPredictor(), enable_pd=True)
    assert any("KV-transfer cost is a planner-side" in c for c in out.caveats)
    assert "pd_transfer" in out.provenance
    assert out.provenance["pd_transfer"]["candidates"]


def test_non_pd_output_has_no_transfer_caveat(spec) -> None:
    cluster, islands, profiles = _connected_pd_cluster(50.0)
    out = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert not any("KV-transfer cost is a planner-side" in c for c in out.caveats)
    assert "pd_transfer" not in out.provenance


def test_recommended_class_default_pd_gets_prominent_caveat(spec) -> None:
    """When the recommendation's transfer was priced on a class-default (the
    islands have no declared link), the caveat is present and prominent (first)."""
    cluster, islands, profiles = _connected_pd_cluster(100.0)
    cluster = cluster.model_copy(update={"links": []})  # disconnect the islands
    out = exhaustive.search(
        spec, cluster, islands, profiles, _StubPredictor(), enable_pd=True
    )
    assert out.recommended is not None
    assert out.recommended.plan.candidate.serving_arch is ServingArch.PD_SPLIT
    assert out.caveats[0].startswith("The recommended P/D plan's prefill and decode")


def test_connected_recommended_pd_has_no_class_default_caveat(spec) -> None:
    """A P/D recommendation over a real declared link must not carry the
    class-default caveat (only the ordinary planner-side add-on caveat)."""
    cluster, islands, profiles = _connected_pd_cluster(1000.0)
    out = exhaustive.search(
        spec, cluster, islands, profiles, _StubPredictor(), enable_pd=True
    )
    assert out.recommended is not None
    assert out.recommended.plan.candidate.serving_arch is ServingArch.PD_SPLIT
    assert not any(c.startswith("The recommended P/D plan's prefill") for c in out.caveats)
    assert any("KV-transfer cost is a planner-side" in c for c in out.caveats)
