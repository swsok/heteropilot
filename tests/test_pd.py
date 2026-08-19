"""Prefill/Decode-split candidates: generation, routing, compilation, KV cost.

Phase 5 increment 1 (docs/phase5_plan.md). P/D is a planner-only feature here:
the generator emits `ServingArch.PD_SPLIT` candidates and the existing
compile+sim path scores them. These tests pin the structural guarantees and,
critically, that the oracle-agreement invariant still holds with P/D candidates
in the search space.
"""

from __future__ import annotations

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    Accelerator,
    AcceleratorProfile,
    AcceleratorType,
    ClusterSpecV2,
    Link,
    LinkType,
    Node,
    SupportedModel,
    detect_islands,
)
from planner.optimizer import exhaustive
from planner.plan import Role, RoutingPolicy, ServingArch
from planner.predictor.llmservingsim import compile_to_sim_config
from planner.topology import TopologyGraph
from planner.util.kv_transfer import _kv_bytes_per_token, kv_transfer_cost

from .conftest import MockPredictor


def _pd(spec, cluster, islands, profiles):
    res = CandidateGenerator(spec, cluster, islands, profiles, enable_pd=True).generate()
    return [c for c in res.candidates if c.serving_arch is ServingArch.PD_SPLIT]


def _slow_prefill_fast_decode_cluster():
    """Synthetic simulator-only cluster: one memory-starved island whose decode
    roofline blows the TPOT budget, and one fast island that meets it.

    source=placeholder throughout - no hardware here was measured (absolute rule
    3). Both islands are single-device (tp=1), so only the stage-5 memory
    roofline is in play, not the stage-4 all-reduce. The slow island fits the
    model in memory (so it is a valid prefill engine) but its decode-step
    roofline far exceeds any sane TPOT, which is exactly the case that must not
    prune the P/D candidate that uses it for prefill only.
    """
    def _node(node_id: str, model: str) -> Node:
        return Node(
            id=node_id,
            accelerators=[
                Accelerator(
                    id="gpu0", type=AcceleratorType.GPU, vendor="synthetic",
                    model=model, backend="cuda", memory_gb=80.0,
                )
            ],
        )

    cluster = ClusterSpecV2(
        cluster_id="pd-invariant-synthetic",
        nodes=[_node("nodeslow", "SLOW-GPU"), _node("nodefast", "FAST-GPU")],
    )
    supported = [SupportedModel(pattern="*", dtypes=["bfloat16"])]
    profiles = {
        "SLOW-GPU": AcceleratorProfile(
            profile_id="slow", vendor="synthetic", model="SLOW-GPU", backend="cuda",
            memory_gb=80.0, memory_bandwidth_gbps=5.0, sim_hardware="SLOW",
            supported_models=supported, max_tp_size=1,
        ),
        "FAST-GPU": AcceleratorProfile(
            profile_id="fast", vendor="synthetic", model="FAST-GPU", backend="cuda",
            memory_gb=80.0, memory_bandwidth_gbps=3000.0, sim_hardware="FAST",
            supported_models=supported, max_tp_size=1,
        ),
    }
    islands = detect_islands(cluster, profiles)
    return cluster, islands, profiles


# --- enumeration ----------------------------------------------------------


def test_pd_candidates_are_generated_when_enabled(spec, cluster, islands, profiles) -> None:
    pd = _pd(spec, cluster, islands, profiles)
    assert pd, "two compatible islands must produce cross-island P/D candidates"


def test_pd_candidates_have_one_prefill_and_one_decode(
    spec, cluster, islands, profiles
) -> None:
    for cand in _pd(spec, cluster, islands, profiles):
        roles = sorted(a.role for a in cand.assignments)
        assert roles == [Role.DECODE, Role.PREFILL]
        # cross-island only: prefill and decode must sit on different islands
        assert cand.assignments[0].island_id != cand.assignments[1].island_id


def test_pd_enumerates_both_directions(spec, cluster, islands, profiles) -> None:
    """Prefill-on-A/decode-on-B is a different deployment from the reverse, so
    both orderings must appear."""
    directed = {
        (
            next(a.island_id for a in c.assignments if a.role is Role.PREFILL),
            next(a.island_id for a in c.assignments if a.role is Role.DECODE),
        )
        for c in _pd(spec, cluster, islands, profiles)
    }
    a, b = "cuda-rtx-a5000-node0", "cuda-rtxpro6000-node1"
    assert (a, b) in directed and (b, a) in directed


def test_pd_uniform_devices_per_replica(spec, cluster, islands, profiles) -> None:
    """D14: the simulator mis-scopes collectives for unequal instance sizes, so
    P/D pairs must be uniform in devices-per-replica - same rule as mixed."""
    for cand in _pd(spec, cluster, islands, profiles):
        assert len({a.devices_per_replica for a in cand.assignments}) == 1


def test_enable_pd_false_produces_no_pd_candidates(spec, cluster, islands, profiles) -> None:
    """Default OFF: Phase 2 behaviour and frozen outputs must be unchanged."""
    res = CandidateGenerator(spec, cluster, islands, profiles).generate()
    assert all(c.serving_arch is ServingArch.AGGREGATED for c in res.candidates)


def test_pd_is_purely_additive(spec, cluster, islands, profiles) -> None:
    """Enabling P/D must not remove or alter any non-P/D candidate."""
    base = CandidateGenerator(spec, cluster, islands, profiles).generate()
    withpd = CandidateGenerator(spec, cluster, islands, profiles, enable_pd=True).generate()
    non_pd = [c.id for c in withpd.candidates if c.serving_arch is ServingArch.AGGREGATED]
    assert non_pd == [c.id for c in base.candidates]


def test_pd_generation_is_deterministic(spec, cluster, islands, profiles) -> None:
    runs = [
        [c.id for c in CandidateGenerator(
            spec, cluster, islands, profiles, enable_pd=True).generate().candidates]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# --- oracle agreement (THE key test) --------------------------------------


def test_oracle_agreement_holds_with_pd_candidates(spec, cluster, islands, profiles) -> None:
    """§9 with P/D in the space: the pruned optimum must equal the exhaustive
    optimum. If a bound ever removes an achievable P/D configuration this fails.
    """
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


def test_slow_prefill_fast_decode_pd_is_not_over_pruned(spec) -> None:
    """Invariant defence for the prefill role (§9, HIGH review finding).

    The prefill island's *decode* roofline is far above the TPOT budget, but it
    is a valid prefill engine. Charging it a decode TPOT floor would prune the
    P/D candidate that uses it for prefill only - a pruning stage that is
    stricter than feasibility, which oracle mode does not apply. So the P/D
    candidate (slow-prefill + fast-decode) must appear in BOTH the pruned and the
    exhaustive candidate lists. Before the role-aware bounds it was present only
    in oracle mode, and this test would fail.
    """
    cluster, islands, profiles = _slow_prefill_fast_decode_cluster()

    def pd_ids(enable_bound_pruning: bool) -> set[str]:
        gen = CandidateGenerator(
            spec, cluster, islands, profiles,
            enable_pd=True, enable_bound_pruning=enable_bound_pruning,
        ).generate()
        return {
            c.id for c in gen.candidates
            if c.serving_arch is ServingArch.PD_SPLIT
            and c.assignments[0].island_id == "cuda-slow-gpu-nodeslow"
            and c.assignments[1].island_id == "cuda-fast-gpu-nodefast"
        }

    pruned = pd_ids(enable_bound_pruning=True)
    oracle = pd_ids(enable_bound_pruning=False)
    assert oracle, "the fixture must produce a slow-prefill/fast-decode candidate"
    assert pruned == oracle, (
        "the slow-prefill/fast-decode P/D candidate was pruned in bound mode but "
        "survived oracle mode: the prefill role is being charged a decode bound"
    )


def test_oracle_evaluates_at_least_as_many_with_pd(spec, cluster, islands, profiles) -> None:
    pruned_pred, oracle_pred = MockPredictor(), MockPredictor()
    exhaustive.search(spec, cluster, islands, profiles, pruned_pred, enable_pd=True)
    exhaustive.oracle(spec, cluster, islands, profiles, oracle_pred, enable_pd=True)
    assert len(oracle_pred.calls) >= len(pruned_pred.calls)


def test_pd_search_is_reproducible(spec, cluster, islands, profiles) -> None:
    runs = [
        exhaustive.search(
            spec, cluster, islands, profiles, MockPredictor(), enable_pd=True
        ).model_dump(mode="json", exclude={"provenance"})
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# --- routing --------------------------------------------------------------


def test_pd_plans_use_pd_split_routing(spec, cluster, islands, profiles) -> None:
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), enable_pd=True
    )
    everything = ([output.recommended] if output.recommended else []) + output.alternatives
    seen = [s.plan for s in everything if s.plan.candidate.serving_arch is ServingArch.PD_SPLIT]
    unscored = [u.plan for u in output.unscored
                if u.plan.candidate.serving_arch is ServingArch.PD_SPLIT]
    assert seen or unscored, "a P/D plan should survive the mock search"
    for plan in seen + unscored:
        assert plan.routing is RoutingPolicy.PD_SPLIT


def test_aggregated_routing_is_unchanged(spec, cluster, islands, profiles) -> None:
    """P/D routing must not leak into non-P/D plans."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    everything = ([output.recommended] if output.recommended else []) + output.alternatives
    for s in everything:
        assert s.plan.routing is not RoutingPolicy.PD_SPLIT


# --- compilation ----------------------------------------------------------


def test_pd_compiles_to_two_pd_typed_instances(spec, cluster, islands, profiles) -> None:
    by_id = {i.id: i for i in islands}
    cand = next(
        c for c in _pd(spec, cluster, islands, profiles)
        if all(a.dp_replicas == 1 for a in c.assignments)
    )
    config, _ = compile_to_sim_config(
        cand, cluster, by_id, profiles, topology=TopologyGraph(cluster)
    )
    instances = [inst for node in config["nodes"] for inst in node["instances"]]
    assert len(instances) == 2
    assert sorted(inst["pd_type"] for inst in instances) == ["decode", "prefill"]


# --- KV-transfer estimator ------------------------------------------------


def test_kv_transfer_single_hop_matches_hand_computation() -> None:
    model, dtype = "meta-llama/Llama-3.1-8B", "bfloat16"
    kvpt = _kv_bytes_per_token(model, dtype, "auto")
    link = Link(
        id="l", src="n0/gpu0", dst="n1/gpu0", type=LinkType.ETHERNET,
        bandwidth_gbps=100.0, latency_ns=20000.0, energy_per_bit_pj=5.0,
    )
    time_ms, energy_j = kv_transfer_cost(model, dtype, 1000, [link])
    kv_bytes = kvpt * 1000
    expected_time = 20000.0 / 1e6 + kv_bytes / (100.0 * 1e9) * 1e3
    expected_energy = 5.0 * kv_bytes * 8 * 1e-12
    assert abs(time_ms - expected_time) < 1e-9
    assert abs(energy_j - expected_energy) < 1e-12


def test_kv_transfer_zero_energy_link_contributes_no_energy() -> None:
    link = Link(
        id="l", src="n0/gpu0", dst="n1/gpu0", type=LinkType.ETHERNET,
        bandwidth_gbps=100.0, latency_ns=1000.0,
    )
    time_ms, energy_j = kv_transfer_cost("meta-llama/Llama-3.1-8B", "bfloat16", 10, [link])
    assert time_ms > 0.0
    assert energy_j == 0.0


def test_kv_transfer_empty_path_is_free() -> None:
    time_ms, energy_j = kv_transfer_cost("meta-llama/Llama-3.1-8B", "bfloat16", 10, [])
    assert time_ms == 0.0
    assert energy_j == 0.0
