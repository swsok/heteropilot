"""Mixed (cross-island) replica candidates: generation, compilation, honesty.

Work order §1.3: heterogeneity is exploited at replica granularity, never
inside a TP group. These tests pin the structural guarantees that make that
true, plus the D14 representational limit of the simulator.
"""

from __future__ import annotations

from planner.candidate_generator import CandidateGenerator
from planner.optimizer import exhaustive
from planner.plan import RoutingPolicy
from planner.predictor.llmservingsim import compile_to_sim_config
from planner.topology import TopologyGraph

from .conftest import MockPredictor


def _mixed(spec, cluster, islands, profiles):
    res = CandidateGenerator(spec, cluster, islands, profiles).generate()
    return [c for c in res.candidates if len(c.assignments) > 1]


def test_mixed_candidates_are_generated(spec, cluster, islands, profiles) -> None:
    mixed = _mixed(spec, cluster, islands, profiles)
    assert mixed, "two compatible islands must produce cross-island candidates"
    ids = {frozenset(a.island_id for a in c.assignments) for c in mixed}
    assert frozenset({"cuda-rtx-a5000-node0", "cuda-rtxpro6000-node1"}) in ids


def test_tp_never_crosses_an_island(spec, cluster, islands, profiles) -> None:
    """Absolute rule 2, structurally: every assignment carries its own island
    and its own tp; there is no such thing as a TP group spanning two."""
    for cand in _mixed(spec, cluster, islands, profiles):
        assert len({a.island_id for a in cand.assignments}) == len(cand.assignments)


def test_uniform_devices_per_replica(spec, cluster, islands, profiles) -> None:
    """D14: the simulator's topology inference (npus_per_group by integer
    division) mis-scopes collectives for unequal instance sizes, so unequal
    mixes must not be enumerated at all."""
    for cand in _mixed(spec, cluster, islands, profiles):
        assert len({a.devices_per_replica for a in cand.assignments}) == 1


def test_enable_mixed_false_restores_single_island_search(
    spec, cluster, islands, profiles
) -> None:
    res = CandidateGenerator(
        spec, cluster, islands, profiles, enable_mixed=False
    ).generate()
    assert all(len(c.assignments) == 1 for c in res.candidates)


def test_mixed_generation_is_deterministic(spec, cluster, islands, profiles) -> None:
    runs = [
        [c.id for c in CandidateGenerator(spec, cluster, islands, profiles)
         .generate().candidates]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_oracle_agreement_holds_with_mixed_candidates(
    spec, cluster, islands, profiles
) -> None:
    pruned = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    oracle = exhaustive.oracle(spec, cluster, islands, profiles, MockPredictor())
    assert pruned.feasible == oracle.feasible
    assert pruned.recommended is not None and oracle.recommended is not None
    assert pruned.recommended.plan.candidate.id == oracle.recommended.plan.candidate.id


def test_mixed_plans_use_load_routing(spec, cluster, islands, profiles) -> None:
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    everything = (
        ([output.recommended] if output.recommended else [])
        + output.alternatives
    )
    mixed_plans = [s.plan for s in everything if len(s.plan.candidate.assignments) > 1]
    unscored_mixed = [u.plan for u in output.unscored
                      if len(u.plan.candidate.assignments) > 1]
    assert mixed_plans or unscored_mixed, "a mixed plan should survive the mock search"
    for plan in mixed_plans + unscored_mixed:
        assert plan.routing is RoutingPolicy.LOAD


# --- compilation ------------------------------------------------------------


def _compile_first_mixed(spec, cluster, islands, profiles):
    by_id = {i.id: i for i in islands}
    cand = _mixed(spec, cluster, islands, profiles)[0]
    config, reduction = compile_to_sim_config(
        cand, cluster, by_id, profiles, topology=TopologyGraph(cluster)
    )
    return cand, config, reduction


def test_mixed_compiles_to_two_nodes(spec, cluster, islands, profiles) -> None:
    cand, config, _ = _compile_first_mixed(spec, cluster, islands, profiles)
    assert config["num_nodes"] == 2
    total_instances = sum(n["num_instances"] for n in config["nodes"])
    assert total_instances == sum(a.dp_replicas for a in cand.assignments)
    hardwares = {
        inst["hardware"] for node in config["nodes"] for inst in node["instances"]
    }
    assert hardwares == {"A5000", "RTXPRO6000"}
    # D14 precondition the simulator relies on:
    sizes = {
        inst["num_npus"] for node in config["nodes"] for inst in node["instances"]
    }
    assert len(sizes) == 1


def test_partial_power_coverage_is_detected(spec, cluster, islands, profiles) -> None:
    """node1 has a power block, node0 does not. Upstream disables power
    modeling wholesale in this case (config_builder.py:326), and the
    predictor's power_complete guard additionally refuses partial totals
    should that behavior ever change. This pins the fixture shape both rely
    on: a genuinely partially-covered deployment."""
    _, config, _ = _compile_first_mixed(spec, cluster, islands, profiles)
    powered = ["power" in n for n in config["nodes"]]
    assert any(powered) and not all(powered)


def test_mixed_energy_is_withheld_by_the_mock_too(
    spec, cluster, islands, profiles
) -> None:
    """The mock must mirror the real predictor's partial-coverage guard, or
    mixed plans would rank on energy the real pipeline cannot produce."""
    by_id = {i.id: i for i in islands}
    cand = _mixed(spec, cluster, islands, profiles)[0]
    sim = MockPredictor().predict(cand, spec, cluster, by_id, profiles)
    assert sim.ok and sim.metrics is not None
    assert sim.metrics.total_energy_j is None
