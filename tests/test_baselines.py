"""Baselines + ablation harness (work order §12).

The end-to-end experiment (experiments/scripts/exp_baselines.py) needs a real
simulator; these tests exercise the SELECTION logic and the greedy optimizer with
the deterministic MockPredictor, so the invariants are guarded without a live sim.

Pinned:
- greedy never beats the oracle (regret >= 0) and is deterministic;
- proposed's pick is feasible and its regret is >= 0 (typically 0);
- No-PD selects no PD_SPLIT candidate; No-Energy selects the max-goodput plan;
- the three N/A ablations are emitted as labeled n/a rows, never fabricated;
- placeholder-profile islands (Ascend stub) are excluded from the simulatable set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scripts"))

import exp_baselines as xb

from planner.candidate_generator import CandidateGenerator
from planner.optimizer import exhaustive, greedy, pareto
from planner.plan import ServingArch


def _harness_inputs(spec, cluster, islands, profiles, mock):
    islands_by_id = {i.id: i for i in islands}
    gen_all = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_bound_pruning=False, enable_pd=True,
    ).generate()
    gen_pruned = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_bound_pruning=True, enable_pd=True,
    ).generate()
    simulatable = [c for c in gen_all.candidates if xb._simulatable(c, islands_by_id, profiles)]
    raw = {c.id: mock.predict(c, spec, cluster, islands_by_id, profiles) for c in simulatable}

    class _Replay:
        def predict(self, candidate, spec, cluster, islands, profiles):
            return raw[candidate.id]

    evaluation = exhaustive.evaluate_candidates(
        simulatable, spec, cluster, islands_by_id, profiles, _Replay()
    )
    feasible = {p.candidate.id: p for p in evaluation.feasible_plans}
    every = dict(feasible)
    for p, _r in evaluation.infeasible_plans:
        every.setdefault(p.candidate.id, p)
    rows = xb.build_strategies(
        feasible, every, simulatable, {c.id for c in gen_pruned.candidates}, raw,
        spec, cluster, islands_by_id, profiles,
    )
    return simulatable, feasible, rows, islands_by_id


def _oracle_value(feasible, spec):
    best = xb._best(list(feasible.values()), spec)
    return pareto.objective_value(best, spec.objective.primary)


def test_placeholder_islands_excluded(spec, cluster, islands, profiles):
    """The Ascend stub (sim_hardware=None) must not appear in the simulatable set."""
    islands_by_id = {i.id: i for i in islands}
    gen = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False, enable_pd=True
    ).generate()
    sim = [c for c in gen.candidates if xb._simulatable(c, islands_by_id, profiles)]
    assert sim, "some GPU candidates must be simulatable"
    for c in sim:
        for a in c.assignments:
            assert profiles[islands_by_id[a.island_id].accelerator_model].sim_hardware is not None


def test_greedy_never_beats_oracle(spec, cluster, islands, profiles, mock_predictor):
    simulatable, feasible, _rows, islands_by_id = _harness_inputs(
        spec, cluster, islands, profiles, mock_predictor)
    pick = greedy.greedy(simulatable, spec, islands_by_id, profiles)
    assert pick is not None
    oracle_value = _oracle_value(feasible, spec)
    if pick.id in feasible:
        val = pareto.objective_value(feasible[pick.id], spec.objective.primary)
        assert val <= oracle_value + 1e-9  # greedy can tie the oracle, never beat it


def test_greedy_is_deterministic(spec, cluster, islands, profiles):
    islands_by_id = {i.id: i for i in islands}
    gen = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False, enable_pd=True
    ).generate()
    sim = [c for c in gen.candidates if xb._simulatable(c, islands_by_id, profiles)]
    runs = [[e.candidate_id for e in greedy.rank(sim, spec, islands_by_id, profiles)]
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_strategy_rows_and_regret(spec, cluster, islands, profiles, mock_predictor):
    _sim, feasible, rows, _ = _harness_inputs(spec, cluster, islands, profiles, mock_predictor)
    by_name = {r["name"]: r for r in rows}
    oracle_value = _oracle_value(feasible, spec)

    # proposed pick is feasible and never beats the oracle.
    prop = by_name["proposed"]
    assert prop["chosen_id"] in feasible
    prop_val = pareto.objective_value(feasible[prop["chosen_id"]], spec.objective.primary)
    assert prop_val <= oracle_value + 1e-9

    # oracle picks the argmax (regret 0 by construction).
    orc = by_name["exhaustive-oracle"]
    assert abs(pareto.objective_value(feasible[orc["chosen_id"]], spec.objective.primary)
               - oracle_value) < 1e-9


def test_no_pd_selects_no_pd(spec, cluster, islands, profiles, mock_predictor):
    sim, _feasible, rows, _islands = _harness_inputs(
        spec, cluster, islands, profiles, mock_predictor)
    by_name = {r["name"]: r for r in rows}
    by_cand = {c.id: c for c in sim}
    chosen = by_name["No-PD-Specialization"]["chosen_id"]
    if chosen is not None:
        assert by_cand[chosen].serving_arch is not ServingArch.PD_SPLIT


def test_no_energy_selects_max_goodput(spec, cluster, islands, profiles, mock_predictor):
    _sim, feasible, rows, _ = _harness_inputs(spec, cluster, islands, profiles, mock_predictor)
    by_name = {r["name"]: r for r in rows}
    chosen = by_name["No-Energy"]["chosen_id"]
    if chosen is not None and feasible:
        best_goodput = max(feasible.values(), key=lambda p: (xb._goodput(p), p.candidate.id))
        assert chosen == best_goodput.candidate.id


def test_na_ablations_are_labeled(spec, cluster, islands, profiles, mock_predictor):
    _sim, _feasible, rows, _ = _harness_inputs(spec, cluster, islands, profiles, mock_predictor)
    by_name = {r["name"]: r for r in rows}
    for na in ("No-Calibration", "No-Uncertainty", "Static"):
        assert by_name[na]["chosen_id"] is None
        assert "N/A" in by_name[na]["note"]


def test_strategies_are_deterministic(spec, cluster, islands, profiles, mock_predictor):
    r1 = _harness_inputs(spec, cluster, islands, profiles, mock_predictor)[2]
    r2 = _harness_inputs(spec, cluster, islands, profiles, mock_predictor)[2]
    assert [(r["name"], r["chosen_id"]) for r in r1] == [(r["name"], r["chosen_id"]) for r in r2]
