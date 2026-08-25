"""Stage-6 surrogate top-K (work order §5.4).

Fast tests (MockPredictor) for the surrogate ranker, the opt-in top-K wiring in
`exhaustive.search`, oracle immunity, and the accuracy-measurement contract. The
end-to-end recall/regret curve on the real simulator is
experiments/scripts/exp_surrogate.py.

Pinned:
- surrogate-off is byte-identical (default path unchanged);
- top-K simulates only K, records SURROGATE_PRUNED for the rest, keeps a subset of
  the generated candidates, and is reproducible;
- the exhaustive oracle IGNORES a surrogate/top_k passed to it (evaluates all);
- top-K can drop the optimum (that is surrogate error, measured, not a bug);
- the CLI rejects --oracle together with --top-k.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scripts"))

import exp_surrogate as xs

from planner.__main__ import cmd_plan
from planner.optimizer import exhaustive, pareto
from planner.optimizer.surrogate import AnalyticalRooflineRanker


def _search(spec, cluster, islands, profiles, mock, **kw):
    return exhaustive.search(spec, cluster, islands, profiles, mock, enable_pd=True, **kw)


def test_ranker_is_deterministic(spec, cluster, islands, profiles):
    islands_by_id = {i.id: i for i in islands}
    from planner.candidate_generator import CandidateGenerator
    cands = CandidateGenerator(spec, cluster, islands, profiles,
                               enable_prefix_caching=False, enable_pd=True).generate().candidates
    r = AnalyticalRooflineRanker()
    a = [c.id for c in r.order(cands, spec, islands_by_id, profiles)]
    b = [c.id for c in r.order(cands, spec, islands_by_id, profiles)]
    assert a == b and sorted(a) == sorted(c.id for c in cands)  # permutation, deterministic


def test_surrogate_off_is_byte_identical(spec, cluster, islands, profiles, mock_predictor):
    base = _search(spec, cluster, islands, profiles, mock_predictor)
    off = _search(spec, cluster, islands, profiles, mock_predictor, surrogate=None, top_k=None)
    assert base.model_dump(exclude={"provenance"}) == off.model_dump(exclude={"provenance"})
    assert "surrogate_pruned" not in base.rejected_summary


def test_topk_simulates_only_k_and_records_stage(spec, cluster, islands, profiles, mock_predictor):
    full = _search(spec, cluster, islands, profiles, mock_predictor)
    k = 3
    tk = _search(spec, cluster, islands, profiles, mock_predictor,
                 surrogate=AnalyticalRooflineRanker(), top_k=k)
    assert tk.generated_candidates == full.generated_candidates
    assert tk.evaluated_candidates == k
    assert tk.rejected_summary.get("surrogate_pruned") == full.generated_candidates - k
    assert any("surrogate top-K" in c or "Stage-6" in c for c in tk.caveats)


def test_topk_ge_n_is_a_noop(spec, cluster, islands, profiles, mock_predictor):
    full = _search(spec, cluster, islands, profiles, mock_predictor)
    big = _search(spec, cluster, islands, profiles, mock_predictor,
                  surrogate=AnalyticalRooflineRanker(), top_k=10_000)
    assert "surrogate_pruned" not in big.rejected_summary
    assert big.evaluated_candidates == full.evaluated_candidates


def test_oracle_ignores_surrogate(spec, cluster, islands, profiles, mock_predictor):
    full = _search(spec, cluster, islands, profiles, mock_predictor)
    orc = exhaustive.oracle(spec, cluster, islands, profiles, mock_predictor, enable_pd=True,
                            surrogate=AnalyticalRooflineRanker(), top_k=1)
    assert orc.evaluated_candidates == full.generated_candidates  # simulated everything
    assert "surrogate_pruned" not in orc.rejected_summary


def test_topk_is_reproducible(spec, cluster, islands, profiles, mock_predictor):
    a = _search(spec, cluster, islands, profiles, mock_predictor,
                surrogate=AnalyticalRooflineRanker(), top_k=4)
    b = _search(spec, cluster, islands, profiles, mock_predictor,
                surrogate=AnalyticalRooflineRanker(), top_k=4)
    assert a.model_dump(exclude={"provenance"}) == b.model_dump(exclude={"provenance"})


def test_topk_pick_never_beats_oracle(spec, cluster, islands, profiles, mock_predictor):
    """The top-K recommendation is scored on true metrics, so it can tie the oracle
    but never beat it (a beat would mean the oracle wasn't the argmax)."""
    full = _search(spec, cluster, islands, profiles, mock_predictor)
    tk = _search(spec, cluster, islands, profiles, mock_predictor,
                 surrogate=AnalyticalRooflineRanker(), top_k=2)
    if full.recommended and tk.recommended:
        assert tk.recommended.value <= full.recommended.value + 1e-9


def test_cli_rejects_oracle_with_topk():
    ns = argparse.Namespace(oracle=True, top_k=1)
    assert cmd_plan(ns) == 1  # returns before touching any spec/cluster


# --- driver accuracy-measurement contract ---------------------------------


def test_driver_recall_regret_at_k_equals_n(spec, cluster, islands, profiles, mock_predictor):
    """The honesty contract: K == N must give recall 1.0 and regret 0.0. Small-K
    accuracy is MEASURED by the driver, never asserted to be high here."""
    islands_by_id = {i.id: i for i in islands}
    from planner.candidate_generator import CandidateGenerator
    cands = [c for c in CandidateGenerator(spec, cluster, islands, profiles,
             enable_prefix_caching=False, enable_pd=True).generate().candidates
             if xs._simulatable(c, islands_by_id, profiles)]
    raw = {c.id: mock_predictor.predict(c, spec, cluster, islands_by_id, profiles) for c in cands}

    class _Replay:
        def predict(self, candidate, spec, cluster, islands, profiles):
            return raw[candidate.id]

    ev = exhaustive.evaluate_candidates(cands, spec, cluster, islands_by_id, profiles, _Replay())
    feasible = {p.candidate.id: p for p in ev.feasible_plans}
    oracle_best = xs._best(list(feasible.values()), spec)
    oracle_val = pareto.objective_value(oracle_best, spec.objective.primary)
    ordered_ids = [c.id for c in
                   AnalyticalRooflineRanker().order(cands, spec, islands_by_id, profiles)]

    # K = N: everything kept -> the oracle optimum survives, regret is exactly 0.
    top = set(ordered_ids)
    assert oracle_best.candidate.id in top
    surr_best = xs._best([feasible[i] for i in top if i in feasible], spec)
    assert abs(pareto.objective_value(surr_best, spec.objective.primary) - oracle_val) < 1e-9
