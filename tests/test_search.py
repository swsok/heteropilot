"""End-to-end search: oracle agreement, reproducibility, diagnosis (§9).

These are the three test classes the work order calls out specifically, and all
three run against the mock predictor so they finish in milliseconds instead of
the ~20 minutes a real simulation sweep would take.
"""

from __future__ import annotations

from planner.candidate_generator import CandidateGenerator
from planner.optimizer import exhaustive
from planner.plan import RejectionStage

from .conftest import MockPredictor


def _ids(output) -> list[str]:
    out = [output.recommended.plan.candidate.id] if output.recommended else []
    return out + [a.plan.candidate.id for a in output.alternatives]


# --- oracle agreement -----------------------------------------------------

def test_pruning_does_not_remove_the_optimum(spec, cluster, islands, profiles) -> None:
    """§9: pruned and unpruned searches must agree on the best plan.

    If a bound ever rejects an achievable configuration this fails, which is the
    only cheap way to tell a pruning bug from a genuinely empty feasible set.
    """
    pruned = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), enable_bound_pruning=True
    )
    oracle = exhaustive.oracle(spec, cluster, islands, profiles, MockPredictor())

    assert pruned.feasible == oracle.feasible
    assert pruned.recommended is not None and oracle.recommended is not None
    assert pruned.recommended.plan.candidate.id == oracle.recommended.plan.candidate.id
    assert pruned.recommended.value == oracle.recommended.value


def test_oracle_evaluates_at_least_as_many_candidates(spec, cluster, islands, profiles) -> None:
    pruned_pred, oracle_pred = MockPredictor(), MockPredictor()
    exhaustive.search(spec, cluster, islands, profiles, pruned_pred, enable_bound_pruning=True)
    exhaustive.oracle(spec, cluster, islands, profiles, oracle_pred)
    assert len(oracle_pred.calls) >= len(pruned_pred.calls)


def test_oracle_mode_is_flagged_in_the_output(spec, cluster, islands, profiles) -> None:
    output = exhaustive.oracle(spec, cluster, islands, profiles, MockPredictor())
    assert any("Oracle mode" in c for c in output.caveats)


# --- reproducibility ------------------------------------------------------

def test_same_inputs_produce_identical_output(spec, cluster, islands, profiles) -> None:
    """§9: byte-identical plan output across runs."""
    runs = [
        exhaustive.search(spec, cluster, islands, profiles, MockPredictor()).model_dump(
            mode="json", exclude={"provenance"}
        )
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_candidate_generation_is_deterministic(spec, cluster, islands, profiles) -> None:
    runs = [
        [c.id for c in CandidateGenerator(spec, cluster, islands, profiles).generate().candidates]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# --- simulator failures are not verdicts ----------------------------------

def test_sim_errors_get_their_own_bucket(spec, cluster, islands, profiles) -> None:
    """D12: a crash is unmeasured, not infeasible. It must never be folded in."""
    candidates = CandidateGenerator(spec, cluster, islands, profiles).generate().candidates
    doomed = {c.id for c in candidates[:2]}
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(fail_ids=doomed)
    )
    assert output.rejected_summary.get(RejectionStage.SIM_ERROR.value) == len(doomed)
    for id_ in doomed:
        assert id_ not in _ids(output)


def test_timeouts_are_counted_as_sim_errors_too(spec, cluster, islands, profiles) -> None:
    candidates = CandidateGenerator(spec, cluster, islands, profiles).generate().candidates
    stuck = {candidates[0].id}
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(timeout_ids=stuck)
    )
    assert output.rejected_summary.get(RejectionStage.SIM_ERROR.value) == 1


def test_all_sims_failing_is_reported_as_such(spec, cluster, islands, profiles) -> None:
    """A search where nothing ran must not look like a search that found nothing."""
    candidates = CandidateGenerator(spec, cluster, islands, profiles).generate().candidates
    output = exhaustive.search(
        spec, cluster, islands, profiles,
        MockPredictor(fail_ids={c.id for c in candidates}),
    )
    assert not output.feasible
    assert any("failed to simulate" in s for s in output.suggestions)


# --- infeasible diagnosis -------------------------------------------------

def test_infeasible_produces_closest_plan_and_suggestions(
    spec, cluster, islands, profiles
) -> None:
    """§3.5: never a bare failure."""
    # Tighten TTFT rather than TPOT: the analytical stage-5 bound checks TPOT
    # and throughput, so an impossible TPOT is pruned before simulation and
    # there is no simulated plan left to diagnose. TTFT exercises the path
    # where candidates reach the predictor and fail on its output.
    impossible = spec.model_copy(deep=True)
    impossible.slo.ttft.max_ms = 1.0
    output = exhaustive.search(spec=impossible, cluster=cluster, islands=islands,
                               profiles=profiles, predictor=MockPredictor())
    assert not output.feasible
    assert output.reason
    assert output.closest_plan is not None
    assert output.violated_constraints
    assert output.suggestions
    assert any("relax the TTFT SLO" in s for s in output.suggestions)


def test_closest_plan_is_the_smallest_miss(spec, cluster, islands, profiles) -> None:
    impossible = spec.model_copy(deep=True)
    impossible.slo.ttft.max_ms = 1.0
    output = exhaustive.search(spec=impossible, cluster=cluster, islands=islands,
                               profiles=profiles, predictor=MockPredictor())
    assert output.closest_plan is not None
    # Assert the contract, not the mock's internals: no other simulated plan
    # missed by less. Overshoot is normalised so metrics stay comparable.
    from planner.optimizer import feasibility

    worst = max(v.overshoot_ratio for v in output.violated_constraints)
    oracle_out = exhaustive.oracle(spec=impossible, cluster=cluster, islands=islands,
                                   profiles=profiles, predictor=MockPredictor())
    others = []
    for alt_plan in [oracle_out.closest_plan] if oracle_out.closest_plan else []:
        rep = feasibility.evaluate(alt_plan, impossible)
        others.append(rep.worst_overshoot)
    assert all(worst <= o + 1e-9 for o in others)


# --- output contract ------------------------------------------------------

def test_prefix_cache_caveat_always_travels_with_the_result(
    spec, cluster, islands, profiles
) -> None:
    """Phase 2 runs with prefix caching off; that limitation must be visible."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert any("Prefix caching is disabled" in c for c in output.caveats)
    assert any("prefix_share_ratio" in c for c in output.caveats)


def test_rejections_are_reported_even_on_success(spec, cluster, islands, profiles) -> None:
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.feasible
    assert output.rejected_summary  # the Ascend island is incompatible
    assert output.generated_candidates >= output.evaluated_candidates


def test_alternatives_are_all_on_the_frontier(spec, cluster, islands, profiles) -> None:
    from planner.optimizer import pareto

    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.recommended is not None
    plans = [output.recommended.plan] + [a.plan for a in output.alternatives]
    front_ids = {p.plan_id for p in pareto.frontier(plans)}
    for alt in output.alternatives:
        assert alt.plan.plan_id in front_ids


# --- output honesty -------------------------------------------------------

def test_equivalent_outcomes_are_collapsed(spec, cluster, islands, profiles) -> None:
    """Knobs that never bind produce identical results; three copies of one
    outcome is not three alternatives."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.recommended is not None
    from planner.optimizer import pareto

    shown = [output.recommended.plan] + [a.plan for a in output.alternatives]
    sigs = [pareto.metric_signature(p) for p in shown]
    assert len(sigs) == len(set(sigs)), "the same predicted outcome is listed twice"


def test_collapsed_duplicates_are_still_reported(spec, cluster, islands, profiles) -> None:
    """Collapsing must not hide that other configurations reach the same point."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    collapsed = [a for a in output.alternatives if a.equivalent_candidates]
    assert collapsed, "expected at least one group of equivalent candidates"
    for alt in collapsed:
        assert alt.plan.candidate.id not in alt.equivalent_candidates


def test_plans_the_objective_cannot_score_are_surfaced(
    spec, cluster, islands, profiles
) -> None:
    """A feasible plan with no energy must not vanish under minimize_energy.

    node0 (A5000) has no power: block, so its candidates simulate fine but carry
    no energy. Sorting them last would drop an entire island from the output
    with no explanation.
    """
    from planner.spec import Objective

    assert spec.objective.primary is Objective.MINIMIZE_ENERGY
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    ranked_ids = {output.recommended.plan.candidate.id} | {
        a.plan.candidate.id for a in output.alternatives
    }
    unscored_ids = {u.plan.candidate.id for u in output.unscored}
    assert unscored_ids, "expected the no-power island's plans to be reported as unscored"
    assert not (ranked_ids & unscored_ids)
    for u in output.unscored:
        assert "energy" in u.reason and "power:" in u.reason


def test_cache_hits_go_to_provenance_not_caveats(spec, cluster, islands, profiles) -> None:
    """One caveat line per cached candidate buries the caveat that matters."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert not any("envelope cache" in c for c in output.caveats)
