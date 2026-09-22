"""The service contract H1 adds: throughput floor, completion ratio, cost objective.

All three are opt-in. With none of them set the declared constraint set is
exactly what it was, which is the property the rest of the planner leans on:
`candidate_generator` removed its throughput lower bound because §5.6 declared
no throughput constraint to relax, and the comment there says restoring it
needs `slo.min_goodput_rps` to be what both sides read. This file pins both
halves - that the new constraints bite when set, and that nothing moves when
they are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from planner.optimizer import feasibility, pareto
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    IslandAssignment,
    PlannerOutput,
    PredictedMetrics,
    RejectionStage,
    ScoredPlan,
)
from planner.spec import Objective, SpecError, load_service_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC = ROOT / "examples/service_specs/llama31-8b.yaml"


def raw_spec() -> dict:
    return yaml.safe_load(EXAMPLE_SPEC.read_text())


def metrics(**kw) -> PredictedMetrics:
    base = {
        "p50_ttft_ms": 100.0, "p95_ttft_ms": 150.0, "p99_ttft_ms": 200.0,
        "p50_tpot_ms": 10.0, "p95_tpot_ms": 15.0, "p99_tpot_ms": 20.0,
        "throughput_tps": 1000.0, "slo_goodput_rps": 10.0, "slo_attainment": 1.0,
        "completed_requests": 100, "completed_tokens": 10_000,
    }
    base.update(kw)
    return PredictedMetrics(**base)


def plan(plan_id: str = "hp-1", *, cost: float | None = None, **metric_kw) -> DeploymentPlan:
    cand = CandidateConfig(
        id=plan_id, model="m", dtype="bfloat16",
        assignments=[IslandAssignment(island_id="isl", tp_size=1)],
    )
    return DeploymentPlan(
        plan_id=plan_id, model="m", candidate=cand,
        predicted=metrics(**metric_kw), cost_per_hour_usd=cost,
    )


def spec_with(spec, **slo_kw):
    """`spec` with the SLO block overridden, leaving everything else alone."""
    return spec.model_copy(update={"slo": spec.slo.model_copy(update=slo_kw)})


# --- (i) throughput floor -------------------------------------------------

def test_goodput_below_the_floor_is_an_slo_violation(spec) -> None:
    report = feasibility.evaluate(
        plan(p99_ttft_ms=1.0, p99_tpot_ms=1.0, slo_goodput_rps=10.0),
        spec_with(spec, min_goodput_rps=12.0),
    )
    assert not report.passed
    assert report.stage is RejectionStage.SLO_VIOLATED
    v = next(v for v in report.violations if v.metric == "slo_goodput_rps")
    assert v.target == pytest.approx(12.0)
    assert v.predicted == pytest.approx(10.0)


def test_goodput_at_the_floor_passes(spec) -> None:
    """The constraint is `>=`, so exactly meeting it is not a miss."""
    report = feasibility.evaluate(
        plan(p99_ttft_ms=1.0, p99_tpot_ms=1.0, slo_goodput_rps=12.0),
        spec_with(spec, min_goodput_rps=12.0),
    )
    assert report.passed


# --- (ii) completion ratio ------------------------------------------------

def test_completion_ratio_uses_offered_requests(spec) -> None:
    report = feasibility.evaluate(
        plan(p99_ttft_ms=1.0, p99_tpot_ms=1.0, completed_requests=98, offered_requests=100),
        spec_with(spec, min_completion_ratio=0.99),
    )
    assert not report.passed
    v = next(v for v in report.violations if v.metric == "completion_ratio")
    assert v.predicted == pytest.approx(0.98)
    assert v.target == pytest.approx(0.99)


# --- (iii) an unknown figure is a note, never a verdict -------------------

def test_unknown_offered_count_is_reported_not_assumed(spec) -> None:
    """D2's rule: a constraint that could not be measured is not satisfied.

    It is not violated either. A predictor with no per-request records leaves
    `offered_requests` None, and inventing either answer would be worse than
    saying so - a silent pass ships a plan that was never checked, a silent
    violation drops one that may well have been fine.
    """
    report = feasibility.evaluate(
        plan(p99_ttft_ms=1.0, p99_tpot_ms=1.0, completed_requests=98, offered_requests=None),
        spec_with(spec, min_completion_ratio=0.99),
    )
    assert report.passed
    assert not [v for v in report.violations if v.metric == "completion_ratio"]
    assert any("could not be checked" in n for n in report.notes)


def test_offered_count_of_zero_is_also_unknown(spec) -> None:
    report = feasibility.evaluate(
        plan(p99_ttft_ms=1.0, p99_tpot_ms=1.0, completed_requests=0, offered_requests=0),
        spec_with(spec, min_completion_ratio=0.99),
    )
    assert report.passed
    assert any("could not be checked" in n for n in report.notes)


# --- (iv) neither set: nothing moves --------------------------------------

def test_both_unset_is_a_no_op(spec) -> None:
    assert spec.slo.min_goodput_rps is None
    assert spec.slo.min_completion_ratio is None
    violations, notes = feasibility.check_throughput(
        metrics(slo_goodput_rps=0.0, completed_requests=1, offered_requests=None), spec
    )
    assert violations == []
    assert notes == []


# --- (v) the cost objective -----------------------------------------------

def test_a_plan_with_no_price_cannot_be_scored_on_cost() -> None:
    ok, reason = pareto.can_score(plan(cost=None), Objective.MINIMIZE_COST_PER_HOUR)
    assert not ok
    assert "price" in reason


def test_cheaper_plan_ranks_first() -> None:
    plans = [plan("hp-expensive", cost=3.5), plan("hp-cheap", cost=2.0)]
    ranked = pareto.rank(plans, Objective.MINIMIZE_COST_PER_HOUR)
    assert ranked[0].plan.cost_per_hour_usd == pytest.approx(2.0)
    assert ranked[0].value == pytest.approx(-2.0)


def test_cost_is_a_pareto_dimension_only_when_both_plans_carry_it() -> None:
    """A priced plan must not dominate an unpriced one on price alone.

    `dominates` skips a dimension either side is missing, so the unpriced plan
    survives into the alternatives instead of vanishing with no explanation.
    """
    priced, unpriced = plan("hp-a", cost=2.0), plan("hp-b", cost=None)
    assert not pareto.dominates(priced, unpriced)
    assert not pareto.dominates(unpriced, priced)
    cheap, dear = plan("hp-c", cost=1.0), plan("hp-d", cost=9.0)
    assert pareto.dominates(cheap, dear)


# --- (vi) the default path's YAML is unchanged ----------------------------

def test_default_dump_omits_the_three_new_keys(tmp_path) -> None:
    """Rule A4. The dump has no `exclude_none` and the frozen outputs DO carry
    nulls for older optional fields, so these have to be dropped on their own."""
    from planner.__main__ import _write_output

    scored = ScoredPlan(plan=plan(), objective=Objective.MINIMIZE_ACTIVE_ACCELERATORS,
                        value=-1.0)
    out = PlannerOutput(feasible=True, service_model="m", cluster_id="c",
                        recommended=scored, alternatives=[scored])
    path = tmp_path / "plan.yaml"
    _write_output(out, path)
    text = path.read_text()
    for key in ("offered_requests", "cost_per_hour_usd", "cost_basis"):
        assert key not in text
    loaded = yaml.safe_load(text)
    assert loaded["recommended"]["plan"]["predicted"]["p50_ttft_ms"] == pytest.approx(100.0)


def test_dump_carries_the_keys_once_they_are_filled(tmp_path) -> None:
    from planner.__main__ import _write_output

    filled = plan(cost=2.5, offered_requests=100).model_copy(
        update={"cost_basis": "datasheet list price"}
    )
    scored = ScoredPlan(plan=filled, objective=Objective.MINIMIZE_COST_PER_HOUR, value=-2.5)
    out = PlannerOutput(feasible=True, service_model="m", cluster_id="c",
                        recommended=scored, alternatives=[scored])
    path = tmp_path / "plan.yaml"
    _write_output(out, path)
    loaded = yaml.safe_load(path.read_text())
    plan_out = loaded["recommended"]["plan"]
    assert plan_out["cost_per_hour_usd"] == pytest.approx(2.5)
    assert plan_out["cost_basis"] == "datasheet list price"
    assert plan_out["predicted"]["offered_requests"] == 100


# --- (vii) validation -----------------------------------------------------

@pytest.mark.parametrize(
    "key, value",
    [
        ("min_goodput_rps", -1),
        ("min_goodput_rps", 0),
        ("min_completion_ratio", 0),
        ("min_completion_ratio", 1.5),
        ("observation_window_s", -1),
    ],
)
def test_out_of_range_slo_values_are_rejected(tmp_path, key, value) -> None:
    raw = raw_spec()
    raw["slo"][key] = value
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SpecError):
        load_service_spec(path)


def test_the_three_fields_round_trip(tmp_path) -> None:
    raw = raw_spec()
    raw["slo"].update(
        {"min_goodput_rps": 12.0, "min_completion_ratio": 0.99, "observation_window_s": 60.0}
    )
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = load_service_spec(path)
    assert loaded.slo.min_goodput_rps == pytest.approx(12.0)
    assert loaded.slo.min_completion_ratio == pytest.approx(0.99)
    assert loaded.slo.observation_window_s == pytest.approx(60.0)


def test_cost_objective_parses_from_yaml(tmp_path) -> None:
    raw = raw_spec()
    raw["objective"]["primary"] = "minimize_cost_per_hour"
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_service_spec(path).objective.primary is Objective.MINIMIZE_COST_PER_HOUR
