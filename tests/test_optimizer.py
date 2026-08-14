"""Feasibility, Pareto dominance and lexicographic ranking (§5.6, §9)."""

from __future__ import annotations

import pytest

from planner.optimizer import feasibility, pareto
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    IslandAssignment,
    PredictedMetrics,
    RejectionStage,
)
from planner.spec import Objective


def metrics(**kw) -> PredictedMetrics:
    base = {
        "p50_ttft_ms": 100.0, "p95_ttft_ms": 150.0, "p99_ttft_ms": 200.0,
        "p50_tpot_ms": 10.0, "p95_tpot_ms": 15.0, "p99_tpot_ms": 20.0,
        "throughput_tps": 1000.0, "slo_goodput_rps": 10.0, "slo_attainment": 1.0,
        "completed_requests": 100, "completed_tokens": 10_000,
        "total_energy_j": 1000.0, "average_power_w": 100.0,
        "peak_power_w": 120.0, "tokens_per_joule": 10.0,
    }
    base.update(kw)
    return PredictedMetrics(**base)


def plan(plan_id: str = "hp-1", *, devices: int = 1, tp: int = 1, **metric_kw) -> DeploymentPlan:
    cand = CandidateConfig(
        id=plan_id, model="m", dtype="bfloat16",
        assignments=[IslandAssignment(island_id="isl", tp_size=tp, dp_replicas=devices // tp)],
    )
    return DeploymentPlan(
        plan_id=plan_id, model="m", candidate=cand, predicted=metrics(**metric_kw)
    )


# --- feasibility ----------------------------------------------------------

def test_all_constraints_met(spec) -> None:
    report = feasibility.evaluate(plan(p99_ttft_ms=100.0, p99_tpot_ms=10.0), spec)
    assert report.passed
    assert report.violations == []


def test_ttft_violation_reported_with_numbers(spec) -> None:
    over = spec.slo.ttft.max_ms * 2
    report = feasibility.evaluate(plan(p99_ttft_ms=over), spec)
    assert not report.passed
    assert report.stage is RejectionStage.SLO_VIOLATED
    v = next(v for v in report.violations if "ttft" in v.metric)
    assert v.target == spec.slo.ttft.max_ms
    assert v.predicted == pytest.approx(over)


def test_robust_margin_can_flip_a_pass_to_a_fail(spec) -> None:
    """§5.8: feasibility is checked on predicted * (1 + p95_error)."""
    just_under = spec.slo.tpot.max_ms * 0.95
    assert feasibility.evaluate(plan(p99_tpot_ms=just_under), spec).passed
    assert not feasibility.evaluate(
        plan(p99_tpot_ms=just_under), spec, tpot_margin_percent=20.0
    ).passed


def test_power_cap_violation(spec) -> None:
    tight = spec.model_copy(deep=True)
    tight.slo.max_cluster_power_w = 100.0
    report = feasibility.evaluate(plan(peak_power_w=500.0), tight)
    assert not report.passed
    assert report.stage is RejectionStage.POWER_VIOLATED


def test_efficiency_floor_violation(spec) -> None:
    tight = spec.model_copy(deep=True)
    tight.slo.min_tokens_per_joule = 100.0
    report = feasibility.evaluate(plan(tokens_per_joule=1.0), tight)
    assert not report.passed
    assert report.stage is RejectionStage.EFFICIENCY_VIOLATED


def test_unmeasurable_power_constraint_is_noted_not_silently_passed(spec) -> None:
    """A constraint that could not be evaluated must never read as satisfied."""
    tight = spec.model_copy(deep=True)
    tight.slo.max_cluster_power_w = 100.0
    report = feasibility.evaluate(
        plan(peak_power_w=None, total_energy_j=None, tokens_per_joule=None), tight
    )
    assert report.passed  # nothing to violate...
    assert any("no power output" in n for n in report.notes)  # ...but it is flagged


# --- Pareto ---------------------------------------------------------------

def test_dominance_requires_better_somewhere() -> None:
    a = plan("a", p99_ttft_ms=100.0, p99_tpot_ms=10.0)
    b = plan("b", p99_ttft_ms=200.0, p99_tpot_ms=20.0)
    assert pareto.dominates(a, b)
    assert not pareto.dominates(b, a)


def test_identical_plans_do_not_dominate_each_other() -> None:
    assert not pareto.dominates(plan("a"), plan("b"))


def test_mixed_trade_off_is_not_domination() -> None:
    fast_hungry = plan("a", p99_ttft_ms=50.0, peak_power_w=900.0)
    slow_frugal = plan("b", p99_ttft_ms=500.0, peak_power_w=100.0)
    assert not pareto.dominates(fast_hungry, slow_frugal)
    assert not pareto.dominates(slow_frugal, fast_hungry)


def test_missing_dimension_cannot_confer_dominance() -> None:
    """A plan without energy must not beat one that was measured."""
    measured = plan("a", tokens_per_joule=1.0)
    unmeasured = plan("b", total_energy_j=None, tokens_per_joule=None, peak_power_w=None)
    assert not pareto.dominates(unmeasured, measured)


def test_frontier_keeps_only_non_dominated() -> None:
    best = plan("best", p99_ttft_ms=10.0, p99_tpot_ms=1.0)
    worse = plan("worse", p99_ttft_ms=100.0, p99_tpot_ms=10.0)
    trade = plan("trade", p99_ttft_ms=5.0, p99_tpot_ms=50.0)
    front = pareto.frontier([best, worse, trade])
    ids = {p.plan_id for p in front}
    assert "worse" not in ids
    assert {"best", "trade"} <= ids


# --- ranking --------------------------------------------------------------

def test_minimize_energy_prefers_less_energy() -> None:
    lo = plan("lo", total_energy_j=100.0)
    hi = plan("hi", total_energy_j=900.0)
    ranked = pareto.rank([hi, lo], Objective.MINIMIZE_ENERGY)
    assert ranked[0].plan.plan_id == "lo"


def test_minimize_accelerators_prefers_fewer() -> None:
    small = plan("small", devices=1)
    big = plan("big", devices=4)
    ranked = pareto.rank([big, small], Objective.MINIMIZE_ACTIVE_ACCELERATORS)
    assert ranked[0].plan.plan_id == "small"


def test_goodput_per_joule_uses_slo_satisfying_tokens_only() -> None:
    """§4: SLO-goodput/J counts tokens of requests that met BOTH SLOs."""
    full = plan("full", slo_attainment=1.0, completed_tokens=10_000, total_energy_j=1000.0)
    half = plan("half", slo_attainment=0.5, completed_tokens=10_000, total_energy_j=1000.0)
    v_full = pareto.objective_value(full, Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE)
    v_half = pareto.objective_value(half, Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE)
    assert v_full == pytest.approx(2 * v_half)


def test_tie_breaks_run_in_the_fixed_order() -> None:
    """Equal on the objective, so fewest accelerators must win (§5.6 stage 3)."""
    a = plan("a", devices=4, total_energy_j=500.0)
    b = plan("b", devices=1, total_energy_j=500.0)
    ranked = pareto.rank([a, b], Objective.MINIMIZE_ENERGY)
    assert ranked[0].plan.plan_id == "b"


def test_ranking_is_deterministic() -> None:
    plans = [plan(f"p{i}", total_energy_j=500.0, devices=1) for i in range(6)]
    first = [s.plan.plan_id for s in pareto.rank(plans, Objective.MINIMIZE_ENERGY)]
    for _ in range(3):
        assert [s.plan.plan_id for s in pareto.rank(plans, Objective.MINIMIZE_ENERGY)] == first


def test_alternatives_are_annotated_with_the_trade_off() -> None:
    best = plan("best", p99_ttft_ms=10.0, peak_power_w=500.0)
    alt = plan("alt", p99_ttft_ms=100.0, peak_power_w=50.0)
    scored = pareto.rank([alt], Objective.MINIMIZE_ENERGY)
    annotated = pareto.annotate_alternatives(best, scored)
    assert annotated[0].note
    assert "power" in annotated[0].note or "ttft" in annotated[0].note
