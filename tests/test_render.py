"""Rendering contract (work order §6).

The `plan` stdout is the primary interface, so its structure needs the same
protection as the data behind it. These tests exist because an earlier edit
inserted the "Feasible but not ranked" block *between* the feasible branch and
its `else`, which rebound the `else` to the wrong condition and printed
INFEASIBLE on a successful search. Nothing caught it until the output was read
by eye.
"""

from __future__ import annotations

from planner.optimizer import exhaustive
from planner.plan import PlannerOutput
from planner.render import render

from .conftest import MockPredictor


def test_feasible_output_never_says_infeasible(spec, cluster, islands, profiles) -> None:
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.feasible
    text = render(output)
    assert "INFEASIBLE" not in text
    assert "Recommended plan" in text


def test_infeasible_output_never_shows_a_recommendation(
    spec, cluster, islands, profiles
) -> None:
    impossible = spec.model_copy(deep=True)
    impossible.slo.ttft.max_ms = 1.0
    output = exhaustive.search(spec=impossible, cluster=cluster, islands=islands,
                               profiles=profiles, predictor=MockPredictor())
    assert not output.feasible
    text = render(output)
    assert "INFEASIBLE" in text
    assert "Recommended plan" not in text
    assert "suggestions:" in text


def test_required_sections_are_always_present(spec, cluster, islands, profiles) -> None:
    """§6: feasible count, rejected counts with reasons, recommendation,
    alternatives and predicted metrics must all appear."""
    text = render(exhaustive.search(spec, cluster, islands, profiles, MockPredictor()))
    for section in (
        "Rejected candidates",
        "Feasible candidates",
        "Recommended plan",
        "Pareto alternatives",
        "TTFT  p50/p95/p99",
        "TPOT  p50/p95/p99",
        "SLO attainment",
        "Caveats",
    ):
        assert section in text, f"missing required section: {section}"


def test_unscored_plans_are_rendered_on_a_feasible_result(
    spec, cluster, islands, profiles
) -> None:
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.unscored, "fixture should produce unrankable plans (node0 has no power block)"
    text = render(output)
    assert "Feasible but not ranked" in text
    assert "INFEASIBLE" not in text


def test_missing_energy_is_labelled_not_shown_as_zero(spec, cluster, islands, profiles) -> None:
    """A plan with no energy must say so rather than print a misleading 0 J."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    text = render(output)
    assert "not simulated" in text or "energy            :" in text
    assert "0.0 J  (avg 0.0 W" not in text


def test_render_survives_an_empty_search(spec) -> None:
    """No candidates at all must still produce a readable diagnosis."""
    output = PlannerOutput(
        feasible=False,
        service_model=spec.model,
        cluster_id="empty",
        reason="no candidate survived generation, so nothing was simulated",
        suggestions=["check the cluster spec and profiles"],
    )
    text = render(output)
    assert "INFEASIBLE" in text
    assert "check the cluster spec" in text
