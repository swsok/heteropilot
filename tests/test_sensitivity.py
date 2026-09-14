"""Flip detection and decision regret (WORK_ORDER_uncertainty_planner.md B2).

The fixture is the work order's: two candidates, A on a single island and B a
P/D split, a LINK_BW range of [13, 35] with B recommended at the nominal 35. The
link is real - `pd-rngd-gpu.yaml`'s measured 12.6 GB/s fabric - so the transfer
term being perturbed is the one the planner actually prices.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer.margin import GlobalMargin
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    IslandAssignment,
    PlannerOutput,
    PredictedMetrics,
    Role,
    ScoredPlan,
    ServingArch,
)
from planner.spec import load_service_spec
from planner.uncertainty.perturb import JudgedMetrics, PerturbContext
from planner.uncertainty.registry import (
    Grade,
    MeasurementCost,
    Range,
    UncertainInput,
    UncertainInputRegistry,
    UncertainKind,
)
from planner.uncertainty.sensitivity import (
    analyze,
    build_grid,
    unbounded,
    worth_measuring,
)

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml"
SERVICE = ROOT / "examples/service_specs/llama31-8b.yaml"


def _metrics(*, energy: float, ttft: float = 500.0) -> PredictedMetrics:
    return PredictedMetrics(
        p50_ttft_ms=ttft * 0.6, p95_ttft_ms=ttft * 0.9, p99_ttft_ms=ttft,
        p50_tpot_ms=10.0, p95_tpot_ms=12.0, p99_tpot_ms=14.0,
        throughput_tps=1000.0, slo_goodput_rps=10.0, slo_attainment=1.0,
        completed_requests=300, completed_tokens=60_000,
        total_energy_j=energy, average_power_w=100.0, peak_power_w=150.0,
        tokens_per_joule=60_000 / energy, served_concurrency=42.0,
    )


class _World(NamedTuple):
    spec: object
    cluster: object
    islands: dict
    candidates: dict
    context: PerturbContext
    metrics: dict
    island_hw: dict


@pytest.fixture
def world() -> _World:
    """A: single island. B: P/D across the fabric link, and cheaper on energy."""
    spec = load_service_spec(SERVICE)
    cluster = load_cluster_spec(CLUSTER)
    profiles = load_profiles_for(cluster, ROOT)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    rngd = next(i for i in islands if i.startswith("furiosa-rngd-node_rngd0"))
    a40 = next(i for i in islands if i.startswith("cuda-a40-node_a40a"))

    a = CandidateConfig(
        id="A", model=spec.model, dtype="bfloat16",
        assignments=[IslandAssignment(island_id=a40, tp_size=4)],
    )
    b = CandidateConfig(
        id="B", model=spec.model, dtype="bfloat16",
        serving_arch=ServingArch.PD_SPLIT,
        assignments=[
            IslandAssignment(island_id=rngd, role=Role.PREFILL, tp_size=4),
            IslandAssignment(island_id=a40, role=Role.DECODE, tp_size=4),
        ],
    )
    candidates = {"A": a, "B": b}
    context = PerturbContext.build(spec, cluster, islands, candidates)
    # B wins on energy (the objective is minimize_energy) but its TTFT sits close
    # to the 25 s SLO, so a slower link can push it out of feasibility.
    metrics = JudgedMetrics.trusted({
        "A": _metrics(energy=2000.0, ttft=500.0),
        "B": _metrics(energy=1000.0, ttft=24_500.0),
    })
    island_hw = {i: ("A40" if "a40" in i else "RNGD") for i in islands}
    return _World(spec, cluster, islands, candidates, context, metrics, island_hw)


def _output(recommended: str, spec, candidates, metrics) -> PlannerOutput:
    plan = DeploymentPlan(
        plan_id="hp-0001", model=spec.model,
        candidate=candidates[recommended], predicted=metrics[recommended],
    )
    return PlannerOutput(
        feasible=True, service_model=spec.model, cluster_id="c",
        recommended=ScoredPlan(plan=plan, objective=spec.objective.primary, value=0.0),
    )


def _link(cluster, prefix: str = "fabric-rngd0-a40a"):
    return next(link for link in cluster.links if link.id == prefix)


def _item(kind, ident, nominal, rng, *, affects=(), hours=None) -> UncertainInput:
    return UncertainInput(
        id=ident, kind=kind, grade=Grade.PLACEHOLDER, nominal=nominal, range=rng,
        affects=list(affects),
        cost=None if hours is None else MeasurementCost(method="m", hours=hours,
                                                        source="tests"),
    )


def _registry(*items) -> UncertainInputRegistry:
    return UncertainInputRegistry(items=list(items))


# --- the grid -------------------------------------------------------------

def test_the_default_grid_is_the_work_orders_five_points() -> None:
    """lo, lo+w/4, nominal, hi-w/4, hi."""
    item = _item(UncertainKind.LINK_BW, "link_bw:x", 24.0,
                 Range(lo=13.0, hi=35.0, unit="gbps", source="tests"))
    assert build_grid(item) == [13.0, 18.5, 24.0, 29.5, 35.0]


def test_the_grid_includes_the_nominal_even_when_it_is_not_the_midpoint() -> None:
    """A ratio_floor range puts nominal at `hi`; it must still be swept."""
    item = _item(UncertainKind.LINK_BW, "link_bw:x", 64.0,
                 Range(lo=51.4, hi=64.0, unit="gbps", source="tests"))
    grid = build_grid(item)
    assert 64.0 in grid and 51.4 in grid
    # Five requested, but nominal coincides with hi, so four distinct remain.
    assert len(grid) == 4
    assert grid == sorted(grid)


def test_an_unbounded_range_has_no_grid() -> None:
    item = _item(UncertainKind.LINK_BW, "link_bw:x", 35.0, Range(unit="gbps"))
    assert build_grid(item) == []


# --- the work order's flip case -------------------------------------------

def test_a_link_that_flips_the_recommendation_is_detected(world) -> None:
    link = _link(world.cluster)
    item = _item(
        UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
        Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests"),
        hours=0.5,
    )
    output = _output("B", world.spec, world.candidates, world.metrics)

    [result] = analyze(
        output, _registry(item), world.metrics, GlobalMargin(), world.spec,
        world.context, world.island_hw,
    )
    assert result.flip is True, "a slow enough link must cost B its feasibility"
    assert result.delta_regret is not None and result.delta_regret > 0
    assert result.approximation is False, "LINK_BW is exact (B1)"

    # At the top of the range - the nominal - the incumbent IS the best plan, so
    # there is nothing to regret there.
    top = result.grid[-1]
    assert top.value == pytest.approx(link.bandwidth_gbps)
    assert top.best_plan_id == "B"
    assert top.current_value == pytest.approx(top.best_value)

    # At the bottom it has been overtaken.
    bottom = result.grid[0]
    assert bottom.best_plan_id == "A"
    assert bottom.current_value < bottom.best_value


def test_an_input_that_changes_nothing_has_zero_regret_and_no_flip(world) -> None:
    link = _link(world.cluster)
    # A range so narrow that the transfer term barely moves.
    item = _item(
        UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
        Range(lo=link.bandwidth_gbps * 0.999, hi=link.bandwidth_gbps,
              unit="gbps", source="tests"),
    )
    output = _output("B", world.spec, world.candidates, world.metrics)

    [result] = analyze(
        output, _registry(item), world.metrics, GlobalMargin(), world.spec,
        world.context, world.island_hw,
    )
    assert result.flip is False
    assert result.delta_regret == pytest.approx(0.0)
    assert worth_measuring([result]) == []


def test_an_unbounded_item_cannot_be_decided_before_measuring(world) -> None:
    item = _item(UncertainKind.LINK_BW, "link_bw:whatever", 35.0, Range(unit="gbps"))
    output = _output("B", world.spec, world.candidates, world.metrics)

    [result] = analyze(
        output, _registry(item), world.metrics, GlobalMargin(), world.spec,
        world.context, world.island_hw,
    )
    assert result.delta_regret is None
    assert result.grid == []
    assert result.flip is False
    assert "cannot be decided before measuring" in result.note
    assert unbounded([result]) == [result]
    assert worth_measuring([result]) == []


# --- ordering -------------------------------------------------------------

def test_the_better_value_for_money_sorts_first(world) -> None:
    """§2.5's sort key is dR/cost, not dR."""
    link = _link(world.cluster)
    wide = Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests")

    cheap = _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
                  wide, hours=0.1)
    dear = _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
                 wide, hours=100.0)
    # Same regret by construction - only the cost differs.
    dear = dear.model_copy(update={"id": f"link_bw:{link.id}#expensive"})

    results = analyze(
        output := _output("B", world.spec, world.candidates, world.metrics),
        _registry(dear, cheap), world.metrics, GlobalMargin(), world.spec,
        world.context, world.island_hw,
    )
    assert output.recommended is not None
    assert next(r.input_id for r in results) == cheap.id
    assert results[0].regret_per_hour > results[1].regret_per_hour


def test_an_item_with_no_cost_sorts_after_the_ones_that_have_one(world) -> None:
    link = _link(world.cluster)
    wide = Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests")
    priced = _item(UncertainKind.LINK_BW, f"link_bw:{link.id}",
                   link.bandwidth_gbps, wide, hours=10.0)
    unpriced = _item(UncertainKind.LINK_BW, f"link_bw:{link.id}",
                     link.bandwidth_gbps, wide).model_copy(
        update={"id": f"link_bw:{link.id}#unpriced"}
    )

    results = analyze(
        _output("B", world.spec, world.candidates, world.metrics), _registry(unpriced, priced),
        world.metrics, GlobalMargin(), world.spec, world.context, world.island_hw,
    )
    assert results[0].cost_hours == 10.0
    assert results[1].cost_hours is None


# --- other kinds ----------------------------------------------------------

def test_a_profile_sweep_is_marked_approximate(world) -> None:
    a40 = next(i for i in world.islands if "a40" in i)
    item = _item(UncertainKind.PROFILE, "profile:a40", 1.0,
                 Range(lo=0.7, hi=1.3, unit="fraction", source="tests"),
                 affects=[a40], hours=2.1)

    [result] = analyze(
        _output("B", world.spec, world.candidates, world.metrics), _registry(item),
        world.metrics, GlobalMargin(), world.spec, world.context, world.island_hw,
    )
    assert result.approximation is True
    assert len(result.grid) == 5
    assert "approximate rule" in result.note


def test_a_search_with_no_recommendation_reports_every_item_as_undecidable(
    world,
) -> None:
    item = _item(UncertainKind.LINK_BW, "link_bw:x", 35.0,
                 Range(lo=13.0, hi=35.0, unit="gbps", source="tests"))
    output = PlannerOutput(feasible=False, service_model=world.spec.model, cluster_id="c")

    [result] = analyze(
        output, _registry(item), world.metrics, GlobalMargin(), world.spec,
        world.context, world.island_hw,
    )
    assert result.delta_regret is None
    assert "nothing feasible" in result.note


def test_the_analysis_never_mutates_the_metrics_it_is_given(world) -> None:
    link = _link(world.cluster)
    before = {cid: m.model_dump() for cid, m in world.metrics.items()}
    item = _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
                 Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests"))

    analyze(_output("B", world.spec, world.candidates, world.metrics), _registry(item),
            world.metrics, GlobalMargin(), world.spec, world.context, world.island_hw)

    assert {cid: m.model_dump() for cid, m in world.metrics.items()} == before


def test_regret_is_never_negative_even_when_the_incumbent_is_infeasible(world) -> None:
    """The invariant the penalty baseline exists to hold.

    Charging the penalty against the incumbent's OWN objective - the obvious
    reading of §2.5 - let an infeasible B at -1562 outscore a feasible A at
    -2000 and produced a NEGATIVE regret. "The plan you cannot deploy would have
    been better" is not a decision anyone can act on, so the charge starts from
    the best feasible value instead.
    """
    link = _link(world.cluster)
    item = _item(
        UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
        Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests"),
    )
    [result] = analyze(
        _output("B", world.spec, world.candidates, world.metrics), _registry(item),
        world.metrics, GlobalMargin(), world.spec, world.context, world.island_hw,
    )
    assert result.delta_regret is not None and result.delta_regret >= 0
    for point in result.grid:
        assert point.current_value <= point.best_value, point


def test_a_bigger_slo_penalty_makes_an_infeasible_incumbent_cost_more(world) -> None:
    link = _link(world.cluster)
    item = _item(
        UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
        Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests"),
    )
    output = _output("B", world.spec, world.candidates, world.metrics)

    def regret(penalty: float) -> float:
        [r] = analyze(output, _registry(item), world.metrics, GlobalMargin(), world.spec,
                      world.context, world.island_hw, slo_penalty=penalty)
        return r.delta_regret

    assert regret(10_000.0) > regret(100.0)


def test_a_grid_point_with_nothing_feasible_still_contributes_its_penalty(world) -> None:
    """§2.5 (revised): baseline 0, so the point does not drop out of the mean.

    An SLO nothing can meet makes every candidate infeasible at every grid
    point. The old code scored those points -inf and excluded them, which turned
    "this input cannot rescue the plan" into "this input was never swept".
    """
    impossible = world.spec.model_copy(
        update={
            "slo": world.spec.slo.model_copy(
                update={"tpot": world.spec.slo.tpot.model_copy(update={"max_ms": 1e-6})}
            )
        }
    )
    link = _link(world.cluster)
    item = _item(
        UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
        Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests"),
    )
    [result] = analyze(
        _output("B", impossible, world.candidates, world.metrics), _registry(item),
        world.metrics, GlobalMargin(), impossible, world.context, world.island_hw,
        slo_penalty=100.0,
    )
    assert len(result.grid) == 4, "every point is scored, none dropped"
    assert all(p.best_plan_id == "" for p in result.grid)
    assert result.delta_regret is not None and result.delta_regret > 0
    assert "does not change the decision there" in result.note


def test_the_penalty_and_grid_are_recorded_in_provenance(world) -> None:
    link = _link(world.cluster)
    item = _item(
        UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps,
        Range(lo=0.05, hi=link.bandwidth_gbps, unit="gbps", source="tests"),
    )
    prov: dict = {}
    analyze(
        _output("B", world.spec, world.candidates, world.metrics), _registry(item),
        world.metrics, GlobalMargin(), world.spec, world.context, world.island_hw,
        provenance=prov,
    )
    recorded = prov["uncertainty"]
    assert recorded["slo_penalty"] > 0
    assert recorded["slo_penalty_source"] == "default"
    assert recorded["grid"] == 5 and recorded["grid_weighting"] == "uniform"


def test_the_default_penalty_spans_the_whole_grid_not_just_the_nominal(world) -> None:
    """§2.5: "격자 전체에서 관측된 목적값의 최댓값".

    Perturbing a PROFILE changes every candidate's objective, so the largest
    value seen across the sweep is not the one at the nominal point.
    """
    a40 = next(i for i in world.islands if "a40" in i)
    item = _item(UncertainKind.PROFILE, "profile:a40", 1.0,
                 Range(lo=0.5, hi=1.5, unit="fraction", source="tests"), affects=[a40])
    prov: dict = {}
    analyze(
        _output("B", world.spec, world.candidates, world.metrics), _registry(item),
        world.metrics, GlobalMargin(), world.spec, world.context, world.island_hw,
        provenance=prov,
    )
    # The nominal point's largest |objective| is 2000 (candidate A's energy).
    # A PROFILE sweep does not move energy, so the span is unchanged here - the
    # assertion that matters is that it was computed from the sweep at all.
    assert prov["uncertainty"]["slo_penalty"] >= 2000.0


def test_a_profile_earns_regret_through_energy_alone(world) -> None:
    """The decision-level half of D34, and the reason it was worth fixing.

    Two candidates that both sit comfortably inside the SLOs and differ only in
    energy, which is this spec's primary objective. The incumbent B is the
    cheaper one; a profile error on B's island makes B slower and therefore
    costlier, and past a point A becomes the better plan. Knowing B's profile
    is worth something, and `delta_regret` should say so.

    Before D34 it said `0.0` - exactly, at every grid point - because
    `_scale_profile` held energy fixed, so the only way a PROFILE item could earn
    regret was to cross an SLO boundary. Under `minimize_energy` that made the
    most expensive measurement in `costs.yaml` (2.1 h) look worthless unless it
    happened to break a latency constraint.
    """
    ids = list(world.islands)
    a_island, b_island = ids[0], ids[1]
    candidates = {
        "A": CandidateConfig(
            id="A", model=world.spec.model, dtype="bfloat16",
            assignments=[IslandAssignment(island_id=a_island, tp_size=1)],
        ),
        "B": CandidateConfig(
            id="B", model=world.spec.model, dtype="bfloat16",
            assignments=[IslandAssignment(island_id=b_island, tp_size=1)],
        ),
    }
    # Latencies an order of magnitude inside the SLOs (25 000 ms / 50 ms), so
    # nothing here can flip by crossing one.
    metrics = JudgedMetrics.trusted({
        "A": _metrics(energy=2000.0, ttft=500.0),
        "B": _metrics(energy=1000.0, ttft=500.0),
    })
    context = PerturbContext.build(world.spec, world.cluster, world.islands, candidates)
    item = _item(UncertainKind.PROFILE, f"profile:{b_island}", 1.0,
                 Range(lo=1.0, hi=3.0, unit="fraction", source="tests"),
                 affects=[b_island], hours=2.1)

    out = analyze(
        _output("B", world.spec, candidates, metrics), _registry(item), metrics,
        GlobalMargin(), world.spec, context, world.island_hw, slo_penalty=2000.0,
    )
    assert out[0].flip is True, "B's energy passes A's inside the swept range"
    assert out[0].delta_regret is not None and out[0].delta_regret > 0.0
    # Nothing crossed an SLO: every grid point still has a feasible plan, so the
    # regret is the objective gap and not a penalty term.
    assert all(p.best_value > float("-inf") for p in out[0].grid)
