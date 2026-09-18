"""Flip detection and decision-regret reduction (§2.5, STEP B2).

For each uncertain input, walk a grid across its sourced range, re-price every
candidate at each point with the closed forms of `perturb.py`, and re-run the
planner's own verdict and ranking. Two things come out:

* **flip** - does the recommendation change anywhere in the range? A yes means
  the plan currently on the table rests on a number nobody has measured.
* **delta_regret** (`dR_i`) - the mean objective the current recommendation
  gives up across the grid against the best plan at each point. It is what a
  measurement would be worth, and B3 ranks the measurement plan by `dR_i/cost`.

Nothing here re-simulates and nothing here re-implements a decision. The
verdict comes from `exhaustive.judge` and the ranking from
`exhaustive.rank_plans` - the same functions that produced the recommendation
being tested, because a sensitivity analysis run against a copy would report
flips that came from the copy.

**Which metrics to pass in.** `metrics_by_candidate` must be the metrics as the
planner JUDGED them, i.e. after `apply_pd_transfer_cost`. The envelope cache
stores the raw simulator output from before that adjustment, so reading it
directly would leave a P/D candidate's transfer term out and the LINK_BW rule -
which adds the DIFFERENCE between two transfer times - would then be adding it
to a number that never contained it. Use `metrics_from_evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from planner.optimizer import pareto
from planner.optimizer.exhaustive import Verdict, judge, rank_plans
from planner.plan import DeploymentPlan, PredictedMetrics
from planner.uncertainty.perturb import (
    JudgedMetrics,
    PerturbContext,
    judged_metrics,
    perturb,
)
from planner.uncertainty.registry import (
    UncertainInput,
    UncertainInputRegistry,
)

if TYPE_CHECKING:
    from planner.optimizer.margin import MarginDecision, MarginPolicy
    from planner.plan import PlannerOutput
    from planner.spec import ServiceSpec

#: §2.5's default grid size.
DEFAULT_GRID = 5


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GridPoint(_Strict):
    """One point of one item's grid, and what the planner decided there."""

    value: float
    #: Candidate id of the best plan at this point; "" when nothing is feasible.
    best_plan_id: str
    #: V(pi*(g)) - the best achievable objective here. Always maximised.
    best_value: float
    #: V_g(pi_hat) - the CURRENT recommendation scored here, penalised if it
    #: stopped being feasible.
    current_value: float


class Sensitivity(_Strict):
    """What one uncertain input is worth knowing."""

    input_id: str
    kind: str
    flip: bool
    #: None when the range is unbounded: regret cannot be defined over a range
    #: nobody has bounded, and the honest report is "cannot be decided before
    #: measuring" rather than a zero that reads as "does not matter".
    delta_regret: float | None
    grid: list[GridPoint] = Field(default_factory=list)
    approximation: bool = False
    #: Set when --resimulate-top replaced this item's closed-form dR.
    resimulated: bool = False
    #: Hours from costs.yaml, for B3's dR/cost ordering. None sorts last.
    cost_hours: float | None = None
    #: ``default`` when the swept interval is the grade's default rather than
    #: this input's own sourced width (S2, D111). The regret is then a regret
    #: over an assumed range, and every consumer says so.
    range_source: str = "sourced"
    note: str = ""

    @property
    def regret_per_hour(self) -> float | None:
        if self.delta_regret is None or self.cost_hours in (None, 0):
            return None
        return self.delta_regret / self.cost_hours


def metrics_from_evaluation(evaluation) -> JudgedMetrics:
    """Post-adjustment metrics for every candidate that got any.

    Kept as the name B2's callers use; the builder itself lives with the type it
    produces, beside the rule that needs it.
    """
    return judged_metrics(evaluation)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def build_grid(item: UncertainInput, size: int = DEFAULT_GRID) -> list[float]:
    """§2.5's grid: `lo, lo+w/4, nominal, hi-w/4, hi` at the default size.

    Nominal is included deliberately and is NOT assumed to be the midpoint - a
    `ratio_floor` range puts it at `hi`. Duplicates are collapsed so a point
    cannot take double weight in the mean.
    """
    rng = item.range
    if rng.lo is None or rng.hi is None:
        return []
    lo, hi = rng.lo, rng.hi
    if size <= 1:
        return [item.nominal]
    if size == 5:
        width = hi - lo
        points = [lo, lo + width / 4, item.nominal, hi - width / 4, hi]
    else:
        points = [lo + (hi - lo) * k / (size - 1) for k in range(size)]
        points.append(item.nominal)
    return sorted({round(p, 12) for p in points})


# ---------------------------------------------------------------------------
# Judging one grid point
# ---------------------------------------------------------------------------

def _plans_at(
    metrics: dict[str, PredictedMetrics],
    context: PerturbContext,
    spec: ServiceSpec,
    policy: MarginPolicy,
    island_hw: dict[str, str],
    sim_error_override: float | None,
) -> tuple[list[DeploymentPlan], dict[str, DeploymentPlan], dict[str, MarginDecision]]:
    """(feasible plans, every judged plan, the decision each was judged under).

    `sim_error_override` replaces the margin the policy would have derived,
    which is how a SIM_ERROR item perturbs: the prediction does not move, the
    margin does (§2.7).

    The decisions come back because the caller needs the SAME one to ask how
    badly a plan missed. Re-deriving it from the policy drops the override and
    silently re-judges the plan under the margin it was NOT rejected with --
    which reported `worst_overshoot` 0.0 for a plan the sweep had just found
    infeasible, and so gave every SIM_ERROR item a regret of exactly zero.
    """
    feasible: list[DeploymentPlan] = []
    everything: dict[str, DeploymentPlan] = {}
    decisions: dict[str, MarginDecision] = {}
    for cid in sorted(metrics):
        candidate = context.candidates.get(cid)
        if candidate is None:
            continue
        decision = policy.decide(
            candidate, context.sim_for(cid, metrics[cid]), metrics[cid], island_hw
        )
        if sim_error_override is not None and not decision.is_unmeasured:
            # A SIM_ERROR item's range is the scalar bucket's fraction, (real -
            # sim) / sim; the margin it implies is one-sided, like every other.
            percent = max(0.0, sim_error_override) * 100.0
            decision = decision.model_copy(
                update={"ttft_percent": percent, "tpot_percent": percent}
            )
        plan = DeploymentPlan(
            plan_id=f"hp-{cid}", model=spec.model, candidate=candidate,
            predicted=metrics[cid],
            robust_margin_ttft_percent=decision.ttft_percent,
            robust_margin_tpot_percent=decision.tpot_percent,
            operating_point=decision.operating_point,
            margin_source=decision.source,
        )
        everything[cid] = plan
        decisions[cid] = decision
        if judge(plan, spec, decision).verdict is Verdict.FEASIBLE:
            feasible.append(plan)
    return feasible, everything, decisions


def _value_of(plan: DeploymentPlan, spec: ServiceSpec) -> float:
    return pareto.objective_value(plan, spec.objective.primary)


@dataclass(frozen=True)
class _Swept:
    """What one grid point looked like, before the penalty was known."""

    value: float
    best_plan_id: str
    #: -inf when no candidate was feasible here.
    best_value: float
    #: The incumbent's own objective when it stayed feasible, else None.
    incumbent_value: float | None
    #: How badly the incumbent missed, when it did not.
    overshoot: float
    approximation: bool
    #: True when the best plan here is not the incumbent.
    flip: bool
    #: Every objective value seen here, for the penalty's default.
    values: list[float]


def _sweep_point(
    item: UncertainInput,
    value: float,
    metrics: JudgedMetrics,
    context: PerturbContext,
    spec: ServiceSpec,
    policy: MarginPolicy,
    island_hw: dict[str, str],
    incumbent_id: str,
) -> _Swept:
    result = perturb(item, value, metrics, context)
    return _swept_from(
        value, result.metrics, result.approximation, result.sim_error_override,
        context, spec, policy, island_hw, incumbent_id,
    )


def _swept_from(
    value: float,
    metrics: dict[str, PredictedMetrics],
    approximation: bool,
    sim_error_override: float | None,
    context: PerturbContext,
    spec: ServiceSpec,
    policy: MarginPolicy,
    island_hw: dict[str, str],
    incumbent_id: str,
) -> _Swept:
    """Judge one already-computed metric set.

    Split out of `_sweep_point` so `--resimulate-top` can score REAL predictions
    with the same arithmetic the closed form is scored with. If the two paths
    scored differently, E-B3 would be measuring its own harness.
    """
    feasible, plans, decisions = _plans_at(
        metrics, context, spec, policy, island_hw, sim_error_override
    )
    ranking = rank_plans(feasible, spec)
    feasible_ids = {p.candidate.id for p in feasible}

    if ranking.best is None:
        best_id, best_value = "", float("-inf")
    else:
        best_id, best_value = ranking.best.plan.candidate.id, ranking.best.value

    incumbent = plans.get(incumbent_id)
    incumbent_value: float | None = None
    overshoot = 0.0
    if incumbent is None:
        overshoot = 1.0
    elif incumbent.candidate.id in feasible_ids:
        incumbent_value = _value_of(incumbent, spec)
    else:
        # The decision this plan was actually judged under, override included.
        judgement = judge(incumbent, spec, decisions[incumbent.candidate.id])
        overshoot = judgement.report.worst_overshoot if judgement.report else 1.0
        if judgement.verdict is Verdict.UNMEASURED:
            # An undecidable plan has no "how narrowly it missed" -- `judge` says
            # so itself when it refuses to let one surface as `closest_plan` --
            # and a passing report's overshoot of 0.0 would price an
            # unverifiable recommendation at no regret at all.
            overshoot = 1.0

    return _Swept(
        value=value, best_plan_id=best_id, best_value=best_value,
        incumbent_value=incumbent_value, overshoot=overshoot,
        approximation=approximation,
        flip=bool(best_id) and best_id != incumbent_id,
        values=[_value_of(p, spec) for p in feasible],
    )


def _score(point: _Swept, penalty: float) -> tuple[float, float]:
    """(best_value, current_value) at one grid point, given the penalty.

    §2.5, as revised: an infeasible incumbent cannot be deployed, so what the
    operator actually gets is the best plan that CAN be, and the penalty prices
    how far outside the SLO the incumbent fell on top of that. The regret
    contribution is then `penalty * overshoot_ratio` and is >= 0 by
    construction.

    When NO candidate is feasible at this point the baseline is 0 rather than
    -inf, so the contribution is still `penalty * overshoot` instead of the
    point dropping silently out of the mean.
    """
    if point.incumbent_value is not None:
        return point.best_value, point.incumbent_value
    baseline = 0.0 if point.best_value == float("-inf") else point.best_value
    return baseline, baseline - penalty * point.overshoot


def _default_penalty(values: list[float]) -> float:
    """§2.5's default: the largest objective value observed across the grid.

    One unit of overshoot then costs a whole objective's worth. It is the knob
    that dominates the ranking, so the caller records it in provenance and E-B1
    re-runs at a tenth of it.
    """
    finite = [abs(v) for v in values if v not in (float("inf"), float("-inf"))]
    return max(finite) if finite else 1.0


def _analyze_one(
    item: UncertainInput, points: list[_Swept], penalty: float
) -> Sensitivity:
    cost_hours = item.cost.hours if item.cost is not None else None
    range_source = item.range.range_source
    if not points:
        return Sensitivity(
            input_id=item.id, kind=item.kind.value, flip=False, delta_regret=None,
            cost_hours=cost_hours, range_source=range_source,
            note=(
                "range is unbounded, so there is no interval to sweep and no regret "
                "to define - this input cannot be decided before measuring it"
            ),
        )

    grid: list[GridPoint] = []
    regrets: list[float] = []
    notes: list[str] = []
    flipped = any(p.flip for p in points)
    approximation = any(p.approximation for p in points)

    for point in points:
        best_value, current_value = _score(point, penalty)
        grid.append(GridPoint(
            value=point.value, best_plan_id=point.best_plan_id,
            best_value=best_value, current_value=current_value,
        ))
        regrets.append(best_value - current_value)
        if point.best_value == float("-inf"):
            notes.append(
                f"at {point.value:.6g} no candidate is feasible, so measuring this "
                f"input does not change the decision there"
            )

    delta_regret = sum(regrets) / len(regrets)
    note = (
        f"{len(points)} grid point(s) over "
        f"[{item.range.lo:.6g}, {item.range.hi:.6g}]"
        f"{' (approximate rule)' if approximation else ''}"
    )
    if notes:
        note += "; " + "; ".join(dict.fromkeys(notes))
    if range_source == "default":
        note += (
            f"; the interval is the {item.grade.value} DEFAULT from grades.yaml "
            f"({item.range.source}), not a measured width of this input"
        )
    return Sensitivity(
        input_id=item.id, kind=item.kind.value, flip=flipped,
        delta_regret=delta_regret, grid=grid, approximation=approximation,
        cost_hours=cost_hours, range_source=range_source, note=note,
    )


def analyze(
    output: PlannerOutput,
    registry: UncertainInputRegistry,
    metrics_by_candidate: JudgedMetrics,
    policy: MarginPolicy,
    spec: ServiceSpec,
    context: PerturbContext,
    island_hw: dict[str, str],
    *,
    grid: int = DEFAULT_GRID,
    slo_penalty: float | None = None,
    provenance: dict | None = None,
) -> list[Sensitivity]:
    """Flip and regret for every registry entry, sorted by what to measure first.

    `context` and `island_hw` are beyond the work order's signature and are not
    optional: the perturbation needs the topology and the candidate set, and the
    margin policy needs each island's hardware label. Passing them explicitly
    keeps this a pure function of its arguments.

    `provenance`, when given, receives the knobs that decide the answer -
    §2.5 requires the penalty to travel with the result, since it dominates the
    ranking.
    """
    if output.recommended is None:
        return [
            Sensitivity(
                input_id=item.id, kind=item.kind.value, flip=False,
                delta_regret=None,
                cost_hours=item.cost.hours if item.cost is not None else None,
                range_source=item.range.range_source,
                note="no recommendation to flip: the search found nothing feasible",
            )
            for item in registry.items
        ]

    incumbent_id = output.recommended.plan.candidate.id

    # Two passes, because §2.5's default penalty is "the largest objective value
    # observed ACROSS THE GRID", not at the nominal point. Pass one walks every
    # item's grid and records what it found; pass two turns those records into
    # regrets once the penalty is known. Perturbation is closed-form arithmetic,
    # so walking twice costs nothing worth optimising.
    swept: dict[str, list[_Swept]] = {}
    observed: list[float] = []
    for item in registry.items:
        swept[item.id] = [
            _sweep_point(item, value, metrics_by_candidate, context, spec, policy,
                         island_hw, incumbent_id)
            for value in build_grid(item, grid)
        ]
        for point in swept[item.id]:
            observed.extend(point.values)

    penalty = slo_penalty if slo_penalty is not None else _default_penalty(observed)
    if provenance is not None:
        provenance.setdefault("uncertainty", {}).update({
            "slo_penalty": penalty,
            "slo_penalty_source": "explicit" if slo_penalty is not None else "default",
            "grid": grid,
            "grid_weighting": "uniform",
        })

    out = [_analyze_one(item, swept[item.id], penalty) for item in registry.items]

    def sort_key(s: Sensitivity) -> tuple:
        # Worth-per-hour first, then raw regret, then the unbounded ones. An item
        # with no cost cannot be ordered by value for money and sorts after the
        # ones that can (§2.6), not before.
        if s.delta_regret is None:
            return (2, 0.0, 0.0, s.input_id)
        per_hour = s.regret_per_hour
        if per_hour is None:
            return (1, -s.delta_regret, 0.0, s.input_id)
        return (0, -per_hour, -s.delta_regret, s.input_id)

    return sorted(out, key=sort_key)


def unbounded(sensitivities: list[Sensitivity]) -> list[Sensitivity]:
    """The ones no sweep can settle - B3 lists these separately."""
    return [s for s in sensitivities if s.delta_regret is None]


def worth_measuring(sensitivities: list[Sensitivity]) -> list[Sensitivity]:
    """§2.5: only `dR_i > 0` entries are measurement-plan candidates."""
    return [s for s in sensitivities if s.delta_regret is not None and s.delta_regret > 0]


class Refinement(_Strict):
    """One item's closed-form ΔR beside the ΔR real simulation gives it.

    Both are computed on the two-point grid `{lo, hi}`, because that is all a
    resimulation buys; the closed form's own five-point figure stays in
    `Sensitivity.delta_regret` until it is replaced, and `closed_form` here is
    the like-for-like comparison E-B3 reports.
    """

    input_id: str
    kind: str
    closed_form: float
    resimulated: float
    #: Wall seconds the two simulations took, and how many candidates they ran.
    seconds: float = 0.0
    simulated: int = 0
    #: Set when this item could not be resimulated at all, with the reason.
    skipped: str = ""


def refine(
    sensitivities: list[Sensitivity],
    registry: UncertainInputRegistry,
    resimulate,
    metrics_by_candidate: JudgedMetrics,
    policy: MarginPolicy,
    spec: ServiceSpec,
    context: PerturbContext,
    island_hw: dict[str, str],
    incumbent_id: str,
    *,
    top: int,
    slo_penalty: float,
) -> tuple[list[Sensitivity], list[Refinement]]:
    """Replace the top `top` items' ΔR with one computed from real simulation.

    `resimulate(item)` returns an object with `.lo` and `.hi`, each carrying
    `.value`, `.metrics`, `.seconds` and `.simulated`; it raises to say an item
    has no simulator input to move. Passing it in rather than importing it keeps
    this module free of the predictor, which every one of its tests depends on.

    Items are taken in the order given - already `ΔR/cost` order out of
    `analyze` - and an item that cannot be resimulated is RECORDED and skipped
    without consuming one of the `top` slots, because refusing to check
    something is not the same as checking it.
    """
    items = {item.id: item for item in registry.items}
    refinements: list[Refinement] = []
    replaced: dict[str, Sensitivity] = {}
    done = 0
    for sensitivity in sensitivities:
        if done >= top:
            break
        item = items.get(sensitivity.input_id)
        if item is None or sensitivity.delta_regret is None:
            continue
        try:
            result = resimulate(item)
        except Exception as error:  # the reason IS the result, so it is recorded
            refinements.append(Refinement(
                input_id=sensitivity.input_id, kind=sensitivity.kind,
                closed_form=sensitivity.delta_regret,
                resimulated=sensitivity.delta_regret,
                skipped=str(error),
            ))
            continue

        exact_points = [
            _swept_from(
                endpoint.value, endpoint.metrics, False, None, context, spec,
                policy, island_hw, incumbent_id,
            )
            for endpoint in (result.lo, result.hi)
        ]
        closed_points = [
            _sweep_point(item, endpoint.value, metrics_by_candidate, context,
                         spec, policy, island_hw, incumbent_id)
            for endpoint in (result.lo, result.hi)
        ]
        exact = _analyze_one(item, exact_points, slo_penalty)
        closed = _analyze_one(item, closed_points, slo_penalty)
        refinements.append(Refinement(
            input_id=sensitivity.input_id, kind=sensitivity.kind,
            closed_form=closed.delta_regret or 0.0,
            resimulated=exact.delta_regret or 0.0,
            seconds=result.lo.seconds + result.hi.seconds,
            simulated=result.lo.simulated + result.hi.simulated,
        ))
        replaced[sensitivity.input_id] = sensitivity.model_copy(update={
            "delta_regret": exact.delta_regret,
            "flip": exact.flip,
            "approximation": False,
            "resimulated": True,
            "grid": exact.grid,
            "note": (
                f"{sensitivity.note} | RESIMULATED at both endpoints: closed form "
                f"said {closed.delta_regret:,.4g}, simulation says "
                f"{exact.delta_regret:,.4g} over {result.lo.simulated + result.hi.simulated} "
                f"candidate run(s) in {result.lo.seconds + result.hi.seconds:.1f} s"
            ),
        })
        done += 1

    out = [replaced.get(s.input_id, s) for s in sensitivities]

    def sort_key(s: Sensitivity) -> tuple:
        if s.delta_regret is None:
            return (2, 0.0, 0.0, s.input_id)
        per_hour = s.regret_per_hour
        if per_hour is None:
            return (1, -s.delta_regret, 0.0, s.input_id)
        return (0, -per_hour, -s.delta_regret, s.input_id)

    return sorted(out, key=sort_key), refinements
