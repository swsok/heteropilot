"""What to measure next, and what it buys (§2.5, §2.6, STEP B3).

B2 says what each uncertain input is worth knowing (`delta_regret`) and B1's
`costs.yaml` says what finding out costs. This turns the two into an ordered,
budgeted plan: the operator reads it top down, runs the named command, and the
regret it removes is stated rather than implied.

Three rules the work order sets and this enforces:

* only `delta_regret > 0` entries are candidates (§2.5). An input the sweep
  cannot move is not worth a server-hour however cheap it is;
* the order is `delta_regret / cost` - value for money, not raw value;
* an input with an unbounded range is NOT in the plan and NOT dropped either.
  It has no regret to compare because nothing bounds it, so it goes in its own
  list under "cannot be decided before measuring", which is a stronger statement
  than any ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from planner.uncertainty.grades import CostsTable
    from planner.uncertainty.sensitivity import Sensitivity


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeasurementItem(_Strict):
    """One measurement, and why it is worth its place in the queue."""

    rank: int
    input_id: str
    kind: str
    #: What to run, from costs.yaml's `method`.
    how_to_measure: str
    cost_hours: float | None = None
    #: True when the measurement owns the device for its duration, so a budget
    #: can count occupancy separately from elapsed time (§2.6).
    exclusive: bool = False
    delta_regret: float
    #: Regret per hour - the ordering key. None when the cost is unknown.
    regret_per_hour: float | None = None
    flip: bool = False
    #: True when the regret rests on a first-order perturbation rather than the
    #: planner's own arithmetic. Shown as "(approx)" wherever it is quoted.
    approximation: bool = False
    note: str = ""


class MeasurementPlan(_Strict):
    """An ordered queue of measurements under a budget."""

    items: list[MeasurementItem] = Field(default_factory=list)
    #: None means no budget was set and everything worth doing is in `items`.
    budget_hours: float | None = None
    #: Regret the planned items remove between them.
    covered_regret: float = 0.0
    #: Worth doing, but the budget ran out.
    uncovered: list[MeasurementItem] = Field(default_factory=list)
    #: Inputs with no sourced range: no regret can be defined, so they are
    #: neither ranked nor budgeted. Listed by id with their reason.
    undecidable: list[str] = Field(default_factory=list)
    #: Hours the planned items occupy a device exclusively, which a shared lab
    #: schedules differently from wall-clock.
    exclusive_hours: float = 0.0

    @property
    def total_hours(self) -> float:
        return sum(i.cost_hours or 0.0 for i in self.items)


def build(
    sensitivities: list[Sensitivity],
    costs: CostsTable | None = None,
    *,
    budget_hours: float | None = None,
) -> MeasurementPlan:
    """Order the worthwhile measurements and fit them into a budget.

    `sensitivities` arrive already sorted by `analyze`; the order is recomputed
    here anyway so a caller that filtered or concatenated lists still gets a
    correct ranking.

    An item whose cost is unknown is still planned - it is worth doing, and
    §2.6 says an unpriced kind sorts last rather than being dropped - but it
    consumes no budget, because pretending to know its cost would be the
    invention the whole work order forbids.
    """
    undecidable = [s.input_id for s in sensitivities if s.delta_regret is None]
    worthwhile = [
        s for s in sensitivities if s.delta_regret is not None and s.delta_regret > 0
    ]

    def order(s: Sensitivity) -> tuple:
        per_hour = s.regret_per_hour
        if per_hour is None:
            return (1, -s.delta_regret, s.input_id)  # type: ignore[operator]
        return (0, -per_hour, s.input_id)

    planned: list[MeasurementItem] = []
    deferred: list[MeasurementItem] = []
    spent = 0.0
    for rank, s in enumerate(sorted(worthwhile, key=order), start=1):
        row = costs.cost_for(s.kind) if costs is not None else None
        item = MeasurementItem(
            rank=rank,
            input_id=s.input_id,
            kind=s.kind,
            how_to_measure=row.method if row is not None else "(no method recorded)",
            cost_hours=s.cost_hours,
            exclusive=row.exclusive if row is not None else False,
            delta_regret=s.delta_regret,  # type: ignore[arg-type]
            regret_per_hour=s.regret_per_hour,
            flip=s.flip,
            approximation=s.approximation,
            note=s.note,
        )
        cost = item.cost_hours or 0.0
        if budget_hours is not None and spent + cost > budget_hours:
            deferred.append(item)
            continue
        spent += cost
        planned.append(item)

    # Ranks number the QUEUE, so a deferred item keeps the rank it would have
    # had: "number 3 did not fit" is more useful than a renumbered list that
    # hides where the budget ran out.
    return MeasurementPlan(
        items=planned,
        budget_hours=budget_hours,
        covered_regret=sum(i.delta_regret for i in planned),
        uncovered=deferred,
        undecidable=undecidable,
        exclusive_hours=sum(
            (i.cost_hours or 0.0) for i in planned if i.exclusive
        ),
    )
