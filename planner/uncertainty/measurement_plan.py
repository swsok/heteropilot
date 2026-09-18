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

Since S3 (D112) an input can be unranked for a THIRD reason, beside "no
interval" and "the interval moves nothing". An input that reaches a prediction
only through the simulator - an intra-island link's effective collective
bandwidth - has a known interval that no closed form can price, so it goes in
``needs_resimulation`` and ``--resimulate-top`` is what turns it into a rank.
Collapsing it into ``inert`` is what made V3's link look not worth measuring.

Since the domain-scoping work order S2 (D111), that last list is much shorter
and means something narrower. An input whose own (kind, grade) sources no width
now takes the GRADE'S DEFAULT range from ``grades.yaml`` and is ranked on it,
marked ``range_source: default``; only a grade with no default row at all is
undecidable. The path this closes cost a real measurement: V3's
``link_bw:pcie-a40a-02`` was a ``vendor_spec`` link with no sourced range, so it
sat in "cannot be decided" and out of the ranking, while being the input that
explained a -43.4 % TPOT error for 0.114 h of measurement.
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
    #: True when `--resimulate-top` checked this item by really simulating both
    #: ends of its range, so `delta_regret` is not a first-order stand-in.
    resimulated: bool = False
    #: ``default`` when the regret was computed over the grade's default range
    #: rather than a width measured for this input (S2, D111). Shown wherever
    #: the rank is: it is the difference between "this is worth 3 hours" and
    #: "this is worth 3 hours IF the assumed range is right".
    range_source: str = "sourced"
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
    #: Inputs with no range at all - their own (kind, grade) sources no width
    #: AND their grade has no default (S2, D111). No regret can be defined, so
    #: they are neither ranked nor budgeted. Listed by id.
    undecidable: list[str] = Field(default_factory=list)
    #: Inputs that WERE swept and moved nothing: a defined regret of zero.
    #: §2.5 keeps them out of the queue - an input the sweep cannot move is not
    #: worth a server-hour - but they are not undecidable either, and the
    #: difference is the whole of what a measurement would buy. Before S2 most
    #: of these were undecidable for want of a range, so the two answers were
    #: indistinguishable; now they are not, and each input lands in exactly one
    #: of items / uncovered / undecidable / inert / needs_resimulation.
    inert: list[str] = Field(default_factory=list)
    #: Inputs whose interval is known but whose regret no closed form can
    #: compute, because they reach a prediction only through the simulator
    #: (S3, D112: an intra-island link's effective collective bandwidth).
    #: The third way to be unranked, and the one that had been collapsed into
    #: `inert`: these may be the most valuable measurements on the list and the
    #: sweep cannot say. `--resimulate-top` moves them into the ranking.
    needs_resimulation: list[str] = Field(default_factory=list)
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
    needs_resim = [
        s.input_id for s in sensitivities
        if s.delta_regret is None and s.requires_resimulation
    ]
    undecidable = [
        s.input_id for s in sensitivities
        if s.delta_regret is None and not s.requires_resimulation
    ]
    worthwhile = [
        s for s in sensitivities if s.delta_regret is not None and s.delta_regret > 0
    ]
    inert = [
        s.input_id for s in sensitivities
        if s.delta_regret is not None and s.delta_regret <= 0
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
            resimulated=s.resimulated,
            range_source=s.range_source,
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
        inert=inert,
        needs_resimulation=needs_resim,
        exclusive_hours=sum(
            (i.cost_hours or 0.0) for i in planned if i.exclusive
        ),
    )
