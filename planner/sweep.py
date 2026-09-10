"""RPS as a search axis: switchover table and backend crossovers.

`docs/rps_aware_planning_design.md` §6, `WORK_ORDER_rps_aware.md` rev 2 STEP 4.5.

The planner answers "what should I deploy?" for one arrival rate. The question
the design says the system owes the user is "at what arrival rate does the answer
CHANGE?" -- which a scalar profile cannot express at all, because a scalar has no
crossover. Running the plan once per rate and diffing the winners is the whole
mechanism.

Two things are reported and they are not the same:

  * a **switchover** is a change of recommended plan between adjacent rates. It is
    observed, not inferred.
  * a **crossover** is the rate at which two backends' tok/J curves cross. It is
    estimated by linear interpolation between the two bracketing measured rates
    and is labelled `estimated` everywhere it appears, because nothing was run at
    the crossing point itself.
"""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from planner.plan import PlannerOutput


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SwitchoverRow(_Strict):
    """One arrival rate's answer, with everything needed to read it critically."""

    rps: float
    feasible: bool
    plan_id: str | None = None
    candidate_id: str | None = None
    backend_mix: str | None = None
    accelerators: int | None = None
    tokens_per_joule: float | None = None
    average_power_w: float | None = None
    p99_tpot_ms: float | None = None
    #: Served concurrency per hardware at this rate, from the run itself.
    operating_point: list[dict[str, Any]] = Field(default_factory=list)
    applied_tpot_margin_pct: float | None = None
    #: "measured" when every operating point sat inside its calibration domain,
    #: "extrapolated" when any did not, "unknown" when no domain applied. The
    #: label travels with the row so a table cannot be read without it.
    validity: str = "unknown"
    rejected_summary: dict[str, int] = Field(default_factory=dict)


class Crossover(_Strict):
    """Where two backends swap places, and how that was arrived at."""

    between_rps: tuple[float, float]
    from_backend: str
    to_backend: str
    #: Linear interpolation between the bracketing rates. Never measured.
    estimated_rps: float | None = None
    basis: str = "recommended plan changed backend"
    label: str = "estimated"


class SweepOutput(_Strict):
    service_model: str
    cluster_id: str
    rps_values: list[float]
    switchover: list[SwitchoverRow] = Field(default_factory=list)
    crossovers: list[Crossover] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)


def _backend_mix(output: PlannerOutput) -> str | None:
    if output.recommended is None:
        return None
    ids = [a.island_id for a in output.recommended.plan.candidate.assignments]
    backends = sorted({i.split("-", 1)[0] for i in ids})
    return "+".join(backends)


def _validity(output: PlannerOutput) -> str:
    if output.recommended is None:
        return "unknown"
    points = output.recommended.plan.operating_point
    if not points:
        return "unknown"
    flags = [p.in_calibration_domain for p in points]
    if any(f is None for f in flags):
        return "unknown"
    return "measured" if all(flags) else "extrapolated"


def row_for(rps: float, output: PlannerOutput) -> SwitchoverRow:
    if output.recommended is None:
        return SwitchoverRow(rps=rps, feasible=False,
                             rejected_summary=dict(output.rejected_summary))
    plan = output.recommended.plan
    m = plan.predicted
    return SwitchoverRow(
        rps=rps,
        feasible=True,
        plan_id=plan.plan_id,
        candidate_id=plan.candidate.id,
        backend_mix=_backend_mix(output),
        accelerators=plan.active_accelerators,
        tokens_per_joule=m.tokens_per_joule,
        average_power_w=m.average_power_w,
        p99_tpot_ms=m.p99_tpot_ms,
        operating_point=[p.model_dump() for p in plan.operating_point],
        applied_tpot_margin_pct=plan.robust_margin_tpot_percent,
        validity=_validity(output),
        rejected_summary=dict(output.rejected_summary),
    )


def find_crossovers(rows: list[SwitchoverRow]) -> list[Crossover]:
    """Adjacent rates whose winning backend differs.

    The estimated crossing rate uses the two rows' tok/J. It is a straight line
    between two points on curves that are not straight, so it is an estimate and
    says so; when either row lacks tok/J the crossing is reported without one
    rather than with a fabricated midpoint.
    """
    out: list[Crossover] = []
    usable = [r for r in rows if r.feasible and r.backend_mix]
    for lo, hi in itertools.pairwise(usable):
        if lo.backend_mix == hi.backend_mix:
            continue
        est: float | None = None
        if lo.tokens_per_joule is not None and hi.tokens_per_joule is not None:
            gap_lo = lo.tokens_per_joule
            gap_hi = hi.tokens_per_joule
            if gap_lo != gap_hi:
                # Where the two efficiencies would meet if both moved linearly in
                # rps between the bracketing points.
                t = gap_lo / (gap_lo - gap_hi) if (gap_lo - gap_hi) else 0.5
                t = min(max(t, 0.0), 1.0)
                est = lo.rps + t * (hi.rps - lo.rps)
        out.append(Crossover(
            between_rps=(lo.rps, hi.rps),
            from_backend=lo.backend_mix or "",
            to_backend=hi.backend_mix or "",
            estimated_rps=est,
        ))
    return out


_RULE = "-" * 78


def render(sweep: SweepOutput) -> str:
    lines = [
        f"--- RPS switchover {_RULE[:60]}",
        f"service : {sweep.service_model}",
        f"cluster : {sweep.cluster_id}",
        "",
        f"{'rps':>7}  {'recommended':<44} {'backend':<12} {'acc':>4} "
        f"{'tok/J':>8} {'avg W':>8} {'p99 TPOT':>9} {'margin':>7}  validity",
    ]
    for r in sweep.switchover:
        if not r.feasible:
            why = ", ".join(f"{k}={v}" for k, v in sorted(r.rejected_summary.items()))
            lines.append(f"{r.rps:>7}  INFEASIBLE  ({why or 'no reason recorded'})")
            continue
        def num(v: float | None, nd: int) -> str:
            return "-" if v is None else f"{round(v, nd)}"

        lines.append(
            f"{r.rps:>7}  {(r.candidate_id or '')[:44]:<44} {(r.backend_mix or ''):<12} "
            f"{r.accelerators or 0:>4} {num(r.tokens_per_joule, 3):>8} "
            f"{num(r.average_power_w, 1):>8} {num(r.p99_tpot_ms, 2):>9} "
            f"{round(r.applied_tpot_margin_pct or 0.0, 2):>6}%  {r.validity}"
        )
    lines.append("")
    if sweep.crossovers:
        lines.append(f"--- Crossovers {_RULE[:64]}")
        for c in sweep.crossovers:
            est = "" if c.estimated_rps is None else f" near {c.estimated_rps:.3g} rps"
            lines.append(
                f"  {c.from_backend} -> {c.to_backend} between {c.between_rps[0]} and "
                f"{c.between_rps[1]} rps{est}  [{c.label}]"
            )
    else:
        lines.append(f"--- Crossovers {_RULE[:64]}")
        lines.append("  none: the same backend wins at every rate swept.")
    for caveat in sweep.caveats:
        lines.append(f"  - {caveat}")
    return "\n".join(lines)
