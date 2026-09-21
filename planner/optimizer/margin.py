"""Per-candidate robustness margins (WORK_ORDER_uncertainty_planner.md §2.1, A3).

Until STEP 4 of the rps work order one pair of percentages was applied to every
candidate in a search. D22 showed why that is the wrong shape: the simulator's
TPOT error on RNGD is about 3 % at the concurrency the profile was fitted at and
about 18 % once the card is serving ~76 requests in flight. One scalar is
simultaneously too harsh on the light candidates and too generous on the heavy
ones - and, worse, it is applied just as confidently to candidates whose
operating point nobody ever measured.

A `MarginPolicy` decides the margin for ONE candidate from ITS operating point:

* `GlobalMargin` is the manual behaviour, unchanged and still the default;
* `AccuracyDomainMargin` reads each hardware's measured error
  (`calibration.AccuracyDomain`) at the served concurrency the run put that
  hardware at (`SimResult.operating_point`, per hardware and per P/D phase),
  takes the worst, and returns ``unmeasured`` when a domain under `refuse`
  does not cover the point, when the domain was measured on a different
  token mix, or when the hardware carries no calibration at all.

``unmeasured`` is not a failure - it is the absence of a verdict, and the search
charges it to `RejectionStage.OUTSIDE_CALIBRATION_DOMAIN` so it can never be
read as "infeasible" (deviations D33).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from planner.plan import OperatingPointRecord
from planner.predictor.calibration import (
    AccuracyDomain,
    CandidateConditions,
    ConditionCheck,
    DeviceBinding,
    DomainIndex,
)

if TYPE_CHECKING:
    from planner.plan import CandidateConfig, IslandAssignment, PredictedMetrics
    from planner.predictor import SimResult
    from planner.predictor.calibration import CalibrationModel

#: Why a decision came out the way it did.
#:   in_domain    - interpolated from measured points at this operating point
#:   extrapolated - outside the measured points under `widen_error_bars`
#:   scalar       - one error for the whole bucket (no domain for the hardware,
#:                  or the caller chose GlobalMargin)
#:   unmeasured   - no measurement covers this operating point; no verdict
MarginStatus = Literal["in_domain", "extrapolated", "scalar", "unmeasured"]

#: WHY a decision is `unmeasured`, when it is. The two are both epistemic and
#: they are not the same gap (domain-scoping S1, D110):
#:   outside_domain     - the right domain exists, the operating point is past
#:                        the end of its measured load axis;
#:   condition_mismatch - no domain was measured under this candidate's
#:                        conditions at all, so none may be consulted;
#:   no_domain          - the hardware carries no calibration whatsoever;
#:   unreadable         - the run reported no operating point to look up.
#: `search` maps the first to `OUTSIDE_CALIBRATION_DOMAIN` and the second to
#: `CALIBRATION_CONDITION_MISMATCH`; the last two keep the historical bucket.
RefusalKind = Literal[
    "", "outside_domain", "condition_mismatch", "no_domain", "unreadable"
]

#: Which SLO metric each P/D phase owns. In a disaggregated deployment the
#: PREFILL hardware determines TTFT and the DECODE hardware determines TPOT, so
#: each side is charged only for the metric it owns; "total" (an aggregated
#: deployment, or one device carrying both roles) owns both.
_PHASE_METRICS = {
    "prefill": ("ttft",),
    "decode": ("tpot",),
    "total": ("ttft", "tpot"),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarginDecision(_Strict):
    """The margin for one candidate, plus why.

    Coverage can be PARTIAL. A domain built by comparing a burst simulation
    against a closed-loop bench has TPOT points and no TTFT ones - sim TTFT there
    is inflated two orders of magnitude by queued requests that the real client
    had not yet sent, so there is nothing honest to compare. Such a decision
    margins TPOT and reports TTFT in `unmeasured_metrics`; the search keeps the
    verdict and carries the gap as a caveat (D33).
    """

    ttft_percent: float
    tpot_percent: float
    #: Overall: `unmeasured` only when no metric could be margined.
    status: MarginStatus
    ttft_status: MarginStatus = "unmeasured"
    tpot_status: MarginStatus = "unmeasured"
    #: Metric names with no margin available. Empty means full coverage.
    unmeasured_metrics: list[str] = Field(default_factory=list)
    #: The busiest hardware's served concurrency, when the run reported one.
    concurrency: float | None = None
    #: Which domain, and between which measured points, this came from.
    basis: str = ""
    #: Where each hardware kind ran and what it was charged (rps STEP 4.3).
    operating_point: list[OperatingPointRecord] = Field(default_factory=list)
    #: "accuracy_domain" | "manual" | "" - which of the two bounds produced the
    #: margin actually applied.
    source: str = ""
    #: hardware -> served concurrency, for margins EXTRAPOLATED under
    #: `widen_error_bars`. The search aggregates these into one note per hardware.
    extrapolated: dict[str, float] = Field(default_factory=dict)
    #: Hardware that carried no accuracy domain and fell back to a scalar bucket.
    no_domain: list[str] = Field(default_factory=list)
    #: The run produced no readable operating point at all.
    unreadable: bool = False
    #: Which gap this is, when `status` is `unmeasured`; "" otherwise.
    refusal: RefusalKind = ""
    #: Application conditions stated on BOTH sides that disagreed, e.g. ["tp"].
    #: Only ever non-empty for `refusal == "condition_mismatch"` or, under the
    #: `warn` policy, alongside a decision that was allowed through anyway.
    mismatch_fields: list[str] = Field(default_factory=list)
    #: The configuration a measurement would have to be taken at to decide this
    #: candidate - the patent's "additional measurement condition". None unless
    #: a condition mismatch was found.
    required_measurement: dict[str, Any] | None = None
    #: Conditions one side left UNSTATED, so the match test could not check
    #: them. Not a refusal; a caveat that the domain was applied on trust.
    condition_warnings: list[str] = Field(default_factory=list)

    @property
    def is_unmeasured(self) -> bool:
        """True only when nothing at all could be margined."""
        return self.status == "unmeasured"

    @property
    def is_partial(self) -> bool:
        return bool(self.unmeasured_metrics) and not self.is_unmeasured


class MarginPolicy(Protocol):
    """How a search turns one candidate's prediction into a robustness margin."""

    def decide(
        self,
        candidate: CandidateConfig,
        sim: SimResult,
        metrics: PredictedMetrics,
        island_hw: dict[str, str],
    ) -> MarginDecision:
        ...


class GlobalMargin:
    """One pair of percentages for every candidate - the behaviour before A3.

    Kept as the default so a search that does not opt in produces byte-identical
    output (absolute rule A4). `source` is "manual" whatever the percentages,
    which is what the frozen outputs carry.
    """

    def __init__(self, ttft_percent: float = 0.0, tpot_percent: float = 0.0) -> None:
        self.ttft_percent = ttft_percent
        self.tpot_percent = tpot_percent

    def decide(
        self,
        candidate: CandidateConfig,
        sim: SimResult,
        metrics: PredictedMetrics,
        island_hw: dict[str, str],
    ) -> MarginDecision:
        return MarginDecision(
            ttft_percent=self.ttft_percent,
            tpot_percent=self.tpot_percent,
            status="scalar",
            ttft_status="scalar",
            tpot_status="scalar",
            concurrency=_busiest(sim),
            basis=(
                f"global margin ttft={self.ttft_percent:g}% tpot={self.tpot_percent:g}%, "
                f"applied to every candidate regardless of operating point"
            ),
            source="manual",
        )


def _busiest(sim: SimResult) -> float | None:
    points = sim.operating_point
    if not points:
        return None
    return max(float(raw["concurrency"]) for raw in points.values())


def _scalar_percent(error: float) -> float:
    """A bucket's fractional mean error as a margin percentage.

    `ErrorStats.mean_error` is `(real - sim) / sim`: positive means the simulator
    predicted faster than reality. `max(0, e)` per §2.1 - a negative error means
    the simulator was already pessimistic there, and taking credit for that would
    shrink the prediction rather than harden it.
    """
    return max(0.0, error) * 100.0


class AccuracyDomainMargin:
    """Each hardware gets the error measured at ITS operating point (§2.4).

    A candidate spanning several hardware kinds - a P/D split, or a mixed
    deployment - takes the WORST margin per metric, and is `unmeasured` if ANY
    of its hardware is refused. Both follow from the margin being a claim about
    the whole prediction: it is only as trustworthy as its least trustworthy
    part.

    `ttft_floor` / `tpot_floor` are the manual `--*-margin-percent` values: an
    explicit instruction not to go below them, so the LARGER of floor and
    measured margin is used (rps STEP 4.3). `calibration` and `bucket` give the
    scalar fallback for hardware that carries a fitted bucket but no domain.
    """

    def __init__(
        self,
        domains: dict[str, AccuracyDomain],
        *,
        shape: str = "",
        model: str = "",
        variant: str = "",
        calibration: CalibrationModel | None = None,
        bucket: str = "",
        ttft_floor: float = 0.0,
        tpot_floor: float = 0.0,
        arrival_process: str = "unknown",
        device_binding: dict[str, DeviceBinding] | None = None,
        condition_mismatch: Literal["refuse", "warn"] = "refuse",
        index: DomainIndex | None = None,
    ) -> None:
        from planner.envelope import is_canonical_bucket, is_canonical_shape

        if shape and not is_canonical_shape(shape):
            raise ValueError(
                f"shape {shape!r} is not a canonical token-mix shape (§2.4.1); build it "
                f"with envelope.workload_shape(spec)"
            )
        if bucket and not is_canonical_bucket(bucket):
            raise ValueError(
                f"bucket {bucket!r} is not a canonical bucket key (§2.4.1); build it with "
                f"envelope.workload_bucket(spec) - a human label is never a lookup key"
            )
        self.domains = domains
        self.shape = shape
        self.model = model
        self.variant = variant
        self.calibration = calibration
        self.bucket = bucket
        self.ttft_floor = ttft_floor
        self.tpot_floor = tpot_floor
        #: How the caller offers load. `plan` replays an arrival trace, so the
        #: CLI passes `open_loop`; left `unknown` the field is not compared.
        self.arrival_process = arrival_process
        #: island id -> how that island's host is bound, from the cluster spec.
        #: Keyed by ISLAND, not by hardware: one cluster can hold a bound and an
        #: unbound node of the same accelerator, and collapsing them would hide
        #: the very difference this field exists to catch. Missing or `unknown`
        #: means the check skips it and flags it.
        self.device_binding = device_binding or {}
        #: `refuse` (default, symmetric with D33's `outside_domain: refuse`) or
        #: `warn`, which applies the domain anyway and records the mismatch.
        #: `warn` exists to measure what the refusal costs, not to plan with.
        self.condition_mismatch = condition_mismatch
        #: The registry, used only to say what DOES exist in a refusal.
        self.index = index

    # -- the application conditions (S1) --------------------------------------

    def _conditions(
        self, candidate: CandidateConfig, hardware: str, island_hw: dict[str, str]
    ) -> list[CandidateConditions]:
        """What this candidate presents to `hardware`, one entry per assignment.

        PER ASSIGNMENT rather than aggregated, because a candidate may put two
        islands of one hardware at different parallelism, and there is no honest
        single tp for that pair: each island must match the domain on its own.
        `islands` is the one genuinely per-hardware field - how many islands of
        this hardware the candidate lights up - and every entry carries it.

        With no island->hardware attribution (the map is optional on `decide`)
        the parallelism is NOT STATED rather than assumed: the check then skips
        those fields and flags them, which is the same treatment an unrecorded
        binding gets. The CLI always supplies the map.
        """
        def conditions(assignment: IslandAssignment | None, islands: int | None
                       ) -> CandidateConditions:
            return CandidateConditions(
                hardware=hardware,
                model=self.model,
                variant=self.variant,
                workload_shape=self.shape,
                arrival_process=self.arrival_process,
                tp=None if assignment is None else assignment.tp_size,
                pp=None if assignment is None else assignment.pp_size,
                dp=None if assignment is None else assignment.dp_replicas,
                islands=islands,
                device_binding=(
                    "unknown" if assignment is None
                    else self.device_binding.get(assignment.island_id, "unknown")
                ),
            )

        mine = [a for a in candidate.assignments if island_hw.get(a.island_id) == hardware]
        if not mine:
            return [conditions(None, None)]
        islands = len({a.island_id for a in mine})
        return [conditions(a, islands) for a in mine]

    @staticmethod
    def _merge(checks: list[tuple[CandidateConditions, ConditionCheck]]) -> ConditionCheck:
        mismatch: list[str] = []
        skipped: list[str] = []
        detail: list[str] = []
        for _cond, check in checks:
            for name in check.mismatch:
                if name not in mismatch:
                    mismatch.append(name)
            for name in check.skipped:
                if name not in skipped:
                    skipped.append(name)
            for line in check.detail:
                if line not in detail:
                    detail.append(line)
        return ConditionCheck(tuple(mismatch), tuple(skipped), tuple(detail))

    def _required_measurement(
        self, hardware: str, conds: list[CandidateConditions]
    ) -> dict[str, Any]:
        out = dict(conds[0].required_measurement)
        out["hardware"] = hardware
        shapes = [(c.tp, c.pp, c.dp) for c in conds]
        if len(set(shapes)) > 1:
            # No single tp/pp/dp describes this hardware's share of the
            # candidate, so state the islands rather than one of them.
            for name in ("tp", "pp", "dp"):
                out.pop(name, None)
            out["per_island"] = [
                {"tp": tp, "pp": pp, "dp": dp} for tp, pp, dp in shapes
            ]
        return out

    def _known_conditions(self, hardware: str) -> str:
        if self.index is None:
            return ""
        rows = self.index.for_hardware(hardware)
        if not rows:
            return ""
        return " | registered domains for this hardware: " + "; ".join(
            f"{r.path} ({r.conditions_summary()})" for r in rows
        )

    # -- the decision ---------------------------------------------------------

    def decide(
        self,
        candidate: CandidateConfig,
        sim: SimResult,
        metrics: PredictedMetrics,
        island_hw: dict[str, str],
    ) -> MarginDecision:
        # Read off the SimResult, which carries the operating point whether the
        # run just happened or came from the cache. Re-deriving it from
        # `artifacts` is what once made the margin vanish on a cache hit.
        points = sim.operating_point
        if not points:
            return self._unmeasured(
                None,
                "the predictor reported no operating point for this run, so no "
                "accuracy domain can be consulted - the candidate is undecidable, "
                "not infeasible",
                unreadable=True,
                refusal="unreadable",
            )

        worst = {"ttft": 0.0, "tpot": 0.0}
        covered: dict[str, set[str]] = {"ttft": set(), "tpot": set()}
        owned: set[str] = set()
        unmeasured: set[str] = set()
        closed_loop: set[str] = set()
        #: Hardware whose domain declares it was fitted on a p50. The margin
        #: below is applied to a p99 (D101), so such a domain is being consulted
        #: on a basis it was not measured on. "" means NOT STATED and earns no
        #: warning: every domain written before 2026-09-18 is p50 in fact and
        #: says nothing, and warning on silence would fire on every run while
        #: telling nobody anything new.
        p50_fitted: set[str] = set()
        reasons: list[str] = []
        records: list[OperatingPointRecord] = []
        extrapolated: dict[str, float] = {}
        no_domain: list[str] = []
        condition_warnings: list[str] = []
        mismatched: list[str] = []
        consulted = False

        for hw, raw in sorted(points.items()):
            conc = float(raw["concurrency"])
            phase = str(raw["phase"])
            metrics_owned = _PHASE_METRICS.get(phase, ("ttft", "tpot"))
            owned.update(metrics_owned)
            domain = self.domains.get(hw)

            if domain is None:
                entry = self._scalar_entry(hw)
                if entry is None:
                    return self._unmeasured(
                        _busiest(sim),
                        f"no accuracy domain for {hw}"
                        + self._available(hw)
                        + ": the simulator's error on this hardware at this workload "
                          "has never been measured",
                        refusal="no_domain",
                    )
                records.append(OperatingPointRecord(hardware=hw, concurrency=conc, phase=phase))
                no_domain.append(hw)
                for metric in metrics_owned:
                    stats = entry.ttft if metric == "ttft" else entry.tpot
                    worst[metric] = max(worst[metric], _scalar_percent(stats.mean_error))
                    covered[metric].add("scalar")
                reasons.append(
                    f"{hw}[{phase}]: scalar mode - no accuracy domain, so the whole-bucket "
                    f"mean error of {entry.workload_bucket} applies at every concurrency"
                )
                consulted = True
                continue

            # The APPLICATION CONDITIONS, before the operating point: may this
            # domain be consulted for this candidate at all? (S1, D110.) V3 is
            # the case - a tp=1 domain answering for a tp=4 island, with a
            # 1.13 % margin against a measured -44.6 %.
            conds = self._conditions(candidate, hw, island_hw)
            check = self._merge([(c, domain.check_conditions(c)) for c in conds])
            if check.mismatch:
                fields = ", ".join(check.mismatch)
                detail = "; ".join(check.detail)
                if self.condition_mismatch == "refuse":
                    return self._unmeasured(
                        _busiest(sim),
                        f"{hw}: the accuracy domain was measured under different "
                        f"conditions ({fields}) - {detail}. It may not be consulted "
                        f"here, so this candidate is unmeasured AT ITS OWN "
                        f"CONFIGURATION, not infeasible (S1, D110)"
                        + self._known_conditions(hw),
                        refusal="condition_mismatch",
                        mismatch_fields=list(check.mismatch),
                        required_measurement=self._required_measurement(hw, conds),
                    )
                mismatched.extend(f for f in check.mismatch if f not in mismatched)
                condition_warnings.append(
                    f"{hw}: APPLIED ACROSS A CONDITION MISMATCH ({fields}) - {detail}. "
                    f"policy condition_mismatch=warn, so the margin below is an error "
                    f"measured under conditions this candidate does not meet"
                )
            if check.skipped:
                condition_warnings.append(
                    f"{hw}: unchecked application conditions "
                    f"({', '.join(check.skipped)}) - one side does not state them, "
                    f"so the domain is applied on trust for those fields"
                )

            errs = domain.errors_at(conc)
            if errs is None:
                return self._unmeasured(
                    _busiest(sim),
                    f"served concurrency {conc:.4g} on {hw} outside its accuracy domain "
                    f"[{domain.conc_min:.4g}, {domain.conc_max:.4g}] (policy refuse); "
                    f"measure the envelope at c>={conc:.0f} or set outside_domain: "
                    f"widen_error_bars to accept an extrapolation",
                )
            ttft_err, tpot_err = errs
            inside = domain.in_domain(conc)
            records.append(OperatingPointRecord(
                hardware=hw, concurrency=conc, phase=phase,
                tpot_error_pct=tpot_err, ttft_error_pct=ttft_err,
                in_calibration_domain=inside,
            ))
            kind = "in_domain" if inside else "extrapolated"
            if not inside:
                extrapolated[hw] = conc
            for metric, err in (("ttft", ttft_err), ("tpot", tpot_err)):
                if metric not in metrics_owned:
                    continue
                if err is None:
                    unmeasured.add(metric)
                    continue
                worst[metric] = max(worst[metric], AccuracyDomain.margin_from_error(err))
                covered[metric].add(kind)
            if domain.arrival_process == "closed_loop":
                closed_loop.add(hw)
            if domain.compared_metric == "tpot_p50":
                p50_fitted.add(hw)
            reasons.append(f"{hw}[{phase}]: {domain.basis_at(conc)}")
            consulted = True

        # A metric is unmeasured only if NO hardware that owns it measured it.
        unmeasured = {m for m in unmeasured if not covered[m] and m in owned}

        def metric_status(metric: str) -> MarginStatus:
            kinds = covered[metric]
            if not kinds:
                return "unmeasured"
            # Weakest wins, as with the hardware: one scalar leg makes the
            # metric's margin a scalar one, one extrapolated leg an extrapolation.
            if "scalar" in kinds:
                return "scalar"
            if "extrapolated" in kinds:
                return "extrapolated"
            return "in_domain"

        ttft_status, tpot_status = metric_status("ttft"), metric_status("tpot")
        if ttft_status == "unmeasured" and tpot_status == "unmeasured":
            return self._unmeasured(
                _busiest(sim),
                "in domain, but neither metric has a measured point here: "
                + "; ".join(reasons),
            )
        # Overall status: weakest wins across the two metrics.
        status: MarginStatus
        if "scalar" in (ttft_status, tpot_status):
            status = "scalar"
        elif "extrapolated" in (ttft_status, tpot_status):
            status = "extrapolated"
        else:
            status = "in_domain"

        basis = "; ".join(reasons)
        if unmeasured:
            basis += (
                f" | NO margin for {', '.join(sorted(unmeasured))}: unmeasured in this "
                f"domain, so that check ran unmargined"
            )
        if closed_loop:
            basis += (
                f" | measured closed-loop ({', '.join(sorted(closed_loop))}), so its "
                f"TTFT does not transfer to an open-loop deployment (D19)"
            )
        if p50_fitted:
            basis += (
                f" | fitted on a p50 ({', '.join(sorted(p50_fitted))}) and applied "
                f"to a p99 (D101): the margin is charged on a basis it was not "
                f"measured on. The domain is still consulted -- this is a warning, "
                f"not a refusal, because the mismatch is a known wart and not an "
                f"unmeasured configuration"
            )
        for warning in condition_warnings:
            basis += f" | {warning}"

        # A hand-set floor is an explicit instruction not to go below it: the
        # LARGER wins. `source` follows the rps STEP 4.3 rule exactly - the
        # domain is credited when its TPOT margin is at least the floor.
        ttft_used = max(self.ttft_floor, worst["ttft"])
        tpot_used = max(self.tpot_floor, worst["tpot"])
        if ttft_used > worst["ttft"] or tpot_used > worst["tpot"]:
            basis += (
                f" | manual floor ttft={self.ttft_floor:g}% tpot={self.tpot_floor:g}% "
                f"binds where it exceeds the measured margin"
            )
        source = "accuracy_domain" if consulted and worst["tpot"] >= self.tpot_floor else "manual"

        return MarginDecision(
            ttft_percent=ttft_used,
            tpot_percent=tpot_used,
            status=status,
            ttft_status=ttft_status,
            tpot_status=tpot_status,
            unmeasured_metrics=sorted(unmeasured),
            concurrency=_busiest(sim),
            basis=basis,
            operating_point=records,
            source=source,
            extrapolated=extrapolated,
            no_domain=no_domain,
            mismatch_fields=mismatched,
            condition_warnings=condition_warnings,
        )

    # -- helpers --------------------------------------------------------------

    def _scalar_entry(self, hardware: str):
        if self.calibration is None or not self.bucket:
            return None
        cal = self.calibration.get(hardware)
        if cal is None:
            return None
        return cal.bucket_for(self.bucket)

    def _available(self, hardware: str) -> str:
        if self.calibration is None:
            return ""
        cal = self.calibration.get(hardware)
        if cal is None or not cal.errors:
            return ""
        fitted = ", ".join(cal.available_buckets())
        return f" at bucket {self.bucket or '(none)'} (fitted: {fitted})"

    @staticmethod
    def _unmeasured(
        concurrency: float | None,
        reason: str,
        *,
        unreadable: bool = False,
        refusal: RefusalKind = "outside_domain",
        mismatch_fields: list[str] | None = None,
        required_measurement: dict[str, Any] | None = None,
    ) -> MarginDecision:
        return MarginDecision(
            ttft_percent=0.0, tpot_percent=0.0, status="unmeasured",
            ttft_status="unmeasured", tpot_status="unmeasured",
            unmeasured_metrics=["ttft", "tpot"],
            concurrency=concurrency, basis=reason, unreadable=unreadable,
            refusal=refusal,
            mismatch_fields=mismatch_fields or [],
            required_measurement=required_measurement,
        )
