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

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from planner.plan import OperatingPointRecord
from planner.predictor.calibration import AccuracyDomain

if TYPE_CHECKING:
    from planner.plan import CandidateConfig, PredictedMetrics
    from planner.predictor import SimResult
    from planner.predictor.calibration import CalibrationModel

#: Why a decision came out the way it did.
#:   in_domain    - interpolated from measured points at this operating point
#:   extrapolated - outside the measured points under `widen_error_bars`
#:   scalar       - one error for the whole bucket (no domain for the hardware,
#:                  or the caller chose GlobalMargin)
#:   unmeasured   - no measurement covers this operating point; no verdict
MarginStatus = Literal["in_domain", "extrapolated", "scalar", "unmeasured"]

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
            )

        worst = {"ttft": 0.0, "tpot": 0.0}
        covered: dict[str, set[str]] = {"ttft": set(), "tpot": set()}
        owned: set[str] = set()
        unmeasured: set[str] = set()
        closed_loop: set[str] = set()
        reasons: list[str] = []
        records: list[OperatingPointRecord] = []
        extrapolated: dict[str, float] = {}
        no_domain: list[str] = []
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

            mismatch = self._scope_mismatch(domain)
            if mismatch is not None:
                return self._unmeasured(
                    _busiest(sim),
                    f"{hw} accuracy domain was measured {mismatch}; refusing to apply "
                    f"an error measured under different conditions (§2.4.1 rev 2)",
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
        )

    # -- helpers --------------------------------------------------------------

    def _scope_mismatch(self, domain: AccuracyDomain) -> str | None:
        """Why this domain does not apply to the service, or None if it does.

        Each scope field is checked only when BOTH sides state it: an unscoped
        domain applies to anything, and a policy built without a model or shape
        cannot refuse on one.
        """
        if domain.workload_shape and self.shape and domain.workload_shape != self.shape:
            return f"on token mix {domain.workload_shape}, this service is {self.shape}"
        if domain.model and self.model and domain.model != self.model:
            return f"on model {domain.model}, this service runs {self.model}"
        if domain.variant and self.variant and domain.variant != self.variant:
            return f"at precision {domain.variant}, this service runs {self.variant}"
        return None

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
        concurrency: float | None, reason: str, *, unreadable: bool = False
    ) -> MarginDecision:
        return MarginDecision(
            ttft_percent=0.0, tpot_percent=0.0, status="unmeasured",
            ttft_status="unmeasured", tpot_status="unmeasured",
            unmeasured_metrics=["ttft", "tpot"],
            concurrency=concurrency, basis=reason, unreadable=unreadable,
        )
