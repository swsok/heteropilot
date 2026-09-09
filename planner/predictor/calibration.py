"""Sim-vs-real calibration and robust planning margins (work order §5.8).

The first calibration model is a per-hardware linear correction::

    real_ttft ~= alpha_hw * sim_ttft + beta_hw
    real_tpot ~= gamma_hw * sim_tpot + delta_hw

fitted from the ``bench/`` validation summaries (the ``vLLM`` and ``Sim`` columns
of ``summary.txt``). Per (hardware, workload_bucket) it also stores an error
distribution - ``mean_error``, ``p95_abs_error``, ``worst_error``,
``sample_count`` - from which a *robust* metric is derived::

    robust_metric = predicted * (1 + p95_error)

so feasibility can be checked against a pessimistic value. Per HANDOVER.md §7,
the correction sign is hardware-dependent (A5000 under-predicts, RTXPRO6000
over-predicts), so calibration is always per-hardware and never a single global
fudge factor.

Absolute rule 3 governs this file: nothing here invents numbers. With no bench
data the model is the identity (alpha=1, beta=0) and every margin is 0, labelled
``source="identity"``. Real numbers come only from real summaries.

Applying calibration or robust margins to planning is **opt-in**: the `plan`
command does not import this module, so its default output - and the frozen
golden tests - are unaffected. A caller wires it in explicitly via
`apply_robust_margins`.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from planner.envelope import workload_bucket
from planner.plan import DeploymentPlan
from planner.spec import ServiceSpec
from planner.util.percentile import percentile

IDENTITY = "identity"
INSUFFICIENT = "insufficient_data"
MEASURED = "measured"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LinearFit(_Strict):
    """A one-dimensional ``real = alpha * sim + beta`` correction."""

    alpha: float = 1.0
    beta: float = 0.0
    sample_count: int = 0
    source: str = IDENTITY

    def apply(self, sim_value: float) -> float:
        return self.alpha * sim_value + self.beta


class ErrorStats(_Strict):
    """Prediction-error distribution for one metric on one workload bucket.

    Error is fractional under-prediction of the simulator relative to reality,
    ``(real - sim) / sim``: positive means the simulator predicted faster than
    reality, so a robust plan must inflate the prediction by that much.
    """

    mean_error: float = 0.0
    p95_abs_error: float = 0.0
    worst_error: float = 0.0
    sample_count: int = 0

    @property
    def robust_margin_percent(self) -> float:
        """The p95 error as a percentage, for a DeploymentPlan robust margin."""
        return self.p95_abs_error * 100.0


class BucketError(_Strict):
    """Per-metric error stats for one workload bucket."""

    workload_bucket: str
    ttft: ErrorStats = Field(default_factory=ErrorStats)
    tpot: ErrorStats = Field(default_factory=ErrorStats)


class AccuracyPoint(_Strict):
    """The simulator's SIGNED error at one served concurrency.

    Sign convention: `(sim - measured) / measured * 100`. Negative TPOT means the
    simulator is optimistic -- it predicts a shorter time per token than the
    hardware delivers -- which is the direction that produced D22.
    """

    conc: float = Field(gt=0)
    tpot_err_pct: float
    tput_err_pct: float | None = None
    ttft_err_pct: float | None = None
    note: str = ""


class AccuracyDomain(_Strict):
    """Where a predictor's error has been measured, and what it was.

    `docs/rps_aware_planning_design.md` §5, the most transferable lesson from D22:
    the card fixture's winner passed a 50 ms TPOT SLO at a predicted 48.41 ms, and
    48.41 x 1.18 = 57.1. The winner was not optimistic, it was infeasible, and
    nothing in the pipeline could see it because the 18 % was a fact about the
    simulator that the simulator did not carry.

    Outside the measured points the policy is `widen_error_bars`: take the nearest
    measured point's error and add |slope| x distance, uncapped. That is
    deliberately pessimistic and deliberately unbounded -- an extrapolated margin
    should get worse the further it reaches, so that a candidate far outside the
    domain is rejected by its own uncertainty rather than by a guess.
    """

    fitted_at_concurrency: float = Field(gt=0)
    points: list[AccuracyPoint] = Field(min_length=1)
    outside_domain: Literal["widen_error_bars", "refuse"] = "widen_error_bars"
    source: str = "measured"
    note: str = ""

    @model_validator(mode="after")
    def _sorted(self) -> AccuracyDomain:
        concs = [p.conc for p in self.points]
        if concs != sorted(concs):
            raise ValueError(f"accuracy_domain points must be sorted by conc: {concs}")
        if len(set(concs)) != len(concs):
            raise ValueError(f"duplicate concurrency in accuracy_domain: {concs}")
        return self

    @property
    def conc_min(self) -> float:
        return self.points[0].conc

    @property
    def conc_max(self) -> float:
        return self.points[-1].conc

    def _err_at(self, conc: float, field: str) -> float | None:
        vals = [(p.conc, getattr(p, field)) for p in self.points
                if getattr(p, field) is not None]
        if not vals:
            return None
        if len(vals) == 1:
            # One point: no slope to widen along, so the error is carried flat and
            # `in_domain` is what tells the caller it is an extrapolation.
            return vals[0][1]
        if conc <= vals[0][0]:
            (x0, y0), (x1, y1) = vals[0], vals[1]
            slope = (y1 - y0) / (x1 - x0)
            return y0 - abs(slope) * (x0 - conc)
        if conc >= vals[-1][0]:
            (x0, y0), (x1, y1) = vals[-2], vals[-1]
            slope = (y1 - y0) / (x1 - x0)
            return y1 - abs(slope) * (conc - x1)
        for (x0, y0), (x1, y1) in itertools.pairwise(vals):
            if x0 <= conc <= x1:
                t = (conc - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        raise AssertionError("unreachable: conc is bracketed")

    def in_domain(self, conc: float) -> bool:
        return self.conc_min <= conc <= self.conc_max

    def tpot_error_at(self, conc: float) -> float:
        """Signed simulator error in TPOT, in percent, at this operating point."""
        v = self._err_at(conc, "tpot_err_pct")
        assert v is not None  # tpot_err_pct is required on every point
        return v

    def ttft_error_at(self, conc: float) -> float | None:
        return self._err_at(conc, "ttft_err_pct")

    @staticmethod
    def margin_from_error(err_pct: float | None) -> float:
        """The inflation to apply, in percent: only where the simulator is optimistic.

        A pessimistic predictor is left alone. Deflating a prediction because the
        simulator ran slow would make plans look better than the hardware measured,
        which is the exact direction of the D22 retraction -- so the margin is
        one-sided by construction.
        """
        if err_pct is None:
            return 0.0
        return max(0.0, -err_pct)

    def tpot_margin_pct(self, conc: float) -> float:
        return self.margin_from_error(self.tpot_error_at(conc))

    def ttft_margin_pct(self, conc: float) -> float:
        return self.margin_from_error(self.ttft_error_at(conc))


class HardwareCalibration(_Strict):
    """Linear fits and error distributions for one hardware kind."""

    hardware: str
    ttft: LinearFit = Field(default_factory=LinearFit)
    tpot: LinearFit = Field(default_factory=LinearFit)
    #: keyed by workload_bucket
    errors: dict[str, BucketError] = Field(default_factory=dict)
    #: Where this predictor's error has been measured (design §5). Optional: a
    #: hardware without one gets margin 0 and a caveat, never a guessed margin.
    accuracy_domain: AccuracyDomain | None = None


class CalibrationModel(_Strict):
    """The calibration store, keyed by hardware label (work order §5.8)."""

    hardware: dict[str, HardwareCalibration] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def identity(cls) -> CalibrationModel:
        """An empty model: every correction is the identity, every margin 0."""
        return cls()

    def get(self, hardware: str) -> HardwareCalibration | None:
        return self.hardware.get(hardware)

    def apply_ttft(self, hardware: str, sim_ttft_ms: float) -> float:
        cal = self.hardware.get(hardware)
        return cal.ttft.apply(sim_ttft_ms) if cal else sim_ttft_ms

    def apply_tpot(self, hardware: str, sim_tpot_ms: float) -> float:
        cal = self.hardware.get(hardware)
        return cal.tpot.apply(sim_tpot_ms) if cal else sim_tpot_ms

    def margins(self, hardware: str, bucket: str) -> tuple[float, float]:
        """Robust (ttft_percent, tpot_percent) for a hardware/bucket pair.

        Returns ``(0.0, 0.0)`` when no data exists - never a guessed margin.
        """
        cal = self.hardware.get(hardware)
        if cal is None:
            return (0.0, 0.0)
        bucket_error = cal.errors.get(bucket)
        if bucket_error is None:
            return (0.0, 0.0)
        return (
            bucket_error.ttft.robust_margin_percent,
            bucket_error.tpot.robust_margin_percent,
        )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_linear(sim: Sequence[float], real: Sequence[float]) -> LinearFit:
    """Least-squares fit of ``real = alpha * sim + beta``. Pure.

    With fewer than two points, or when every ``sim`` value is identical (no
    slope is determinable), fall back to a pure offset ``alpha=1`` and let
    ``beta`` absorb the mean difference. That is honest about what one point can
    say, and never fabricates a slope.
    """
    if len(sim) != len(real):
        raise ValueError("sim and real must be the same length")
    n = len(sim)
    if n == 0:
        return LinearFit(source=IDENTITY)
    mean_sim = sum(sim) / n
    mean_real = sum(real) / n
    denom = sum((x - mean_sim) ** 2 for x in sim)
    if n < 2 or denom == 0.0:
        return LinearFit(
            alpha=1.0, beta=mean_real - mean_sim, sample_count=n, source=INSUFFICIENT
        )
    cov = sum((x - mean_sim) * (y - mean_real) for x, y in zip(sim, real, strict=True))
    alpha = cov / denom
    beta = mean_real - alpha * mean_sim
    return LinearFit(alpha=alpha, beta=beta, sample_count=n, source=MEASURED)


def compute_error_stats(sim: Sequence[float], real: Sequence[float]) -> ErrorStats:
    """Fractional-error distribution of a simulator against reality. Pure."""
    if len(sim) != len(real):
        raise ValueError("sim and real must be the same length")
    errors = [(r - s) / s for s, r in zip(sim, real, strict=True) if s != 0.0]
    if not errors:
        return ErrorStats()
    abs_errors = [abs(e) for e in errors]
    worst = max(errors, key=abs)
    return ErrorStats(
        mean_error=sum(errors) / len(errors),
        p95_abs_error=percentile(abs_errors, 95),
        worst_error=worst,
        sample_count=len(errors),
    )


def robust_metric(predicted: float, p95_error: float) -> float:
    """``predicted * (1 + p95_error)`` (work order §5.8)."""
    return predicted * (1.0 + p95_error)


# ---------------------------------------------------------------------------
# bench summary.txt ingestion
# ---------------------------------------------------------------------------

#: One data row of bench/core/plots.py::write_summary, e.g.
#: ``TTFT P95  102712.6  100859.8  -1.8%`` -> metric, stat, vLLM(real), Sim(sim).
_SUMMARY_ROW = re.compile(
    r"^(?P<metric>TTFT|TPOT|Latency)\s+(?P<stat>Mean|Median|P90|P95|P99)\s+"
    r"(?P<vllm>[-+]?[0-9.]+)\s+(?P<sim>[-+]?[0-9.]+)\s+[-+]?[0-9.]+%\s*$"
)


class SummaryPairs:
    """(sim, real) pairs parsed from a bench validation summary, by metric."""

    def __init__(self) -> None:
        self.ttft: list[tuple[float, float]] = []
        self.tpot: list[tuple[float, float]] = []

    def extend(self, other: SummaryPairs) -> None:
        self.ttft.extend(other.ttft)
        self.tpot.extend(other.tpot)


def parse_validation_summary(text: str) -> SummaryPairs:
    """Parse a ``summary.txt`` into (sim, real) pairs. Pure.

    ``vLLM`` is reality, ``Sim`` is the prediction, so a pair is ``(sim, real)``
    to fit ``real = alpha * sim + beta`` directly. Only TTFT and TPOT rows are
    kept - Latency is not a calibrated metric.

    All five stat rows (Mean, Median, P90/95/99) are kept as regression points.
    Mean and the percentiles are different estimators of one distribution;
    pooling them is a deliberate choice to widen the fit's range with the few
    points a summary offers, accepting a mild bias toward whichever range
    dominates. Switch to percentiles-only here if that bias ever matters.
    """
    pairs = SummaryPairs()
    for line in text.splitlines():
        match = _SUMMARY_ROW.match(line.strip())
        if match is None:
            continue
        sim = float(match.group("sim"))
        real = float(match.group("vllm"))
        if match.group("metric") == "TTFT":
            pairs.ttft.append((sim, real))
        elif match.group("metric") == "TPOT":
            pairs.tpot.append((sim, real))
    return pairs


def _fit_pairs(pairs: list[tuple[float, float]]) -> tuple[LinearFit, ErrorStats]:
    sim = [s for s, _ in pairs]
    real = [r for _, r in pairs]
    return fit_linear(sim, real), compute_error_stats(sim, real)


def fit_hardware(
    pairs: SummaryPairs, *, hardware: str, workload_bucket: str
) -> HardwareCalibration:
    """Build a `HardwareCalibration` from parsed summary pairs."""
    ttft_fit, ttft_err = _fit_pairs(pairs.ttft)
    tpot_fit, tpot_err = _fit_pairs(pairs.tpot)
    return HardwareCalibration(
        hardware=hardware,
        ttft=ttft_fit,
        tpot=tpot_fit,
        errors={
            workload_bucket: BucketError(
                workload_bucket=workload_bucket, ttft=ttft_err, tpot=tpot_err
            )
        },
    )


def fit_from_summaries(
    entries: Sequence[tuple[str | Path, str, str]],
) -> CalibrationModel:
    """Fit a `CalibrationModel` from ``(summary_path, hardware, workload_bucket)``.

    Multiple summaries for one hardware are pooled: their (sim, real) pairs are
    concatenated before fitting. When two entries share a (hardware, bucket)
    their error samples merge into one distribution.

    With no readable data an entry contributes nothing; an empty ``entries``
    yields `CalibrationModel.identity`.
    """
    by_hw_pairs: dict[str, SummaryPairs] = {}
    by_hw_bucket_pairs: dict[tuple[str, str], SummaryPairs] = {}
    sources: dict[str, list[str]] = {}

    for path, hardware, bucket in entries:
        text = Path(path).read_text()
        pairs = parse_validation_summary(text)
        by_hw_pairs.setdefault(hardware, SummaryPairs()).extend(pairs)
        by_hw_bucket_pairs.setdefault((hardware, bucket), SummaryPairs()).extend(pairs)
        sources.setdefault(hardware, []).append(str(path))

    model = CalibrationModel.identity()
    for hardware, pooled in by_hw_pairs.items():
        ttft_fit, _ = _fit_pairs(pooled.ttft)
        tpot_fit, _ = _fit_pairs(pooled.tpot)
        errors: dict[str, BucketError] = {}
        for (hw, bucket), bpairs in by_hw_bucket_pairs.items():
            if hw != hardware:
                continue
            _, ttft_err = _fit_pairs(bpairs.ttft)
            _, tpot_err = _fit_pairs(bpairs.tpot)
            errors[bucket] = BucketError(
                workload_bucket=bucket, ttft=ttft_err, tpot=tpot_err
            )
        model.hardware[hardware] = HardwareCalibration(
            hardware=hardware, ttft=ttft_fit, tpot=tpot_fit, errors=errors
        )
    model.provenance = {"fitted_from": sources}
    return model


# ---------------------------------------------------------------------------
# Applying to a plan (opt-in)
# ---------------------------------------------------------------------------

def apply_robust_margins(
    plan: DeploymentPlan,
    model: CalibrationModel,
    *,
    hardware: str,
    spec: ServiceSpec | None = None,
    bucket: str | None = None,
) -> DeploymentPlan:
    """Return a copy of ``plan`` with robust margins from ``model`` filled in.

    Opt-in by construction: nothing calls this unless a caller chooses to. The
    bucket is taken from ``spec`` (via `envelope.workload_bucket`) or passed
    directly. Missing calibration leaves the margins at 0 - never a guess.
    """
    if bucket is None:
        if spec is None:
            raise ValueError("provide either spec or bucket to select a workload bucket")
        bucket = workload_bucket(spec)
    ttft_pct, tpot_pct = model.margins(hardware, bucket)
    return plan.model_copy(
        update={
            "robust_margin_ttft_percent": ttft_pct,
            "robust_margin_tpot_percent": tpot_pct,
        }
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATION_DIR = Path("profiles/calibration")


def save_calibration(model: CalibrationModel, path: str | Path) -> None:
    """Write a calibration model to YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))


def load_calibration(path: str | Path) -> CalibrationModel:
    """Load a calibration model from YAML. A missing file is the identity model."""
    path = Path(path)
    if not path.is_file():
        return CalibrationModel.identity()
    raw = yaml.safe_load(path.read_text())
    if not raw:
        return CalibrationModel.identity()
    return CalibrationModel.model_validate(raw)
