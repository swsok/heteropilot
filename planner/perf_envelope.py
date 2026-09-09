"""Performance envelopes: a measured curve, and the refusal to read past its ends.

`docs/rps_aware_planning_design.md` §2 and §4. **Not** `planner/envelope.py`, which
is the simulation-result cache; the names are close and the design says so
explicitly.

A profile's scalar `throughput` is a claim about one operating point pretending to
be a claim about the device. D22 is what that costs: an exponent fitted across a
pool-capped point read a x1.74 interval as a doubling, and the headline built on it
was retracted. So an envelope here is a list of measured points plus the two things
that make it safe to use -- `concurrency_metric: served`, and a `validity` block
whose default is to refuse rather than extrapolate. A file missing either is not
loaded.

The solver answers "what operating point does this arrival rate put the instance
at?" and may answer "it saturates", which is a RESULT the caller acts on by adding
replicas. It distinguishes two saturations that must never be collapsed:

  * `genuine`   -- the measured curve has flattened; more concurrency buys nothing;
  * `unmeasured` -- the demand is past the last measured point and the curve was
                    still climbing. We do not know. Under `extrapolation: refuse`
                    this becomes a rejection reason, never a silent guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EnvelopeError(ValueError):
    """The envelope file is unusable. Never downgraded to a warning: a planner
    running on a mis-parsed envelope is worse than one running on none."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvelopePoint(_Strict):
    """One measured operating point. `conc` is SERVED concurrency, always."""

    conc: float = Field(gt=0)
    tput_tok_s: float = Field(gt=0)
    tpot_p50: float | None = Field(default=None, gt=0)
    tpot_p99: float | None = Field(default=None, gt=0)
    ttft_p99: float | None = Field(default=None, gt=0)
    util_pct: float | None = Field(default=None, ge=0, le=100)
    power_w: float | None = Field(default=None, ge=0)
    tok_per_j: float | None = Field(default=None, ge=0)
    rps: float | None = Field(default=None, ge=0)


class EnvelopeValidity(_Strict):
    """Where the curve may be read. `refuse` is the default and the only policy
    implemented: silently widening a range is the D22 failure mode."""

    conc_min: float = Field(gt=0)
    conc_max: float = Field(gt=0)
    extrapolation: Literal["refuse", "widen_error_bars"] = "refuse"
    pool_binding_above: float | None = Field(default=None, gt=0)
    power_valid_range: list[float] | None = None

    @model_validator(mode="after")
    def _ordered(self) -> EnvelopeValidity:
        if self.conc_max < self.conc_min:
            raise ValueError(f"conc_max {self.conc_max} < conc_min {self.conc_min}")
        return self


class MeasuredWorkload(_Strict):
    """The curve is conditional on the token distribution (design §10).

    `output_tokens_mean` is the one the solver needs and the p50 is not: Little's
    law relates throughput to the MEAN tokens per request, and on a skewed
    distribution the two differ enough to matter. On the RNGD workload the p50 is
    632 and the mean 652.5 -- a 3.2 % gap that moves the solved concurrency by
    about the same, which is larger than the round-trip tolerance.
    """

    dataset: str
    input_tokens_p50: float = Field(gt=0)
    output_tokens_p50: float = Field(gt=0)
    output_tokens_mean: float | None = Field(default=None, gt=0)


class PerfEnvelope(_Strict):
    envelope_id: str
    unit: str
    source: str
    concurrency_metric: str
    measured_on_workload: MeasuredWorkload
    points: list[EnvelopePoint] = Field(min_length=2)
    validity: EnvelopeValidity
    closed_loop: bool = True
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    idle: dict[str, Any] | None = None
    notes: str = ""

    @field_validator("concurrency_metric")
    @classmethod
    def _must_be_served(cls, v: str) -> str:
        # The requested/served conflation IS D22. A file that does not say which
        # it recorded cannot be trusted to have recorded served.
        if v != "served":
            raise ValueError(
                f"concurrency_metric must be 'served', got {v!r}; requested "
                f"concurrency is what D22 retracted"
            )
        return v

    @model_validator(mode="after")
    def _sorted_and_monotone(self) -> PerfEnvelope:
        concs = [p.conc for p in self.points]
        if concs != sorted(concs):
            raise ValueError("points must be sorted by ascending conc")
        if len(set(concs)) != len(concs):
            raise ValueError(f"duplicate concurrency values: {concs}")
        tputs = [p.tput_tok_s for p in self.points]
        if tputs != sorted(tputs):
            # A non-monotone throughput curve cannot be inverted, and every
            # measured curve so far is monotone. Refuse rather than pick a branch.
            raise ValueError(
                f"throughput must be non-decreasing in served concurrency to be "
                f"invertible; got {tputs}"
            )
        return self

    # -- reading the curve --------------------------------------------------

    def _bracket(self, conc: float) -> tuple[EnvelopePoint, EnvelopePoint]:
        for lo, hi in zip(self.points, self.points[1:], strict=False):
            if lo.conc <= conc <= hi.conc:
                return lo, hi
        raise EnvelopeError(f"{conc} is not bracketed by any measured interval")

    def in_domain(self, conc: float) -> bool:
        return self.validity.conc_min <= conc <= self.validity.conc_max

    def throughput_at(self, conc: float) -> float:
        """Log-linear in both axes: the curve is a power law piecewise, which is
        how its scaling exponent is quoted (D22's 0.675 -> 0.241)."""
        self._require_in_domain(conc)
        lo, hi = self._bracket(conc)
        return _loglog(conc, lo.conc, lo.tput_tok_s, hi.conc, hi.tput_tok_s)

    def metric_at(self, conc: float, field: str) -> float | None:
        """Any per-point field, log-linear, or None if the field is unmeasured
        at either end of the bracketing interval -- never interpolated across a
        gap in the data."""
        self._require_in_domain(conc)
        lo, hi = self._bracket(conc)
        a, b = getattr(lo, field), getattr(hi, field)
        if a is None or b is None:
            return None
        return _loglog(conc, lo.conc, a, hi.conc, b)

    def _require_in_domain(self, conc: float) -> None:
        if not self.in_domain(conc):
            raise EnvelopeError(
                f"{self.envelope_id}: served concurrency {conc:.3f} is outside the "
                f"measured range [{self.validity.conc_min}, {self.validity.conc_max}] "
                f"and validity.extrapolation is {self.validity.extrapolation!r}"
            )

    def workload_mismatch_pct(self, out_tokens: float) -> float | None:
        """How far the caller's mean output length is from the measured one.

        None when the envelope did not record a mean. The caller decides what to
        do with it; this module will not rescale the curve.
        """
        measured = self.measured_on_workload.output_tokens_mean
        if measured is None:
            return None
        return (out_tokens - measured) / measured * 100.0

    @property
    def top_exponent(self) -> float:
        """d log(throughput) / d log(conc) over the final measured interval.

        How hard the curve is still climbing where the measurements stop. It is
        what separates "we stopped measuring" from "the device stopped scaling".
        """
        lo, hi = self.points[-2], self.points[-1]
        return (math.log(hi.tput_tok_s) - math.log(lo.tput_tok_s)) / (
            math.log(hi.conc) - math.log(lo.conc)
        )


def _loglog(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------

#: Below this scaling exponent over the final measured interval, the curve is
#: treated as flat: doubling concurrency would buy under 3.5 % more throughput.
#: Named and exposed rather than buried, because it is the one judgement in this
#: module -- D22's top interval reads 0.241, which is NOT flat, so an RNGD
#: instance asked for more than it was measured at returns `unmeasured`.
SATURATION_EXPONENT = 0.05

#: Relative tolerance for the endpoint comparisons. Without it, asking for exactly
#: the throughput of the top measured point can round to "past the end" and return
#: Saturated for a point that IS measured -- which the round-trip test caught.
_ENDPOINT_REL_TOL = 1e-9


@dataclass(frozen=True)
class OperatingPoint:
    """Where an arrival rate puts one instance, on the measured curve."""

    concurrency: float
    throughput_tok_s: float
    rps_per_instance: float
    tpot_p50_ms: float | None = None
    tpot_p99_ms: float | None = None
    power_w: float | None = None
    util_pct: float | None = None

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Saturated:
    """This instance cannot serve this arrival rate on the measured curve.

    `kind` is load-bearing. `genuine` means adding concurrency would not help and
    the caller should add replicas. `unmeasured` means we ran out of measurements,
    which under `extrapolation: refuse` is a rejection reason and never a guess --
    the distinction D22 collapsed.
    """

    kind: Literal["genuine", "unmeasured"]
    rps_per_instance: float
    max_measured_tput_tok_s: float
    max_measured_conc: float
    top_exponent: float
    reason: str

    @property
    def ok(self) -> bool:
        return False


def solve_operating_point(
    env: PerfEnvelope,
    rps_per_instance: float,
    out_tokens: float,
) -> OperatingPoint | Saturated:
    """Little's law on the measured curve.

    The design writes the fixed point as `L = lambda * W(L)` with
    `W(L) = TTFT(L) + out_tokens * TPOT(L)`. This inverts the THROUGHPUT curve
    instead -- solve `T(L) = lambda * out_tokens` -- which is the same fixed point
    expressed in the quantities the envelope measures most reliably. The reason to
    prefer it is concrete: the envelope's `ttft_*` is closed-loop (the whole pool
    is fired at once, D19), so feeding it into an open-loop residence time would
    import a number the envelope's own notes forbid comparing. Served concurrency
    and throughput carry no such caveat, and they were measured together.

    `out_tokens` must be the MEAN output tokens per request, not the median:
    Little's law is a statement about means, and on a skewed distribution the two
    differ enough to move the answer. Use `env.workload_mismatch_pct` to see how
    far the caller's workload is from the one the curve was measured on; the
    mismatch is reported rather than corrected, because rescaling a curve to a
    different token distribution is exactly the kind of unmeasured inference this
    module exists to refuse (design §10).
    """
    if rps_per_instance <= 0:
        raise EnvelopeError(f"rps_per_instance must be positive, got {rps_per_instance}")
    if out_tokens <= 0:
        raise EnvelopeError(f"out_tokens must be positive, got {out_tokens}")

    demand_tok_s = rps_per_instance * out_tokens
    top = env.points[-1]
    lo_pt = env.points[0]

    if demand_tok_s > top.tput_tok_s * (1.0 + _ENDPOINT_REL_TOL):
        exponent = env.top_exponent
        if exponent < SATURATION_EXPONENT:
            kind: Literal["genuine", "unmeasured"] = "genuine"
            reason = (
                f"{env.envelope_id}: {rps_per_instance:.4g} rps needs "
                f"{demand_tok_s:.1f} tok/s but the curve tops out at "
                f"{top.tput_tok_s:.1f} and has flattened there (exponent "
                f"{exponent:.3f} < {SATURATION_EXPONENT}); more concurrency would "
                f"not help -- add replicas"
            )
        else:
            kind = "unmeasured"
            reason = (
                f"{env.envelope_id}: {rps_per_instance:.4g} rps needs "
                f"{demand_tok_s:.1f} tok/s, past the last measured point "
                f"({top.tput_tok_s:.1f} tok/s at conc {top.conc}), and the curve was "
                f"still climbing there (exponent {exponent:.3f}). Whether the device "
                f"serves this is UNMEASURED, not impossible"
            )
        return Saturated(
            kind=kind, rps_per_instance=rps_per_instance,
            max_measured_tput_tok_s=top.tput_tok_s, max_measured_conc=top.conc,
            top_exponent=exponent, reason=reason,
        )

    if demand_tok_s < lo_pt.tput_tok_s * (1.0 - _ENDPOINT_REL_TOL):
        # Under-loaded past the bottom of the curve. Also outside the domain, and
        # for the same reason: nobody measured it.
        return Saturated(
            kind="unmeasured", rps_per_instance=rps_per_instance,
            max_measured_tput_tok_s=top.tput_tok_s, max_measured_conc=top.conc,
            top_exponent=env.top_exponent,
            reason=(
                f"{env.envelope_id}: {rps_per_instance:.4g} rps needs only "
                f"{demand_tok_s:.1f} tok/s, below the lowest measured point "
                f"({lo_pt.tput_tok_s:.1f} tok/s at conc {lo_pt.conc}). The operating "
                f"point is below the measured range -- UNMEASURED, not saturated"
            ),
        )

    # Clamp to the measured span so an endpoint request lands inside a bracket.
    demand_tok_s = min(max(demand_tok_s, lo_pt.tput_tok_s), top.tput_tok_s)
    conc = _invert_throughput(env, demand_tok_s)
    return OperatingPoint(
        concurrency=conc,
        throughput_tok_s=demand_tok_s,
        rps_per_instance=rps_per_instance,
        tpot_p50_ms=env.metric_at(conc, "tpot_p50"),
        tpot_p99_ms=env.metric_at(conc, "tpot_p99"),
        power_w=env.metric_at(conc, "power_w"),
        util_pct=env.metric_at(conc, "util_pct"),
    )


def _invert_throughput(env: PerfEnvelope, tput: float) -> float:
    """The concurrency at which the curve delivers `tput`. Monotonicity is
    enforced at load time, so the bracketing interval is unique."""
    for lo, hi in zip(env.points, env.points[1:], strict=False):
        if lo.tput_tok_s <= tput <= hi.tput_tok_s:
            return _loglog(tput, lo.tput_tok_s, lo.conc, hi.tput_tok_s, hi.conc)
    raise EnvelopeError(f"{tput} tok/s is not bracketed by any measured interval")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_envelope(path: str | Path) -> PerfEnvelope:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise EnvelopeError(f"{path}: no such envelope") from exc
    if not isinstance(raw, dict):
        raise EnvelopeError(f"{path}: expected a mapping at the top level")
    for required in ("concurrency_metric", "validity"):
        if required not in raw:
            raise EnvelopeError(
                f"{path}: missing '{required}'. An envelope without it cannot be "
                f"used safely (design §2), so it is refused rather than defaulted"
            )
    try:
        return PerfEnvelope.model_validate(raw)
    except Exception as exc:
        raise EnvelopeError(f"{path}: {exc}") from exc


ENVELOPE_ROOT = Path(__file__).resolve().parents[1] / "profiles/envelopes"


#: The design's path uses the short dtype spelling (`bf16`); a ServiceSpec uses
#: the long one (`bfloat16`). Mapping them here rather than renaming either keeps
#: the envelope path readable and the spec faithful to what vLLM accepts.
_DTYPE_DIR = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float32": "fp32",
}


def dtype_dir(dtype: str) -> str:
    return _DTYPE_DIR.get(dtype, dtype)


def find_envelope(
    hardware: str, model: str, dtype: str, tp: int, root: Path | None = None
) -> PerfEnvelope | None:
    """`profiles/envelopes/<HW>/<org>/<model>/<variant>/tp<N>.yaml`, or None.

    None means "no envelope for this hardware", which callers treat as "the
    envelope stage does not apply" -- never as "the envelope permits anything".
    """
    base = (root or ENVELOPE_ROOT) / hardware / model / dtype_dir(dtype) / f"tp{tp}.yaml"
    return load_envelope(base) if base.exists() else None
