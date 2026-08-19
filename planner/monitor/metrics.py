"""vLLM Prometheus scraping and GPU power sampling (work order §5.7).

Two independent, pure-where-possible pieces:

1. ``parse_vllm_metrics`` turns the raw text of vLLM's ``/metrics`` endpoint
   into TTFT/TPOT percentiles and token/request counters. The parser is a pure
   function of the text so it is unit-testable against a fixture string with no
   running server. Percentiles come from Prometheus histogram buckets via the
   standard ``histogram_quantile`` linear interpolation.

2. ``PowerSampler`` polls ``nvidia-smi --query-gpu=power.draw`` for a set of
   device indices; ``integrate_energy`` turns a series of (timestamp, watts)
   samples into joules with the trapezoidal rule. The integration is a pure
   function so it is testable without a GPU. Default sampling period is 1.0 s -
   coarse enough not to perturb the workload, fine enough to integrate energy to
   within a percent or two over a multi-second run.

The percentiles here intentionally use the same numpy-``linear`` method as
``planner/util/percentile.py`` for the power path; histogram percentiles are a
different estimator (they operate on pre-bucketed data, not raw samples) and are
implemented in ``histogram_quantile`` accordingly.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Prometheus text parsing
# ---------------------------------------------------------------------------

#: One Prometheus sample line: name, optional {labels}, value.
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:[iI]nf|[nN]a[nN]|[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?))"
    r"\s*(?:[0-9]+)?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> list[Sample]:
    """Parse Prometheus exposition text into samples, ignoring comments.

    Tolerant by design: a line that does not match is skipped rather than
    raising, because a metrics endpoint can carry vendor extensions we do not
    model, and a scrape must never fail on an unknown line.
    """
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        labels = dict(_LABEL_RE.findall(match.group("labels") or ""))
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        samples.append(Sample(match.group("name"), labels, value))
    return samples


def histogram_quantile(buckets: Sequence[tuple[float, float]], q: float) -> float:
    """Prometheus-style quantile over cumulative histogram buckets.

    ``buckets`` is a list of ``(le, cumulative_count)`` pairs; ``q`` is a
    fraction in [0, 1]. Uses linear interpolation within the target bucket, the
    same estimator Prometheus' own ``histogram_quantile`` uses. Returns ``nan``
    when the histogram is empty.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    ordered = sorted(buckets, key=lambda b: b[0])
    if not ordered:
        return float("nan")
    total = ordered[-1][1]
    if total <= 0:
        return float("nan")

    rank = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in ordered:
        if count >= rank:
            if le == float("inf"):
                # Everything is in the overflow bucket; the best estimate is the
                # largest finite boundary we saw.
                return prev_le
            span = le - prev_le
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return le
            return prev_le + span * (rank - prev_count) / bucket_count
        prev_le, prev_count = le, count
    return ordered[-1][0]


@dataclass(frozen=True)
class VllmScrape:
    """One scrape of a vLLM ``/metrics`` endpoint, reduced to headline metrics.

    Percentiles are milliseconds; token/request fields are cumulative counters
    (monotonic since server start), so a rate must be derived over a window by
    the caller, not read off a single scrape.
    """

    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float
    completed_requests: int
    completed_tokens: int
    prompt_tokens: int


#: vLLM histogram metric base names (v0.6+). ``_bucket``/``_sum``/``_count``
#: are the three series each histogram exposes.
_TTFT_METRIC = "vllm:time_to_first_token_seconds"
_TPOT_METRIC = "vllm:time_per_output_token_seconds"
_GEN_TOKENS = "vllm:generation_tokens_total"
_PROMPT_TOKENS = "vllm:prompt_tokens_total"
_REQ_SUCCESS = "vllm:request_success_total"


def _buckets_for(samples: Sequence[Sample], metric: str) -> list[tuple[float, float]]:
    # Sum counts by `le` so a histogram carrying an extra label dimension
    # (multiple series) collapses into one cumulative-count curve instead of
    # feeding duplicate boundaries to histogram_quantile.
    by_edge: dict[float, float] = {}
    for s in samples:
        if s.name != f"{metric}_bucket":
            continue
        le = s.labels.get("le")
        if le is None:
            continue
        edge = float("inf") if le in ("+Inf", "Inf", "inf") else float(le)
        by_edge[edge] = by_edge.get(edge, 0.0) + s.value
    return sorted(by_edge.items())


def _counter_total(samples: Sequence[Sample], metric: str) -> float:
    return sum(s.value for s in samples if s.name == metric)


def _percentiles_ms(buckets: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    return (
        histogram_quantile(buckets, 0.50) * 1000.0,
        histogram_quantile(buckets, 0.95) * 1000.0,
        histogram_quantile(buckets, 0.99) * 1000.0,
    )


def parse_vllm_metrics(text: str) -> VllmScrape:
    """Reduce raw ``/metrics`` text to a `VllmScrape`. Pure function."""
    samples = parse_prometheus(text)
    p50_ttft, p95_ttft, p99_ttft = _percentiles_ms(_buckets_for(samples, _TTFT_METRIC))
    p50_tpot, p95_tpot, p99_tpot = _percentiles_ms(_buckets_for(samples, _TPOT_METRIC))
    return VllmScrape(
        p50_ttft_ms=p50_ttft,
        p95_ttft_ms=p95_ttft,
        p99_ttft_ms=p99_ttft,
        p50_tpot_ms=p50_tpot,
        p95_tpot_ms=p95_tpot,
        p99_tpot_ms=p99_tpot,
        completed_requests=int(_counter_total(samples, _REQ_SUCCESS)),
        completed_tokens=int(_counter_total(samples, _GEN_TOKENS)),
        prompt_tokens=int(_counter_total(samples, _PROMPT_TOKENS)),
    )


# ---------------------------------------------------------------------------
# Power sampling
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_INTERVAL_S = 1.0


@dataclass(frozen=True)
class PowerSample:
    timestamp: float
    watts: float


def integrate_energy(timestamps: Sequence[float], watts: Sequence[float]) -> float:
    """Trapezoidal integral of power (W) over time (s), in joules. Pure.

    Fewer than two samples integrate to 0.0: energy needs an interval. Samples
    must be time-ordered; out-of-order timestamps would give a nonsensical
    (possibly negative) area, which is the caller's bug to avoid.
    """
    if len(timestamps) != len(watts):
        raise ValueError("timestamps and watts must be the same length")
    if len(timestamps) < 2:
        return 0.0
    energy = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        energy += dt * (watts[i - 1] + watts[i]) / 2.0
    return energy


@dataclass
class PowerSeries:
    """A collected set of power samples for one device group."""

    samples: list[PowerSample] = field(default_factory=list)

    def add(self, watts: float, *, at: float | None = None) -> None:
        self.samples.append(PowerSample(time.time() if at is None else at, watts))

    @property
    def average_power_w(self) -> float | None:
        if not self.samples:
            return None
        return sum(s.watts for s in self.samples) / len(self.samples)

    @property
    def peak_power_w(self) -> float | None:
        if not self.samples:
            return None
        return max(s.watts for s in self.samples)

    @property
    def total_energy_j(self) -> float:
        return integrate_energy(
            [s.timestamp for s in self.samples],
            [s.watts for s in self.samples],
        )

    @property
    def window_seconds(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp


class PowerSampler:
    """Polls ``nvidia-smi`` for the summed power draw of a device group.

    One sampler owns a set of device indices (the ints a backend put in
    ``CUDA_VISIBLE_DEVICES``). ``sample_watts`` reads all of them once and
    returns their sum; ``collect`` loops for a duration. Neither is pure - they
    shell out - so tests exercise `integrate_energy` and `PowerSeries` instead.
    """

    def __init__(
        self,
        device_indices: Sequence[int],
        *,
        interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        smi_path: str = "nvidia-smi",
    ) -> None:
        self.device_indices = list(device_indices)
        self.interval_s = interval_s
        self.smi_path = smi_path

    @property
    def available(self) -> bool:
        return shutil.which(self.smi_path) is not None

    def sample_watts(self) -> float:
        """Summed instantaneous power draw of the group, in watts.

        Raises `DeploymentError`-free: on any failure it raises RuntimeError so
        the caller can treat power as unavailable rather than crash a status
        read.
        """
        query = "--query-gpu=power.draw"
        ids = ",".join(str(i) for i in self.device_indices)
        cmd = [self.smi_path, query, "--format=csv,noheader,nounits", f"--id={ids}"]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"nvidia-smi power query failed: {exc}") from exc
        total = 0.0
        for line in out.stdout.splitlines():
            line = line.strip()
            if line:
                total += float(line)
        return total

    def collect(self, duration_s: float) -> PowerSeries:
        """Sample every ``interval_s`` for ``duration_s`` and return the series."""
        series = PowerSeries()
        deadline = time.time() + duration_s
        while time.time() < deadline:
            series.add(self.sample_watts())
            time.sleep(self.interval_s)
        return series
