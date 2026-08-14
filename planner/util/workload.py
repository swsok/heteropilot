"""Workload trace generation (work order §5.5).

Turns a `ServiceSpec.traffic` distribution into the JSONL the simulator and
`bench` both consume. This is where reproducibility lives: `python -m serving`
has no `--seed` flag and does not need one - it is a deterministic discrete-event
simulation - so §9's "same spec + seed => byte-identical output" is a property of
*this* generator (docs/deviations.md D5).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from planner.spec import ServiceSpec, TokenDistribution

#: Standard-normal quantile at p95, used to fit a lognormal to (p50, p95).
Z95 = 1.6448536269514722


@dataclass(frozen=True)
class WorkloadTrace:
    path: Path
    num_requests: int
    seed: int
    total_input_tokens: int
    total_output_tokens: int
    horizon_s: float

    def as_provenance(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "num_requests": self.num_requests,
            "random_seed": self.seed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "horizon_s": round(self.horizon_s, 3),
        }


def _lognormal_sigma(dist: TokenDistribution) -> float:
    """Fit sigma from the p50/p95 pair; 0 when p95 is absent or degenerate."""
    if dist.p95 is None or dist.p95 <= dist.p50:
        return 0.0
    return math.log(dist.p95 / dist.p50) / Z95


def _sample_lengths(rng: np.random.Generator, dist: TokenDistribution, n: int) -> np.ndarray:
    sigma = _lognormal_sigma(dist)
    if sigma == 0.0:
        return np.full(n, dist.p50, dtype=np.int64)
    samples = rng.lognormal(mean=math.log(dist.p50), sigma=sigma, size=n)
    if dist.p99 is not None:
        # Respect the declared tail instead of letting the fit run away.
        samples = np.minimum(samples, dist.p99)
    return np.maximum(1, np.rint(samples)).astype(np.int64)


def _sample_arrivals(
    rng: np.random.Generator, rate_rps: float, burstiness: float, n: int
) -> np.ndarray:
    """Inter-arrival times in seconds.

    burstiness == 1 gives a Poisson process (exponential gaps). Larger values
    use a Gamma with shape 1/burstiness, which keeps the mean rate but makes the
    gaps more dispersed - short bursts separated by longer idles.
    """
    shape = 1.0 / max(burstiness, 1e-9)
    scale = 1.0 / (rate_rps * shape)
    gaps = rng.gamma(shape=shape, scale=scale, size=n)
    return np.cumsum(gaps)


def generate_trace(
    spec: ServiceSpec,
    out_path: str | Path,
    *,
    num_requests: int,
    seed: int,
) -> WorkloadTrace:
    """Write a LLMServingSim-format JSONL trace and describe it.

    Token ids are synthetic. They only need to be self-consistent: the simulator
    hashes them for prefix matching and never interprets them as text.
    """
    if num_requests < 1:
        raise ValueError(f"num_requests must be >= 1, got {num_requests}")

    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = _sample_lengths(rng, spec.traffic.input_tokens, num_requests)
    outputs = _sample_lengths(rng, spec.traffic.output_tokens, num_requests)
    arrivals_s = _sample_arrivals(
        rng, spec.traffic.arrival_rate_rps, spec.traffic.burstiness, num_requests
    )

    # Shared-prefix pool. With prefix caching disabled (the Phase 2 default) this
    # changes nothing in the simulator, but generating it keeps the trace honest
    # to the spec and correct for later phases that turn caching back on.
    share = spec.traffic.prefix_share_ratio
    prefix_len = int(spec.traffic.input_tokens.p50 * share) if share > 0 else 0
    shared_prefix = [int(t) for t in rng.integers(1, 32000, size=prefix_len)] if prefix_len else []

    total_in = total_out = 0
    with out_path.open("w") as fh:
        for i in range(num_requests):
            n_in = int(inputs[i])
            n_out = int(outputs[i])
            if prefix_len and n_in > prefix_len:
                tail = rng.integers(1, 32000, size=n_in - prefix_len)
                input_ids = shared_prefix + [int(t) for t in tail]
            else:
                input_ids = [int(t) for t in rng.integers(1, 32000, size=n_in)]
            output_ids = [int(t) for t in rng.integers(1, 32000, size=n_out)]

            fh.write(
                json.dumps(
                    {
                        "input_toks": n_in,
                        "output_toks": n_out,
                        "arrival_time_ns": int(arrivals_s[i] * 1e9),
                        "input_tok_ids": input_ids,
                        "output_tok_ids": output_ids,
                    }
                )
                + "\n"
            )
            total_in += n_in
            total_out += n_out

    return WorkloadTrace(
        path=out_path,
        num_requests=num_requests,
        seed=seed,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        horizon_s=float(arrivals_s[-1]),
    )
