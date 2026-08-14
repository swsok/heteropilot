"""The single percentile implementation for the whole planner (work order §4).

Every latency percentile in HeteroPilot goes through here. Different
interpolation methods disagree by several percent on small samples, and a
TTFT P99 that shifts depending on which module computed it makes SLO
feasibility non-reproducible. numpy's `linear` method is fixed by the work
order; do not add a `method` parameter.

Note this deliberately does not match the simulator's own printed
"P99 TTFT" line, which uses its own method. Parse the per-request CSV and
compute percentiles here instead (docs/phase0_formats.md §3.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np

INTERPOLATION: Final[Literal["linear"]] = "linear"


def percentile(values: Sequence[float], p: float) -> float:
    """The p-th percentile (0-100) with numpy `linear` interpolation."""
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    if len(values) == 0:
        raise ValueError("cannot take a percentile of an empty sequence")
    return float(np.percentile(np.asarray(values, dtype=float), p, method=INTERPOLATION))


def percentiles(values: Sequence[float], ps: Sequence[float]) -> dict[float, float]:
    return {p: percentile(values, p) for p in ps}


def summary(values: Sequence[float]) -> dict[str, float]:
    """The P50/P95/P99 triple the SLO checks are defined on."""
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }
