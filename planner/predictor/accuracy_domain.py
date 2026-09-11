"""Served concurrency - the axis an accuracy domain is indexed by (§2.4).

The accuracy-domain CURVE itself is `planner.predictor.calibration.AccuracyDomain`
(one per hardware, `profiles/calibration/*.yaml`), and the per-candidate margin
that reads it is `planner.optimizer.margin.AccuracyDomainMargin`. This module
holds the two ways of getting the x coordinate:

* `served_concurrency_from_sim` reads it off a trace - Little's law as a
  time-average, ``sum(latency) / wall`` - which is what D22 published and what
  `planner/util/operating_point.py` computes per hardware for a simulated run;
* `served_concurrency_little` solves it forward from predicted latencies where
  no trace exists (the surrogate stage).

Both are SERVED concurrency, never the client's requested concurrency: a
request pool too small to keep the server busy makes those two differ by 30 %
or more, which is precisely how D22's retracted top point came about.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Metric names an accuracy domain carries, in the order `errors_at` returns them.
METRICS = ("ttft", "tpot")


def served_concurrency_from_sim(latencies_s: Sequence[float], wall_s: float) -> float:
    """Mean in-flight requests over a run: ``sum(latency) / wall``.

    This is Little's law read off the trace rather than solved: the time-average
    number in the system is the total time-in-system divided by the observation
    window. It is the same arithmetic D22 used, and it is deliberately NOT the
    client's requested concurrency.

    ``latencies_s`` and ``wall_s`` must share a unit; seconds by convention.
    """
    if wall_s <= 0:
        return 0.0
    return float(sum(latencies_s)) / float(wall_s)


def served_concurrency_little(
    rps: float, ttft_ms: float, tpot_ms: float, mean_out_tokens: float
) -> float:
    """Little's law forward: ``L = lambda * W`` with ``W`` built from the SLOs.

    Used where no trace exists - the surrogate stage, which has predicted
    latencies but no per-request records. ``W = TTFT + n_out * TPOT``.

    Note this is the one-shot form, not the fixed point ``L = lambda * W(L)``
    of the rps design §2.1: the caller supplies the latencies it already
    predicted, so there is nothing to iterate here.
    """
    wait_s = (ttft_ms + mean_out_tokens * tpot_ms) / 1000.0
    return float(rps) * wait_s
