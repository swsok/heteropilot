"""Concurrent candidate simulation (work order §5.5 note on --run-id isolation).

Each candidate is simulated by an independent `python -m serving` subprocess that
LLMServingSimPredictor stages under a per-candidate run directory and a unique
`--run-id` (so ASTRA-Sim inputs are isolated - CLAUDE.md: "parallel simulations
need no extra locking"). A `predict()` call therefore spends almost all its time
in `subprocess.run`, which releases the GIL, so a thread pool runs many candidate
sims at once and turns a machine's idle cores into near-linear speedup.

Determinism is preserved: results are keyed by `candidate.id`, so completion order
never affects the returned mapping, and the caller iterates candidates in its own
(generation) order downstream - identical output to the sequential loop, faster.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
from planner.plan import CandidateConfig
from planner.predictor import Predictor, SimResult
from planner.spec import ServiceSpec


def default_workers(n_candidates: int) -> int:
    """A safe default worker count. Each sim subprocess uses ~1-2 cores, so half
    the CPUs keeps them busy without oversubscribing; capped at 32 and at the
    candidate count."""
    cpus = os.cpu_count() or 4
    return max(1, min(n_candidates, cpus // 2, 32))


def predict_all(
    predictor: Predictor,
    candidates: list[CandidateConfig],
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    *,
    max_workers: int | None = None,
    progress: Callable[[int, int, CandidateConfig], None] | None = None,
) -> dict[str, SimResult]:
    """Simulate every candidate concurrently; return {candidate.id: SimResult}.

    `max_workers=1` runs sequentially (useful for debugging). Any exception from a
    `predict` propagates after the pool drains, exactly as a sequential loop would
    raise. The predictor must isolate per-candidate state (LLMServingSimPredictor
    does, via per-id run dirs and unique --run-id)."""
    if not candidates:
        return {}
    # Concurrent isolation (per-id run dir and --run-id) and the returned mapping
    # both key on candidate.id, so duplicate ids would race and silently drop a
    # result. The generator produces unique ids; assert it before dispatching.
    ids = [c.id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("predict_all requires unique candidate ids")
    workers = max_workers if max_workers is not None else default_workers(len(candidates))
    workers = max(1, min(workers, len(candidates)))

    if workers == 1:
        results: dict[str, SimResult] = {}
        for i, cand in enumerate(candidates):
            results[cand.id] = predictor.predict(cand, spec, cluster, islands, profiles)
            if progress is not None:
                progress(i, len(candidates), cand)
        return results

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(predictor.predict, cand, spec, cluster, islands, profiles): cand
            for cand in candidates
        }
        for i, fut in enumerate(as_completed(futures)):
            cand = futures[fut]
            results[cand.id] = fut.result()  # re-raises any predict() exception
            if progress is not None:
                progress(i, len(candidates), cand)
    return results
