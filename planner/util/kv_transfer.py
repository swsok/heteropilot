"""KV-cache transfer cost for Prefill/Decode split (work order §5.3, §5.9).

A P/D-split deployment computes a request's prompt KV cache on the prefill
island and ships it to the decode island, which then generates. This module
estimates the wire cost of that transfer - time and energy - from the already
built topology path between the two islands.

Pure arithmetic, no I/O, so it is unit-testable in isolation. It is **not** a
pruning stage: §5.6 declares no P/D constraint, so nothing here may remove a
candidate (the two-invariants rule). It feeds the §5.9 adoption analysis
(`Benefit_of_split > KV_transfer_latency + KV_transfer_energy + queueing`) and
the Level-2 path-aware model.

NOTE: `kv_transfer_cost` is no longer informational only. As of Phase 5 increment
2 (docs/phase5_plan.md) it is the P/D transfer-cost term: `evaluate_candidates`
(via `apply_pd_transfer_cost` in planner/optimizer/exhaustive.py) calls it to add
the prefill->decode transfer penalty to a pd_split candidate's predicted TTFT and
energy, which the simulator leaves free. This is a post-predict metric adjustment
applied identically in oracle and pruned modes - it still removes no candidate,
so oracle-agreement is untouched.
"""

from __future__ import annotations

from planner.inventory import Link
from planner.topology import TopologyGraph
from planner.util import memory as memutil


def _kv_bytes_per_token(model: str, dtype: str, kv_cache_dtype: str) -> int:
    """Whole-model KV bytes for one token, obtained from the simulator's own
    memory model rather than reimplemented (same discipline as memory.py).

    tp is irrelevant to what crosses the wire: the full per-token KV cache is
    transferred regardless of how the prefill engine sharded it, so this sizes
    at tp=1. Device memory is set effectively unbounded because only the KV
    *rate* is wanted, not a fit verdict.
    """
    report = memutil.evaluate(
        model,
        tp_size=1,
        device_memory_gb=1e9,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=1.0,
    )
    return report.kv_bytes_per_token


def kv_transfer_cost(
    model: str,
    dtype: str,
    prompt_tokens: int,
    path: list[Link],
    *,
    kv_cache_dtype: str = "auto",
) -> tuple[float, float]:
    """Estimate (transfer_time_ms, transfer_energy_j) for one request's prompt KV.

    - ``kv_bytes = kv_bytes_per_token * prompt_tokens``
    - ``time_ms = path_latency_ns / 1e6 + kv_bytes / (effective_bw_gbps * 1e9) * 1e3``
    - ``energy_j = sum(link.energy_per_bit_pj) * kv_bytes * 8 * 1e-12``

    An empty path (same endpoint) costs nothing. A link with no
    ``energy_per_bit_pj`` contributes zero energy - the caller that needs the
    accounting to be complete should inspect the links, as silently treating
    unknown as free would understate energy exactly where topology matters.
    """
    if prompt_tokens < 0:
        raise ValueError(f"prompt_tokens must be >= 0, got {prompt_tokens}")

    kv_bytes = _kv_bytes_per_token(model, dtype, kv_cache_dtype) * prompt_tokens

    latency_ms = TopologyGraph.path_latency_ns(path) / 1e6
    bw_gbps = TopologyGraph.effective_bandwidth_gbps(path)
    # bw == inf for an empty path -> the division is 0.0, leaving only latency.
    xfer_ms = latency_ms + (kv_bytes / (bw_gbps * 1e9)) * 1e3

    energy_per_bit_pj = sum(
        link.energy_per_bit_pj for link in path if link.energy_per_bit_pj is not None
    )
    energy_j = energy_per_bit_pj * kv_bytes * 8 * 1e-12

    return xfer_ms, energy_j
