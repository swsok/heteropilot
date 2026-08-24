"""Greedy optimizer baseline (work order §12).

A cheap analytical optimizer: it ranks candidates by a memory-roofline
goodput/J *proxy* and picks the argmax WITHOUT ever simulating. It exists purely
as a §12 optimizer baseline — the gap between greedy's pick and the sim-guided
"proposed" pick (and the exhaustive oracle optimum) measures what running the
simulator actually buys.

The proxy mirrors the candidate generator's stage-5 roofline physics
(`candidate_generator._stage5_analytical_ok`) so greedy respects the same physics
as the pruning bounds — the discipline CLAUDE.md demands of the mock predictor.
It never reads a `SimResult`; the caller looks up the pick's true simulated
metrics to report the optimality gap.

Definitions (all proxies, higher is better unless noted):
  step_s        = (weight_bytes + active*kv_bytes_per_token) / (mem_bw * 1e9)
  throughput    = sum over decode/aggregated replicas of active/step_s  (tok/s)
  power         = sum over ALL devices of active_power                  (W)
  goodput/J     = throughput / power                                   (tok/J)
  roofline_tpot = max decode step_s over replicas                      (ms)
A candidate whose roofline TPOT floor already exceeds the SLO is flagged
`likely_infeasible` and sorted below every likely-feasible candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

from planner.inventory import AcceleratorProfile, ExecutionIsland
from planner.plan import CandidateConfig, Role
from planner.spec import ServiceSpec
from planner.util import memory as memutil


@dataclass(frozen=True)
class GreedyEstimate:
    candidate_id: str
    proxy_tokens_per_joule: float
    proxy_throughput_tps: float
    proxy_power_w: float | None       # None when a touched profile has no power
    roofline_tpot_ms: float
    likely_infeasible: bool


def _active_power_w(profile: AcceleratorProfile) -> float | None:
    if profile.power is not None:
        return profile.power.active_power
    return profile.tdp_w  # may be None -> power unknown for this candidate


def estimate(
    candidate: CandidateConfig,
    spec: ServiceSpec,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    *,
    gpu_memory_utilization: float = 0.90,
) -> GreedyEstimate:
    seqs = candidate.knobs.max_num_seqs
    total_tps = 0.0
    total_power = 0.0
    worst_floor_ms = 0.0
    power_known = True
    for a in candidate.assignments:
        island = islands[a.island_id]
        profile = profiles[island.accelerator_model]
        per_device_gb = island.total_memory_gb / island.size
        _, report = memutil.feasible(
            spec.model,
            tp_size=a.tp_size,
            device_memory_gb=per_device_gb,
            dtype=spec.service.dtype,
            kv_cache_dtype=spec.service.kv_cache_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        is_prefill = a.role is Role.PREFILL
        median_len = (
            spec.traffic.input_tokens.p50 if is_prefill
            else spec.traffic.input_tokens.p50 + spec.traffic.output_tokens.p50
        )
        kv_capacity = report.kv_tokens // max(1, median_len)
        active = max(1, min(seqs, max(1, kv_capacity)))
        bytes_per_step = report.weight_bytes + active * report.kv_bytes_per_token
        step_s = bytes_per_step / (profile.memory_bandwidth_gbps * 1e9)

        p = _active_power_w(profile)
        if p is None:
            power_known = False
        else:
            total_power += p * a.total_devices

        # Only decode/aggregated engines emit tokens; a disaggregated prefill
        # engine runs no decode steps (mirrors the stage-5 role split).
        if not is_prefill:
            worst_floor_ms = max(worst_floor_ms, step_s * 1e3)
            total_tps += (active / step_s) * a.dp_replicas

    proxy_power: float | None = total_power if power_known else None
    proxy_tpj = (
        total_tps / proxy_power if proxy_power and proxy_power > 0 else 0.0
    )
    likely_infeasible = worst_floor_ms > spec.slo.tpot.max_ms
    return GreedyEstimate(
        candidate_id=candidate.id,
        proxy_tokens_per_joule=proxy_tpj,
        proxy_throughput_tps=total_tps,
        proxy_power_w=proxy_power,
        roofline_tpot_ms=worst_floor_ms,
        likely_infeasible=likely_infeasible,
    )


def rank(
    candidates: list[CandidateConfig],
    spec: ServiceSpec,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    *,
    gpu_memory_utilization: float = 0.90,
) -> list[GreedyEstimate]:
    """Candidates ordered by the greedy proxy, best first. Deterministic:
    likely-feasible before likely-infeasible, then higher proxy goodput/J, then
    candidate id (matches pareto's id tie-break for §9 reproducibility)."""
    ests = [
        estimate(c, spec, islands, profiles, gpu_memory_utilization=gpu_memory_utilization)
        for c in candidates
    ]
    ests.sort(key=lambda e: (e.likely_infeasible, -e.proxy_tokens_per_joule, e.candidate_id))
    return ests


def greedy(
    candidates: list[CandidateConfig],
    spec: ServiceSpec,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    *,
    gpu_memory_utilization: float = 0.90,
) -> CandidateConfig | None:
    """The single candidate a greedy planner would pick (and then simulate).
    None if there are no candidates."""
    ranked = rank(candidates, spec, islands, profiles,
                  gpu_memory_utilization=gpu_memory_utilization)
    if not ranked:
        return None
    top_id = ranked[0].candidate_id
    return next(c for c in candidates if c.id == top_id)
