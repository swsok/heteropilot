"""M4 TieredPredictor - P1 increment: the analytic surrogate tier (DESIGN §7).

P1 ships Tier-2 only: a deterministic, physics-based analytic predictor that
lets a batch run in milliseconds per candidate with NO simulator. Tier-1
(envelope cache reuse) and Tier-3 (sampled full-sim verification) land in P2.

HONESTY CONTRACT (FR-T3): everything this predictor emits is fidelity =
"surrogate". It is NOT a sound bound and NOT a simulation; the numbers exist
to rank candidates and to be error-measured against full sim by the P2
verification pass. The fidelity label must follow the numbers into the DB and
the UI, and results must never be reported as simulated.

Physics: the same memory-roofline arithmetic as the candidate generator's
stage-5 bounds and the oracle-agreement mock (tests/conftest.py) - step time
from weight+KV bytes over profile memory bandwidth, padded by a fixed slack,
with an M/M/1-style queueing blow-up near saturation. Power derives from the
profile's measured/declared idle+active figures, utilization-blended; host
(node) power is NOT included and the label says so.
"""

from __future__ import annotations

from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
from planner.plan import CandidateConfig, PredictedMetrics, Role
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.spec import ServiceSpec
from planner.util import memory as memutil
from planner.util.workload import WorkloadTrace

#: Fixed pessimism over the roofline, mirroring tests/conftest.py
#: MOCK_ROOFLINE_SLACK. Must stay >= 1.0: a predictor faster than the
#: stage-4/5 bounds would contradict the pruning relaxation invariant.
ROOFLINE_SLACK = 1.2

#: TTFT shape constants, copied from the oracle-agreement mock: prefill work is
#: charged as a fixed multiple of the step roofline plus a per-sequence setup
#: cost, then scaled by the queueing factor. Uncalibrated by construction.
TTFT_PREFILL_STEPS = 8.0
TTFT_PER_SEQ_MS = 0.5

FIDELITY_SURROGATE = "surrogate"
FIDELITY_SIM = "sim"
FIDELITY_ENVELOPE = "envelope"


class SurrogatePredictor(Predictor):
    """Tier-2 analytic predictor. Deterministic; never touches the simulator."""

    def __init__(
        self,
        trace: WorkloadTrace,
        *,
        gpu_memory_utilization: float = 0.90,
        activation_reserve_gb: float = 0.0,
    ) -> None:
        self.trace = trace
        self.gpu_memory_utilization = gpu_memory_utilization
        self.activation_reserve_gb = activation_reserve_gb

    def predict(
        self,
        candidate: CandidateConfig,
        spec: ServiceSpec,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
    ) -> SimResult:
        try:
            metrics = self._analytic_metrics(candidate, spec, islands, profiles)
        except Exception as exc:  # pragma: no cover - defensive, per Predictor ABC
            return SimResult(
                candidate.id, SimOutcome.CRASHED,
                detail=f"surrogate arithmetic failed: {exc}",
            )
        return SimResult(candidate.id, SimOutcome.OK, metrics=metrics)

    def _analytic_metrics(
        self,
        candidate: CandidateConfig,
        spec: ServiceSpec,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
    ) -> PredictedMetrics:
        seqs = candidate.knobs.max_num_seqs
        decode_tpot: list[float] = []
        prefill_tpot: list[float] = []
        decode_tps = 0.0
        peak_power = 0.0
        idle_power = 0.0
        power_known = True

        for a in candidate.assignments:
            island = islands[a.island_id]
            profile = profiles[island.accelerator_model]
            report = memutil.evaluate(
                candidate.model,
                tp_size=a.tp_size,
                device_memory_gb=island.total_memory_gb / island.size,
                dtype=candidate.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                activation_reserve_gb=self.activation_reserve_gb,
            )
            is_prefill = a.role is Role.PREFILL
            median_len = (
                spec.traffic.input_tokens.p50 if is_prefill
                else spec.traffic.input_tokens.p50 + spec.traffic.output_tokens.p50
            )
            active = max(1, min(seqs, max(1, report.kv_tokens // max(1, median_len))))
            bytes_per_step = report.weight_bytes + active * report.kv_bytes_per_token
            step_ms = bytes_per_step / (profile.memory_bandwidth_gbps * 1e9) * 1e3
            latency_ms = step_ms * ROOFLINE_SLACK

            if is_prefill:
                prefill_tpot.append(latency_ms)
            else:
                decode_tpot.append(latency_ms)
                decode_tps += (active / (latency_ms / 1e3)) * a.dp_replicas

            if profile.power is not None:
                peak_power += profile.power.active_power * a.total_devices
                idle_power += profile.power.idle_power * a.total_devices
            elif profile.tdp_w is not None:
                peak_power += profile.tdp_w * a.total_devices
            else:
                power_known = False

        tpot = max(decode_tpot) if decode_tpot else max(prefill_tpot)
        offered_tps = spec.traffic.arrival_rate_rps * spec.traffic.output_tokens.p50
        utilization = offered_tps / decode_tps if decode_tps > 0 else float("inf")
        attainment = min(1.0, 1.0 / utilization) if utilization > 0 else 1.0
        queue_factor = 1.0 / max(0.02, 1.0 - min(utilization, 0.98))
        prefill_ms = max(prefill_tpot) if prefill_tpot else tpot
        ttft = (prefill_ms * TTFT_PREFILL_STEPS + seqs * TTFT_PER_SEQ_MS) * queue_factor

        completed_tokens = self.trace.total_output_tokens
        completed_requests = self.trace.num_requests
        makespan_s = max(
            self.trace.horizon_s,
            completed_tokens / decode_tps if decode_tps > 0 else self.trace.horizon_s,
        )

        if power_known:
            util_frac = min(1.0, utilization)
            avg_power = idle_power + (peak_power - idle_power) * util_frac
            energy = avg_power * makespan_s
            tokens_per_joule = completed_tokens / energy if energy > 0 else None
        else:
            avg_power = None
            energy = None
            tokens_per_joule = None

        return PredictedMetrics(
            p50_ttft_ms=ttft * 0.6,
            p95_ttft_ms=ttft * 0.9,
            p99_ttft_ms=ttft,
            p50_tpot_ms=tpot * 0.8,
            p95_tpot_ms=tpot * 0.95,
            p99_tpot_ms=tpot,
            throughput_tps=decode_tps,
            slo_goodput_rps=spec.traffic.arrival_rate_rps * attainment,
            slo_attainment=attainment,
            completed_requests=completed_requests,
            completed_tokens=completed_tokens,
            total_energy_j=energy,
            average_power_w=avg_power,
            peak_power_w=peak_power if power_known else None,
            tokens_per_joule=tokens_per_joule,
            sim_wall_seconds=0.0,
        )


def make_predictor(
    kind: str,
    trace: WorkloadTrace,
    *,
    gpu_memory_utilization: float = 0.90,
    activation_reserve_gb: float = 0.0,
) -> Predictor:
    """Named predictor factory, picklable across worker processes by name.

    P1 knows only "surrogate". P2 adds the envelope-cache tier and the full
    LLMServingSim path for verification samples.
    """
    if kind == FIDELITY_SURROGATE:
        return SurrogatePredictor(
            trace,
            gpu_memory_utilization=gpu_memory_utilization,
            activation_reserve_gb=activation_reserve_gb,
        )
    raise ValueError(
        f"unknown predictor kind '{kind}' (P1 supports only 'surrogate'; "
        "'sim'/'envelope' arrive with the P2 tier work)"
    )
