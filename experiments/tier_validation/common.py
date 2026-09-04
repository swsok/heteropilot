"""Shared pieces of the E1-E4 harnesses: ranking metrics, report writing,
and the datasheet-driven analytical predictor used where full simulation
would make a sweep intractable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
from planner.plan import CandidateConfig, PredictedMetrics
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.spec import ServiceSpec
from planner.util import memory as memutil
from planner.util import provenance as prov

# ---------------------------------------------------------------------------
# Ranking-agreement metrics (E1)
# ---------------------------------------------------------------------------


def top1_match(ranking: list[str], truth: list[str]) -> bool:
    """Do the two rankings agree on the single best candidate?"""
    return bool(ranking and truth and ranking[0] == truth[0])


def topk_contains(ranking: list[str], truth: list[str], k: int = 3) -> bool:
    """Is the ranking's pick within the ground truth's top-k?"""
    return bool(ranking and ranking[0] in truth[:k])


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau over the ids present in BOTH rankings (no scipy).

    tau = (concordant - discordant) / (n*(n-1)/2). 1.0 for identical order,
    -1.0 for reversed. Ids missing from either side are excluded rather than
    silently ranked last - a candidate one leg never evaluated carries no
    ordering information.
    """
    shared = [x for x in a if x in set(b)]
    if len(shared) < 2:
        return 1.0
    pos_b = {x: i for i, x in enumerate(b)}
    n = len(shared)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            # shared is in a-order, so pair (i, j) is ascending in a.
            if pos_b[shared[i]] < pos_b[shared[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


# ---------------------------------------------------------------------------
# Report writing (all experiments)
# ---------------------------------------------------------------------------


def write_report(
    out_dir: Path,
    name: str,
    payload: dict[str, Any],
    *,
    table: str,
    provenance_extra: dict[str, Any] | None = None,
) -> Path:
    """Write <name>.json (with the §3.8 provenance block) and <name>.txt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["provenance"] = prov.collect(extra=provenance_extra or {})
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / f"{name}.txt").write_text(table)
    return json_path


# ---------------------------------------------------------------------------
# Datasheet-driven analytical predictor (E4; also a fast E1 smoke mode)
# ---------------------------------------------------------------------------


class AnalyticalPredictor(Predictor):
    """A Predictor whose numbers come from the accelerator DATASHEET.

    Same structure as tests' MockPredictor (memory roofline + queueing blow-
    up), but the bandwidth/peak/efficiency terms are read from each profile's
    datasheet block, so sweeping datasheet parameters (E4) actually moves the
    predictions - which a datasheet-blind mock cannot do. Absolute numbers
    are proxies; only ordering and internal consistency matter here.
    """

    ROOFLINE_SLACK = 1.2

    def predict(
        self,
        candidate: CandidateConfig,
        spec: ServiceSpec,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
    ) -> SimResult:
        from planner.plan import Role

        seqs = candidate.knobs.max_num_seqs
        median_len = spec.traffic.input_tokens.p50 + spec.traffic.output_tokens.p50

        decode_tpot: list[float] = []
        prefill_tpot: list[float] = []
        decode_tps = 0.0
        for a in candidate.assignments:
            island = islands[a.island_id]
            profile = profiles[island.accelerator_model]
            ds = profile.datasheet
            bw_gbps = (
                ds.memory_bandwidth_gbps
                if ds is not None and ds.memory_bandwidth_gbps is not None
                else profile.memory_bandwidth_gbps
            )
            mem_eff = (
                ds.mem_efficiency if ds is not None and ds.mem_efficiency else 1.0
            )
            flops_eff = (
                ds.flops_efficiency if ds is not None and ds.flops_efficiency else 1.0
            )
            peak_flops = None
            if ds is not None and ds.peak_tflops:
                short = {"bfloat16": "bf16", "float16": "fp16"}.get(
                    candidate.dtype, candidate.dtype
                )
                peak = ds.peak_tflops.get(short)
                peak_flops = peak * 1e12 if peak is not None else None

            report = memutil.evaluate(
                candidate.model, tp_size=a.tp_size,
                device_memory_gb=island.total_memory_gb / island.size,
                dtype=candidate.dtype,
            )
            active = max(1, min(seqs, report.kv_tokens // max(1, median_len)))
            bytes_per_step = report.weight_bytes + active * report.kv_bytes_per_token
            t_mem_s = bytes_per_step / (bw_gbps * 1e9 * mem_eff)
            # Decode compute: ~2 FLOPs per weight byte / dtype byte per token.
            t_comp_s = 0.0
            if peak_flops is not None:
                flops_per_step = 2.0 * (report.weight_bytes / 2) * active
                t_comp_s = flops_per_step / (peak_flops * flops_eff)
            latency_ms = max(t_mem_s, t_comp_s) * 1e3 * self.ROOFLINE_SLACK
            if a.role is Role.PREFILL:
                prefill_tpot.append(latency_ms)
            else:
                decode_tpot.append(latency_ms)
                decode_tps += (active / (latency_ms / 1e3)) * a.dp_replicas

        tpot = max(decode_tpot) if decode_tpot else max(prefill_tpot)
        offered_tps = spec.traffic.arrival_rate_rps * spec.traffic.output_tokens.p50
        utilization = offered_tps / decode_tps if decode_tps > 0 else float("inf")
        attainment = min(1.0, 1.0 / utilization) if utilization > 0 else 1.0
        queue_factor = 1.0 / max(0.02, 1.0 - min(utilization, 0.98))
        prefill_ms = max(prefill_tpot) if prefill_tpot else tpot
        ttft = (prefill_ms * 8.0 + seqs * 0.5) * queue_factor

        tokens = 20_000
        # Energy proxy so goodput/J objectives stay rankable: charge each
        # device its profile active power (or a constant when absent).
        power_w = 0.0
        for a in candidate.assignments:
            island = islands[a.island_id]
            profile = profiles[island.accelerator_model]
            per_dev = profile.power.active_power if profile.power else 300.0
            power_w += per_dev * a.total_devices
        window_s = tokens / decode_tps if decode_tps > 0 else 60.0
        energy = power_w * window_s

        return SimResult(
            candidate.id,
            SimOutcome.OK,
            metrics=PredictedMetrics(
                p50_ttft_ms=ttft * 0.6,
                p95_ttft_ms=ttft * 0.9,
                p99_ttft_ms=ttft,
                p50_tpot_ms=tpot * 0.8,
                p95_tpot_ms=tpot * 0.95,
                p99_tpot_ms=tpot,
                throughput_tps=decode_tps,
                slo_goodput_rps=spec.traffic.arrival_rate_rps * attainment,
                slo_attainment=attainment,
                completed_requests=100,
                completed_tokens=tokens,
                total_energy_j=energy,
                average_power_w=power_w,
                peak_power_w=power_w * 1.1,
                tokens_per_joule=tokens / energy if energy > 0 else None,
                sim_wall_seconds=0.0,
            ),
        )
