"""Interactive fast-path planning for /api/plan (DESIGN §2.2, FR-T5/FR-A2..A4).

Same pipeline as a batch scenario - islands, candidate generation, Tier-1
envelope lookups, Tier-2 surrogate with calibration margins - but bounded for
interactivity: full simulation is forbidden, and when the candidate space is
large the surrogate keeps only the top-K, marking the response `truncated`.
Everything returned carries its fidelity labels; an interactive answer is
never a verified result.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.optimizer.surrogate import AnalyticalRooflineRanker
from planner.spec import ServiceSpec
from planner.topology import TopologyGraph
from planner.util.workload import generate_trace
from scenariolab.runner.tiers import (
    FIDELITY_ENVELOPE,
    FIDELITY_SURROGATE,
    SharedEnvelope,
    SurrogatePredictor,
    calibration_margins,
    load_calibrations,
    npu_concurrency_extrapolated,
)

#: FR-T5 defaults: candidates beyond the top-K are surrogate-pruned so the
#: response stays interactive; the wall-clock budget is reported alongside.
INTERACTIVE_TOP_K = 64
INTERACTIVE_NUM_REQUESTS = 100
INTERACTIVE_SEED = 42
TIME_BUDGET_S = 10.0

#: FR-S4-style multipliers used to complete a p50-only interactive SLO.
P95_MULTIPLIER = 4
P99_MULTIPLIER = 8


class InteractivePlanError(ValueError):
    """Raised for invalid interactive requests (maps to HTTP 400)."""


def build_service_spec(slo: dict[str, Any]) -> ServiceSpec:
    """ServiceSpec from the interactive request body (FR-A3: same validation
    as load_service_spec, including the traffic-is-required rule)."""
    missing = [k for k in ("rps", "input_p50", "output_p50") if not slo.get(k)]
    if missing:
        raise InteractivePlanError(
            "traffic is required: an SLO alone cannot size a deployment "
            f"(missing: {', '.join(missing)})"
        )
    raw = {
        "service": {
            "model": slo.get("model", "meta-llama/Llama-3.1-8B"),
            "dtype": slo.get("dtype", "bfloat16"),
        },
        "traffic": {
            "arrival_rate_rps": slo["rps"],
            "input_tokens": {
                "p50": int(slo["input_p50"]),
                "p95": int(slo["input_p50"] * P95_MULTIPLIER),
                "p99": int(slo["input_p50"] * P99_MULTIPLIER),
            },
            "output_tokens": {
                "p50": int(slo["output_p50"]),
                "p95": int(slo["output_p50"] * P95_MULTIPLIER),
                "p99": int(slo["output_p50"] * P99_MULTIPLIER),
            },
        },
        "slo": {
            "ttft": {"percentile": 99, "max_ms": slo["ttft_p99_ms"]},
            "tpot": {"percentile": 99, "max_ms": slo["tpot_p99_ms"]},
            **(
                {"max_cluster_power_w": slo["power_cap_w"]}
                if slo.get("power_cap_w") else {}
            ),
        },
        "objective": {
            "primary": "minimize_energy",
            "secondary": "minimize_active_accelerators",
        },
    }
    try:
        return ServiceSpec.model_validate(raw)
    except Exception as exc:
        raise InteractivePlanError(f"invalid SLO request: {exc}") from exc


def plan_interactive(
    cluster_yaml: str | Path,
    slo: dict[str, Any],
    *,
    root: str | Path = ".",
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
    top_k: int = INTERACTIVE_TOP_K,
    num_requests: int = INTERACTIVE_NUM_REQUESTS,
    seed: int = INTERACTIVE_SEED,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the fast path once against a cluster YAML file."""
    return plan_fast(
        build_service_spec(slo), load_cluster_spec(cluster_yaml),
        root=root, envelope_dir=envelope_dir, calibration_dir=calibration_dir,
        top_k=top_k, num_requests=num_requests, seed=seed, work_dir=work_dir,
    )


def plan_fast(
    spec: ServiceSpec,
    cluster: Any,
    *,
    root: str | Path = ".",
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
    top_k: int = INTERACTIVE_TOP_K,
    num_requests: int = INTERACTIVE_NUM_REQUESTS,
    seed: int = INTERACTIVE_SEED,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """The fast path over an in-memory ClusterSpecV2 - the workspace
    placement engine plans on occupancy OVERLAY copies that exist only in
    memory (workspace work order §5.1), so this must not require a file."""
    import tempfile

    started = time.perf_counter()
    root = Path(root)
    profiles = load_profiles_for(cluster, root)
    islands = detect_islands(cluster, profiles)

    ttft_margin = tpot_margin = 0.0
    calibrated = False
    if calibration_dir is not None:
        hardware = {
            profiles[i.accelerator_model].sim_hardware
            or f"<no-sim-hardware:{i.accelerator_model}>"
            for i in islands if i.accelerator_model in profiles
        }
        ttft_margin, tpot_margin, calibrated = calibration_margins(
            load_calibrations(root / calibration_dir), hardware, spec
        )

    with tempfile.TemporaryDirectory(prefix="slab-plan-", dir=work_dir) as tmp:
        trace = generate_trace(
            spec, Path(tmp) / "workload.jsonl", num_requests=num_requests, seed=seed
        )
        cache = None
        if envelope_dir is not None:
            reduction = TopologyGraph(cluster).reduce_for_simulator(islands)
            cache = SharedEnvelope(
                root / envelope_dir, spec,
                accelerator_of={i.id: i.accelerator_model for i in islands},
                link_bw_gbps=reduction.link_bw_gbps,
                readonly=True,  # FR-T5: the interactive path never simulates
            )
        predictor = SurrogatePredictor(trace)
        try:
            output = exhaustive.search(
                spec, cluster, islands, profiles, predictor,
                cache=cache,
                ttft_margin_percent=ttft_margin,
                tpot_margin_percent=tpot_margin,
                surrogate=AnalyticalRooflineRanker(),
                top_k=top_k,
                max_workers=1,
                provenance={
                    "scenariolab": {
                        "interactive": True,
                        "seed": seed,
                        "num_requests": num_requests,
                        "top_k": top_k,
                    }
                },
            )
        finally:
            predictor.close()

    truncated = output.rejected_summary.get("surrogate_pruned", 0) > 0
    plan = output.recommended.plan if output.recommended is not None else None
    hit_ids = set(output.provenance.get("envelope_cache_hit_ids", []))
    fidelity = (
        FIDELITY_ENVELOPE
        if plan is not None and plan.candidate.id in hit_ids
        else FIDELITY_SURROGATE
    )
    npu_flag = plan is not None and npu_concurrency_extrapolated(
        plan.candidate, spec, {i.id: i for i in islands}, profiles
    )
    return {
        "feasible": output.feasible,
        "fidelity": fidelity,
        "calibrated": calibrated,
        "npu_extrapolated": npu_flag,
        "truncated": truncated,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "seed": seed,
        "num_requests": num_requests,
        "planner_output": output.model_dump(mode="json"),
        "calibration": {
            "calibrated": calibrated,
            "ttft_margin_percent": ttft_margin,
            "tpot_margin_percent": tpot_margin,
        },
    }
