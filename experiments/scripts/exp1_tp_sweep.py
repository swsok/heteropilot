"""Exp 1 — same-GPU TP=1/2/4 sweep (work order §12 Exp 1).

Validates the planner pipeline across tensor-parallel degrees on ONE real
accelerator class (A40): it builds a single-replica aggregated candidate at
TP=1, 2 and 4 on the same size-4 A40 island, simulates each through the real
`LLMServingSimPredictor`, scores them with the same `evaluate_candidates` path
the `plan` command uses, and tabulates the §4 metrics (TTFT/TPOT percentiles,
throughput, SLO attainment, average/peak power, energy, tokens/J).

It also reports planner-level metrics (§12): generated vs evaluated candidate
counts and the prune ratio from the real generator, plus per-TP planning wall
time. Prediction error and oracle regret are out of scope here — the former
needs a real deployment (Phase 4 calibration), the latter is covered by the
oracle-agreement tests.

PROVENANCE (absolute rule 3): every number is an LLMServingSim prediction on the
MEASURED A40 profile bundle (profiler/perf/A40/, dummy-weight layerwise, TP 1/2/4
profiled 2026-08). The size-4 island's intra bandwidth is the PCIe bottleneck
(64 GB/s, vendor_spec) — see the cluster file; the TP=2 row is therefore
conservative vs a dedicated NVLink pair.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    ExecutionIsland,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    Role,
    ServingArch,
    VllmKnobs,
)
from planner.predictor import Predictor, SimResult
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300
TP_DEGREES = (1, 2, 4)


def _pick_island(islands: list[ExecutionIsland], want_tp: int) -> ExecutionIsland:
    """Smallest island that can host `want_tp` devices, deterministically."""
    fits = sorted((i for i in islands if i.size >= want_tp), key=lambda i: (i.size, i.id))
    if not fits:
        raise SystemExit(f"no island has >= {want_tp} devices for TP={want_tp}")
    return fits[0]


def _candidate(spec, island: ExecutionIsland, tp: int, knobs: VllmKnobs) -> CandidateConfig:
    return CandidateConfig(
        id=f"exp1-tp{tp}",
        model=spec.model,
        dtype=spec.service.dtype,
        assignments=[
            IslandAssignment(island_id=island.id, role=Role.AGGREGATED, tp_size=tp)
        ],
        serving_arch=ServingArch.AGGREGATED,
        knobs=knobs,
    )


def _metrics_row(tp: int, devices: int, feasible: bool, m) -> dict:
    j_per_req = (m.total_energy_j / m.completed_requests
                if m.total_energy_j and m.completed_requests else None)
    return {
        "tp": tp,
        "devices": devices,
        "feasible": feasible,
        "p50_ttft_ms": m.p50_ttft_ms,
        "p99_ttft_ms": m.p99_ttft_ms,
        "p50_tpot_ms": m.p50_tpot_ms,
        "p99_tpot_ms": m.p99_tpot_ms,
        "throughput_tps": m.throughput_tps,
        "slo_goodput_rps": m.slo_goodput_rps,
        "slo_attainment": m.slo_attainment,
        "average_power_w": m.average_power_w,
        "peak_power_w": m.peak_power_w,
        "total_energy_j": m.total_energy_j,
        "tokens_per_joule": m.tokens_per_joule,
        "j_per_request": j_per_req,
    }


def _fmt(v, spec: str = ",.1f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def render_table(rows: list[dict]) -> str:
    out = [
        "| TP | devices | feasible | p50/p99 TTFT (ms) | p50/p99 TPOT (ms) "
        "| throughput (tok/s) | SLO attain | avg/peak W | energy (J) | tok/J | J/req |",
        "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        ttft = f"{_fmt(r.get('p50_ttft_ms'))} / {_fmt(r.get('p99_ttft_ms'))}"
        tpot = f"{_fmt(r.get('p50_tpot_ms'))} / {_fmt(r.get('p99_tpot_ms'))}"
        power = f"{_fmt(r.get('average_power_w'))} / {_fmt(r.get('peak_power_w'))}"
        out.append(
            f"| {r['tp']} | {r['devices']} | {'yes' if r.get('feasible') else 'no'} "
            f"| {ttft} | {tpot} | {_fmt(r.get('throughput_tps'))} "
            f"| {_fmt(r.get('slo_attainment'), '.2f')} | {power} "
            f"| {_fmt(r.get('total_energy_j'), ',.0f')} | {_fmt(r.get('tokens_per_joule'), '.3f')} "
            f"| {_fmt(r.get('j_per_request'), ',.1f')} |"
        )
    return "\n".join(out)


def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    if not islands:
        print("error: no execution islands in this cluster", file=sys.stderr)
        return 1
    islands_by_id = {i.id: i for i in islands}

    knobs = VllmKnobs(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        kv_cache_dtype=spec.service.kv_cache_dtype,
    )
    tps = [int(x) for x in args.tp.split(",")]
    candidates = [_candidate(spec, _pick_island(islands, tp), tp, knobs) for tp in tps]

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(
        spec, work_root / "workload.jsonl", num_requests=args.num_requests, seed=args.seed
    )

    # Planner-level metrics (§12): what the real generator enumerates/prunes.
    gen = CandidateGenerator(
        spec, cluster, islands, profiles,
        max_num_seqs=(args.max_num_seqs,),
        max_num_batched_tokens=(args.max_num_batched_tokens,),
        enable_prefix_caching=False,
        enable_bound_pruning=True,
    ).generate()
    planner_metrics = {
        "generated_candidates": gen.generated,
        "survivors": len(gen.candidates),
        "rejected_candidates": len(gen.rejections),
        "prune_ratio": round(len(gen.rejections) / gen.generated, 3) if gen.generated else None,
    }

    predictor = LLMServingSimPredictor(trace, work_dir=work_root / "sims", timeout_s=args.timeout)
    raw: dict[str, SimResult] = {}
    wall: dict[str, float] = {}
    try:
        for i, cand in enumerate(candidates):
            if not args.quiet:
                print(f"  [{i + 1}/{len(candidates)}] simulating {cand.id}", file=sys.stderr)
            t0 = time.monotonic()
            raw[cand.id] = predictor.predict(cand, spec, cluster, islands_by_id, profiles)
            wall[cand.id] = time.monotonic() - t0
    finally:
        predictor.close()

    class _Replay(Predictor):
        def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
            return raw[candidate.id]

    evaluation = exhaustive.evaluate_candidates(
        candidates, spec, cluster, islands_by_id, profiles, _Replay()
    )
    feasible_ids = {p.candidate.id for p in evaluation.feasible_plans}
    plan_by_id = {p.candidate.id: p for p in evaluation.feasible_plans}
    infeasible = {p.candidate.id: (p, r) for (p, r) in evaluation.infeasible_plans}
    sim_error = {
        r.candidate_id: r.reason for r in evaluation.rejections
        if r.stage.value == "sim_error"
    }

    rows = []
    for cand in candidates:
        tp = cand.assignments[0].tp_size
        devices = cand.total_devices
        if cand.id in sim_error:
            rows.append({"tp": tp, "devices": devices, "feasible": False,
                         "outcome": "sim_error", "detail": sim_error[cand.id]})
            continue
        if cand.id in plan_by_id:
            m = plan_by_id[cand.id].predicted
        else:
            m = infeasible[cand.id][0].predicted
        row = _metrics_row(tp, devices, cand.id in feasible_ids, m)
        row["sim_wall_seconds"] = round(wall.get(cand.id, 0.0), 1)
        if cand.id in infeasible:
            row["violations"] = [
                f"{v.metric}={v.predicted:.1f} vs {v.target:.1f}"
                for v in infeasible[cand.id][1].violations
            ]
        rows.append(row)

    profile_paths = [
        Path(args.root) / a.profile
        for node in cluster.nodes for a in node.accelerators if a.profile
    ]
    provenance = prov.collect(
        service_spec_path=args.service,
        cluster_spec_path=args.cluster,
        profile_paths=profile_paths,
        dataset_path=trace.path,
        random_seed=args.seed,
        extra={
            "experiment": "exp1_tp_sweep",
            "tp_degrees": tps,
            "planner_metrics": planner_metrics,
            "intra_bandwidth_note": (
                "size-4 A40 island intra bandwidth = PCIe bottleneck 64 GB/s "
                "(vendor_spec); the TP=2 row is conservative vs a dedicated NVLink pair"
            ),
            "knobs": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
            "workload": trace.as_provenance(),
        },
    )
    result = {"provenance": provenance, "planner_metrics": planner_metrics, "rows": rows}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp1_tp_sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    table = render_table(rows)
    (out_dir / "exp1_tp_sweep_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nplanner: generated {planner_metrics['generated_candidates']} candidate(s), "
          f"{planner_metrics['survivors']} survived, "
          f"rejected {planner_metrics['rejected_candidates']} "
          f"(prune ratio {planner_metrics['prune_ratio']})")
    print(f"wrote {out_dir / 'exp1_tp_sweep.json'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 1 — same-GPU TP=1/2/4 sweep (§12 Exp 1).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--tp", default="1,2,4", help="comma-separated TP degrees")
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--work-dir", type=Path, default=Path("outputs/exp1"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
