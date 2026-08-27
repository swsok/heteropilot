"""4-combo P/D comparison driver (Phase 5 increment 4, docs/phase5_plan.md §12 Exp 5).

Compares the four role x backend Prefill/Decode combinations against an aggregated
baseline on a fixed fabric:

    GPU-P + GPU-D   (realizable end-to-end on the measured A5000 profile)
    GPU-P + NPU-D   |
    NPU-P + GPU-D   |  SIM-PROXY: any NPU island uses the A5000 compute model as a
    NPU-P + NPU-D   |  stand-in (ascend-sim-proxy.yaml). NOT an NPU measurement.

Efficiency: the generator enumerates many structurally-equivalent P/D candidates
(both island choices, both directions). This driver picks ONE representative per
combo plus the smallest aggregated baseline and simulates only those, so the whole
comparison is a handful of simulator runs, not the full candidate space. Metrics are
then run through the planner's real transfer-cost + feasibility path
(exhaustive.evaluate_candidates), identical to the `plan` command.

Absolute rule 3: every NPU-touching row is labeled SIM-PROXY in the output; the
results doc repeats it. GPU-P+GPU-D is the only combo with a real (A5000) profile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    ExecutionIsland,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive
from planner.plan import CandidateConfig, Role, ServingArch
from planner.predictor import Predictor, SimResult
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300


def _backend_tag(island: ExecutionIsland) -> str:
    return "GPU" if island.backend == "cuda" else "NPU"


def _combo_of(cand: CandidateConfig, islands_by_id: dict[str, ExecutionIsland]) -> str | None:
    """Combo label 'XPU-P + YPU-D' for a pd_split candidate, else None."""
    if cand.serving_arch is not ServingArch.PD_SPLIT:
        return None
    prefill = next(a for a in cand.assignments if a.role is Role.PREFILL)
    decode = next(a for a in cand.assignments if a.role is Role.DECODE)
    p = _backend_tag(islands_by_id[prefill.island_id])
    d = _backend_tag(islands_by_id[decode.island_id])
    return f"{p}-P + {d}-D"


def _select(
    candidates: list[CandidateConfig], islands_by_id: dict[str, ExecutionIsland]
) -> tuple[list[CandidateConfig], dict[str, str]]:
    """One representative pd candidate per combo + the smallest aggregated baseline.

    Deterministic: candidates are visited in sorted-id order so the representative
    is stable across runs (§9 reproducibility)."""
    ordered = sorted(candidates, key=lambda c: c.id)
    chosen: dict[str, CandidateConfig] = {}
    combo_of: dict[str, str] = {}
    for cand in ordered:
        combo = _combo_of(cand, islands_by_id)
        if combo is not None and combo not in chosen:
            chosen[combo] = cand
            combo_of[cand.id] = combo
    # Smallest aggregated baseline (single island, fewest devices) for context.
    aggregated = [
        c for c in ordered
        if c.serving_arch is ServingArch.AGGREGATED and len(c.assignments) == 1
    ]
    if aggregated:
        # Prefer a real (cuda/GPU) island so the baseline is a measured-profile run,
        # not a SIM-PROXY one; then fewest devices, then id for determinism.
        baseline = min(
            aggregated,
            key=lambda c: (
                0 if islands_by_id[c.assignments[0].island_id].backend == "cuda" else 1,
                c.total_devices,
                c.id,
            ),
        )
        tag = _backend_tag(islands_by_id[baseline.assignments[0].island_id])
        chosen[f"aggregated ({tag}-D)"] = baseline
        combo_of[baseline.id] = f"aggregated ({tag} baseline)"
    return list(chosen.values()), combo_of


def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    if not islands:
        print("error: no execution islands in this cluster", file=sys.stderr)
        return 1
    islands_by_id = {i.id: i for i in islands}

    generator = CandidateGenerator(
        spec, cluster, islands, profiles,
        max_num_seqs=(args.max_num_seqs,),
        max_num_batched_tokens=(args.max_num_batched_tokens,),
        enable_prefix_caching=False,
        enable_bound_pruning=not args.oracle,
        enable_pd=True,
    )
    generation = generator.generate()
    selected, combo_of = _select(generation.candidates, islands_by_id)
    print(
        f"generated {len(generation.candidates)} candidate(s); "
        f"simulating {len(selected)} representative(s)",
        file=sys.stderr,
    )

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(
        spec, work_root / "workload.jsonl", num_requests=args.num_requests, seed=args.seed
    )

    predictor = LLMServingSimPredictor(trace, work_dir=work_root / "sims", timeout_s=args.timeout)
    raw: dict[str, SimResult] = {}
    try:
        for i, cand in enumerate(selected):
            if not args.quiet:
                print(f"  [{i + 1}/{len(selected)}] simulating {cand.id}", file=sys.stderr)
            raw[cand.id] = predictor.predict(cand, spec, cluster, islands_by_id, profiles)
    finally:
        predictor.close()

    class _Replay(Predictor):
        def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
            return raw[candidate.id]

    evaluation = exhaustive.evaluate_candidates(
        selected, spec, cluster, islands_by_id, profiles, _Replay()
    )
    xfer_by_id = {t["candidate_id"]: t for t in evaluation.pd_transfers}

    # Build one row per selected candidate.
    feasible_ids = {p.candidate.id for p in evaluation.feasible_plans}
    plan_by_id = {p.candidate.id: p for p in evaluation.feasible_plans}
    infeasible_by_id = {
        p.candidate.id: report for (p, report) in evaluation.infeasible_plans
    }
    sim_error_by_id = {
        r.candidate_id: r.reason
        for r in evaluation.rejections
        if r.stage.value == "sim_error"
    }

    # Provenance is DERIVED from the profiles the candidate actually uses, never
    # hardcoded. The original version of this driver hardwired
    # "SIM-PROXY (RTXPRO6000 model)" for any NPU row, which was true for
    # pd-4combo-sim.yaml but silently mislabels a measured RNGD profile as a proxy
    # the moment the fixture changes - the exact failure absolute rule 3 exists to
    # prevent, in the direction that understates the evidence.
    def provenance_of(cand) -> str:
        parts = []
        for assignment in cand.assignments:
            island = islands_by_id[assignment.island_id]
            profile = profiles.get(island.accelerator_model)
            source = getattr(getattr(profile, "source", None), "value", "unknown")
            role = getattr(assignment.role, "value", str(assignment.role))
            parts.append(f"{island.accelerator_model}:{source}[{role}]")
        return " + ".join(parts)

    # Any rejection stage, not just sim_error. A candidate rejected at, say, the
    # analytical bound stage appears in NEITHER plan_by_id NOR infeasible_plans, so
    # without this its row came out entirely blank - "-" in every column - which
    # reads as "no result" when the truth is "rejected, and here is why".
    rejection_by_id = {
        r.candidate_id: f"{r.stage.value}: {r.reason}"
        for r in evaluation.rejections
    }

    rows = []
    for cand in selected:
        combo = combo_of.get(cand.id, "?")
        row: dict[str, object] = {
            "combo": combo,
            "candidate_id": cand.id,
            "provenance": provenance_of(cand),
            "feasible": cand.id in feasible_ids,
        }
        if cand.id in sim_error_by_id:
            row["outcome"] = "sim_error"
            row["detail"] = sim_error_by_id[cand.id]
        elif cand.id in plan_by_id:
            m = plan_by_id[cand.id].predicted
            row.update(_metrics_row(m))
        elif cand.id in infeasible_by_id:
            report = infeasible_by_id[cand.id]
            # metrics still exist on the infeasible plan
            m = next(
                p.predicted for (p, _r) in evaluation.infeasible_plans
                if p.candidate.id == cand.id
            )
            row.update(_metrics_row(m))
            row["violations"] = [
                f"{v.metric}={v.predicted:.1f} vs {v.target:.1f}" for v in report.violations
            ]
        elif cand.id in rejection_by_id:
            row["outcome"] = "rejected"
            row["detail"] = rejection_by_id[cand.id]
        info = xfer_by_id.get(cand.id)
        if info is not None:
            row["xfer_ms_p99"] = info.get("xfer_ms_p99")
            row["xfer_class_default"] = info.get("class_default")
        rows.append(row)

    # Order rows: the four combos, then the aggregated baseline.
    combo_order = {
        "GPU-P + GPU-D": 0, "GPU-P + NPU-D": 1, "NPU-P + GPU-D": 2, "NPU-P + NPU-D": 3,
    }
    rows.sort(key=lambda r: combo_order.get(str(r["combo"]), 9))

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
            "experiment": "phase5_increment4_pd_4combo",
            "npu_is_sim_proxy": True,
            "npu_proxy_note": (
                "NPU islands use the A5000 compute model (ascend-sim-proxy.yaml); every "
                "NPU-touching combo is simulator-only proxy data, not an NPU measurement"
            ),
            "knobs": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
            "workload": trace.as_provenance(),
        },
    )
    result = {"provenance": provenance, "rows": rows}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pd_4combo.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )

    table = render_table(rows)
    (out_dir / "pd_4combo_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwrote {out_dir / 'pd_4combo.json'}", file=sys.stderr)
    return 0


def _metrics_row(m) -> dict:
    return {
        "p99_ttft_ms": m.p99_ttft_ms,
        "p99_tpot_ms": m.p99_tpot_ms,
        "slo_attainment": m.slo_attainment,
        "total_energy_j": m.total_energy_j,
        "tokens_per_joule": m.tokens_per_joule,
    }


def render_table(rows: list[dict]) -> str:
    out = [
        "| combo | provenance | feasible | p99 TTFT (ms) | p99 TPOT (ms) | attainment "
        "| energy (J) | tok/J | P/D xfer p99 (ms) |",
        "| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(v, spec: str = ",.1f") -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    for r in rows:
        out.append(
            f"| {r.get('combo', '-')} | {r.get('provenance', '-')} "
            f"| {'yes' if r.get('feasible') else 'no'} | {fmt(r.get('p99_ttft_ms'))} "
            f"| {fmt(r.get('p99_tpot_ms'))} | {fmt(r.get('slo_attainment'), '.2f')} "
            f"| {fmt(r.get('total_energy_j'), ',.0f')} | {fmt(r.get('tokens_per_joule'), '.3f')} "
            f"| {fmt(r.get('xfer_ms_p99'))} |"
        )
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="4-combo P/D comparison (Phase 5 increment 4).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/.hp-pd-combo/work"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/.hp-pd-combo"))
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
