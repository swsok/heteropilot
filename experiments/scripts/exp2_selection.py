"""Exp 2 — heterogeneous resource selection, per placement class (work order §12).

Emits the structured JSON the Exp 2 figure needs: the best SLO-goodput/J plan in
each placement class (single-accelerator-class vs mixed). The committed plan YAML
only preserves the Pareto frontier, so a dominated class (e.g. A5000-only, beaten
by RTXPRO6000) is absent from it; this driver evaluates ALL feasible candidates
and reports the best per class.

Simulate-once (one pass over every simulatable candidate) then group by class and
take the max objective per group. All numbers are LLMServingSim predictions
(absolute rule 3); placeholder-profile islands are excluded and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive, pareto
from planner.predictor import Predictor, SimResult
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.parallel import predict_all
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300


def _class_of(candidate, islands) -> str:
    models = sorted({islands[a.island_id].accelerator_model for a in candidate.assignments})
    return models[0] if len(models) == 1 else "mixed(" + "+".join(models) + ")"


def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    islands_by_id = {i.id: i for i in islands}

    # Use the generator's DEFAULT knob sweep (not a single value): the per-class
    # best goodput/J depends on picking the best knobs per class, which is how the
    # committed Exp 2 result (exp2_summary.md) was produced.
    gen = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False,
    ).generate()
    simulatable = [
        c for c in gen.candidates
        if all(profiles[islands_by_id[a.island_id].accelerator_model].sim_hardware
               for a in c.assignments)
    ]
    print(f"generated {len(gen.candidates)}, simulatable {len(simulatable)}", file=sys.stderr)

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(spec, work_root / "workload.jsonl",
                           num_requests=args.num_requests, seed=args.seed)

    predictor = LLMServingSimPredictor(trace, work_dir=work_root / "sims", timeout_s=args.timeout)

    def _progress(i, total, c):
        if not args.quiet:
            print(f"  [{i + 1}/{total}] {c.id}", file=sys.stderr)

    try:
        raw = predict_all(predictor, simulatable, spec, cluster, islands_by_id, profiles,
                          progress=_progress)
    finally:
        predictor.close()

    class _Replay(Predictor):
        def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
            return raw[candidate.id]

    evaluation = exhaustive.evaluate_candidates(
        simulatable, spec, cluster, islands_by_id, profiles, _Replay()
    )
    by_cand = {c.id: c for c in simulatable}

    # Best feasible plan per placement class, by the primary objective.
    best: dict[str, object] = {}
    for plan in evaluation.feasible_plans:
        if not pareto.can_score(plan, spec.objective.primary)[0]:
            continue
        cls = _class_of(by_cand[plan.candidate.id], islands_by_id)
        val = pareto.objective_value(plan, spec.objective.primary)
        if cls not in best or val > best[cls][0]:
            best[cls] = (val, plan)

    rows = []
    for cls, (val, plan) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        m = plan.predicted
        rows.append({
            "placement_class": cls,
            "candidate_id": plan.candidate.id,
            "goodput_per_joule": val,
            "devices": plan.active_accelerators,
            "p99_ttft_ms": m.p99_ttft_ms,
            "p99_tpot_ms": m.p99_tpot_ms,
            "peak_power_w": m.peak_power_w,
            "slo_attainment": m.slo_attainment,
        })

    profile_paths = [Path(args.root) / a.profile for node in cluster.nodes
                     for a in node.accelerators if a.profile]
    provenance = prov.collect(
        service_spec_path=args.service, cluster_spec_path=args.cluster,
        profile_paths=profile_paths, dataset_path=trace.path, random_seed=args.seed,
        extra={"experiment": "exp2_heterogeneous_selection",
               "generated": len(gen.candidates), "simulatable": len(simulatable),
               "feasible": len(evaluation.feasible_plans),
               "workload": trace.as_provenance(),
               "knobs": "generator default sweep (max_num_seqs / max_num_batched_tokens)"},
    )
    result = {"provenance": provenance, "rows": rows}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp2_selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    lines = ["| placement class | goodput/J | devices | p99 TTFT (ms) | peak W | attain |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(
            f"| {r['placement_class']} | {r['goodput_per_joule']:.3f} | {r['devices']} "
            f"| {r['p99_ttft_ms']:,.0f} | {r['peak_power_w']:,.0f} "
            f"| {r['slo_attainment']:.2f} |"
        )
    (out_dir / "exp2_selection_table.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {out_dir / 'exp2_selection.json'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp 2 per-class selection (§12).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--work-dir", type=Path, default=Path("outputs/exp2_selection"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
