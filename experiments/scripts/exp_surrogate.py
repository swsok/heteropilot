"""Exp: surrogate top-K accuracy (work order §5.4 stage 6).

Measures — never asserts — the cost of the stage-6 surrogate top-K. The analytical
roofline ranks all candidates; only the top-K would be simulated in a real run.
The question this driver answers empirically: how much optimality does top-K trade
for the N/K simulation speedup?

Simulate-once, replay-select (same shape as exp_baselines.py): every candidate is
simulated exactly once into a shared cache; the oracle optimum and each K's
surrogate pick are then read off that cache. Per K:

  recall@K          = 1.0 if the oracle optimum survived the top-K else 0.0
  regret@K          = max(0, (oracle_value - surrogate_value) / oracle_value)
                      (0 even when the exact optimum drops, if a tied/near
                       candidate survived)
  speedup@K         = N / K (candidates simulated: oracle N vs surrogate K)
  false_infeasible  = oracle has a feasible plan but the top-K has none

The recall/regret-vs-K curve IS the honest accuracy claim; no accuracy number is
hardcoded anywhere. All numbers are LLMServingSim predictions (rule 3);
placeholder-profile islands are excluded and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive, pareto
from planner.optimizer.surrogate import AnalyticalRooflineRanker
from planner.predictor import Predictor, SimResult
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.parallel import predict_all
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 120


def _simulatable(candidate, islands, profiles) -> bool:
    return all(
        profiles[islands[a.island_id].accelerator_model].sim_hardware is not None
        for a in candidate.assignments
    )


def _best(plans, spec):
    scorable = [p for p in plans if pareto.can_score(p, spec.objective.primary)[0]]
    if not scorable:
        return None
    return pareto.rank(scorable, spec.objective.primary, spec.objective.secondary)[0].plan


def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    islands_by_id = {i.id: i for i in islands}

    gen = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_bound_pruning=True, enable_pd=True,
    ).generate()
    simulatable = [c for c in gen.candidates if _simulatable(c, islands_by_id, profiles)]
    n = len(simulatable)
    print(f"generated {len(gen.candidates)}, simulatable {n}", file=sys.stderr)
    if n == 0:
        print("error: nothing simulatable", file=sys.stderr)
        return 1

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
                          max_workers=args.workers, progress=_progress)
    finally:
        predictor.close()

    class _Replay(Predictor):
        def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
            return raw[candidate.id]

    evaluation = exhaustive.evaluate_candidates(
        simulatable, spec, cluster, islands_by_id, profiles, _Replay()
    )
    feasible = {p.candidate.id: p for p in evaluation.feasible_plans}
    oracle_best = _best(list(feasible.values()), spec)
    oracle_value = (
        pareto.objective_value(oracle_best, spec.objective.primary) if oracle_best else None
    )
    oracle_id = oracle_best.candidate.id if oracle_best else None

    ordered = AnalyticalRooflineRanker().order(simulatable, spec, islands_by_id, profiles)
    ordered_ids = [c.id for c in ordered]

    ks = sorted({k for k in args.k_values if 1 <= k <= n} | {n})
    rows = []
    for k in ks:
        top_ids = set(ordered_ids[:k])
        sub = [feasible[i] for i in top_ids if i in feasible]
        surr_best = _best(sub, spec)
        surr_value = (
            pareto.objective_value(surr_best, spec.objective.primary) if surr_best else None
        )
        recall = 1.0 if oracle_id in top_ids else 0.0
        if oracle_value and oracle_value > 0 and surr_value is not None:
            regret = max(0.0, (oracle_value - surr_value) / oracle_value)
        elif surr_value is None:
            regret = 1.0
        else:
            regret = None
        rows.append({
            "k": k,
            "recall": recall,
            "regret": regret,
            "speedup": round(n / k, 2),
            "surrogate_best_id": surr_best.candidate.id if surr_best else None,
            "surrogate_goodput_per_joule": surr_value,
            "false_infeasible": bool(feasible) and not sub,
        })

    profile_paths = [Path(args.root) / a.profile for node in cluster.nodes
                     for a in node.accelerators if a.profile]
    provenance = prov.collect(
        service_spec_path=args.service, cluster_spec_path=args.cluster,
        profile_paths=profile_paths, dataset_path=trace.path, random_seed=args.seed,
        extra={"experiment": "exp_surrogate_topk", "candidates": n,
               "oracle_best_id": oracle_id, "oracle_goodput_per_joule": oracle_value,
               "surrogate": "roofline", "workload": trace.as_provenance()},
    )
    result = {"provenance": provenance, "candidates": n,
              "oracle_best_id": oracle_id, "oracle_goodput_per_joule": oracle_value,
              "rows": rows}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surrogate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )

    def fmt(v, s=".3f"):
        return format(v, s) if isinstance(v, (int, float)) else "-"
    lines = [
        f"# Surrogate top-K accuracy (N={n} candidates, oracle goodput/J = "
        f"{fmt(oracle_value)})",
        "",
        "| K | recall@K | regret@K | speedup | false-infeasible |",
        "| ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | {fmt(r['recall'], '.0f')} | {fmt(r['regret'])} "
            f"| {fmt(r['speedup'], '.1f')}x | {'yes' if r['false_infeasible'] else 'no'} |"
        )
    (out_dir / "surrogate_table.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {out_dir / 'surrogate.json'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp: surrogate top-K accuracy (§5.4 stage 6).")
    p.add_argument("--service", required=True, type=Path)
    p.add_argument("--cluster", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20],
                   help="K values to sweep (N is always appended).")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--workers", type=int, default=None,
                   help="concurrent candidate simulations (default: ~half the CPUs, capped 32)")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/exp_surrogate"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
