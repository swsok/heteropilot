"""Where does the surrogate put the true optimum? (WORK_ORDER_rps_aware.md STEP 4)

The top-k experiment of STEP 1 found `--top-k 20` turns a FEASIBLE plan
INFEASIBLE on the `pd-rngd-gpu` fixture. This ranks the same candidate set with
the same surrogate and reports the optimum's rank, so the failure can be
attributed to an ordering rather than to the simulator.

No simulation: generation and the analytical ranker only, seconds not hours.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import greedy
from planner.optimizer.surrogate import AnalyticalRooflineRanker
from planner.spec import load_service_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--enable-pd", action="store_true")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--target", action="append", default=[],
                    help="candidate id whose rank to report; repeatable")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}

    generation = CandidateGenerator(
        spec, cluster, islands, profiles,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False, enable_bound_pruning=True,
        enable_pd=args.enable_pd,
    ).generate()
    candidates = generation.candidates
    print(f"generated {len(candidates)} candidates that survive stages 1-5")

    ordered = AnalyticalRooflineRanker().order(
        candidates, spec, by_id, profiles,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    rank = {c.id: i for i, c in enumerate(ordered)}
    est = {
        e.candidate_id: e for e in greedy.rank(
            candidates, spec, by_id, profiles,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    }

    def row(i: int, c) -> str:
        e = est[c.id]
        return (f"  #{i + 1:<4} {c.id:<62} tok/J {e.proxy_tokens_per_joule:9.4f}  "
                f"tps {e.proxy_throughput_tps:9.1f}  W {e.proxy_power_w or 0:7.1f}  "
                f"floor {e.roofline_tpot_ms:7.2f} ms"
                f"{'  LIKELY-INFEASIBLE' if e.likely_infeasible else ''}")

    print(f"\ntop {args.top} by the surrogate (TPOT SLO {spec.slo.tpot.max_ms} ms):")
    for i, c in enumerate(ordered[:args.top]):
        print(row(i, c))

    for t in args.target:
        if t not in rank:
            print(f"\n!! target {t} is not in the generated set")
            continue
        i = rank[t]
        print(f"\ntarget {t}")
        print(row(i, ordered[i]))
        print(f"  -> surrogate rank {i + 1} of {len(ordered)}; "
              f"{'KEPT' if i < args.top else 'DROPPED'} at --top-k {args.top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
