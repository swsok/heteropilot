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

Extended 2026-09-08 (`WORK_ORDER_rps_aware.md` STEP 4) with two options, because
STEP 1 found `--top-k 20` turning a FEASIBLE plan INFEASIBLE on `pd-rngd-gpu` --
a case the shipped curve had not covered:

  --cache-dir  replay an existing `EnvelopeCache` corpus instead of simulating.
               A past sweep's cache is hours of simulation already paid for, and
               the whole driver is a pure function of {candidate -> SimResult}.
               Candidates with no cache entry are reported and excluded, so a
               partial corpus narrows the claim instead of silently biasing it.
  --rankers    score several orderings on the same corpus. The shipped ranker is
               `roofline`; the others exist to be compared against it, and NONE of
               them is the default until a curve says it should be.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive, pareto
from planner.optimizer.surrogate import AnalyticalRooflineRanker, SurrogateRanker
from planner.predictor import Predictor, SimOutcome, SimResult
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


def _rankers() -> dict[str, SurrogateRanker]:
    """Orderings to compare. `roofline` is what ships; the rest are candidates.

    `floor` and `floor_then_tpj` exist because the shipped proxy's tok/J is
    algebraically invariant to TP and DP -- throughput and power both scale with
    `tp * dp`, so the ratio cancels -- which leaves the parallelism axis invisible
    to it. The roofline TPOT floor is the one term in `greedy.estimate` that does
    vary with parallelism, and it is currently used only as a discarded binary flag.
    """
    from planner.optimizer import greedy as _greedy

    class _KeyRanker(AnalyticalRooflineRanker):
        def __init__(self, keyfn):
            self._keyfn = keyfn

        def order(self, candidates, spec, islands, profiles, *,
                  gpu_memory_utilization: float = 0.90):
            est = {
                e.candidate_id: e for e in _greedy.rank(
                    candidates, spec, islands, profiles,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
            }
            return sorted(candidates, key=lambda c: (self._keyfn(est[c.id]), c.id))

    return {
        "roofline": AnalyticalRooflineRanker(),
        "floor": _KeyRanker(lambda e: e.roofline_tpot_ms),
        "floor_then_tpj": _KeyRanker(
            lambda e: (e.roofline_tpot_ms, -e.proxy_tokens_per_joule)),
        "tpj_then_floor": _KeyRanker(
            lambda e: (-e.proxy_tokens_per_joule, e.roofline_tpot_ms)),
    }


def _load_cache_corpus(cache_dir: Path) -> dict[str, SimResult]:
    """Read an EnvelopeCache directory as {candidate_id: SimResult}.

    Read by candidate_id rather than by recomputing the cache key: the key binds
    the trace digest and link bandwidth of the run that wrote it, and the point
    here is to reuse a corpus produced under those conditions, not to re-derive
    them. What that costs is stated in the output -- the corpus is whatever that
    sweep simulated, so it fixes the workload this measurement speaks about.
    """
    from planner.plan import PredictedMetrics

    out: dict[str, SimResult] = {}
    for path in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            metrics = PredictedMetrics.model_validate(payload["metrics"])
        except Exception as exc:  # a half-written or foreign file is not a corpus entry
            print(f"  skipping unreadable {path.name}: {exc}", file=sys.stderr)
            continue
        out[payload["candidate_id"]] = SimResult(
            candidate_id=payload["candidate_id"],
            outcome=SimOutcome.OK,
            metrics=metrics,
            warnings=["metrics replayed from a cached corpus"],
        )
    return out


def run(args: argparse.Namespace) -> int:
    spec = load_service_spec(args.service)
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    islands_by_id = {i.id: i for i in islands}

    gen = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_bound_pruning=True,
        enable_pd=args.enable_pd,
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

    replayed_from = None
    if args.cache_dir:
        # Replay: the corpus decides which candidates this measurement can speak
        # about. Anything the sweep did not simulate is dropped here and counted,
        # so a partial corpus narrows the claim rather than skewing it.
        replayed_from = str(args.cache_dir)
        corpus = _load_cache_corpus(Path(args.cache_dir))
        hits = [c for c in simulatable if c.id in corpus]
        if not hits:
            print("error: the cache covers none of the generated candidates -- "
                  "wrong fixture, or a different --enable-pd setting?", file=sys.stderr)
            return 1
        # A candidate absent from the corpus is NOT removed from the ranking. The
        # cache stores only `result.ok`, so an absent candidate is one the sweep
        # simulated and got nothing usable from -- a failure or a timeout. It still
        # consumed a top-K slot in the real run, and dropping it here would delete
        # exactly the candidates that rank high and deliver nothing, flattering
        # whichever ranker likes them. So it stays in the order and yields no plan.
        raw = {
            c.id: corpus.get(c.id) or SimResult(
                candidate_id=c.id, outcome=SimOutcome.CRASHED,
                detail="no cache entry: the sweep simulated this and stored no result",
            )
            for c in simulatable
        }
        print(f"replaying {len(hits)} cached results from {args.cache_dir}; "
              f"{n - len(hits)} of {n} candidates have no entry and count as "
              f"simulated-without-result", file=sys.stderr)
    else:
        predictor = LLMServingSimPredictor(
            trace, work_dir=work_root / "sims", timeout_s=args.timeout)

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

    available = _rankers()
    unknown = [r for r in args.rankers if r not in available]
    if unknown:
        print(f"error: unknown ranker(s) {unknown}; have {sorted(available)}",
              file=sys.stderr)
        return 1

    ks = sorted({k for k in args.k_values if 1 <= k <= n} | {n})
    by_ranker: dict[str, list[dict]] = {}
    for rname in args.rankers:
        ordered_ids = [
            c.id for c in available[rname].order(
                simulatable, spec, islands_by_id, profiles)
        ]
        rows = []
        for k in ks:
            top_ids = set(ordered_ids[:k])
            sub = [feasible[i] for i in top_ids if i in feasible]
            surr_best = _best(sub, spec)
            surr_value = (
                pareto.objective_value(surr_best, spec.objective.primary)
                if surr_best else None
            )
            recall = 1.0 if oracle_id in top_ids else 0.0
            # `objective_value` is always maximised and NEGATES minimisation
            # objectives, so the denominator must be |oracle|. Dividing by the
            # signed value made regret unmeasurable (None) for every
            # `minimize_energy` spec, which is what `pd-rngd-gpu` uses.
            if surr_value is None:
                regret = 1.0
            elif oracle_value is None or not math.isfinite(oracle_value) or oracle_value == 0:
                regret = None
            else:
                regret = max(0.0, (oracle_value - surr_value) / abs(oracle_value))
            rows.append({
                "k": k,
                "recall": recall,
                "regret": regret,
                "speedup": round(n / k, 2),
                "surrogate_best_id": surr_best.candidate.id if surr_best else None,
                "surrogate_goodput_per_joule": surr_value,
                "false_infeasible": bool(feasible) and not sub,
            })
        by_ranker[rname] = rows
    rows = by_ranker[args.rankers[0]]

    profile_paths = [Path(args.root) / a.profile for node in cluster.nodes
                     for a in node.accelerators if a.profile]
    provenance = prov.collect(
        service_spec_path=args.service, cluster_spec_path=args.cluster,
        profile_paths=profile_paths, dataset_path=trace.path, random_seed=args.seed,
        extra={"experiment": "exp_surrogate_topk", "candidates": n,
               "oracle_best_id": oracle_id, "oracle_goodput_per_joule": oracle_value,
               "surrogate": args.rankers[0], "rankers": list(args.rankers),
               "enable_pd": args.enable_pd, "replayed_from": replayed_from,
               "workload": trace.as_provenance()},
    )
    result = {"provenance": provenance, "candidates": n,
              "oracle_best_id": oracle_id, "oracle_goodput_per_joule": oracle_value,
              "replayed_from": replayed_from,
              "rows": rows, "rows_by_ranker": by_ranker}
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
        f"Oracle optimum: `{oracle_id}`.",
    ]
    if replayed_from:
        lines += ["", f"Replayed from `{replayed_from}` -- the corpus, not a fresh "
                      "sweep, so this speaks about the workload that sweep ran."]
    for rname in args.rankers:
        lines += [
            "", f"## ranker `{rname}`" + ("  (shipped)" if rname == "roofline" else ""),
            "",
            "| K | recall@K | regret@K | speedup | false-infeasible |",
            "| ---: | ---: | ---: | ---: | :---: |",
        ]
        for r in by_ranker[rname]:
            lines.append(
                f"| {r['k']} | {fmt(r['recall'], '.0f')} | {fmt(r['regret'])} "
                f"| {fmt(r['speedup'], '.1f')}x "
                f"| {'yes' if r['false_infeasible'] else 'no'} |"
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
    p.add_argument("--enable-pd", action="store_true", default=True,
                   help="generate P/D candidates (default: on, as before)")
    p.add_argument("--no-enable-pd", dest="enable_pd", action="store_false")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="replay an EnvelopeCache corpus instead of simulating")
    p.add_argument("--rankers", nargs="+", default=["roofline"],
                   help="orderings to score; the first is the headline")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
