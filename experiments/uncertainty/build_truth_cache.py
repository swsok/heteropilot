"""F2's truth cache: P/D candidates, simulated once, with D40's mirrors flagged.

`WORK_ORDER_uq_stage_b_plus.md` STEP C1. E-B1 to E-B3 all ran on a corpus built
with ``enable_pd=False``, so `link_bw` flipped nothing and one input carried the
whole decision. F2 turns P/D on and moves the TTFT SLO into the middle regime so
that more than one input kind can matter.

**The E-A1 cache is reused, not rebuilt.** The TTFT SLO enters neither
`EnvelopeKey` -- model, dtype, placement, scheduler_config_hash, network_class,
workload_bucket -- nor the trace digest, so every aggregated candidate E-A1
already simulated is a hit. Only the P/D candidates are new. The reuse is
reported rather than assumed: hits, misses and the resulting simulation count all
go to `f2_cache_summary.json`.

**D40 is worked around, not fixed.** `EnvelopeCache`'s placement key records each
island's shape but not WHICH island got which share, so a pair differing only by
mirroring the split collapses to one entry and the cache serves one candidate's
metrics to both. Re-keying invalidates every committed corpus, which is a
decision and not this script's to make (absolute rule A4). Instead the mirror
pairs are identified structurally, one member of each is kept as the
representative, and the rest are excluded from the truth with the exclusion
recorded. A truth built over both members would count one simulation twice.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/build_truth_cache.py \\
        --fixture experiments/uncertainty/fixtures/f2.json \\
        --reuse outputs/uncertainty/ea1/cache \\
        --out-dir outputs/uncertainty/f2 --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from experiments.uncertainty.ea1_margin_modes import _IslandOperatingPoint, _refusing
from planner.candidate_generator import CandidateGenerator
from planner.envelope import EnvelopeCache, workload_bucket, workload_shape
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.optimizer.margin import AccuracyDomainMargin
from planner.plan import CandidateConfig, RejectionStage, ServingArch
from planner.predictor.calibration import load_accuracy_domains, load_calibrations
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.util import provenance as prov
from planner.util.workload import generate_trace

#: E-A1's TTFT SLO. A F2 fixture that still carries it is a copy-paste,
#: not a fixture (`docs/d23_revalidation.md` §3: 25 000 ms is the loose
#: regime, where the winner is aggregated and no link input matters).
E_A1_TTFT_MAX_MS = 25000.0


def load_fixture(path: Path) -> dict:
    """Read an F2 fixture, refusing one that would silently rebuild E-A1.

    Two mistakes are cheap to make and expensive to notice: leaving `enable_pd`
    off, which produces an aggregated corpus indistinguishable from E-A1's, and
    pointing at the E-A1 service spec, whose 25 000 ms TTFT SLO puts every
    candidate in the loose regime where link bandwidth flips nothing. Both are
    the premise of STEP C1 rather than an option in it, so both are checked here
    instead of in a comment.
    """
    fixture = json.loads(path.read_text())
    if not fixture.get("enable_pd"):
        raise ValueError(f"{path}: f2 fixtures must set enable_pd: true -- "
                         "that is the point")
    spec = load_service_spec(Path(fixture["service"]))
    if spec.slo.ttft.max_ms == E_A1_TTFT_MAX_MS:
        raise ValueError(
            f"{path}: the service spec's TTFT SLO is {E_A1_TTFT_MAX_MS:g} ms, "
            "which is E-A1's -- F2 exists to move it into the middle regime")
    return fixture


def mirror_groups(candidates: list[CandidateConfig], cache: EnvelopeCache
                  ) -> dict[str, list[str]]:
    """{representative id: [every id that shares its cache entry]}.

    Grouped by the planner's OWN key rather than by a re-derived one. Two
    candidates are mirrors exactly when `EnvelopeCache` cannot tell them apart,
    so asking the cache is both the correct test and the one that cannot drift
    from it: `placement` is built from `(accelerator, role, tp, pp, ep, dp)` per
    assignment, sorted -- the island id is absent, which is D40 in one line.

    The representative is the lexicographically smallest id, so a re-run picks
    the same one.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for c in candidates:
        key = cache.cache_key(c)
        if key is None:      # unkeyable (unknown island); left alone
            continue
        groups[key].append(c.id)
    out = {}
    for ids in groups.values():
        if len(ids) > 1:
            ordered = sorted(ids)
            out[ordered[0]] = ordered
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--reuse", type=Path, default=None,
                    help="an existing cache to seed from; its entries are copied, "
                         "never written through, so the source stays as it was")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--truth-out", type=Path,
                    default=Path("experiments/uncertainty/results/f2_truth.md"))
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate, key and report; simulate nothing")
    args = ap.parse_args()

    try:
        fixture = load_fixture(args.fixture)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    spec = load_service_spec(Path(fixture["service"]))
    cluster = load_cluster_spec(Path(fixture["cluster"]))
    profiles = load_profiles_for(cluster, Path("."))
    islands = detect_islands(cluster, profiles)

    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pre_existing = len(list(cache_dir.glob("*.json")))
    seeded = 0
    if args.reuse:
        for src in sorted(Path(args.reuse).glob("*.json")):
            dst = cache_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                seeded += 1
    print(f"{pre_existing} entries already here; seeded {seeded} more from "
          f"{args.reuse}", file=sys.stderr)

    work = args.out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(spec, work / "workload.jsonl",
                           num_requests=fixture["num_requests"], seed=fixture["seed"])
    _tiers, island_hw, _w = exhaustive._profile_tiers(spec, islands, profiles)
    # `_IslandOperatingPoint`, not a plain `EnvelopeCache`: the E-A1 entries this
    # run seeds from predate the cache recording `operating_point`, so without the
    # backfill every reused candidate would reach the margin policy with no
    # operating point and come back `unmeasured` -- refusing the whole aggregated
    # corpus for a reason that has nothing to do with the domains. The backfill is
    # exact only where a run has one phase per hardware, which is why it matters
    # that the entries needing it are E-A1's aggregated ones: the P/D entries this
    # run writes carry the operating point the predictor actually recorded.
    cache = _IslandOperatingPoint(
        cache_dir, spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=TopologyGraph(cluster).reduce_for_simulator(islands).link_bw_gbps,
        trace_digest=prov.hash_file(trace.path),
        island_hw=island_hw,
    )

    generation = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_pd=True,
    ).generate()
    candidates = generation.candidates
    mirrors = mirror_groups(candidates, cache)
    excluded = {i for ids in mirrors.values() for i in ids[1:]}
    kept = [c for c in candidates if c.id not in excluded]
    print(f"generated {len(candidates)}; {len(mirrors)} mirror group(s) collapse "
          f"{len(excluded)} candidate(s); {len(kept)} kept", file=sys.stderr)

    hits = sum(1 for c in kept if cache.get(c) is not None)
    print(f"cache covers {hits} of {len(kept)}; {len(kept) - hits} need simulating",
          file=sys.stderr)

    summary = {
        "fixture": str(args.fixture),
        "generated": len(candidates),
        "mirror_groups": len(mirrors),
        "excluded_mirror": len(excluded),
        "kept": len(kept),
        "seeded_from": str(args.reuse) if args.reuse else None,
        "entries_present_before_run": pre_existing,
        "seeded_entries": seeded,
        "cache_hits_before_run": hits,
        "to_simulate": len(kept) - hits,
        "dry_run": args.dry_run,
    }
    (args.out_dir / "mirror_pairs.json").write_text(
        json.dumps({"representative_to_members": mirrors,
                    "excluded": sorted(excluded)}, indent=2) + "\n")

    if args.dry_run:
        (args.out_dir / "f2_cache_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0

    # The truth is judged by `main`'s committed domains under `outside_domain:
    # refuse`, per STEP C1. One pass, not two: the margin policy is applied after
    # the metrics exist, so judging and cache-filling are the same walk over the
    # corpus and cannot disagree about what was simulated.
    domains = _refusing(load_accuracy_domains(".", None))
    policy = AccuracyDomainMargin(
        domains, shape=workload_shape(spec),
        calibration=load_calibrations(".", None), bucket=workload_bucket(spec),
    )

    predictor = LLMServingSimPredictor(trace, work_dir=work / "sims",
                                       timeout_s=args.timeout)
    captured: list[exhaustive.SearchResult] = []
    started = time.monotonic()
    try:
        output = exhaustive.search(
            spec, cluster, islands, profiles, predictor,
            cache=cache, enable_pd=True, margin_policy=policy,
            candidate_filter=lambda c: c.id not in excluded,
            max_workers=args.workers,
            on_evaluation=lambda result, _cands: captured.append(result),
        )
    finally:
        predictor.close()
    evaluation = captured[0]
    pd_ids = {c.id for c in kept if c.serving_arch is ServingArch.PD_SPLIT}
    feasible_pd = [p for p in evaluation.feasible_plans if p.candidate.id in pd_ids]
    best = output.recommended

    summary["wall_seconds"] = time.monotonic() - started
    summary["cache_stats"] = cache.stats()
    summary["evaluated"] = evaluation.evaluated
    summary["feasible"] = len(evaluation.feasible_plans)
    summary["feasible_pd"] = len(feasible_pd)
    summary["infeasible"] = len(evaluation.infeasible_plans)
    summary["unmeasured"] = len(evaluation.unmeasured)
    summary["rejected_summary"] = output.rejected_summary
    summary["recommended_candidate"] = best.plan.candidate.id if best else None
    summary["objective"] = best.value if best else None
    summary["operating_point_backfilled"] = {
        "per_island": len(set(cache.filled_per_island)),
        "run_level": len(set(cache.filled_run_level)),
    }
    (args.out_dir / "f2_cache_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))

    args.truth_out.parent.mkdir(parents=True, exist_ok=True)
    args.truth_out.write_text(_render_truth(
        fixture, spec, summary, evaluation, output, feasible_pd, pd_ids, domains, trace,
    ))
    print(f"wrote {args.truth_out}", file=sys.stderr)
    return 0


def _shape(candidate_id: str) -> str:
    """The tp/dp pairs in a candidate id, which is what the failures share."""
    return "+".join(sorted(set(re.findall(r"tp\d+-dp\d+", candidate_id))))


def _row(plan) -> str:
    m = plan.predicted
    return (f"| `{plan.candidate.id}` | {plan.candidate.serving_arch.value} | "
            f"{m.p95_ttft_ms:.0f} | {m.p95_tpot_ms:.2f} | "
            f"{m.tokens_per_joule:.1f} | {plan.candidate.total_devices} |")


def _render_truth(fixture, spec, summary, evaluation, output, feasible_pd,
                  pd_ids, domains, trace) -> str:
    """The generated half of F2's truth. The reading of it is authored beside it."""
    best = output.recommended
    lines = [
        "# F2 truth -- P/D on, TTFT SLO in the middle regime",
        "",
        "Generated by `experiments/uncertainty/build_truth_cache.py`; do not hand-edit.",
        "`WORK_ORDER_uq_stage_b_plus.md` STEP C1.",
        "",
        f"- fixture: `{summary['fixture']}`",
        f"- service: `{fixture['service']}` (TTFT SLO "
        f"p{spec.slo.ttft.percentile} <= {spec.slo.ttft.max_ms:g} ms, TPOT SLO "
        f"p{spec.slo.tpot.percentile} <= {spec.slo.tpot.max_ms:g} ms)",
        f"- cluster: `{fixture['cluster']}`",
        f"- {fixture['num_requests']} requests, seed {fixture['seed']}, "
        f"trace digest `{prov.hash_file(trace.path)[:12]}`",
        "- judged by the committed domains in `profiles/calibration/` under "
        "`outside_domain: refuse`",
        "",
        "## Corpus",
        "",
        "| | count |",
        "| --- | ---: |",
        f"| generated (`enable_pd=True`) | {summary['generated']} |",
        f"| D40 mirror groups | {summary['mirror_groups']} |",
        f"| excluded as mirror members | {summary['excluded_mirror']} |",
        f"| kept (the truth corpus) | {summary['kept']} |",
        f"| of which P/D | {len(pd_ids)} |",
        "",
        "The exclusion is D40, not a modelling choice: `EnvelopeCache`'s placement key "
        "records each island's shape but not which island got which share, so the two "
        "members of a mirrored split read one entry. Keeping both would count one "
        "simulation twice. Members are listed in `mirror_pairs.json`.",
        "",
        "## Cache",
        "",
        "| | count |",
        "| --- | ---: |",
        f"| present before this run | {summary['entries_present_before_run']} |",
        f"| copied this run from `{summary['seeded_from']}` | "
        f"{summary['seeded_entries']} |",
        f"| covered before the run | {summary['cache_hits_before_run']} |",
        f"| needed simulating | {summary['to_simulate']} |",
        f"| operating point backfilled per-island | "
        f"{summary['operating_point_backfilled']['per_island']} |",
        f"| operating point backfilled run-level | "
        f"{summary['operating_point_backfilled']['run_level']} |",
        "",
        "## Verdicts",
        "",
        "| verdict | count |",
        "| --- | ---: |",
        f"| feasible | {summary['feasible']} |",
        f"| of which P/D | {summary['feasible_pd']} |",
        f"| infeasible | {summary['infeasible']} |",
        f"| unmeasured (`outside_calibration_domain`) | {summary['unmeasured']} |",
        "",
        "Rejected, by stage: "
        + (", ".join(f"`{k}` {v}" for k, v in sorted(output.rejected_summary.items()))
           or "none") + ".",
        "",
    ]

    # `sim_error` is not a verdict about a candidate, it is the absence of one:
    # the run died, so the candidate was never judged. Reported in full rather
    # than as a count, because STEP C2-C4 inherit this corpus and a silently
    # missing sub-family would look like a family the planner rejected.
    failures = [r for r in evaluation.rejections if r.stage is RejectionStage.SIM_ERROR]
    if failures:
        tails: dict[str, list[str]] = defaultdict(list)
        for r in failures:
            tail = r.reason.strip().splitlines()[-1].strip()
            # Group by the failure, not by the numbers in it: the byte counts
            # differ per run and would split one cause into twenty-two.
            tails[re.sub(r"[0-9]+(\.[0-9]+)?", "N", tail)].append(r.candidate_id)
        lines += ["### Runs that did not finish", "",
                  f"{len(failures)} of the {summary['to_simulate']} candidates that "
                  "needed simulating raised instead of producing metrics. They are "
                  "absent from the truth, not rejected by it.", ""]
        for tail, ids in sorted(tails.items(), key=lambda kv: -len(kv[1])):
            shapes = {_shape(i) for i in ids}
            lines += [f"- {len(ids)} run(s), numbers elided: `{tail}`",
                      f"  - parallelism shape(s): {', '.join(sorted(shapes))}",
                      f"  - e.g. `{sorted(ids)[0]}`"]
        lines += [""]

    lines += [
        "## Winner",
        "",
    ]
    if best is None:
        lines += ["No feasible plan. The completion criterion is not met."]
    else:
        lines += [
            f"- candidate `{best.plan.candidate.id}` "
            f"(`{best.plan.candidate.serving_arch.value}`, plan `{best.plan.plan_id}`)",
            f"- objective `{best.objective.value}` = {best.value:.4g}",
            f"- {best.plan.predicted.tokens_per_joule:.2f} tok/J, "
            f"TTFT p95 {best.plan.predicted.p95_ttft_ms:.0f} ms, "
            f"TPOT p95 {best.plan.predicted.p95_tpot_ms:.2f} ms",
            f"- margin basis: `{best.plan.margin_basis}`",
            "",
            "Top feasible plans as ranked:",
            "",
            "| candidate | arch | TTFT p95 (ms) | TPOT p95 (ms) | tok/J | devices |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        ranked = [best, *output.alternatives]
        lines += [_row(entry.plan) for entry in ranked[:10]]
    lines += [
        "",
        "## Completion criterion (STEP C1)",
        "",
        f"- feasible >= 20: {summary['feasible']} -> "
        f"{'PASS' if summary['feasible'] >= 20 else 'FAIL'}",
        f"- feasible P/D >= 1: {summary['feasible_pd']} -> "
        f"{'PASS' if summary['feasible_pd'] >= 1 else 'FAIL'}",
        "",
        "## Domains in force",
        "",
        "| hardware | conc range | points | outside_domain |",
        "| --- | --- | ---: | --- |",
    ]
    for hw, d in sorted(domains.items()):
        lines.append(f"| {hw} | {d.conc_min:g} to {d.conc_max:g} | "
                     f"{len(d.points)} | {d.outside_domain} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
