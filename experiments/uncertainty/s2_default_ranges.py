#!/usr/bin/env python
"""S2: what the grade DEFAULT ranges change in the measurement plan.

Domain-scoping work order STEP S2; deviations D111. One question, asked on the
fixture V3 actually measured: with the default ranges in `grades.yaml`, where
does `link_bw:pcie-a40a-02` rank?

V3's answer without them was "nowhere". The link is `source: vendor_spec`, its
(kind, grade) row sources no width, so it had no interval to sweep, no regret to
compare, and it left the ranking for the "cannot be decided before measuring"
list. It was in fact the input that explained the whole of a -43.4 % TPOT error,
and measuring it took 0.114 h.

The run is the same fixture, cache and margin policy as E-A1 condition (c) - the
per-point accuracy-domain rule - so nothing here re-simulates and nothing here
re-decides the plan. Only the registry's ranges differ between the two arms:

    with_defaults      profiles/uncertainty/grades.yaml as committed
    without_defaults   the same table with `defaults:` dropped (pre-S2)

`_IslandOperatingPoint` is E-A1's: the committed cache entries carry the served
concurrency per island but not the per-hardware operating point the margin
policy reads, and it fills the one from the other. Without it every candidate is
`unmeasured`, there is no recommendation, and every input is undecidable for a
reason that has nothing to do with this experiment - which is exactly what a
plain `python -m planner plan` against this cache reports.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/s2_default_ranges.py \\
        --cache-dir outputs/uncertainty/ea1/cache \\
        --out experiments/uncertainty/results/s2_default_ranges_table.md \\
        --json-out outputs/uncertainty/s2/s2_default_ranges.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.uncertainty.ea1_margin_modes import MARGIN18, _IslandOperatingPoint
from planner.candidate_generator import CandidateGenerator
from planner.envelope import workload_bucket, workload_shape
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.optimizer.margin import AccuracyDomainMargin
from planner.predictor.calibration import load_accuracy_domains, load_calibrations
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.uncertainty import build_registry, load_costs, load_grades
from planner.uncertainty import measurement_plan as mplan
from planner.uncertainty.perturb import PerturbContext, judged_metrics
from planner.uncertainty.sensitivity import analyze
from planner.util import provenance as prov
from planner.util.workload import generate_trace

#: The input V3 traced its error to, and the reason this experiment exists.
V3_INPUT = "link_bw:pcie-a40a-02"


def _plan_for(grades, *, registry_args, analyze_args, costs, budget_hours):
    """Registry -> sensitivities -> ranked plan, for one grade table."""
    registry = build_registry(*registry_args, grades)
    sensitivities = analyze(*analyze_args[:2], registry, *analyze_args[2:])
    return registry, mplan.build(sensitivities, costs, budget_hours=budget_hours)


def _rows(registry, plan) -> list[dict]:
    """One row per registry input: where it landed and on what interval."""
    rank_of = {i.input_id: i.rank for i in [*plan.items, *plan.uncovered]}
    regret_of = {i.input_id: i.delta_regret for i in [*plan.items, *plan.uncovered]}
    per_hour_of = {i.input_id: i.regret_per_hour for i in [*plan.items, *plan.uncovered]}
    out = []
    for item in registry.items:
        if item.id in rank_of:
            where = "ranked"
        elif item.id in set(plan.undecidable):
            where = "undecidable"
        elif item.id in set(plan.inert):
            where = "inert"
        else:  # pragma: no cover - the four buckets are exhaustive
            where = "unaccounted"
        out.append({
            "input_id": item.id,
            "kind": item.kind.value,
            "grade": item.grade.value,
            "nominal": item.nominal,
            "range": str(item.range),
            "range_source": item.range.range_source,
            "where": where,
            "rank": rank_of.get(item.id),
            "delta_regret": regret_of.get(item.id),
            "regret_per_hour": per_hour_of.get(item.id),
            "cost_hours": item.cost.hours if item.cost is not None else None,
        })
    return sorted(out, key=lambda r: (r["rank"] is None, r["rank"] or 0, r["input_id"]))


def _table(rows: list[dict]) -> str:
    lines = [
        "| input | grade | range | from | where | rank | dR | dR/h |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        rank = "-" if r["rank"] is None else str(r["rank"])
        dr = "-" if r["delta_regret"] is None else f"{r['delta_regret']:,.4g}"
        per_hour = "-" if r["regret_per_hour"] is None else f"{r['regret_per_hour']:,.4g}"
        lines.append(
            f"| `{r['input_id']}` | {r['grade']} | {r['range']} | {r['range_source']} "
            f"| {r['where']} | {rank} | {dr} | {per_hour} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/uncertainty/s2/work"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    fixture = json.loads(MARGIN18.read_text())
    spec = load_service_spec(fixture["service"])
    cluster = load_cluster_spec(fixture["cluster"])
    profiles = load_profiles_for(cluster, Path("."))
    islands = detect_islands(cluster, profiles)
    bucket, shape = workload_bucket(spec), workload_shape(spec)
    domains = load_accuracy_domains(".", None)
    calibration = load_calibrations(".", None)
    _tiers, island_hw, _warnings = exhaustive._profile_tiers(spec, islands, profiles)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(
        spec, args.work_dir / "workload.jsonl",
        num_requests=fixture["num_requests"], seed=fixture["seed"],
    )
    reduction = TopologyGraph(cluster).reduce_for_simulator(islands)
    cache = _IslandOperatingPoint(
        args.cache_dir, spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=reduction.link_bw_gbps,
        trace_digest=prov.hash_file(trace.path),
        island_hw=island_hw,
    )
    predictor = LLMServingSimPredictor(
        trace, work_dir=args.work_dir / "sims", timeout_s=args.timeout,
    )

    by_id = {i.id: i for i in islands}
    policy = AccuracyDomainMargin(
        domains, shape=shape, calibration=calibration, bucket=bucket
    )
    try:
        generation = CandidateGenerator(
            spec, cluster, islands, profiles, enable_prefix_caching=False,
        ).generate()
        evaluation = exhaustive.evaluate_candidates(
            generation.candidates, spec, cluster, by_id, profiles, predictor,
            cache=cache, margin_policy=policy, island_hw=island_hw,
            max_workers=args.workers,
        )
        output = exhaustive.search(
            spec, cluster, islands, profiles, predictor,
            cache=cache, margin_policy=policy, max_workers=args.workers,
        )
    finally:
        predictor.close()

    costs = load_costs()
    committed = load_grades()
    arms = {"with_defaults": committed, "without_defaults": committed.without_defaults()}
    context = PerturbContext.build(
        spec, cluster, by_id, {c.id: c for c in generation.candidates},
        evaluation.operating_points,
    )
    results: dict[str, dict] = {}
    for key, grades in arms.items():
        registry = build_registry(
            cluster, profiles, islands, calibration, grades, spec, costs
        )
        sensitivities = analyze(
            output, registry, judged_metrics(evaluation), policy, spec, context,
            island_hw, grid=args.grid,
        )
        plan = mplan.build(sensitivities, costs, budget_hours=args.budget_hours)
        rows = _rows(registry, plan)
        results[key] = {
            # The digest is the COMMITTED file's on both arms - `without_defaults`
            # edits the loaded table, not a file - so `defaults` is what says
            # which arm this is.
            "grades_digest": grades.digest,
            "defaults": len(grades.defaults),
            "defaults_dropped": key == "without_defaults",
            "ranked": [i.input_id for i in plan.items],
            "undecidable": plan.undecidable,
            "inert": plan.inert,
            "covered_regret": plan.covered_regret,
            "rows": rows,
            "v3_input_rank": next(
                (r["rank"] for r in rows if r["input_id"] == V3_INPUT), None
            ),
            "v3_input_where": next(
                (r["where"] for r in rows if r["input_id"] == V3_INPUT), None
            ),
        }
        print(f"{key}: ranked={len(plan.items)} inert={len(plan.inert)} "
              f"undecidable={len(plan.undecidable)} "
              f"{V3_INPUT} -> {results[key]['v3_input_where']} "
              f"rank={results[key]['v3_input_rank']}")

    recommended = output.recommended
    body = [
        "<!-- GENERATED by experiments/uncertainty/s2_default_ranges.py - do not edit -->",
        "",
        f"Fixture `{fixture['cluster']}` / `{fixture['service']}`, "
        f"{fixture['num_requests']} requests, seed {fixture['seed']}, "
        f"grid {args.grid}, cache `{args.cache_dir}`.",
        "",
        f"Recommendation: `{recommended.plan.candidate.id if recommended else None}`"
        f" - unchanged between the two arms by construction (only the registry's "
        f"ranges differ).",
        "",
    ]
    for key in ("without_defaults", "with_defaults"):
        res = results[key]
        body += [
            f"## {key} ({res['defaults']} default row(s))",
            "",
            _table(res["rows"]),
            "",
            f"`{V3_INPUT}`: **{res['v3_input_where']}**"
            + (f", rank **{res['v3_input_rank']}**" if res["v3_input_rank"] else ""),
            "",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {args.out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "fixture": fixture,
            "bucket": bucket,
            "shape": shape,
            "grid": args.grid,
            "budget_hours": args.budget_hours,
            "recommended_candidate": (
                recommended.plan.candidate.id if recommended else None
            ),
            "arms": results,
            "provenance": prov.collect(),
        }, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
