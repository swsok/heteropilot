"""E-B3: does the closed form rank the same things first? (STEP B4).

B1's whole design bet is that a measurement plan can be built by post-processing
cached predictions instead of re-running the simulator - `perturb` moves the
numbers by arithmetic, and the sweep of `items x grid x candidates` finishes in
seconds rather than weeks. Two of the rules are exact and the rest are
first-order stand-ins, so the bet has a price, and this is where it is paid:
run the plan both ways and see whether the stand-ins put the same items at the
top.

    off : ΔR from `analyze` alone - closed form, five-point grid
    on  : `--resimulate-top N` - the top N items' ΔR recomputed from predictions
          the simulator actually produced at both endpoints of their range

Both ΔR values in a refinement record come from the same two-point grid `{lo,
hi}` and the same scoring, so the only thing that differs is where the metrics
came from. The headline is the Spearman rank correlation between the two
orderings, and beside it the runtime, because a rank correlation of 1.0 is only
interesting next to the hours it saves.

An item with no simulator input to move - a `sim_error`, whose closed form is
exact by construction - is reported as SKIPPED and does not consume one of the N
slots. "We could not check this" and "we checked it and it agreed" are different
claims and this file never merges them.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb3_closed_form_vs_resim.py \\
        --cache-dir outputs/uncertainty/ea1/cache --top 3
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

from experiments.uncertainty.eb1_regret_vs_budget import (
    World,
    _best,
    _default_penalty,
    build_world,
)
from planner.optimizer import pareto
from planner.plan import DeploymentPlan, PlannerOutput
from planner.uncertainty import resimulate as resim
from planner.uncertainty.perturb import JudgedMetrics, PerturbContext
from planner.uncertainty.registry import UncertainInputRegistry, UncertainKind
from planner.uncertainty.sensitivity import analyze, refine
from planner.util import provenance as prov

#: A candidate's x1.0 endpoint must reproduce its cached prediction to this.
#: Anything coarser is a different prediction wearing the same id.
IDENTITY_TOL = 1e-6
#: `LINK_BW` and `LINK_LAT` re-price a transfer term the planner adds itself, so
#: their closed form is EXACT and a resimulation must land on the same dR. This
#: is the second identity control STEP C4 asks for. The tolerance is relative and
#: loose on purpose: the two sides agree on arithmetic, not on float ordering.
LINK_EXACT_RTOL = 1e-3
#: A rank correlation over fewer than this many ACTIVE items says nothing -- PR
#: #86 reported 1.000 over four items of which three were tied at zero, and over
#: one refined item, and neither was evidence. §2.5 makes the rule explicit.
MIN_ACTIVE_FOR_SPEARMAN = 3


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, ties averaged. None when either side is constant.

    Written out rather than pulled from scipy: this repo's venv has no scipy,
    and a rank correlation over a handful of items is four lines.
    """
    if len(a) != len(b) or len(a) < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (stop + 1 < len(order)
                   and values[order[stop + 1]] == values[order[index]]):
                stop += 1
            shared = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = shared
            index = stop + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def mirror_members(path: Path) -> set[str]:
    """Every NON-representative member of C1's mirror groups.

    `mirror_pairs.json` maps a representative id to the full group including
    itself. The representative is the one the cache entry belongs to, so it
    stays; the rest are the duplicates D40 lets read it.
    """
    groups = json.loads(path.read_text())["representative_to_members"]
    return {m for rep, members in groups.items() for m in members if m != rep}


def restrict(world: World, drop: set[str]) -> tuple[World, list[str]]:
    """The same world with `drop` removed from the corpus, truth and context.

    STEP C4 allows a candidate-exclusion argument on `planner/uncertainty/
    resimulate.py`. It is not needed and is not added: excluding candidates
    THERE would exclude them from the resimulation while leaving them in the
    closed form's corpus and in the incumbent search, so the two sides would be
    scored over different candidate sets and the magnitude comparison -- the
    whole point of E-B3 -- would be measuring that difference instead. Removing
    them once, here, keeps `planner/` untouched (A4) and keeps both sides
    honest.
    """
    removed = sorted(drop & set(world.candidates))
    if not removed:
        return world, []
    keep = {k: v for k, v in world.candidates.items() if k not in drop}
    context = PerturbContext(
        spec=world.context.spec, cluster=world.context.cluster,
        islands=world.context.islands,
        candidates={k: v for k, v in world.context.candidates.items()
                    if k not in drop},
        topology=world.context.topology,
        operating_points={k: v for k, v in world.context.operating_points.items()
                          if k not in drop},
    )
    truth = JudgedMetrics.trusted(
        {k: v for k, v in world.truth.items() if k not in drop}
    )
    world.candidates, world.context, world.truth = keep, context, truth
    return world, removed


def endpoint_coverage(endpoint, baseline, touched_ids: set[str]) -> dict:
    """How many of the candidates this endpoint claimed to simulate came back.

    `_endpoint` reports `simulated=len(touched)` -- the number of runs it
    STARTED -- and builds its metric set as `dict(baseline)` updated with
    whatever the evaluation produced. A candidate whose run crashed is simply
    absent from that update, so it keeps the baseline's value and the endpoint
    is indistinguishable from one where the input moved nothing.

    That is not hypothetical. On 2026-09-15 two helper nodes were missing
    `astra-sim/inputs/system/system.json`, every run on them raised before the
    simulator started, and the harness reported four link items as
    `closed=0 resim=0 ... agrees` -- a clean pass produced entirely by failure.

    The discriminator is object IDENTITY, not equality: a candidate that really
    was re-simulated gets a fresh `PredictedMetrics` from `judged_metrics`, even
    when its numbers are unchanged, while one that crashed still holds the very
    object the baseline had. Equality would call a genuinely inert candidate a
    crash, which is the opposite error and just as wrong.
    """
    ran = sorted(
        cid for cid in touched_ids
        if endpoint.metrics.get(cid) is not baseline.get(cid)
    )
    return {
        "value": endpoint.value,
        "touched": len(touched_ids),
        "resimulated": len(ran),
        "missing": sorted(touched_ids - set(ran)),
    }


def coverage_gate(coverages: list[dict]) -> dict:
    """Endpoints that simulated nothing at all, and whether any did.

    Zero is the bright line: a partially failed endpoint is a real corpus with
    holes, which D71 already describes and `f2_truth.md` already lists, but an
    endpoint with no successful run at all carries no information and any dR
    computed from it is the baseline compared with itself.
    """
    empty = [c for c in coverages if c["touched"] and not c["resimulated"]]
    partial = [c for c in coverages
               if c["resimulated"] and c["missing"]]
    return {
        "endpoints": len(coverages),
        "endpoints_with_no_successful_run": len(empty),
        "endpoints_partially_failed": len(partial),
        "total_touched": sum(c["touched"] for c in coverages),
        "total_resimulated": sum(c["resimulated"] for c in coverages),
        "passed": not empty,
    }


def active_records(records) -> list:
    """Checked refinements whose dR is non-zero on at least one side.

    "Active" is the §2.3 sense -- an item that carries regret -- and it is read
    from BOTH sides on purpose. An item the closed form calls zero and the
    resimulation does not is the most interesting row this experiment can
    produce, and taking `active` from the closed form alone would drop it.
    """
    return [r for r in records if not r.skipped
            and max(abs(r.closed_form), abs(r.resimulated)) > 0.0]


def spearman_refusal(n_active: int, n_checked: int) -> str | None:
    """§2.5's reason to refuse a rank correlation, or None to go ahead.

    PR #86 committed two correlations of 1.000 that were evidence of nothing:
    one over four items with three tied at zero, one over a single refined item
    where the two orderings were identical by construction. A correlation over
    ties measures the ties.
    """
    if n_active >= MIN_ACTIVE_FOR_SPEARMAN:
        return None
    return (
        f"not computed: {n_active} item(s) with a non-zero dR among the "
        f"{n_checked} checked, below the {MIN_ACTIVE_FOR_SPEARMAN} §2.5 "
        f"requires. A correlation over items tied at zero measures the ties."
    )


def identity_gate(controls: list[dict], expected: set[str]) -> dict:
    """Which candidates failed the x1.0 control, and whether D40 explains them.

    A candidate whose x1.0 endpoint does not reproduce its cached prediction is
    reading a prediction that is not its own. D40 -- one cache entry serving two
    mirrored placements -- is the one cause known to do that, so mirror members
    still in the corpus are expected. Anything else is a new defect, and every
    magnitude this experiment reports would be carrying it.
    """
    seen = {c for ctl in controls for c in ctl["deviating_candidates"]}
    unexplained = sorted(seen - expected)
    return {
        "tolerance": IDENTITY_TOL,
        "controls_run": len(controls),
        "deviating_candidates": sorted(seen),
        "expected_from_d40_mirrors": sorted(expected),
        "unexplained": unexplained,
        "passed": not unexplained,
    }


# ---------------------------------------------------------------------------
# Sharding: one item is one independent resimulation, so N nodes can share
# ---------------------------------------------------------------------------

#: Fields every shard must agree on before their refinements may be merged.
#: A shard is a resimulation of ONE world; merging two that disagree about the
#: corpus, the incumbent or the penalty would produce a table whose rows were
#: scored against different baselines, which is the failure this experiment is
#: least able to notice from the inside.
SHARD_INVARIANTS = ("corpus_size", "slo_penalty", "incumbent", "grid",
                    "fixture_path", "include_mirrors")


def _as_record(row: dict):
    """A refinement dict from a shard, in the shape the gates read."""
    from types import SimpleNamespace

    return SimpleNamespace(
        input_id=row["input_id"], kind=row.get("kind", ""),
        closed_form=row["closed_form"], resimulated=row["resimulated"],
        skipped=row.get("skipped") or "",
    )


def merge_shards(paths: list[Path]) -> dict:
    """One payload from several single-node runs of disjoint items.

    Every shard recomputes the closed form -- it is 0.7 s and deterministic --
    so the merge checks rather than assumes that they describe the same world:
    the §2.3 invariants must match, and any item appearing in two shards must
    carry the same closed-form dR. A shard from a node with a different corpus
    is a different experiment and is refused rather than averaged.

    Item ownership is exclusive: a refinement for the same input in two shards
    means the split was wrong, and the run is stopped so the duplicate cannot
    quietly take the last writer's value.
    """
    shards = [(path, json.loads(path.read_text())) for path in paths]
    if not shards:
        raise SystemExit("--merge needs at least one shard")
    head_path, head = shards[0]
    for path, shard in shards[1:]:
        for field in SHARD_INVARIANTS:
            if shard.get(field) != head.get(field):
                raise SystemExit(
                    f"{path} and {head_path} disagree on {field!r}: "
                    f"{shard.get(field)!r} vs {head.get(field)!r}. These are "
                    f"different worlds and their refinements cannot be merged."
                )

    closed_by_id: dict[str, float] = {}
    for path, shard in shards:
        for row in shard.get("closed", []):
            prior = closed_by_id.get(row["input_id"])
            value = row.get("delta_regret")
            if prior is not None and value is not None and prior != value:
                raise SystemExit(
                    f"{path}: closed-form dR for {row['input_id']} is {value}, "
                    f"but another shard computed {prior}. The closed form is "
                    f"deterministic, so this is not the same world."
                )
            if value is not None:
                closed_by_id[row["input_id"]] = value

    refinements: dict[str, dict] = {}
    owner: dict[str, Path] = {}
    controls: list[dict] = []
    link_checks: list[dict] = []
    coverages: list[dict] = []
    for path, shard in shards:
        for row in shard.get("refinements", []):
            key = row["input_id"]
            if key in refinements:
                raise SystemExit(
                    f"{key} was resimulated by both {owner[key]} and {path}. "
                    f"Shards must own disjoint items; re-run the split."
                )
            refinements[key], owner[key] = row, path
        controls.extend(shard.get("identity_controls", []))
        link_checks.extend(shard.get("link_exactness_checks", []))
        coverages.extend(shard.get("resimulation_coverage_detail", []))

    records = [_as_record(r) for r in refinements.values()]
    checked = [r for r in records if not r.skipped]
    active = active_records(records)
    note = spearman_refusal(len(active), len(checked))
    rho = None if note else spearman(
        [r.closed_form for r in checked], [r.resimulated for r in checked]
    )
    # Each shard already worked out which candidates D40 entitles to deviate --
    # the mirrors still IN its corpus, which is empty whenever the mirrors were
    # excluded at build. Union those rather than re-deriving it here: the
    # excluded lists name candidates that are absent and so cannot deviate at
    # all, and using them as the expectation would excuse a deviation by a
    # candidate that is not even present.
    expected = {c for _, shard in shards
                for c in shard.get("identity_control_verdict", {})
                .get("expected_from_d40_mirrors", [])}
    verdict = identity_gate(controls, expected)
    # A shard whose runs all crashed must fail the merged result too. The first
    # sharded run of this experiment produced exactly that on two of three
    # nodes, and its rows looked like clean zeros.
    coverage_verdict = coverage_gate(coverages)
    return {
        "merged_from": [str(path) for path in paths],
        "shards": [
            {"path": str(path), "items": [r["input_id"]
                                          for r in shard.get("refinements", [])],
             "resimulate_seconds": shard.get("resimulate_seconds"),
             "provenance": shard.get("provenance")}
            for path, shard in shards
        ],
        **{field: head.get(field) for field in SHARD_INVARIANTS},
        "closed_form_delta_regret": closed_by_id,
        "resimulate_seconds_total": sum(
            shard.get("resimulate_seconds") or 0.0 for _, shard in shards),
        "resimulate_seconds_wall_clock_max": max(
            (shard.get("resimulate_seconds") or 0.0 for _, shard in shards),
            default=0.0),
        "spearman_delta_regret_on_checked": rho,
        "spearman_not_computed_because": note,
        "active_checked_items": [r.input_id for r in active],
        "identity_control_verdict": verdict,
        "resimulation_coverage": coverage_verdict,
        "resimulation_coverage_detail": coverages,
        "link_exactness_checks": link_checks,
        "refinements": list(refinements.values()),
        "identity_controls": controls,
        "provenance": prov.collect(random_seed=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not `required`: `--merge` combines finished shards and reads no cache.
    # Checked below instead, so the error names the actual mistake.
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("outputs/uncertainty/eb3/work"))
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--degrade", nargs="+", default=None,
        help="Pool ids to degrade. Default: the kinds --kinds names.",
    )
    parser.add_argument(
        "--kinds", default="profile,sim_error",
        help="Comma-separated kinds to sweep, or 'all'. The default is the "
             "first-order set the resimulation exists to check, which is what "
             "the committed E-A1 run used. §2.5 asks F2 for 'all': every "
             "resimulable kind, with sim_error still reported SKIPPED.",
    )
    parser.add_argument(
        "--fixture", type=Path, default=None,
        help="Fixture json. Defaults to E-B1's (E-A1's corpus), so the "
             "committed result reproduces.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--merge", type=Path, nargs="+", default=None,
        help="Shard JSONs to merge into one result. Each shard is an ordinary "
             "run over a disjoint --degrade subset, so the items can be split "
             "across machines; this only combines them. Nothing else runs.",
    )
    parser.add_argument(
        "--exclude", type=Path, default=None,
        help="C1's mirror_pairs.json. Drops every non-representative member "
             "from the corpus, the truth and the context before either side "
             "runs (D40).",
    )
    parser.add_argument(
        "--expect-mirrors", type=Path, default=None,
        help="C1's mirror_pairs.json, used ONLY to tell the x1.0 identity gate "
             "which candidates D40 entitles to deviate. Unlike --exclude it "
             "removes nothing. Needed with --include-mirrors: there the mirrors "
             "are in the corpus on purpose, so their deviation is the thing "
             "being measured and must not stop the run.",
    )
    parser.add_argument(
        "--include-mirrors", action="store_true",
        help="Build the corpus WITHOUT C1's mirror exclusion, so D40's "
             "duplicates are present. The 'before' half of §2.5's comparison; "
             "on its own it makes a corpus that is known to double-count.",
    )
    args = parser.parse_args()

    if args.merge:
        payload = merge_shards(list(args.merge))
        out = args.out_json or Path(
            "outputs/uncertainty/eb3/eb3_merged.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"merged {len(args.merge)} shard(s) -> {out}")
        print(f"  resimulated {payload['resimulate_seconds_total']:,.0f}s of "
              f"work; slowest shard {payload['resimulate_seconds_wall_clock_max']:,.0f}s")
        for r in payload["refinements"]:
            if r.get("skipped"):
                print(f"  SKIPPED {r['input_id']}: {str(r['skipped'])[:80]}")
            else:
                print(f"  {r['input_id']}: closed={r['closed_form']:,.4g} "
                      f"resim={r['resimulated']:,.4g}")
        note = payload["spearman_not_computed_because"]
        print(f"  spearman = {payload['spearman_delta_regret_on_checked']}"
              + (f"  [{note}]" if note else ""))
        cov = payload["resimulation_coverage"]
        print(f"  coverage: {cov['total_resimulated']}/{cov['total_touched']} "
              f"runs came back across {cov['endpoints']} endpoint(s)")
        if not cov["passed"]:
            print(f"\nSTOP: {cov['endpoints_with_no_successful_run']} endpoint(s) "
                  f"produced no successful simulation at all; their dR is the "
                  f"baseline compared with itself.")
            return 4
        unexplained = payload["identity_control_verdict"]["unexplained"]
        if unexplained:
            print(f"\nSTOP: {len(unexplained)} candidate(s) fail the x1.0 "
                  f"identity control and are not explained by D40.")
            return 3
        return 0

    if args.cache_dir is None:
        raise SystemExit("--cache-dir is required unless --merge is given")
    kwargs = {"fixture_path": args.fixture} if args.fixture else {}
    if args.include_mirrors:
        kwargs["exclude_mirrors"] = False
    world = build_world(args.cache_dir, args.work_dir, args.workers,
                        keep_predictor=True, **kwargs)
    mirrors = mirror_members(args.exclude) if args.exclude else set()
    world, dropped = restrict(world, mirrors)
    print(f"corpus {len(world.candidates)} candidates"
          + (f"; {len(world.excluded_mirror)} excluded as D40 mirrors at build"
             if world.excluded_mirror else "")
          + (f"; {len(world.excluded_uncached)} with no cache entry (D71)"
             if world.excluded_uncached else "")
          + (f"; {len(dropped)} dropped by --exclude" if dropped else ""))
    # The identity control's expectation, and the reason it is a gate rather
    # than a report: a candidate whose x1.0 endpoint does not reproduce its
    # cached prediction is a candidate reading someone else's cache entry. D40
    # is the one cause known to do that, so the only deviations allowed are
    # mirror members still in the corpus. Anything else is a NEW defect and the
    # magnitude comparison below would silently carry it.
    declared = mirror_members(args.expect_mirrors) if args.expect_mirrors else set()
    expected_deviants = (
        (mirrors | declared | set(world.excluded_mirror)) & set(world.candidates)
    )
    if args.degrade:
        chosen = [d for d in world.pool if d.id in set(args.degrade)]
        missing = set(args.degrade) - {d.id for d in chosen}
        if missing:
            raise SystemExit(f"unknown pool id(s): {sorted(missing)}")
    elif args.kinds.strip().lower() == "all":
        chosen = list(world.pool)
    else:
        wanted = {k.strip() for k in args.kinds.split(",") if k.strip()}
        unknown = wanted - {k.value for k in UncertainKind}
        if unknown:
            raise SystemExit(f"unknown kind(s): {sorted(unknown)}")
        chosen = [d for d in world.pool if d.item.kind.value in wanted]
    if not chosen:
        raise SystemExit(f"--kinds {args.kinds!r} selected nothing from the pool")
    print(f"sweeping {len(chosen)}: {[d.id for d in chosen]}")

    penalty = _default_penalty([
        pareto.objective_value(
            DeploymentPlan(
                plan_id="x", model=world.spec.model,
                candidate=world.candidates[cid], predicted=world.truth[cid],
            ),
            world.spec.objective.primary,
        )
        for cid in world.truth if cid in world.candidates
    ])

    # Both sides sweep from TRUTH, and this is the whole correctness of the
    # experiment. An earlier run swept the closed form from a DEGRADED state
    # while `resimulate` rebuilt its endpoints from the truth cluster and
    # profiles - so the two figures differed by their baseline as well as by
    # their rule, and the "disagreement" it reported was partly the harness.
    # From truth the two are exactly comparable: same starting metrics, same
    # scoring, one item moved to the same endpoint by two different means.
    metrics, policy = world.truth, world.policy_truth
    best = _best(world, metrics, policy)
    if best is None:
        raise SystemExit("no feasible plan in the truth state; nothing to rank")
    output = PlannerOutput(
        feasible=True, service_model=world.spec.model,
        cluster_id=world.cluster.cluster_id, recommended=best,
    )
    # Re-based to the truth nominal, for the same reason: an item's `nominal` is
    # what the planner currently believes, and here that is the measured value.
    items = [
        d.item.model_copy(update={"nominal": d.truth_value}) for d in chosen
    ]
    registry = UncertainInputRegistry(items=items)

    started = time.monotonic()
    closed = analyze(
        output, registry, metrics, policy, world.spec, world.context,
        world.island_hw, grid=args.grid, slo_penalty=penalty,
    )
    closed_seconds = time.monotonic() - started
    print(f"closed form: {closed_seconds:.2f}s")
    for s in closed:
        print(f"  {s.input_id:42s} dR={s.delta_regret} approx={s.approximation}")

    base = partial(
        resim.resimulate, baseline=metrics, spec=world.spec, cluster=world.cluster,
        islands=world.islands, profiles=world.profiles,
        candidates=list(world.candidates.values()), predictor=world.predictor,
        island_hw=world.island_hw, max_workers=args.workers,
    )
    #: The `lo` endpoint of a PROFILE item is the multiplier 1.0 - the MEASURED
    #: bundle - so its resimulation must reproduce the cached truth prediction.
    #: It is the control for the whole bundle-copy path, and it is checked
    #: rather than assumed: a copy that quietly perturbed something would make
    #: every disagreement below unattributable.
    controls: list[dict] = []
    coverages: list[dict] = []
    corpus = list(world.candidates.values())

    def runner(item):
        result = base(item)
        touched_ids = {c.id for c in resim._touched(item, corpus)}
        for endpoint in (result.lo, result.hi):
            coverages.append(
                {"input_id": item.id,
                 **endpoint_coverage(endpoint, world.truth, touched_ids)}
            )
        for endpoint in (result.lo, result.hi):
            if abs(endpoint.value - 1.0) > 1e-12:
                continue
            worst, where = 0.0, ""
            deviants: dict[str, float] = {}
            for cid, m in endpoint.metrics.items():
                truth = world.truth.get(cid)
                if truth is None:
                    continue
                for field in ("p99_ttft_ms", "p99_tpot_ms", "total_energy_j"):
                    a, b = getattr(truth, field), getattr(m, field)
                    if a in (None, 0) or b is None:
                        continue
                    rel = abs(b - a) / abs(a)
                    if rel > worst:
                        worst, where = rel, f"{cid}.{field}"
                    # §2.5 compares the LIST of candidates that fail this
                    # control against C1's mirror list, so the worst one alone
                    # is not enough: a second, smaller deviation somewhere else
                    # is exactly the "cause other than D40" the gate looks for.
                    if rel > IDENTITY_TOL:
                        deviants[cid] = max(deviants.get(cid, 0.0), rel)
            controls.append({
                "input_id": item.id, "value": endpoint.value,
                "candidates": len(endpoint.metrics),
                "worst_relative_deviation_from_cache": worst, "worst_at": where,
                "deviating_candidates": sorted(deviants),
                "deviations": {k: deviants[k] for k in sorted(deviants)},
            })
        return result
    started = time.monotonic()
    refined, records = refine(
        closed, registry, runner, metrics, policy, world.spec, world.context,
        world.island_hw, best.plan.candidate.id,
        top=args.top, slo_penalty=penalty,
    )
    resim_seconds = time.monotonic() - started
    print(f"resimulated: {resim_seconds:.1f}s")

    order_closed = {s.input_id: i for i, s in enumerate(closed)}
    order_refined = {s.input_id: i for i, s in enumerate(refined)}
    ids = sorted(order_closed)
    checked = [r for r in records if not r.skipped]

    # §2.5: a rank correlation over fewer than three ACTIVE items is not
    # evidence, and PR #86 committed two that were not. The rule is applied to
    # both correlations and the reason is recorded beside the null, so a reader
    # cannot mistake "not computed" for "came out zero".
    active = active_records(records)
    spearman_note = spearman_refusal(len(active), len(checked))
    if spearman_note is not None:
        rho_rank = rho_value = None
    else:
        rho_rank = spearman(
            [float(order_closed[i]) for i in ids],
            [float(order_refined[i]) for i in ids],
        )
        rho_value = spearman(
            [r.closed_form for r in checked], [r.resimulated for r in checked]
        ) if len(checked) >= 2 else None

    # The x1.0 identity control, as a gate (§2.5).
    identity_verdict = identity_gate(controls, expected_deviants)
    unexplained = identity_verdict["unexplained"]
    coverage_verdict = coverage_gate(coverages)

    # The second identity control: LINK_BW and LINK_LAT re-price a term the
    # planner adds itself, so their closed form is exact and the resimulation
    # must land on it. A disagreement here is a bug in the rule, not a cost of
    # approximation, and it is the one place this experiment can say so.
    link_checks = []
    for r in checked:
        if r.kind not in ("link_bw", "link_lat"):
            continue
        scale = max(abs(r.closed_form), abs(r.resimulated), 1.0)
        rel = abs(r.closed_form - r.resimulated) / scale
        link_checks.append({
            "input_id": r.input_id, "kind": r.kind,
            "closed_form": r.closed_form, "resimulated": r.resimulated,
            "relative_difference": rel, "agrees": rel <= LINK_EXACT_RTOL,
        })

    payload = {
        "top": args.top,
        "grid": args.grid,
        "degraded": [d.id for d in chosen],
        "slo_penalty": penalty,
        "incumbent": best.plan.candidate.id,
        "closed_form_seconds": closed_seconds,
        "resimulate_seconds": resim_seconds,
        "speedup": (resim_seconds / closed_seconds) if closed_seconds else None,
        "spearman_rank": rho_rank,
        "spearman_delta_regret_on_checked": rho_value,
        "spearman_not_computed_because": spearman_note,
        "active_checked_items": [r.input_id for r in active],
        "identity_control_verdict": identity_verdict,
        "resimulation_coverage": coverage_verdict,
        "resimulation_coverage_detail": coverages,
        "link_exactness_checks": link_checks,
        "corpus_size": len(world.candidates),
        "excluded_mirror_at_build": world.excluded_mirror,
        "excluded_uncached": world.excluded_uncached,
        "dropped_by_exclude": dropped,
        "include_mirrors": args.include_mirrors,
        "fixture_path": str(args.fixture) if args.fixture else None,
        "order_closed": [s.input_id for s in closed],
        "order_refined": [s.input_id for s in refined],
        "refinements": [r.model_dump() for r in records],
        "identity_controls": controls,
        "closed": [s.model_dump(exclude={"grid"}) for s in closed],
        "refined": [s.model_dump(exclude={"grid"}) for s in refined],
        "provenance": prov.collect(random_seed=0),
    }
    out = args.out_json or Path(
        "outputs/uncertainty/eb3/eb3_closed_form_vs_resim.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")
    print(f"  spearman(rank) = {rho_rank}"
          + (f"  [{spearman_note}]" if spearman_note else ""))
    for c in controls:
        print(f"  control {c['input_id']} at x1.0: worst deviation from cache "
              f"{c['worst_relative_deviation_from_cache']:.3e} ({c['worst_at']}); "
              f"{len(c['deviating_candidates'])} candidate(s) over {IDENTITY_TOL:g}")
    for c in link_checks:
        verdict = "agrees" if c["agrees"] else "DISAGREES"
        print(f"  link exactness {c['input_id']}: closed={c['closed_form']:,.4g} "
              f"resim={c['resimulated']:,.4g} rel={c['relative_difference']:.3e} "
              f"-> {verdict}")
    for r in records:
        if r.skipped:
            print(f"  SKIPPED {r.input_id}: {r.skipped[:90]}")
        else:
            print(f"  {r.input_id}: closed={r.closed_form:,.4g} "
                  f"resim={r.resimulated:,.4g} in {r.seconds:.0f}s "
                  f"({r.simulated} runs)")
    if world.predictor is not None:
        world.predictor.close()
    # §2.5: the x1.0 control is a GATE. The artifact is written first, because
    # the deviating list IS the finding, but the run does not report success --
    # a candidate that fails this control is reading another candidate's cache
    # entry, and every magnitude below it would carry that.
    if not coverage_verdict["passed"]:
        n = coverage_verdict["endpoints_with_no_successful_run"]
        print(f"\nSTOP: {n} endpoint(s) produced no successful simulation at "
              f"all, so their dR is the baseline compared with itself. Nothing "
              f"in {out} may be quoted. Check a sim log under the work dir; the "
              f"first time this fired it was a missing ASTRA-Sim input template.")
        for c in coverages:
            if c["touched"] and not c["resimulated"]:
                print(f"  {c['input_id']} at {c['value']}: 0 of {c['touched']}")
        return 4
    if unexplained:
        print(f"\nSTOP: {len(unexplained)} candidate(s) fail the x1.0 identity "
              f"control and are NOT D40 mirrors. This is a cause other than "
              f"D40; the magnitude comparison in {out} must not be quoted "
              f"until it is explained.")
        for cid in unexplained[:10]:
            print(f"  {cid}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
