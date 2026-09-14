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

from eb1_regret_vs_budget import (
    _best,
    _default_penalty,
    build_world,
)

from planner.optimizer import pareto
from planner.plan import DeploymentPlan, PlannerOutput
from planner.uncertainty import resimulate as resim
from planner.uncertainty.registry import UncertainInputRegistry
from planner.uncertainty.sensitivity import analyze, refine
from planner.util import provenance as prov


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("outputs/uncertainty/eb3/work"))
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--degrade", nargs="+", default=None,
        help="Pool ids to degrade. Default: every PROFILE item plus the accuracy "
             "domain - the set whose rules are first-order, which is what the "
             "resimulation exists to check.",
    )
    args = parser.parse_args()

    world = build_world(args.cache_dir, args.work_dir, args.workers,
                        keep_predictor=True)
    if args.degrade:
        chosen = [d for d in world.pool if d.id in set(args.degrade)]
        missing = set(args.degrade) - {d.id for d in chosen}
        if missing:
            raise SystemExit(f"unknown pool id(s): {sorted(missing)}")
    else:
        chosen = [
            d for d in world.pool
            if d.item.kind.value in ("profile", "sim_error")
        ]
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

    def runner(item):
        result = base(item)
        for endpoint in (result.lo, result.hi):
            if abs(endpoint.value - 1.0) > 1e-12:
                continue
            worst, where = 0.0, ""
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
            controls.append({
                "input_id": item.id, "value": endpoint.value,
                "candidates": len(endpoint.metrics),
                "worst_relative_deviation_from_cache": worst, "worst_at": where,
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
    rho_rank = spearman(
        [float(order_closed[i]) for i in ids],
        [float(order_refined[i]) for i in ids],
    )
    checked = [r for r in records if not r.skipped]
    rho_value = spearman(
        [r.closed_form for r in checked], [r.resimulated for r in checked]
    ) if len(checked) >= 2 else None

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
        "order_closed": [s.input_id for s in closed],
        "order_refined": [s.input_id for s in refined],
        "refinements": [r.model_dump() for r in records],
        "identity_controls": controls,
        "closed": [s.model_dump(exclude={"grid"}) for s in closed],
        "refined": [s.model_dump(exclude={"grid"}) for s in refined],
        "provenance": prov.collect(random_seed=0),
    }
    out = Path("outputs/uncertainty/eb3/eb3_closed_form_vs_resim.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")
    print(f"  spearman(rank) = {rho_rank}")
    for c in controls:
        print(f"  control {c['input_id']} at x1.0: worst deviation from cache "
              f"{c['worst_relative_deviation_from_cache']:.3e} ({c['worst_at']})")
    for r in records:
        if r.skipped:
            print(f"  SKIPPED {r.input_id}: {r.skipped[:90]}")
        else:
            print(f"  {r.input_id}: closed={r.closed_form:,.4g} "
                  f"resim={r.resimulated:,.4g} in {r.seconds:.0f}s "
                  f"({r.simulated} runs)")
    if world.predictor is not None:
        world.predictor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
