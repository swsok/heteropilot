"""E-B2: does `flip=True` mean the recommendation really changes? (STEP B4).

B2's sweep reports a flip when, somewhere inside an uncertain input's range,
the best plan stops being the incumbent. A measurement plan that says "this one
flips the decision" is making a falsifiable claim, and this is the experiment
that falsifies it: degrade a set of inputs, ask the planner which of them flip,
then restore each one to truth and see whether the recommendation actually
moved.

    predicted flip : `Sensitivity.flip` from `analyze` on the degraded state
    actual flip    : `argmax` changes when THAT input alone is restored

The two are not the same statement, and the asymmetry is the point. The sweep
walks the whole range, so it flips if ANY point in the range would change the
decision; the restore tests one point, the truth value. So a predicted flip that
does not materialise is not automatically a false alarm - it can be a flip that
lives at a part of the range truth does not sit at. Precision is therefore a
LOWER bound on the rule's correctness, and this file says so wherever it quotes
one, rather than reporting the friendlier number.

The grid size is the variable: m = 3, 5, 9 (§2.5's grid is the m = 5 shape).
A coarser grid can miss a flip that lives between its points; a finer one costs
proportionally more sweep. Items whose perturbation rule is a first-order
stand-in (`approximation=True`: PROFILE, POWER) are counted separately, because
"the approximate rules cry wolf" is a different defect from "the sweep is too
coarse" and the work order asks for their false-positive rate on its own.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb2_flip_detection.py \\
        --cache-dir outputs/uncertainty/ea1/cache
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from eb1_regret_vs_budget import (
    Degraded,
    World,
    _best,
    _default_penalty,
    _state,
    build_world,
)

from planner.optimizer import pareto
from planner.plan import DeploymentPlan, PlannerOutput
from planner.uncertainty.registry import UncertainInputRegistry
from planner.uncertainty.sensitivity import analyze
from planner.util import provenance as prov

GRIDS = (3, 5, 9)


@dataclass
class Case:
    """One (degradation set, input) pair: what was claimed, what happened."""

    grid: int
    k: int
    degraded: list[str]
    input_id: str
    kind: str
    predicted: bool
    actual: bool
    approximation: bool
    delta_regret: float | None
    incumbent: str | None
    restored_winner: str | None


def _winner(world: World, degraded: list[Degraded], restored: set[str]) -> str | None:
    metrics, policy = _state(world, degraded, restored)
    best = _best(world, metrics, policy)
    return best.plan.candidate.id if best else None


def cases_for(
    world: World, degraded: list[Degraded], grid: int, penalty: float
) -> list[Case]:
    metrics, policy = _state(world, degraded, set())
    best = _best(world, metrics, policy)
    incumbent = best.plan.candidate.id if best else None
    output = PlannerOutput(
        feasible=best is not None, service_model=world.spec.model,
        cluster_id=world.cluster.cluster_id, recommended=best,
    )
    registry = UncertainInputRegistry(items=[d.item for d in degraded])
    ranked = analyze(
        output, registry, metrics, policy, world.spec, world.context,
        world.island_hw, grid=grid, slo_penalty=penalty,
    )
    by_id = {s.input_id: s for s in ranked}
    out: list[Case] = []
    for d in degraded:
        s = by_id.get(d.id)
        if s is None:
            continue
        after = _winner(world, degraded, {d.id})
        out.append(Case(
            grid=grid, k=len(degraded), degraded=[x.id for x in degraded],
            input_id=d.id, kind=d.item.kind.value, predicted=s.flip,
            actual=after != incumbent, approximation=s.approximation,
            delta_regret=s.delta_regret, incumbent=incumbent,
            restored_winner=after,
        ))
    return out


def score(cases: list[Case]) -> dict:
    tp = sum(1 for c in cases if c.predicted and c.actual)
    fp = sum(1 for c in cases if c.predicted and not c.actual)
    fn = sum(1 for c in cases if not c.predicted and c.actual)
    tn = sum(1 for c in cases if not c.predicted and not c.actual)
    return {
        "n": len(cases), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("outputs/uncertainty/eb1/work"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--slo-penalty", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    world = build_world(args.cache_dir, args.work_dir, args.workers)
    penalty = args.slo_penalty
    if penalty is None:
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
    print(f"pool {len(world.pool)}  penalty {penalty:,.1f}")

    cases: list[Case] = []
    for grid in GRIDS:
        for k in args.k:
            for combo in itertools.combinations(world.pool, k):
                cases.extend(cases_for(world, list(combo), grid, penalty))
        print(f"  m={grid} done ({len(cases)} cases so far)")

    payload = {
        "grids": list(GRIDS),
        "slo_penalty": penalty,
        "overall": {
            str(g): score([c for c in cases if c.grid == g]) for g in GRIDS
        },
        "approximate_rules": {
            str(g): score([c for c in cases if c.grid == g and c.approximation])
            for g in GRIDS
        },
        "exact_rules": {
            str(g): score([c for c in cases if c.grid == g and not c.approximation])
            for g in GRIDS
        },
        "by_kind": {
            str(g): {
                kind: score([
                    c for c in cases if c.grid == g and c.kind == kind
                ])
                for kind in sorted({c.kind for c in cases})
            }
            for g in GRIDS
        },
        "cases": [vars(c) for c in cases],
        "provenance": prov.collect(random_seed=0),
    }
    out = Path("outputs/uncertainty/eb2/eb2_flip_detection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")
    for g in GRIDS:
        print(f"  m={g}: {payload['overall'][str(g)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
