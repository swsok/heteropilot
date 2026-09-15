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

from experiments.uncertainty.eb1_regret_vs_budget import (
    Degraded,
    World,
    _best,
    _default_penalty,
    _state,
    build_world,
)
from planner.optimizer import pareto
from planner.plan import DeploymentPlan, PlannerOutput
from planner.uncertainty.perturb import JudgedMetrics, perturb
from planner.uncertainty.registry import UncertainInputRegistry, UncertainKind
from planner.uncertainty.sensitivity import GridPoint, analyze
from planner.util import provenance as prov

GRIDS = (3, 5, 9)
#: Two objective values are a tie below this. `collapse_equivalent` rounds a
#: plan's outcome to 6 decimals before comparing, so anything finer than that is
#: not a distinction the planner itself makes.
TIE_TOL = 1e-6


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
    #: §2.4's category, on the rows where the prediction and the single-truth
    #: outcome disagree. None where they agree - there is nothing to explain.
    category: str | None = None
    #: §2.4's second criterion: does the flip exist ANYWHERE in the range?
    #: None unless --truth-sweep asked for it.
    realisable: bool | None = None


def classify_false_positive(
    grid: list[GridPoint],
    incumbent: str,
    nominal: float,
    truth: float,
    equivalent_to_incumbent: set[str],
    tol: float = TIE_TOL,
) -> str:
    """Why a predicted flip did not materialise: `alpha`, `beta` or `gamma`.

    The three categories are §2.4's, and the point of separating them is that
    only one of them is the detector being wrong.

    **gamma - equivalence or tie.** Every point where the sweep saw the argmax
    change went to a candidate that is `equivalent_candidates`-identical to the
    incumbent, or that wins nothing over it. The id moved and the outcome did
    not; `collapse_equivalent` dissolves it.

    **alpha - structural.** A real crossing exists inside the item's range -- a
    point where a strictly better, genuinely different candidate takes over --
    but it does not lie between the degraded nominal and the truth, so restoring
    to truth never crosses it. The plan's claim was "measuring this COULD change
    the decision", and that claim is true; scoring it against one truth value
    asks a different question. This is a property of where the fixture's truth
    happens to sit, not of the rule.

    **delta - the two paths are not modelling the same thing.** The sweep places
    a strict crossing AT the truth value itself, and restoring to that same value
    produces no flip. Two computations of the same point cannot disagree unless
    they are computing different functions, so this says nothing about where the
    crossing is; it says the sweep and the restore disagree about what the input
    MEANS. Not one of §2.4's three, and added because on E-A1 it is all of them:
    see `eb2_f2.md`.

    **beta - approximation.** Everything else, and in particular a crossing the
    sweep placed strictly between nominal and truth that restoring to truth did
    not reproduce. That is the closed-form rule putting the crossing in the wrong
    place, which is the only category that counts against the rule.

    A grid with no argmax change at all is also `beta`: the sweep reported a flip
    that its own grid does not show, which can only be the rule.
    """
    changing = [g for g in grid if g.best_plan_id and g.best_plan_id != incumbent]
    if not changing:
        return "beta"
    strict = [
        g for g in changing
        if g.best_plan_id not in equivalent_to_incumbent
        and g.best_value - g.current_value > tol
    ]
    if not strict:
        return "gamma"
    values = [g.value for g in grid]
    span = max(values) - min(values) if values else 0.0
    at_truth = max(tol, span * 1e-9)
    if any(abs(g.value - truth) <= at_truth for g in strict):
        return "delta"
    lo, hi = min(nominal, truth), max(nominal, truth)
    if any(lo - tol <= g.value <= hi + tol for g in strict):
        return "beta"
    return "alpha"


def equivalents_of(world: World, incumbent: str | None) -> set[str]:
    """Candidate ids whose TRUTH outcome is identical to the incumbent's.

    Uses the planner's own `metric_signature`, so "equivalent" here means what
    `equivalent_candidates` means in a `PlannerOutput` and not merely "same
    objective value".
    """
    if incumbent is None or incumbent not in world.truth:
        return set()
    def sig(cid: str):
        return pareto.metric_signature(DeploymentPlan(
            plan_id="x", model=world.spec.model,
            candidate=world.candidates[cid], predicted=world.truth[cid],
        ))
    target = sig(incumbent)
    return {cid for cid in world.truth
            if cid in world.candidates and cid != incumbent and sig(cid) == target}


def realisable_flip(
    world: World, degraded: list[Degraded], item: Degraded, grid: int
) -> bool:
    """Does restoring this input to ANY value in its range move the winner?

    §2.4's second scoring criterion. The first asks whether the flip happens at
    the one value truth turned out to have, which makes a wide-ranged input
    structurally a false positive; this asks whether the flip is there to be
    found at all, which is what the detector actually claims.

    Deliberately NOT read off the sweep: this walks the same restore path that
    produces `actual` -- `_state` then `_best` then `judge` under the margin
    policy -- while `analyze` reaches its verdict through `_plans_at`. Two
    implementations agreeing is a control; reading the answer off the thing
    being scored would be none.
    """
    incumbent = _winner(world, degraded, set())
    lo, hi = item.item.range.lo, item.item.range.hi
    if lo is None or hi is None:
        return False
    others = [d for d in degraded if d.id != item.id]
    for i in range(grid):
        value = lo + (hi - lo) * i / (grid - 1) if grid > 1 else lo
        metrics: JudgedMetrics = world.truth
        policy = world.policy_truth
        for d in others:
            if d.item.kind is UncertainKind.SIM_ERROR:
                policy = world.policy_scalar
                continue
            rebased = d.item.model_copy(update={"nominal": d.truth_value})
            metrics = JudgedMetrics.trusted(
                perturb(rebased, d.degraded_value, metrics, world.context).metrics
            )
        if item.item.kind is UncertainKind.SIM_ERROR:
            # The domain is not a metric perturbation: it is scalar mode or it is
            # not, so the only two points its "range" has are the two policies.
            policy = world.policy_scalar if value > (lo + hi) / 2 else world.policy_truth
        else:
            rebased = item.item.model_copy(update={"nominal": item.truth_value})
            metrics = JudgedMetrics.trusted(
                perturb(rebased, value, metrics, world.context).metrics
            )
        best = _best(world, metrics, policy)
        if (best.plan.candidate.id if best else None) != incumbent:
            return True
    return False


def _winner(world: World, degraded: list[Degraded], restored: set[str]) -> str | None:
    metrics, policy = _state(world, degraded, restored)
    best = _best(world, metrics, policy)
    return best.plan.candidate.id if best else None


def cases_for(
    world: World, degraded: list[Degraded], grid: int, penalty: float,
    *, truth_sweep: bool = False,
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
    equivalents = equivalents_of(world, incumbent)
    out: list[Case] = []
    for d in degraded:
        s = by_id.get(d.id)
        if s is None:
            continue
        after = _winner(world, degraded, {d.id})
        actual = after != incumbent
        category = None
        if s.flip and not actual and incumbent is not None:
            category = classify_false_positive(
                s.grid, incumbent, d.degraded_value, d.truth_value, equivalents,
            )
        out.append(Case(
            grid=grid, k=len(degraded), degraded=[x.id for x in degraded],
            input_id=d.id, kind=d.item.kind.value, predicted=s.flip,
            actual=actual, approximation=s.approximation,
            delta_regret=s.delta_regret, incumbent=incumbent,
            restored_winner=after, category=category,
            realisable=(realisable_flip(world, degraded, d, grid)
                        if truth_sweep else None),
        ))
    return out


def score(cases: list[Case], *, against: str = "actual") -> dict:
    """Precision and recall against one of the two criteria of §2.4.

    `against="actual"` is the REALISED transition: did restoring this input to
    the value truth turned out to have move the recommendation? That number is
    as much a statement about where this fixture's truth sits as about the rule.

    `against="realisable"` is the POSSIBLE transition: is the flip anywhere in
    the range? That is what the measurement plan claims, so it is the detector's
    own accuracy. Cases where it was not computed are skipped rather than
    counted as negatives.
    """
    if against == "realisable":
        cases = [c for c in cases if c.realisable is not None]
    def truth(c: Case) -> bool:
        return c.actual if against == "actual" else bool(c.realisable)
    tp = sum(1 for c in cases if c.predicted and truth(c))
    fp = sum(1 for c in cases if c.predicted and not truth(c))
    fn = sum(1 for c in cases if not c.predicted and truth(c))
    tn = sum(1 for c in cases if not c.predicted and not truth(c))
    return {
        "n": len(cases), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def categories(cases: list[Case]) -> dict[str, int]:
    """How the false positives break down, §2.4."""
    out: dict[str, int] = {}
    for c in cases:
        if c.category:
            out[c.category] = out.get(c.category, 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--fixture", type=Path, default=None,
        help="Fixture json. Defaults to E-B1's (E-A1's corpus), so the "
             "committed result reproduces.",
    )
    parser.add_argument(
        "--truth-sweep", action="store_true",
        help="Also score against the POSSIBLE transition (§2.4): restore each "
             "input to every grid point of its range, not only to truth.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("outputs/uncertainty/eb1/work"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--slo-penalty", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    kwargs = {"fixture_path": args.fixture} if args.fixture else {}
    world = build_world(args.cache_dir, args.work_dir, args.workers, **kwargs)
    if world.excluded_mirror or world.excluded_uncached:
        print(f"corpus: {len(world.candidates)} candidates; "
              f"{len(world.excluded_mirror)} excluded as D40 mirrors, "
              f"{len(world.excluded_uncached)} with no cache entry (D71)")
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
                cases.extend(cases_for(world, list(combo), grid, penalty,
                                       truth_sweep=args.truth_sweep))
        print(f"  m={grid} done ({len(cases)} cases so far)")

    payload = {
        "grids": list(GRIDS),
        "slo_penalty": penalty,
        "fixture_path": str(args.fixture) if args.fixture else None,
        "excluded_mirror": world.excluded_mirror,
        "excluded_uncached": world.excluded_uncached,
        "corpus_size": len(world.candidates),
        # §2.4: the same cases scored both ways. `realised` is the criterion the
        # committed E-B2 result used; `realisable` is the detector's own.
        "realisable": {
            str(g): score([c for c in cases if c.grid == g], against="realisable")
            for g in GRIDS
        } if args.truth_sweep else None,
        "categories": {
            str(g): categories([c for c in cases if c.grid == g]) for g in GRIDS
        },
        "categories_by_kind": {
            str(g): {
                kind: categories([c for c in cases
                                  if c.grid == g and c.kind == kind])
                for kind in sorted({c.kind for c in cases})
            }
            for g in GRIDS
        },
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
    out = args.out_json or Path("outputs/uncertainty/eb2/eb2_flip_detection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")
    for g in GRIDS:
        print(f"  m={g}: {payload['overall'][str(g)]}")
        print(f"        categories {payload['categories'][str(g)]}")
        if args.truth_sweep:
            print(f"        realisable {payload['realisable'][str(g)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
