"""E-B1 / E-B2: truth degradation (uncertainty work order STEP B4).

The quantitative basis for the patent's §6 effect claim. Start from a fully
measured fixture, DEGRADE some of its inputs to a lower grade, then let each
measurement strategy buy the truth back one input at a time and watch the regret
fall. The strategy that reaches zero regret for the fewest server-hours wins.

Nothing here simulates. The truth metrics come from the E-A1 envelope cache, and
every degraded or restored state is B1's closed-form perturbation of them - which
is the whole reason a sweep of `strategies x seeds x k x steps` is minutes
instead of weeks.

**Where the ranges come from, and why that is not cheating.** In a real run an
uncertain input's range comes from `grades.yaml`, and for a `placeholder` link
that is `unbounded` - no regret is computable, which is the honest answer and
exactly what B3 reports. A truth-degradation experiment is different: it KNOWS
both endpoints, because it chose the degraded value itself. So the range handed
to the analysis is `[min(truth, degraded), max(truth, degraded)]`, supplied by
the experiment rather than discovered by the planner. That is what makes every
strategy - including the ones this one is compared against - rankable at all.

**What this fixture can and cannot exercise.** Three facts about
`pd-rngd-gpu-card` under `minimize_energy` shape every number below, and all
three were found by running the experiment, not assumed:

* the E-A1 cache was generated with ``enable_pd=False``, so the corpus holds no
  P/D candidate. `LINK_BW`'s rule re-prices a KV transfer between a prefill and
  a decode engine, so with no such candidate all six fabric links are
  **structurally inert**: `perturb` reports `affected=0` for each. They are kept
  in the pool anyway, because they are genuine uncertain inputs of this cluster
  and a measurement strategy has to decide whether to spend on them - which is
  exactly where `widest` goes wrong;
* `PROFILE` scales latency and throughput but never energy (§2.7 gives energy to
  `POWER`), so with energy as the ranked objective a profile degradation can only
  move the recommendation by pushing a candidate across an SLO boundary. One
  does: degrading `cuda-a40-node_a40a` hands the plan to its twin `a40b` - a
  genuine flip worth exactly zero regret, because the two nodes are identical;
* that leaves `SIM_ERROR` - the accuracy domain - carrying essentially all the
  decision regret in this fixture.

That is a result, not a defect, and it is reported as one. It does mean most
degradation sets move nothing at all, so every aggregate below is given twice:
over all sets, and over only those sets where the recommendation actually
changed (`initial_regret > 0`).

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb1_regret_vs_budget.py \\
        --cache-dir outputs/uncertainty/ea1/cache \\
        --out-dir experiments/uncertainty/results
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from planner.candidate_generator import CandidateGenerator
from planner.envelope import EnvelopeCache
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive, pareto
from planner.optimizer.exhaustive import Verdict, judge, rank_plans
from planner.optimizer.margin import AccuracyDomainMargin, MarginPolicy
from planner.plan import DeploymentPlan, PlannerOutput, ScoredPlan
from planner.predictor.accuracy_domain import accuracy_domain_key
from planner.predictor.calibration import CalibrationModel, load_calibration
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.uncertainty.grades import load_costs
from planner.uncertainty.perturb import JudgedMetrics, PerturbContext, judged_metrics, perturb
from planner.uncertainty.registry import (
    Grade,
    MeasurementCost,
    Range,
    UncertainInput,
    UncertainInputRegistry,
    UncertainKind,
)
from planner.uncertainty.sensitivity import _default_penalty, analyze
from planner.util import provenance as prov
from planner.util.workload import generate_trace

FIXTURE = Path("outputs/pd_slo_sweep_margin18/pd-rngd-gpu-card.json")
#: What a degraded link is set to: the placeholder 35 GB/s the fixtures carried
#: before D18 remeasured the fabric at ~13.
DEGRADED_LINK_GBPS = 35.0
#: A Tier 0 profile is ~39 % slower per operator than the measured bundle
#: (profiles/uncertainty/grades.yaml, profile/analytical).
DEGRADED_PROFILE = 1.38875811006420014
#: Truth: Stage A's accuracy domain (work order STEP B4, "calibration = Stage A의
#: 도메인 yaml"). RNGD's entry carries the four measured operating points.
DOMAIN_CALIBRATION = (
    Path("profiles/calibration/a40.yaml"),
    Path("profiles/calibration/rngd_card_edf.domain.yaml"),
)
#: The degraded twin: the SAME fit before E-A2 added operating points, i.e. one
#: error for the whole bucket. This is the work order's third degradation,
#: "도메인을 스칼라 모드로", done at file granularity rather than by editing the
#: model - `rngd_card_edf.yaml` IS what the domain file was built from
#: (`provenance.accuracy_domain.base`), so the contrast is a real pair of
#: artifacts and not a synthetic one.
SCALAR_CALIBRATION = (
    Path("profiles/calibration/a40.yaml"),
    Path("profiles/calibration/rngd_card_edf.yaml"),
)
#: The worst error the domain measured (L=107.2, tpot 0.4725). The SIM_ERROR
#: item's range runs from 0 to this: "the error here is somewhere between none
#: and the worst we have ever seen on this hardware".
WORST_MEASURED_ERROR = 0.47252811693773233
STRATEGIES = ("ours", "random", "round_robin", "widest", "oracle")


def _calibration(paths: tuple[Path, ...]) -> CalibrationModel:
    model = CalibrationModel.identity()
    for path in paths:
        model.hardware.update(load_calibration(path).hardware)
    return model


@dataclass
class Degraded:
    """One input moved off its measured value, and the way back."""

    item: UncertainInput
    truth_value: float
    degraded_value: float

    @property
    def id(self) -> str:
        return self.item.id


@dataclass
class World:
    spec: object
    cluster: object
    islands: dict
    candidates: dict
    context: PerturbContext
    truth: JudgedMetrics
    island_hw: dict
    costs: object
    #: Truth policy: the Stage A domain. Degrading SIM_ERROR swaps in `scalar`.
    policy_truth: MarginPolicy = None  # type: ignore[assignment]
    policy_scalar: MarginPolicy = None  # type: ignore[assignment]
    pool: list[Degraded] = field(default_factory=list)
    #: E-B3 needs these to re-run the simulator; E-B1 and E-B2 never touch them.
    profiles: dict = field(default_factory=dict)
    predictor: object | None = None


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

def build_world(
    cache_dir: Path, work_dir: Path, workers: int, *, keep_predictor: bool = False
) -> World:
    fixture = json.loads(FIXTURE.read_text())
    spec = load_service_spec(fixture["service"])
    cluster = load_cluster_spec(fixture["cluster"])
    profiles = load_profiles_for(cluster, Path("."))
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}

    work_dir.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(
        spec, work_dir / "workload.jsonl",
        num_requests=fixture["num_requests"], seed=fixture["seed"],
    )
    reduction = TopologyGraph(cluster).reduce_for_simulator(islands)
    cache = EnvelopeCache(
        cache_dir, spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=reduction.link_bw_gbps,
        trace_digest=prov.hash_file(trace.path),
    )
    predictor = LLMServingSimPredictor(trace, work_dir=work_dir / "sims")
    generation = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False
    ).generate()
    _tiers, island_hw, _w = exhaustive._profile_tiers(spec, islands, profiles)
    try:
        evaluation = exhaustive.evaluate_candidates(
            generation.candidates, spec, cluster, by_id, profiles, predictor,
            cache=cache, island_hw=island_hw, max_workers=workers,
        )
    finally:
        # E-B3 re-runs the simulator through this same predictor, so it asks for
        # the process pool to stay up; E-B1 and E-B2 never simulate and close it
        # here, which is also what makes their "zero cache misses" check load
        # bearing.
        if not keep_predictor:
            predictor.close()
    if cache.stats()["misses"]:
        raise SystemExit(
            f"{cache.stats()['misses']} cache miss(es): this experiment must not "
            f"simulate. Fill the cache with the E-A1 command first."
        )

    key = accuracy_domain_key(spec, spec.model, spec.service.dtype)
    candidates = {c.id: c for c in generation.candidates}
    world = World(
        spec=spec, cluster=cluster, islands=by_id, candidates=candidates,
        context=PerturbContext.build(spec, cluster, by_id, candidates),
        truth=judged_metrics(evaluation), island_hw=island_hw, costs=load_costs(),
        policy_truth=AccuracyDomainMargin(_calibration(DOMAIN_CALIBRATION), key),
        policy_scalar=AccuracyDomainMargin(_calibration(SCALAR_CALIBRATION), key),
        profiles=profiles,
        predictor=predictor if keep_predictor else None,
    )
    world.pool = _degradable(world)
    return world


def _degradable(world: World) -> list[Degraded]:
    """The inputs this experiment knows how to degrade and restore.

    The work order's three degradations, in its own order:

    * every measured `fabric-*` link to the pre-D18 placeholder 35 GB/s;
    * every island profile to the Tier 0 multiplier from `grades.yaml`;
    * the accuracy domain to scalar mode - one item, because the two calibration
      files are a pair and there is no half-scalar state between them.

    LINK_BW's rule is exact and the other two are first-order, which is what
    E-B2 needs in order to report the approximate rules' false-positive rate on
    its own.
    """
    out: list[Degraded] = []
    cost = world.costs
    for link in world.cluster.links:
        if link.source.value != "measured" or not link.id.startswith("fabric-"):
            continue
        row = cost.cost_for("link_bw")
        out.append(Degraded(
            item=UncertainInput(
                id=f"link_bw:{link.id}", kind=UncertainKind.LINK_BW,
                grade=Grade.PLACEHOLDER, nominal=DEGRADED_LINK_GBPS,
                range=Range(lo=min(link.bandwidth_gbps, DEGRADED_LINK_GBPS),
                            hi=max(link.bandwidth_gbps, DEGRADED_LINK_GBPS),
                            unit="gbps", source="E-B1 degradation endpoints"),
                cost=MeasurementCost(method=row.method, hours=row.hours,
                                     exclusive=row.exclusive, source=row.source)
                if row else None,
            ),
            truth_value=link.bandwidth_gbps, degraded_value=DEGRADED_LINK_GBPS,
        ))
    row = cost.cost_for("profile")
    for island_id in sorted(world.islands):
        out.append(Degraded(
            item=UncertainInput(
                id=f"profile:{island_id}", kind=UncertainKind.PROFILE,
                grade=Grade.ANALYTICAL, nominal=DEGRADED_PROFILE,
                range=Range(lo=1.0, hi=DEGRADED_PROFILE, unit="fraction",
                            source="E-B1 degradation endpoints"),
                affects=[island_id],
                cost=MeasurementCost(method=row.method, hours=row.hours,
                                     exclusive=row.exclusive, source=row.source)
                if row else None,
            ),
            truth_value=1.0, degraded_value=DEGRADED_PROFILE,
        ))
    row = cost.cost_for("sim_error")
    out.append(Degraded(
        item=UncertainInput(
            id="sim_error:domain", kind=UncertainKind.SIM_ERROR,
            grade=Grade.MEASURED, nominal=0.0,
            range=Range(lo=0.0, hi=WORST_MEASURED_ERROR, unit="fraction",
                        source=(
                            "E-B1: 0 to the worst operating point the E-A2 domain "
                            "measured (L=107.2, tpot=0.4725)"
                        )),
            cost=MeasurementCost(method=row.method, hours=row.hours,
                                 exclusive=row.exclusive, source=row.source)
            if row else None,
        ),
        truth_value=0.0, degraded_value=WORST_MEASURED_ERROR,
    ))
    return out


# ---------------------------------------------------------------------------
# States and scoring
# ---------------------------------------------------------------------------

def _state(
    world: World, degraded: list[Degraded], restored: set[str]
) -> tuple[JudgedMetrics, MarginPolicy]:
    """(metrics, policy) with every still-degraded input at its degraded value.

    A SIM_ERROR degradation moves the MARGIN, not the prediction (§2.7), so it
    swaps the accuracy domain for its scalar twin instead of perturbing metrics.
    """
    metrics: JudgedMetrics = world.truth
    policy = world.policy_truth
    for d in degraded:
        if d.id in restored:
            continue
        if d.item.kind is UncertainKind.SIM_ERROR:
            policy = world.policy_scalar
            continue
        # `perturb` moves an input from `item.nominal` to `value`, and the pool's
        # items carry the DEGRADED value as their nominal because that is what
        # the planner believes while degraded. Here the baseline is truth, so the
        # item is re-based before it is applied. Getting this wrong is silent: a
        # PROFILE item perturbed from its own nominal scales by exactly 1.0, and
        # the first run of this experiment reported "no degradation moves the
        # recommendation" for that reason alone.
        item = d.item.model_copy(update={"nominal": d.truth_value})
        metrics = JudgedMetrics.trusted(
            perturb(item, d.degraded_value, metrics, world.context).metrics
        )
    return metrics, policy


def _best(
    world: World, metrics: JudgedMetrics, policy: MarginPolicy | None = None
) -> ScoredPlan | None:
    feasible: list[DeploymentPlan] = []
    policy = policy or world.policy_truth
    for cid in sorted(metrics):
        candidate = world.candidates.get(cid)
        if candidate is None:
            continue
        decision = policy.decide(candidate, metrics[cid], world.island_hw)
        plan = DeploymentPlan(
            plan_id=f"hp-{cid}", model=world.spec.model, candidate=candidate,
            predicted=metrics[cid],
        )
        if judge(plan, world.spec, decision).verdict is Verdict.FEASIBLE:
            feasible.append(plan)
    return rank_plans(feasible, world.spec).best


def _truth_value(world: World, candidate_id: str | None, penalty: float) -> float:
    """What the chosen plan is really worth, scored on the truth metrics.

    An incumbent that turns out infeasible under truth is charged from the best
    truth-feasible value, the same convention B2 settled on - so regret stays
    non-negative and a plan that cannot be deployed never outscores one that can.
    """
    truth_best = _best(world, world.truth)
    if truth_best is None:
        return 0.0
    if candidate_id is None:
        return truth_best.value - penalty
    plan = DeploymentPlan(
        plan_id="x", model=world.spec.model,
        candidate=world.candidates[candidate_id],
        predicted=world.truth[candidate_id],
    )
    judgement = judge(plan, world.spec, world.policy_truth.decide(
        plan.candidate, plan.predicted, world.island_hw))
    if judgement.verdict is Verdict.FEASIBLE:
        return pareto.objective_value(plan, world.spec.objective.primary)
    if judgement.verdict is Verdict.UNMEASURED:
        # A full unit, never the report's overshoot. An UNMEASURED plan that
        # PASSED its measured metric carries a report whose `worst_overshoot` is
        # 0.0, and charging that would price an unverifiable recommendation at
        # zero regret - the exact reading `judge` forbids when it says an
        # undecidable candidate must not surface as `closest_plan`, "which is a
        # claim about how narrowly something missed". There is no narrowly here.
        return truth_best.value - penalty
    overshoot = judgement.report.worst_overshoot if judgement.report else 1.0
    return truth_best.value - penalty * overshoot


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _order(
    world: World, strategy: str, degraded: list[Degraded], restored: set[str],
    rng: random.Random, penalty: float, grid: int,
) -> list[str]:
    """The remaining inputs, best first under this strategy."""
    remaining = [d for d in degraded if d.id not in restored]
    if not remaining:
        return []
    if strategy == "random":
        ids = [d.id for d in remaining]
        rng.shuffle(ids)
        return ids
    if strategy == "round_robin":
        by_kind: dict[str, list[str]] = {}
        for d in remaining:
            by_kind.setdefault(d.item.kind.value, []).append(d.id)
        out: list[str] = []
        while any(by_kind.values()):
            for kind in sorted(by_kind):
                if by_kind[kind]:
                    out.append(by_kind[kind].pop(0))
        return out
    if strategy == "widest":
        return [
            d.id for d in sorted(
                remaining,
                key=lambda d: -((d.item.range.hi or 0) - (d.item.range.lo or 0)),
            )
        ]
    if strategy == "oracle":
        scored = []
        for d in remaining:
            trial = restored | {d.id}
            state_m, state_p = _state(world, degraded, trial)
            best = _best(world, state_m, state_p)
            scored.append((
                -_truth_value(world, best.plan.candidate.id if best else None, penalty),
                d.id,
            ))
        return [cid for _v, cid in sorted(scored)]
    # ours
    metrics, policy = _state(world, degraded, restored)
    best = _best(world, metrics, policy)
    output = PlannerOutput(
        feasible=best is not None, service_model=world.spec.model,
        cluster_id=world.cluster.cluster_id, recommended=best,
    )
    registry = UncertainInputRegistry(items=[d.item for d in remaining])
    ranked = analyze(
        output, registry, metrics, policy, world.spec, world.context,
        world.island_hw, grid=grid, slo_penalty=penalty,
    )
    return [s.input_id for s in ranked]


def run_curve(
    world: World, degraded: list[Degraded], strategy: str, seed: int,
    penalty: float, grid: int,
) -> list[tuple[float, float]]:
    """(hours spent, regret) after each purchase, starting from none."""
    rng = random.Random(seed)
    restored: set[str] = set()
    spent = 0.0
    curve: list[tuple[float, float]] = []
    by_id = {d.id: d for d in degraded}
    while True:
        state_m, state_p = _state(world, degraded, restored)
        best = _best(world, state_m, state_p)
        truth_best = _best(world, world.truth)
        regret = (truth_best.value if truth_best else 0.0) - _truth_value(
            world, best.plan.candidate.id if best else None, penalty
        )
        curve.append((spent, max(0.0, regret)))
        order = _order(world, strategy, degraded, restored, rng, penalty, grid)
        if not order:
            break
        nxt = order[0]
        restored.add(nxt)
        spent += by_id[nxt].item.cost.hours if by_id[nxt].item.cost else 0.0
    return curve


def _degradation_sets(world: World, k: int, args) -> list[list[Degraded]]:
    """Which k-subsets of the pool to degrade.

    The work order asks for 10 random seeds. `--exhaustive` enumerates every
    subset instead, which is strictly more informative here and costs nothing:
    the whole experiment is closed form, and C(11,3) = 165.
    """
    if k > len(world.pool):
        return []
    if args.exhaustive:
        return [list(c) for c in itertools.combinations(world.pool, k)]
    return [
        random.Random(1000 * k + seed).sample(world.pool, k)
        for seed in range(args.seeds)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path,
                        default=Path("outputs/uncertainty/eb1/work"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("experiments/uncertainty/results"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--slo-penalty", type=float, default=None)
    parser.add_argument(
        "--exhaustive", action="store_true",
        help="Enumerate EVERY degradation set of each size instead of sampling "
             "--seeds of them. The pool is small enough that C(11,3)=165 is "
             "cheap, and it removes the seed lottery: with one item carrying "
             "most of the regret, 10 random draws of size 1 hit it about once.",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    world = build_world(args.cache_dir, args.work_dir, args.workers)
    truth_best = _best(world, world.truth)
    if truth_best is None:
        raise SystemExit("the truth fixture has no feasible plan; nothing to degrade")
    penalty = args.slo_penalty
    if penalty is None:
        # The same default the planner uses (`sensitivity._default_penalty`):
        # the largest objective magnitude observed, so one unit of overshoot
        # costs a whole objective's worth. Computed over the truth corpus here
        # because that is this experiment's "grid".
        penalty = _default_penalty([
            pareto.objective_value(
                DeploymentPlan(
                    plan_id="x", model=world.spec.model,
                    candidate=world.candidates[cid], predicted=world.truth[cid],
                ),
                world.spec.objective.primary,
            )
            for cid in world.truth
            if cid in world.candidates
        ])

    print(f"truth winner: {truth_best.plan.candidate.id}  V={truth_best.value:,.1f}")
    print(f"degradable pool: {len(world.pool)}  penalty: {penalty:,.1f}")

    # §2.5's penalty is "결과를 지배하는 노브", so the work order asks for the
    # default and a tenth of it and for any ranking that flips between them.
    penalties = [("default", penalty), ("tenth", penalty / 10.0)]

    runs: list[dict] = []
    for k in args.k:
        for index, chosen in enumerate(_degradation_sets(world, k, args)):
            for label, value in penalties:
                for strategy in STRATEGIES:
                    curve = run_curve(world, chosen, strategy, index, value, args.grid)
                    runs.append({
                        "k": k, "seed": index, "penalty": label, "strategy": strategy,
                        "degraded": [d.id for d in chosen],
                        "curve": curve,
                        # Regret before any measurement: 0 means this degradation
                        # set never moved the recommendation, so every strategy
                        # scores 0 on it and it dilutes the mean. Reported both
                        # ways rather than filtered silently.
                        "initial_regret": curve[0][1] if curve else 0.0,
                    })
            print(f"  k={k} set={index} {[d.id for d in chosen]} done")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fixture": json.loads(FIXTURE.read_text()),
        "truth_winner": truth_best.plan.candidate.id,
        "truth_value": truth_best.value,
        "slo_penalty": penalty,
        "grid": args.grid,
        "degraded_link_gbps": DEGRADED_LINK_GBPS,
        "degraded_profile_multiplier": DEGRADED_PROFILE,
        "pool": [
            {"id": d.id, "kind": d.item.kind.value, "truth": d.truth_value,
             "degraded": d.degraded_value,
             "cost_hours": d.item.cost.hours if d.item.cost else None}
            for d in world.pool
        ],
        "penalties": dict(penalties),
        "sampling": "exhaustive" if args.exhaustive else f"random x{args.seeds}",
        "runs": runs,
        "provenance": prov.collect(random_seed=0),
    }
    out_json = Path("outputs/uncertainty/eb1/eb1_regret_vs_budget.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
