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

from experiments.uncertainty.build_truth_cache import mirror_groups
from experiments.uncertainty.ea1_margin_modes import _IslandOperatingPoint
from planner.candidate_generator import CandidateGenerator
from planner.envelope import workload_bucket, workload_shape
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive, pareto
from planner.optimizer.exhaustive import Verdict, judge, rank_plans
from planner.optimizer.margin import AccuracyDomainMargin, MarginPolicy
from planner.plan import DeploymentPlan, PlannerOutput, ScoredPlan
from planner.predictor.calibration import (
    AccuracyPoint,
    load_accuracy_domains,
    load_calibrations,
)
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
from planner.util.tier import resolve_variant
from planner.util.workload import generate_trace

FIXTURE = Path("outputs/pd_slo_sweep_margin18/pd-rngd-gpu-card.json")
#: What a degraded link is set to: the placeholder 35 GB/s the fixtures carried
#: before D18 remeasured the fabric at ~13.
DEGRADED_LINK_GBPS = 35.0
#: A Tier 0 profile is ~39 % slower per operator than the measured bundle
#: (profiles/uncertainty/grades.yaml, profile/analytical).
DEGRADED_PROFILE = 1.38875811006420014
#: Truth: the committed accuracy domains (work order STEP B4, "calibration =
#: Stage A의 도메인 yaml"). On the reconciled tree that is `rngd_card_edf.yaml`
#: itself: D33 dropped `rngd_card_edf.domain.yaml` because it paired each real
#: run with a sim at the same REQUESTED concurrency -- its sim side ran at served
#: 71-189 against real 15-107, the D32 mis-pairing -- and the nine-point D32
#: domain supersedes it. So the truth here is a DIFFERENT and better-founded
#: domain than the one E-B1 first ran against, and its numbers are not
#: comparable with that run's.
DOMAIN_CALIBRATION = (
    Path("profiles/calibration/a40.yaml"),
    Path("profiles/calibration/a40.accuracy.yaml"),
    Path("profiles/calibration/rngd_card_edf.yaml"),
)
#: The degraded twin: each domain COLLAPSED TO ONE POINT at the concurrency it
#: was fitted at, so `widen_error_bars` has no slope to widen along and every
#: margin is that one value held flat. That is the work order's third
#: degradation, "도메인을 스칼라 모드로" -- one error for the whole workload --
#: and it is exactly the state `a40.accuracy.yaml`'s own header describes the A40
#: domain as having been in before it gained a second and third point.
#:
#: It used to be done at FILE granularity: `rngd_card_edf.yaml` was then the
#: pre-domain artifact that `rngd_card_edf.domain.yaml` was built from, so the
#: contrast was a real pair rather than a synthetic one. D33 merged the domain
#: into that file and dropped the other, so the pair no longer exists.
#:
#: Passing NO domains is not the same degradation and was tried first: the
#: scalar fallback is a fitted BUCKET error, and neither `a40.yaml` nor
#: `rngd_card_edf.yaml` carries one for this canonical bucket
#: (`in_lt1024-out_ge512-rps_lt20`), so all 324 candidates came back `unmeasured`
#: and the degraded search had nothing feasible at all. `analyze` then returns
#: dR None for every item -- "no recommendation to flip" -- and the ranking
#: degenerates to alphabetical order, which put the inert links ahead of the
#: domain and made `ours` look like the worst strategy. That was the harness, not
#: the invention.
#: The worst error the domain measured (L=107.2, tpot 0.4725). The SIM_ERROR
#: item's range runs from 0 to this: "the error here is somewhere between none
#: and the worst we have ever seen on this hardware".
WORST_MEASURED_ERROR = 0.47252811693773233
STRATEGIES = ("ours", "random", "round_robin", "widest", "oracle")


def _scalar_domains(domains: dict) -> dict:
    """Each domain reduced to the single point it was fitted at.

    One measured error for the whole workload, with no operating-point axis --
    which is what the domain replaced, and what `widen_error_bars` degenerates to
    when there is nothing to interpolate between.
    """
    out = {}
    for hw, d in domains.items():
        # `model_copy` does not validate, so the point is built as the model.
        point = AccuracyPoint(
            conc=d.fitted_at_concurrency,
            tpot_err_pct=d.tpot_error_at(d.fitted_at_concurrency),
            ttft_err_pct=d.ttft_error_at(d.fitted_at_concurrency),
            note=f"scalar mode: {hw}'s domain collapsed to its fitted point",
        )
        out[hw] = d.model_copy(update={"points": [point]})
    return out


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
    #: Candidates dropped before evaluation, and why. Empty on the E-A1 fixture,
    #: which enumerates no P/D candidate and therefore has neither problem.
    #: `mirror` is D40; `uncached` is D71 -- runs that raised in the simulator and
    #: so have no entry to read. Both are reported rather than silently absent:
    #: the second in particular removes exactly the tp1-decode P/D family, which
    #: is where a link degradation would have had the most to say.
    excluded_mirror: list[str] = field(default_factory=list)
    excluded_uncached: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

def build_world(
    cache_dir: Path, work_dir: Path, workers: int, *, keep_predictor: bool = False,
    fixture_path: Path = FIXTURE, exclude_mirrors: bool | None = None,
) -> World:
    """The corpus, its truth, and the pool of inputs that can be degraded.

    `exclude_mirrors` defaults to the fixture's `enable_pd`, which is what C1
    chose and what every committed result was built on. Passing `False` on a P/D
    fixture deliberately puts D40's mirrored placements back in the corpus, which
    E-B3 needs in order to measure what D40 contributes rather than assuming it
    (STEP C4). Nothing else should pass it.
    """
    fixture = json.loads(fixture_path.read_text())
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
    _tiers, island_hw, _w = exhaustive._profile_tiers(spec, islands, profiles)
    # The E-A1 truth cache predates operating-point caching: all 162 entries
    # carry `metrics` and no `operating_point`, so main's margin policy -- which
    # reads the operating point off the SimResult -- returned `unmeasured` for
    # all 324 candidates and the experiment had no feasible truth to degrade.
    # `_IslandOperatingPoint` is E-A1's own answer to that: it fills the missing
    # point from the cached per-island served concurrency. Imported rather than
    # copied so the two experiments cannot disagree about what the operating
    # point of a cached run is.
    cache = _IslandOperatingPoint(
        cache_dir, spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=reduction.link_bw_gbps,
        trace_digest=prov.hash_file(trace.path),
        island_hw=island_hw,
    )
    predictor = LLMServingSimPredictor(trace, work_dir=work_dir / "sims")
    # `enable_pd` comes from the fixture so the E-A1 json keeps its aggregated
    # corpus and reproduces byte-for-byte, while F2 turns P/D on (STEP C1).
    enable_pd = bool(fixture.get("enable_pd"))
    generation = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False,
        enable_pd=enable_pd,
    ).generate()
    drop_mirrors = enable_pd if exclude_mirrors is None else exclude_mirrors
    corpus, excluded_mirror, excluded_uncached = _corpus(
        generation.candidates, cache, exclude_mirrors=drop_mirrors)
    try:
        evaluation = exhaustive.evaluate_candidates(
            corpus, spec, cluster, by_id, profiles, predictor,
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

    # main's signature (D33 §1): the token-mix scope, the model and the precision
    # variant are separate arguments, and the domains come in as a dict.
    shape = workload_shape(spec)
    bucket = workload_bucket(spec)
    variant = resolve_variant(spec.service.dtype, spec.service.kv_cache_dtype)
    # The planner's own loaders, so the experiment and the planner cannot
    # disagree about which calibration an island maps to. The domain dict and the
    # scalar fallback are separate arguments: dropping the former IS scalar mode.
    domains = load_accuracy_domains(".", DOMAIN_CALIBRATION)
    calibration = load_calibrations(".", DOMAIN_CALIBRATION)
    candidates = {c.id: c for c in corpus}
    world = World(
        spec=spec, cluster=cluster, islands=by_id, candidates=candidates,
        # The operating points the evaluation recorded. main's margin policy
        # reads them off the SimResult `sim_for` builds, so leaving them out
        # makes every candidate `unmeasured` -- which is what it did.
        context=PerturbContext.build(spec, cluster, by_id, candidates,
                                     evaluation.operating_points),
        truth=judged_metrics(evaluation), island_hw=island_hw, costs=load_costs(),
        policy_truth=AccuracyDomainMargin(
            domains, shape=shape, model=spec.model, variant=variant,
            calibration=calibration, bucket=bucket),
        policy_scalar=AccuracyDomainMargin(
            _scalar_domains(domains), shape=shape, model=spec.model,
            variant=variant, calibration=calibration, bucket=bucket),
        profiles=profiles,
        predictor=predictor if keep_predictor else None,
        excluded_mirror=excluded_mirror,
        excluded_uncached=excluded_uncached,
    )
    world.pool = _degradable(world)
    return world


def _corpus(candidates: list, cache, *, exclude_mirrors: bool
            ) -> tuple[list, list[str], list[str]]:
    """The candidates this experiment may reason about, and the two it may not.

    Returns `(kept, excluded_mirror, excluded_uncached)`.

    **Mirrors (D40), and why they are excluded only with P/D on.** The placement
    key records each island's shape but not WHICH island got which share, so any
    two candidates differing only by that read one cache entry -- which is why
    E-A1's 324 aggregated candidates are served by 162 entries. Whether that is
    a defect depends on the corpus:

    * aggregated, over the two identical A40 nodes: `...node_a40a-tp4` and
      `...node_a40b-tp4` ARE the same deployment on the same hardware, so one
      simulation is the right answer for both, and E-B1 NEEDS both present --
      "degrading `profile:a40a` hands the plan to its twin `a40b`" is one of the
      three findings the committed result rests on. Collapsing the pair would
      delete that finding rather than clean it up;
    * P/D: the two members differ in which island PREFILLS, which the key cannot
      see and the simulator does not treat as equivalent. That is D40, and there
      one entry voting twice is exactly the double count STEP C1 removed.

    So the fixture decides, via `enable_pd`. The pair is identified by asking the
    cache for its own key -- the same function STEP C1 built the truth with,
    imported rather than re-derived so the two cannot disagree about what a
    mirror is.

    **Uncached (D71).** A candidate whose run raised in the simulator has no
    entry. Evaluating it here would be a cache MISS, which this experiment
    treats as a fatal error precisely because a miss means it is about to
    simulate. They are dropped with their ids kept, because on F2 they are not a
    random 23: they are every `tp1-dp1` P/D split, the family a link degradation
    would have had the most to say about.

    **Uncached is checked either way.** It is the guard that keeps "this
    experiment never simulates" true, and on the E-A1 fixture it finds nothing:
    every one of the 324 candidates keys to one of the 162 committed entries.
    """
    mirror_ids: set[str] = set()
    if exclude_mirrors:
        mirrors = mirror_groups(candidates, cache)
        mirror_ids = {i for ids in mirrors.values() for i in ids[1:]}
    kept, uncached = [], []
    for c in candidates:
        if c.id in mirror_ids:
            continue
        key = cache.cache_key(c)
        # `cache.get` would count a miss and trip the no-simulation guard below,
        # so coverage is read off the path instead of through the accessor.
        if key is None or not Path(key).exists():
            uncached.append(c.id)
            continue
        kept.append(c)
    return kept, sorted(mirror_ids), uncached


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
        # main's four-argument signature (D33 §3): the policy reads the
        # operating point off a SimResult, which `sim_for` rebuilds around
        # the perturbed metrics.
        decision = policy.decide(
            candidate, world.context.sim_for(cid, metrics[cid]),
            metrics[cid], world.island_hw,
        )
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
        plan.candidate, world.context.sim_for(candidate_id, plan.predicted),
        plan.predicted, world.island_hw))
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
    return [s.input_id for s in sensitivities(
        world, degraded, restored, penalty, grid)]


def sensitivities(
    world: World, degraded: list[Degraded], restored: set[str],
    penalty: float, grid: int,
) -> list:
    """`analyze`'s ranking of the still-degraded inputs, dR and all.

    Split out of `_order` because §2.3's headline metric is `n_active` -- how
    many inputs, and how many KINDS of input, have dR > 0 -- and the strategy
    loop only ever needed the order. Throwing the dR away made the one number
    the fixture is judged on unrecoverable from the raw output.
    """
    remaining = [d for d in degraded if d.id not in restored]
    metrics, policy = _state(world, degraded, restored)
    best = _best(world, metrics, policy)
    output = PlannerOutput(
        feasible=best is not None, service_model=world.spec.model,
        cluster_id=world.cluster.cluster_id, recommended=best,
    )
    registry = UncertainInputRegistry(items=[d.item for d in remaining])
    return analyze(
        output, registry, metrics, policy, world.spec, world.context,
        world.island_hw, grid=grid, slo_penalty=penalty,
    )


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


def stratified_subsets(pool: list[Degraded], k: int) -> list[list[Degraded]]:
    """Every k-subset that mixes at least two KINDS of uncertain input.

    Plain random sampling is dominated by the kind with the most items: this
    cluster has six links against four profiles and one domain, so more than a
    third of the k=2 draws are link-only and nothing in them can interact. A
    set that degrades one link and one profile is the only kind of set that can
    show a strategy choosing BETWEEN kinds, which is what E-B1 measures.

    `k <= 1` is returned whole: a single-item set cannot mix kinds, and the work
    order asks for those to be enumerated exhaustively anyway.

    The pool is small enough (C(11,3) = 165) to enumerate and filter exactly,
    rather than rejection-sample until a draw happens to qualify -- so the
    result is unbiased over the qualifying subsets by construction, and a
    caller cannot get a single-kind set no matter how it draws.
    """
    subsets = [list(c) for c in itertools.combinations(pool, k)]
    if k <= 1:
        return subsets
    return [s for s in subsets if len({d.item.kind for d in s}) >= 2]


def _degradation_sets(world: World, k: int, args) -> list[list[Degraded]]:
    """Which k-subsets of the pool to degrade.

    The work order asks for 10 random seeds. `--exhaustive` enumerates every
    subset instead, which is strictly more informative here and costs nothing:
    the whole experiment is closed form, and C(11,3) = 165.
    """
    if k > len(world.pool):
        return []
    if args.stratified:
        subsets = stratified_subsets(world.pool, k)
        if not subsets:
            raise SystemExit(
                f"--stratified: no k={k} subset of the pool mixes two kinds; "
                f"the pool has {len({d.item.kind for d in world.pool})} kind(s)"
            )
        if args.exhaustive:
            return subsets
        return [
            random.Random(1000 * k + seed).choice(subsets)
            for seed in range(args.seeds)
        ]
    if args.exhaustive:
        return [list(c) for c in itertools.combinations(world.pool, k)]
    return [
        random.Random(1000 * k + seed).sample(world.pool, k)
        for seed in range(args.seeds)
    ]


def verdict_counts(
    world: World, metrics: JudgedMetrics, policy: MarginPolicy
) -> dict[str, int]:
    """How the corpus was judged in one state: feasible / rejected / unmeasured."""
    counts: dict[str, int] = {}
    for cid in sorted(metrics):
        candidate = world.candidates.get(cid)
        if candidate is None:
            continue
        decision = policy.decide(
            candidate, world.context.sim_for(cid, metrics[cid]),
            metrics[cid], world.island_hw,
        )
        plan = DeploymentPlan(
            plan_id=f"hp-{cid}", model=world.spec.model, candidate=candidate,
            predicted=metrics[cid],
        )
        verdict = judge(plan, world.spec, decision).verdict.value
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def require_judged_degraded(
    world: World, degraded: list[Degraded]
) -> dict[str, int]:
    """The degraded state must JUDGE its corpus. Returns the verdict counts.

    E-B1's first run passed no domains at all, which is not scalar mode but no
    mode: every candidate came back `unmeasured`, the degraded search had
    nothing feasible, and `analyze` returned dR None for every item. The ranking
    then fell back to alphabetical order, which put the structurally inert links
    ahead of the domain and made `ours` look like the worst of the five
    strategies. Nothing in the output said so.

    **The condition is "nothing was judged", not "nothing was feasible",** and
    the difference is the whole point. On F2, degrading `profile:cuda-a40-node_
    a40a` alone leaves all 223 candidates SLO-`rejected`: a real verdict, in the
    tight TTFT regime this fixture was built for, and the state where a
    measurement is worth the most -- `_truth_value` already prices a
    recommendation of None at a full penalty. Halting there would refuse to
    measure the most informative degradation in the sweep. `unmeasured` is the
    accident: the candidate was not judged at all, so there is no verdict to
    have regret about, and a comparison between strategies degenerates into a
    comparison of tie-breaking.
    """
    metrics, policy = _state(world, degraded, set())
    counts = verdict_counts(world, metrics, policy)
    if counts.get("feasible", 0) == 0 and counts.get("rejected", 0) == 0:
        raise SystemExit(
            f"the degraded state for {[d.id for d in degraded]} judged nothing: "
            f"{counts or 'an empty corpus'}. Every dR would be None and the "
            "ranking would fall back to alphabetical order. Check that the "
            "degraded accuracy domain is scalar mode (one point per domain) "
            "and not an empty domain dict."
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--fixture", type=Path, default=FIXTURE,
        help="Fixture json (service, cluster, num_requests, seed, and optionally "
             "enable_pd). Defaults to E-A1's, so the committed result reproduces.",
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="Draw only degradation sets that mix at least two KINDS of input "
             "(§2.2). Without it the six links dominate the k=2 and k=3 draws.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
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

    world = build_world(args.cache_dir, args.work_dir, args.workers,
                        fixture_path=args.fixture)
    if world.excluded_mirror or world.excluded_uncached:
        print(f"corpus: {len(world.candidates)} candidates; "
              f"{len(world.excluded_mirror)} excluded as D40 mirrors, "
              f"{len(world.excluded_uncached)} with no cache entry (D71)")
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
            counts = require_judged_degraded(world, chosen)
            for label, value in penalties:
                # §2.3's n_active: the dR the ranking saw at the fully degraded
                # start, kept per item so the results can count how many inputs
                # -- and how many kinds -- were active at all.
                ranked = sensitivities(world, chosen, set(), value, args.grid)
                deltas = [
                    {"input_id": s.input_id, "kind": s.kind,
                     "delta_regret": s.delta_regret, "flip": s.flip,
                     "per_hour": s.regret_per_hour}
                    for s in ranked
                ]
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
                        #: How the corpus was judged while fully degraded. A set
                        #: with `feasible: 0` is not an error (see
                        #: `require_judged_degraded`) -- it is the state where a
                        #: measurement buys the most.
                        "degraded_verdicts": counts,
                        # Identical across the five strategies -- it describes
                        # the degraded STATE, not the strategy -- so it is
                        # attached once rather than five times.
                        "sensitivities": deltas if strategy == "ours" else None,
                    })
            print(f"  k={k} set={index} {[d.id for d in chosen]} done")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fixture": json.loads(args.fixture.read_text()),
        "fixture_path": str(args.fixture),
        "excluded_mirror": world.excluded_mirror,
        "excluded_uncached": world.excluded_uncached,
        "corpus_size": len(world.candidates),
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
        "sampling": ("exhaustive" if args.exhaustive else f"random x{args.seeds}")
                    + (", stratified by kind" if args.stratified else ""),
        "stratified": args.stratified,
        "runs": runs,
        "provenance": prov.collect(random_seed=0),
    }
    out_json = args.out_json or Path(
        "outputs/uncertainty/eb1/eb1_regret_vs_budget.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
