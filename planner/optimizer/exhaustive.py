"""Exhaustive search and the planning driver (work order §5.6).

The oracle exists to be *slow and correct*. Running every structurally valid
candidate through full simulation is how we measure a heuristic's optimality
gap, catch optimizer bugs, provide a paper baseline, and separate surrogate
error from search error. The work order is explicit: **never delete this.**

`plan()` is the same search with the bound-based filters on. Oracle agreement -
both returning the same optimum on a small cluster - is what proves the bounds
never prune an achievable configuration (§9).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from planner.candidate_generator import CandidateGenerator
from planner.envelope import EnvelopeCache
from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
from planner.optimizer import feasibility, pareto
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    PlannerOutput,
    Rejection,
    RejectionStage,
    RoutingPolicy,
    UnscoredPlan,
    summarize_rejections,
)
from planner.predictor import Predictor
from planner.spec import ServiceSpec

PHASE2_PREFIX_CACHE_CAVEAT = (
    "Prefix caching is disabled for every candidate. With it enabled the simulator's "
    "prefix-cache memory grows monotonically and the run dies on any device small "
    "enough to saturate (docs/deviations.md D12). Consequence: "
    "ServiceSpec.traffic.prefix_share_ratio has no effect on these predictions, and "
    "TTFT is pessimistic for workloads with genuinely shared prefixes."
)


@dataclass
class SearchResult:
    feasible_plans: list[DeploymentPlan] = field(default_factory=list)
    infeasible_plans: list[tuple[DeploymentPlan, feasibility.FeasibilityReport]] = field(
        default_factory=list
    )
    rejections: list[Rejection] = field(default_factory=list)
    generated: int = 0
    evaluated: int = 0
    notes: list[str] = field(default_factory=list)
    #: Reported as a provenance count, not as caveats - one line per cached
    #: candidate would bury the caveat that actually matters.
    cache_hits: list[str] = field(default_factory=list)


def _plan_id(index: int) -> str:
    return f"hp-{index:05d}"


def evaluate_candidates(
    candidates: list[CandidateConfig],
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    predictor: Predictor,
    *,
    cache: EnvelopeCache | None = None,
    ttft_margin_percent: float = 0.0,
    tpot_margin_percent: float = 0.0,
    progress: Callable[[int, int, CandidateConfig], None] | None = None,
) -> SearchResult:
    """Simulate every candidate and split by feasibility."""
    result = SearchResult()

    for index, candidate in enumerate(candidates):
        if progress is not None:
            progress(index, len(candidates), candidate)

        sim = cache.get(candidate) if cache else None
        cached = sim is not None
        if sim is None:
            sim = predictor.predict(candidate, spec, cluster, islands, profiles)
            if cache is not None and sim.ok:
                cache.put(candidate, sim)

        result.evaluated += 1

        if not sim.ok:
            # A crash or a timeout is a broken run, not a verdict on the
            # configuration. It gets its own bucket so a search full of them
            # cannot be mistaken for a search that found nothing feasible.
            result.rejections.append(
                Rejection(
                    candidate_id=candidate.id,
                    stage=RejectionStage.SIM_ERROR,
                    reason=f"{sim.outcome.value}: {sim.detail}",
                )
            )
            continue

        assert sim.metrics is not None
        plan = DeploymentPlan(
            plan_id=_plan_id(index),
            model=spec.model,
            candidate=candidate,
            predicted=sim.metrics,
            routing=(
                RoutingPolicy.SINGLE
                if candidate.total_devices == candidate.assignments[0].devices_per_replica
                else RoutingPolicy.LOAD
            ),
            robust_margin_ttft_percent=ttft_margin_percent,
            robust_margin_tpot_percent=tpot_margin_percent,
        )
        if cached:
            result.cache_hits.append(candidate.id)

        report = feasibility.evaluate(
            plan,
            spec,
            ttft_margin_percent=ttft_margin_percent,
            tpot_margin_percent=tpot_margin_percent,
        )
        result.notes.extend(report.notes)

        if report.passed:
            result.feasible_plans.append(plan)
        else:
            result.infeasible_plans.append((plan, report))
            result.rejections.append(
                Rejection(
                    candidate_id=candidate.id,
                    stage=report.stage or RejectionStage.SLO_VIOLATED,
                    reason="; ".join(
                        f"{v.metric}={v.predicted:.1f} vs target {v.target:.1f}"
                        for v in report.violations
                    ),
                )
            )
    return result


def _suggestions(
    spec: ServiceSpec,
    closest: DeploymentPlan | None,
    report: feasibility.FeasibilityReport | None,
    rejected: dict[str, int],
) -> list[str]:
    """Rule-based advice for an infeasible search (§3.5).

    Deliberately concrete: "relax TTFT from 500ms to 620ms" is actionable in a
    way that "consider relaxing the SLO" is not.
    """
    out: list[str] = []
    if rejected.get(RejectionStage.SIM_ERROR.value):
        out.append(
            f"{rejected[RejectionStage.SIM_ERROR.value]} candidate(s) failed to simulate at "
            f"all - fix those before trusting this result; they are not infeasible, "
            f"they are unmeasured"
        )
    if report is not None and closest is not None:
        for v in report.violations:
            if v.metric.endswith("_ttft_ms"):
                out.append(
                    f"relax the TTFT SLO from {v.target:.0f}ms to at least {v.predicted:.0f}ms, "
                    f"or add faster prefill capacity"
                )
            elif v.metric.endswith("_tpot_ms"):
                out.append(
                    f"relax the TPOT SLO from {v.target:.0f}ms to at least {v.predicted:.0f}ms, "
                    f"or raise tensor parallelism to cut per-token latency"
                )
            elif v.metric == "peak_power_w":
                out.append(
                    f"raise the power cap from {v.target:.0f}W to at least {v.predicted:.0f}W, "
                    f"or use fewer/lower-TDP accelerators"
                )
            elif v.metric == "tokens_per_joule":
                out.append(
                    f"lower the efficiency floor from {v.target:.2f} to at most "
                    f"{v.predicted:.2f} tokens/J, or move to more efficient hardware"
                )
        rate = spec.traffic.arrival_rate_rps
        if closest.predicted.slo_goodput_rps < rate:
            out.append(
                f"lower the admitted request rate from {rate:.1f} to about "
                f"{closest.predicted.slo_goodput_rps:.1f} rps"
            )
    if rejected.get(RejectionStage.MEMORY_INFEASIBLE.value):
        out.append("add accelerators with more memory, or use a smaller dtype / fp8 KV cache")
    if rejected.get(RejectionStage.BACKEND_INCOMPATIBLE.value):
        out.append(
            "some islands were excluded for missing or incompatible profiles - check "
            "supported_models and sim_hardware in profiles/accelerators/"
        )
    if not out:
        out.append("no candidates were generated at all; check the cluster spec and profiles")
    return out


def search(
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands: list[ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    predictor: Predictor,
    *,
    enable_bound_pruning: bool = True,
    cache: EnvelopeCache | None = None,
    gpu_memory_utilization: float = 0.90,
    activation_reserve_gb: float = 0.0,
    ttft_margin_percent: float = 0.0,
    tpot_margin_percent: float = 0.0,
    provenance: dict | None = None,
    progress: Callable[[int, int, CandidateConfig], None] | None = None,
) -> PlannerOutput:
    """Generate, simulate, filter, rank. `enable_bound_pruning=False` is oracle mode."""
    generator = CandidateGenerator(
        spec,
        cluster,
        islands,
        profiles,
        gpu_memory_utilization=gpu_memory_utilization,
        activation_reserve_gb=activation_reserve_gb,
        enable_prefix_caching=False,
        enable_bound_pruning=enable_bound_pruning,
    )
    generation = generator.generate()

    by_id = {island.id: island for island in islands}
    evaluation = evaluate_candidates(
        generation.candidates,
        spec,
        cluster,
        by_id,
        profiles,
        predictor,
        cache=cache,
        ttft_margin_percent=ttft_margin_percent,
        tpot_margin_percent=tpot_margin_percent,
        progress=progress,
    )

    all_rejections = generation.rejections + evaluation.rejections
    summary = summarize_rejections(all_rejections)
    caveats = [PHASE2_PREFIX_CACHE_CAVEAT]
    if not enable_bound_pruning:
        caveats.append(
            "Oracle mode: bound-based pruning was disabled, so every structurally "
            "valid candidate was simulated. Slow by design."
        )

    prov = dict(provenance or {})
    if evaluation.cache_hits:
        prov["envelope_cache_hits"] = len(evaluation.cache_hits)
        prov["envelope_cache_hit_ids"] = sorted(evaluation.cache_hits)

    # Split before ranking: a plan the objective cannot score must be surfaced,
    # not sorted to the bottom where it silently disappears.
    scorable: list[DeploymentPlan] = []
    unscored: list[UnscoredPlan] = []
    for plan in evaluation.feasible_plans:
        ok, why = pareto.can_score(plan, spec.objective.primary)
        (scorable.append(plan) if ok else unscored.append(UnscoredPlan(plan=plan, reason=why)))

    if scorable:
        ranked = pareto.rank(scorable, spec.objective.primary, spec.objective.secondary)
        front_ids = {p.plan_id for p in pareto.frontier(scorable)}
        scored_by_plan_id = {s.plan.plan_id: s for s in ranked}

        # Collapse identical outcomes across the recommendation *and* the
        # alternatives. Grouping only within the alternatives would still leave
        # an "alternative" that predicts exactly what the recommendation does,
        # which is the opposite of an alternative. Ranked order is preserved, so
        # the first group's representative is still the best plan.
        shown = [ranked[0]] + [s for s in ranked[1:] if s.plan.plan_id in front_ids]
        collapsed = pareto.collapse_equivalent([s.plan for s in shown])

        best = scored_by_plan_id[collapsed[0][0].plan_id].model_copy(
            update={"equivalent_candidates": collapsed[0][1]}
        )
        alternatives = [
            scored_by_plan_id[rep.plan_id].model_copy(
                update={"equivalent_candidates": dupes}
            )
            for rep, dupes in collapsed[1:]
        ]

        return PlannerOutput(
            feasible=True,
            service_model=spec.model,
            cluster_id=cluster.cluster_id,
            recommended=best,
            alternatives=pareto.annotate_alternatives(best.plan, alternatives),
            unscored=unscored,
            rejected_summary=summary,
            evaluated_candidates=evaluation.evaluated,
            generated_candidates=generation.generated,
            provenance=prov,
            caveats=caveats + evaluation.notes,
        )

    closest_pair = min(
        evaluation.infeasible_plans, key=lambda pair: pair[1].worst_overshoot, default=None
    )
    closest, report = closest_pair if closest_pair else (None, None)
    return PlannerOutput(
        feasible=False,
        service_model=spec.model,
        cluster_id=cluster.cluster_id,
        reason=(
            "no currently available configuration satisfies all constraints"
            if evaluation.infeasible_plans
            else "no candidate survived generation, so nothing was simulated"
        ),
        closest_plan=closest,
        violated_constraints=report.violations if report else [],
        suggestions=_suggestions(spec, closest, report, summary),
        unscored=unscored,
        rejected_summary=summary,
        evaluated_candidates=evaluation.evaluated,
        generated_candidates=generation.generated,
        provenance=prov,
        caveats=caveats + evaluation.notes,
    )


def oracle(*args, **kwargs) -> PlannerOutput:
    """Exhaustive search with every bound-based filter disabled (§5.6).

    Do not delete. It is the only way to tell a pruning bug from a genuinely
    empty feasible set.
    """
    kwargs["enable_bound_pruning"] = False
    return search(*args, **kwargs)
