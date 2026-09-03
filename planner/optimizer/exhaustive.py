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
from planner.optimizer.surrogate import SurrogateRanker
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    PlannerOutput,
    PredictedMetrics,
    Rejection,
    RejectionStage,
    Role,
    RoutingPolicy,
    ServingArch,
    UnscoredPlan,
    summarize_rejections,
)
from planner.predictor import Predictor, SimResult
from planner.spec import ServiceSpec
from planner.topology import TopologyError, TopologyGraph
from planner.util import kv_transfer
from planner.util import tier as tierutil
from planner.util.parallel import predict_all

PD_TRANSFER_CAVEAT = (
    "P/D KV-transfer cost is a planner-side analytical add-on; the simulator models "
    "the prefill->decode handoff as free (docs/phase5_plan.md increment 2). Reported "
    "TTFT and energy for pd_split plans include it. TTFT offsets are per-percentile "
    "(p99 uses the p99 prompt length, etc.); the energy offset is a p50/median "
    "approximation of the mean, so it is slightly low on skewed prompt distributions. "
    "TPOT is unchanged (the transfer is one-time, pre-decode) and transfer power is not "
    "modeled. Bandwidth/path/prompt assumptions are recorded in provenance['pd_transfer']."
)

PD_TRANSFER_CLASS_DEFAULT_CAVEAT = (
    "The recommended P/D plan's prefill and decode islands are NOT joined by any "
    "declared link, so its KV-transfer cost was charged an interconnect class-default "
    "bandwidth (the same fallback the simulator's compiler uses), not a measured or "
    "declared path. Declare the fabric link between these islands to price it properly."
)

PHASE2_PREFIX_CACHE_CAVEAT = (
    "Prefix caching is disabled for every candidate. With it enabled the simulator's "
    "prefix-cache memory grows monotonically and the run dies on any device small "
    "enough to saturate (docs/deviations.md D12). Consequence: "
    "ServiceSpec.traffic.prefix_share_ratio has no effect on these predictions, and "
    "TTFT is pessimistic for workloads with genuinely shared prefixes."
)

SURROGATE_TOPK_CAVEAT = (
    "Stage-6 surrogate top-K was active: only the K candidates the analytical "
    "roofline ranked best were fully simulated; the rest were HEURISTICALLY dropped "
    "(not a sound bound - this can drop the true optimum). Metrics come from full "
    "simulation of the survivors only. If this search is infeasible or the "
    "recommendation looks weak, re-run with --oracle or a larger --top-k to rule out "
    "surrogate error (measured by experiments/scripts/exp_surrogate.py)."
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
    #: One entry per P/D-split candidate whose predicted metrics were adjusted
    #: for the KV transfer. Surfaced into provenance so the assumptions behind
    #: the (analytical, un-simulated) transfer cost travel with the plan.
    pd_transfers: list[dict] = field(default_factory=list)


def _plan_id(index: int) -> str:
    return f"hp-{index:05d}"


def apply_pd_transfer_cost(
    candidate: CandidateConfig,
    metrics: PredictedMetrics,
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands: dict[str, ExecutionIsland],
    topology: TopologyGraph,
) -> tuple[PredictedMetrics, dict]:
    """Add the prefill->decode KV-transfer cost to a P/D candidate's metrics.

    The simulator models the P/D handoff as free (`transfer_prefill_request`
    reallocates KV with zero simulated time; docs/phase5_plan.md increment 2), so
    HeteroPilot prices the transfer itself. This runs *after* prediction and
    outside the predictor, identically in oracle and pruned modes, so it is a
    metric adjustment of an already-selected candidate, never a pruning stage -
    oracle-agreement is untouched (§5.6 declares no P/D constraint).

    TTFT offsets are computed **per percentile**: transfer time scales with prompt
    length and the SLO is gated at a percentile, so p99 uses `input_tokens.p99`
    (falling back to p50 when that percentile is unset), p95 uses `input_tokens.p95`,
    p50 uses `input_tokens.p50`. Using the median for the p99 tail would understate
    it and could admit a P/D that really violates the SLO. TPOT is left alone: the
    transfer is one-time, before decode begins. Energy adds a per-request transfer
    term (a p50/median approximation of the mean, slightly low on skewed prompt
    distributions) times the request count; average/peak power are left unmodeled.

    When no declared link joins the two islands the transfer is **not** free: it is
    charged the same interconnect class-default the simulator's compiler uses
    (`topology._inter_island` -> `reduce_for_simulator`), so a P/D over an undeclared
    fabric cannot look artificially cheap and win the ranking silently. The
    class-default assumption is recorded. Non-P/D candidates are returned unchanged
    with an empty info.
    """
    if candidate.serving_arch is not ServingArch.PD_SPLIT:
        return metrics, {}

    prefills = [a for a in candidate.assignments if a.role is Role.PREFILL]
    decodes = [a for a in candidate.assignments if a.role is Role.DECODE]
    if len(prefills) != 1 or len(decodes) != 1:
        # _build_pd emits exactly one of each today; fail loud if a future
        # multi-assignment generator change would silently under-charge here.
        raise ValueError(
            f"PD_SPLIT candidate {candidate.id!r} must have exactly one PREFILL and "
            f"one DECODE assignment, got {len(prefills)} prefill / {len(decodes)} decode"
        )
    prefill, decode = prefills[0], decodes[0]
    pf_island = islands[prefill.island_id]
    dc_island = islands[decode.island_id]
    prefill_ep = f"{pf_island.node_id}/{pf_island.accelerator_ids[0]}"
    decode_ep = f"{dc_island.node_id}/{dc_island.accelerator_ids[0]}"

    assumptions: list[str] = []
    class_default = False
    try:
        path = topology.path(prefill_ep, decode_ep)
        bw_gbps = TopologyGraph.effective_bandwidth_gbps(path)
        latency_ns = TopologyGraph.path_latency_ns(path)
        energy_carrying = any(link.energy_per_bit_pj is not None for link in path)
    except TopologyError:
        # No path in the spec. Match the compiler rather than charging zero: the
        # sim's link_bw for this pair is the class default, not 0, so our transfer
        # must be too (otherwise an undeclared-fabric P/D looks free and can win).
        class_default = True
        bw_gbps, latency_ns, notes = topology._inter_island(pf_island, dc_island)
        assumptions.extend(notes)
        # An unpriced (class-default) hop carries no per-link energy figure.
        energy_carrying = False

    kv_per_token = kv_transfer._kv_bytes_per_token(
        spec.model, spec.service.dtype, spec.service.kv_cache_dtype
    )

    def _xfer_ms(prompt_tokens: int) -> float:
        return latency_ns / 1e6 + (kv_per_token * prompt_tokens / (bw_gbps * 1e9)) * 1e3

    tok = spec.traffic.input_tokens
    p50_tok = tok.p50
    p95_tok = tok.p95 if tok.p95 is not None else tok.p50
    p99_tok = tok.p99 if tok.p99 is not None else tok.p50
    xfer_p50, xfer_p95, xfer_p99 = _xfer_ms(p50_tok), _xfer_ms(p95_tok), _xfer_ms(p99_tok)

    # Energy from the median prompt over the priced path only; an undeclared-fabric
    # (class-default) hop has no per-link energy, so it contributes none.
    energy_j = 0.0
    if energy_carrying:
        _, energy_j = kv_transfer.kv_transfer_cost(
            spec.model, spec.service.dtype, p50_tok, path,
            kv_cache_dtype=spec.service.kv_cache_dtype,
        )

    updates: dict[str, object] = {
        "p50_ttft_ms": metrics.p50_ttft_ms + xfer_p50,
        "p95_ttft_ms": metrics.p95_ttft_ms + xfer_p95,
        "p99_ttft_ms": metrics.p99_ttft_ms + xfer_p99,
    }
    if metrics.total_energy_j is not None:
        new_energy = metrics.total_energy_j + energy_j * metrics.completed_requests
        updates["total_energy_j"] = new_energy
        updates["tokens_per_joule"] = (
            metrics.completed_tokens / new_energy if new_energy > 0 else None
        )
    adjusted = metrics.model_copy(update=updates)

    info = {
        "candidate_id": candidate.id,
        "xfer_ms_p50": xfer_p50,
        "xfer_ms_p95": xfer_p95,
        "xfer_ms_p99": xfer_p99,
        "energy_j_per_req": energy_j,
        "energy_prompt_tokens_p50": p50_tok,
        "prompt_tokens": {"p50": p50_tok, "p95": p95_tok, "p99": p99_tok},
        "path_bw_gbps": bw_gbps,
        "path_latency_ns": latency_ns,
        "class_default": class_default,
        "assumptions": assumptions,
    }
    return adjusted, info


def _routing_for(candidate: CandidateConfig) -> RoutingPolicy:
    """Pick the routing policy the deployer/simulator should use.

    A P/D-split deployment routes prefill and decode to different engines, so it
    always carries `PD_SPLIT`. Everything else keeps the Phase 2 rule: a single
    replica needs no router (`SINGLE`), several replicas are load-balanced
    (`LOAD`).
    """
    if candidate.serving_arch is ServingArch.PD_SPLIT:
        return RoutingPolicy.PD_SPLIT
    if candidate.total_devices == candidate.assignments[0].devices_per_replica:
        return RoutingPolicy.SINGLE
    return RoutingPolicy.LOAD


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
    max_workers: int | None = None,
    progress: Callable[[int, int, CandidateConfig], None] | None = None,
) -> SearchResult:
    """Simulate every candidate and split by feasibility.

    The expensive simulations run concurrently (``max_workers``), but the result
    ASSEMBLY stays sequential in candidate order, so plan_ids (assigned by index)
    and every appended list are byte-identical to a sequential run - parallelism
    only speeds it up. §9 reproducibility is a property of the assembly order, not
    of the simulation order. The envelope cache is read before and written after
    the parallel phase, never from a worker thread, so no locking is needed.
    """
    result = SearchResult()
    topology = TopologyGraph(cluster)

    # Phase 1: resolve a SimResult for every candidate. Cache hits are free; the
    # misses are simulated concurrently, then their ok results are memoized.
    sims: dict[str, SimResult] = {}
    cached_ids: set[str] = set()
    to_sim: list[CandidateConfig] = []
    for candidate in candidates:
        hit = cache.get(candidate) if cache else None
        if hit is not None:
            sims[candidate.id] = hit
            cached_ids.add(candidate.id)
        else:
            to_sim.append(candidate)

    if to_sim:
        # Dedup misses that share an envelope-cache entry (two islands of the same
        # accelerator model map to the same key): simulate ONE representative per
        # key, then serve its result to the twins. This reproduces exactly what the
        # sequential get/predict/put loop did - the second same-key candidate was a
        # cache hit - so cache-hit accounting stays byte-identical AND a homogeneous
        # multi-island cluster memoizes instead of re-simulating.
        rep_by_key: dict[str, CandidateConfig] = {}
        reps: list[CandidateConfig] = []
        twins: list[CandidateConfig] = []
        for candidate in to_sim:
            key = cache.cache_key(candidate) if cache is not None else None
            if key is not None and key in rep_by_key:
                twins.append(candidate)
            else:
                if key is not None:
                    rep_by_key[key] = candidate
                reps.append(candidate)
        sims.update(predict_all(
            predictor, reps, spec, cluster, islands, profiles,
            max_workers=max_workers, progress=progress,
        ))
        if cache is not None:
            for rep in reps:
                if sims[rep.id].ok:
                    cache.put(rep, sims[rep.id])
            for twin in twins:
                served = cache.get(twin)  # hits iff the representative's sim was ok
                if served is not None:
                    sims[twin.id] = served
                    cached_ids.add(twin.id)
                else:
                    # The representative's sim failed (nothing memoized), so this
                    # twin would also miss and re-run to the same deterministic
                    # failure; reuse it rather than re-simulate.
                    key = cache.cache_key(twin)
                    assert key is not None  # a twin is only formed when it has a key
                    sims[twin.id] = sims[rep_by_key[key].id]

    # Phase 2: assemble in candidate order (deterministic plan_ids and lists).
    for index, candidate in enumerate(candidates):
        sim = sims[candidate.id]
        cached = candidate.id in cached_ids
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
        # Price the prefill->decode KV transfer the simulator leaves free, before
        # building the plan so feasibility checks the SLO *with* it included: a
        # P/D whose transfer blows the TTFT budget must become infeasible. Applied
        # identically here in oracle and pruned modes, so oracle-agreement holds.
        metrics = sim.metrics
        if candidate.serving_arch is ServingArch.PD_SPLIT:
            metrics, pd_info = apply_pd_transfer_cost(
                candidate, metrics, spec, cluster, islands, topology
            )
            result.pd_transfers.append(pd_info)

        plan = DeploymentPlan(
            plan_id=_plan_id(index),
            model=spec.model,
            candidate=candidate,
            predicted=metrics,
            routing=_routing_for(candidate),
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
    if rejected.get(RejectionStage.SURROGATE_PRUNED.value):
        out.append(
            "surrogate top-K was active and dropped candidates heuristically - this "
            "'infeasible' may be surrogate error; re-run with --oracle or a larger "
            "--top-k before trusting it"
        )
    if not out:
        out.append("no candidates were generated at all; check the cluster spec and profiles")
    return out


def _profile_tiers(
    spec: ServiceSpec,
    islands: list[ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    perf_root=None,
) -> tuple[dict[str, tierutil.ProfileTier], dict[str, str], list[str]]:
    """Resolve every island's bundle tier (STEP 2 of the tiered-profile work).

    Returns (island_id -> tier, island_id -> hardware label, warnings). An
    island with no profile or no sim_hardware has no bundle at all, which is
    PLACEHOLDER by definition.
    """
    variant = tierutil.resolve_variant(spec.service.dtype, spec.service.kv_cache_dtype)
    tiers: dict[str, tierutil.ProfileTier] = {}
    hw_labels: dict[str, str] = {}
    warnings: list[str] = []
    for island in islands:
        profile = profiles.get(island.accelerator_model)
        hardware = profile.sim_hardware if profile is not None else None
        if hardware is None:
            tiers[island.id] = tierutil.ProfileTier.PLACEHOLDER
            hw_labels[island.id] = island.accelerator_model
            continue
        res = tierutil.resolve_bundle_tier_report(perf_root, hardware, spec.model, variant)
        tiers[island.id] = res.tier
        hw_labels[island.id] = hardware
        warnings.extend(res.warnings)
    return tiers, hw_labels, warnings


def _tier_summary(
    used_island_ids: set[str],
    tiers: dict[str, tierutil.ProfileTier],
    hw_labels: dict[str, str],
) -> tuple[str, list[str]]:
    """Weakest tier over the islands the reported plans actually use, plus the
    mandatory caveat per distinct non-measurement (tier, hardware) pair."""
    used = sorted(used_island_ids & set(tiers)) or sorted(tiers)
    overall = tierutil.min_tier(tiers[i] for i in used)
    caveats: list[str] = []
    for island_id in used:
        c = tierutil.caveat_for(tiers[island_id], hw_labels[island_id])
        if c is not None and c not in caveats:
            caveats.append(c)
    return overall.value, caveats


def _islands_of_plans(plans) -> set[str]:
    out: set[str] = set()
    for plan in plans:
        if plan is None:
            continue
        out.update(a.island_id for a in plan.candidate.assignments)
    return out


def search(
    spec: ServiceSpec,
    cluster: ClusterSpecV2,
    islands: list[ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    predictor: Predictor,
    *,
    perf_root=None,
    enable_bound_pruning: bool = True,
    enable_pd: bool = False,
    cache: EnvelopeCache | None = None,
    gpu_memory_utilization: float = 0.90,
    activation_reserve_gb: float = 0.0,
    ttft_margin_percent: float = 0.0,
    tpot_margin_percent: float = 0.0,
    surrogate: SurrogateRanker | None = None,
    top_k: int | None = None,
    max_workers: int | None = None,
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
        enable_pd=enable_pd,
    )
    generation = generator.generate()

    by_id = {island.id: island for island in islands}

    # Stage-6 surrogate top-K (opt-in, HEURISTIC). The surrogate order only
    # chooses WHICH candidates to simulate; it never REORDERS them, because
    # evaluate_candidates assigns plan_ids by index (see _plan_id) - reordering
    # would renumber plans and break byte-identity for the surviving subset. So
    # we build a keep-set from the surrogate's top-K, then filter in generation
    # order. When surrogate/top_k is unset this block is skipped entirely, so the
    # default path is byte-identical.
    candidates = generation.candidates
    surrogate_rejections: list[Rejection] = []
    if surrogate is not None and top_k is not None and top_k < len(candidates):
        ordered = surrogate.order(
            candidates, spec, by_id, profiles,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        keep = {c.id for c in ordered[:top_k]}
        surrogate_rejections = [
            Rejection(
                candidate_id=c.id,
                stage=RejectionStage.SURROGATE_PRUNED,
                reason=f"surrogate rank below top-{top_k}; heuristically dropped "
                       f"(NOT a sound bound - this stage can drop the optimum)",
            )
            for c in candidates if c.id not in keep
        ]
        candidates = [c for c in candidates if c.id in keep]

    evaluation = evaluate_candidates(
        candidates,
        spec,
        cluster,
        by_id,
        profiles,
        predictor,
        cache=cache,
        ttft_margin_percent=ttft_margin_percent,
        tpot_margin_percent=tpot_margin_percent,
        max_workers=max_workers,
        progress=progress,
    )

    island_tiers, island_hw, tier_warnings = _profile_tiers(spec, islands, profiles, perf_root)

    all_rejections = generation.rejections + surrogate_rejections + evaluation.rejections
    summary = summarize_rejections(all_rejections)
    caveats = [PHASE2_PREFIX_CACHE_CAVEAT]
    caveats.extend(tier_warnings)
    if not enable_bound_pruning:
        caveats.append(
            "Oracle mode: bound-based pruning was disabled, so every structurally "
            "valid candidate was simulated. Slow by design."
        )

    if surrogate_rejections:
        caveats.append(SURROGATE_TOPK_CAVEAT)

    if evaluation.pd_transfers:
        caveats.append(PD_TRANSFER_CAVEAT)

    prov = dict(provenance or {})
    if evaluation.cache_hits:
        prov["envelope_cache_hits"] = len(evaluation.cache_hits)
        prov["envelope_cache_hit_ids"] = sorted(evaluation.cache_hits)
    if evaluation.pd_transfers:
        prov["pd_transfer"] = {
            "note": (
                "planner-side analytical KV-transfer cost added to pd_split metrics; "
                "the simulator models the prefill->decode handoff as free"
            ),
            "candidates": evaluation.pd_transfers,
        }

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

        # If the recommendation's KV transfer was priced on a class-default (no
        # declared link) rather than a real path, surface that prominently - it is
        # the difference between a measured cost and a fallback guess.
        feasible_caveats = list(caveats)
        by_cand = {t["candidate_id"]: t for t in evaluation.pd_transfers}
        if by_cand.get(best.plan.candidate.id, {}).get("class_default"):
            feasible_caveats.insert(0, PD_TRANSFER_CLASS_DEFAULT_CAVEAT)

        used = _islands_of_plans(
            [best.plan, *(s.plan for s in alternatives), *(u.plan for u in unscored)]
        )
        tier_value, tier_caveats = _tier_summary(used, island_tiers, island_hw)
        prov["profile_tier"] = tier_value
        prov["profile_tiers"] = {i: t.value for i, t in sorted(island_tiers.items())}

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
            caveats=feasible_caveats + evaluation.notes + tier_caveats,
            profile_tier=tier_value,
            profile_tiers=prov["profile_tiers"],
        )

    closest_pair = min(
        evaluation.infeasible_plans, key=lambda pair: pair[1].worst_overshoot, default=None
    )
    closest, report = closest_pair if closest_pair else (None, None)

    used = _islands_of_plans(
        [closest, *(u.plan for u in unscored), *(p for p, _ in evaluation.infeasible_plans)]
    )
    tier_value, tier_caveats = _tier_summary(used, island_tiers, island_hw)
    prov["profile_tier"] = tier_value
    prov["profile_tiers"] = {i: t.value for i, t in sorted(island_tiers.items())}

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
        caveats=caveats + evaluation.notes + tier_caveats,
        profile_tier=tier_value,
        profile_tiers=prov["profile_tiers"],
    )


def oracle(*args, **kwargs) -> PlannerOutput:
    """Exhaustive search with every bound-based filter disabled (§5.6).

    Do not delete. It is the only way to tell a pruning bug from a genuinely
    empty feasible set.
    """
    kwargs["enable_bound_pruning"] = False
    # The oracle simulates EVERYTHING by definition, so it can never be limited by
    # the heuristic surrogate top-K - hard-strip it even if a caller passes it.
    kwargs["surrogate"] = None
    kwargs["top_k"] = None
    return search(*args, **kwargs)
