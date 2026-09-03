"""E1 — does Tier 0 change the planner's DECISION? (STEP 11)

The planner's job is configuration RANKING, not absolute latency prediction,
so the headline question for Tier 0 is whether plans chosen on analytical
bundles agree with plans chosen on measured ones. Four legs per condition:

    (1) greedy    roofline proxy, no simulation (work order §12 baseline)
    (2) tier0     synthetic bundle + full simulation
    (3) tier2     measured bundle + full simulation   <- ground truth
    (4) oracle    tier2 with bound pruning disabled

Metrics per leg vs (3): top-1 agreement, top-3 containment, relative error
of the chosen plan's primary-objective value AS SCORED BY (3), and Kendall
tau over the shared feasible ranking.

Usage:
    python -m experiments.tier_validation.e1_plan_agreement --dry-run
    python -m experiments.tier_validation.e1_plan_agreement \
        --out outputs/tier_validation/e1 [--num-requests 50]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.tier_validation.common import (
    kendall_tau,
    top1_match,
    topk_contains,
    write_report,
)
from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    AcceleratorProfile,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive, greedy, pareto
from planner.predictor import Predictor
from planner.spec import ServiceSpec, load_service_spec

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class E1Condition:
    """One (cluster, service) pair plus how to reach its Tier 0 twin.

    The tier0 leg reuses the SAME cluster topology; only the profiles'
    sim_hardware is rewritten to the -t0 label, whose bundle must exist under
    the perf root (`scripts/gen-tier0-bundles.sh` / a mirror emit)."""

    name: str
    cluster: Path
    service: Path
    #: measured hardware label -> tier0 label (usually just <HW> -> <HW>-t0).
    tier0_labels: dict[str, str] = field(default_factory=dict)


#: The a40x8 production cluster yields 252 candidates (x3 simulated legs x2
#: specs ~ 1500 simulations); the Exp-1 TP-sweep subset yields 42, keeping a
#: full E1 run in the hours range. Both conditions use real measured A40
#: bundles as ground truth.
DEFAULT_CONDITIONS = (
    E1Condition(
        name="a40-tp-sweep-llama31-8b",
        cluster=REPO / "experiments" / "configs" / "clusters" / "exp1-a40-tp-sweep.yaml",
        service=REPO / "examples" / "service_specs" / "llama31-8b.yaml",
        tier0_labels={"A40": "A40-t0"},
    ),
    E1Condition(
        name="a40-tp-sweep-llama31-8b-light",
        cluster=REPO / "experiments" / "configs" / "clusters" / "exp1-a40-tp-sweep.yaml",
        service=REPO / "examples" / "service_specs" / "llama31-8b-light.yaml",
        tier0_labels={"A40": "A40-t0"},
    ),
)


def _tier0_profiles(
    profiles: dict[str, AcceleratorProfile], labels: dict[str, str]
) -> dict[str, AcceleratorProfile]:
    out = {}
    for model, profile in profiles.items():
        if profile.sim_hardware in labels:
            out[model] = profile.model_copy(
                update={"sim_hardware": labels[profile.sim_hardware]}
            )
        else:
            out[model] = profile
    return out


def _full_ranking(
    spec: ServiceSpec, cluster, islands, profiles, predictor: Predictor,
    *, enable_bound_pruning: bool = True, cache=None, max_workers: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Ranked feasible candidate ids + id -> primary-objective value."""
    generation = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_bound_pruning=enable_bound_pruning,
    ).generate()
    by_id = {i.id: i for i in islands}
    evaluation = exhaustive.evaluate_candidates(
        generation.candidates, spec, cluster, by_id, profiles, predictor,
        cache=cache, max_workers=max_workers,
    )
    scorable = [
        p for p in evaluation.feasible_plans
        if pareto.can_score(p, spec.objective.primary)[0]
    ]
    ranked = pareto.rank(scorable, spec.objective.primary, spec.objective.secondary)
    return (
        [s.plan.candidate.id for s in ranked],
        {s.plan.candidate.id: s.value for s in ranked},
    )


def run_condition(
    cond: E1Condition,
    predictor_factory: Callable[[ServiceSpec], Predictor],
    *,
    dry_run: bool = False,
    max_workers: int | None = None,
) -> dict[str, Any]:
    cluster = load_cluster_spec(cond.cluster)
    profiles_t2 = load_profiles_for(cluster, REPO)
    islands = detect_islands(cluster, profiles_t2)
    spec = load_service_spec(cond.service)
    profiles_t0 = _tier0_profiles(profiles_t2, cond.tier0_labels)

    generation = CandidateGenerator(spec, cluster, islands, profiles_t2).generate()
    if dry_run:
        return {
            "condition": cond.name,
            "dry_run": True,
            "generated_candidates": generation.generated,
            "surviving_candidates": len(generation.candidates),
            # 3 simulated legs (tier0, tier2, oracle); greedy is free.
            "simulations_upper_bound": 3 * generation.generated,
        }

    # (1) greedy: proxy ranking, zero simulations.
    greedy_ids = [
        e.candidate_id
        for e in greedy.rank(
            generation.candidates, spec, {i.id: i for i in islands}, profiles_t2
        )
        if not e.likely_infeasible
    ]
    # (2) tier0 + sim, (3) tier2 + sim (ground truth), (4) oracle on tier2.
    # tier2 and oracle share an envelope cache: the cache key is built from the
    # accelerator MODEL (not sim_hardware), so the tier0 leg must NOT share it
    # - a shared cache would silently serve tier2 results to the tier0 leg.
    import tempfile

    from planner.envelope import EnvelopeCache

    accelerator_of = {i.id: i.accelerator_model for i in islands}
    cache_t2 = EnvelopeCache(
        tempfile.mkdtemp(prefix="e1-cache-", dir=REPO / "outputs"),
        spec, accelerator_of=accelerator_of, link_bw_gbps=64.0,
    )
    t0_ids, _ = _full_ranking(
        spec, cluster, islands, profiles_t0, predictor_factory(spec),
        max_workers=max_workers,
    )
    t2_ids, t2_value = _full_ranking(
        spec, cluster, islands, profiles_t2, predictor_factory(spec),
        cache=cache_t2, max_workers=max_workers,
    )
    oracle_ids, _ = _full_ranking(
        spec, cluster, islands, profiles_t2, predictor_factory(spec),
        enable_bound_pruning=False, cache=cache_t2, max_workers=max_workers,
    )

    def leg_metrics(ids: list[str]) -> dict[str, Any]:
        chosen = ids[0] if ids else None
        rel_err = None
        if chosen is not None and t2_ids:
            truth_best = t2_value[t2_ids[0]]
            chosen_value = t2_value.get(chosen)
            if chosen_value is not None and truth_best:
                rel_err = abs(chosen_value - truth_best) / abs(truth_best)
        return {
            "top1_match": top1_match(ids, t2_ids),
            "top3_contains": topk_contains(ids, t2_ids, 3),
            "chosen": chosen,
            "rel_err_primary_objective": rel_err,
            "kendall_tau": kendall_tau(ids, t2_ids),
        }

    return {
        "condition": cond.name,
        "ground_truth": "tier2",
        "truth_top3": t2_ids[:3],
        "n_feasible_tier2": len(t2_ids),
        "legs": {
            "greedy": leg_metrics(greedy_ids),
            "tier0": leg_metrics(t0_ids),
            "tier2": leg_metrics(t2_ids),  # trivially perfect; sanity anchor
            "oracle": leg_metrics(oracle_ids),
        },
    }


def render_table(results: list[dict[str, Any]]) -> str:
    lines = [
        f"{'condition':<28} {'leg':<8} {'top1':>5} {'top3':>5} "
        f"{'rel.err':>8} {'tau':>6}"
    ]
    for r in results:
        if r.get("dry_run"):
            lines.append(
                f"{r['condition']:<28} dry-run: {r['surviving_candidates']} candidates, "
                f"<= {r['simulations_upper_bound']} simulations"
            )
            continue
        for leg, m in r["legs"].items():
            rel = "n/a" if m["rel_err_primary_objective"] is None else (
                f"{m['rel_err_primary_objective'] * 100:.1f}%"
            )
            lines.append(
                f"{r['condition']:<28} {leg:<8} "
                f"{m['top1_match']!s:>5} {m['top3_contains']!s:>5} "
                f"{rel:>8} {m['kendall_tau']:>6.3f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e1_plan_agreement")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "tier_validation" / "e1")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    def predictor_factory(spec: ServiceSpec) -> Predictor:
        import tempfile

        from planner.predictor.llmservingsim import LLMServingSimPredictor
        from planner.util.workload import generate_trace

        work_root = Path(tempfile.mkdtemp(prefix="e1-", dir=REPO / "outputs"))
        # One trace per leg from THIS condition's spec; the shared seed keeps
        # legs of one condition byte-identical.
        trace = generate_trace(
            spec,
            work_root / "workload.jsonl",
            num_requests=args.num_requests,
            seed=args.seed,
        )
        return LLMServingSimPredictor(trace, work_dir=work_root)

    results = [
        run_condition(
            c, predictor_factory, dry_run=args.dry_run, max_workers=args.workers
        )
        for c in DEFAULT_CONDITIONS
    ]
    table = render_table(results)
    print(table)
    if not args.dry_run:
        write_report(
            args.out, "e1_plan_agreement", {"results": results}, table=table,
            provenance_extra={"num_requests": args.num_requests, "seed": args.seed},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
