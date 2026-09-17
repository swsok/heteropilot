"""Human-readable rendering of a PlannerOutput (work order §6).

The `plan` command's stdout must always carry: feasible candidate count and top
list, rejected counts by stage with reasons, the recommendation, the Pareto
alternatives, and the predicted metrics. An infeasible result must diagnose
rather than just say no.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from planner.plan import DeploymentPlan, PlannerOutput, ScoredPlan
from planner.uncertainty.measurement_plan import MeasurementPlan
from planner.uncertainty.registry import (
    UncertainInput,
    UncertainInputRegistry,
    UncertainKind,
)

if TYPE_CHECKING:
    from planner.deploy.base import DeploymentHandle, DeploymentMetrics

WIDTH = 78


def _rule(title: str = "") -> str:
    if not title:
        return "-" * WIDTH
    return f"--- {title} " + "-" * max(0, WIDTH - len(title) - 5)


def _fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}{unit}"


def _placement(plan: DeploymentPlan) -> str:
    parts = []
    for a in plan.candidate.assignments:
        parts.append(
            f"{a.island_id} x{a.dp_replicas} (tp={a.tp_size}"
            + (f", pp={a.pp_size}" if a.pp_size > 1 else "")
            + f", role={a.role.value})"
        )
    return "; ".join(parts)


def render_metrics(plan: DeploymentPlan, indent: str = "  ") -> str:
    m = plan.predicted
    lines = [
        f"{indent}TTFT  p50/p95/p99 : "
        f"{_fmt(m.p50_ttft_ms)} / {_fmt(m.p95_ttft_ms)} / {_fmt(m.p99_ttft_ms)} ms",
        f"{indent}TPOT  p50/p95/p99 : "
        f"{_fmt(m.p50_tpot_ms)} / {_fmt(m.p95_tpot_ms)} / {_fmt(m.p99_tpot_ms)} ms",
        f"{indent}throughput        : {_fmt(m.throughput_tps)} tok/s",
        f"{indent}SLO attainment    : {m.slo_attainment * 100:.1f}%  "
        f"(goodput {_fmt(m.slo_goodput_rps, ' rps', 2)})",
        f"{indent}accelerators      : {plan.active_accelerators}",
    ]
    if m.has_energy:
        lines += [
            f"{indent}energy            : {_fmt(m.total_energy_j, ' J')}  "
            f"(avg {_fmt(m.average_power_w, ' W')}, peak {_fmt(m.peak_power_w, ' W')})",
            f"{indent}tokens/J          : {_fmt(m.tokens_per_joule, '', 3)}",
        ]
    else:
        lines.append(
            f"{indent}energy            : not simulated "
            f"(no power: block in the cluster spec - see deviations D2)"
        )
    return "\n".join(lines)


def render_deployment_metrics(metrics: DeploymentMetrics, indent: str = "  ") -> str:
    """Measured runtime metrics, mirroring `render_metrics` for the sim side."""
    lines = [
        f"{indent}TTFT  p50/p95/p99 : "
        f"{_fmt(metrics.p50_ttft_ms)} / {_fmt(metrics.p95_ttft_ms)} / "
        f"{_fmt(metrics.p99_ttft_ms)} ms",
        f"{indent}TPOT  p50/p95/p99 : "
        f"{_fmt(metrics.p50_tpot_ms)} / {_fmt(metrics.p95_tpot_ms)} / "
        f"{_fmt(metrics.p99_tpot_ms)} ms",
        f"{indent}throughput        : {_fmt(metrics.throughput_tps)} tok/s",
        f"{indent}completed         : {metrics.completed_requests} req, "
        f"{metrics.completed_tokens} tok",
    ]
    if metrics.has_energy:
        lines += [
            f"{indent}energy            : {_fmt(metrics.total_energy_j, ' J')}  "
            f"(avg {_fmt(metrics.average_power_w, ' W')}, "
            f"peak {_fmt(metrics.peak_power_w, ' W')})",
            f"{indent}tokens/J          : {_fmt(metrics.tokens_per_joule, '', 3)}",
        ]
    else:
        lines.append(
            f"{indent}energy            : not sampled "
            f"(no nvidia-smi or no device indices recorded)"
        )
    lines.append(
        f"{indent}sampling          : {metrics.sample_count} power sample(s) over "
        f"{_fmt(metrics.window_seconds, ' s')}"
    )
    return "\n".join(lines)


def render_deployment_handle(handle: DeploymentHandle, indent: str = "  ") -> str:
    """One-block summary of a launched deployment."""
    lines = [
        f"{indent}deployment id : {handle.deployment_id}",
        f"{indent}backend       : {handle.backend}",
        f"{indent}plan          : {handle.plan_id}",
        f"{indent}endpoint      : {handle.base_url}",
        f"{indent}pid           : {handle.pid if handle.pid is not None else 'n/a'}",
        f"{indent}started       : {handle.started_at}",
    ]
    devices = handle.extra.get("cuda_visible_devices")
    if devices is not None:
        lines.append(f"{indent}CUDA devices  : {devices}")
    return "\n".join(lines)


def _render_plan(scored: ScoredPlan, label: str) -> str:
    plan = scored.plan
    out = [
        f"{label}: {plan.plan_id}  [{plan.candidate.id}]",
        f"  placement         : {_placement(plan)}",
        f"  knobs             : max_num_seqs={plan.candidate.knobs.max_num_seqs}, "
        f"max_num_batched_tokens={plan.candidate.knobs.max_num_batched_tokens}, "
        f"prefix_caching={plan.candidate.knobs.enable_prefix_caching}",
        f"  routing           : {plan.routing.value}",
        f"  score             : {scored.objective.value} = {scored.value:,.4f}",
        render_metrics(plan),
    ]
    if plan.margin_basis:
        out.append(f"  margin basis      : {plan.margin_basis}")
    return "\n".join(out)


def render(output: PlannerOutput, *, top_n: int = 5) -> str:
    lines: list[str] = []
    # Non-measured inputs get a banner ABOVE everything else: a plan built on
    # analytical/calibrated/placeholder/unknown profiles must not be readable
    # as a measured result (tiered-profiles work, absolute rule A1).
    if output.profile_tier not in ("measured", "imported"):
        lines.append("!" * WIDTH)
        lines.append(
            f"!!  PROFILE TIER: {output.profile_tier.upper()} - this plan rests on "
            f"non-measured profile inputs  !!"
        )
        lines.append("!" * WIDTH)
        lines.append("")
    override = (output.provenance.get("uncertainty") or {}).get("bucket_override")
    if override:
        lines.append("!" * WIDTH)
        lines.append(
            f"!!  BUCKET OVERRIDE: accuracy domains looked up under "
            f"{override.get('used')} instead of this service's own "
            f"{override.get('requested')}  !!"
        )
        lines.append(
            "!!  This asserts that an error measured on a DIFFERENT workload applies "
            "here.  !!"
        )
        lines.append("!" * WIDTH)
        lines.append("")
    lines.append(_rule("HeteroPilot plan"))
    lines.append(f"service : {output.service_model}")
    lines.append(f"cluster : {output.cluster_id}")
    lines.append(
        f"searched: {output.generated_candidates} generated, "
        f"{output.evaluated_candidates} simulated "
        f"(prune ratio {output.prune_ratio * 100:.0f}%)"
    )
    lines.append("")

    # Rejections always, even on success: a search that pruned everything
    # interesting looks identical to a search that had nothing to prune.
    lines.append(_rule("Rejected candidates"))
    if output.rejected_summary:
        for stage, count in output.rejected_summary.items():
            lines.append(f"  {stage:<26} {count}")
        if "sim_error" in output.rejected_summary:
            lines.append(
                "  NOTE: sim_error means the simulator crashed or timed out. Those "
                "candidates are unmeasured, not infeasible."
            )
        if "outside_calibration_domain" in output.rejected_summary:
            lines.append(
                "  NOTE: outside_calibration_domain means a value the candidate needs "
                "was never measured - its operating point lies outside an accuracy "
                "domain under `refuse`, its hardware has no calibration, or the slab3d "
                "table does not cover it. Those candidates were NOT judged infeasible "
                "- they were not judged at all."
            )
        if "calibration_condition_mismatch" in output.rejected_summary:
            lines.append(
                "  NOTE: calibration_condition_mismatch means no accuracy domain was "
                "measured under this candidate's CONFIGURATION - its parallelism, "
                "island count, device binding, model, precision, token mix or arrival "
                "process differs from every domain's. Also unmeasured rather than "
                "infeasible, but it asks for a different experiment: measure at that "
                "configuration, not further along the load axis of another one."
            )
    else:
        lines.append("  (none)")
    lines.append("")

    if output.feasible and output.recommended is not None:
        feasible_count = 1 + len(output.alternatives)
        lines.append(_rule("Feasible candidates"))
        lines.append(f"  {feasible_count} on the Pareto frontier")
        lines.append("")
        lines.append(_rule("Recommended plan"))
        lines.append(_render_plan(output.recommended, "recommended"))
        lines.append("")
        lines.append(_rule("Pareto alternatives"))
        if output.alternatives:
            for alt in output.alternatives[:top_n]:
                lines.append(f"  {alt.plan.plan_id}  [{alt.plan.candidate.id}]")
                lines.append(f"    trade-off : {alt.note}")
                lines.append(f"    score     : {alt.value:,.4f}")
                if alt.equivalent_candidates:
                    lines.append(
                        f"    identical : {len(alt.equivalent_candidates)} other "
                        f"candidate(s) predict exactly this outcome "
                        f"({', '.join(alt.equivalent_candidates[:3])}"
                        + (", ..." if len(alt.equivalent_candidates) > 3 else "")
                        + ")"
                    )
                lines.append(render_metrics(alt.plan, indent="    "))
                lines.append("")
            if len(output.alternatives) > top_n:
                lines.append(f"  ... {len(output.alternatives) - top_n} more")
        else:
            lines.append("  (none; the recommendation dominates every other candidate)")

    else:
        lines.append(_rule("INFEASIBLE"))
        lines.append(f"  {output.reason}")
        lines.append("")
        if output.closest_plan is not None:
            lines.append("  closest plan:")
            lines.append(f"    {output.closest_plan.plan_id} "
                         f"[{output.closest_plan.candidate.id}]")
            lines.append(f"    placement : {_placement(output.closest_plan)}")
            lines.append(render_metrics(output.closest_plan, indent="    "))
            lines.append("")
        if output.violated_constraints:
            lines.append("  violated constraints:")
            for v in output.violated_constraints:
                lines.append(
                    f"    {v.metric:<22} target {v.target:,.2f}  "
                    f"predicted {v.predicted:,.2f}  (+{v.overshoot_ratio * 100:.0f}%)"
                )
            lines.append("")
        lines.append("  suggestions:")
        for s in output.suggestions:
            lines.append(f"    - {s}")

    # Applies to both branches: plans can be feasible-but-unrankable whether or
    # not anything else was rankable. Keep this outside the if/else above - an
    # earlier version sat between the if-body and the else, which silently
    # rebound the else to this condition and printed INFEASIBLE on success.
    if output.unscored:
        lines.append("")
        lines.append(_rule("Feasible but not ranked"))
        lines.append(
            f"  {len(output.unscored)} plan(s) met every constraint but could not be scored "
            f"on the primary objective:"
        )
        by_reason: dict[str, list[str]] = {}
        for u in output.unscored:
            by_reason.setdefault(u.reason, []).append(u.plan.candidate.id)
        for reason, ids in by_reason.items():
            lines.append(f"    {reason}")
            for cid in ids[:4]:
                lines.append(f"      - {cid}")
            if len(ids) > 4:
                lines.append(f"      ... {len(ids) - 4} more")

    if output.uncertain_inputs is not None:
        lines.append("")
        lines.append(render_uncertain_inputs(output.uncertain_inputs))

    if output.measurement_plan is not None:
        lines.append("")
        lines.append(render_measurement_plan(output.measurement_plan))

    if output.caveats:
        lines.append("")
        lines.append(_rule("Caveats"))
        for c in dict.fromkeys(output.caveats):
            lines.append(f"  - {c}")

    return "\n".join(lines)


#: Order the uncertain-input table walks its kinds; stable so the section reads
#: the same across runs and diffs cleanly.
_KIND_ORDER = ("sim_error", "profile", "power", "link_bw", "link_lat")


def _coverage_line(registry: UncertainInputRegistry) -> str:
    """"26 links, 16 uncertain (placeholder 14, vendor_spec 2)" per kind.

    Printed before the table because a table of uncertain inputs alone cannot
    say whether it covers two inputs or two hundred.
    """
    parts: list[str] = []
    for kind_value in _KIND_ORDER:
        kind = UncertainKind(kind_value)
        items = registry.by_kind(kind)
        total = registry.total_for(kind)
        if total == 0:
            continue
        by_grade: dict[str, int] = {}
        for item in items:
            by_grade[item.grade.value] = by_grade.get(item.grade.value, 0) + 1
        detail = ", ".join(f"{g} {n}" for g, n in sorted(by_grade.items()))
        parts.append(
            f"{kind_value} {len(items)}/{total} uncertain" + (f" ({detail})" if detail else "")
        )
    return "; ".join(parts) if parts else "no inputs classified"


def render_uncertain_inputs(registry: UncertainInputRegistry, *, top_n: int = 40) -> str:
    """The "Uncertain inputs" section (§2.3, STEP A1).

    Unbounded items come first and are marked: they are the ones no amount of
    analysis can settle, so they head the operator's reading order.
    """
    lines = [_rule("Uncertain inputs")]
    lines.append(f"  coverage: {_coverage_line(registry)}")
    if not registry.items:
        lines.append("  (every classified input is a measurement)")
        return "\n".join(lines)

    unbounded = registry.unbounded()
    if unbounded:
        lines.append(
            f"  {len(unbounded)} input(s) have NO sourced range - a plan cannot be "
            f"decided against them before measuring"
        )
    lines.append("")
    lines.append(
        f"  {'':2}{'id':<44} {'kind':<10} {'grade':<12} "
        f"{'nominal':>12}  {'range':<26} affects"
    )

    def sort_key(item: UncertainInput) -> tuple:
        return (
            0 if item.range.is_unbounded else 1,
            _KIND_ORDER.index(item.kind.value) if item.kind.value in _KIND_ORDER else 99,
            item.id,
        )

    ordered = sorted(registry.items, key=sort_key)
    for item in ordered[:top_n]:
        mark = "**" if item.range.is_unbounded else "  "
        affects = ", ".join(item.affects[:2]) + (
            f", +{len(item.affects) - 2}" if len(item.affects) > 2 else ""
        )
        lines.append(
            f"  {mark}{item.id:<44} {item.kind.value:<10} {item.grade.value:<12} "
            f"{item.nominal:>12,.4g}  {item.range!s:<26} {affects or '-'}"
        )
    if len(ordered) > top_n:
        lines.append(f"    ... {len(ordered) - top_n} more")
    lines.append("  ** = unbounded (no sourced range)")
    return "\n".join(lines)


def render_measurement_plan(plan: MeasurementPlan, *, top_n: int = 20) -> str:
    """The "Measurement plan" section (§2.5, STEP B3).

    Ordered by what each measurement is worth per hour, with the regret the
    budget removes stated on its own line - the number the operator is actually
    deciding on.
    """
    lines = [_rule("Measurement plan")]
    if not plan.items and not plan.uncovered and not plan.undecidable:
        lines.append("  nothing to measure: no uncertain input moves the recommendation")
        return "\n".join(lines)

    if plan.budget_hours is None:
        lines.append("  budget: unlimited")
    else:
        lines.append(
            f"  budget: {plan.budget_hours:,.3g} h, of which "
            f"{plan.total_hours:,.3g} h planned "
            f"({plan.exclusive_hours:,.3g} h with the device to itself)"
        )
    lines.append(
        f"  this budget removes {plan.covered_regret:,.4g} of decision regret "
        f"across {len(plan.items)} measurement(s)"
    )

    if plan.items:
        lines.append("")
        lines.append(
            f"  {'#':>3} {'input':<38} {'dR':>12} {'dR/h':>12} {'cost':>8}  measure by"
        )
        for item in plan.items[:top_n]:
            per_hour = "n/a" if item.regret_per_hour is None else f"{item.regret_per_hour:,.4g}"
            cost = "unknown" if item.cost_hours is None else f"{item.cost_hours:,.3g} h"
            flag = " FLIPS" if item.flip else ""
            approx = " (approx)" if item.approximation else ""
            if item.resimulated:
                approx = " (resimulated)"
            lines.append(
                f"  {item.rank:>3} {item.input_id:<38} {item.delta_regret:>12,.4g} "
                f"{per_hour:>12} {cost:>8}  {item.how_to_measure}{flag}{approx}"
            )
        if len(plan.items) > top_n:
            lines.append(f"      ... {len(plan.items) - top_n} more")

    if plan.uncovered:
        lines.append("")
        lines.append(
            f"  {len(plan.uncovered)} measurement(s) worth doing did not fit the budget, "
            f"starting at rank {plan.uncovered[0].rank}:"
        )
        for item in plan.uncovered[:5]:
            cost = "unknown" if item.cost_hours is None else f"{item.cost_hours:,.3g} h"
            lines.append(f"      {item.rank:>3} {item.input_id} ({cost})")
        if len(plan.uncovered) > 5:
            lines.append(f"      ... {len(plan.uncovered) - 5} more")

    if plan.undecidable:
        lines.append("")
        lines.append(
            f"  {len(plan.undecidable)} input(s) CANNOT BE DECIDED BEFORE MEASURING - "
            f"no sourced range, so no regret can be computed for them at all:"
        )
        for input_id in plan.undecidable[:8]:
            lines.append(f"      {input_id}")
        if len(plan.undecidable) > 8:
            lines.append(f"      ... {len(plan.undecidable) - 8} more")
    return "\n".join(lines)
