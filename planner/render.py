"""Human-readable rendering of a PlannerOutput (work order §6).

The `plan` command's stdout must always carry: feasible candidate count and top
list, rejected counts by stage with reasons, the recommendation, the Pareto
alternatives, and the predicted metrics. An infeasible result must diagnose
rather than just say no.
"""

from __future__ import annotations

from planner.plan import DeploymentPlan, PlannerOutput, ScoredPlan

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
    return "\n".join(out)


def render(output: PlannerOutput, *, top_n: int = 5) -> str:
    lines: list[str] = []
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

    if output.caveats:
        lines.append("")
        lines.append(_rule("Caveats"))
        for c in dict.fromkeys(output.caveats):
            lines.append(f"  - {c}")

    return "\n".join(lines)
