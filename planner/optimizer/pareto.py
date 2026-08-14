"""Pareto frontier and lexicographic ranking (work order §5.6).

Two separate jobs:

* **Ranking** picks the single recommendation. It is strictly lexicographic —
  primary objective first, then the fixed tie-break order. Weighted sums are
  forbidden by the work order, and for good reason: a weight silently encodes a
  trade-off the user never stated, and small weight changes reorder the result.
* **The frontier** supplies the alternatives. A user shown one plan cannot tell
  what they gave up; the non-dominated set makes the trade-offs explicit.
"""

from __future__ import annotations

from planner.plan import DeploymentPlan, ScoredPlan
from planner.spec import Objective

#: Objective dimensions, with the direction that counts as better.
#: (name, extractor, lower_is_better)
_DIMENSIONS: list[tuple[str, str, bool]] = [
    ("p99_ttft_ms", "p99_ttft_ms", True),
    ("p99_tpot_ms", "p99_tpot_ms", True),
    ("peak_power_w", "peak_power_w", True),
    ("tokens_per_joule", "tokens_per_joule", False),
    ("active_accelerators", "__devices__", True),
]


def _dimension_values(plan: DeploymentPlan) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, attr, _ in _DIMENSIONS:
        if attr == "__devices__":
            out[name] = float(plan.active_accelerators)
        else:
            value = getattr(plan.predicted, attr, None)
            out[name] = None if value is None else float(value)
    return out


def dominates(a: DeploymentPlan, b: DeploymentPlan) -> bool:
    """True when `a` is at least as good everywhere and strictly better somewhere.

    Dimensions missing from either plan (energy, when the config has no power
    block) are skipped rather than assumed. Comparing against an absent value
    would let a plan dominate purely by not having been measured.
    """
    va, vb = _dimension_values(a), _dimension_values(b)
    strictly_better = False
    compared = 0
    for name, _, lower_better in _DIMENSIONS:
        x, y = va[name], vb[name]
        if x is None or y is None:
            continue
        compared += 1
        if lower_better:
            if x > y:
                return False
            if x < y:
                strictly_better = True
        else:
            if x < y:
                return False
            if x > y:
                strictly_better = True
    return compared > 0 and strictly_better


def frontier(plans: list[DeploymentPlan]) -> list[DeploymentPlan]:
    """Non-dominated subset, order preserved."""
    out: list[DeploymentPlan] = []
    for plan in plans:
        if any(dominates(other, plan) for other in plans if other is not plan):
            continue
        out.append(plan)
    return out


def metric_signature(plan: DeploymentPlan) -> tuple:
    """Identity of a plan's *outcome*, ignoring how it was configured."""
    vals = _dimension_values(plan)
    out: list[float | None] = []
    for name, _, _ in _DIMENSIONS:
        value = vals[name]
        out.append(None if value is None else round(value, 6))
    return tuple(out)


def collapse_equivalent(plans: list[DeploymentPlan]) -> list[tuple[DeploymentPlan, list[str]]]:
    """Group plans whose predicted outcomes are identical.

    Knobs that do not bind produce byte-identical results - a 20-request trace
    never reaches 32 concurrent sequences, so max_num_seqs 32/128/256 all
    simulate the same. Listing them as three separate "alternatives" inflates
    the apparent trade-off space with choices that are not choices. The first
    plan of each group represents it; the rest are reported as equivalents.
    """
    groups: dict[tuple, list[DeploymentPlan]] = {}
    for plan in plans:
        groups.setdefault(metric_signature(plan), []).append(plan)
    return [
        (members[0], [p.candidate.id for p in members[1:]])
        for members in groups.values()
    ]


def can_score(plan: DeploymentPlan, objective: Objective) -> tuple[bool, str]:
    """Whether `plan` carries the metric this objective needs.

    A feasible plan that cannot be scored must not simply sort last: it would
    vanish from the output with no explanation, which is how an entire island
    can disappear from a search because its node had no `power:` block. Same
    failure class as an unmeasurable constraint reading as satisfied (D2).
    """
    m = plan.predicted
    if objective in (Objective.MINIMIZE_ENERGY, Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE):
        if m.total_energy_j is None:
            return False, (
                f"objective '{objective.value}' needs energy, but this plan has none - "
                f"the node running it has no power: block, so the simulator emitted no "
                f"energy (deviations D2)"
            )
        if m.total_energy_j <= 0:
            return False, f"objective '{objective.value}' needs energy > 0, got {m.total_energy_j}"
    return True, ""


def objective_value(plan: DeploymentPlan, objective: Objective) -> float:
    """Objective score. Always *maximised* by the caller, so minimisation
    objectives are negated here rather than at the comparison site."""
    m = plan.predicted
    if objective is Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE:
        if m.total_energy_j is None or m.total_energy_j <= 0:
            return float("-inf")
        # SLO-goodput/J: tokens of SLO-satisfying requests per joule (§4).
        return (m.completed_tokens * m.slo_attainment) / m.total_energy_j
    if objective is Objective.MINIMIZE_ENERGY:
        if m.total_energy_j is None:
            return float("-inf")
        return -m.total_energy_j
    if objective is Objective.MINIMIZE_ACTIVE_ACCELERATORS:
        return -float(plan.active_accelerators)
    raise ValueError(f"unhandled objective {objective}")


def _fragmentation(plan: DeploymentPlan) -> int:
    """How scattered a placement is: more islands touched is worse."""
    return len({a.island_id for a in plan.candidate.assignments})


def rank(
    plans: list[DeploymentPlan], primary: Objective, secondary: Objective | None = None
) -> list[ScoredPlan]:
    """Lexicographic ordering (§5.6 stage 2-3).

    Order: primary objective, then secondary if given, then the fixed tie-breaks
    — fewest active accelerators, least fragmentation, lowest reconfiguration
    cost (zero until Phase 6, so it contributes nothing yet and is represented
    by the stable sort). Candidate id breaks any remaining tie so the result is
    byte-identical across runs (§9).
    """

    def key(plan: DeploymentPlan) -> tuple:
        parts: list[float] = [-objective_value(plan, primary)]
        if secondary is not None:
            parts.append(-objective_value(plan, secondary))
        parts.append(float(plan.active_accelerators))
        parts.append(float(_fragmentation(plan)))
        return (*parts, plan.candidate.id)

    ordered = sorted(plans, key=key)
    return [
        ScoredPlan(plan=p, objective=primary, value=objective_value(p, primary))
        for p in ordered
    ]


def annotate_alternatives(
    recommended: DeploymentPlan, alternatives: list[ScoredPlan]
) -> list[ScoredPlan]:
    """Describe how each alternative differs from the recommendation.

    "lower latency, higher power" is what makes a Pareto set actionable; a bare
    list of plan ids is not.
    """
    out: list[ScoredPlan] = []
    base = _dimension_values(recommended)
    for alt in alternatives:
        vals = _dimension_values(alt.plan)
        notes: list[str] = []
        for name, _, lower_better in _DIMENSIONS:
            x, y = vals[name], base[name]
            if x is None or y is None or x == y:
                continue
            better = (x < y) if lower_better else (x > y)
            direction = "lower" if x < y else "higher"
            notes.append(f"{direction} {name}" + ("" if better else " (worse)"))
        out.append(alt.model_copy(update={"note": ", ".join(notes) or "equivalent"}))
    return out
