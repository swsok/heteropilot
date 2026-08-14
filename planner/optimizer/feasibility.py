"""Hard-constraint checking (work order §5.6).

Every check returns `(passed, Violation | None)` so the same code both filters
candidates and builds the infeasible diagnosis - §3.5 requires the planner to
explain *how far off* the closest plan was, which is impossible if failures are
reduced to a boolean on the way through.

Checks run against **robust** values: `predicted * (1 + p95_error)` per §5.8.
Until Phase 4 calibration supplies a margin it is zero and robust equals
predicted, but the plumbing is here so turning it on later changes one number
rather than the control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from planner.plan import DeploymentPlan, PredictedMetrics, RejectionStage, Violation
from planner.spec import ServiceSpec


@dataclass
class FeasibilityReport:
    passed: bool
    violations: list[Violation] = field(default_factory=list)
    #: The stage a candidate should be charged to when it fails.
    stage: RejectionStage | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def worst_overshoot(self) -> float:
        """Largest normalised miss, for ranking near-misses in the diagnosis."""
        if not self.violations:
            return 0.0
        return max(v.overshoot_ratio for v in self.violations)


def _percentile_value(metrics: PredictedMetrics, kind: str, percentile: int) -> float:
    key = f"p{percentile}_{kind}_ms"
    value = getattr(metrics, key, None)
    if value is None:
        raise ValueError(f"metrics carry no {key}; cannot check the {kind.upper()} SLO")
    return float(value)


def check_latency(
    metrics: PredictedMetrics,
    spec: ServiceSpec,
    *,
    ttft_margin_percent: float = 0.0,
    tpot_margin_percent: float = 0.0,
) -> list[Violation]:
    out: list[Violation] = []

    ttft = _percentile_value(metrics, "ttft", spec.slo.ttft.percentile)
    robust_ttft = ttft * (1.0 + ttft_margin_percent / 100.0)
    if robust_ttft > spec.slo.ttft.max_ms:
        out.append(
            Violation(
                metric=f"p{spec.slo.ttft.percentile}_ttft_ms",
                target=spec.slo.ttft.max_ms,
                predicted=robust_ttft,
            )
        )

    tpot = _percentile_value(metrics, "tpot", spec.slo.tpot.percentile)
    robust_tpot = tpot * (1.0 + tpot_margin_percent / 100.0)
    if robust_tpot > spec.slo.tpot.max_ms:
        out.append(
            Violation(
                metric=f"p{spec.slo.tpot.percentile}_tpot_ms",
                target=spec.slo.tpot.max_ms,
                predicted=robust_tpot,
            )
        )
    return out


def check_power(metrics: PredictedMetrics, spec: ServiceSpec) -> tuple[list[Violation], list[str]]:
    """Power cap and energy-efficiency floor.

    When the cluster config has no `power:` block the simulator emits no energy
    at all (docs/deviations.md D2). An unmeasurable constraint is reported as a
    note, never silently treated as satisfied.
    """
    violations: list[Violation] = []
    notes: list[str] = []

    if spec.slo.max_cluster_power_w is not None:
        if metrics.peak_power_w is None:
            notes.append(
                "slo.max_cluster_power_w is set but the simulation produced no power "
                "output; add a power: block to the cluster spec or drop the constraint"
            )
        elif metrics.peak_power_w > spec.slo.max_cluster_power_w:
            violations.append(
                Violation(
                    metric="peak_power_w",
                    target=spec.slo.max_cluster_power_w,
                    predicted=metrics.peak_power_w,
                )
            )

    if spec.slo.min_tokens_per_joule is not None:
        if metrics.tokens_per_joule is None:
            notes.append(
                "slo.min_tokens_per_joule is set but the simulation produced no energy "
                "output; the constraint could not be checked"
            )
        elif metrics.tokens_per_joule < spec.slo.min_tokens_per_joule:
            violations.append(
                Violation(
                    metric="tokens_per_joule",
                    target=spec.slo.min_tokens_per_joule,
                    predicted=metrics.tokens_per_joule,
                )
            )
    return violations, notes


def evaluate(
    plan: DeploymentPlan,
    spec: ServiceSpec,
    *,
    ttft_margin_percent: float = 0.0,
    tpot_margin_percent: float = 0.0,
) -> FeasibilityReport:
    """Full hard-constraint pass over a simulated plan.

    Resource and compatibility constraints (`RequiredDevices <= FreeDevices`,
    `BackendCompatible`, `ModelSupported`, `TopologyCompatible`) are enforced
    earlier, during candidate generation - a candidate that reaches simulation
    has already satisfied them by construction.
    """
    metrics = plan.predicted
    violations = check_latency(
        metrics,
        spec,
        ttft_margin_percent=ttft_margin_percent,
        tpot_margin_percent=tpot_margin_percent,
    )
    stage = RejectionStage.SLO_VIOLATED if violations else None

    power_violations, notes = check_power(metrics, spec)
    if power_violations and stage is None:
        stage = (
            RejectionStage.POWER_VIOLATED
            if any(v.metric == "peak_power_w" for v in power_violations)
            else RejectionStage.EFFICIENCY_VIOLATED
        )
    violations.extend(power_violations)

    return FeasibilityReport(
        passed=not violations, violations=violations, stage=stage, notes=notes
    )
