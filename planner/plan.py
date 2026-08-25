"""Candidate, plan and planner-output data structures (work order §3.4, §3.5, §5.4).

`CandidateConfig` is what the generator enumerates and the predictor evaluates.
`DeploymentPlan` is what the planner emits and a deployer consumes.
`PlannerOutput` wraps the recommendation, the Pareto alternatives, and — when
nothing is feasible — the diagnosis the work order requires instead of a bare
failure.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from planner.spec import Objective


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    AGGREGATED = "aggregated"

    @property
    def pd_type(self) -> str | None:
        """The simulator's per-instance `pd_type` value."""
        return None if self is Role.AGGREGATED else self.value


class ServingArch(str, enum.Enum):
    AGGREGATED = "aggregated"
    PD_SPLIT = "pd_split"


class RoutingPolicy(str, enum.Enum):
    SINGLE = "single"
    RR = "rr"
    LOAD = "load"
    PD_SPLIT = "pd_split"

    @property
    def sim_flag(self) -> str:
        """Simulator `--request-routing-policy`. RAND is never used: it would
        break the byte-identical reproducibility the work order §9 demands."""
        return {"single": "LOAD", "rr": "RR", "load": "LOAD", "pd_split": "LOAD"}[self.value]


class VllmKnobs(_Strict):
    """The runtime overrides the simulator accepts per instance.

    Field names match `configs/cluster/*.json` exactly so the compiler is a copy
    rather than a translation (docs/phase0_formats.md §4).

    `enable_prefix_caching` defaults to False for Phase 2: with it on, the
    simulator's prefix-cache memory grows monotonically until the run dies on any
    device small enough to saturate (docs/deviations.md D12). Turning it on is
    allowed but currently unsafe on tight candidates.
    """

    max_num_seqs: int = Field(default=128, ge=0)
    max_num_batched_tokens: int = Field(default=2048, ge=0)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    prioritize_prefill: bool = False
    block_size: int = Field(default=16, gt=0)
    kv_cache_dtype: str = "auto"
    #: vLLM `--max-model-len`. None = the model's own max (max_position_embeddings),
    #: which is also what the simulator uses (it has no separate max_model_len; it
    #: caps at max_position_embeddings). Set it to pin a deployment's context/KV
    #: budget; leave None to match the simulator's effective context.
    max_model_len: int | None = Field(default=None, gt=0)


class IslandAssignment(_Strict):
    """One island doing one job, at one parallelism degree, N times over."""

    island_id: str
    role: Role = Role.AGGREGATED
    tp_size: int = Field(ge=1)
    pp_size: int = Field(default=1, ge=1)
    dp_replicas: int = Field(default=1, ge=1)

    @property
    def devices_per_replica(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def total_devices(self) -> int:
        return self.devices_per_replica * self.dp_replicas


class CandidateConfig(_Strict):
    """One point in the search space (§5.4)."""

    id: str
    model: str
    dtype: str
    assignments: list[IslandAssignment] = Field(min_length=1)
    serving_arch: ServingArch = ServingArch.AGGREGATED
    knobs: VllmKnobs = Field(default_factory=VllmKnobs)

    @property
    def total_devices(self) -> int:
        return sum(a.total_devices for a in self.assignments)

    @property
    def islands(self) -> list[str]:
        return [a.island_id for a in self.assignments]

    def signature(self) -> tuple:
        """Order-independent identity, for dedup and cache keys."""
        return (
            self.model,
            self.dtype,
            tuple(sorted(
                (a.island_id, a.role.value, a.tp_size, a.pp_size, a.dp_replicas)
                for a in self.assignments
            )),
            self.serving_arch.value,
            tuple(sorted(self.knobs.model_dump().items())),
        )


class RejectionStage(str, enum.Enum):
    """Pruning stages, in the fixed order of §5.4. The value strings are the
    keys of `PlannerOutput.rejected_summary`."""

    BACKEND_INCOMPATIBLE = "backend_incompatible"
    MEMORY_INFEASIBLE = "memory_infeasible"
    PARALLELISM_INFEASIBLE = "parallelism_infeasible"
    TOPOLOGY_INFEASIBLE = "topology_infeasible"
    ANALYTICAL_LOWER_BOUND = "analytical_lower_bound"
    #: Stage-6 surrogate top-K (§5.4). Heuristic, NOT a sound bound: unlike the
    #: stages above it CAN drop the true optimum. That loss is *surrogate error*,
    #: measured against the exhaustive oracle (experiments/scripts/exp_surrogate.py),
    #: never a correctness bug. Do not "fix" an oracle-agreement test by folding
    #: this in - the "pruning must be a relaxation" rule is for the sound stages
    #: 4-5 only. Only appears when the caller opts in via top_k.
    SURROGATE_PRUNED = "surrogate_pruned"
    SLO_VIOLATED = "slo_violated"
    POWER_VIOLATED = "power_violated"
    EFFICIENCY_VIOLATED = "efficiency_violated"
    #: Simulator crashed or timed out. Never fold this into a feasibility
    #: bucket - a broken run is not a planning result (docs/deviations.md D12).
    SIM_ERROR = "sim_error"


class Rejection(_Strict):
    candidate_id: str
    stage: RejectionStage
    reason: str


class Violation(_Strict):
    metric: str
    target: float
    predicted: float

    @property
    def overshoot_ratio(self) -> float:
        """How badly missed, normalised so metrics can be compared."""
        if self.target == 0:
            return float("inf")
        return abs(self.predicted - self.target) / abs(self.target)


class PredictedMetrics(_Strict):
    """What the predictor returns for one candidate (§3.6 metric set)."""

    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float
    throughput_tps: float
    slo_goodput_rps: float
    slo_attainment: float
    completed_requests: int
    completed_tokens: int
    total_energy_j: float | None = None
    average_power_w: float | None = None
    peak_power_w: float | None = None
    tokens_per_joule: float | None = None
    sim_wall_seconds: float | None = None

    @property
    def has_energy(self) -> bool:
        return self.total_energy_j is not None


class DeploymentPlan(_Strict):
    """Planner output, deployer input (§3.4)."""

    plan_id: str
    model: str
    candidate: CandidateConfig
    predicted: PredictedMetrics
    routing: RoutingPolicy = RoutingPolicy.LOAD
    robust_margin_ttft_percent: float = 0.0
    robust_margin_tpot_percent: float = 0.0

    @property
    def active_accelerators(self) -> int:
        return self.candidate.total_devices


class ScoredPlan(_Strict):
    plan: DeploymentPlan
    objective: Objective
    value: float
    note: str = ""
    #: Candidate ids whose predicted outcome is byte-identical to this one.
    #: Knobs that never bind (a batch cap above the workload's concurrency)
    #: produce duplicates; collapsing them keeps the alternatives list honest
    #: about how many genuinely different outcomes exist.
    equivalent_candidates: list[str] = Field(default_factory=list)


class UnscoredPlan(_Strict):
    """Feasible, but missing the metric the primary objective ranks on.

    Kept visible rather than sorted to the bottom: a plan that silently vanishes
    looks the same as one that was never generated.
    """

    plan: DeploymentPlan
    reason: str


class PlannerOutput(_Strict):
    """§3.5. Feasible and infeasible share one type so callers cannot forget
    the diagnosis branch."""

    feasible: bool
    service_model: str
    cluster_id: str
    recommended: ScoredPlan | None = None
    alternatives: list[ScoredPlan] = Field(default_factory=list)
    unscored: list[UnscoredPlan] = Field(default_factory=list)
    rejected_summary: dict[str, int] = Field(default_factory=dict)
    evaluated_candidates: int = 0
    generated_candidates: int = 0

    # Infeasible branch
    reason: str = ""
    closest_plan: DeploymentPlan | None = None
    violated_constraints: list[Violation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    provenance: dict[str, Any] = Field(default_factory=dict)
    #: Caveats that must travel with the result, e.g. the Phase 2 prefix-cache
    #: restriction. Printed with the plan and written into the YAML.
    caveats: list[str] = Field(default_factory=list)

    @property
    def prune_ratio(self) -> float:
        if self.generated_candidates == 0:
            return 0.0
        return 1.0 - (self.evaluated_candidates / self.generated_candidates)


def summarize_rejections(rejections: list[Rejection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rejections:
        counts[r.stage.value] = counts.get(r.stage.value, 0) + 1
    return dict(sorted(counts.items()))
