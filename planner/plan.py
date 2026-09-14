"""Candidate, plan and planner-output data structures (work order §3.4, §3.5, §5.4).

`CandidateConfig` is what the generator enumerates and the predictor evaluates.
`DeploymentPlan` is what the planner emits and a deployer consumes.
`PlannerOutput` wraps the recommendation, the Pareto alternatives, and — when
nothing is feasible — the diagnosis the work order requires instead of a bare
failure.
"""

from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from planner.spec import Objective
from planner.uncertainty.measurement_plan import MeasurementPlan
from planner.uncertainty.registry import UncertainInputRegistry


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
    #: Which ASTRA-Sim topology encoding this placement needs (deviations D28).
    #: "auto" for everything the pre-D28 path could express; "slab3d" only for
    #: asymmetric P/D (`tp_d == 2 * tp_p`), which `auto` cannot represent at all.
    #: Deliberately NOT the simulator's third mode: `split2` is a calibration
    #: instrument and a plan must never name it (tests/test_slab3d_config.py).
    topology_mode: Literal["auto", "slab3d"] = "auto"

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
            self.topology_mode,
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
    #: The candidate is representable but a value it needs sits outside a
    #: measured calibration domain (D28/D31, A6): either it was never simulated
    #: because the slab3d latency table does not cover it, or it WAS simulated and
    #: its operating point lies outside the hardware's accuracy domain under
    #: `refuse` -- or the hardware has no calibration at all -- so no margin
    #: exists and no verdict is possible (uncertainty work order A3, D33). This is
    #: an EPISTEMIC refusal, not a failure: "we do not know" rather than "it does
    #: not work". It gets its own bucket for the same reason SIM_ERROR does --
    #: folding it into a feasibility stage would report an unmeasured
    #: configuration as an infeasible one. STEP 4's envelope rejection joins this
    #: category.
    OUTSIDE_CALIBRATION_DOMAIN = "outside_calibration_domain"
    #: The candidate's PREDICTED operating point falls outside the hardware's
    #: measured performance envelope, and the envelope's policy is `refuse`
    #: (STEP 4.4, A6). Like the stage above it is epistemic and, unlike stages
    #: 4-5, NOT a relaxation of feasibility -- it can drop the true optimum, and
    #: when it does that is a RESULT ("the optimum is outside what was measured"),
    #: which the oracle-agreement test reports rather than treats as a bug.
    #: Opt-in: hardware without an envelope is never touched by it.
    OUTSIDE_MEASURED_ENVELOPE = "outside_measured_envelope"
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
    #: The BUSIEST instance's mean in-flight requests over the run,
    #: `sum(latency) / wall` per simulator instance (uncertainty work order
    #: §2.4.1 rev 2): a domain is measured on one card, so a four-replica
    #: candidate holding 200 requests runs each card at 50, and the headline
    #: figure is the instance whose error decides the verdict. The per-hardware,
    #: per-phase operating points a margin is actually read at live on
    #: `SimResult.operating_point`. Filled by the real predictor from the
    #: per-request CSV; None for predictors that have no per-request records.
    #: `_write_output` drops the key unless the caller opted into the
    #: accuracy-domain machinery, so default plans are unchanged (rule A4).
    served_concurrency: float | None = None
    #: Per ISLAND, from the per-request CSV's instance attribution; the busiest
    #: instance of each island. None when the predictor has no per-request
    #: records.
    served_concurrency_per_island: dict[str, float] | None = None

    @property
    def has_energy(self) -> bool:
        return self.total_energy_j is not None


class OperatingPointRecord(_Strict):
    """Where one hardware kind actually ran, and what margin that earned it."""

    hardware: str
    concurrency: float
    phase: str
    tpot_error_pct: float | None = None
    ttft_error_pct: float | None = None
    in_calibration_domain: bool | None = None


class DeploymentPlan(_Strict):
    """Planner output, deployer input (§3.4)."""

    plan_id: str
    model: str
    candidate: CandidateConfig
    predicted: PredictedMetrics
    routing: RoutingPolicy = RoutingPolicy.LOAD
    robust_margin_ttft_percent: float = 0.0
    robust_margin_tpot_percent: float = 0.0
    #: The served concurrency each hardware kind ran at, and the accuracy-domain
    #: error read off it (STEP 4.3). Empty when no accuracy domain applied, which
    #: is the default path and keeps the frozen output unchanged.
    operating_point: list[OperatingPointRecord] = Field(default_factory=list)
    #: "manual" | "accuracy_domain" | "" -- which of the two produced the margin
    #: actually applied. Both are recorded in provenance; this says which bound.
    margin_source: str = ""
    #: How those two margins were arrived at - which accuracy domain, and
    #: between which measured points (uncertainty work order A3). None (not "")
    #: so `_write_output` can drop the key entirely and leave default plans
    #: byte-identical (rule A4).
    margin_basis: str | None = None

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

    #: Weakest profile tier among the islands of the reported plans
    #: (planner/util/tier.py, work order tiered-profiles STEP 2). Anything but
    #: measured/imported means the numbers rest on non-measured inputs and the
    #: renderer shows a banner. Lives here rather than on DeploymentPlan: every
    #: plan of one output shares the same island->tier map, the renderer and
    #: the YAML writer both consume PlannerOutput, and per-plan copies would
    #: only duplicate profile_tiers.
    profile_tier: str = "unknown"
    #: island_id -> tier for every island the search saw.
    profile_tiers: dict[str, str] = Field(default_factory=dict)

    #: Every planner input that is not a measurement, with its sourced error
    #: range (WORK_ORDER_uncertainty_planner.md §2.3). None unless the caller
    #: opted in via `--accuracy-domain`; `_write_output` drops the key entirely
    #: in that case, so the default path's YAML is byte-identical (rule A4).
    uncertain_inputs: UncertainInputRegistry | None = None

    #: What to measure next and what it buys (§2.5, STEP B3). None unless the
    #: caller asked for it with `--measurement-plan`; dropped from the YAML in
    #: that case, so the default path is unchanged (rule A4).
    measurement_plan: MeasurementPlan | None = None

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
