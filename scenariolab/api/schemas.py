"""Pydantic response models for the ScenarioLab web API (FR-A1).

Every numeric result travels with its honesty labels: `fidelity`,
`calibrated`, `npu_extrapolated` (FR-U1 starts here - the UI cannot show a
badge the API never sent).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class BatchInfo(BaseModel):
    batch_id: str
    config_hash: str
    master_seed: int
    status: str
    started_at: str | None
    finished_at: str | None
    scenario_counts: dict[str, int]
    done: int
    feasible: int
    feasible_rate: float | None
    median_power_saving_pct: float | None


class HistogramBin(BaseModel):
    label: str
    counts: dict[str, int]


class RateBin(BaseModel):
    label: str
    total: int
    feasible: int


class HeatmapCell(BaseModel):
    x_label: str
    y_label: str
    total: int
    feasible: int


class DashboardCharts(BaseModel):
    feasible_by_power_cap: list[RateBin]
    saving_histogram_by_fidelity: list[HistogramBin]
    heatmap_cluster_size_vs_tpot: list[HeatmapCell]


class SummaryResponse(BaseModel):
    batches: list[BatchInfo]
    selected_batch: str | None
    npu_extrapolated_count: int | None = None
    uncalibrated_count: int | None = None
    verification: VerificationStats | None = None
    charts: DashboardCharts | None = None


class ScenarioRow(BaseModel):
    scenario_id: str
    batch_id: str
    cluster_id: str
    service_id: str
    status: str
    feasible: bool
    fidelity: str
    calibrated: bool
    npu_extrapolated: bool
    p99_ttft_ms: float | None
    p99_tpot_ms: float | None
    avg_power_w: float | None
    peak_power_w: float | None
    tokens_per_joule: float | None
    slo_goodput: float | None
    active_devices: int | None
    baseline_power_w: float | None
    power_saving_pct: float | None
    baseline_note: str | None
    has_npu: bool


class ScenarioListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    rows: list[ScenarioRow]


class GraphNode(BaseModel):
    id: str
    node: str
    device: str
    cls: str
    state: str
    kind: Literal["accelerator", "nic"]
    #: GPU | NPU for accelerators (drives the collapsed per-node counts).
    accel_type: str | None = None
    role: str | None = None
    in_plan: bool = False


class GraphLink(BaseModel):
    src: str
    dst: str
    type: str
    bandwidth_gbps: float
    in_plan: bool = False


class ClusterGraph(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]


class ClusterInfo(BaseModel):
    cluster_id: str
    seed: int
    yaml_path: str
    num_nodes: int
    num_accels: int
    num_free_accels: int
    classes: list[str]
    num_islands: int
    has_npu: bool
    origin: str = "random"
    link_summary: str | None = None
    workspaces: int = 0


class ClusterDetail(ClusterInfo):
    graph: ClusterGraph


class ServiceInfo(BaseModel):
    service_id: str
    seed: int
    yaml_path: str
    model: str
    rps: float
    ttft_p99_ms: float
    tpot_p99_ms: float
    power_cap_w: float


class VerificationRecord(BaseModel):
    scenario_id: str
    sim_p99_ttft_ms: float | None
    sim_p99_tpot_ms: float | None
    sim_avg_power_w: float | None
    err_ttft_pct: float | None
    err_tpot_pct: float | None
    err_power_pct: float | None
    selection_flipped: bool | None
    feasibility_flipped: bool | None
    regret_energy_pct: float | None


class VerificationStats(BaseModel):
    verified: int
    selection_flips: int
    feasibility_flips: int
    err_ttft_pct_p50: float | None
    err_ttft_pct_p95: float | None
    err_tpot_pct_p50: float | None
    err_tpot_pct_p95: float | None
    err_power_pct_p50: float | None
    err_power_pct_p95: float | None


class VerificationPoint(BaseModel):
    scenario_id: str
    fast_p99_ttft_ms: float | None
    fast_p99_tpot_ms: float | None
    fast_avg_power_w: float | None
    sim_p99_ttft_ms: float | None
    sim_p99_tpot_ms: float | None
    sim_avg_power_w: float | None
    selection_flipped: bool | None
    feasibility_flipped: bool | None
    fidelity: str


class VerificationResponse(BaseModel):
    batch_id: str
    stats: VerificationStats
    points: list[VerificationPoint]
    flipped: list[str]


class ScenarioDetail(BaseModel):
    row: ScenarioRow
    service: ServiceInfo
    cluster: ClusterInfo
    graph: ClusterGraph
    #: The full result document (planner_output + baseline + calibration),
    #: verbatim from the plan JSON file.
    document: dict[str, Any]
    verification: VerificationRecord | None


class PlanSloRequest(BaseModel):
    """Interactive SLO body (§9.2). rps/input_p50/output_p50 are the traffic -
    without them the request is rejected with 400 (FR-A3)."""

    model: str = "meta-llama/Llama-3.1-8B"
    dtype: str = "bfloat16"
    rps: float | None = None
    input_p50: int | None = None
    output_p50: int | None = None
    ttft_p99_ms: float
    tpot_p99_ms: float
    power_cap_w: float | None = None


class PlanRequest(BaseModel):
    cluster_id: str
    slo: PlanSloRequest


class PlanResponse(BaseModel):
    cluster_id: str
    feasible: bool
    #: FR-A2 honesty block, always present.
    fidelity: str
    calibrated: bool
    npu_extrapolated: bool
    truncated: bool
    elapsed_s: float
    seed: int
    num_requests: int
    calibration: dict[str, Any]
    planner_output: dict[str, Any]
    graph: ClusterGraph


# -- workspace mode (workspace work order §6.1/§7) ---------------------------

class IslandInfo(BaseModel):
    id: str
    accelerators: int
    model: str
    tp_candidates: list[int]


class BuildClusterResponse(BaseModel):
    cluster: ClusterInfo
    warnings: list[str]
    islands: list[IslandInfo]
    already_existed: bool


class WorkspaceCreateRequest(BaseModel):
    cluster_id: str
    name: str
    total_power_cap_w: float | None = None


class WorkspaceInfo(BaseModel):
    workspace_id: str
    cluster_id: str
    name: str
    created_at: str
    status: str
    total_power_cap_w: float | None
    placed_count: int = 0
    #: FR-CAT3: set when the cluster YAML changed after this workspace was made.
    cluster_changed: bool = False


class PlacementSloRequest(BaseModel):
    """Body of POST /placements: either slo='random' (+count/seed) or a
    user-typed SLO object (same fields as /api/plan)."""

    slo: str | PlanSloRequest
    count: int = 1
    seed: int = 42
    #: confirm=True places immediately (CLI --yes analogue; also what the
    #: random-count path uses, since each placement must land before the next
    #: is planned). confirm=False returns a PLANNING preview (FR-W4).
    confirm: bool = False


class PlacementRow(BaseModel):
    placement_id: str
    workspace_id: str
    service_id: str
    seq: int
    status: str
    devices: list[str]
    fidelity: str | None
    calibrated: bool | None
    slo_ttft_ok: bool | None
    slo_tpot_ok: bool | None
    p99_ttft_ms: float | None
    p99_tpot_ms: float | None
    avg_power_w: float | None
    peak_power_w: float | None
    tokens_per_joule: float | None
    shared_fabric_warning: bool | None
    npu_extrapolated: bool | None
    rejected_reason: dict[str, Any] | None
    service: dict[str, Any]
    created_at: str
    removed_at: str | None


class PlacementResponse(BaseModel):
    placements: list[PlacementRow]
    #: Full planner outputs, index-aligned with `placements` (preview detail).
    results: list[dict[str, Any]]


class WorkspaceResources(BaseModel):
    total_accels: int
    free_accels: int
    by_class: dict[str, dict[str, int]]


class WorkspacePower(BaseModel):
    #: Sum of PLACED avg predictions.
    sum_avg_w: float
    #: Sum of PLACED peaks - a conservative upper bound that assumes
    #: simultaneous peaks (FR-W3); labelled as such in the UI.
    sum_peak_w: float
    total_power_cap_w: float | None


class WorkspaceSummaryResponse(BaseModel):
    workspace: WorkspaceInfo
    cluster: ClusterInfo
    resources: WorkspaceResources
    power: WorkspacePower
    placements: list[PlacementRow]
    graph: ClusterGraph
    #: device id -> {service seq (stable color index), role}
    topology_overlay: dict[str, dict[str, Any]]
    #: Standing notice (work order §9): every prediction assumes sole use of
    #: its devices; inter-service interference is not modelled.
    interference_notice: str


SummaryResponse.model_rebuild()
