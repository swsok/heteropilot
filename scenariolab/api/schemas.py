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


SummaryResponse.model_rebuild()
