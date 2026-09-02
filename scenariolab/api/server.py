"""M6 FastAPI app: read-only serving of the ResultStore (DESIGN §9).

A lab-internal tool with NO auth: the CLI binds 0.0.0.0 by default so other
lab machines can reach it - do not expose it beyond the lab network. The DB
is opened read-only per request (FR-A6); the interactive /api/plan endpoint
arrives with P4 and is declared here so /docs shows the intended surface.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from planner.spec import load_service_spec
from planner.util import provenance as prov
from planner.util.percentile import percentile
from scenariolab.api.graph import build_cluster_graph
from scenariolab.api.schemas import (
    BatchInfo,
    BuildClusterResponse,
    ClusterDetail,
    ClusterGraph,
    ClusterInfo,
    DashboardCharts,
    HeatmapCell,
    HistogramBin,
    IslandInfo,
    PlacementResponse,
    PlacementRow,
    PlacementSloRequest,
    PlanRequest,
    PlanResponse,
    RateBin,
    ScenarioDetail,
    ScenarioListResponse,
    ScenarioRow,
    ServiceInfo,
    SummaryResponse,
    VerificationPoint,
    VerificationRecord,
    VerificationResponse,
    VerificationStats,
    WorkspaceCreateRequest,
    WorkspaceInfo,
    WorkspacePower,
    WorkspaceResources,
    WorkspaceSummaryResponse,
)
from scenariolab.config import LabConfigError
from scenariolab.generator.cluster_builder import ClusterBuildRequest, build_cluster
from scenariolab.generator.cluster_gen import ClusterGenError
from scenariolab.runner.interactive import InteractivePlanError
from scenariolab.runner.workspace import (
    place_service,
    random_workspace_service,
    replan_workspace,
    save_user_service,
)
from scenariolab.store.db import ResultStore, StoreError, devices_from_json

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class _RowShim(dict):
    """Dict with sqlite3.Row's keys() shape, so helpers accept either."""

    def keys(self):
        return list(super().keys())


def create_app(
    db_path: str | Path,
    root: str | Path = ".",
    *,
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
    clusters_dir: str | Path | None = None,
    services_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="ScenarioLab",
        description="Random-scenario power-optimal placement results (read-only), "
        "plus the interactive fast-path planner.",
        version="0.2.0",
    )
    root = Path(root)
    base = Path(db_path).parent
    clusters_dir = Path(clusters_dir) if clusters_dir else base / "clusters"
    services_dir = Path(services_dir) if services_dir else base / "services"
    results_dir = Path(results_dir) if results_dir else base / "results"

    def store() -> Iterator[ResultStore]:
        try:
            reader = ResultStore(db_path, readonly=True)
        except StoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            yield reader
        finally:
            reader.close()

    def write_store() -> Iterator[ResultStore]:
        """Workspace endpoints mutate state, so FR-A6 is amended for them (work
        order §7): writes go through this dedicated store only; every read
        endpoint keeps the mode=ro connection."""
        writer = ResultStore(db_path)
        try:
            yield writer
        finally:
            writer.close()

    # -- helpers ------------------------------------------------------------ #

    def _scenario_row(row: Any) -> ScenarioRow:
        return ScenarioRow(
            scenario_id=row["scenario_id"],
            batch_id=row["batch_id"],
            cluster_id=row["cluster_id"],
            service_id=row["service_id"],
            status=row["status"],
            feasible=bool(row["feasible"]),
            fidelity=row["fidelity"],
            calibrated=bool(row["calibrated"]),
            npu_extrapolated=bool(row["npu_extrapolated"]),
            p99_ttft_ms=row["p99_ttft_ms"],
            p99_tpot_ms=row["p99_tpot_ms"],
            avg_power_w=row["avg_power_w"],
            peak_power_w=row["peak_power_w"],
            tokens_per_joule=row["tokens_per_joule"],
            slo_goodput=row["slo_goodput"],
            active_devices=row["active_devices"],
            baseline_power_w=row["baseline_power_w"],
            power_saving_pct=row["power_saving_pct"],
            baseline_note=row["baseline_note"],
            has_npu=bool(row["has_npu"]),
        )

    def _cluster_info(row: Any) -> ClusterInfo:
        return ClusterInfo(
            cluster_id=row["cluster_id"],
            seed=row["seed"],
            yaml_path=row["yaml_path"],
            num_nodes=row["num_nodes"],
            num_accels=row["num_accels"],
            num_free_accels=row["num_free_accels"],
            classes=json.loads(row["classes_json"]),
            num_islands=row["num_islands"],
            has_npu=bool(row["has_npu"]),
            origin=row["origin"],
            link_summary=row["link_summary"],
        )

    def _service_info(row: Any) -> ServiceInfo:
        return ServiceInfo(
            service_id=row["service_id"],
            seed=row["seed"],
            yaml_path=row["yaml_path"],
            model=row["model"],
            rps=row["rps"],
            ttft_p99_ms=row["ttft_p99_ms"],
            tpot_p99_ms=row["tpot_p99_ms"],
            power_cap_w=row["power_cap_w"],
        )

    def _verification_stats(rows: list[Any]) -> VerificationStats:
        stats: dict[str, Any] = {
            "verified": len(rows),
            "selection_flips": sum(1 for r in rows if r["selection_flipped"]),
            "feasibility_flips": sum(1 for r in rows if r["feasibility_flipped"]),
        }
        for metric in ("err_ttft_pct", "err_tpot_pct", "err_power_pct"):
            values = [abs(r[metric]) for r in rows if r[metric] is not None]
            stats[f"{metric}_p50"] = (
                round(percentile(values, 50), 3) if values else None
            )
            stats[f"{metric}_p95"] = (
                round(percentile(values, 95), 3) if values else None
            )
        return VerificationStats(**stats)

    def _charts(rows: list[Any]) -> DashboardCharts:
        # Chart A: feasible rate by power-cap band.
        cap_edges = [0.0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, math.inf]
        cap_pairs = list(itertools.pairwise(cap_edges))
        cap_bins = [
            RateBin(
                label=f"{int(lo)}-{'∞' if hi == math.inf else int(hi)}W",
                total=0, feasible=0,
            )
            for lo, hi in cap_pairs
        ]
        # Chart B: power-saving histogram, one series per fidelity.
        saving_pairs = list(itertools.pairwise(range(-20, 71, 10)))
        saving_bins = [
            HistogramBin(label=f"{lo}..{hi}%", counts={}) for lo, hi in saving_pairs
        ]
        # Chart C: feasible-rate heatmap, cluster size x TPOT strictness.
        heat: dict[tuple[str, str], HeatmapCell] = {}

        def size_label(n: int) -> str:
            return "1-2" if n <= 2 else ("3-6" if n <= 6 else "7+")

        def tpot_label(ms: float) -> str:
            if ms < 60:
                return "<60ms"
            return "<150ms" if ms < 150 else ">=150ms"

        for row in rows:
            cap = row["power_cap_w"]
            for (lo, hi), rate_bin in zip(cap_pairs, cap_bins, strict=True):
                if lo <= cap < hi:
                    rate_bin.total += 1
                    rate_bin.feasible += int(row["feasible"])
                    break
            saving = row["power_saving_pct"]
            if saving is not None:
                for (lo, hi), hist in zip(saving_pairs, saving_bins, strict=True):
                    if lo <= saving < hi:
                        fid = row["fidelity"]
                        hist.counts[fid] = hist.counts.get(fid, 0) + 1
                        break
            key = (size_label(row["num_accels"]), tpot_label(row["tpot_p99_ms"]))
            cell = heat.setdefault(
                key, HeatmapCell(x_label=key[0], y_label=key[1], total=0, feasible=0)
            )
            cell.total += 1
            cell.feasible += int(row["feasible"])

        return DashboardCharts(
            feasible_by_power_cap=[b for b in cap_bins if b.total],
            saving_histogram_by_fidelity=saving_bins,
            heatmap_cluster_size_vs_tpot=sorted(
                heat.values(), key=lambda c: (c.x_label, c.y_label)
            ),
        )

    # -- endpoints ------------------------------------------------------------ #

    @app.get("/api/summary", response_model=SummaryResponse)
    def summary(
        batch_id: str | None = None, db: ResultStore = Depends(store)
    ) -> SummaryResponse:
        batches = []
        for row in db.list_batches():
            agg = db.batch_summary(row["batch_id"])
            batches.append(
                BatchInfo(
                    batch_id=row["batch_id"],
                    config_hash=row["config_hash"],
                    master_seed=row["master_seed"],
                    status=row["status"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    scenario_counts=agg["counts"],
                    done=agg["done"],
                    feasible=agg["feasible"],
                    feasible_rate=agg["feasible_rate"],
                    median_power_saving_pct=agg["median_power_saving_pct"],
                )
            )
        selected = batch_id or (batches[-1].batch_id if batches else None)
        response = SummaryResponse(batches=batches, selected_batch=selected)
        if selected is not None:
            rows = db.dashboard_rows(selected)
            response.npu_extrapolated_count = sum(
                1 for r in rows if r["npu_extrapolated"]
            )
            response.uncalibrated_count = sum(1 for r in rows if not r["calibrated"])
            response.charts = _charts(rows)
            response.verification = _verification_stats(db.verification_rows(selected))
        return response

    @app.get("/api/scenarios", response_model=ScenarioListResponse)
    def scenarios(
        batch_id: str | None = None,
        feasible: bool | None = None,
        fidelity: str | None = None,
        has_npu: bool | None = None,
        min_saving: float | None = None,
        cluster_id: str | None = None,
        service_id: str | None = None,
        sort: str = "scenario_id",
        descending: bool = False,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=500),
        db: ResultStore = Depends(store),
    ) -> ScenarioListResponse:
        filters: dict[str, Any] = {
            "feasible": feasible,
            "fidelity": fidelity,
            "has_npu": has_npu,
            "min_saving_pct": min_saving,
            "cluster_id": cluster_id,
            "service_id": service_id,
        }
        try:
            rows = db.query_results(
                batch_id, order_by=sort, descending=descending,
                page=page, page_size=page_size, **filters,
            )
        except StoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        total = db.count_results(batch_id, **filters)
        return ScenarioListResponse(
            total=total, page=page, page_size=page_size,
            rows=[_scenario_row(r) for r in rows],
        )

    @app.get("/api/scenarios/{scenario_id}", response_model=ScenarioDetail)
    def scenario_detail(
        scenario_id: str, db: ResultStore = Depends(store)
    ) -> ScenarioDetail:
        row = db.get_scenario(scenario_id)
        if row is None or row["feasible"] is None:
            raise HTTPException(status_code=404, detail=f"no result for '{scenario_id}'")
        cluster_row = db.get_cluster(row["cluster_id"])
        service_row = db.get_service(row["service_id"])
        assert cluster_row is not None and service_row is not None
        document = json.loads((root / row["plan_json_path"]).read_text())
        graph = build_cluster_graph(
            root / cluster_row["yaml_path"], root, document=document
        )
        verification_row = db.get_verification(scenario_id)
        verification = (
            VerificationRecord(
                scenario_id=scenario_id,
                sim_p99_ttft_ms=verification_row["sim_p99_ttft_ms"],
                sim_p99_tpot_ms=verification_row["sim_p99_tpot_ms"],
                sim_avg_power_w=verification_row["sim_avg_power_w"],
                err_ttft_pct=verification_row["err_ttft_pct"],
                err_tpot_pct=verification_row["err_tpot_pct"],
                err_power_pct=verification_row["err_power_pct"],
                selection_flipped=_opt_bool(verification_row["selection_flipped"]),
                feasibility_flipped=_opt_bool(verification_row["feasibility_flipped"]),
                regret_energy_pct=verification_row["regret_energy_pct"],
            )
            if verification_row is not None else None
        )
        # The has_npu column lives on clusters; get_scenario doesn't join it.
        merged = dict(row)
        merged["has_npu"] = cluster_row["has_npu"]
        return ScenarioDetail(
            row=_scenario_row(merged),
            service=_service_info(service_row),
            cluster=_cluster_info(cluster_row),
            graph=graph,
            document=document,
            verification=verification,
        )

    @app.get("/api/clusters", response_model=list[ClusterInfo])
    def clusters(
        origin: str | None = None, db: ResultStore = Depends(store)
    ) -> list[ClusterInfo]:
        rows = db.list_clusters()
        if origin in ("random", "custom"):
            rows = [r for r in rows if r["origin"] == origin]
        out = []
        for row in rows:
            info = _cluster_info(row)
            info.workspaces = db.workspace_count_for_cluster(row["cluster_id"])
            out.append(info)
        return out

    @app.get("/api/clusters/{cluster_id}", response_model=ClusterDetail)
    def cluster_detail(
        cluster_id: str, db: ResultStore = Depends(store)
    ) -> ClusterDetail:
        row = db.get_cluster(cluster_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no cluster '{cluster_id}'")
        info = _cluster_info(row)
        graph: ClusterGraph = build_cluster_graph(root / row["yaml_path"], root)
        return ClusterDetail(**info.model_dump(), graph=graph)

    @app.get("/api/services", response_model=list[ServiceInfo])
    def services(db: ResultStore = Depends(store)) -> list[ServiceInfo]:
        return [_service_info(r) for r in db.list_services()]

    @app.get("/api/services/{service_id}", response_model=ServiceInfo)
    def service_detail(
        service_id: str, db: ResultStore = Depends(store)
    ) -> ServiceInfo:
        row = db.get_service(service_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no service '{service_id}'")
        return _service_info(row)

    @app.get("/api/verification", response_model=VerificationResponse)
    def verification(
        batch_id: str, db: ResultStore = Depends(store)
    ) -> VerificationResponse:
        rows = db.verification_rows(batch_id)
        points = [
            VerificationPoint(
                scenario_id=r["scenario_id"],
                fast_p99_ttft_ms=r["fast_p99_ttft_ms"],
                fast_p99_tpot_ms=r["fast_p99_tpot_ms"],
                fast_avg_power_w=r["fast_avg_power_w"],
                sim_p99_ttft_ms=r["sim_p99_ttft_ms"],
                sim_p99_tpot_ms=r["sim_p99_tpot_ms"],
                sim_avg_power_w=r["sim_avg_power_w"],
                selection_flipped=_opt_bool(r["selection_flipped"]),
                feasibility_flipped=_opt_bool(r["feasibility_flipped"]),
                fidelity=r["fidelity"],
            )
            for r in rows
        ]
        return VerificationResponse(
            batch_id=batch_id,
            stats=_verification_stats(rows),
            points=points,
            flipped=[p.scenario_id for p in points if p.selection_flipped],
        )

    @app.post("/api/plan", response_model=PlanResponse)
    def plan(request: PlanRequest, db: ResultStore = Depends(store)) -> PlanResponse:
        from scenariolab.runner.interactive import (
            InteractivePlanError,
            plan_interactive,
        )

        cluster_row = db.get_cluster(request.cluster_id)
        if cluster_row is None:
            raise HTTPException(
                status_code=404, detail=f"no cluster '{request.cluster_id}'"
            )
        slo = request.slo.model_dump()
        try:
            result = plan_interactive(
                root / cluster_row["yaml_path"], slo,
                root=root,
                envelope_dir=envelope_dir,
                calibration_dir=calibration_dir,
            )
        except InteractivePlanError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Query history is the single thing the web layer writes (FR-A6),
        # through a dedicated short-lived read-write store.
        try:
            with ResultStore(db_path) as writer:
                writer.record_plan_query(
                    cluster_id=request.cluster_id,
                    slo=slo,
                    seed=result["seed"],
                    num_requests=result["num_requests"],
                    feasible=result["feasible"],
                    fidelity=result["fidelity"],
                    truncated=result["truncated"],
                    elapsed_s=result["elapsed_s"],
                )
        except StoreError:
            pass  # history must never break the answer itself

        graph = build_cluster_graph(
            root / cluster_row["yaml_path"], root,
            document={"planner_output": result["planner_output"]},
        )
        return PlanResponse(
            cluster_id=request.cluster_id,
            feasible=result["feasible"],
            fidelity=result["fidelity"],
            calibrated=result["calibrated"],
            npu_extrapolated=result["npu_extrapolated"],
            truncated=result["truncated"],
            elapsed_s=result["elapsed_s"],
            seed=result["seed"],
            num_requests=result["num_requests"],
            calibration=result["calibration"],
            planner_output=result["planner_output"],
            graph=graph,
        )

    # -- workspace mode (workspace work order §7) ----------------------------- #

    INTERFERENCE_NOTICE = (
        "Each service's prediction assumes SOLE use of its devices; "
        "inter-service interference is not modelled. Overlapping fabric is "
        "flagged (shared_fabric_warning), never quantified."
    )

    def _workspace_info(row: Any, db: ResultStore) -> WorkspaceInfo:
        changed = False
        if row["cluster_yaml_hash"]:
            cluster_row = db.get_cluster(row["cluster_id"])
            if cluster_row is not None:
                current = prov.hash_file(root / cluster_row["yaml_path"])
                changed = current is not None and current != row["cluster_yaml_hash"]
        keys = row.keys()
        return WorkspaceInfo(
            workspace_id=row["workspace_id"],
            cluster_id=row["cluster_id"],
            name=row["name"],
            created_at=row["created_at"],
            status=row["status"],
            total_power_cap_w=row["total_power_cap_w"],
            placed_count=row["placed_count"] if "placed_count" in keys else 0,
            cluster_changed=changed,
        )

    def _placement_row(row: Any) -> PlacementRow:
        return PlacementRow(
            placement_id=row["placement_id"],
            workspace_id=row["workspace_id"],
            service_id=row["service_id"],
            seq=row["seq"],
            status=row["status"],
            devices=sorted(devices_from_json(row["devices_json"])),
            fidelity=row["fidelity"],
            calibrated=_opt_bool(row["calibrated"]),
            slo_ttft_ok=_opt_bool(row["slo_ttft_ok"]),
            slo_tpot_ok=_opt_bool(row["slo_tpot_ok"]),
            p99_ttft_ms=row["p99_ttft_ms"],
            p99_tpot_ms=row["p99_tpot_ms"],
            avg_power_w=row["avg_power_w"],
            peak_power_w=row["peak_power_w"],
            tokens_per_joule=row["tokens_per_joule"],
            shared_fabric_warning=_opt_bool(row["shared_fabric_warning"]),
            npu_extrapolated=_opt_bool(row["npu_extrapolated"]),
            rejected_reason=(
                json.loads(row["rejected_reason_json"])
                if row["rejected_reason_json"] else None
            ),
            service={
                "model": row["model"],
                "rps": row["rps"],
                "ttft_p99_ms": row["slo_ttft_ms"],
                "tpot_p99_ms": row["slo_tpot_ms"],
                "power_cap_w": row["power_cap_w"],
                "origin": row["service_origin"],
            },
            created_at=row["created_at"],
            removed_at=row["removed_at"],
        )

    @app.post("/api/clusters/build", response_model=BuildClusterResponse)
    def clusters_build(
        request: ClusterBuildRequest, db: ResultStore = Depends(write_store)
    ) -> BuildClusterResponse:
        try:
            summary, warnings, islands = build_cluster(request, clusters_dir, root)
        except (LabConfigError, ClusterGenError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        existed = db.get_cluster(summary.cluster_id) is not None
        db.upsert_cluster(summary)  # idempotent: same request -> same id (FR-CB4)
        row = db.get_cluster(summary.cluster_id)
        assert row is not None
        return BuildClusterResponse(
            cluster=_cluster_info(row),
            warnings=warnings,
            islands=[IslandInfo(**i) for i in islands],
            already_existed=existed,
        )

    @app.post("/api/workspaces", response_model=WorkspaceInfo)
    def workspaces_create(
        request: WorkspaceCreateRequest, db: ResultStore = Depends(write_store)
    ) -> WorkspaceInfo:
        cluster_row = db.get_cluster(request.cluster_id)
        if cluster_row is None:
            raise HTTPException(
                status_code=404, detail=f"no cluster '{request.cluster_id}'"
            )
        workspace_id = db.create_workspace(
            request.cluster_id, request.name,
            cluster_yaml_hash=prov.hash_file(root / cluster_row["yaml_path"]),
            total_power_cap_w=request.total_power_cap_w,
        )
        row = db.get_workspace(workspace_id)
        assert row is not None
        return _workspace_info(row, db)

    @app.get("/api/workspaces", response_model=list[WorkspaceInfo])
    def workspaces_list(db: ResultStore = Depends(store)) -> list[WorkspaceInfo]:
        return [_workspace_info(row, db) for row in db.list_workspaces()]

    @app.get("/api/workspaces/{workspace_id}", response_model=WorkspaceSummaryResponse)
    @app.get(
        "/api/workspaces/{workspace_id}/summary",
        response_model=WorkspaceSummaryResponse,
    )
    def workspace_summary(
        workspace_id: str, db: ResultStore = Depends(store)
    ) -> WorkspaceSummaryResponse:
        row = db.get_workspace(workspace_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no workspace '{workspace_id}'")
        cluster_row = db.get_cluster(row["cluster_id"])
        assert cluster_row is not None
        placements = db.workspace_placements(workspace_id)
        placed = [p for p in placements if p["status"] == "PLACED"]

        from planner.inventory import load_cluster_spec

        cluster = load_cluster_spec(root / cluster_row["yaml_path"])
        by_class: dict[str, dict[str, int]] = {}
        occupied = db.placed_devices(workspace_id)
        for node in cluster.nodes:
            for accel in node.accelerators:
                entry = by_class.setdefault(accel.model, {"total": 0, "free": 0})
                entry["total"] += 1
                if f"{node.id}/{accel.id}" not in occupied:
                    entry["free"] += 1
        total_accels = sum(v["total"] for v in by_class.values())

        overlay: dict[str, dict[str, Any]] = {}
        for placement in placed:
            stored = json.loads(placement["devices_json"])
            roles = stored if isinstance(stored, dict) else dict.fromkeys(stored)
            for device, role in roles.items():
                overlay[device] = {
                    "placement_id": placement["placement_id"],
                    "service_id": placement["service_id"],
                    # FR-W2: color index = seq, stable across add/remove.
                    "color_index": placement["seq"],
                    "role": role,
                }

        graph = build_cluster_graph(root / cluster_row["yaml_path"], root)
        merged = _RowShim(dict(row))
        merged["placed_count"] = len(placed)

        return WorkspaceSummaryResponse(
            workspace=_workspace_info(merged, db),
            cluster=_cluster_info(cluster_row),
            resources=WorkspaceResources(
                total_accels=total_accels,
                free_accels=total_accels - len(occupied),
                by_class=by_class,
            ),
            power=WorkspacePower(
                sum_avg_w=sum(p["avg_power_w"] or 0.0 for p in placed),
                sum_peak_w=sum(p["peak_power_w"] or 0.0 for p in placed),
                total_power_cap_w=row["total_power_cap_w"],
            ),
            placements=[_placement_row(p) for p in placements],
            graph=graph,
            topology_overlay=overlay,
            interference_notice=INTERFERENCE_NOTICE,
        )

    @app.post(
        "/api/workspaces/{workspace_id}/placements", response_model=PlacementResponse
    )
    def placements_create(
        workspace_id: str,
        request: PlacementSloRequest,
        db: ResultStore = Depends(write_store),
    ) -> PlacementResponse:
        workspace = db.get_workspace(workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"no workspace '{workspace_id}'")

        jobs: list[tuple[str, Any]] = []  # (service_id, spec)
        if request.slo == "random":
            # Index runs 0..count-1 for THIS request: (seed, index) fully
            # determines the SLO sequence (FR-P7), independent of history.
            for k in range(request.count):
                summary = random_workspace_service(
                    workspace_id, k, request.seed, services_dir
                )
                db.upsert_service(summary, origin="random")
                jobs.append((summary.service_id, load_service_spec(summary.yaml_path)))
        elif isinstance(request.slo, str):
            raise HTTPException(
                status_code=400, detail="slo must be 'random' or an SLO object"
            )
        else:
            seq = len(db.workspace_placements(workspace_id)) + 1
            try:
                summary, spec = save_user_service(
                    workspace_id, seq, request.slo.model_dump(), services_dir
                )
            except InteractivePlanError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            db.upsert_service(summary, origin="user")
            jobs.append((summary.service_id, spec))

        # count > 1 forces sequential confirmation: each placement must land
        # before the next one can see the devices it took.
        confirm = request.confirm or (request.slo == "random" and request.count > 1)
        rows: list[PlacementRow] = []
        results: list[dict[str, Any]] = []
        for service_id, spec in jobs:
            try:
                outcome = place_service(
                    db, workspace_id, service_id, spec,
                    root=root, results_dir=results_dir,
                    envelope_dir=envelope_dir, calibration_dir=calibration_dir,
                    confirm=confirm,
                )
            except StoreError as exc:  # e.g. ARCHIVED workspace
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            placement = db.workspace_placements(workspace_id)
            row = next(
                p for p in placement
                if p["placement_id"] == outcome["placement_id"]
            )
            rows.append(_placement_row(row))
            results.append(outcome["result"])
        return PlacementResponse(placements=rows, results=results)

    @app.post("/api/workspaces/{workspace_id}/replan")
    def workspace_replan(
        workspace_id: str,
        order: str = "seq",
        apply: bool = False,
        db: ResultStore = Depends(write_store),
    ) -> dict[str, Any]:
        """§5.5 replan-all. Preview by default; apply=true swaps atomically.
        NOT a joint optimization - see `note` in the response."""
        if db.get_workspace(workspace_id) is None:
            raise HTTPException(status_code=404, detail=f"no workspace '{workspace_id}'")
        try:
            outcome = replan_workspace(
                db, workspace_id, order=order, apply=apply,
                root=root, results_dir=results_dir,
                envelope_dir=envelope_dir, calibration_dir=calibration_dir,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Strip the bulky planner outputs from the wire response; the applied
        # rows carry their documents on disk like any placement.
        for entry in outcome["entries"]:
            entry.pop("result", None)
        return outcome

    @app.post("/api/workspaces/{workspace_id}/archive")
    def workspace_archive(
        workspace_id: str, db: ResultStore = Depends(write_store)
    ) -> dict[str, str]:
        try:
            db.archive_workspace(workspace_id)
        except StoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"workspace_id": workspace_id, "status": "ARCHIVED"}

    @app.get("/api/workspaces/{workspace_id}/placements/{placement_id}")
    def placement_detail(
        workspace_id: str,
        placement_id: str,
        db: ResultStore = Depends(store),
    ) -> dict[str, Any]:
        """Row + full fast-path document, so the UI can reuse the scenario
        detail rendering for a placement (FR-W5)."""
        rows = db.workspace_placements(workspace_id)
        row = next((p for p in rows if p["placement_id"] == placement_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no placement '{placement_id}'")
        document: dict[str, Any] = {}
        if row["plan_json_path"]:
            document = json.loads(Path(row["plan_json_path"]).read_text())
        return {
            "placement": _placement_row(row).model_dump(),
            "document": document,
        }

    @app.post(
        "/api/workspaces/{workspace_id}/placements/{placement_id}/confirm",
        response_model=PlacementRow,
    )
    def placements_confirm(
        workspace_id: str,
        placement_id: str,
        db: ResultStore = Depends(write_store),
    ) -> PlacementRow:
        try:
            db.confirm_placement(placement_id)
        except StoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        row = next(
            p for p in db.workspace_placements(workspace_id)
            if p["placement_id"] == placement_id
        )
        return _placement_row(row)

    @app.delete("/api/workspaces/{workspace_id}/placements/{placement_id}")
    def placements_delete(
        workspace_id: str,
        placement_id: str,
        db: ResultStore = Depends(write_store),
    ) -> dict[str, str]:
        try:
            db.remove_placement(placement_id)
        except StoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"placement_id": placement_id, "status": "REMOVED"}

    @app.get("/api/batches/{batch_id}/progress")
    def progress(batch_id: str, db: ResultStore = Depends(store)) -> dict[str, Any]:
        counts = db.scenario_counts(batch_id)
        total = sum(counts.values())
        done = sum(
            counts.get(s, 0) for s in ("DONE_FEASIBLE", "DONE_INFEASIBLE", "VERIFIED")
        )
        return {"batch_id": batch_id, "total": total, "done": done, "counts": counts}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _opt_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
