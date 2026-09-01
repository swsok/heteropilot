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

from planner.util.percentile import percentile
from scenariolab.api.graph import build_cluster_graph
from scenariolab.api.schemas import (
    BatchInfo,
    ClusterDetail,
    ClusterGraph,
    ClusterInfo,
    DashboardCharts,
    HeatmapCell,
    HistogramBin,
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
)
from scenariolab.store.db import ResultStore, StoreError

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(
    db_path: str | Path,
    root: str | Path = ".",
    *,
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
) -> FastAPI:
    app = FastAPI(
        title="ScenarioLab",
        description="Random-scenario power-optimal placement results (read-only), "
        "plus the interactive fast-path planner.",
        version="0.2.0",
    )
    root = Path(root)

    def store() -> Iterator[ResultStore]:
        try:
            reader = ResultStore(db_path, readonly=True)
        except StoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            yield reader
        finally:
            reader.close()

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
    def clusters(db: ResultStore = Depends(store)) -> list[ClusterInfo]:
        return [_cluster_info(r) for r in db.list_clusters()]

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
