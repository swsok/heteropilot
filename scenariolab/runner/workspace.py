"""F3 Incremental placement engine (workspace work order §5).

A workspace owns a device-occupancy OVERLAY over an immutable cluster YAML:
placements that are PLACED mark their devices ALLOCATED in an in-memory copy,
and the next service is planned by the unchanged planner pipeline, which
already restricts islands to state == FREE. No planner modification, no
side effects on the YAML or on other workspaces (FR-P1).

Honesty rules carried through (work order §0.1): devices are exclusively
occupied (no co-location - the simulator does not model device sharing);
inter-service interference is NOT modelled - every prediction assumes sole
use of its devices, and overlapping contention groups only raise
`shared_fabric_warning`, never a fabricated adjusted number.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from planner.inventory import (
    AcceleratorState,
    ClusterSpecV2,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.spec import ServiceSpec, load_service_spec
from planner.topology import TopologyError, TopologyGraph
from scenariolab.config import SloGeneratorConfig
from scenariolab.generator.sampling import derive_seed
from scenariolab.generator.slo_gen import ServiceSummary, generate_service
from scenariolab.runner.interactive import build_service_spec, plan_fast
from scenariolab.store.db import PLANNING, REJECTED, ResultStore, StoreError

#: Random workspace services sample from the same distributions as the
#: default batch config - one place, so the two modes stay comparable.
DEFAULT_WORKSPACE_SLO = SloGeneratorConfig.model_validate({
    "num_specs": 1,
    "models": ["meta-llama/Llama-3.1-8B"],
    "arrival_rate_rps": {"dist": "loguniform", "min": 0.5, "max": 30},
    "input_tokens_p50": {"dist": "choice", "values": [256, 512, 1024, 4096]},
    "output_tokens_p50": {"dist": "choice", "values": [64, 128, 512]},
    "ttft_p99_ms": {"dist": "loguniform", "min": 200, "max": 5000},
    "tpot_p99_ms": {"dist": "loguniform", "min": 30, "max": 300},
    "power_cap_w": {"dist": "uniform", "min": 400, "max": 4000},
    "min_tokens_per_joule": {"dist": "fixed", "value": 0.0},
    "objective": {
        "primary": "minimize_energy",
        "secondary": "minimize_active_accelerators",
    },
})


def cluster_overlay(cluster: ClusterSpecV2, occupied: set[str]) -> ClusterSpecV2:
    """A deep copy with occupied devices marked ALLOCATED (work order §5.1)."""
    copy = cluster.model_copy(deep=True)
    for node in copy.nodes:
        for accel in node.accelerators:
            if f"{node.id}/{accel.id}" in occupied:
                accel.state = AcceleratorState.ALLOCATED
    return copy


def plan_devices_of(
    planner_output: dict[str, Any],
    cluster: ClusterSpecV2,
    root: str | Path = ".",
) -> dict[str, str]:
    """Map 'node/device' -> role for the devices a plan occupies.

    Islands are re-detected on the SAME cluster view the plan was computed
    against, so island ids and their deterministic accelerator order match
    the assignments exactly.
    """
    recommended = planner_output.get("recommended")
    plan = (
        recommended["plan"] if recommended
        else planner_output.get("closest_plan")
    )
    if not plan:
        return {}
    profiles = load_profiles_for(cluster, root)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    occupied: dict[str, str] = {}
    for assignment in plan["candidate"]["assignments"]:
        island = islands.get(assignment["island_id"])
        if island is None:
            continue
        used = assignment["tp_size"] * assignment["pp_size"] * assignment["dp_replicas"]
        for accel_id in island.accelerator_ids[:used]:
            occupied[f"{island.node_id}/{accel_id}"] = assignment["role"]
    return occupied


def plan_contention_groups(cluster: ClusterSpecV2, devices: set[str]) -> set[str]:
    """Contention groups a placement's traffic can touch (§5.4): the groups of
    every link on shortest paths between its device pairs (TP collectives,
    CPU bridges, NIC fabric for multi-node placements)."""
    if len(devices) < 2:
        # A single device still occupies its root complex.
        groups: set[str] = set()
        for link in cluster.links:
            if link.contention_group and (link.src in devices or link.dst in devices):
                groups.add(link.contention_group)
        return groups
    topology = TopologyGraph(cluster)
    groups = set()
    for src, dst in itertools.combinations(sorted(devices), 2):
        try:
            path = topology.path(src, dst)
        except TopologyError:
            continue
        groups.update(
            link.contention_group for link in path if link.contention_group
        )
    return groups


def random_workspace_service(
    workspace_id: str,
    index: int,
    seed: int,
    services_dir: Path,
) -> ServiceSummary:
    """One random service for a workspace, on the DESIGN §3.2 seed chain.

    FR-P7: (seed, index) fully determines the SLO, so the same seed replays
    the same sequence regardless of workspace history. The seed is part of
    the service id so different seeds never overwrite each other's specs."""
    return generate_service(
        DEFAULT_WORKSPACE_SLO,
        index,
        derive_seed(seed, "wslo", workspace_id, index),
        services_dir,
        "workspace",
        service_id=f"{workspace_id}-s{seed}-r{index:04d}",
    )


def save_user_service(
    workspace_id: str,
    seq: int,
    slo: dict[str, Any],
    services_dir: Path,
) -> tuple[ServiceSummary, ServiceSpec]:
    """Persist a user-typed SLO as a ServiceSpec YAML (work order §5.2: both
    input paths get a service_id for reproducibility and F4 display)."""
    spec = build_service_spec(slo)
    service_id = f"{workspace_id}-u{seq:04d}"
    services_dir.mkdir(parents=True, exist_ok=True)
    path = services_dir / f"{service_id}.yaml"
    path.write_text(
        "# generated_by: scenariolab-workspace (user-typed SLO)\n"
        + yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True),
                         sort_keys=False)
    )
    summary = ServiceSummary(
        service_id=service_id,
        seed=0,
        yaml_path=path,
        model=spec.model,
        rps=spec.traffic.arrival_rate_rps,
        input_p50=spec.traffic.input_tokens.p50,
        output_p50=spec.traffic.output_tokens.p50,
        ttft_p99_ms=spec.slo.ttft.max_ms,
        tpot_p99_ms=spec.slo.tpot.max_ms,
        power_cap_w=spec.slo.max_cluster_power_w or 0.0,
    )
    return summary, spec


def evaluate_placement(
    cluster: ClusterSpecV2,
    occupied: set[str],
    spec: ServiceSpec,
    *,
    root: str | Path = ".",
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
    total_power_cap_w: float | None = None,
    placed_peak_sum_w: float = 0.0,
) -> dict[str, Any]:
    """Pure evaluation of one service against an occupancy state (§5.1).

    Shared by incremental placement and replan-all; persists nothing.
    Returns {"feasible", "record", "devices_roles", "result"}.
    """
    root = Path(root)
    overlay = cluster_overlay(cluster, occupied)
    result = plan_fast(
        spec, overlay,
        root=root, envelope_dir=envelope_dir, calibration_dir=calibration_dir,
    )
    output = result["planner_output"]
    cal = result["calibration"]

    record: dict[str, Any] = {
        "fidelity": result["fidelity"],
        "calibrated": result["calibrated"],
        "npu_extrapolated": result["npu_extrapolated"],
    }

    # Workspace-level power cap (FR-P3): accumulated PLACED peaks + this peak.
    cap_violation = None
    plan = output.get("recommended")
    if plan is not None and total_power_cap_w is not None:
        peak = plan["plan"]["predicted"].get("peak_power_w")
        if peak is not None and placed_peak_sum_w + peak > total_power_cap_w:
            cap_violation = {
                "metric": "workspace_power_cap",
                "target": total_power_cap_w,
                "predicted": placed_peak_sum_w + peak,
            }

    if not output["feasible"] or cap_violation is not None:
        record["rejected_reason"] = {
            "reason": (
                "workspace power cap exceeded" if cap_violation
                else output.get("reason", "")
            ),
            "violated_constraints": (
                [cap_violation] if cap_violation
                else output.get("violated_constraints", [])
            ),
            "suggestions": output.get("suggestions", []),
            # FR-P5: which devices were unavailable to this plan.
            "occupied_devices": sorted(occupied),
        }
        return {
            "feasible": False, "record": record,
            "devices_roles": {}, "result": result,
        }

    devices_roles = plan_devices_of(output, overlay, root)
    metrics = plan["plan"]["predicted"]

    # SLO verdicts on ROBUST values: predicted * (1 + margin%). With no
    # calibration coverage the margins are 0 and the verdict is raw - the
    # calibrated flag says which one the user is looking at (§2.2).
    robust_ttft = metrics["p99_ttft_ms"] * (1 + cal["ttft_margin_percent"] / 100)
    robust_tpot = metrics["p99_tpot_ms"] * (1 + cal["tpot_margin_percent"] / 100)
    shared = plan_contention_groups(cluster, set(devices_roles)) & (
        plan_contention_groups(cluster, occupied)
    )
    record.update(
        slo_ttft_ok=robust_ttft <= spec.slo.ttft.max_ms,
        slo_tpot_ok=robust_tpot <= spec.slo.tpot.max_ms,
        p99_ttft_ms=metrics["p99_ttft_ms"],
        p99_tpot_ms=metrics["p99_tpot_ms"],
        avg_power_w=metrics.get("average_power_w"),
        peak_power_w=metrics.get("peak_power_w"),
        tokens_per_joule=metrics.get("tokens_per_joule"),
        shared_fabric_warning=bool(shared),
        shared_groups=sorted(shared),
    )
    return {
        "feasible": True, "record": record,
        "devices_roles": devices_roles, "result": result,
    }


def place_service(
    store: ResultStore,
    workspace_id: str,
    service_id: str,
    spec: ServiceSpec,
    *,
    root: str | Path = ".",
    results_dir: str | Path,
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
    confirm: bool = False,
) -> dict[str, Any]:
    """Plan one service on the workspace's remaining FREE devices (§5.1).

    feasible  -> a PLANNING preview row (or PLACED when confirm=True)
    infeasible -> a REJECTED row; the overlay is untouched
    Returns the placement record including the full planner output.
    """
    workspace = store.get_workspace(workspace_id)
    if workspace is None:
        raise KeyError(f"no workspace '{workspace_id}'")
    if workspace["status"] != "ACTIVE":
        raise StoreError(
            f"workspace '{workspace_id}' is {workspace['status']}; "
            "archived workspaces accept no new placements"
        )
    cluster_row = store.get_cluster(workspace["cluster_id"])
    assert cluster_row is not None
    root = Path(root)
    cluster = load_cluster_spec(root / cluster_row["yaml_path"])

    occupied = store.placed_devices(workspace_id)
    placed_peak = sum(
        row["peak_power_w"] or 0.0
        for row in store.workspace_placements(workspace_id, include_removed=False)
        if row["status"] == "PLACED"
    )
    evaluation = evaluate_placement(
        cluster, occupied, spec,
        root=root, envelope_dir=envelope_dir, calibration_dir=calibration_dir,
        total_power_cap_w=workspace["total_power_cap_w"],
        placed_peak_sum_w=placed_peak,
    )
    record = evaluation["record"]
    result = evaluation["result"]

    if not evaluation["feasible"]:
        placement_id = store.insert_placement(
            workspace_id, service_id, status=REJECTED, devices=[], record=record,
        )
        document_path = _write_document(
            results_dir, workspace_id, placement_id, service_id, result
        )
        store.set_placement_plan_path(placement_id, str(document_path))
        return {
            "placement_id": placement_id, "status": REJECTED,
            "devices": [], "record": record, "result": result,
        }

    devices_roles = evaluation["devices_roles"]
    placement_id = store.insert_placement(
        workspace_id, service_id, status=PLANNING, devices=devices_roles,
        record=record,
    )
    document_path = _write_document(
        results_dir, workspace_id, placement_id, service_id, result
    )
    store.set_placement_plan_path(placement_id, str(document_path))
    if confirm:
        store.confirm_placement(placement_id)
    return {
        "placement_id": placement_id,
        "status": "PLACED" if confirm else PLANNING,
        "devices": sorted(devices_roles),
        "devices_roles": devices_roles,
        "record": record,
        "result": result,
    }


#: The replan disclaimer, stated wherever replan results travel (work order
#: §5.5/§9: three places - code, API response, docs).
REPLAN_NOTE = (
    "replan-all is a sequential re-run of the SAME greedy per-service "
    "placement in a chosen order; it is NOT a joint multi-service "
    "optimization (that is out of scope by the parent work order)."
)


def replan_workspace(
    store: ResultStore,
    workspace_id: str,
    *,
    order: str = "seq",
    apply: bool = False,
    root: str | Path = ".",
    results_dir: str | Path,
    envelope_dir: str | Path | None = None,
    calibration_dir: str | Path | None = "profiles/calibration",
) -> dict[str, Any]:
    """§5.5 replan-all: re-run sequential placement of the current PLACED
    services from an empty overlay. Preview by default; `apply=True` swaps
    the whole placement set atomically."""
    if order not in ("seq", "rps_desc"):
        raise ValueError("order must be 'seq' or 'rps_desc'")
    workspace = store.get_workspace(workspace_id)
    if workspace is None:
        raise KeyError(f"no workspace '{workspace_id}'")
    cluster_row = store.get_cluster(workspace["cluster_id"])
    assert cluster_row is not None
    root = Path(root)
    cluster = load_cluster_spec(root / cluster_row["yaml_path"])

    placed = [
        row for row in store.workspace_placements(workspace_id, include_removed=False)
        if row["status"] == "PLACED"
    ]
    if order == "rps_desc":
        placed = sorted(placed, key=lambda r: (-r["rps"], r["seq"]))

    occupied: set[str] = set()
    peak_sum = 0.0
    entries: list[dict[str, Any]] = []
    for row in placed:
        service_row = store.get_service(row["service_id"])
        assert service_row is not None
        spec = load_service_spec(root / service_row["yaml_path"])
        evaluation = evaluate_placement(
            cluster, occupied, spec,
            root=root, envelope_dir=envelope_dir, calibration_dir=calibration_dir,
            total_power_cap_w=workspace["total_power_cap_w"],
            placed_peak_sum_w=peak_sum,
        )
        if evaluation["feasible"]:
            occupied |= set(evaluation["devices_roles"])
            peak_sum += evaluation["record"].get("peak_power_w") or 0.0
        entries.append({
            "service_id": row["service_id"],
            "previous_placement_id": row["placement_id"],
            "status": "PLACED" if evaluation["feasible"] else REJECTED,
            "devices_roles": evaluation["devices_roles"],
            "record": evaluation["record"],
            "result": evaluation["result"],
        })

    response: dict[str, Any] = {
        "order": order,
        "applied": False,
        "note": REPLAN_NOTE,
        "entries": entries,
    }
    if not apply:
        return response

    placement_ids = store.replace_all_placements(
        workspace_id,
        [
            (e["service_id"], e["status"],
             e["devices_roles"] if e["status"] == "PLACED" else [], e["record"])
            for e in entries
        ],
    )
    for entry, placement_id in zip(entries, placement_ids, strict=True):
        entry["placement_id"] = placement_id
        path = _write_document(
            results_dir, workspace_id, placement_id, entry["service_id"],
            entry["result"],
        )
        store.set_placement_plan_path(placement_id, str(path))
    response["applied"] = True
    return response


def _write_document(
    results_dir: str | Path,
    workspace_id: str,
    placement_id: str,
    service_id: str,
    result: dict[str, Any],
) -> Path:
    out = Path(results_dir) / "workspaces" / workspace_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{placement_id}.json"
    path.write_text(json.dumps(
        {
            "placement_id": placement_id,
            "workspace_id": workspace_id,
            "service_id": service_id,
            **result,
        },
        indent=2, sort_keys=True,
    ))
    return path
