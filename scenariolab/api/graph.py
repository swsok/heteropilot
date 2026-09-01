"""Cluster topology graph JSON for the UI (FR-A5), with plan overlay.

The graph is derived from the generated ClusterSpecV2 YAML, so what the UI
draws is exactly what the planner consumed. When a plan document is supplied,
the devices its island assignments occupy get `in_plan`/`role`, and links
with both endpoints in-plan are highlighted too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from scenariolab.api.schemas import ClusterGraph, GraphLink, GraphNode


def _plan_devices(
    document: dict[str, Any] | None, cluster: Any, root: Path
) -> dict[str, str]:
    """Map 'node/device' -> role for every device the plan occupies."""
    if not document:
        return {}
    output = document.get("planner_output", {})
    recommended = output.get("recommended")
    plan = (
        recommended["plan"] if recommended
        else output.get("closest_plan")  # infeasible: overlay the near-miss
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


def build_cluster_graph(
    cluster_yaml: str | Path,
    root: str | Path = ".",
    document: dict[str, Any] | None = None,
) -> ClusterGraph:
    root = Path(root)
    cluster = load_cluster_spec(cluster_yaml)
    occupied = _plan_devices(document, cluster, root)

    nodes: list[GraphNode] = []
    for node in cluster.nodes:
        for accel in node.accelerators:
            key = f"{node.id}/{accel.id}"
            nodes.append(
                GraphNode(
                    id=key,
                    node=node.id,
                    device=accel.id,
                    cls=accel.model,
                    state=accel.state.value,
                    kind="accelerator",
                    role=occupied.get(key),
                    in_plan=key in occupied,
                )
            )
        for nic in node.nics:
            nodes.append(
                GraphNode(
                    id=f"{node.id}/{nic.id}",
                    node=node.id,
                    device=nic.id,
                    cls=nic.type,
                    state="FREE",
                    kind="nic",
                )
            )

    links = [
        GraphLink(
            src=link.src,
            dst=link.dst,
            type=link.type.value,
            bandwidth_gbps=link.bandwidth_gbps,
            in_plan=link.src in occupied and link.dst in occupied,
        )
        for link in cluster.links
    ]
    return ClusterGraph(nodes=nodes, links=links)
