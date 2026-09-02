"""F1 ClusterBuilder: user-defined clusters (workspace work order §3).

Input is a ClusterBuildRequest; output is an ordinary ClusterSpecV2 YAML - no
new cluster schema. The node/link construction is the SAME code the random
generator uses (`cluster_gen._make_node`, topology v2), so a custom cluster
gets the identical host-structured PCIe model, contention groups and
connectivity invariant.

Provenance (work order §0.1-2): a user-typed inter-node bandwidth is labelled
`source: user_defined` - it is a deliberate what-if input, never a
measurement. Preset links inherit their profile's own source label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from planner.inventory import (
    AcceleratorProfile,
    ClusterSpecV2,
    Link,
    LinkType,
    Source,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from scenariolab.config import LabConfigError, profile_path_for
from scenariolab.generator.cluster_gen import (
    CLASS_FACTS,
    ClusterGenError,
    ClusterSummary,
    LinkProfile,
    _fully_connected,
    _load_profile,
    _make_node,
    describe_fabric,
    load_link_profile,
)

#: Fast-path responsiveness guard (work order §2.1): bigger studies belong in
#: batch mode, not the interactive builder.
MAX_TOTAL_ACCELERATORS = 64

#: Above this, warn that the value exceeds any shipping fabric (never reject:
#: sweeps beyond current hardware are a legitimate what-if).
FABRIC_SANITY_GBPS = 1600

#: Inter-node link types a user may pick for a custom fabric.
INTER_NODE_TYPES = ("NVLINK", "PCIE", "INFINIBAND", "ETHERNET", "HCCS")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeGroup(_Strict):
    """One homogeneous node configuration: `num_nodes` nodes of `class`."""

    cls: str = Field(alias="class")
    count_per_node: int = Field(ge=1)
    num_nodes: int = Field(ge=1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CustomLink(_Strict):
    type: str
    bandwidth_gbps: float = Field(gt=0)
    latency_ns: float = Field(ge=0)

    @model_validator(mode="after")
    def _type_allowed(self) -> CustomLink:
        if self.type not in INTER_NODE_TYPES:
            raise ValueError(
                f"inter-node link type must be one of {INTER_NODE_TYPES}, "
                f"got '{self.type}'"
            )
        return self


class InterNode(_Strict):
    preset: str | None = None
    custom: CustomLink | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> InterNode:
        if (self.preset is None) == (self.custom is None):
            raise ValueError("interconnect.inter_node needs exactly one of preset/custom")
        return self


class Interconnect(_Strict):
    #: Always 'auto': the intra-node interconnect is a hardware fact dictated
    #: by the accelerator profile (FR-CB2), never a user choice.
    intra_node: str = "auto"
    inter_node: InterNode

    @model_validator(mode="after")
    def _intra_auto(self) -> Interconnect:
        if self.intra_node != "auto":
            raise ValueError(
                "interconnect.intra_node must be 'auto' - the intra-node fabric "
                "is dictated by the accelerator profile (FR-CB2)"
            )
        return self


class ClusterBuildRequest(_Strict):
    name: str = Field(min_length=1, max_length=48)
    nodes: list[NodeGroup] = Field(min_length=1)
    interconnect: Interconnect
    initial_state: str = "FREE"

    @model_validator(mode="after")
    def _free_only(self) -> ClusterBuildRequest:
        if self.initial_state != "FREE":
            raise ValueError(
                "initial_state must be FREE (partial occupancy is out of F1's scope)"
            )
        return self

    def request_hash(self) -> str:
        payload = json.dumps(self.model_dump(by_alias=True), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def cluster_id(self) -> str:
        slug = "".join(c if c.isalnum() or c == "-" else "-" for c in self.name.lower())
        return f"custom-{slug}-{self.request_hash()}"


def _validate_classes(request: ClusterBuildRequest, root: Path) -> dict[str, AcceleratorProfile]:
    profiles: dict[str, AcceleratorProfile] = {}
    for group in request.nodes:
        entry = group.cls
        if entry not in CLASS_FACTS:
            raise LabConfigError(
                f"unknown accelerator class '{entry}'; available: "
                f"{sorted(CLASS_FACTS)}"
            )
        path = root / profile_path_for(entry)
        profile = _load_profile(entry, root)
        if profile.source == Source.PLACEHOLDER:
            usable = [
                c for c in CLASS_FACTS
                if _load_profile(c, root).source != Source.PLACEHOLDER
            ]
            raise LabConfigError(
                f"class '{entry}' ({path}) has source=placeholder and cannot be "
                f"used; available classes: {sorted(usable)}"
            )
        cap = CLASS_FACTS[entry]["max_per_node"]
        if group.count_per_node > cap:
            raise LabConfigError(
                f"class '{entry}': count_per_node={group.count_per_node} exceeds "
                f"the per-node cap {cap}"
            )
        profiles[entry] = profile
    return profiles


def _inter_node_profile(
    inter: InterNode, root: Path, warnings: list[str]
) -> LinkProfile:
    if inter.preset is not None:
        profile = load_link_profile(inter.preset, root)
        if profile.scope != "internode":
            raise LabConfigError(
                f"link preset '{inter.preset}' is scope={profile.scope}, "
                "not an inter-node fabric"
            )
        return profile
    assert inter.custom is not None
    if inter.custom.bandwidth_gbps > FABRIC_SANITY_GBPS:
        warnings.append(
            f"custom bandwidth {inter.custom.bandwidth_gbps:g} Gbps exceeds any "
            f"shipping fabric (> {FABRIC_SANITY_GBPS} Gbps); kept for what-if "
            "sweeps, labelled user_defined"
        )
    return LinkProfile(
        link_id="user-defined",
        type=LinkType(inter.custom.type),
        scope="internode",
        bandwidth_gbps=inter.custom.bandwidth_gbps,
        latency_ns=inter.custom.latency_ns,
        source=Source.USER_DEFINED,
    )


def build_cluster(
    request: ClusterBuildRequest,
    out_dir: str | Path,
    root: str | Path = ".",
) -> tuple[ClusterSummary, list[str], list[dict]]:
    """Build (or idempotently return) a custom cluster.

    Returns (summary, warnings, islands-info). FR-CB4: the cluster_id embeds
    the request hash, so re-submitting the same request reuses the same file
    and DB row instead of creating a duplicate.
    """
    root = Path(root)
    out_dir = Path(out_dir)
    warnings: list[str] = []

    total = sum(g.count_per_node * g.num_nodes for g in request.nodes)
    if total > MAX_TOTAL_ACCELERATORS:
        raise LabConfigError(
            f"{total} accelerators exceed the interactive builder cap of "
            f"{MAX_TOTAL_ACCELERATORS} (fast-path responsiveness); use batch mode "
            "(a LabConfig + `scenariolab run`) for larger studies"
        )
    profiles = _validate_classes(request, root)
    fabric = _inter_node_profile(request.interconnect.inter_node, root, warnings)
    pcie = load_link_profile("pcie_gen4", root)

    cluster_id = request.cluster_id()
    path = out_dir / f"{cluster_id}.yaml"
    already_existed = path.exists()

    if not already_existed:
        nodes = []
        links: list[Link] = []
        index = 0
        for group in request.nodes:
            fast_name = CLASS_FACTS[group.cls]["fast_fabric"]
            fast = load_link_profile(fast_name, root) if fast_name else None
            for _ in range(group.num_nodes):
                node, node_links = _make_node(
                    index, group.cls, profiles[group.cls], group.count_per_node,
                    pcie, fast,
                    nic_speed_gbps=fabric.bandwidth_gbps,
                    nic_type=fabric.type.value.lower(),
                )
                nodes.append(node)
                links.extend(node_links)
                index += 1
        for i in range(index):
            for j in range(i + 1, index):
                links.append(
                    Link(
                        id=f"{fabric.link_id}-node{i}-node{j}",
                        src=f"node{i}/nic0",
                        dst=f"node{j}/nic0",
                        type=fabric.type,
                        bandwidth_gbps=fabric.bandwidth_gbps,
                        latency_ns=fabric.latency_ns,
                        energy_per_bit_pj=fabric.energy_per_bit_pj,
                        contention_group=f"nic-node{i}-node{j}",
                        source=fabric.source,
                    )
                )
        spec = ClusterSpecV2(cluster_id=cluster_id, nodes=nodes, links=links)
        out_dir.mkdir(parents=True, exist_ok=True)
        header = (
            "# generated_by: scenariolab-builder\n"
            f"# request_hash: {request.request_hash()}\n"
            "# Inter-node link source label: "
            f"{fabric.source.value}\n"
        )
        path.write_text(
            header
            + yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True),
                             sort_keys=False)
        )

    loaded = load_cluster_spec(path)
    islands = detect_islands(loaded, load_profiles_for(loaded, root))
    if not islands:
        path.unlink(missing_ok=True)
        raise ClusterGenError(f"{cluster_id}: no execution island")
    if not _fully_connected(loaded):
        path.unlink(missing_ok=True)
        raise ClusterGenError(f"{cluster_id}: cluster graph is not connected")

    summary = ClusterSummary(
        cluster_id=cluster_id,
        seed=0,  # deterministic build, no RNG involved
        yaml_path=path,
        num_nodes=len(loaded.nodes),
        num_accels=sum(len(n.accelerators) for n in loaded.nodes),
        num_free_accels=sum(len(n.accelerators) for n in loaded.nodes),
        classes=sorted({g.cls for g in request.nodes}),
        num_islands=len(islands),
        has_npu=any(
            a.type.value == "NPU" for n in loaded.nodes for a in n.accelerators
        ),
        origin="custom",
        link_summary=describe_fabric(loaded),
        build_request_json=json.dumps(request.model_dump(by_alias=True), sort_keys=True),
    )
    islands_info = [
        {
            "id": island.id,
            "accelerators": island.size,
            "model": island.accelerator_model,
            "tp_candidates": island.max_tp_candidates,
        }
        for island in islands
    ]
    return summary, warnings, islands_info


def load_build_request(path: str | Path) -> ClusterBuildRequest:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise LabConfigError(f"{path}: expected a YAML mapping")
    try:
        return ClusterBuildRequest.model_validate(raw)
    except LabConfigError:
        raise
    except Exception as exc:
        raise LabConfigError(f"{path}: {exc}") from exc
