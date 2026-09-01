"""M1 ClusterGenerator: random, valid ClusterSpecV2 instances (DESIGN §4).

Outputs are instances of the existing planner schema - no new schema. Every
hardware number is copied from profiles/ (accelerator profiles and the link
classes under profiles/networks/); the generator invents nothing (FR-C8).

Interconnects are dictated by the accelerator class (FR-C4): PCIe-attached
cards get PCIE links on a shared root complex, NVLink-capable cards get
NVLINK, and an RNGD card is a single device whose fabric is internal, so it
gets no intra-node link at all and forms singleton islands by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from planner.inventory import (
    Accelerator,
    AcceleratorProfile,
    AcceleratorState,
    ClusterSpecV2,
    Link,
    LinkType,
    Nic,
    Node,
    NodePower,
    Source,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from scenariolab.config import ClusterGeneratorConfig, profile_path_for
from scenariolab.generator.sampling import rng_for

NETWORK_DIR = Path("profiles/networks")

#: Per-class generation facts that are not (and should not be) in the
#: accelerator profile: which intra-node link class the card family uses and a
#: realistic per-node card count cap. Adding a pool entry requires a row here -
#: failing loudly beats guessing an interconnect (FR-C4).
CLASS_FACTS: dict[str, dict] = {
    "a40": {"intra_link": "pcie_gen4", "max_per_node": 8},
    "a5000": {"intra_link": "pcie_gen4", "max_per_node": 8},
    "rtxpro6000": {"intra_link": "nvlink", "max_per_node": 8},
    # One accelerator is one whole RNGD card (TP=8 internally); the card's
    # fabric is on-package, so there is no intra-node accelerator link.
    "furiosa_rngd_card": {"intra_link": None, "max_per_node": 4},
}


class LinkProfile(BaseModel):
    """One link class under profiles/networks/ (values are copy-referenced)."""

    model_config = ConfigDict(extra="forbid")

    link_id: str
    type: LinkType
    scope: Literal["intranode", "internode"]
    bandwidth_gbps: float = Field(gt=0)
    latency_ns: float = Field(ge=0)
    energy_per_bit_pj: float | None = Field(default=None, ge=0)
    source: Source = Source.PLACEHOLDER


class ClusterGenError(ValueError):
    """Raised when the configured ranges cannot produce a valid cluster."""


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: str
    seed: int
    yaml_path: Path
    num_nodes: int
    num_accels: int
    num_free_accels: int
    classes: list[str]
    num_islands: int
    has_npu: bool


def load_link_profile(name: str, root: Path) -> LinkProfile:
    path = root / NETWORK_DIR / f"{name}.yaml"
    if not path.exists():
        raise ClusterGenError(f"link class '{name}': no profile file at {path}")
    raw = yaml.safe_load(path.read_text())
    return LinkProfile.model_validate(raw)


def _accel_kind(profile: AcceleratorProfile) -> str:
    """GPU/NPU from the backend; cuda is the only GPU backend in this repo."""
    return "GPU" if profile.backend == "cuda" else "NPU"


def _make_node(
    node_index: int,
    pool_entry: str,
    profile: AcceleratorProfile,
    num_accels: int,
    intra_link: LinkProfile | None,
    nic_speed_gbps: float,
    nic_type: str,
) -> tuple[Node, list[Link]]:
    node_id = f"node{node_index}"
    prefix = "gpu" if _accel_kind(profile) == "GPU" else "npu"
    accels = [
        Accelerator(
            id=f"{prefix}{k}",
            type=_accel_kind(profile),  # type: ignore[arg-type]
            vendor=profile.vendor,
            model=profile.model,
            backend=profile.backend,
            memory_gb=profile.memory_gb,
            state=AcceleratorState.FREE,
            profile=str(profile_path_for(pool_entry)),
        )
        for k in range(num_accels)
    ]
    nic = Nic(id="nic0", type=nic_type, speed_gbps=nic_speed_gbps)
    # Always emit a power block so the energy objective is computable; the
    # NodePower defaults are upstream's and stay source: placeholder.
    node = Node(id=node_id, accelerators=accels, nics=[nic], power=NodePower())

    links: list[Link] = []
    if intra_link is not None:
        group = (
            f"{node_id}-pcie-root" if intra_link.type == LinkType.PCIE
            else f"{node_id}-{intra_link.link_id}"
        )
        for a in range(num_accels):
            for b in range(a + 1, num_accels):
                links.append(
                    Link(
                        id=f"{intra_link.link_id}-{node_id}-{accels[a].id}-{accels[b].id}",
                        src=f"{node_id}/{accels[a].id}",
                        dst=f"{node_id}/{accels[b].id}",
                        type=intra_link.type,
                        bandwidth_gbps=intra_link.bandwidth_gbps,
                        latency_ns=intra_link.latency_ns,
                        energy_per_bit_pj=intra_link.energy_per_bit_pj,
                        contention_group=group,
                        source=intra_link.source,
                    )
                )
    return node, links


def _mark_allocated(
    nodes: list[Node], free_ratio: float, rng
) -> int:
    """Set some accelerators ALLOCATED so partially-occupied clusters exist.

    Returns the number of accelerators left FREE (always >= 1)."""
    flat = [(n, a) for n in nodes for a in n.accelerators]
    num_free = max(1, round(len(flat) * free_ratio))
    free_idx = set(rng.choice(len(flat), size=num_free, replace=False).tolist())
    for idx, (_, accel) in enumerate(flat):
        if idx not in free_idx:
            accel.state = AcceleratorState.ALLOCATED
    return num_free


def _build_once(
    config: ClusterGeneratorConfig,
    cluster_id: str,
    rng,
    profiles: dict[str, AcceleratorProfile],
    intra_links: dict[str, LinkProfile | None],
    internode_links: dict[str, LinkProfile],
) -> ClusterSpecV2:
    num_nodes = config.nodes_per_cluster.sample(rng)
    link_name = config.internode_link_pool[int(rng.integers(len(config.internode_link_pool)))]
    fabric = internode_links[link_name]
    nic_type = fabric.type.value.lower()

    nodes: list[Node] = []
    links: list[Link] = []
    for i in range(num_nodes):
        pool = config.accelerator_pool
        entry = pool[int(rng.integers(len(pool)))]
        profile = profiles[entry]
        cap = CLASS_FACTS[entry]["max_per_node"]
        count = min(config.accelerators_per_node.sample(rng), cap)
        node, node_links = _make_node(
            i, entry, profile, count, intra_links[entry],
            nic_speed_gbps=fabric.bandwidth_gbps, nic_type=nic_type,
        )
        nodes.append(node)
        links.extend(node_links)
        # Host-side NIC attachment mirrors the example clusters: the first
        # accelerator reaches the NIC over PCIe (values copied from pcie_gen4).
        pcie = internode_links.get("__pcie_attach__")
        if pcie is not None:
            links.append(
                Link(
                    id=f"pcie-{node.id}-{node.accelerators[0].id}-nic0",
                    src=f"{node.id}/{node.accelerators[0].id}",
                    dst=f"{node.id}/nic0",
                    type=pcie.type,
                    bandwidth_gbps=pcie.bandwidth_gbps,
                    latency_ns=pcie.latency_ns,
                    contention_group=f"{node.id}-pcie-root",
                    source=pcie.source,
                )
            )

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
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

    free_ratio = config.free_ratio.sample(rng)
    _mark_allocated(nodes, free_ratio, rng)
    return ClusterSpecV2(cluster_id=cluster_id, nodes=nodes, links=links)


def generate_cluster(
    config: ClusterGeneratorConfig,
    index: int,
    seed: int,
    out_dir: Path,
    root: Path,
    lab_config_hash: str,
) -> ClusterSummary:
    """Generate, self-validate and write cluster c{index:04d} (FR-C1..C9)."""
    for entry in config.accelerator_pool:
        if entry not in CLASS_FACTS:
            raise ClusterGenError(
                f"accelerator_pool entry '{entry}' has no CLASS_FACTS row; add its "
                "intra-node interconnect and per-node cap before generating with it"
            )
    profiles = {
        entry: _load_profile(entry, root) for entry in config.accelerator_pool
    }
    intra_links = {
        entry: (
            load_link_profile(CLASS_FACTS[entry]["intra_link"], root)
            if CLASS_FACTS[entry]["intra_link"] else None
        )
        for entry in config.accelerator_pool
    }
    internode_links = {
        name: load_link_profile(name, root) for name in config.internode_link_pool
    }
    internode_links["__pcie_attach__"] = load_link_profile("pcie_gen4", root)

    cluster_id = f"c{index:04d}"
    rng = rng_for(seed)
    failures: list[str] = []
    for _attempt in range(20):
        spec = _build_once(config, cluster_id, rng, profiles, intra_links, internode_links)
        path = _write_yaml(spec, seed, out_dir, lab_config_hash)
        try:
            loaded = load_cluster_spec(path)
            islands = detect_islands(loaded, load_profiles_for(loaded, root))
        except Exception as exc:
            failures.append(str(exc))
            path.unlink(missing_ok=True)
            continue
        if not islands:
            failures.append("no FREE execution island")
            path.unlink(missing_ok=True)
            continue
        num_accels = sum(len(n.accelerators) for n in loaded.nodes)
        num_free = sum(
            1 for n in loaded.nodes for a in n.accelerators
            if a.state == AcceleratorState.FREE
        )
        classes = sorted({
            entry
            for n in loaded.nodes
            for a in n.accelerators
            for entry, p in profiles.items()
            if p.model == a.model
        })
        return ClusterSummary(
            cluster_id=cluster_id,
            seed=seed,
            yaml_path=path,
            num_nodes=len(loaded.nodes),
            num_accels=num_accels,
            num_free_accels=num_free,
            classes=classes,
            num_islands=len(islands),
            has_npu=any(
                a.type.value == "NPU" for n in loaded.nodes for a in n.accelerators
            ),
        )
    raise ClusterGenError(
        f"cluster {cluster_id}: 20 attempts produced no valid cluster. "
        f"Adjust the configured ranges. Failure log: {failures}"
    )


def _load_profile(entry: str, root: Path) -> AcceleratorProfile:
    from planner.inventory import load_accelerator_profile

    return load_accelerator_profile(root / profile_path_for(entry))


def _write_yaml(
    spec: ClusterSpecV2, seed: int, out_dir: Path, lab_config_hash: str
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.cluster_id}.yaml"
    header = (
        "# generated_by: scenariolab\n"
        f"# cluster_seed: {seed}\n"
        f"# lab_config_hash: {lab_config_hash}\n"
        "# All hardware numbers are copy-references into profiles/ (accelerator\n"
        "# profiles and profiles/networks/ link classes); the generator invents\n"
        "# no numbers. Link values carry their own source labels.\n"
    )
    body = yaml.safe_dump(
        spec.model_dump(mode="json", exclude_none=True), sort_keys=False
    )
    path.write_text(header + body)
    return path
