"""M1 ClusterGenerator: random, valid ClusterSpecV2 instances (DESIGN §4).

Outputs are instances of the existing planner schema - no new schema. Every
hardware number is copied from profiles/ (accelerator profiles and the link
classes under profiles/networks/); the generator invents nothing (FR-C8).

Intra-node topology models the host, not just the fabric (topology v2):

- Every accelerator sits on a PCIe ROOT COMPLEX; a root hosts at most
  DEVICES_PER_ROOT (4) devices, mirroring how real servers hang ~4 cards off
  one CPU's bus. Devices on one root are fully meshed with PCIE links sharing
  that root's contention group; roots are bridged through the CPU
  interconnect (one PCIE link between root representatives, its own
  contention group).
- The NIC hangs off root 0, so every device reaches the fabric through the
  host - the whole node is one connected component by construction.
- Classes with a fast fabric (FR-C4: rtxpro6000 -> NVLINK) additionally get
  an all-pairs fabric mesh on top of the host PCIe tree, exactly like the
  example clusters (NVLink pair + PCIe NIC attach).
- An RNGD card's internal fabric stays internal (no ONPACKAGE links between
  cards), but the cards live on the host PCIe tree like any other device -
  a card with no path to its own node's NIC would be unusable in reality.
"""

from __future__ import annotations

import itertools
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
#: accelerator profile: whether the card family has a fast peer fabric on top
#: of the host PCIe tree, and a realistic per-node card count cap. Adding a
#: pool entry requires a row here - failing loudly beats guessing an
#: interconnect (FR-C4).
CLASS_FACTS: dict[str, dict] = {
    "a40": {"fast_fabric": None, "max_per_node": 8},
    "a5000": {"fast_fabric": None, "max_per_node": 8},
    "rtxpro6000": {"fast_fabric": "nvlink", "max_per_node": 8},
    # One accelerator is one whole RNGD card (TP=8 internally); its fabric is
    # on-package, so cards get no peer fabric - only the host PCIe tree.
    "furiosa_rngd_card": {"fast_fabric": None, "max_per_node": 4},
}

#: How many devices share one PCIe root complex (one CPU's bus). Real servers
#: typically hang ~4 cards off each socket's root; an 8-GPU node therefore
#: has two roots bridged by the CPU interconnect.
DEVICES_PER_ROOT = 4


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
    #: 'random' (this generator) or 'custom' (the workspace cluster builder).
    origin: str = "random"
    #: One-line inter-node link description for the catalog (FR-CAT1: the
    #: list view reads only the DB, never re-parses YAML).
    link_summary: str | None = None
    #: The verbatim ClusterBuildRequest for custom clusters (idempotency key).
    build_request_json: str | None = None


def load_link_profile(name: str, root: Path) -> LinkProfile:
    path = root / NETWORK_DIR / f"{name}.yaml"
    if not path.exists():
        raise ClusterGenError(f"link class '{name}': no profile file at {path}")
    raw = yaml.safe_load(path.read_text())
    return LinkProfile.model_validate(raw)


def _accel_kind(profile: AcceleratorProfile) -> str:
    """GPU/NPU from the backend; cuda is the only GPU backend in this repo."""
    return "GPU" if profile.backend == "cuda" else "NPU"


def _pcie_link(
    link_id: str, src: str, dst: str, pcie: LinkProfile, contention_group: str
) -> Link:
    return Link(
        id=link_id, src=src, dst=dst, type=pcie.type,
        bandwidth_gbps=pcie.bandwidth_gbps, latency_ns=pcie.latency_ns,
        energy_per_bit_pj=pcie.energy_per_bit_pj,
        contention_group=contention_group, source=pcie.source,
    )


def _make_node(
    node_index: int,
    pool_entry: str,
    profile: AcceleratorProfile,
    num_accels: int,
    pcie: LinkProfile,
    fast_fabric: LinkProfile | None,
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

    # Host PCIe tree: full mesh within each root complex...
    roots = [accels[i:i + DEVICES_PER_ROOT] for i in range(0, num_accels, DEVICES_PER_ROOT)]
    for r, members in enumerate(roots):
        for a, b in itertools.combinations(members, 2):
            links.append(_pcie_link(
                f"pcie-{node_id}-{a.id}-{b.id}",
                f"{node_id}/{a.id}", f"{node_id}/{b.id}",
                pcie, f"{node_id}-pcie-root{r}",
            ))
    # ...roots bridged through the CPU interconnect (one link per root pair,
    # between the roots' first devices - the path any cross-root P2P takes)...
    for r1, r2 in itertools.combinations(range(len(roots)), 2):
        links.append(_pcie_link(
            f"cpu-{node_id}-root{r1}-root{r2}",
            f"{node_id}/{roots[r1][0].id}", f"{node_id}/{roots[r2][0].id}",
            pcie, f"{node_id}-cpu-interconnect",
        ))
    # ...and the NIC hanging off root 0, so every device reaches the fabric.
    links.append(_pcie_link(
        f"pcie-{node_id}-{roots[0][0].id}-nic0",
        f"{node_id}/{roots[0][0].id}", f"{node_id}/nic0",
        pcie, f"{node_id}-pcie-root0",
    ))

    # Fast peer fabric (e.g. NVLink/NVSwitch) on top of the host tree.
    if fast_fabric is not None:
        for a, b in itertools.combinations(accels, 2):
            links.append(
                Link(
                    id=f"{fast_fabric.link_id}-{node_id}-{a.id}-{b.id}",
                    src=f"{node_id}/{a.id}",
                    dst=f"{node_id}/{b.id}",
                    type=fast_fabric.type,
                    bandwidth_gbps=fast_fabric.bandwidth_gbps,
                    latency_ns=fast_fabric.latency_ns,
                    energy_per_bit_pj=fast_fabric.energy_per_bit_pj,
                    contention_group=f"{node_id}-{fast_fabric.link_id}-switch",
                    source=fast_fabric.source,
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
    pcie: LinkProfile,
    fast_fabrics: dict[str, LinkProfile | None],
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
            i, entry, profile, count, pcie, fast_fabrics[entry],
            nic_speed_gbps=fabric.bandwidth_gbps, nic_type=nic_type,
        )
        nodes.append(node)
        links.extend(node_links)

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
    pcie = load_link_profile("pcie_gen4", root)
    fast_fabrics = {
        entry: (
            load_link_profile(CLASS_FACTS[entry]["fast_fabric"], root)
            if CLASS_FACTS[entry]["fast_fabric"] else None
        )
        for entry in config.accelerator_pool
    }
    internode_links = {
        name: load_link_profile(name, root) for name in config.internode_link_pool
    }

    cluster_id = f"c{index:04d}"
    rng = rng_for(seed)
    failures: list[str] = []
    for _attempt in range(20):
        spec = _build_once(
            config, cluster_id, rng, profiles, pcie, fast_fabrics, internode_links
        )
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
        if not _fully_connected(loaded):
            failures.append("cluster graph is not one connected component")
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
            link_summary=describe_fabric(loaded),
        )
    raise ClusterGenError(
        f"cluster {cluster_id}: 20 attempts produced no valid cluster. "
        f"Adjust the configured ranges. Failure log: {failures}"
    )


def describe_fabric(cluster: ClusterSpecV2) -> str:
    """One-line inter-node fabric description for the catalog view."""
    nics = {f"{n.id}/{nic.id}" for n in cluster.nodes for nic in n.nics}
    for link in cluster.links:
        if link.src in nics and link.dst in nics:
            return (
                f"{link.type.value} {link.bandwidth_gbps:g}Gbps "
                f"({link.source.value})"
            )
    return "single-node"


def _fully_connected(cluster: ClusterSpecV2) -> bool:
    """Every accelerator and NIC in one component over the union of links.

    A device with no path to its node's NIC (and through it, to the rest of
    the cluster) would be unusable in reality; the generator must never emit
    one (topology v2 invariant).
    """
    vertices: set[str] = set()
    for n in cluster.nodes:
        vertices.update(f"{n.id}/{a.id}" for a in n.accelerators)
        vertices.update(f"{n.id}/{nic.id}" for nic in n.nics)
    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for link in cluster.links:
        adjacency[link.src].add(link.dst)
        adjacency[link.dst].add(link.src)
    start = next(iter(vertices))
    seen = {start}
    stack = [start]
    while stack:
        for peer in adjacency[stack.pop()]:
            if peer not in seen:
                seen.add(peer)
                stack.append(peer)
    return seen == vertices


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
