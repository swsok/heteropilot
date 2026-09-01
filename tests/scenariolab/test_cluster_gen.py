"""M1 ClusterGenerator tests (DESIGN §4.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.inventory import (
    AcceleratorState,
    LinkType,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from scenariolab.config import ClusterGeneratorConfig, LabConfigError, load_lab_config
from scenariolab.generator.cluster_gen import generate_cluster
from scenariolab.generator.sampling import derive_seed
from tests.scenariolab.conftest import ROOT, write_lab_config


def _gen_config(**overrides) -> ClusterGeneratorConfig:
    raw = {
        "num_clusters": 4,
        "nodes_per_cluster": {"min": 1, "max": 3},
        "accelerators_per_node": {"min": 1, "max": 4},
        "accelerator_pool": ["a40", "a5000", "rtxpro6000", "furiosa_rngd_card"],
        "same_class_per_node": True,
        "internode_link_pool": ["ib_100g", "ib_400g"],
        "free_ratio": {"min": 0.5, "max": 1.0},
    }
    raw.update(overrides)
    return ClusterGeneratorConfig.model_validate(raw)


def _generate(tmp_path: Path, index: int, seed: int, config=None, subdir: str = "a"):
    return generate_cluster(
        config or _gen_config(), index, seed, tmp_path / subdir, ROOT, "deadbeef"
    )


def test_determinism_byte_identical(tmp_path: Path) -> None:
    seed = derive_seed(777, "cluster", 0)
    first = _generate(tmp_path, 0, seed, subdir="a")
    second = _generate(tmp_path, 0, seed, subdir="b")
    assert first.yaml_path.read_bytes() == second.yaml_path.read_bytes()


def test_independence_from_batch_size(tmp_path: Path) -> None:
    """FR: growing num_clusters must not change existing clusters - guaranteed
    because each index gets its own derived seed."""
    texts_small = [
        _generate(tmp_path, i, derive_seed(9, "cluster", i), subdir="s").yaml_path.read_text()
        for i in range(3)
    ]
    texts_large = [
        _generate(tmp_path, i, derive_seed(9, "cluster", i), subdir="l").yaml_path.read_text()
        for i in range(6)
    ]
    assert texts_small == texts_large[:3]


def test_all_generated_clusters_valid(tmp_path: Path) -> None:
    for i in range(10):
        summary = _generate(tmp_path, i, derive_seed(5, "cluster", i))
        cluster = load_cluster_spec(summary.yaml_path)
        islands = detect_islands(cluster, load_profiles_for(cluster, ROOT))
        assert len(islands) >= 1
        assert summary.num_islands == len(islands)
        free = [
            a for n in cluster.nodes for a in n.accelerators
            if a.state == AcceleratorState.FREE
        ]
        assert len(free) == summary.num_free_accels >= 1


def test_placeholder_pool_rejected(tmp_path: Path) -> None:
    path = write_lab_config(
        tmp_path, **{"cluster_generator.accelerator_pool": ["a5000", "ascend_target"]}
    )
    with pytest.raises(LabConfigError, match="placeholder"):
        load_lab_config(path, ROOT)


def test_unknown_pool_entry_rejected(tmp_path: Path) -> None:
    path = write_lab_config(
        tmp_path, **{"cluster_generator.accelerator_pool": ["no_such_card"]}
    )
    with pytest.raises(LabConfigError, match="no profile file"):
        load_lab_config(path, ROOT)


def _accel_to_accel_links(cluster) -> list:
    nics = {f"{n.id}/{nic.id}" for n in cluster.nodes for nic in n.nics}
    return [
        link for link in cluster.links
        if link.src not in nics and link.dst not in nics
    ]


def test_interconnect_dictated_by_profile(tmp_path: Path) -> None:
    """FR-C4 (topology v2): every class sits on the host PCIe tree; only
    NVLink-capable classes additionally get a fast peer fabric."""
    cases = {
        "rtxpro6000": {LinkType.NVLINK, LinkType.PCIE},
        "a40": {LinkType.PCIE},
        "furiosa_rngd_card": {LinkType.PCIE},
    }
    for entry, expected_types in cases.items():
        config = _gen_config(
            accelerator_pool=[entry],
            nodes_per_cluster={"min": 1, "max": 1},
            accelerators_per_node={"min": 2, "max": 4},
        )
        summary = _generate(tmp_path, 0, derive_seed(11, entry), config, subdir=entry)
        cluster = load_cluster_spec(summary.yaml_path)
        peer_types = {link.type for link in _accel_to_accel_links(cluster)}
        assert peer_types == expected_types, (
            f"{entry}: peer links {peer_types}, expected {expected_types}"
        )


def test_contention_groups_assigned(tmp_path: Path) -> None:
    """FR-C5: PCIe peer links share the node's root-complex group."""
    config = _gen_config(
        accelerator_pool=["a40"],
        nodes_per_cluster={"min": 1, "max": 1},
        accelerators_per_node={"min": 3, "max": 3},
    )
    summary = _generate(tmp_path, 0, derive_seed(12, "cg"), config, subdir="cg")
    cluster = load_cluster_spec(summary.yaml_path)
    pcie_groups = {
        link.contention_group for link in cluster.links if link.type == LinkType.PCIE
    }
    assert pcie_groups == {"node0-pcie-root0"}  # 3 devices + NIC on one root


def _adjacency(cluster) -> dict[str, set[str]]:
    vertices = {
        f"{n.id}/{d.id}" for n in cluster.nodes for d in [*n.accelerators, *n.nics]
    }
    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for link in cluster.links:
        adjacency[link.src].add(link.dst)
        adjacency[link.dst].add(link.src)
    return adjacency


def test_every_device_connected(tmp_path: Path) -> None:
    """Topology v2 invariant: no isolated accelerators or NICs, ever - the
    Explorer bug this rework fixes (RNGD cards used to float unconnected)."""
    for i in range(12):
        summary = _generate(tmp_path, i, derive_seed(31, "conn", i), subdir="conn")
        cluster = load_cluster_spec(summary.yaml_path)
        adjacency = _adjacency(cluster)
        assert all(peers for peers in adjacency.values()), (
            f"{summary.cluster_id}: isolated device(s) "
            f"{[v for v, p in adjacency.items() if not p]}"
        )
        start = next(iter(adjacency))
        seen, stack = {start}, [start]
        while stack:
            for peer in adjacency[stack.pop()]:
                if peer not in seen:
                    seen.add(peer)
                    stack.append(peer)
        assert seen == set(adjacency), f"{summary.cluster_id}: graph is disconnected"


def test_pcie_root_complex_structure(tmp_path: Path) -> None:
    """An 8-device node splits into two 4-device roots bridged by the CPU;
    the NIC hangs off root 0."""
    config = _gen_config(
        accelerator_pool=["a40"],
        nodes_per_cluster={"min": 1, "max": 1},
        accelerators_per_node={"min": 8, "max": 8},
    )
    summary = _generate(tmp_path, 0, derive_seed(32, "rc"), config, subdir="rc")
    cluster = load_cluster_spec(summary.yaml_path)
    groups: dict[str, list] = {}
    for link in cluster.links:
        if link.type == LinkType.PCIE:
            groups.setdefault(link.contention_group, []).append(link)
    # Two roots (full mesh of 4 = 6 links each; root0 carries the NIC attach
    # too), one CPU-interconnect bridge between them.
    assert set(groups) == {
        "node0-pcie-root0", "node0-pcie-root1", "node0-cpu-interconnect",
    }
    assert len(groups["node0-pcie-root0"]) == 7  # 6 peer + 1 NIC
    assert len(groups["node0-pcie-root1"]) == 6
    assert len(groups["node0-cpu-interconnect"]) == 1
    nic_links = [
        link for link in groups["node0-pcie-root0"] if link.dst == "node0/nic0"
    ]
    assert len(nic_links) == 1


def test_rngd_cards_share_one_island_via_host_bus(tmp_path: Path) -> None:
    """RNGD cards on one host bus now form one island (dp across cards inside
    the island); TP stays 1 per card (profile max_tp_size=1)."""
    config = _gen_config(
        accelerator_pool=["furiosa_rngd_card"],
        nodes_per_cluster={"min": 1, "max": 1},
        accelerators_per_node={"min": 4, "max": 4},
    )
    summary = _generate(tmp_path, 0, derive_seed(33, "rngd"), config, subdir="rngd")
    cluster = load_cluster_spec(summary.yaml_path)
    islands = detect_islands(cluster, load_profiles_for(cluster, ROOT))
    free_islands = [i for i in islands if i.size >= 1]
    assert len(free_islands) >= 1
    biggest = max(free_islands, key=lambda i: i.size)
    assert biggest.max_tp_candidates == [1]  # cards never TP across each other


def test_minimal_ranges_still_valid(tmp_path: Path) -> None:
    config = _gen_config(
        nodes_per_cluster={"min": 1, "max": 1},
        accelerators_per_node={"min": 1, "max": 1},
    )
    summary = _generate(tmp_path, 0, derive_seed(13, "min"), config, subdir="min")
    assert summary.num_nodes == 1
    assert summary.num_accels == 1
    assert summary.num_islands == 1


def test_statistical_class_coverage(tmp_path: Path) -> None:
    """Weak coverage check with a fixed seed: 40 clusters must touch every
    pool class at least once (golden-by-seed, so not flaky)."""
    seen: set[str] = set()
    for i in range(40):
        summary = _generate(tmp_path, i, derive_seed(20, "cov", i), subdir="cov")
        seen.update(summary.classes)
    assert seen == {"a40", "a5000", "rtxpro6000", "furiosa_rngd_card"}


def test_generated_header_records_provenance(tmp_path: Path) -> None:
    """FR-C9: seed and config hash in the YAML header."""
    seed = derive_seed(777, "cluster", 0)
    summary = _generate(tmp_path, 0, seed)
    head = summary.yaml_path.read_text().splitlines()[:4]
    assert head[0] == "# generated_by: scenariolab"
    assert f"# cluster_seed: {seed}" in head
    assert "# lab_config_hash: deadbeef" in head
