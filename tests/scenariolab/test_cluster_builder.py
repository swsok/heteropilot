"""F1 ClusterBuilder tests (workspace work order §3.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from planner.inventory import LinkType, load_cluster_spec
from scenariolab.config import LabConfigError
from scenariolab.generator.cluster_builder import (
    ClusterBuildRequest,
    build_cluster,
    load_build_request,
)
from tests.scenariolab.conftest import ROOT


def _request(**overrides) -> ClusterBuildRequest:
    raw = {
        "name": "my-hetero",
        "nodes": [
            {"class": "a40", "count_per_node": 8, "num_nodes": 2},
            {"class": "furiosa_rngd_card", "count_per_node": 4, "num_nodes": 1},
        ],
        "interconnect": {"inter_node": {"preset": "ib_100g"}},
    }
    raw.update(overrides)
    return ClusterBuildRequest.model_validate(raw)


def test_valid_request_builds_matching_cluster(tmp_path: Path) -> None:
    summary, warnings, islands = build_cluster(_request(), tmp_path, ROOT)
    assert warnings == []
    assert summary.origin == "custom"
    assert summary.num_nodes == 3
    assert summary.num_accels == 20  # 8+8 GPUs + 4 NPU cards
    assert summary.has_npu is True
    assert "INFINIBAND 100Gbps" in (summary.link_summary or "")
    cluster = load_cluster_spec(summary.yaml_path)
    inter = [
        link for link in cluster.links
        if link.src.endswith("nic0") and link.dst.endswith("nic0")
    ]
    assert all(link.bandwidth_gbps == 100 for link in inter)
    assert len(islands) == summary.num_islands >= 3
    # FR-CB5: TP candidates come back for immediate inspection.
    assert all(island["tp_candidates"] for island in islands)


def test_placeholder_class_rejected(tmp_path: Path) -> None:
    request = _request(nodes=[{"class": "a40", "count_per_node": 2, "num_nodes": 1}])
    # ascend_target is not even in CLASS_FACTS -> unknown-class error with the
    # available list; a placeholder WITH facts would hit the source check.
    with pytest.raises(LabConfigError, match="unknown accelerator class"):
        build_cluster(
            _request(nodes=[{"class": "ascend_target", "count_per_node": 2,
                             "num_nodes": 1}]),
            tmp_path, ROOT,
        )
    build_cluster(request, tmp_path, ROOT)  # sane baseline still works


def test_preset_and_custom_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one of preset/custom"):
        _request(interconnect={"inter_node": {
            "preset": "ib_100g",
            "custom": {"type": "ETHERNET", "bandwidth_gbps": 50, "latency_ns": 8000},
        }})
    with pytest.raises(ValueError, match="exactly one of preset/custom"):
        _request(interconnect={"inter_node": {}})


def test_total_accelerator_cap(tmp_path: Path) -> None:
    request = _request(nodes=[{"class": "a40", "count_per_node": 8, "num_nodes": 9}])
    with pytest.raises(LabConfigError, match="exceed the interactive builder cap"):
        build_cluster(request, tmp_path, ROOT)


def test_per_node_cap_enforced(tmp_path: Path) -> None:
    request = _request(
        nodes=[{"class": "furiosa_rngd_card", "count_per_node": 8, "num_nodes": 1}]
    )
    with pytest.raises(LabConfigError, match="per-node cap"):
        build_cluster(request, tmp_path, ROOT)


def test_custom_bandwidth_labelled_user_defined(tmp_path: Path) -> None:
    request = _request(interconnect={"inter_node": {
        "custom": {"type": "ETHERNET", "bandwidth_gbps": 50, "latency_ns": 8000},
    }})
    summary, warnings, _ = build_cluster(request, tmp_path, ROOT)
    assert warnings == []
    cluster = load_cluster_spec(summary.yaml_path)
    inter = [
        link for link in cluster.links
        if link.src.endswith("nic0") and link.dst.endswith("nic0")
    ]
    assert inter
    assert all(link.source.value == "user_defined" for link in inter)
    assert all(link.bandwidth_gbps == 50 for link in inter)
    assert "user_defined" in (summary.link_summary or "")


def test_absurd_bandwidth_warns_but_builds(tmp_path: Path) -> None:
    request = _request(interconnect={"inter_node": {
        "custom": {"type": "ETHERNET", "bandwidth_gbps": 5000, "latency_ns": 100},
    }})
    _, warnings, _ = build_cluster(request, tmp_path, ROOT)
    assert any("exceeds any shipping fabric" in w for w in warnings)


def test_intra_node_dictated_by_profile(tmp_path: Path) -> None:
    """FR-CB2: the user's inter-node choice never leaks into the intra-node
    fabric - rtxpro6000 keeps NVLink regardless."""
    request = _request(
        nodes=[{"class": "rtxpro6000", "count_per_node": 4, "num_nodes": 1}],
        interconnect={"inter_node": {
            "custom": {"type": "ETHERNET", "bandwidth_gbps": 10, "latency_ns": 20000},
        }},
    )
    summary, _, _ = build_cluster(request, tmp_path, ROOT)
    cluster = load_cluster_spec(summary.yaml_path)
    types = {link.type for link in cluster.links}
    assert LinkType.NVLINK in types
    assert LinkType.PCIE in types  # host tree stays


def test_idempotent_rebuild(tmp_path: Path) -> None:
    first, _, _ = build_cluster(_request(), tmp_path, ROOT)
    second, _, _ = build_cluster(_request(), tmp_path, ROOT)
    assert first.cluster_id == second.cluster_id
    assert first.yaml_path.read_bytes() == second.yaml_path.read_bytes()
    assert len(list(tmp_path.glob("custom-*.yaml"))) == 1


def test_load_build_request_from_yaml(tmp_path: Path) -> None:
    spec = tmp_path / "req.yaml"
    spec.write_text(yaml.safe_dump({
        "name": "from-file",
        "nodes": [{"class": "a5000", "count_per_node": 2, "num_nodes": 1}],
        "interconnect": {"inter_node": {"preset": "ib_400g"}},
    }))
    request = load_build_request(spec)
    summary, _, _ = build_cluster(request, tmp_path, ROOT)
    assert summary.cluster_id.startswith("custom-from-file-")
