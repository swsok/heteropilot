"""`schema_version: 2` - the graph-aware inventory H2 adds.

The whole point of the version field is that a v1 file means today exactly what
it meant yesterday. So this file spends more of its length on what v1 must keep
refusing than on what v2 accepts: a v2 field silently ignored in a v1 file is
as bad as one silently honoured, and only an error distinguishes "not supported
here" from "accepted and had no effect".

`bandwidth_gbps` keeps its v1 meaning throughout - GB/s in spite of the name.
Reinterpreting a committed number's unit is how a 64 quietly becomes an 8.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from planner.inventory import (
    AcceleratorProfile,
    InventoryError,
    Source,
    load_cluster_spec,
    load_profiles_for,
)

ROOT = Path(__file__).resolve().parents[1]
V2_MIN = ROOT / "tests" / "data" / "cluster_v2_min.yaml"
EXAMPLES = sorted((ROOT / "examples" / "clusters").glob("*.yaml"))


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cluster.yaml"
    p.write_text(textwrap.dedent(body))
    return p


V1_BASE = """
cluster_id: t
nodes:
  - id: node0
    accelerators:
      - {id: gpu0, type: GPU, vendor: NVIDIA, model: H100, backend: cuda, memory_gb: 80}
      - {id: gpu1, type: GPU, vendor: NVIDIA, model: H100, backend: cuda, memory_gb: 80}
links:
  - {id: l01, src: node0/gpu0, dst: node0/gpu1, type: NVLINK,
     bandwidth_gbps: 900, latency_ns: 1000}
"""


# --- (i) every committed cluster is v1, and reads as it always did ---------

@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_committed_clusters_are_schema_version_1(path: Path) -> None:
    cluster = load_cluster_spec(path)
    assert cluster.schema_version == 1
    assert cluster.net_switches == []
    assert cluster.shared_resources == []
    assert cluster.snapshot_id is None


def test_v1_bandwidth_is_still_read_as_gb_per_s() -> None:
    """D120. The name says gbps and the number has always meant GB/s."""
    cluster = load_cluster_spec(EXAMPLES[0])
    for link in cluster.links:
        assert link.bandwidth_unit == "GB/s"


# --- (ii) a v1 file may not carry v2 fields -------------------------------

@pytest.mark.parametrize(
    "patch",
    [
        pytest.param("    cpu_sockets:\n      - {id: sock0}\n", id="cpu_sockets"),
        pytest.param("    pcie_switches:\n      - {id: pcisw0}\n", id="pcie_switches"),
        pytest.param("    host_price_per_hour_usd: 1.0\n", id="host_price"),
    ],
)
def test_v1_node_rejects_v2_fields(tmp_path: Path, patch: str) -> None:
    body = V1_BASE.replace("    accelerators:\n", patch + "    accelerators:\n", 1)
    with pytest.raises(InventoryError, match="schema_version"):
        load_cluster_spec(write(tmp_path, body))


def test_v1_link_rejects_a_changed_bandwidth_unit(tmp_path: Path) -> None:
    body = V1_BASE.replace("bandwidth_gbps: 900", 'bandwidth_gbps: 900, bandwidth_unit: "Gbit/s"')
    with pytest.raises(InventoryError, match="schema_version"):
        load_cluster_spec(write(tmp_path, body))


def test_v1_link_may_restate_the_default_unit(tmp_path: Path) -> None:
    """Stating GB/s in a v1 file changes nothing, so it is not an error."""
    body = V1_BASE.replace("bandwidth_gbps: 900", 'bandwidth_gbps: 900, bandwidth_unit: "GB/s"')
    assert load_cluster_spec(write(tmp_path, body)).schema_version == 1


def test_v1_cluster_rejects_shared_resources(tmp_path: Path) -> None:
    body = V1_BASE + (
        'shared_resources:\n  - {id: up, kind: nic, capacity: 10.0}\n'
    )
    with pytest.raises(InventoryError, match="schema_version"):
        load_cluster_spec(write(tmp_path, body))


def test_v1_cluster_rejects_net_switches(tmp_path: Path) -> None:
    body = V1_BASE + "net_switches:\n  - {id: sw0}\n"
    with pytest.raises(InventoryError, match="schema_version"):
        load_cluster_spec(write(tmp_path, body))


def test_v1_accelerator_rejects_a_price(tmp_path: Path) -> None:
    body = V1_BASE.replace("memory_gb: 80}", "memory_gb: 80, price_per_hour_usd: 2.0}", 1)
    with pytest.raises(InventoryError, match="schema_version"):
        load_cluster_spec(write(tmp_path, body))


def test_v1_rejects_a_bare_switch_endpoint(tmp_path: Path) -> None:
    body = V1_BASE.replace("dst: node0/gpu1", "dst: sw0", 1)
    with pytest.raises(InventoryError, match="schema_version 2"):
        load_cluster_spec(write(tmp_path, body))


# --- (iii) the v2 minimal example -----------------------------------------

def test_v2_min_loads_with_every_vertex_kind() -> None:
    cluster = load_cluster_spec(V2_MIN)
    assert cluster.schema_version == 2
    assert cluster.snapshot_id == "2026-09-22T00:00:00Z"
    assert [s.id for s in cluster.net_switches] == ["sw0"]
    node0 = cluster.node("node0")
    assert [s.id for s in node0.cpu_sockets] == ["sock0"]
    assert [s.id for s in node0.pcie_switches] == ["pcisw0"]
    assert node0.host_price_per_hour_usd == pytest.approx(1.0)
    assert cluster.accelerator("node0", "gpu0").price_per_hour_usd == pytest.approx(2.0)
    resource = cluster.shared_resources[0]
    assert (resource.capacity, resource.reserved) == (10.0, 6.0)


def test_a_switch_endpoint_carries_no_node() -> None:
    """How a cluster-scoped vertex is told from a node-scoped one.

    `endpoints` reports the node half as "" so the intra-node callers, which
    compare it against a real node id, can never match a switch.
    """
    cluster = load_cluster_spec(V2_MIN)
    link = next(link for link in cluster.links if link.id == "l-nic0-sw0")
    (src_node, src_dev), (dst_node, dst_dev) = link.endpoints
    assert (src_node, src_dev) == ("node0", "nic0")
    assert (dst_node, dst_dev) == ("", "sw0")


def test_contention_group_and_shared_resource_are_mutually_exclusive(tmp_path: Path) -> None:
    body = V1_BASE.replace("cluster_id: t", "cluster_id: t\nschema_version: 2").replace(
        "latency_ns: 1000}",
        "latency_ns: 1000, contention_group: g, shared_resource: up}",
    ) + 'shared_resources:\n  - {id: up, kind: nic, capacity: 10.0}\n'
    with pytest.raises(InventoryError, match="state one"):
        load_cluster_spec(write(tmp_path, body))


def test_unknown_shared_resource_id_is_rejected(tmp_path: Path) -> None:
    body = V1_BASE.replace("cluster_id: t", "cluster_id: t\nschema_version: 2").replace(
        "latency_ns: 1000}", "latency_ns: 1000, shared_resource: ghost}"
    )
    with pytest.raises(InventoryError, match="names no entry in shared_resources"):
        load_cluster_spec(write(tmp_path, body))


def test_unregistered_switch_endpoint_is_rejected(tmp_path: Path) -> None:
    body = V1_BASE.replace("cluster_id: t", "cluster_id: t\nschema_version: 2").replace(
        "dst: node0/gpu1", "dst: ghostsw", 1
    )
    with pytest.raises(InventoryError, match="must name a net_switch"):
        load_cluster_spec(write(tmp_path, body))


def test_a_reservation_may_not_exceed_capacity(tmp_path: Path) -> None:
    body = V1_BASE.replace("cluster_id: t", "cluster_id: t\nschema_version: 2") + (
        'shared_resources:\n  - {id: up, kind: nic, capacity: 10.0, reserved: 11.0}\n'
    )
    with pytest.raises(InventoryError, match="exceeds capacity"):
        load_cluster_spec(write(tmp_path, body))


def test_pcie_switch_upstream_must_resolve(tmp_path: Path) -> None:
    body = V1_BASE.replace("cluster_id: t", "cluster_id: t\nschema_version: 2").replace(
        "    accelerators:\n", "    pcie_switches:\n      - {id: sw, upstream: ghost}\n"
        "    accelerators:\n", 1,
    )
    with pytest.raises(InventoryError, match="names no cpu_socket or pcie_switch"):
        load_cluster_spec(write(tmp_path, body))


def test_one_id_namespace_per_node(tmp_path: Path) -> None:
    """A link endpoint is `<node>/<id>` whatever the id names, so kinds share one."""
    body = V1_BASE.replace("cluster_id: t", "cluster_id: t\nschema_version: 2").replace(
        "    accelerators:\n", "    cpu_sockets:\n      - {id: gpu0}\n    accelerators:\n", 1,
    )
    with pytest.raises(InventoryError, match="duplicate"):
        load_cluster_spec(write(tmp_path, body))


# --- (iv) a price with no source is a made-up number ----------------------

def test_profile_price_without_a_source_is_refused() -> None:
    with pytest.raises(ValueError, match="price_source"):
        AcceleratorProfile.model_validate(
            {
                "profile_id": "p", "vendor": "v", "model": "m", "backend": "cuda",
                "memory_gb": 80, "memory_bandwidth_gbps": 2000,
                "price_per_hour_usd": 2.0,
            }
        )


def test_profile_price_with_a_source_is_accepted() -> None:
    profile = AcceleratorProfile.model_validate(
        {
            "profile_id": "p", "vendor": "v", "model": "m", "backend": "cuda",
            "memory_gb": 80, "memory_bandwidth_gbps": 2000,
            "price_per_hour_usd": 2.0, "price_source": "placeholder",
        }
    )
    assert profile.price_source is Source.PLACEHOLDER


def test_runtime_capabilities_default_to_unstated() -> None:
    """Unstated is not unsupported; a checker must be able to tell them apart."""
    profile = AcceleratorProfile.model_validate(
        {
            "profile_id": "p", "vendor": "v", "model": "m", "backend": "cuda",
            "memory_gb": 80, "memory_bandwidth_gbps": 2000,
        }
    )
    assert profile.runtime_capabilities is None


def test_runtime_capabilities_round_trip() -> None:
    profile = AcceleratorProfile.model_validate(
        {
            "profile_id": "p", "vendor": "v", "model": "m", "backend": "cuda",
            "memory_gb": 80, "memory_bandwidth_gbps": 2000,
            "runtime_capabilities": {
                "collectives": ["all_reduce", "p2p"], "max_world_size": 8,
                "kv_transfer": True, "source": "placeholder",
            },
        }
    )
    caps = profile.runtime_capabilities
    assert caps is not None
    assert caps.collectives == ["all_reduce", "p2p"]
    assert caps.max_world_size == 8
    assert caps.kv_transfer is True


# --- (v) nothing downstream moved -----------------------------------------

def test_committed_profiles_still_resolve() -> None:
    for path in EXAMPLES:
        cluster = load_cluster_spec(path)
        assert load_profiles_for(cluster, ROOT)
