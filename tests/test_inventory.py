"""ClusterSpecV2, island detection and compatibility (work order §5.2, §9)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from planner.inventory import (
    AcceleratorProfile,
    InventoryError,
    LinkType,
    compatibility,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
    model_slug,
    tp_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "clusters" / "heterogeneous-lab.yaml"


def cluster_yaml(body: str) -> str:
    return textwrap.dedent(body)


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cluster.yaml"
    p.write_text(cluster_yaml(body))
    return p


def accel(aid: str, backend: str = "cuda", model: str = "H100", state: str = "FREE") -> str:
    return f"""
      - id: {aid}
        type: GPU
        vendor: NVIDIA
        model: {model}
        backend: {backend}
        memory_gb: 80
        state: {state}
"""


FOUR_GPU_NVLINK = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0')}{accel('gpu1')}{accel('gpu2')}{accel('gpu3')}
links:
  - {{id: l01, src: node0/gpu0, dst: node0/gpu1, type: NVLINK,
     bandwidth_gbps: 900, latency_ns: 1000}}
  - {{id: l12, src: node0/gpu1, dst: node0/gpu2, type: NVLINK,
     bandwidth_gbps: 900, latency_ns: 1000}}
  - {{id: l23, src: node0/gpu2, dst: node0/gpu3, type: NVLINK,
     bandwidth_gbps: 900, latency_ns: 1000}}
"""


# --- schema validation ----------------------------------------------------

def test_example_cluster_loads_and_profiles_resolve() -> None:
    cluster = load_cluster_spec(EXAMPLE)
    profiles = load_profiles_for(cluster, ROOT)
    assert {"RTX-A5000", "RTXPRO6000", "ASCEND_TARGET"} <= set(profiles)


def test_link_endpoint_must_reference_a_real_device(tmp_path: Path) -> None:
    body = FOUR_GPU_NVLINK.replace("dst: node0/gpu1", "dst: node0/ghost", 1)
    with pytest.raises(InventoryError, match="does not name any accelerator or nic"):
        load_cluster_spec(write(tmp_path, body))


def test_link_endpoint_format_enforced(tmp_path: Path) -> None:
    body = FOUR_GPU_NVLINK.replace("src: node0/gpu0", "src: gpu0", 1)
    with pytest.raises(InventoryError):
        load_cluster_spec(write(tmp_path, body))


def test_duplicate_accelerator_id_rejected(tmp_path: Path) -> None:
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0')}{accel('gpu0')}
"""
    with pytest.raises(InventoryError, match="duplicate"):
        load_cluster_spec(write(tmp_path, body))


# --- island detection -----------------------------------------------------

def test_connected_same_backend_gpus_form_one_island(tmp_path: Path) -> None:
    islands = detect_islands(load_cluster_spec(write(tmp_path, FOUR_GPU_NVLINK)))
    assert len(islands) == 1
    assert islands[0].size == 4
    assert islands[0].interconnect_type is LinkType.NVLINK
    assert islands[0].total_memory_gb == 320


def test_backends_are_never_mixed_in_one_island(tmp_path: Path) -> None:
    """Absolute rule 2, enforced at the earliest possible point."""
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0', 'cuda', 'H100')}{accel('npu0', 'ascend', 'ASCEND')}
links:
  - {{id: l, src: node0/gpu0, dst: node0/npu0, type: PCIE, bandwidth_gbps: 64, latency_ns: 1500}}
"""
    islands = detect_islands(load_cluster_spec(write(tmp_path, body)))
    assert len(islands) == 2
    for island in islands:
        backends = {island.backend}
        assert len(backends) == 1
    assert {i.backend for i in islands} == {"cuda", "ascend"}


def test_unlinked_accelerators_become_singleton_islands(tmp_path: Path) -> None:
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0')}{accel('gpu1')}
"""
    islands = detect_islands(load_cluster_spec(write(tmp_path, body)))
    assert [i.size for i in islands] == [1, 1]
    assert all(i.interconnect_type is None for i in islands)
    assert all(i.max_tp_candidates == [1] for i in islands)


def test_infiniband_does_not_merge_islands(tmp_path: Path) -> None:
    """Cross-island fabrics must not be mistaken for intra-island interconnect."""
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0')}{accel('gpu1')}
links:
  - {{id: l, src: node0/gpu0, dst: node0/gpu1, type: INFINIBAND,
     bandwidth_gbps: 400, latency_ns: 5000}}
"""
    assert len(detect_islands(load_cluster_spec(write(tmp_path, body)))) == 2


def test_non_free_accelerators_are_excluded(tmp_path: Path) -> None:
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0')}{accel('gpu1', state='DEGRADED')}{accel('gpu2', state='ALLOCATED')}
"""
    cluster = load_cluster_spec(write(tmp_path, body))
    assert sum(i.size for i in detect_islands(cluster)) == 1
    assert sum(i.size for i in detect_islands(cluster, free_only=False)) == 3


def test_disconnected_same_model_groups_get_distinct_ids(tmp_path: Path) -> None:
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0')}{accel('gpu1')}{accel('gpu2')}{accel('gpu3')}
links:
  - {{id: a, src: node0/gpu0, dst: node0/gpu1, type: NVLINK, bandwidth_gbps: 900, latency_ns: 1000}}
  - {{id: b, src: node0/gpu2, dst: node0/gpu3, type: NVLINK, bandwidth_gbps: 900, latency_ns: 1000}}
"""
    islands = detect_islands(load_cluster_spec(write(tmp_path, body)))
    assert len(islands) == 2
    assert len({i.id for i in islands}) == 2


def test_example_cluster_islands() -> None:
    cluster = load_cluster_spec(EXAMPLE)
    profiles = load_profiles_for(cluster, ROOT)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    assert set(islands) == {
        "cuda-rtx-a5000-node0",
        "cuda-rtxpro6000-node1",
        "ascend-ascend-target-node2",
    }
    # npu2 is DEGRADED and must not appear.
    assert islands["ascend-ascend-target-node2"].accelerator_ids == ["npu0", "npu1"]
    # a5000.yaml declares max_tp_size 1 because only tp1 has been profiled.
    assert islands["cuda-rtx-a5000-node0"].max_tp_candidates == [1]
    assert islands["cuda-rtxpro6000-node1"].max_tp_candidates == [1, 2]


def test_island_detection_is_deterministic() -> None:
    """Reproducibility (work order §9): same input, identical output."""
    cluster = load_cluster_spec(EXAMPLE)
    profiles = load_profiles_for(cluster, ROOT)
    runs = [
        [i.model_dump() for i in detect_islands(cluster, profiles)]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# --- TP enumeration -------------------------------------------------------

@pytest.mark.parametrize(
    "size,max_tp,expected",
    [
        (1, 8, [1]),
        (2, 8, [1, 2]),
        (4, 8, [1, 2, 4]),
        (8, 8, [1, 2, 4, 8]),
        (8, 4, [1, 2, 4]),
        (6, 8, [1, 2, 3, 6]),
        (4, 1, [1]),
    ],
)
def test_tp_candidates_are_divisors_capped_by_profile(size, max_tp, expected) -> None:
    assert tp_candidates(size, max_tp) == expected


# --- compatibility --------------------------------------------------------

def profile(**kw) -> AcceleratorProfile:
    base = {
        "profile_id": "p", "vendor": "v", "model": "m", "backend": "cuda",
        "memory_gb": 80, "memory_bandwidth_gbps": 1000,
        "supported_models": [
            {"pattern": "meta-llama/Llama-3.1-*", "dtypes": ["bfloat16", "fp8"]}
        ],
    }
    base.update(kw)
    return AcceleratorProfile.model_validate(base)


@pytest.mark.parametrize(
    "model,dtype,expected",
    [
        ("meta-llama/Llama-3.1-8B", "bfloat16", True),
        ("meta-llama/Llama-3.1-70B", "fp8", True),
        ("meta-llama/Llama-3.1-8B", "float32", False),   # dtype not declared
        ("Qwen/Qwen3-32B", "bfloat16", False),           # pattern does not match
    ],
)
def test_compatibility_matches_pattern_and_dtype(model, dtype, expected) -> None:
    assert compatibility(model, dtype, profile()) is expected


def test_no_declared_support_means_unsupported() -> None:
    """Unknown must not read as permitted (work order §11 Risk 1)."""
    assert compatibility("anything/at-all", "bfloat16", profile(supported_models=[])) is False


def test_profile_model_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "p.yaml").write_text(
        "profile_id: p\nvendor: v\nmodel: WRONG\nbackend: cuda\n"
        "memory_gb: 80\nmemory_bandwidth_gbps: 1000\n"
    )
    body = f"""
cluster_id: t
nodes:
  - id: node0
    accelerators:{accel('gpu0').rstrip()}
        profile: p.yaml
"""
    cluster = load_cluster_spec(write(tmp_path, body))
    with pytest.raises(InventoryError, match="declares model"):
        load_profiles_for(cluster, tmp_path)


def test_model_slug_keeps_capacity_suffix() -> None:
    assert model_slug("H100-80GB") == "h100-80gb"
    assert model_slug("RTX-A5000") == "rtx-a5000"
    assert model_slug("ASCEND_TARGET") == "ascend-target"
