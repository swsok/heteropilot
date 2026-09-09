"""The compiler emits slab3d's mode and its corrected dim-1 latency (STEP 2.3).

`WORK_ORDER_rps_aware.md` rev 2. Three things have to be true together, and each
was a way to get this wrong:

  1. an asymmetric candidate carries `topology_mode: slab3d` into the cluster
     config, or `config_builder` computes the flat dims and the whole encoding
     silently does nothing;
  2. dim 1's `link_latency` is multiplied by the MEASURED factor, or the run is
     13-24 % optimistic on TPOT (docs/slab3d_calibration.md);
  3. outside the measured domain the compile REFUSES. A factor of 1.0 there would
     look harmless and would in fact reinstate the full error, unchecked.

And one thing must not change: a symmetric or aggregated candidate compiles
exactly as it did before D28.
"""

from __future__ import annotations

import pytest

from planner.inventory import (
    Accelerator,
    AcceleratorProfile,
    AcceleratorType,
    ClusterSpecV2,
    Link,
    LinkType,
    Node,
    SupportedModel,
    detect_islands,
)
from planner.plan import CandidateConfig, IslandAssignment, Role, ServingArch
from planner.predictor.llmservingsim import (
    OutsideCalibrationDomain,
    compile_to_sim_config,
)
from planner.topology import TopologyGraph

MODEL = "meta-llama/Llama-3.1-8B"
BASE_LAT = 20000.0


def _profiles() -> dict[str, AcceleratorProfile]:
    return {
        "GPU-X": AcceleratorProfile(
            profile_id="gpux", vendor="synthetic", model="GPU-X", backend="cuda",
            memory_gb=80.0, memory_bandwidth_gbps=2000.0, sim_hardware="GPU-X",
            supported_models=[SupportedModel(pattern="*", dtypes=["bfloat16"])],
            max_tp_size=16,
        )
    }


def _cluster(bw: float, size: int = 8) -> ClusterSpecV2:
    """Two islands of `size`, joined at `bw` so the compiled scalar link_bw is `bw`."""
    nodes, links = [], []
    for n in ("n0", "n1"):
        nodes.append(Node(id=n, accelerators=[
            Accelerator(id=f"gpu{i}", type=AcceleratorType.GPU, vendor="synthetic",
                        model="GPU-X", backend="cuda", memory_gb=80.0)
            for i in range(size)
        ]))
        for i in range(size):
            for j in range(i + 1, size):
                links.append(Link(id=f"{n}-nv-{i}{j}", src=f"{n}/gpu{i}",
                                  dst=f"{n}/gpu{j}", type=LinkType.NVLINK,
                                  bandwidth_gbps=bw, latency_ns=BASE_LAT))
    links.append(Link(id="x", src="n0/gpu0", dst="n1/gpu0", type=LinkType.ETHERNET,
                      bandwidth_gbps=bw, latency_ns=BASE_LAT))
    return ClusterSpecV2(cluster_id="slab3d-synth", nodes=nodes, links=links)


def _pd(tp_p: int, tp_d: int, mode: str, island_ids: list[str]) -> CandidateConfig:
    return CandidateConfig(
        id=f"pd-tp{tp_p}-tp{tp_d}", model=MODEL, dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=island_ids[0], role=Role.PREFILL, tp_size=tp_p),
            IslandAssignment(island_id=island_ids[1], role=Role.DECODE, tp_size=tp_d),
        ],
        serving_arch=ServingArch.PD_SPLIT, topology_mode=mode,
    )


def _compile(bw: float, tp_p: int, tp_d: int, mode: str, size: int = 8):
    cluster = _cluster(bw, size)
    profiles = _profiles()
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    ids = sorted(islands)
    assert len(ids) >= 2, f"fixture should give two islands, got {ids}"
    cand = _pd(tp_p, tp_d, mode, ids)
    return compile_to_sim_config(cand, cluster, islands, profiles,
                                 topology=TopologyGraph(cluster))


@pytest.mark.parametrize("bw,tp_p,tp_d,factor", [
    (16.0, 4, 8, 4.0),
    (35.2, 4, 8, 4.0),
    (100.0, 4, 8, 4.0),
    (16.0, 2, 4, 2.0),
    (35.2, 2, 4, 2.0),
    (100.0, 2, 4, 2.0),
])
def test_dim1_latency_carries_the_measured_factor(bw, tp_p, tp_d, factor):
    config, _ = _compile(bw, tp_p, tp_d, "slab3d")
    assert config["topology_mode"] == "slab3d"
    lat = config["link_latency"]
    assert isinstance(lat, list), "slab3d must emit a per-dim list, not a scalar"
    assert lat[0] == pytest.approx(BASE_LAT), "dim 0 keeps the fitted per-hop latency"
    assert lat[1] == pytest.approx(BASE_LAT * factor), (
        f"dim 1 should carry the measured {factor}x for tp{tp_d} at {bw} Gbps"
    )
    assert len(config["link_bw"]) == len(lat), (
        "link_bw and link_latency must have the same length or "
        "_normalize_network_dim_values rejects the config"
    )


@pytest.mark.parametrize("bw,tp_p,tp_d,size", [
    (50.0, 4, 8, 8),        # measured split, unmeasured bandwidth
    (35.2, 8, 16, 16),      # unmeasured split -- tp/2 would predict 8, but nobody measured it
])
def test_outside_the_domain_the_compile_refuses(bw, tp_p, tp_d, size):
    with pytest.raises(OutsideCalibrationDomain, match="outside the measured"):
        _compile(bw, tp_p, tp_d, "slab3d", size)


def test_refusal_is_not_a_compile_error():
    """It must not land in the SIM_ERROR bucket -- see test_calibration_domain."""
    from planner.predictor.llmservingsim import CompileError
    assert not issubclass(OutsideCalibrationDomain, CompileError)


@pytest.mark.parametrize("tp_p,tp_d", [(1, 2), (2, 4), (4, 8)])
def test_the_emitted_list_length_matches_what_config_builder_will_compute(tp_p, tp_d):
    """The invariant that broke: `_normalize_network_dim_values` rejects a list
    whose length differs from the topology's dimension count, and the compiler
    computed those dims from instances it had not stamped with the mode -- so it
    sized the list from `auto`'s dims while the simulator used slab3d's. `[1,2]`
    died with "'link_bw' must have exactly 3 value(s) ... but got 2"."""
    from serving.core.config_builder import (
        TOPOLOGY_MODE_KEY,
        _compute_network_dims,
        _normalize_network_dim_values,
    )
    config, _ = _compile(35.2, tp_p, tp_d, "slab3d")
    flattened = [
        {**inst, TOPOLOGY_MODE_KEY: config["topology_mode"]}
        for node in config["nodes"] for inst in node["instances"]
    ]
    num_dims = len(_compute_network_dims(flattened))
    # the call the simulator makes; it raises if the lengths disagree
    _normalize_network_dim_values(config["link_bw"], num_dims, "link_bw")
    _normalize_network_dim_values(config["link_latency"], num_dims, "link_latency")
    assert len(config["link_latency"]) == num_dims


def test_a_symmetric_pd_candidate_is_untouched():
    """`auto` is the pre-D28 path: scalar latency, no topology_mode key."""
    config, _ = _compile(35.2, 4, 4, "auto")
    assert "topology_mode" not in config
    assert not isinstance(config["link_latency"], list)
    assert config["link_latency"] == pytest.approx(BASE_LAT)


def test_slab3d_without_a_decode_assignment_is_a_compile_error():
    """The mode exists for asymmetric P/D; anything else is a bug, not a refusal."""
    from planner.predictor.llmservingsim import CompileError
    cluster = _cluster(35.2)
    profiles = _profiles()
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    ids = sorted(islands)
    cand = CandidateConfig(
        id="bad", model=MODEL, dtype="bfloat16",
        assignments=[IslandAssignment(island_id=ids[0], role=Role.AGGREGATED,
                                      tp_size=4)],
        topology_mode="slab3d",
    )
    with pytest.raises(CompileError, match="without a decode assignment"):
        compile_to_sim_config(cand, cluster, islands, profiles,
                              topology=TopologyGraph(cluster))
