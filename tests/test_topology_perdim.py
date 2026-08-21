"""Level-2 per-dimension topology compile (Phase 5, work order §5.3, deviations D3).

Level 1 collapses the whole ClusterSpecV2 link graph to one scalar `link_bw`, so a
fast intra-island interconnect is dragged down by a slow cross-instance fabric.
Level 2 (`--topology-level 2`) emits a per-ASTRA-dimension list instead, deriving
`[intra_bottleneck, cross_bottleneck]` from the real graph. These tests pin:

- the discriminator: on a multi-island placement (NVLink intra + Ethernet cross)
  Level 1 emits the slow scalar while Level 2 keeps intra fast;
- the dim-count contract: the emitted list length always equals
  serving.core.config_builder._compute_network_dims, so config_builder accepts it;
- single-island placements are byte-identical between levels;
- the all-tp=1 inert-intra-dim case emits a finite placeholder, not inf;
- provenance never overclaims (path_aware stays False; contention not modeled).

Every synthetic cluster here is source=placeholder - no hardware was measured
(absolute rule 3).
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
from planner.plan import CandidateConfig, IslandAssignment, Role, ServingArch, VllmKnobs
from planner.predictor.llmservingsim import compile_to_sim_config
from planner.topology import TopologyGraph
from serving.core.config_builder import (
    _compute_network_dims,
    _normalize_network_dim_values,
)

MODEL = "meta-llama/Llama-3.1-8B"
NVLINK_BW = 900.0
NVLINK_LAT = 500.0
ETH_BW = 10.0
ETH_LAT = 20000.0


def _profiles(devices_per_node: int) -> dict[str, AcceleratorProfile]:
    return {
        "GPU-X": AcceleratorProfile(
            profile_id="gpux", vendor="synthetic", model="GPU-X", backend="cuda",
            memory_gb=80.0, memory_bandwidth_gbps=2000.0, sim_hardware="GPU-X",
            supported_models=[SupportedModel(pattern="*", dtypes=["bfloat16"])],
            max_tp_size=devices_per_node,
        )
    }


def _node(node_id: str, num_gpus: int) -> Node:
    return Node(
        id=node_id,
        accelerators=[
            Accelerator(
                id=f"gpu{i}", type=AcceleratorType.GPU, vendor="synthetic",
                model="GPU-X", backend="cuda", memory_gb=80.0,
            )
            for i in range(num_gpus)
        ],
    )


def _two_island_cluster(num_gpus_per_node: int) -> ClusterSpecV2:
    """Two nodes, fast NVLink inside each, a slow Ethernet hop between them."""
    links = [
        Link(id="eth", src="n0/gpu0", dst="n1/gpu0", type=LinkType.ETHERNET,
             bandwidth_gbps=ETH_BW, latency_ns=ETH_LAT),
    ]
    for n in ("n0", "n1"):
        for i in range(num_gpus_per_node):
            for j in range(i + 1, num_gpus_per_node):
                links.append(Link(
                    id=f"{n}-nv-{i}{j}", src=f"{n}/gpu{i}", dst=f"{n}/gpu{j}",
                    type=LinkType.NVLINK, bandwidth_gbps=NVLINK_BW, latency_ns=NVLINK_LAT,
                ))
    return ClusterSpecV2(
        cluster_id="perdim-synthetic",
        nodes=[_node("n0", num_gpus_per_node), _node("n1", num_gpus_per_node)],
        links=links,
    )


def _compile(cluster, islands_list, profiles, candidate, level):
    islands = {i.id: i for i in islands_list}
    return compile_to_sim_config(
        candidate, cluster, islands, profiles,
        topology=TopologyGraph(cluster), topology_level=level,
    )


def _aggregated_candidate(island_ids, tp) -> CandidateConfig:
    return CandidateConfig(
        id="c", model=MODEL, dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=i, role=Role.AGGREGATED, tp_size=tp)
            for i in island_ids
        ],
        serving_arch=ServingArch.AGGREGATED,
        knobs=VllmKnobs(),
    )


# --- the discriminator ----------------------------------------------------


def test_level1_collapses_intra_but_level2_keeps_it_fast() -> None:
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    assert len(islands) == 2 and all(i.size == 2 for i in islands)
    cand = _aggregated_candidate([islands[0].id, islands[1].id], tp=2)

    cfg1, _ = _compile(cluster, islands, _profiles(2), cand, level=1)
    cfg2, _ = _compile(cluster, islands, _profiles(2), cand, level=2)

    # Level 1: one scalar, the global bottleneck (Ethernet) — intra TP penalized.
    assert cfg1["link_bw"] == ETH_BW
    assert cfg1["link_latency"] == ETH_LAT
    # Level 2: [intra=NVLink, cross=Ethernet] — intra TP keeps its real bandwidth.
    assert cfg2["link_bw"] == [NVLINK_BW, ETH_BW]
    assert cfg2["link_latency"] == [NVLINK_LAT, ETH_LAT]


# --- dim-count contract with config_builder -------------------------------


def test_emitted_list_length_matches_compute_network_dims() -> None:
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    cand = _aggregated_candidate([islands[0].id, islands[1].id], tp=2)
    cfg, _ = _compile(cluster, islands, _profiles(2), cand, level=2)

    flattened = [inst for node in cfg["nodes"] for inst in node["instances"]]
    num_dims = len(_compute_network_dims(flattened))
    assert isinstance(cfg["link_bw"], list)
    assert len(cfg["link_bw"]) == num_dims
    # config_builder must accept exactly what we emit (no length mismatch raise).
    normalized = _normalize_network_dim_values(cfg["link_bw"], num_dims, "link_bw")
    assert list(normalized) == [NVLINK_BW, ETH_BW]


# --- single-island: byte-identical between levels --------------------------


def test_single_island_is_identical_between_levels() -> None:
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    cand = _aggregated_candidate([islands[0].id], tp=2)  # one island only

    cfg1, _ = _compile(cluster, islands, _profiles(2), cand, level=1)
    cfg2, _ = _compile(cluster, islands, _profiles(2), cand, level=2)
    # No cross hop -> intra serves the single dimension -> identical scalar.
    assert cfg1["link_bw"] == cfg2["link_bw"] == NVLINK_BW
    assert cfg1["link_latency"] == cfg2["link_latency"] == NVLINK_LAT


# --- inert intra dim (all tp=1) -------------------------------------------


def test_all_tp1_multi_instance_emits_finite_inert_intra() -> None:
    cluster = _two_island_cluster(1)  # single-GPU nodes -> size-1 islands
    islands = detect_islands(cluster, _profiles(1))
    assert all(i.size == 1 for i in islands)
    cand = _aggregated_candidate([islands[0].id, islands[1].id], tp=1)

    cfg, _ = _compile(cluster, islands, _profiles(1), cand, level=2)
    # intra dim has size 1 (no collective); it must still be a finite placeholder.
    assert isinstance(cfg["link_bw"], list)
    intra_bw = cfg["link_bw"][0]
    assert intra_bw != float("inf") and intra_bw > 0
    assert cfg["link_bw"][1] == ETH_BW  # cross is the real Ethernet hop
    perdim = TopologyGraph(cluster).reduce_for_simulator_perdim(
        [{i.id: i for i in islands}[a.island_id] for a in cand.assignments]
    )
    assert any("inert placeholder" in a for a in perdim.assumptions)


# --- same-island pairs contribute no cross hop ----------------------------


def test_same_island_pair_yields_no_cross() -> None:
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    isl = islands[0]
    # Two assignments on the SAME island (e.g. a same-island P/D or DP split).
    perdim = TopologyGraph(cluster).reduce_for_simulator_perdim([isl, isl])
    assert perdim.cross_bw_gbps is None and perdim.cross_lat_ns is None
    assert perdim.intra_bw_gbps == NVLINK_BW


# --- provenance honesty ----------------------------------------------------


def test_level2_provenance_does_not_overclaim() -> None:
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    perdim = TopologyGraph(cluster).reduce_for_simulator_perdim(list(islands))
    prov = perdim.as_provenance()
    assert prov["model_level"] == 2
    assert prov["resolution"] == "per-dimension"
    assert prov["path_aware"] is False          # dimension-resolved != path-aware
    assert prov["contention_modeled"] is False  # analytical backend can't (D3)
    assert prov["link_bw_gbps_perdim"] == [NVLINK_BW, ETH_BW]


# --- reproducibility -------------------------------------------------------


def test_level2_compile_is_deterministic() -> None:
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    cand = _aggregated_candidate([islands[0].id, islands[1].id], tp=2)
    outs = [_compile(cluster, islands, _profiles(2), cand, level=2)[0]["link_bw"]
            for _ in range(3)]
    assert outs[0] == outs[1] == outs[2] == [NVLINK_BW, ETH_BW]


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0]])
def test_config_builder_rejects_wrong_length_list(bad) -> None:
    """The contract we must never violate: a mismatched list is a hard error."""
    with pytest.raises(ValueError):
        _normalize_network_dim_values(bad, 2, "link_bw")


# --- dim-count contract across the flagged shapes -------------------------
# The crash risk is emitting a list whose length != config_builder's dim count.
# _compute_network_dims doubles npus/pp for prefill instances, so P/D and pp>1
# are the shapes most likely to break the contract; assert them at the compile
# boundary (the emitted list must be accepted by config_builder unchanged).


def _assert_perdim_contract(cluster, islands_list, profiles, candidate):
    cfg, _ = _compile(cluster, islands_list, profiles, candidate, level=2)
    flattened = [inst for node in cfg["nodes"] for inst in node["instances"]]
    num_dims = len(_compute_network_dims(flattened))
    # The contract: a scalar broadcasts to any dim count; a LIST must match the
    # dim count exactly or config_builder raises. Both validated by normalizing
    # with the real dim count (raises on a length mismatch).
    _normalize_network_dim_values(cfg["link_bw"], num_dims, "link_bw")
    _normalize_network_dim_values(cfg["link_latency"], num_dims, "link_latency")
    if isinstance(cfg["link_bw"], list):
        assert len(cfg["link_bw"]) == num_dims
    return cfg["link_bw"]


def test_perdim_contract_cross_island_pd_split() -> None:
    """P/D prefill-doubling in _compute_network_dims is the top crash risk."""
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    cand = CandidateConfig(
        id="pd", model=MODEL, dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=islands[0].id, role=Role.PREFILL, tp_size=1),
            IslandAssignment(island_id=islands[1].id, role=Role.DECODE, tp_size=1),
        ],
        serving_arch=ServingArch.PD_SPLIT, knobs=VllmKnobs(),
    )
    link_bw = _assert_perdim_contract(cluster, islands, _profiles(2), cand)
    # Distinct islands -> a real cross hop is present.
    assert isinstance(link_bw, list) and link_bw[-1] == ETH_BW


def test_perdim_contract_pp_greater_than_one() -> None:
    """pp>1 also feeds the total_pp branch of _compute_network_dims."""
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    cand = CandidateConfig(
        id="pp", model=MODEL, dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=islands[0].id, role=Role.AGGREGATED,
                             tp_size=1, pp_size=2),
        ],
        serving_arch=ServingArch.AGGREGATED, knobs=VllmKnobs(),
    )
    _assert_perdim_contract(cluster, islands, _profiles(2), cand)


def test_perdim_contract_dp_replicas_same_island() -> None:
    """DP replicas make a 2-dim topology on ONE island: cross must fall back to a
    scalar (no distinct-island hop), which broadcasts safely to both dims."""
    cluster = _two_island_cluster(2)
    islands = detect_islands(cluster, _profiles(2))
    cand = CandidateConfig(
        id="dp", model=MODEL, dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=islands[0].id, role=Role.AGGREGATED,
                             tp_size=1, dp_replicas=2),
        ],
        serving_arch=ServingArch.AGGREGATED, knobs=VllmKnobs(),
    )
    link_bw = _assert_perdim_contract(cluster, islands, _profiles(2), cand)
    # One island -> no cross hop -> scalar, broadcast to whatever dim count.
    assert not isinstance(link_bw, list)
