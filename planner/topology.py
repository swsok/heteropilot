"""Topology graph and the Level-1 bandwidth model (work order §5.3).

Two-level policy. **Level 1** (this module, Phase 2) scores candidates in bulk
using representative values per interconnect class. **Level 2** — real paths with
`contention_group` bandwidth sharing — needs the `config_builder.py` work that
§7 unlocks at Phase 5, because the simulator's cluster config has nowhere to put
a link graph at all (docs/deviations.md D3).

That makes the compile step lossy by construction: an arbitrary graph collapses
to the scalar `link_bw` / `link_latency` pair the simulator accepts. This module
performs that reduction explicitly and records what it did, so no result is ever
mistaken for a path-aware one.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

from planner.inventory import ClusterSpecV2, ExecutionIsland, Link, LinkType

#: Representative Level-1 values per interconnect class, used only when the
#: cluster spec has no link describing a hop. Every one is a placeholder: no
#: bandwidth here was measured on this machine (absolute rule 3). They exist so
#: a spec with an incomplete graph still scores rather than crashing, and their
#: use is recorded in `TopologyReduction.assumptions`.
CLASS_DEFAULT_GBPS: dict[LinkType, float] = {
    LinkType.NVLINK: 900.0,
    LinkType.HCCS: 392.0,
    LinkType.PCIE: 64.0,
    LinkType.INFINIBAND: 400.0,
    LinkType.ETHERNET: 10.0,
}

CLASS_DEFAULT_LATENCY_NS: dict[LinkType, float] = {
    LinkType.NVLINK: 1000.0,
    LinkType.HCCS: 1200.0,
    LinkType.PCIE: 1500.0,
    LinkType.INFINIBAND: 5000.0,
    LinkType.ETHERNET: 20000.0,
}


class TopologyError(ValueError):
    """Raised when the topology cannot answer a question it was asked."""


@dataclass(frozen=True)
class TopologyReduction:
    """The scalar pair handed to the simulator, plus how it was derived."""

    link_bw_gbps: float
    link_latency_ns: float
    basis: str
    assumptions: list[str] = field(default_factory=list)

    def as_provenance(self) -> dict[str, object]:
        return {
            "link_bw_gbps": self.link_bw_gbps,
            "link_latency_ns": self.link_latency_ns,
            "basis": self.basis,
            "assumptions": list(self.assumptions),
            "model_level": 1,
            "path_aware": False,
        }


@dataclass(frozen=True)
class PerDimReduction:
    """Per-ASTRA-dimension bandwidth/latency for the Level-2 compile (§5.3).

    ASTRA-Sim's network config is dimensional, not a link graph (deviations D3):
    dim 0 is the intra-group (TP) FullyConnected dimension, dim 1 (when present)
    is the cross-instance dimension. Level 1 collapses both into one scalar
    bottleneck, so a fast intra-island interconnect (e.g. NVLink) is dragged down
    by a slow cross-instance fabric. This keeps them apart: each dimension gets
    its own bottleneck. `cross_*` is None when the placement never crosses to a
    distinct island, in which case the intra value serves both dimensions and the
    result is byte-identical to Level 1.

    This is dimension-resolved, NOT per-flow path-aware: the analytical backend
    cannot represent `contention_group` sharing, so it is dropped here just as in
    Level 1 (`contention_modeled` stays False).
    """

    intra_bw_gbps: float
    intra_lat_ns: float
    cross_bw_gbps: float | None
    cross_lat_ns: float | None
    assumptions: list[str] = field(default_factory=list)

    def as_provenance(self) -> dict[str, object]:
        perdim_bw = [self.intra_bw_gbps]
        perdim_lat = [self.intra_lat_ns]
        # cross_bw_gbps and cross_lat_ns are always both set or both None.
        if self.cross_bw_gbps is not None and self.cross_lat_ns is not None:
            perdim_bw.append(self.cross_bw_gbps)
            perdim_lat.append(self.cross_lat_ns)
        return {
            "model_level": 2,
            "resolution": "per-dimension",
            # Still False: dimension-resolved is a weaker, honest claim than
            # per-flow path-aware, which ASTRA-Sim's analytical backend cannot do.
            "path_aware": False,
            "contention_modeled": False,
            "link_bw_gbps_perdim": perdim_bw,
            "link_latency_ns_perdim": perdim_lat,
            "dim_semantics": {
                "0": "intra-group (TP all-reduce/all-gather)",
                "1": "cross-instance (PP / independent instances)",
            },
            "basis": (
                "Level-2 per-dimension bottleneck: intra = min bandwidth / max "
                "latency over the selected islands' island_interconnect; cross = "
                "min bandwidth / max latency over the inter-island paths. "
                "Dimension-resolved, not per-flow path-aware: ASTRA-Sim's "
                "dimensional model cannot represent contention_group (deviations D3)."
            ),
            "assumptions": list(self.assumptions),
        }


class TopologyGraph:
    """Undirected graph over `<node>/<device>` endpoints.

    networkx would do, but the graph is tiny and a hand-rolled BFS keeps the
    traversal order deterministic, which matters for §9 reproducibility.
    """

    def __init__(self, cluster: ClusterSpecV2) -> None:
        self.cluster = cluster
        self._adj: dict[str, list[tuple[str, Link]]] = {}
        for link in cluster.links:
            self._adj.setdefault(link.src, []).append((link.dst, link))
            self._adj.setdefault(link.dst, []).append((link.src, link))
        # Deterministic neighbour order regardless of link declaration order.
        for endpoint in self._adj:
            self._adj[endpoint].sort(key=lambda pair: (pair[0], pair[1].id))

    def endpoints(self) -> list[str]:
        return sorted(self._adj)

    def path(self, src: str, dst: str) -> list[Link]:
        """Fewest-hop path as a link list. Empty when src == dst."""
        if src == dst:
            return []
        if src not in self._adj or dst not in self._adj:
            raise TopologyError(f"unknown endpoint in path({src!r}, {dst!r})")

        prev: dict[str, tuple[str, Link]] = {}
        seen = {src}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt, link in self._adj[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (cur, link)
                if nxt == dst:
                    return self._rebuild(prev, src, dst)
                queue.append(nxt)
        raise TopologyError(f"no path between {src!r} and {dst!r}")

    @staticmethod
    def _rebuild(prev: dict[str, tuple[str, Link]], src: str, dst: str) -> list[Link]:
        out: list[Link] = []
        cur = dst
        while cur != src:
            cur, link = prev[cur]
            out.append(link)
        out.reverse()
        return out

    def connected(self, src: str, dst: str) -> bool:
        try:
            self.path(src, dst)
        except TopologyError:
            return False
        return True

    @staticmethod
    def effective_bandwidth_gbps(
        path: list[Link], concurrent_flows: dict[str, int] | None = None
    ) -> float:
        """Bottleneck bandwidth along a path, GB/s.

        Links sharing a `contention_group` split their nominal bandwidth by the
        number of active flows in that group - the simple model §5.3 specifies
        for Level 1. Links with no group are assumed uncontended.
        """
        if not path:
            return float("inf")
        flows = concurrent_flows or {}
        per_link = []
        for link in path:
            share = flows.get(link.contention_group, 1) if link.contention_group else 1
            per_link.append(link.bandwidth_gbps / max(1, share))
        return min(per_link)

    @staticmethod
    def path_latency_ns(path: list[Link]) -> float:
        return sum(link.latency_ns for link in path)

    @classmethod
    def transfer_time_ns(
        cls, num_bytes: float, path: list[Link], concurrent_flows: dict[str, int] | None = None
    ) -> float:
        if num_bytes <= 0:
            return 0.0
        bw = cls.effective_bandwidth_gbps(path, concurrent_flows)
        if bw == float("inf"):
            return 0.0
        # GB/s -> bytes/ns is a factor of 1e9 bytes per 1e9 ns, i.e. 1:1.
        return cls.path_latency_ns(path) + num_bytes / bw

    @staticmethod
    def transfer_energy_j(num_bytes: float, path: list[Link]) -> tuple[float, list[str]]:
        """Energy for moving `num_bytes`, plus warnings for links lacking data.

        A link with no `energy_per_bit_pj` contributes zero and is reported.
        Silently treating unknown as free would understate energy exactly where
        the topology matters most.
        """
        warnings: list[str] = []
        total_pj = 0.0
        bits = num_bytes * 8
        for link in path:
            if link.energy_per_bit_pj is None:
                warnings.append(f"link {link.id} has no energy_per_bit_pj; counted as 0")
                continue
            total_pj += bits * link.energy_per_bit_pj
        return total_pj * 1e-12, warnings

    # -- Level 1 -----------------------------------------------------------

    def island_interconnect(self, island: ExecutionIsland) -> tuple[float, float, list[str]]:
        """Representative (bandwidth GB/s, latency ns) inside one island."""
        assumptions: list[str] = []
        members = [f"{island.node_id}/{a}" for a in island.accelerator_ids]

        observed = [
            link
            for link in self.cluster.links
            if link.src in members and link.dst in members
        ]
        if observed:
            bw = min(link.bandwidth_gbps for link in observed)
            lat = max(link.latency_ns for link in observed)
            return bw, lat, assumptions

        if island.size == 1:
            # Nothing crosses a wire inside a one-device island.
            return float("inf"), 0.0, assumptions

        kind = island.interconnect_type or LinkType.PCIE
        assumptions.append(
            f"island {island.id}: no link in the spec connects its accelerators; "
            f"used the {kind.value} class default "
            f"({CLASS_DEFAULT_GBPS[kind]} GB/s, {CLASS_DEFAULT_LATENCY_NS[kind]} ns), "
            f"source=placeholder"
        )
        return CLASS_DEFAULT_GBPS[kind], CLASS_DEFAULT_LATENCY_NS[kind], assumptions

    def reduce_for_simulator(self, islands: list[ExecutionIsland]) -> TopologyReduction:
        """Collapse the graph to the scalar pair the simulator accepts (D3).

        Bottleneck semantics: the slowest interconnect any selected island relies
        on, and the highest latency. Taking the minimum rather than an average is
        deliberate - an average would let a fast island paper over a slow one and
        quietly produce optimistic predictions.
        """
        if not islands:
            raise TopologyError("cannot reduce topology for an empty island set")

        assumptions: list[str] = []
        bandwidths: list[float] = []
        latencies: list[float] = []
        for island in islands:
            bw, lat, notes = self.island_interconnect(island)
            assumptions.extend(notes)
            bandwidths.append(bw)
            latencies.append(lat)

        if len(islands) > 1:
            # Multi-island placements also cross the fabric between nodes.
            for a, b in itertools.pairwise(islands):
                bw, lat, notes = self._inter_island(a, b)
                assumptions.extend(notes)
                bandwidths.append(bw)
                latencies.append(lat)

        finite = [b for b in bandwidths if b != float("inf")]
        bw = min(finite) if finite else CLASS_DEFAULT_GBPS[LinkType.NVLINK]
        if not finite:
            assumptions.append(
                "no finite bandwidth anywhere in the selection (all single-device "
                "islands); used the NVLINK class default, source=placeholder"
            )
        lat = max(latencies) if latencies else 0.0

        basis = (
            f"Level-1 bottleneck over {len(islands)} island(s): min bandwidth, max latency. "
            "The simulator has no link graph, so contention_group and per-link energy "
            "are dropped here (deviations D3)."
        )
        return TopologyReduction(bw, lat, basis, assumptions)

    def reduce_for_simulator_perdim(
        self, islands: list[ExecutionIsland]
    ) -> PerDimReduction:
        """Level-2 reduction: separate intra-TP and cross-instance bottlenecks.

        Unlike `reduce_for_simulator`, which collapses everything into one scalar,
        this keeps the two ASTRA-Sim dimensions apart. `islands` is the per-
        assignment list, so it repeats the same island when two assignments land
        on it (a same-island P/D split); DP replicas live inside one assignment
        and do not repeat. Only pairs that actually cross to a distinct island
        contribute a cross bottleneck. When none do, `cross_*` is None and the
        caller emits a scalar, so single-island placements stay identical to
        Level 1 (deviations D3).
        """
        if not islands:
            raise TopologyError("cannot reduce topology for an empty island set")

        assumptions: list[str] = []
        intra_bws: list[float] = []
        intra_lats: list[float] = []
        for island in islands:
            bw, lat, notes = self.island_interconnect(island)
            assumptions.extend(notes)
            intra_bws.append(bw)
            intra_lats.append(lat)

        finite_intra = [b for b in intra_bws if b != float("inf")]
        if finite_intra:
            intra_bw = min(finite_intra)
        else:
            # Every selected island is a single device (tp=1): the intra/TP
            # dimension carries no collective. ASTRA-Sim still needs a finite
            # bandwidth for a dimension that will see no traffic; emit the NVLINK
            # class default as an inert placeholder and record that it is inert.
            intra_bw = CLASS_DEFAULT_GBPS[LinkType.NVLINK]
            assumptions.append(
                "intra/TP dimension has size 1 on every selected island "
                f"(no collective); emitted the NVLINK class default {intra_bw} GB/s "
                "as an inert placeholder, source=placeholder"
            )
        intra_lat = max(intra_lats) if intra_lats else 0.0

        # Cross bottleneck: only pairs whose path is non-empty, i.e. that cross to
        # a distinct island. A same-island pair (DP replicas, same-island P/D)
        # yields an empty path -> inf -> no cross hop, so it is excluded here.
        cross_bw: float | None = None
        cross_lat: float | None = None
        finite_cross: list[tuple[float, float]] = []
        for a, b in itertools.pairwise(islands):
            bw, lat, notes = self._inter_island(a, b)
            assumptions.extend(notes)
            if bw != float("inf"):
                finite_cross.append((bw, lat))
        if finite_cross:
            cross_bw = min(bw for bw, _ in finite_cross)
            cross_lat = max(lat for _, lat in finite_cross)

        return PerDimReduction(intra_bw, intra_lat, cross_bw, cross_lat, assumptions)

    def _inter_island(
        self, a: ExecutionIsland, b: ExecutionIsland
    ) -> tuple[float, float, list[str]]:
        assumptions: list[str] = []
        src = f"{a.node_id}/{a.accelerator_ids[0]}"
        dst = f"{b.node_id}/{b.accelerator_ids[0]}"
        try:
            path = self.path(src, dst)
        except TopologyError:
            assumptions.append(
                f"islands {a.id} and {b.id} are not connected in the spec; used the "
                f"INFINIBAND class default "
                f"({CLASS_DEFAULT_GBPS[LinkType.INFINIBAND]} GB/s), source=placeholder"
            )
            return (
                CLASS_DEFAULT_GBPS[LinkType.INFINIBAND],
                CLASS_DEFAULT_LATENCY_NS[LinkType.INFINIBAND],
                assumptions,
            )
        return (
            self.effective_bandwidth_gbps(path),
            self.path_latency_ns(path),
            assumptions,
        )
