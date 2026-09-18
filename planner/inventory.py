"""ClusterSpecV2, accelerator profiles, and execution-island detection.

Work order §3.2, §3.3, §5.2. This is the planner's own rich view of a cluster;
it is compiled down to the simulator's flat JSON only at prediction time
(see docs/deviations.md D3 for what that compilation loses).
"""

from __future__ import annotations

import enum
import fnmatch
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# vLLM reserves a fraction of VRAM and then subtracts weights, activation peak
# and CUDA-graph capture. The simulator's memory model does none of that
# (docs/deviations.md D10), so the planner derates explicitly and records it.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_ACTIVATION_RESERVE_GB = 0.0


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AcceleratorType(str, enum.Enum):
    GPU = "GPU"
    NPU = "NPU"


class AcceleratorState(str, enum.Enum):
    FREE = "FREE"
    ALLOCATED = "ALLOCATED"
    RESERVED = "RESERVED"
    DEGRADED = "DEGRADED"


class Health(str, enum.Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


class PowerState(str, enum.Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    SLEEP = "SLEEP"


class LinkType(str, enum.Enum):
    NVLINK = "NVLINK"
    PCIE = "PCIE"
    INFINIBAND = "INFINIBAND"
    ETHERNET = "ETHERNET"
    HCCS = "HCCS"
    #: Vendor-neutral on-package fabric between processing elements of one
    #: accelerator die/board, e.g. the 8 PEs of a FuriosaAI RNGD card. Needed
    #: because NVLINK and HCCS are vendor-specific names and PCIE is simply
    #: wrong for elements that never leave the package - mislabelling it PCIE
    #: would also feed the topology model the wrong interconnect class. See
    #: docs/deviations.md D16.
    ONPACKAGE = "ONPACKAGE"


class Source(str, enum.Enum):
    """Provenance of a hardware number. Absolute rule 3: never mislabel."""

    MEASURED = "measured"
    VENDOR_SPEC = "vendor_spec"
    PLACEHOLDER = "placeholder"
    #: A number typed in by a user (ScenarioLab cluster builder). Distinct
    #: from PLACEHOLDER: it is a deliberate what-if input, not a stand-in
    #: awaiting measurement - but it is still not a measurement and must be
    #: displayed as user_defined wherever it flows.
    USER_DEFINED = "user_defined"


#: Link types that keep accelerators inside one execution island. Anything else
#: (INFINIBAND, ETHERNET) crosses an island boundary even within a node.
INTRA_ISLAND_LINKS = frozenset({
    LinkType.NVLINK, LinkType.PCIE, LinkType.HCCS, LinkType.ONPACKAGE,
})


#: What traffic the bandwidth was measured (or is being asked) for. A link does
#: not have one bandwidth: on the A40 node's PCIe bridge a direct
#: device-to-device copy sustains 25.0 GB/s, a two-rank all-reduce over the same
#: path 19.3, and the four-rank all-reduce the TP group actually runs 8.8
#: (experiments/p2_evidence/results/v3_verdict_accuracy.md A.1). Asking for the
#: wrong one is how V3's simulator read a 64.0 GB/s datasheet number for a hop
#: that delivered an eighth of it - domain-scoping S3, deviations D112.
Collective = Literal["all_reduce", "all_gather", "p2p"]

#: The message-size band the figure belongs to. Collective bandwidth is not
#: scale-free: on that same path the measured all-reduce busbw climbs from 0.23
#: GB/s at 8 KiB to a plateau of 8.8-9.1 from 4 MiB up (14 points, PR #100).
#: `bulk` is the plateau and the only band that answers the simulator's
#: question, whose `link_bw` is an asymptotic bytes/bandwidth term with the
#: small-message cost carried separately by `link_latency`.
#:   small - below 1 MiB, latency-dominated and far below plateau
#:   mid   - 1 MiB to below 4 MiB, still climbing
#:   bulk  - 4 MiB and above; 4-64 MiB agree to 4 % on the measured path
MsgSizeClass = Literal["small", "mid", "bulk"]

#: Byte bounds of each band, as [lo, hi). Used to classify a measurement's own
#: sweep and to reject a `msg_size_class` that its `msg_bytes` contradicts.
MSG_SIZE_CLASS_BYTES: dict[str, tuple[int, float]] = {
    "small": (0, 1 << 20),
    "mid": (1 << 20, 4 << 20),
    "bulk": (4 << 20, float("inf")),
}


def msg_size_class_of(num_bytes: float) -> MsgSizeClass:
    """Which band a message size falls in."""
    for name, (lo, hi) in MSG_SIZE_CLASS_BYTES.items():
        if lo <= num_bytes < hi:
            return name  # type: ignore[return-value]
    raise ValueError(f"message size {num_bytes} falls in no band")


class LinkMeasurement(_Strict):
    """One measured effective bandwidth of a link, with the conditions it holds under.

    The link's own `bandwidth_gbps` stays whatever its datasheet says and is
    never edited by anything here (absolute rule A3): a measurement is
    *additional* provenance beside the spec value, so the two can be compared
    rather than one silently replacing the other.

    Every field of the key - `collective`, `world_size`, `msg_size_class`,
    `binding` - is a condition the figure holds under, following the same
    convention S1 gave an accuracy domain (D110): a condition this measurement
    does not state cannot be used to refuse a deployment, so `binding: unknown`
    is compared against nothing and is not the same claim as `unpinned`.
    """

    collective: Collective
    msg_size_class: MsgSizeClass
    #: How the measuring process was bound to the node, in `Node.device_binding`'s
    #: vocabulary. `unknown` means the harness did not record it.
    binding: Literal["numa_pinned", "unpinned", "unknown"] = "unknown"
    #: Effective bus bandwidth, GB/s - the figure a collective actually achieves,
    #: not the wire rate. Compared against `Link.bandwidth_gbps`, same unit.
    bus_bw_gbps: float = Field(gt=0)
    #: How many ranks took part. A two-rank and a four-rank all-reduce over one
    #: path are different measurements (19.3 against 8.8 on the A40 bridge).
    world_size: int = Field(default=2, ge=2)
    source: Source = Source.MEASURED
    #: The tool and its version, e.g. "torch 2.10 + NCCL 2.27.5 under torchrun".
    #: Not free text for its own sake: `nccl-tests all_reduce_perf` and this
    #: repo's torch probe are not interchangeable evidence (S6(ii) is still open
    #: for exactly that reason).
    method: str = ""
    #: The sizes actually swept, bytes. Checked against `msg_size_class`.
    msg_bytes: list[int] = Field(default_factory=list)
    date: str = ""
    #: Path to the raw artefact, repo-relative.
    raw: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _bytes_match_class(self) -> LinkMeasurement:
        lo, hi = MSG_SIZE_CLASS_BYTES[self.msg_size_class]
        for size in self.msg_bytes:
            if not lo <= size < hi:
                raise ValueError(
                    f"LinkMeasurement: msg_size_class={self.msg_size_class!r} covers "
                    f"[{lo}, {hi}) bytes but msg_bytes lists {size}; a figure "
                    f"labelled with the wrong band would be selected for traffic it "
                    f"was never measured at"
                )
        if self.source is Source.MEASURED and not self.method.strip():
            raise ValueError(
                "LinkMeasurement: source=measured must name its `method` - an "
                "unattributed number is not a measurement (absolute rule 3)"
            )
        return self

    @property
    def key(self) -> tuple[str, int, str, str]:
        """What this figure answers for, and nothing else.

        The work order names `(link_id, collective, msg_size_class,
        device_binding)`. `world_size` is a fourth component this repository's
        own data forces: the A40 bridge measures 8.8 GB/s for a four-rank
        all-reduce and 19.29 for a two-rank one, both `bulk`, both `unpinned`,
        both from the same run. Without it the two collide and one silently
        answers for the other - which is the substitution the key exists to
        stop. Recorded as a deviation in D112 rather than left implicit.
        """
        return (self.collective, self.world_size, self.msg_size_class, self.binding)


class Nic(_Strict):
    id: str
    type: str
    speed_gbps: float = Field(gt=0)


class Accelerator(_Strict):
    id: str
    type: AcceleratorType
    vendor: str
    model: str
    backend: str
    memory_gb: float = Field(gt=0)
    state: AcceleratorState = AcceleratorState.FREE
    profile: str | None = None

    # Dynamic fields, populated by the Phase 4 monitor. Optional until then.
    utilization: float | None = Field(default=None, ge=0.0, le=1.0)
    memory_used_gb: float | None = Field(default=None, ge=0.0)
    health: Health | None = None
    power_state: PowerState | None = None
    queue_depth: int | None = Field(default=None, ge=0)

    @property
    def is_free(self) -> bool:
        return self.state == AcceleratorState.FREE


class NodePower(_Strict):
    """Node-level power components for the simulator's `power:` block.

    These describe the *host*, not the accelerator, so per deviations D7 they
    live on the node rather than in `profiles/accelerators/*.yaml`. The
    accelerator's own idle/standby/active figures come from its profile.

    Defaults are copied from upstream's `single_node_power_instance.json`.
    Upstream documents no derivation for them, so they are `placeholder`, never
    `measured` - a plan whose energy came from these numbers must say so.
    """

    base_node_power: float = Field(default=60.0, ge=0)
    cpu_idle_power: float = Field(default=10.0, ge=0)
    cpu_active_power: float = Field(default=200.0, ge=0)
    cpu_util: float = Field(default=0.15, ge=0, le=1)
    dram_dimm_size: float = Field(default=32.0, gt=0)
    dram_idle_power: float = Field(default=2.0, ge=0)
    dram_energy_per_bit: float = Field(default=6.0, ge=0)
    link_num_links: int = Field(default=1, ge=0)
    link_idle_power: float = Field(default=5.0, ge=0)
    link_energy_per_bit: float = Field(default=4.0, ge=0)
    nic_num_nics: int = Field(default=1, ge=0)
    nic_idle_power: float = Field(default=20.0, ge=0)
    storage_num_devices: int = Field(default=2, ge=0)
    storage_idle_power: float = Field(default=5.0, ge=0)
    source: Source = Source.PLACEHOLDER


class Node(_Strict):
    id: str
    accelerators: list[Accelerator] = Field(min_length=1)
    nics: list[Nic] = Field(default_factory=list)
    #: Host memory, needed by the simulator's `cpu_mem` block. Defaults match
    #: upstream's bundled configs; override when the real host differs.
    cpu_memory_gb: float = Field(default=512.0, gt=0)
    cpu_memory_bw_gbps: float = Field(default=256.0, gt=0)
    cpu_memory_latency_ns: float = Field(default=0.0, ge=0)
    #: Omit to run without energy output. Present means the compiler emits a
    #: `power:` block and the simulator reports energy (deviations D2).
    power: NodePower | None = None
    #: How a serving process on this node is bound to it - `numa_pinned` when
    #: the deployment pins CPU and memory to the NUMA node the accelerators sit
    #: on, `unpinned` when it does not. An APPLICATION CONDITION of an accuracy
    #: domain (domain-scoping S1, D110) so that a domain measured unbound is not
    #: consulted for a deployment that pins, or the reverse. Since S3 (D112) it
    #: is also read on the way IN: it selects which of a link's measured
    #: effective bandwidths the simulator is given, because the same link
    #: measured bound and unbound are two different figures. It is worth 1.93x
    #: of throughput on the A40 node's second NUMA group
    #: (experiments/p2_evidence/results/v3_verdict_accuracy.md A.3), which is why
    #: it is a condition rather than a note. `unknown` - the default, and what
    #: every committed cluster says today - means the match test skips it and
    #: flags it rather than assuming either state.
    device_binding: Literal["numa_pinned", "unpinned", "unknown"] = "unknown"

    @model_validator(mode="after")
    def _unique_ids(self) -> Node:
        _reject_duplicates([a.id for a in self.accelerators], f"node {self.id} accelerator")
        _reject_duplicates([n.id for n in self.nics], f"node {self.id} nic")
        return self


class Link(_Strict):
    id: str
    src: str
    dst: str
    type: LinkType
    bandwidth_gbps: float = Field(gt=0)
    latency_ns: float = Field(ge=0)
    energy_per_bit_pj: float | None = Field(default=None, ge=0)
    duplex: str = "full"
    contention_group: str | None = None
    source: Source = Source.PLACEHOLDER
    #: Measured effective bandwidths for this link, one per
    #: `(collective, msg_size_class, binding)`. `bandwidth_gbps` above is left
    #: alone whatever these say (A3); `effective_bandwidth_gbps` picks between
    #: them. Empty - what every committed fixture said before S3 - means the
    #: spec value is all there is, and the registry keeps grading it at
    #: `source` so it stays in the measurement queue.
    measurements: list[LinkMeasurement] = Field(default_factory=list)

    @model_validator(mode="after")
    def _endpoint_format(self) -> Link:
        for role, value in (("src", self.src), ("dst", self.dst)):
            if value.count("/") != 1:
                raise ValueError(
                    f"link {self.id}: {role}='{value}' must be '<node_id>/<device_or_nic_id>'"
                )
        if self.src == self.dst:
            raise ValueError(f"link {self.id}: src and dst are the same endpoint '{self.src}'")
        return self

    @property
    def endpoints(self) -> tuple[tuple[str, str], tuple[str, str]]:
        (sn, sd), (dn, dd) = self.src.split("/"), self.dst.split("/")
        return (sn, sd), (dn, dd)

    @model_validator(mode="after")
    def _one_measurement_per_key(self) -> Link:
        seen: set[tuple[str, int, str, str]] = set()
        for m in self.measurements:
            if m.key in seen:
                raise ValueError(
                    f"link {self.id}: two measurements for {m.collective}/w"
                    f"{m.world_size}/{m.msg_size_class}/{m.binding}; the key exists "
                    f"so one figure answers for one set of conditions"
                )
            seen.add(m.key)
        return self

    def measurement_for(
        self,
        collective: Collective,
        msg_size_class: MsgSizeClass,
        binding: str,
        *,
        world_size: int | None = None,
    ) -> LinkMeasurement | None:
        """The measurement that answers for these conditions, or None.

        `collective` and `msg_size_class` must match exactly - a p2p copy is not
        an all-reduce and a 64 KiB figure does not answer for 16 MiB, and
        substituting either is the category error D112 records.

        `binding` follows S1's convention (D110): `unknown` on either side is
        *not stated*, so it is not compared. The caller is told which fields
        went unchecked through `effective_bandwidth_gbps`'s note, because an
        unchecked condition is not a satisfied one.

        `world_size` is part of the key and must match exactly: the planner asks
        with the TP degree it is about to simulate, and a four-rank figure
        standing in for a two-rank one understates a small group by 2.2x on the
        measured path. `None` means the caller states no group size, which
        selects any - only `island_interconnect`'s island-size default and
        direct callers do that, and both record which figure they got.
        """
        best: LinkMeasurement | None = None
        for m in self.measurements:
            if m.collective != collective or m.msg_size_class != msg_size_class:
                continue
            if world_size is not None and m.world_size != world_size:
                continue
            stated_both = m.binding != "unknown" and binding != "unknown"
            if stated_both and m.binding != binding:
                continue
            # A figure whose binding is stated AND agrees is strictly better
            # evidence than one that leaves it unstated, so prefer it.
            if best is None or (m.binding != "unknown" and best.binding == "unknown"):
                best = m
        return best


class ClusterSpecV2(_Strict):
    cluster_id: str
    nodes: list[Node] = Field(min_length=1)
    links: list[Link] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> ClusterSpecV2:
        _reject_duplicates([n.id for n in self.nodes], "node")
        _reject_duplicates([link.id for link in self.links], "link")
        known: set[str] = set()
        for node in self.nodes:
            known.update(f"{node.id}/{a.id}" for a in node.accelerators)
            known.update(f"{node.id}/{n.id}" for n in node.nics)
        for link in self.links:
            for role, value in (("src", link.src), ("dst", link.dst)):
                if value not in known:
                    raise ValueError(
                        f"link {link.id}: {role}='{value}' does not name any accelerator or nic"
                    )
        return self

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"no node '{node_id}' in cluster {self.cluster_id}")

    def accelerator(self, node_id: str, accel_id: str) -> Accelerator:
        for a in self.node(node_id).accelerators:
            if a.id == accel_id:
                return a
        raise KeyError(f"no accelerator '{accel_id}' on node '{node_id}'")


class SupportedModel(_Strict):
    pattern: str
    dtypes: list[str] = Field(min_length=1)


class ProfilePower(_Strict):
    """Mirrors the simulator's per-node `power.npu.<hardware>` block one-to-one.

    Deviation D7: the work order's profile schema (tdp_w / idle_power_w) does not
    map onto what the simulator needs, and standby_power / standby_duration have
    no source in it at all. Field names here match the simulator exactly so the
    compiler is a copy, not a derivation.
    """

    idle_power: float = Field(ge=0)
    standby_power: float = Field(ge=0)
    active_power: float = Field(ge=0)
    standby_duration: float = Field(ge=0)
    source: Source = Source.PLACEHOLDER


class PowerModelPoint(_Strict):
    """One measured operating point: a wattage and the load it was taken at.

    A5(c): a power figure without the utilisation it was taken at is not a
    measurement, so `util_pct` is required even when the model is not keyed on it.
    """

    power_w: float = Field(ge=0)
    util_pct: float = Field(ge=0, le=100)
    served_conc: float | None = Field(default=None, gt=0)
    tput_tok_s: float | None = Field(default=None, ge=0)


class PowerModelValidity(_Strict):
    """Where the curve may be evaluated. `refuse` outside is the default (A6)."""

    served_conc_min: float = Field(gt=0)
    served_conc_max: float = Field(gt=0)
    extrapolation: str = "refuse"


class PowerModel(_Strict):
    """Power as a function of the operating point, replacing the scalar
    `active_power` where a curve has been measured
    (`docs/rps_aware_planning_design.md` §3).

    `kind` is data, not an enum, because the explanatory variable is a property of
    the device and the measured range rather than of the schema. ATOM's power is
    monotone in utilisation over a 59 pp span; the RNGD card sits at 84.7-92.1 %
    throughout and its power is U-shaped in concurrency, so keying it on
    utilisation does not fit (deviations D31). Both are legitimate models of their
    own measurement, and forcing one form would mean fitting a curve the data does
    not support.

    Optional throughout: a profile with only the scalar `power:` block stays valid.
    """

    kind: str
    idle_w: float = Field(ge=0)
    points: list[PowerModelPoint] = Field(min_length=1)
    validity: PowerModelValidity
    saturation_knee: float | None = None
    additive_across_units: bool | None = None
    quantisation_w: float | None = Field(default=None, ge=0)
    source: Source = Source.PLACEHOLDER


class Datasheet(_Strict):
    """Datasheet values needed for Tier 0 roofline generation.

    Every field is optional. A missing value must make Tier 0 generation FAIL
    (absolute rule A2) - it is never quietly defaulted. ``datasheet_source``
    is mandatory whenever any value is present (absolute rule 3: no
    unattributed hardware numbers). ``Source`` is deliberately NOT extended:
    it labels accelerator-YAML numbers, while ProfileTier labels bundles -
    two different concepts.
    """

    #: dtype -> dense peak TFLOP/s. Keys like 'bf16', 'fp16', 'fp8', 'int8'.
    peak_tflops: dict[str, float] = Field(default_factory=dict)
    #: Vendor-spec memory bandwidth for Tier 0 rooflines. Optional override:
    #: the profile-level memory_bandwidth_gbps sometimes carries an upstream
    #: simulator figure rather than the vendor datasheet number (rtxpro6000:
    #: 1597 vs the whitepaper's 1792); Tier 0 must compute on the sourced one.
    memory_bandwidth_gbps: float | None = Field(default=None, gt=0)
    #: Compute-unit count (GPU: SMs; NPU: PE clusters/cores). Backend-neutral.
    compute_units: int | None = Field(default=None, gt=0)
    clock_mhz: float | None = Field(default=None, gt=0)
    l2_cache_mb: float | None = Field(default=None, gt=0)
    #: Kernel-launch overhead floor in us (KernelSight-LM's t_0 term).
    kernel_launch_us: float | None = Field(default=None, gt=0)
    #: Roofline derating. Absent means Tier 0 generation is impossible (A2);
    #: it is fitted from measurements (STEP 8), never assumed.
    flops_efficiency: float | None = Field(default=None, gt=0, le=1)
    mem_efficiency: float | None = Field(default=None, gt=0, le=1)
    #: Per-kernel-family overrides. Keys like 'gemm','attention','elementwise','moe'.
    family_efficiency: dict[str, float] = Field(default_factory=dict)
    #: Where these numbers come from. Required for A1 compliance.
    datasheet_source: str = ""

    @model_validator(mode="after")
    def _rules(self) -> Datasheet:
        has_values = bool(
            self.peak_tflops
            or self.family_efficiency
            or any(
                v is not None
                for v in (
                    self.compute_units, self.clock_mhz, self.l2_cache_mb,
                    self.kernel_launch_us, self.flops_efficiency, self.mem_efficiency,
                )
            )
        )
        if has_values and not self.datasheet_source.strip():
            raise ValueError(
                "datasheet has values but datasheet_source is empty - "
                "unattributed hardware numbers are forbidden (absolute rule 3)"
            )
        for family, eff in self.family_efficiency.items():
            if not (0 < eff <= 1):
                raise ValueError(
                    f"family_efficiency[{family!r}]={eff} outside (0, 1]"
                )
        return self


class AcceleratorProfile(_Strict):
    profile_id: str
    vendor: str
    model: str
    backend: str
    memory_gb: float = Field(gt=0)
    memory_bandwidth_gbps: float = Field(gt=0)
    #: Simulator `hardware` key; must match a directory under profiler/perf/.
    sim_hardware: str | None = None
    tdp_w: float | None = Field(default=None, gt=0)
    idle_power_w: float | None = Field(default=None, ge=0)
    power: ProfilePower | None = None
    #: Measured power curve, where one exists. The scalar `power:` block above is
    #: kept for back-compatibility and is what the simulator compiler still emits.
    power_model: PowerModel | None = None
    source: Source = Source.PLACEHOLDER
    perf_data: str | None = None
    supported_models: list[SupportedModel] = Field(default_factory=list)
    max_tp_size: int = Field(default=8, ge=1)
    notes: str = ""
    #: Vendor datasheet values for Tier 0/1 synthetic bundles. None for
    #: profiles that only ever use measured bundles.
    datasheet: Datasheet | None = None

    @model_validator(mode="after")
    def _tier_label_rules(self) -> AcceleratorProfile:
        # A synthetic sim_hardware label promises a generated bundle, which is
        # impossible without datasheet values (tiered-profiles STEP 3).
        synthetic_label = self.sim_hardware is not None and (
            self.sim_hardware.endswith("-t0") or self.sim_hardware.endswith("-t1")
        )
        if synthetic_label and self.datasheet is None:
            raise ValueError(
                f"sim_hardware '{self.sim_hardware}' names a synthetic (Tier 0/1) "
                f"bundle but the profile has no datasheet: - Tier 0 generation "
                f"needs datasheet values (absolute rule A2)"
            )
        if self.datasheet is not None and not self.datasheet.datasheet_source.strip():
            raise ValueError(
                f"profile {self.profile_id}: datasheet present but datasheet_source "
                f"is empty - unattributed hardware numbers are forbidden (rule 3)"
            )
        return self


class ExecutionIsland(BaseModel):
    """A set of accelerators that can host exactly one vLLM engine (§1.3)."""

    id: str
    backend: str
    node_id: str
    accelerator_ids: list[str]
    accelerator_model: str
    interconnect_type: LinkType | None
    total_memory_gb: float
    max_tp_candidates: list[int]

    @property
    def size(self) -> int:
        return len(self.accelerator_ids)


def _reject_duplicates(values: list[str], what: str) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"duplicate {what} id '{v}'")
        seen.add(v)


def model_slug(model: str) -> str:
    """Deterministic, filesystem-safe slug used in island ids.

    Note: the work order illustrates `cuda-h100-node0` for a model written
    `H100-80GB`. We do not strip the capacity suffix - dropping it would merge
    an H100-80GB and an H100-94GB island into one id. Golden tests are generated
    from this function, not from the illustration.
    """
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", model.lower())).strip("-")


def compatibility(model: str, dtype: str, profile: AcceleratorProfile) -> bool:
    """True when `profile` declares support for (model, dtype).

    An empty `supported_models` list means "unknown", which we treat as
    unsupported: silently admitting an unprofiled combination is exactly the
    Risk-1 failure the work order §11 asks us to catch at candidate generation.
    """
    for entry in profile.supported_models:
        if fnmatch.fnmatch(model, entry.pattern) and dtype in entry.dtypes:
            return True
    return False


def _intra_node_adjacency(cluster: ClusterSpecV2, node: Node) -> dict[str, set[str]]:
    """Accelerator-to-accelerator adjacency within one node, island links only."""
    adjacency: dict[str, set[str]] = {a.id: set() for a in node.accelerators}
    accel_ids = set(adjacency)
    for link in cluster.links:
        if link.type not in INTRA_ISLAND_LINKS:
            continue
        (sn, sd), (dn, dd) = link.endpoints
        if sn != node.id or dn != node.id:
            continue
        if sd in accel_ids and dd in accel_ids:
            adjacency[sd].add(dd)
            adjacency[dd].add(sd)
    return adjacency


def _components(adjacency: dict[str, set[str]], members: list[str]) -> list[list[str]]:
    order = {m: i for i, m in enumerate(members)}
    seen: set[str] = set()
    out: list[list[str]] = []
    for start in members:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency.get(cur, ()):
                if nxt not in seen and nxt in order:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(sorted(comp, key=order.__getitem__))
    return out


def tp_candidates(size: int, max_tp_size: int) -> list[int]:
    """Divisors of the island size, capped by the profile's max TP degree."""
    return [d for d in range(1, size + 1) if size % d == 0 and d <= max_tp_size]


def detect_islands(
    cluster: ClusterSpecV2,
    profiles: dict[str, AcceleratorProfile] | None = None,
    *,
    free_only: bool = True,
) -> list[ExecutionIsland]:
    """Group accelerators into execution islands (§5.2).

    An island is a maximal set of accelerators on one node that share a backend,
    share an accelerator model, and are mutually reachable over intra-island
    links (NVLINK / PCIE / HCCS). Accelerators with no such link form singleton
    islands, which is correct: one GPU can still host a TP=1 engine.

    Mixing backends inside an island is impossible by construction - that is
    absolute rule 2 enforced at the earliest possible point.
    """
    profiles = profiles or {}
    islands: list[ExecutionIsland] = []

    for node in cluster.nodes:
        adjacency = _intra_node_adjacency(cluster, node)
        usable = [a for a in node.accelerators if a.is_free or not free_only]

        # Group by (backend, model) first: a single island must be homogeneous
        # enough to run one engine, and TP across different models is meaningless.
        groups: dict[tuple[str, str], list[Accelerator]] = {}
        for accel in usable:
            groups.setdefault((accel.backend, accel.model), []).append(accel)

        for (backend, model), members in sorted(groups.items()):
            by_id = {a.id: a for a in members}
            for comp in _components(adjacency, [a.id for a in members]):
                accels = [by_id[i] for i in comp]
                profile = profiles.get(model)
                max_tp = profile.max_tp_size if profile else len(accels)
                islands.append(
                    ExecutionIsland(
                        id=f"{backend}-{model_slug(model)}-{node.id}",
                        backend=backend,
                        node_id=node.id,
                        accelerator_ids=comp,
                        accelerator_model=model,
                        interconnect_type=_dominant_link(cluster, node.id, comp),
                        total_memory_gb=sum(a.memory_gb for a in accels),
                        max_tp_candidates=tp_candidates(len(accels), max_tp),
                    )
                )

    _disambiguate_ids(islands)
    return islands


def _dominant_link(cluster: ClusterSpecV2, node_id: str, accel_ids: list[str]) -> LinkType | None:
    """The intra-island link type connecting these accelerators, if any."""
    members = set(accel_ids)
    kinds: set[LinkType] = set()
    for link in cluster.links:
        if link.type not in INTRA_ISLAND_LINKS:
            continue
        (sn, sd), (dn, dd) = link.endpoints
        if sn == node_id and dn == node_id and sd in members and dd in members:
            kinds.add(link.type)
    if not kinds:
        return None
    # Prefer the fastest class present; a PCIe fallback link alongside NVLink
    # should not downgrade the island's description.
    for kind in (LinkType.ONPACKAGE, LinkType.NVLINK, LinkType.HCCS, LinkType.PCIE):
        if kind in kinds:
            return kind
    return next(iter(kinds))


def _disambiguate_ids(islands: list[ExecutionIsland]) -> None:
    """Two disconnected same-model groups on one node would collide on id."""
    counts: dict[str, int] = {}
    for island in islands:
        counts[island.id] = counts.get(island.id, 0) + 1
    running: dict[str, int] = {}
    for island in islands:
        if counts[island.id] > 1:
            idx = running.get(island.id, 0)
            running[island.id] = idx + 1
            island.id = f"{island.id}-{idx}"


class InventoryError(ValueError):
    """Raised when a cluster spec or profile cannot be loaded."""


def _load_yaml(path: Path, what: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise InventoryError(f"{path}: {what} file not found") from exc
    except yaml.YAMLError as exc:
        raise InventoryError(f"{path}: invalid YAML - {exc}") from exc
    if not isinstance(raw, dict):
        raise InventoryError(f"{path}: expected a YAML mapping at the top level")
    return raw


def load_cluster_spec(path: str | Path) -> ClusterSpecV2:
    path = Path(path)
    try:
        return ClusterSpecV2.model_validate(_load_yaml(path, "cluster spec"))
    except InventoryError:
        raise
    except Exception as exc:
        raise InventoryError(f"{path}: {exc}") from exc


def load_accelerator_profile(path: str | Path) -> AcceleratorProfile:
    path = Path(path)
    try:
        return AcceleratorProfile.model_validate(_load_yaml(path, "accelerator profile"))
    except InventoryError:
        raise
    except Exception as exc:
        raise InventoryError(f"{path}: {exc}") from exc


def load_profiles_for(
    cluster: ClusterSpecV2, root: str | Path = "."
) -> dict[str, AcceleratorProfile]:
    """Load every profile referenced by the cluster, keyed by accelerator model."""
    root = Path(root)
    out: dict[str, AcceleratorProfile] = {}
    for node in cluster.nodes:
        for accel in node.accelerators:
            if accel.profile is None or accel.model in out:
                continue
            profile = load_accelerator_profile(root / accel.profile)
            if profile.model != accel.model:
                raise InventoryError(
                    f"{accel.profile}: profile declares model '{profile.model}' but "
                    f"accelerator {node.id}/{accel.id} declares '{accel.model}'"
                )
            if profile.backend != accel.backend:
                raise InventoryError(
                    f"{accel.profile}: profile declares backend '{profile.backend}' but "
                    f"accelerator {node.id}/{accel.id} declares '{accel.backend}'"
                )
            out[accel.model] = profile
    return out
