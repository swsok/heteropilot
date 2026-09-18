"""The uncertain-input registry (WORK_ORDER_uncertainty_planner.md §2.3).

One place that answers "which planner inputs are not measurements, how far off
can each one be, and what would it cost to find out". Three sources feed it:

* ``cluster.links`` - bandwidth and latency whose ``Source`` is not measured;
* each island's profile - the *weaker* of the bundle tier and the profile
  YAML's own ``source``, plus the accelerator power block;
* the calibration store - the simulator's own prediction error per
  (hardware, workload bucket).

Two invariants shape the code:

**No zero-width item exists in the registry.** An input that is measured, or
whose sourced range collapses to a point, is counted in ``measured_count``
rather than listed. Stage B's perturbation and ranking therefore never need a
special case for a width of zero.

**A range with no source is ``unbounded``, never a plausible default**
(absolute rule A1). ``unbounded`` propagates to "cannot be decided before
measuring" rather than quietly scoring as zero regret.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from planner.inventory import Collective, MsgSizeClass, Source
from planner.topology import TopologyError, TopologyGraph
from planner.uncertainty.grades import CostsTable, DefaultRule, GradesTable, RangeRule
from planner.util import tier as tierutil

if TYPE_CHECKING:
    from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
    from planner.predictor.calibration import CalibrationModel
    from planner.spec import ServiceSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UncertainKind(str, enum.Enum):
    SIM_ERROR = "sim_error"   # (hardware x bucket) prediction error; accuracy domain
    PROFILE = "profile"       # accelerator performance profile trust
    LINK_BW = "link_bw"       # inventory.Link.bandwidth_gbps
    LINK_LAT = "link_lat"     # inventory.Link.latency_ns
    POWER = "power"           # accelerator power model


class Grade(str, enum.Enum):
    """One trust scale spanning `inventory.Source` and `tier.ProfileTier`.

    Neither existing enum covers the other: ``Source`` knows ``vendor_spec``
    but not ``analytical``/``calibrated``, ``ProfileTier`` the reverse. A
    profile's grade is the weaker of the two signals, so the comparison needs a
    scale both map into.
    """

    PLACEHOLDER = "placeholder"
    USER_DEFINED = "user_defined"
    UNKNOWN = "unknown"
    ANALYTICAL = "analytical"
    CALIBRATED = "calibrated"
    VENDOR_SPEC = "vendor_spec"
    MEASURED = "measured"
    IMPORTED = "imported"

    @property
    def rank(self) -> int:
        """§2.3's ordering. Only rank 4 - a real measurement - is certain.

        ``placeholder``, ``user_defined`` and ``unknown`` share rank 0: a
        deliberate what-if and an unattributed number are both "no measurement
        exists", and the registry treats them identically (all three are
        unbounded unless the caller supplies a range).
        """
        return {
            Grade.PLACEHOLDER: 0,
            Grade.USER_DEFINED: 0,
            Grade.UNKNOWN: 0,
            Grade.ANALYTICAL: 1,
            Grade.CALIBRATED: 2,
            Grade.VENDOR_SPEC: 3,
            Grade.MEASURED: 4,
            Grade.IMPORTED: 4,
        }[self]

    @property
    def is_measurement(self) -> bool:
        return self.rank == 4

    @classmethod
    def from_source(cls, source: Source) -> Grade:
        return cls(source.value)

    @classmethod
    def from_tier(cls, tier: tierutil.ProfileTier) -> Grade:
        return cls(tier.value)


def weaker(a: Grade, b: Grade) -> Grade:
    """The less trustworthy of two grades; ties keep the first."""
    return b if b.rank < a.rank else a


class Range(_Strict):
    """The interval an uncertain value may actually take.

    ``lo``/``hi`` of None mean unbounded in that direction: no sourced limit
    exists. A finite range is always non-degenerate - see the module docstring.
    """

    lo: float | None = None
    hi: float | None = None
    unit: str = "fraction"
    #: Row id in grades.yaml, or the calibration file the width came from.
    source: str = ""
    #: ``sourced`` - the width came from this input's own (kind, grade) rule or
    #: from the calibration store. ``default`` - nothing measured this input's
    #: width, so the grade's default from ``grades.yaml`` was applied (S2, D111).
    #: Carried all the way to the rendered measurement plan: a default is a
    #: policy the operator may disagree with, and it says so rather than looking
    #: like a measurement.
    range_source: Literal["sourced", "default"] = "sourced"

    @property
    def is_unbounded(self) -> bool:
        return self.lo is None or self.hi is None

    @property
    def width(self) -> float | None:
        if self.lo is None or self.hi is None:
            return None
        return self.hi - self.lo

    @model_validator(mode="after")
    def _finite_ranges_are_non_degenerate(self) -> Range:
        if self.lo is not None and self.hi is not None:
            if self.hi <= self.lo:
                raise ValueError(
                    f"Range [{self.lo}, {self.hi}]: a bounded range must have hi > lo - "
                    f"a zero-width item does not belong in the registry (§2.3)"
                )
            if not self.source.strip():
                raise ValueError(
                    "Range: a bounded range must name its source (absolute rule A1)"
                )
        return self

    def __str__(self) -> str:
        if self.is_unbounded:
            return "unbounded"
        return f"[{self.lo:,.4g}, {self.hi:,.4g}] {self.unit}"


class MeasurementCost(_Strict):
    """What it costs to replace an uncertain input with a measurement (§2.6)."""

    method: str
    #: Wall-clock hours; None when no timestamped evidence exists. Sorts last.
    hours: float | None = None
    #: True when the measurement owns the device for its duration.
    exclusive: bool = False
    source: str = ""


class UncertainInput(_Strict):
    """One uncertain planner input."""

    id: str
    kind: UncertainKind
    grade: Grade
    nominal: float
    range: Range
    cost: MeasurementCost | None = None
    #: island ids, or ``"<island>-><island>"`` pairs for a link.
    affects: list[str] = Field(default_factory=list)
    note: str = ""


class UncertainInputRegistry(_Strict):
    """Every uncertain input of one planning run (§2.3)."""

    #: Uncertain inputs only. A measured input is counted, never listed.
    items: list[UncertainInput] = Field(default_factory=list)
    #: kind -> how many inputs were excluded because they are measurements.
    #: The renderer turns this into the coverage line ("26 links, 16 uncertain").
    measured_count: dict[str, int] = Field(default_factory=dict)
    #: sha256 of grades.yaml / costs.yaml, for provenance.
    grades_digest: str = ""
    costs_digest: str = ""

    def unbounded(self) -> list[UncertainInput]:
        """Items with no range at all - "cannot be decided before measuring"."""
        return [i for i in self.items if i.range.is_unbounded]

    def defaulted(self) -> list[UncertainInput]:
        """Items ranked on the grade's default range rather than their own (D111)."""
        return [i for i in self.items if i.range.range_source == "default"]

    def by_kind(self, kind: UncertainKind) -> list[UncertainInput]:
        return [i for i in self.items if i.kind is kind]

    def total_for(self, kind: UncertainKind) -> int:
        """How many inputs of this kind existed, uncertain or not."""
        return len(self.by_kind(kind)) + self.measured_count.get(kind.value, 0)


# ---------------------------------------------------------------------------
# Range construction
# ---------------------------------------------------------------------------

def _default_range(
    grades: GradesTable, kind: UncertainKind, grade: Grade, nominal: float, unit: str,
    *, unbounded_source: str = "",
) -> Range:
    """The grade's default range, or unbounded when the grade has no default.

    The S2 layer (D111). Before it, an input whose own (kind, grade) sourced no
    width left the measurement plan's ranking entirely - which is what happened
    to `link_bw:pcie-a40a-02`, the `vendor_spec` link that turned out to explain
    a -43.4 % TPOT error and cost 0.114 h to measure.

    A default never overrides a sourced width; it is only ever reached when
    there is none, and the range it returns says `range_source="default"` so
    nothing downstream can mistake the two.
    """
    default = grades.default_for(kind.value, grade.value)
    if default is None:
        # The honest end of the road, and still a real category: a grade nobody
        # can put a defensible width on (`user_defined`, a deliberate what-if)
        # stays "cannot be decided before measuring". `unbounded_source` keeps
        # the breadcrumb of WHICH row said so.
        return Range(unit=unit, source=unbounded_source)

    if default.rule is DefaultRule.ABSOLUTE and default.half_width is not None:
        lo, hi = nominal - default.half_width, nominal + default.half_width
    elif default.lo is not None and default.hi is not None:
        # Ordered after multiplying: a negative nominal would otherwise invert
        # the interval and trip the non-degenerate check.
        ends = sorted((nominal * default.lo, nominal * default.hi))
        lo, hi = ends[0], ends[1]
    else:  # pragma: no cover - the model validator forbids this shape
        return Range(unit=unit, source=unbounded_source)

    if hi <= lo:
        # A relative default on a nominal of zero. No width exists, and
        # inventing one here would be the guess the whole table refuses.
        return Range(unit=unit, source=unbounded_source)
    return Range(
        lo=lo, hi=hi, unit=unit, source=f"defaults:{default.id}", range_source="default"
    )


def _range_from_rule(
    grades: GradesTable, kind: UncertainKind, grade: Grade, nominal: float, unit: str
) -> Range:
    """Apply grades.yaml's formula for a (kind, grade) to a nominal value.

    A rule that cannot produce a non-degenerate interval - no sourced number, a
    nominal of zero under a multiplicative rule - falls through to the grade's
    default range (S2, D111), and to unbounded when the grade has none. Neither
    step fabricates a width: a default is a sourced policy, labelled as one.
    """
    rule = grades.rule_for(kind.value, grade.value)

    if rule.rule is RangeRule.SYMMETRIC_FRACTION and rule.value is not None:
        lo, hi = nominal * (1.0 - rule.value), nominal * (1.0 + rule.value)
    elif rule.rule is RangeRule.RATIO_FLOOR and rule.r_min is not None:
        lo, hi = nominal * rule.r_min, nominal
    else:
        # UNBOUNDED, or CALIBRATION_DERIVED (whose width comes from the
        # calibration store, applied by the caller, not from this table).
        return _default_range(
            grades, kind, grade, nominal, unit,
            unbounded_source=rule.id if rule.source else "",
        )

    if hi <= lo:
        return _default_range(
            grades, kind, grade, nominal, unit,
            unbounded_source=rule.id if rule.source else "",
        )
    return Range(lo=lo, hi=hi, unit=unit, source=rule.id)


def _cost_for(costs: CostsTable | None, kind: UncertainKind) -> MeasurementCost | None:
    if costs is None:
        return None
    row = costs.cost_for(kind.value)
    if row is None:
        return None
    return MeasurementCost(
        method=row.method, hours=row.hours, exclusive=row.exclusive, source=row.source
    )


# ---------------------------------------------------------------------------
# Link item keys (domain-scoping S3, D112)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkTraffic:
    """What a link carries in one deployment, and for whom.

    A LINK_BW item used to be one per link, which read the link's datasheet
    number as though a wire had one bandwidth. It does not: the A40 node's PCIe
    bridge sustains 25.0 GB/s for a direct copy, 19.3 for a two-rank all-reduce
    and 8.8 for the four-rank all-reduce a tp=4 island actually runs, against a
    `vendor_spec` 64.0. So the item is keyed by the traffic instead, and the
    same link carrying two kinds is two items with two measurement costs.
    """

    collective: Collective
    msg_size_class: MsgSizeClass
    binding: str
    world_size: int
    #: island ids (intra-island traffic) or "<island>-><island>" pairs (inter).
    affects: tuple[str, ...] = ()


def link_item_id(
    kind: UncertainKind, link_id: str, traffic: LinkTraffic | None = None
) -> str:
    """`link_bw:<link>@<collective>/w<N>/<class>/<binding>`, or `link_bw:<link>`.

    The unkeyed form is for a link no selected island's traffic crosses - there
    is no collective to name - and for LINK_LAT, whose value the planner applies
    to every kind of traffic alike (`path_latency_ns` sums it regardless) and of
    which this repository holds no measurement at all.
    """
    if traffic is None:
        return f"{kind.value}:{link_id}"
    return (
        f"{kind.value}:{link_id}@{traffic.collective}/w{traffic.world_size}/"
        f"{traffic.msg_size_class}/{traffic.binding}"
    )


@dataclass(frozen=True)
class LinkItemKey:
    """A LINK_BW/LINK_LAT item id, taken apart. `collective` is None if unkeyed."""

    link_id: str
    collective: Collective | None = None
    world_size: int | None = None
    msg_size_class: MsgSizeClass = "bulk"
    binding: str = "unknown"


def parse_link_item_id(item_id: str) -> LinkItemKey:
    """Split a link item id back into its parts.

    Every consumer that used to do `item.id.split(":", 1)[1]` and treat the
    remainder as a link id goes through this instead; a bare split now yields
    `pcie-a40a-02@all_reduce/bulk/unpinned`, which matches no link.
    """
    rest = item_id.split(":", 1)[1] if ":" in item_id else item_id
    if "@" not in rest:
        return LinkItemKey(link_id=rest)
    link_id, _, cond = rest.partition("@")
    parts = cond.split("/")
    if len(parts) != 4 or not parts[1].startswith("w") or not parts[1][1:].isdigit():
        raise ValueError(
            f"link item id {item_id!r}: expected "
            f"'<link>@<collective>/w<world_size>/<msg_size_class>/<binding>'"
        )
    return LinkItemKey(
        link_id=link_id,
        collective=parts[0],  # type: ignore[arg-type]
        world_size=int(parts[1][1:]),
        msg_size_class=parts[2],  # type: ignore[arg-type]
        binding=parts[3],
    )


def _link_traffic(
    cluster: ClusterSpecV2, islands: list[ExecutionIsland]
) -> dict[str, list[LinkTraffic]]:
    """link id -> every kind of traffic the selected islands put on it.

    Built from `_link_affects`, which already separates the two cases: an entry
    that names an island is that island's TP collective, and one that names a
    pair is the P/D handoff between them. Both are asked for at `bulk`, the
    band the simulator's asymptotic `link_bw` term is about.
    """
    affects = _link_affects(cluster, islands)
    by_id = {island.id: island for island in islands}
    binding = {node.id: node.device_binding for node in cluster.nodes}
    out: dict[str, list[LinkTraffic]] = {}

    for link_id, who in affects.items():
        intra = sorted(w for w in who if "->" not in w)
        inter = sorted(w for w in who if "->" in w)
        traffic: list[LinkTraffic] = []
        if intra:
            # One item per distinct TP degree: two islands of different size on
            # this link need two measurements, not one.
            for size in sorted({by_id[i].size for i in intra if i in by_id}):
                members = [i for i in intra if i in by_id and by_id[i].size == size]
                traffic.append(LinkTraffic(
                    collective="all_reduce", msg_size_class="bulk",
                    binding=binding.get(by_id[members[0]].node_id, "unknown"),
                    world_size=size, affects=tuple(members),
                ))
        if inter:
            first = inter[0].split("->")[0]
            traffic.append(LinkTraffic(
                collective="p2p", msg_size_class="bulk",
                binding=binding.get(
                    by_id[first].node_id if first in by_id else "", "unknown"),
                world_size=2, affects=tuple(inter),
            ))
        out[link_id] = traffic
    return out


# ---------------------------------------------------------------------------
# affects: which islands a link sits between
# ---------------------------------------------------------------------------

def _link_affects(
    cluster: ClusterSpecV2, islands: list[ExecutionIsland]
) -> dict[str, list[str]]:
    """link id -> the islands (or island pairs) whose traffic crosses it.

    Intra-island links report the island itself; inter-island links report every
    ordered-by-name island pair whose fewest-hop path uses them. Every pair's
    path is walked once and inverted, rather than re-running BFS per link.
    """
    members: dict[str, set[str]] = {
        island.id: {f"{island.node_id}/{a}" for a in island.accelerator_ids}
        for island in islands
    }
    affects: dict[str, list[str]] = {}

    for link in cluster.links:
        for island_id, endpoints in members.items():
            if link.src in endpoints and link.dst in endpoints:
                affects.setdefault(link.id, []).append(island_id)

    try:
        graph = TopologyGraph(cluster)
    except TopologyError:
        return affects

    ordered = sorted(islands, key=lambda i: i.id)
    for idx, a in enumerate(ordered):
        for b in ordered[idx + 1:]:
            src = f"{a.node_id}/{a.accelerator_ids[0]}"
            dst = f"{b.node_id}/{b.accelerator_ids[0]}"
            try:
                path = graph.path(src, dst)
            except TopologyError:
                # Disconnected islands are normal (a fixture need not wire every
                # pair); they simply constrain no link.
                continue
            for link in path:
                affects.setdefault(link.id, []).append(f"{a.id}->{b.id}")

    return {k: sorted(dict.fromkeys(v)) for k, v in affects.items()}


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------

def build_registry(
    cluster: ClusterSpecV2,
    profiles: dict[str, AcceleratorProfile],
    islands: list[ExecutionIsland],
    calibration: CalibrationModel,
    grades: GradesTable,
    spec: ServiceSpec,
    costs: CostsTable | None = None,
    *,
    perf_root: Path | None = None,
) -> UncertainInputRegistry:
    """Collect every uncertain input of one planning run (§2.3).

    ``spec`` is needed beyond the work order's five arguments because a
    bundle's tier is resolved per (model, dtype variant), which only the
    ServiceSpec knows.
    """
    items: list[UncertainInput] = []
    measured: dict[str, int] = {}

    def count_measured(kind: UncertainKind) -> None:
        measured[kind.value] = measured.get(kind.value, 0) + 1

    items += _link_items(cluster, islands, grades, costs, count_measured)
    items += _profile_items(cluster, profiles, islands, spec, grades, costs, perf_root,
                            count_measured)
    items += _sim_error_items(profiles, islands, calibration, grades, costs)

    return UncertainInputRegistry(
        items=items,
        measured_count=dict(sorted(measured.items())),
        grades_digest=grades.digest,
        costs_digest=costs.digest if costs is not None else "",
    )


def _link_items(cluster, islands, grades, costs, count_measured) -> list[UncertainInput]:
    """LINK_BW and LINK_LAT items for every link that is not a measurement.

    Since S3 (D112) a LINK_BW item is per (link, traffic): what is uncertain is
    not "this wire" but "what this wire delivers for the collective this
    deployment runs over it", and a `measured` figure for one collective says
    nothing about another. A link whose spec value is itself `measured` is still
    credited wholesale, as before - the schema has no way to say which traffic
    that claim covers, which is exactly why the `measurements:` array exists.
    """
    affects = _link_affects(cluster, islands)
    traffic_by_link = _link_traffic(cluster, islands)
    out: list[UncertainInput] = []

    for link in cluster.links:
        grade = Grade.from_source(link.source)
        if grade.is_measurement:
            count_measured(UncertainKind.LINK_BW)
            count_measured(UncertainKind.LINK_LAT)
            continue

        # One `source` field covers both bandwidth and latency, so both inherit
        # the same grade. At least one committed fixture says in a comment that
        # a link's bandwidth is measured while its latency is not
        # (experiments/configs/clusters/a40x8.yaml); the schema cannot express
        # that split, so a `measured` link's latency is credited as measured
        # too. Recorded rather than silently corrected - see the PR notes.
        note = f"link {link.type.value} {link.src} -> {link.dst}; source={link.source.value}"

        # LINK_BW: one item per kind of traffic this deployment puts on the link.
        traffics = traffic_by_link.get(link.id) or []
        for traffic in traffics:
            hit = link.measurement_for(
                traffic.collective, traffic.msg_size_class, traffic.binding,
                world_size=traffic.world_size,
            )
            if hit is not None and Grade.from_source(hit.source).is_measurement:
                # This traffic's bandwidth IS measured, whatever the datasheet
                # column beside it says. Counted, not listed - the registry's
                # standing rule for a measured input.
                count_measured(UncertainKind.LINK_BW)
                continue
            rng = _range_from_rule(
                grades, UncertainKind.LINK_BW, grade, link.bandwidth_gbps, "gbps"
            )
            out.append(
                UncertainInput(
                    id=link_item_id(UncertainKind.LINK_BW, link.id, traffic),
                    kind=UncertainKind.LINK_BW,
                    grade=grade,
                    nominal=link.bandwidth_gbps,
                    range=rng,
                    cost=_cost_for(costs, UncertainKind.LINK_BW),
                    affects=list(traffic.affects),
                    note=(
                        f"{note}; {traffic.collective} at world_size "
                        f"{traffic.world_size}, {traffic.msg_size_class} messages, "
                        f"binding {traffic.binding}. The nominal is the spec value: "
                        f"no measurement answers for these conditions"
                    ),
                )
            )
        if not traffics:
            # No selected island's traffic crosses this link, so there is no
            # collective to name. Kept rather than dropped: a link outside this
            # deployment's paths is still an uncertain input of the spec, and a
            # later candidate may use it.
            rng = _range_from_rule(
                grades, UncertainKind.LINK_BW, grade, link.bandwidth_gbps, "gbps"
            )
            out.append(
                UncertainInput(
                    id=link_item_id(UncertainKind.LINK_BW, link.id),
                    kind=UncertainKind.LINK_BW,
                    grade=grade,
                    nominal=link.bandwidth_gbps,
                    range=rng,
                    cost=_cost_for(costs, UncertainKind.LINK_BW),
                    affects=affects.get(link.id, []),
                    note=f"{note}; no selected island's traffic crosses it",
                )
            )

        # LINK_LAT stays one item per link: `path_latency_ns` applies it to every
        # kind of traffic alike, and no link latency in this repository is
        # measured at all (grades.yaml's link_lat/vendor_spec row), so there is
        # nothing for a key to select between.
        rng = _range_from_rule(
            grades, UncertainKind.LINK_LAT, grade, link.latency_ns, "ns"
        )
        out.append(
            UncertainInput(
                id=link_item_id(UncertainKind.LINK_LAT, link.id),
                kind=UncertainKind.LINK_LAT,
                grade=grade,
                nominal=link.latency_ns,
                range=rng,
                cost=_cost_for(costs, UncertainKind.LINK_LAT),
                affects=affects.get(link.id, []),
                note=note,
            )
        )
    return out


#: Hardware-label suffixes that mark a synthetic bundle of the SAME hardware
#: (planner/util/tier.py's `_SUFFIX_TIER`), not a different device.
_TIER_SUFFIXES = ("-t0", "-t1")


def _strip_tier_suffix(hardware: str) -> str:
    for suffix in _TIER_SUFFIXES:
        if hardware.endswith(suffix):
            return hardware[: -len(suffix)]
    return hardware


def _profile_items(
    cluster, profiles, islands, spec, grades, costs, perf_root, count_measured
) -> list[UncertainInput]:
    """PROFILE and POWER items, graded on the weaker of two signals (§2.3).

    Reading the bundle tier alone would pass ``ascend-sim-proxy`` as measured -
    it borrows the real RTXPRO6000 bundle while the profile itself is a
    placeholder standing an NPU in for a GPU - and the single most misleading
    input in the repository would never reach the registry.
    """
    variant = tierutil.resolve_variant(spec.service.dtype, spec.service.kv_cache_dtype)
    out: list[UncertainInput] = []

    for island in islands:
        profile = profiles.get(island.accelerator_model)
        if profile is None:
            out.append(
                UncertainInput(
                    id=f"profile:{island.id}",
                    kind=UncertainKind.PROFILE,
                    grade=Grade.PLACEHOLDER,
                    nominal=1.0,
                    range=_default_range(
                        grades, UncertainKind.PROFILE, Grade.PLACEHOLDER, 1.0, "fraction"
                    ),
                    cost=_cost_for(costs, UncertainKind.PROFILE),
                    affects=[island.id],
                    note=f"no profile loaded for accelerator model {island.accelerator_model}",
                )
            )
            continue

        source_grade = Grade.from_source(profile.source)
        hardware = profile.sim_hardware
        if hardware is None:
            bundle_grade = Grade.PLACEHOLDER
            bundle_note = "no sim_hardware: no bundle at all"
        else:
            tier = tierutil.resolve_bundle_tier(perf_root, hardware, spec.model, variant)
            bundle_grade = Grade.from_tier(tier)
            bundle_note = f"bundle {hardware} tier={tier.value}"

        grade = weaker(source_grade, bundle_grade)
        note = f"profile source={profile.source.value}; {bundle_note}"
        if hardware is not None and profile.model != _strip_tier_suffix(hardware):
            # A label that differs only by the -t0/-t1 tier suffix names the
            # same hardware synthetically, which the grade already says; a
            # proxy is a label naming DIFFERENT hardware (ascend-sim-proxy
            # simulated as RTXPRO6000), and that is what must be surfaced.
            note += f"; proxy: {profile.model} simulated as {hardware}"

        if grade.is_measurement:
            count_measured(UncertainKind.PROFILE)
        else:
            out.append(
                UncertainInput(
                    id=f"profile:{island.id}",
                    kind=UncertainKind.PROFILE,
                    grade=grade,
                    # A profile's uncertainty is a multiplier on the predicted
                    # latencies (§2.7), so its nominal is the identity.
                    nominal=1.0,
                    range=_range_from_rule(grades, UncertainKind.PROFILE, grade, 1.0, "fraction"),
                    cost=_cost_for(costs, UncertainKind.PROFILE),
                    affects=[island.id],
                    note=note,
                )
            )

        # The accelerator's own power block. A profile with no power block is
        # not an uncertain number - the simulator emits no energy at all for it
        # (deviations D2/D14) - so it is neither listed nor counted.
        if profile.power is None:
            continue
        power_grade = Grade.from_source(profile.power.source)
        if power_grade.is_measurement:
            count_measured(UncertainKind.POWER)
            continue
        out.append(
            UncertainInput(
                id=f"power:{island.id}",
                kind=UncertainKind.POWER,
                grade=power_grade,
                nominal=1.0,
                range=_range_from_rule(grades, UncertainKind.POWER, power_grade, 1.0, "fraction"),
                cost=_cost_for(costs, UncertainKind.POWER),
                affects=[island.id],
                note=f"{profile.profile_id} power source={profile.power.source.value}",
            )
        )
    return out


def _sim_error_items(
    profiles, islands, calibration, grades, costs
) -> list[UncertainInput]:
    """One SIM_ERROR item per (hardware, workload bucket) the store knows.

    Hardware the store has never been fitted for gets an unbounded item rather
    than the silent zero margin `CalibrationModel.margins` returns today: under
    ``--accuracy-domain`` (STEP A3) that hardware's candidates become
    ``unmeasured`` instead of passing as if the simulator were exact.
    """
    out: list[UncertainInput] = []
    cost = _cost_for(costs, UncertainKind.SIM_ERROR)
    affects_by_hw: dict[str, list[str]] = {}
    for island in islands:
        profile = profiles.get(island.accelerator_model)
        hardware = profile.sim_hardware if profile is not None else None
        if hardware is not None:
            affects_by_hw.setdefault(hardware, []).append(island.id)

    for hardware in sorted(calibration.hardware):
        cal = calibration.hardware[hardware]
        affects = sorted(affects_by_hw.get(hardware, []))
        for bucket in sorted(cal.errors):
            berr = cal.errors[bucket]
            # One item per (hw, bucket), but the store holds two metrics. The
            # binding one is whichever has the wider p95-vs-mean gap: moving it
            # across its range also covers the narrower metric.
            metric, stats = max(
                (("ttft", berr.ttft), ("tpot", berr.tpot)),
                key=lambda kv: kv[1].p95_abs_error - abs(kv[1].mean_error),
            )
            half = stats.p95_abs_error - abs(stats.mean_error)
            nominal = stats.mean_error
            other = "tpot" if metric == "ttft" else "ttft"
            other_stats = berr.tpot if metric == "ttft" else berr.ttft
            note = (
                f"{metric} binds: mean {stats.mean_error:+.4f}, "
                f"p95_abs {stats.p95_abs_error:.4f}, n={stats.sample_count}; "
                f"{other} mean {other_stats.mean_error:+.4f}, "
                f"p95_abs {other_stats.p95_abs_error:.4f}"
            )
            rule = grades.rule_for(UncertainKind.SIM_ERROR.value, Grade.MEASURED.value)
            if half > 0 and rule.rule is RangeRule.CALIBRATION_DERIVED:
                rng = Range(
                    lo=nominal - half, hi=nominal + half, unit="fraction", source=rule.id
                )
            else:
                # A bucket whose p95 equals its |mean| records a spread of
                # exactly zero, which is a sample-count artifact rather than a
                # perfect simulator. Since S2 that falls through to the
                # sim_error/measured default instead of leaving the one input
                # that carries most of this fixture's regret unrankable (D111).
                rng = _default_range(
                    grades, UncertainKind.SIM_ERROR, Grade.MEASURED, nominal, "fraction",
                    unbounded_source=rule.id if rule.source else "",
                )
                if half <= 0:
                    note += (
                        "; p95 equals |mean| so no width is sourced"
                        + (
                            " -> grade default applied"
                            if not rng.is_unbounded
                            else " -> unbounded"
                        )
                    )
            out.append(
                UncertainInput(
                    id=f"sim_error:{hardware}/{bucket}",
                    kind=UncertainKind.SIM_ERROR,
                    grade=Grade.MEASURED,
                    nominal=nominal,
                    range=rng,
                    cost=cost,
                    affects=affects,
                    note=note,
                )
            )

    for hardware in sorted(set(affects_by_hw) - set(calibration.hardware)):
        out.append(
            UncertainInput(
                id=f"sim_error:{hardware}/unfitted",
                kind=UncertainKind.SIM_ERROR,
                grade=Grade.PLACEHOLDER,
                nominal=0.0,
                range=_default_range(
                    grades, UncertainKind.SIM_ERROR, Grade.PLACEHOLDER, 0.0, "fraction"
                ),
                cost=cost,
                affects=sorted(affects_by_hw[hardware]),
                note=(
                    f"no calibration fitted for {hardware}: the simulator's error on this "
                    f"hardware has never been measured"
                ),
            )
        )
    return out
