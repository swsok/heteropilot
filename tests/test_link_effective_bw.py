"""A link's bandwidth is per traffic kind, not per wire (S3; deviations D112).

`LINK_BW` used to be one item per link, holding the datasheet number as though
a wire had one bandwidth. The A40 node says otherwise: over the same PCIe bridge
a direct device-to-device copy sustains 25.0 GB/s, a two-rank all-reduce 19.3,
and the four-rank all-reduce a tp=4 island actually runs 8.8 - against a
`vendor_spec` 64.0 (v3_verdict_accuracy.md A.1, PR #100). Feeding the simulator
the datasheet figure is what produced V3's -43.4 % TPOT error, and the registry
could not point at it because it priced the link as a P/D handoff while the
error came from a TP all-reduce inside one island.

These tests pin the key that separates the two, the selection rule that decides
which figure the simulator is given, and the three ways it must not overreach:
a figure is never used for traffic it was not measured at, the spec value is
never edited, and an item no closed form can price is not reported as inert.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from planner.candidate_generator import CandidateGenerator
from planner.inventory import (
    Link,
    LinkMeasurement,
    LinkType,
    Source,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
    msg_size_class_of,
)
from planner.predictor.calibration import CalibrationModel
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.uncertainty import UncertainKind, build_registry, load_costs, load_grades
from planner.uncertainty.measurement_plan import build as build_plan
from planner.uncertainty.registry import (
    LinkItemKey,
    link_item_id,
    parse_link_item_id,
)
from planner.uncertainty.sensitivity import Sensitivity

ROOT = Path(__file__).resolve().parents[1]

#: E-A1's cluster. `pcie-a40a-02` is the V3 link: 64.0 GB/s, `vendor_spec`.
V3_FIXTURE = "experiments/configs/clusters/pd-rngd-gpu-card.yaml"
#: The same cluster with PR #100's measurements filed under the S3 key.
S3_FIXTURE = "experiments/configs/clusters/pd-rngd-gpu-card-measured-pcie.yaml"

#: The measured TP=4 effective all-reduce bandwidth of the A40 PCIe bridge.
MEASURED_TP4_GBPS = 8.8
#: What the same wire's datasheet says, and what the fixture must keep saying.
SPEC_GBPS = 64.0


def _link(**over) -> Link:
    base: dict = {
        "id": "pcie-a40a-02", "src": "node_a40a/gpu0", "dst": "node_a40a/gpu2",
        "type": LinkType.PCIE, "bandwidth_gbps": SPEC_GBPS, "latency_ns": 2000.0,
        "source": Source.VENDOR_SPEC,
    }
    base.update(over)
    return Link(**base)


def _measurement(**over) -> LinkMeasurement:
    base: dict = {
        "collective": "all_reduce", "msg_size_class": "bulk", "binding": "unpinned",
        "bus_bw_gbps": MEASURED_TP4_GBPS, "world_size": 4,
        "source": Source.MEASURED,
        "method": "torch 2.10 + NCCL 2.27.5 under torchrun",
    }
    base.update(over)
    return LinkMeasurement(**base)


# ---------------------------------------------------------------------------
# (i) the key: binding, collective and message size each split the item
# ---------------------------------------------------------------------------

def test_binding_splits_the_item_id():
    """The same link bound and unbound are two items with two measurements."""
    pinned = parse_link_item_id("link_bw:pcie-a40a-02@all_reduce/w4/bulk/numa_pinned")
    unpinned = parse_link_item_id("link_bw:pcie-a40a-02@all_reduce/w4/bulk/unpinned")
    assert pinned.link_id == unpinned.link_id == "pcie-a40a-02"
    assert pinned.binding != unpinned.binding
    assert pinned.world_size == unpinned.world_size == 4
    assert link_item_id(UncertainKind.LINK_BW, "pcie-a40a-02") == "link_bw:pcie-a40a-02"


def test_a_bound_measurement_does_not_answer_for_an_unbound_deployment():
    link = _link(measurements=[_measurement(binding="numa_pinned")])
    assert link.measurement_for("all_reduce", "bulk", "numa_pinned", world_size=4)
    assert link.measurement_for("all_reduce", "bulk", "unpinned", world_size=4) is None


def test_an_unstated_binding_is_not_compared_but_is_reported():
    """S1's convention (D110): unstated on either side is unchecked, not equal.

    It still selects - refusing would throw away the only measurement this
    repository holds, whose harness recorded no affinity - but the note says the
    condition went unchecked, because an unchecked condition is not a met one.
    """
    link = _link(measurements=[_measurement(binding="unknown")])
    assert link.measurement_for("all_reduce", "bulk", "unpinned", world_size=4)
    bw, note = TopologyGraph.link_bandwidth_gbps(
        link, collective="all_reduce", binding="unpinned", world_size=4
    )
    assert bw == MEASURED_TP4_GBPS
    assert note is not None and "unchecked" in note


def test_a_stated_binding_is_preferred_over_an_unstated_one():
    link = _link(measurements=[
        _measurement(binding="unknown", bus_bw_gbps=9.9),
        _measurement(binding="unpinned", bus_bw_gbps=MEASURED_TP4_GBPS),
    ])
    hit = link.measurement_for("all_reduce", "bulk", "unpinned", world_size=4)
    assert hit is not None and hit.bus_bw_gbps == MEASURED_TP4_GBPS


def test_a_p2p_figure_is_not_an_all_reduce_figure():
    """25.0 GB/s of direct copy does not answer for the 8.8 the TP group gets."""
    link = _link(measurements=[_measurement(
        collective="p2p", bus_bw_gbps=25.0, world_size=2,
    )])
    assert link.measurement_for("p2p", "bulk", "unpinned", world_size=2)
    assert link.measurement_for("all_reduce", "bulk", "unpinned", world_size=4) is None


def test_a_four_rank_figure_is_not_a_two_rank_figure():
    """8.8 (tp=4) against 19.3 (tp=2) over one wire: 2.2x, so the group matters.

    `world_size` is in the key for this reason, which is a documented departure
    from the work order's three-part key - the repo's own data forced it, and
    the fixture that first carried both figures would not load without it.
    """
    link = _link(measurements=[_measurement(world_size=4)])
    assert link.measurement_for("all_reduce", "bulk", "unpinned", world_size=4)
    assert link.measurement_for("all_reduce", "bulk", "unpinned", world_size=2) is None


def test_a_small_message_figure_does_not_answer_for_bulk_traffic():
    """0.23 GB/s at 8 KiB is the same wire and not the same answer."""
    link = _link(measurements=[_measurement(
        msg_size_class="small", bus_bw_gbps=0.225, msg_bytes=[8192],
    )])
    assert link.measurement_for("all_reduce", "small", "unpinned", world_size=4)
    assert link.measurement_for("all_reduce", "bulk", "unpinned", world_size=4) is None


def test_msg_size_class_must_match_the_bytes_swept():
    """A figure labelled with the wrong band would be selected for traffic it
    was never measured at, so the label is checked against the sweep."""
    with pytest.raises(ValueError, match="msg_size_class"):
        _measurement(msg_size_class="bulk", msg_bytes=[8192])
    _measurement(msg_size_class="bulk", msg_bytes=[4 << 20, 64 << 20])


def test_measured_bands_match_the_committed_sweep():
    """The band boundaries are where PR #100's own curve turns, not round numbers."""
    assert msg_size_class_of(8 * 1024) == "small"
    assert msg_size_class_of(1 << 20) == "mid"
    assert msg_size_class_of(4 << 20) == "bulk"
    assert msg_size_class_of(64 << 20) == "bulk"


def test_a_measurement_must_name_its_method():
    with pytest.raises(ValueError, match="method"):
        LinkMeasurement(
            collective="all_reduce", msg_size_class="bulk", binding="unpinned",
            bus_bw_gbps=MEASURED_TP4_GBPS, world_size=4, source=Source.MEASURED,
        )


def test_one_measurement_per_key():
    with pytest.raises(ValueError, match="two measurements"):
        _link(measurements=[_measurement(), _measurement(bus_bw_gbps=9.0)])


# ---------------------------------------------------------------------------
# (ii) the simulator is given the measured value when one answers
# ---------------------------------------------------------------------------

def test_a_measurement_reaches_the_simulator_input():
    """The whole point: the TP dimension's link_bw becomes the measured figure.

    An intra-island link's bandwidth is the min `island_interconnect` reduces
    into the simulator's scalar `link_bw`, so this is the path V3's error came
    down - not the P/D transfer term the registry used to price.
    """
    cluster = load_cluster_spec(ROOT / S3_FIXTURE)
    islands = detect_islands(cluster)
    a40 = next(i for i in islands if "a40a" in i.id)
    graph = TopologyGraph(cluster)

    bw, _lat, notes = graph.island_interconnect(a40, world_size=4)
    assert bw == pytest.approx(MEASURED_TP4_GBPS)
    assert any("measured all_reduce" in n for n in notes)
    assert graph.reduce_for_simulator(
        [a40], world_sizes=[4]
    ).link_bw_gbps == pytest.approx(MEASURED_TP4_GBPS)


def test_the_spec_value_is_used_when_no_measurement_answers():
    """A tp=2 group on the same island gets no tp=4 figure, and says which
    measurements the link does carry rather than substituting one."""
    cluster = load_cluster_spec(ROOT / S3_FIXTURE)
    islands = detect_islands(cluster)
    a40 = next(i for i in islands if "a40a" in i.id)
    graph = TopologyGraph(cluster)

    bw, _lat, notes = graph.island_interconnect(a40, world_size=3)
    assert bw == pytest.approx(SPEC_GBPS)
    assert any("no measurement answers" in n for n in notes)


def test_a_collective_is_never_consulted_unless_it_is_asked_for():
    """Every pre-S3 caller asks for a bandwidth with no traffic kind, and must
    keep getting the nominal value - a factor of 7 hangs on it here."""
    link = _link(measurements=[_measurement()])
    bw, note = TopologyGraph.link_bandwidth_gbps(link)
    assert (bw, note) == (SPEC_GBPS, None)
    assert TopologyGraph.effective_bandwidth_gbps([link]) == SPEC_GBPS


def test_the_unmeasured_fixture_is_unaffected():
    """E-A1's cluster carries no measurements, so its compile cannot move."""
    cluster = load_cluster_spec(ROOT / V3_FIXTURE)
    islands = detect_islands(cluster)
    a40 = next(i for i in islands if "a40a" in i.id)
    graph = TopologyGraph(cluster)
    for ranks in (None, 2, 4):
        bw, _lat, notes = graph.island_interconnect(a40, world_size=ranks)
        assert bw == pytest.approx(SPEC_GBPS)
        assert notes == []


# ---------------------------------------------------------------------------
# (iii) the vendor value is never edited (absolute rule A3)
# ---------------------------------------------------------------------------

def test_the_spec_bandwidth_column_is_untouched_in_the_fixture():
    """The measured figure sits BESIDE the datasheet number, not over it.

    The file this replaces had 8.8 written into `bandwidth_gbps` with
    `source: measured`, which lost both the datasheet value and the fact that
    8.8 is a four-rank collective figure rather than a property of the wire.
    """
    raw = yaml.safe_load((ROOT / S3_FIXTURE).read_text())
    links = {link["id"]: link for link in raw["links"]}
    for link_id in ("pcie-a40a-02", "pcie-a40b-02"):
        link = links[link_id]
        assert link["bandwidth_gbps"] == SPEC_GBPS
        assert link["source"] == "vendor_spec"
        keys = {
            (m["collective"], m["msg_size_class"], m["binding"])
            for m in link["measurements"]
        }
        assert ("all_reduce", "bulk", "unpinned") in keys


def test_measure_apply_appends_rather_than_overwrites(tmp_path):
    """`measure-apply` on a keyed id files the figure and leaves the spec alone."""
    from planner.__main__ import main

    source = tmp_path / "cluster.yaml"
    source.write_text((ROOT / V3_FIXTURE).read_text())
    plan = tmp_path / "plan.yaml"
    plan.write_text("recommended_plan: {}\n")

    rc = main([
        "measure-apply", "--plan", str(plan),
        "--input", "link_bw:pcie-a40a-02@all_reduce/w4/bulk/unpinned",
        "--value", str(MEASURED_TP4_GBPS),
        "--method", "torch 2.10 + NCCL 2.27.5 under torchrun",
        "--raw", "outputs/p2_evidence/link/tp4.json",
        "--cluster", str(source), "--no-replan",
    ])
    assert rc == 0

    written = yaml.safe_load((tmp_path / "cluster.measured.yaml").read_text())
    link = next(link for link in written["links"] if link["id"] == "pcie-a40a-02")
    assert link["bandwidth_gbps"] == SPEC_GBPS          # A3: not edited
    assert link["source"] == "vendor_spec"
    assert link["measurements"] == [{
        "collective": "all_reduce", "msg_size_class": "bulk", "binding": "unpinned",
        "bus_bw_gbps": MEASURED_TP4_GBPS, "world_size": 4, "source": "measured",
        "method": "torch 2.10 + NCCL 2.27.5 under torchrun",
        "raw": "outputs/p2_evidence/link/tp4.json",
    }]
    # And the copy loads, which is what the re-plan would do next.
    assert load_cluster_spec(tmp_path / "cluster.measured.yaml")


def test_measure_apply_refuses_a_measurement_with_no_method(tmp_path):
    """An unattributed figure is not a measurement (absolute rule 3), and the
    tool matters here: this repo's torch probe is not `nccl-tests`."""
    from planner.__main__ import main

    source = tmp_path / "cluster.yaml"
    source.write_text((ROOT / V3_FIXTURE).read_text())
    plan = tmp_path / "plan.yaml"
    plan.write_text("recommended_plan: {}\n")
    rc = main([
        "measure-apply", "--plan", str(plan),
        "--input", "link_bw:pcie-a40a-02@all_reduce/w4/bulk/unpinned",
        "--value", "8.8", "--cluster", str(source), "--no-replan",
    ])
    assert rc == 1


def test_measure_apply_refuses_a_key_with_no_group_size(tmp_path):
    """`@all_reduce/bulk/unpinned` is the pre-world_size spelling and names no
    group, so it is refused rather than silently defaulted."""
    from planner.__main__ import main

    source = tmp_path / "cluster.yaml"
    source.write_text((ROOT / V3_FIXTURE).read_text())
    plan = tmp_path / "plan.yaml"
    plan.write_text("recommended_plan: {}\n")
    with pytest.raises(ValueError, match="expected"):
        main([
            "measure-apply", "--plan", str(plan),
            "--input", "link_bw:pcie-a40a-02@all_reduce/bulk/unpinned",
            "--value", "8.8", "--method", "nccl-tests all_reduce_perf",
            "--cluster", str(source), "--no-replan",
        ])


# ---------------------------------------------------------------------------
# The registry: one item per traffic kind, and none of them inert by default
# ---------------------------------------------------------------------------

def _registry(cluster_path):
    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    cluster = load_cluster_spec(ROOT / cluster_path)
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    return build_registry(
        cluster, profiles, islands, CalibrationModel.identity(),
        load_grades(), spec, load_costs(),
    )


def test_an_intra_island_link_is_keyed_by_its_collective():
    reg = _registry(V3_FIXTURE)
    ids = [i.id for i in reg.by_kind(UncertainKind.LINK_BW)]
    keyed = [i for i in ids if i.startswith("link_bw:pcie-a40a-02@")]
    assert keyed, f"no keyed item for the V3 link among {ids}"
    key = parse_link_item_id(keyed[0])
    assert key.collective == "all_reduce"
    assert key.world_size == 4, "the tp=4 A40 island is what puts traffic here"
    assert key.msg_size_class == "bulk"
    # The fixture states no binding, so the item says so rather than guessing.
    assert key.binding == "unknown"


def test_link_lat_stays_one_item_per_link():
    """No link latency in this repository is measured, so there is nothing for
    a key to select between and `path_latency_ns` applies it to all traffic."""
    reg = _registry(V3_FIXTURE)
    ids = [i.id for i in reg.by_kind(UncertainKind.LINK_LAT)]
    assert "link_lat:pcie-a40a-02" in ids
    assert not any("@" in i for i in ids)


def test_a_measured_collective_leaves_the_registry():
    """The standing rule: a measured input is counted, never listed."""
    before = _registry(V3_FIXTURE)
    after = _registry(S3_FIXTURE)
    all_reduce = [
        i.id for i in after.by_kind(UncertainKind.LINK_BW)
        if parse_link_item_id(i.id).collective == "all_reduce"
        and parse_link_item_id(i.id).link_id in ("pcie-a40a-02", "pcie-a40b-02")
    ]
    assert all_reduce == []
    assert (after.measured_count.get("link_bw", 0)
            > before.measured_count.get("link_bw", 0))


def test_an_all_reduce_item_needs_simulation_rather_than_reading_as_inert():
    """The defect S3 exists to close.

    The closed form prices only the P/D handoff, so an intra-island item came
    back with every metric untouched and scored a regret of exactly zero -
    `inert`, i.e. "measured, it would change no plan". It was the input that
    explained a -43.4 % error. It must land in its own list instead.
    """
    priced = Sensitivity(
        input_id="link_bw:pcie-a40a-02@all_reduce/w4/bulk/unpinned",
        kind="link_bw", flip=False, delta_regret=None,
        requires_resimulation=True, cost_hours=0.114,
    )
    plan = build_plan([priced])
    assert plan.needs_resimulation == [priced.input_id]
    assert plan.inert == []
    assert plan.undecidable == []


def test_an_unbounded_item_is_still_undecidable_not_needing_simulation():
    """The two "no regret" answers stay distinguishable in both directions."""
    plan = build_plan([Sensitivity(
        input_id="link_lat:pcie-a40a-02", kind="link_lat", flip=False,
        delta_regret=None, cost_hours=None,
    )])
    assert plan.undecidable == ["link_lat:pcie-a40a-02"]
    assert plan.needs_resimulation == []


@pytest.mark.parametrize("bad", [
    "link_bw:pcie-a40a-02@all_reduce/bulk",              # no binding
    "link_bw:pcie-a40a-02@all_reduce/bulk/unpinned",     # the pre-world_size form
    "link_bw:pcie-a40a-02@all_reduce/w/bulk/unpinned",   # no number
    "link_bw:pcie-a40a-02@all_reduce/4/bulk/unpinned",   # no w
])
def test_parse_rejects_a_half_written_key(bad):
    with pytest.raises(ValueError, match="expected"):
        parse_link_item_id(bad)


def test_parse_accepts_the_unkeyed_form():
    """A link id with no traffic named still parses - it is what LINK_LAT uses
    and what a pre-S3 invocation passes."""
    assert parse_link_item_id("link_bw:x") == LinkItemKey(link_id="x")


# ---------------------------------------------------------------------------
# The bound and the predictor must read the same wire the same way
# ---------------------------------------------------------------------------

def test_the_all_reduce_floor_uses_the_candidates_own_group_size():
    """A pruning stage must be a relaxation of the feasibility test.

    The generator's TP all-reduce floor reads `island_interconnect`. Taking the
    ISLAND's size rather than the candidate's would price a tp=2 group at the
    four-rank 8.8 GB/s where that group really gets 19.29, make the floor larger
    than the truth, and reject candidates §5.6 accepts. 2.2x of bandwidth is a
    lot of floor.
    """
    cluster = load_cluster_spec(ROOT / S3_FIXTURE)
    islands = detect_islands(cluster)
    a40 = next(i for i in islands if "a40a" in i.id)
    graph = TopologyGraph(cluster)

    four = graph.island_interconnect(a40, world_size=4)[0]
    two = graph.island_interconnect(a40, world_size=2)[0]
    assert two > four, "the two-rank figure must be the faster one here"
    # The island's own size is 4, so the default would hand a tp=2 candidate the
    # slower figure. The generator passes world_size=tp instead.
    assert graph.island_interconnect(a40)[0] == pytest.approx(four)

    # Behavioural check: record what world_size the floor asks with.
    asked: list[int | None] = []
    real = graph.island_interconnect

    def spy(island, *, world_size=None):
        asked.append(world_size)
        return real(island, world_size=world_size)

    graph.island_interconnect = spy  # type: ignore[method-assign]

    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    profiles = load_profiles_for(cluster, ROOT)
    gen = CandidateGenerator(
        spec, cluster, detect_islands(cluster, profiles), profiles, topology=graph
    )
    gen.generate()
    assert asked, "the floor never consulted the topology"
    assert None not in asked, (
        "the all-reduce floor asked with no world_size, so it took the island's "
        "size; a tp=2 candidate would be priced at the four-rank figure"
    )
    assert set(asked) <= {1, 2, 4}, asked


def test_the_handoff_term_and_the_simulator_price_one_hop_alike():
    """`apply_pd_transfer_cost` adds a term beside the simulator's own output,
    so both must ask the topology the same question about the same wire."""
    link = _link(
        id="fabric-x", src="node_a/gpu0", dst="node_b/gpu0",
        measurements=[_measurement(collective="p2p", bus_bw_gbps=25.0, world_size=2)],
    )
    asked = TopologyGraph.effective_bandwidth_gbps(
        [link], collective="p2p", msg_size_class="bulk", world_size=2
    )
    assert asked == pytest.approx(25.0)
    # And the all-reduce question about the same wire gets a different answer,
    # which is why "the bandwidth of this link" cannot be a shared constant.
    assert TopologyGraph.effective_bandwidth_gbps(
        [link], collective="all_reduce", msg_size_class="bulk", world_size=4
    ) == pytest.approx(SPEC_GBPS)
