"""The uncertain-input registry (WORK_ORDER_uncertainty_planner.md STEP A1).

Every count these tests assert is recomputed from the fixture YAML rather than
written in: the work order names concrete numbers for `pd-rngd-gpu.yaml`
(26 links, 10 measured) and hardcoding them would turn an honest registry bug
into a passing test the day someone edits the fixture.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from planner.inventory import (
    ClusterSpecV2,
    Source,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.plan import PlannerOutput
from planner.predictor.calibration import CalibrationModel, load_calibration
from planner.spec import load_service_spec
from planner.uncertainty import (
    Grade,
    UncertainKind,
    build_registry,
    load_costs,
    load_grades,
)
from planner.uncertainty.grades import GradeRule, GradesTable, RangeRule
from planner.util import tier as tierutil

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def grades():
    return load_grades()


@pytest.fixture
def costs():
    return load_costs()


@pytest.fixture
def llama_spec():
    return load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")


def _registry(cluster_path, spec, grades, costs, calibration=None):
    cluster = load_cluster_spec(ROOT / cluster_path)
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    return build_registry(
        cluster,
        profiles,
        islands,
        calibration or CalibrationModel.identity(),
        grades,
        spec,
        costs,
    )


def _link_sources(cluster_path: str) -> Counter:
    """Count link `source` values straight out of the YAML, not the registry."""
    raw = yaml.safe_load((ROOT / cluster_path).read_text())
    return Counter(link.get("source", "placeholder") for link in raw["links"])


# --- links ----------------------------------------------------------------

PD_FIXTURE = "experiments/configs/clusters/pd-rngd-gpu.yaml"


def test_link_items_cover_exactly_the_non_measured_links(llama_spec, grades, costs) -> None:
    """§2.3: a measured link is counted, never listed - no zero-width items."""
    sources = _link_sources(PD_FIXTURE)
    uncertain = sum(n for s, n in sources.items() if s != Source.MEASURED.value)
    measured = sources[Source.MEASURED.value]

    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)

    assert len(reg.by_kind(UncertainKind.LINK_BW)) == uncertain
    assert len(reg.by_kind(UncertainKind.LINK_LAT)) == uncertain
    assert reg.measured_count["link_bw"] == measured
    assert reg.measured_count["link_lat"] == measured
    assert reg.total_for(UncertainKind.LINK_BW) == sum(sources.values())


def test_placeholder_links_are_unbounded_and_vendor_spec_follows_grades(
    llama_spec, grades, costs
) -> None:
    """Which links get a finite range is decided by grades.yaml, not by this test."""
    sources = _link_sources(PD_FIXTURE)
    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)
    bw = {i.id: i for i in reg.by_kind(UncertainKind.LINK_BW)}

    placeholders = [i for i in bw.values() if i.grade is Grade.PLACEHOLDER]
    assert len(placeholders) == sources[Source.PLACEHOLDER.value]
    assert all(i.range.is_unbounded for i in placeholders)

    vendor = [i for i in bw.values() if i.grade is Grade.VENDOR_SPEC]
    assert len(vendor) == sources[Source.VENDOR_SPEC.value]
    rule = grades.rule_for("link_bw", "vendor_spec")
    if rule.rule is RangeRule.RATIO_FLOOR:
        for item in vendor:
            assert not item.range.is_unbounded
            assert item.range.hi == pytest.approx(item.nominal)
            assert item.range.lo == pytest.approx(item.nominal * rule.r_min)
    else:
        assert all(i.range.is_unbounded for i in vendor)


def test_no_item_has_zero_width(llama_spec, grades, costs) -> None:
    """The §2.3 invariant Stage B relies on: a listed item is genuinely uncertain."""
    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)
    assert reg.items
    for item in reg.items:
        assert item.range.is_unbounded or item.range.lo != item.range.hi


def test_link_affects_names_the_islands_the_link_serves(llama_spec, grades, costs) -> None:
    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)
    onpkg = next(i for i in reg.by_kind(UncertainKind.LINK_BW) if i.id.endswith("onpkg-rngd0-01"))
    # An on-package PE-to-PE link sits inside one island and serves only it.
    assert onpkg.affects == ["furiosa-rngd-node_rngd0"]


def test_inter_island_links_report_island_pairs(llama_spec) -> None:
    """The island-pair half of `affects`, tested on the mapping directly.

    Every inter-island link in this fixture is `measured`, so none of them
    becomes a registry item - which is why this exercises `_link_affects`
    rather than the registry: the pairing logic still has to be right for a
    fixture whose fabric is not measured.
    """
    from planner.uncertainty.registry import _link_affects

    cluster = load_cluster_spec(ROOT / PD_FIXTURE)
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    affects = _link_affects(cluster, islands)

    pairs = affects["fabric-rngd0-a40a"]
    assert pairs, "an inter-island link must name the island pairs it carries"
    assert all("->" in p for p in pairs)
    assert "cuda-a40-node_a40a->furiosa-rngd-node_rngd0" in pairs

    # Intra-island links never produce a pair label.
    assert affects["onpkg-rngd0-01"] == ["furiosa-rngd-node_rngd0"]


# --- sim_error ------------------------------------------------------------

def test_calibrated_hardware_gets_one_item_per_bucket(llama_spec, grades, costs) -> None:
    calibration = load_calibration(ROOT / "profiles/calibration/a40.yaml")
    reg = _registry(
        "experiments/configs/clusters/a40x8.yaml", llama_spec, grades, costs, calibration
    )
    items = reg.by_kind(UncertainKind.SIM_ERROR)

    buckets = list(calibration.hardware["A40"].errors)
    assert [i.id for i in items] == [f"sim_error:A40/{b}" for b in buckets]

    item = items[0]
    assert item.grade is Grade.MEASURED
    # Scalar mode: the width comes from the store, centred on the mean error.
    assert not item.range.is_unbounded
    stats = calibration.hardware["A40"].errors[buckets[0]].tpot
    assert item.nominal == pytest.approx(stats.mean_error)
    half = stats.p95_abs_error - abs(stats.mean_error)
    assert item.range.hi - item.range.lo == pytest.approx(2 * half)


def test_hardware_with_no_calibration_is_unbounded_not_silently_zero(
    llama_spec, grades, costs
) -> None:
    """Today `CalibrationModel.margins` returns (0, 0) for unfitted hardware.

    That is indistinguishable from "the simulator is exact here", which is the
    gap STEP A3 closes. The registry must say unbounded instead.
    """
    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)
    items = {i.id: i for i in reg.by_kind(UncertainKind.SIM_ERROR)}
    assert "sim_error:A40/unfitted" in items
    assert items["sim_error:A40/unfitted"].range.is_unbounded


# --- profiles: the weaker of two signals ----------------------------------

PROXY_FIXTURE = "experiments/configs/clusters/pd-4combo-sim.yaml"


def test_bundle_tier_alone_would_call_the_proxy_measured(llama_spec) -> None:
    """Documents the regression this rule exists to prevent (§2.3).

    `ascend-sim-proxy` borrows the real RTXPRO6000 bundle, so a grade read from
    the bundle tier alone passes the repository's most misleading input as a
    measurement. If this assertion ever fails the guard below is testing
    nothing, and it should be reworked rather than deleted.
    """
    variant = tierutil.resolve_variant(
        llama_spec.service.dtype, llama_spec.service.kv_cache_dtype
    )
    tier = tierutil.resolve_bundle_tier(None, "RTXPRO6000", llama_spec.model, variant)
    assert tier is tierutil.ProfileTier.MEASURED


def test_proxy_profile_is_graded_placeholder_despite_a_measured_bundle(
    llama_spec, grades, costs
) -> None:
    reg = _registry(PROXY_FIXTURE, llama_spec, grades, costs)
    proxies = [
        i for i in reg.by_kind(UncertainKind.PROFILE) if "ascend-sim-proxy" in i.id
    ]
    assert proxies, "the NPU islands must produce PROFILE items"
    for item in proxies:
        assert item.grade is Grade.PLACEHOLDER
        assert item.range.is_unbounded
        assert "proxy:" in item.note
        assert "bundle RTXPRO6000 tier=measured" in item.note
        assert "profile source=placeholder" in item.note


def test_vendor_spec_profile_with_a_measured_bundle_is_still_uncertain(
    llama_spec, grades, costs
) -> None:
    """profiles/accelerators/rtxpro6000.yaml declares `source: vendor_spec`.

    The work order's STEP A1 text assumed `measured` here and expected these
    islands to be counted rather than listed. The file wins (CLAUDE.md, "when
    spec and reality diverge"): its memory and bandwidth are copied from
    upstream configs with no stated derivation, so vendor_spec is correct and
    the island is genuinely uncertain.
    """
    cluster = load_cluster_spec(ROOT / PROXY_FIXTURE)
    profiles = load_profiles_for(cluster, ROOT)
    assert profiles["RTXPRO6000"].source is Source.VENDOR_SPEC

    reg = _registry(PROXY_FIXTURE, llama_spec, grades, costs)
    gpu = [i for i in reg.by_kind(UncertainKind.PROFILE) if "rtxpro6000" in i.id]
    assert gpu
    assert all(i.grade is Grade.VENDOR_SPEC for i in gpu)
    assert reg.measured_count.get("profile", 0) == 0


def test_fully_measured_profiles_are_counted_not_listed(llama_spec, grades, costs) -> None:
    """The contrast the proxy test needs: a40.yaml is measured on both signals."""
    cluster = load_cluster_spec(ROOT / "experiments/configs/clusters/a40x8.yaml")
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    assert profiles["A40"].source is Source.MEASURED

    reg = _registry("experiments/configs/clusters/a40x8.yaml", llama_spec, grades, costs)
    assert reg.by_kind(UncertainKind.PROFILE) == []
    assert reg.measured_count["profile"] == len(islands)


@pytest.mark.skipif(
    not (ROOT / "profiler/perf/ASCEND_TARGET-t0").exists(),
    reason="needs the gitignored synthetic Tier 0 bundle; run scripts/gen-tier0-bundles.sh",
)
def test_weaker_of_the_two_signals_wins_for_a_tier0_bundle(grades, costs) -> None:
    """ascend_target: profile vendor_spec (rank 3) + bundle -t0 analytical (1)."""
    spec = load_service_spec(ROOT / "examples/service_specs/qwen3-32b.yaml")
    reg = _registry("examples/clusters/hetero-gpu-ascend.yaml", spec, grades, costs)
    ascend = next(
        i for i in reg.by_kind(UncertainKind.PROFILE) if "ascend-target" in i.id
    )
    assert ascend.grade is Grade.ANALYTICAL
    assert "profile source=vendor_spec" in ascend.note
    assert "tier=analytical" in ascend.note
    # A -t0 suffix names the same hardware synthetically; it is not a proxy.
    assert "proxy:" not in ascend.note

    rule = grades.rule_for("profile", "analytical")
    if rule.rule is RangeRule.SYMMETRIC_FRACTION:
        assert ascend.range.lo == pytest.approx(1.0 - rule.value)
        assert ascend.range.hi == pytest.approx(1.0 + rule.value)


# --- grades.yaml contract -------------------------------------------------

def test_absent_combination_is_unbounded_without_raising(grades) -> None:
    rule = grades.rule_for("link_bw", "no_such_grade")
    assert rule.rule is RangeRule.UNBOUNDED
    assert rule.value is None and rule.r_min is None


def test_a_number_without_a_source_is_rejected() -> None:
    """Absolute rule A1, enforced by the loader rather than by review."""
    with pytest.raises(ValueError, match="no source"):
        GradeRule(kind="profile", grade="analytical", rule="symmetric_fraction", value=0.1)

    # ... and the same row with a source is fine.
    GradeRule(
        kind="profile", grade="analytical", rule="symmetric_fraction",
        value=0.1, source="docs/somewhere.md",
    )


def test_an_unbounded_rule_may_not_carry_a_number() -> None:
    with pytest.raises(ValueError, match="must carry no number"):
        GradeRule(
            kind="profile", grade="placeholder", rule="unbounded",
            value=0.1, source="docs/somewhere.md",
        )


def test_shipped_grades_table_has_a_source_for_every_number(grades) -> None:
    for rule in grades.rules:
        if rule.value is not None or rule.r_min is not None:
            assert rule.source.strip(), f"{rule.id} carries a number with no source"
    assert grades.digest, "the table must digest for provenance"


def test_shipped_costs_table_has_a_source_for_every_hour(costs) -> None:
    for cost in costs.costs:
        if cost.hours is not None:
            assert cost.source.strip(), f"{cost.kind} carries hours with no source"


def test_duplicate_rows_are_rejected() -> None:
    row = {"kind": "link_bw", "grade": "placeholder", "rule": "unbounded"}
    with pytest.raises(ValueError, match="duplicate row"):
        GradesTable.model_validate({"rules": [row, dict(row)]})


# --- golden: the default path is untouched --------------------------------

def test_planner_output_defaults_to_no_registry() -> None:
    out = PlannerOutput(feasible=False, service_model="m", cluster_id="c")
    assert out.uncertain_inputs is None


def test_default_yaml_dump_omits_the_new_key(tmp_path) -> None:
    """Rule A4: the flagless YAML must be byte-identical to before this STEP.

    `_write_output` has no `exclude_none` - other optional fields DO appear as
    null in the golden outputs - so the new key has to be dropped on its own.
    """
    from planner.__main__ import _write_output

    out = PlannerOutput(feasible=False, service_model="m", cluster_id="c")
    path = tmp_path / "plan.yaml"
    _write_output(out, path)
    assert "uncertain_inputs" not in path.read_text()
    assert "uncertain_inputs" not in yaml.safe_load(path.read_text())


def test_dump_carries_the_registry_when_it_is_present(
    tmp_path, llama_spec, grades, costs
) -> None:
    from planner.__main__ import _write_output

    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)
    out = PlannerOutput(
        feasible=False, service_model="m", cluster_id="c", uncertain_inputs=reg
    )
    path = tmp_path / "plan.yaml"
    _write_output(out, path)
    loaded = yaml.safe_load(path.read_text())
    assert len(loaded["uncertain_inputs"]["items"]) == len(reg.items)
    assert loaded["uncertain_inputs"]["measured_count"]["link_bw"] == 10


# --- renderer -------------------------------------------------------------

def test_renderer_shows_coverage_and_marks_unbounded(llama_spec, grades, costs) -> None:
    from planner.render import render_uncertain_inputs

    reg = _registry(PD_FIXTURE, llama_spec, grades, costs)
    text = render_uncertain_inputs(reg)
    assert "coverage:" in text
    assert "link_bw 16/26 uncertain" in text
    assert "**" in text
    # Unbounded items head the table: every marked row precedes every plain one.
    ids = {i.id for i in reg.items}
    rows = [ln for ln in text.splitlines() if any(i in ln for i in ids)]
    assert len(rows) == len(reg.items)
    marked = [ln.strip().startswith("**") for ln in rows]
    assert marked[0] is True
    assert marked == sorted(marked, reverse=True)


def test_renderer_section_is_absent_without_a_registry(llama_spec) -> None:
    from planner.render import render

    out = PlannerOutput(feasible=False, service_model="m", cluster_id="c", reason="none")
    assert "Uncertain inputs" not in render(out)


def test_cluster_fixture_is_the_one_the_work_order_describes() -> None:
    """Guards the premise of every count above (26 links: 10/14/2)."""
    sources = _link_sources(PD_FIXTURE)
    cluster: ClusterSpecV2 = load_cluster_spec(ROOT / PD_FIXTURE)
    assert len(cluster.links) == sum(sources.values())
    assert set(sources) <= {s.value for s in Source}
