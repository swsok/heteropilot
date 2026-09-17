"""Default ranges by grade (domain-scoping work order S2; deviations D111).

Before S2 an uncertain input whose own (kind, grade) sourced no width left the
measurement plan's ranking entirely: no interval to sweep, so no regret, so the
item was listed as "cannot be decided before measuring" and never competed for
a server-hour. V3 measured what that costs. `link_bw:pcie-a40a-02` is a
`vendor_spec` link with no sourced range - undecidable, unranked - and it was
the input that explained a -43.4 % TPOT error, for 0.114 h of measurement.

These tests pin the layer that closes that path and the three ways it must not
overreach: a default never replaces a sourced width, a grade with no default is
still undecidable, and a range built from a default says so everywhere it goes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.predictor.calibration import CalibrationModel
from planner.spec import load_service_spec
from planner.uncertainty import Grade, UncertainKind, build_registry, load_costs, load_grades
from planner.uncertainty.grades import DefaultRule, GradeDefault, GradeRule, GradesTable
from planner.uncertainty.measurement_plan import build as build_plan
from planner.uncertainty.registry import Range, UncertainInput, _range_from_rule
from planner.uncertainty.sensitivity import Sensitivity

ROOT = Path(__file__).resolve().parents[1]

#: The V3 cluster: `pcie-a40a-02` lives here, `source: vendor_spec`, 64.0 GB/s.
V3_FIXTURE = "experiments/configs/clusters/pd-rngd-gpu-card.yaml"


@pytest.fixture
def grades():
    return load_grades()


@pytest.fixture
def costs():
    return load_costs()


def _registry(cluster_path, grades, costs):
    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    cluster = load_cluster_spec(ROOT / cluster_path)
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    return build_registry(
        cluster, profiles, islands, CalibrationModel.identity(), grades, spec, costs
    )


def _s(input_id: str, kind: str, delta_regret, cost_hours, range_source="sourced"):
    return Sensitivity(
        input_id=input_id, kind=kind, flip=False, delta_regret=delta_regret,
        cost_hours=cost_hours, range_source=range_source,
    )


# --- (i) the path V3 found ---------------------------------------------------

def test_the_v3_link_is_ranked_instead_of_undecidable(grades, costs) -> None:
    """`pcie-a40a-02` is the case the whole step exists for."""
    reg = _registry(V3_FIXTURE, grades, costs)
    item = next(i for i in reg.items if i.id == "link_bw:pcie-a40a-02")

    assert item.grade is Grade.VENDOR_SPEC
    assert grades.rule_for("link_bw", "vendor_spec").rule.value == "unbounded", (
        "this test is only meaningful while the link_bw/vendor_spec ROW sources no "
        "width; if it ever gains one, the default is no longer what ranks this link"
    )
    assert not item.range.is_unbounded
    assert item.range.range_source == "default"

    default = grades.default_for("link_bw", "vendor_spec")
    assert item.range.lo == pytest.approx(item.nominal * default.lo)
    assert item.range.hi == pytest.approx(item.nominal * default.hi)
    assert item.range.source == f"defaults:{default.id}"


def test_a_default_ranged_input_reaches_the_ranking_and_is_marked(costs) -> None:
    """It competes on dR/h like any other item, and carries the caveat with it."""
    plan = build_plan(
        [_s("link_bw:pcie-a40a-02", "link_bw", 12.0, 0.114, range_source="default"),
         _s("sim_error:RNGD-CARD/b", "sim_error", 3.0, 1.0)],
        costs,
    )
    assert [i.input_id for i in plan.items] == [
        "link_bw:pcie-a40a-02", "sim_error:RNGD-CARD/b"
    ]
    assert plan.items[0].range_source == "default"
    assert plan.items[1].range_source == "sourced"
    assert not plan.undecidable

    from planner.render import render_measurement_plan

    text = render_measurement_plan(plan)
    assert "[default range]" in text


# --- (ii) a sourced width is never replaced ----------------------------------

def test_a_sourced_width_wins_over_a_grade_default() -> None:
    table = GradesTable(
        rules=[GradeRule(kind="profile", grade="analytical", rule="symmetric_fraction",
                         value=0.25, source="a rule with a number")],
        defaults=[GradeDefault(grade="analytical", rule=DefaultRule.RELATIVE,
                               lo=0.1, hi=9.0, source="a default nobody should reach")],
    )
    rng = _range_from_rule(table, UncertainKind.PROFILE, Grade.ANALYTICAL, 1.0, "fraction")

    assert rng.range_source == "sourced"
    assert (rng.lo, rng.hi) == pytest.approx((0.75, 1.25))
    assert rng.source == "profile/analytical"


def test_every_sourced_range_in_a_real_registry_stays_sourced(grades, costs) -> None:
    """The layer is additive: it may only fill in where there was nothing."""
    reg = _registry(V3_FIXTURE, grades, costs)
    for item in reg.items:
        rule = grades.rule_for(item.kind.value, item.grade.value)
        if rule.rule.value in ("symmetric_fraction", "ratio_floor"):
            assert item.range.range_source == "sourced", item.id


# --- (iii) what remains undecidable ------------------------------------------

def test_only_a_grade_with_no_default_is_undecidable() -> None:
    """`user_defined` has no default row, deliberately: a what-if is the user's."""
    table = GradesTable(
        defaults=[GradeDefault(grade="placeholder", rule=DefaultRule.RELATIVE,
                               lo=0.5, hi=1.5, source="test")],
    )
    assert table.default_for("link_bw", "user_defined") is None

    priced = _range_from_rule(table, UncertainKind.LINK_BW, Grade.PLACEHOLDER, 64.0, "gbps")
    unpriced = _range_from_rule(table, UncertainKind.LINK_BW, Grade.USER_DEFINED, 64.0, "gbps")
    assert not priced.is_unbounded and priced.range_source == "default"
    assert unpriced.is_unbounded

    plan = build_plan([_s("link_bw:a", "link_bw", 1.0, 1.0),
                       _s("link_bw:b", "link_bw", None, 1.0)])
    assert plan.undecidable == ["link_bw:b"]


def test_the_committed_table_keeps_user_defined_undecidable(grades) -> None:
    for kind in ("link_bw", "link_lat", "profile", "power", "sim_error"):
        assert grades.default_for(kind, "user_defined") is None


def test_a_relative_default_cannot_manufacture_a_width_at_a_zero_nominal() -> None:
    """§2.3's invariant: no zero-width item exists in the registry."""
    table = GradesTable(
        defaults=[GradeDefault(grade="placeholder", rule=DefaultRule.RELATIVE,
                               lo=0.5, hi=1.5, source="test")],
    )
    rng = _range_from_rule(table, UncertainKind.PROFILE, Grade.PLACEHOLDER, 0.0, "fraction")
    assert rng.is_unbounded


# --- the shapes the two quantity kinds need ----------------------------------

def test_a_spec_latency_is_a_lower_bound_not_an_upper_one(grades, costs) -> None:
    """The grade-level vendor_spec default would have claimed the opposite."""
    reg = _registry(V3_FIXTURE, grades, costs)
    lat = next(i for i in reg.items if i.id == "link_lat:pcie-a40a-02")

    assert lat.range.range_source == "default"
    assert lat.range.lo >= lat.nominal, "a link cannot beat its spec latency"
    assert lat.range.hi > lat.nominal


def test_an_absolute_default_spans_a_zero_error_nominal() -> None:
    """`sim_error` is additive: hardware nobody calibrated has nominal 0.0."""
    table = GradesTable(
        defaults=[GradeDefault(grade="placeholder", kind="sim_error",
                               rule=DefaultRule.ABSOLUTE, half_width=0.4,
                               source="test")],
    )
    rng = _range_from_rule(table, UncertainKind.SIM_ERROR, Grade.PLACEHOLDER, 0.0, "fraction")
    assert (rng.lo, rng.hi) == pytest.approx((-0.4, 0.4))
    assert rng.range_source == "default"


def test_a_kind_row_overrides_the_grade_row() -> None:
    table = GradesTable(
        defaults=[
            GradeDefault(grade="vendor_spec", rule=DefaultRule.RELATIVE,
                         lo=0.125, hi=1.0, source="test"),
            GradeDefault(grade="vendor_spec", kind="link_lat", rule=DefaultRule.RELATIVE,
                         lo=1.0, hi=8.0, source="test"),
        ],
    )
    assert table.default_for("link_bw", "vendor_spec").kind == ""
    assert table.default_for("link_lat", "vendor_spec").kind == "link_lat"


# --- (iv) what the layer must NOT move ---------------------------------------

def test_an_experiment_supplied_range_is_sourced(costs) -> None:
    """E-B1/B2/B3 hand `analyze` the interval they degraded themselves.

    Those ranges never pass through grades.yaml, so the default layer cannot
    reach them and cannot move E-B2's flip precision or recall. This pins the
    reason rather than the result: a Range built by a caller is `sourced`.
    """
    item = UncertainInput(
        id="link_bw:probe", kind=UncertainKind.LINK_BW, grade=Grade.PLACEHOLDER,
        nominal=64.0,
        range=Range(lo=8.0, hi=64.0, unit="gbps", source="experiment: degraded value"),
    )
    assert item.range.range_source == "sourced"


def test_a_default_must_name_its_source() -> None:
    """Absolute rule A1, in the form S2 sets: a policy, never an invented number."""
    with pytest.raises(ValueError, match="no source"):
        GradeDefault(grade="vendor_spec", rule=DefaultRule.RELATIVE, lo=0.125, hi=1.0)


def test_a_default_must_pick_one_shape() -> None:
    with pytest.raises(ValueError, match="takes no half_width"):
        GradeDefault(grade="x", rule=DefaultRule.RELATIVE, lo=0.5, hi=1.5,
                     half_width=0.1, source="test")
    with pytest.raises(ValueError, match="needs half_width"):
        GradeDefault(grade="x", rule=DefaultRule.ABSOLUTE, source="test")


def test_duplicate_defaults_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate default"):
        GradesTable(defaults=[
            GradeDefault(grade="vendor_spec", rule=DefaultRule.RELATIVE, lo=0.1, hi=1.0,
                         source="test"),
            GradeDefault(grade="vendor_spec", rule=DefaultRule.RELATIVE, lo=0.2, hi=1.0,
                         source="test"),
        ])


def test_without_defaults_restores_the_pre_s2_answer(grades, costs) -> None:
    """The one escape hatch, for an experiment that needs both answers."""
    plain = grades.without_defaults()
    assert plain.rules == grades.rules
    assert plain.defaults == []

    rng = _range_from_rule(plain, UncertainKind.LINK_BW, Grade.VENDOR_SPEC, 64.0, "gbps")
    assert rng.is_unbounded
