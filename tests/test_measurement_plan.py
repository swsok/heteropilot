"""Measurement plan and measure-apply (WORK_ORDER_uncertainty_planner.md B3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from planner.uncertainty.grades import CostRule, CostsTable
from planner.uncertainty.measurement_plan import build
from planner.uncertainty.sensitivity import Sensitivity

ROOT = Path(__file__).resolve().parents[1]


def _s(input_id: str, kind: str, regret: float | None, hours: float | None,
       *, flip: bool = False, approx: bool = False) -> Sensitivity:
    return Sensitivity(
        input_id=input_id, kind=kind, flip=flip, delta_regret=regret,
        cost_hours=hours, approximation=approx,
    )


def _costs() -> CostsTable:
    return CostsTable(costs=[
        CostRule(kind="link_bw", method="gpu_host_bandwidth.py", hours=0.114,
                 exclusive=True, source="tests"),
        CostRule(kind="profile", method="python -m profiler profile", hours=2.1,
                 exclusive=True, source="tests"),
        CostRule(kind="power", method="nvidia-smi sampling", hours=0.172,
                 exclusive=False, source="tests"),
    ])


# --- budgets --------------------------------------------------------------

def test_a_zero_budget_plans_nothing_and_defers_everything() -> None:
    plan = build(
        [_s("link_bw:a", "link_bw", 100.0, 0.114),
         _s("profile:b", "profile", 500.0, 2.1)],
        _costs(), budget_hours=0.0,
    )
    assert plan.items == []
    assert [i.input_id for i in plan.uncovered] == ["link_bw:a", "profile:b"]
    assert plan.covered_regret == 0.0


def test_an_unlimited_budget_plans_every_worthwhile_item_in_order() -> None:
    plan = build(
        [_s("profile:b", "profile", 500.0, 2.1),     # 238/h
         _s("link_bw:a", "link_bw", 100.0, 0.114)],  # 877/h
        _costs(),
    )
    assert [i.input_id for i in plan.items] == ["link_bw:a", "profile:b"]
    assert [i.rank for i in plan.items] == [1, 2]
    assert plan.uncovered == []
    assert plan.covered_regret == pytest.approx(600.0)
    assert plan.total_hours == pytest.approx(2.214)


def test_the_order_is_value_for_money_not_raw_value() -> None:
    """§2.5's key is dR/cost. The bigger regret is not automatically first."""
    plan = build(
        [_s("profile:big", "profile", 500.0, 2.1),      # 238/h
         _s("link_bw:small", "link_bw", 100.0, 0.114)],  # 877/h
        _costs(),
    )
    assert plan.items[0].input_id == "link_bw:small"
    assert plan.items[0].delta_regret < plan.items[1].delta_regret


def test_a_deferred_item_keeps_the_rank_it_would_have_had() -> None:
    """"number 2 did not fit" beats a renumbered list that hides the cut."""
    plan = build(
        [_s("link_bw:a", "link_bw", 100.0, 0.114),
         _s("profile:b", "profile", 500.0, 2.1)],
        _costs(), budget_hours=0.2,
    )
    assert [i.input_id for i in plan.items] == ["link_bw:a"]
    assert [(i.rank, i.input_id) for i in plan.uncovered] == [(2, "profile:b")]


def test_exclusive_hours_are_counted_separately() -> None:
    plan = build(
        [_s("link_bw:a", "link_bw", 10.0, 0.114),   # exclusive
         _s("power:c", "power", 10.0, 0.172)],      # not
        _costs(),
    )
    assert plan.total_hours == pytest.approx(0.286)
    assert plan.exclusive_hours == pytest.approx(0.114)


# --- what is and is not a candidate ---------------------------------------

def test_only_positive_regret_is_worth_a_server_hour() -> None:
    plan = build(
        [_s("link_bw:moves", "link_bw", 10.0, 0.114),
         _s("link_bw:flat", "link_bw", 0.0, 0.114)],
        _costs(),
    )
    assert [i.input_id for i in plan.items] == ["link_bw:moves"]


def test_an_unbounded_input_is_listed_apart_not_ranked_and_not_dropped() -> None:
    """No range means no regret to compare - a stronger statement than a rank."""
    plan = build(
        [_s("link_bw:known", "link_bw", 10.0, 0.114),
         _s("link_bw:unbounded", "link_bw", None, 0.114)],
        _costs(),
    )
    assert [i.input_id for i in plan.items] == ["link_bw:known"]
    assert plan.undecidable == ["link_bw:unbounded"]
    assert all(i.input_id != "link_bw:unbounded" for i in plan.uncovered)


def test_an_item_with_no_cost_is_planned_but_consumes_no_budget() -> None:
    """§2.6: an unpriced kind sorts last rather than being dropped.

    Charging it a made-up number would be exactly the invention the work order
    forbids, so it is planned and the budget arithmetic ignores it.
    """
    plan = build(
        [_s("link_lat:x", "link_lat", 999.0, None),
         _s("link_bw:a", "link_bw", 10.0, 0.114)],
        _costs(), budget_hours=0.2,
    )
    assert [i.input_id for i in plan.items] == ["link_bw:a", "link_lat:x"]
    assert plan.items[1].cost_hours is None
    assert plan.items[1].regret_per_hour is None
    assert plan.total_hours == pytest.approx(0.114)


def test_the_method_comes_from_costs_yaml(tmp_path) -> None:
    plan = build([_s("profile:b", "profile", 1.0, 2.1)], _costs())
    assert plan.items[0].how_to_measure == "python -m profiler profile"
    assert plan.items[0].exclusive is True

    bare = build([_s("profile:b", "profile", 1.0, 2.1)], None)
    assert "no method recorded" in bare.items[0].how_to_measure


def test_the_shipped_costs_table_supplies_a_method_for_every_kind() -> None:
    from planner.uncertainty.grades import load_costs
    from planner.uncertainty.registry import UncertainKind

    costs = load_costs()
    for kind in UncertainKind:
        assert costs.cost_for(kind.value) is not None, kind


# --- rendering ------------------------------------------------------------

def test_the_render_states_the_regret_the_budget_removes() -> None:
    from planner.render import render_measurement_plan

    text = render_measurement_plan(build(
        [_s("link_bw:a", "link_bw", 100.0, 0.114, flip=True),
         _s("profile:b", "profile", 500.0, 2.1, approx=True),
         _s("link_bw:u", "link_bw", None, 0.114)],
        _costs(), budget_hours=0.5,
    ))
    assert "removes 100" in text
    assert "FLIPS" in text
    assert "CANNOT BE DECIDED BEFORE MEASURING" in text
    assert "link_bw:u" in text
    assert "did not fit the budget, starting at rank 2" in text


def test_the_render_says_so_when_there_is_nothing_to_measure() -> None:
    from planner.render import render_measurement_plan

    assert "nothing to measure" in render_measurement_plan(build([], _costs()))


# --- measure-apply --------------------------------------------------------

def test_measure_apply_writes_a_copy_and_leaves_the_original_alone(tmp_path) -> None:
    """Absolute rule A3, which is the whole reason this command exists."""
    from planner.__main__ import build_parser

    source = tmp_path / "cluster.yaml"
    source.write_text((ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml").read_text())
    before = source.read_text()
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({"feasible": True, "provenance": {}}))

    args = build_parser().parse_args([
        "measure-apply", "--plan", str(plan_yaml),
        "--input", "link_bw:fabric-rngd0-a40a", "--value", "13.0",
        "--cluster", str(source), "--no-replan",
    ])
    assert args.func(args) == 0

    assert source.read_text() == before, "the original must not be touched"
    copy = source.with_suffix(".measured.yaml")
    assert copy.is_file()
    links = {link["id"]: link for link in yaml.safe_load(copy.read_text())["links"]}
    assert links["fabric-rngd0-a40a"]["bandwidth_gbps"] == 13.0
    assert links["fabric-rngd0-a40a"]["source"] == "measured"


def test_the_applied_link_is_no_longer_uncertain_in_the_registry(tmp_path) -> None:
    """The point of applying a measurement: it leaves the registry."""
    from planner.__main__ import build_parser
    from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
    from planner.predictor.calibration import CalibrationModel
    from planner.spec import load_service_spec
    from planner.uncertainty import build_registry, load_costs, load_grades

    source = tmp_path / "cluster.yaml"
    source.write_text(
        (ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml").read_text()
    )
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({"feasible": True, "provenance": {}}))
    # A placeholder on-package link: uncertain before, measured after.
    target = "link_bw:onpkg-rngd0-01"

    args = build_parser().parse_args([
        "measure-apply", "--plan", str(plan_yaml), "--input", target,
        "--value", "250.0", "--cluster", str(source), "--no-replan",
    ])
    assert args.func(args) == 0

    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    grades, costs = load_grades(), load_costs()

    def registry_for(path: Path):
        cluster = load_cluster_spec(path)
        profiles = load_profiles_for(cluster, ROOT)
        islands = detect_islands(cluster, profiles)
        return build_registry(
            cluster, profiles, islands, CalibrationModel.identity(), grades, spec, costs
        )

    before = registry_for(source)
    after = registry_for(source.with_suffix(".measured.yaml"))
    assert any(i.id == target for i in before.items)
    assert all(i.id != target for i in after.items), "measured inputs leave the registry"
    assert after.measured_count["link_bw"] == before.measured_count["link_bw"] + 1


def test_measure_apply_refuses_a_sim_error_point_with_no_operating_point(tmp_path) -> None:
    """An error without the concurrency it was measured at cannot be placed."""
    from planner.__main__ import build_parser

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({"feasible": True, "provenance": {}}))
    args = build_parser().parse_args([
        "measure-apply", "--plan", str(plan_yaml),
        "--input", "sim_error:RNGD-CARD/x", "--value", "0.1",
        "--calibration", str(ROOT / "profiles/calibration/a40.yaml"), "--no-replan",
    ])
    assert args.func(args) == 1


def test_measure_apply_adds_an_accuracy_point_to_a_calibration_copy(tmp_path) -> None:
    """A sim_error measurement is an `AccuracyPoint` on the hardware's domain
    (D33: the curve is `calibration.AccuracyDomain`), created under `refuse` when
    the calibration carried none."""
    from planner.__main__ import build_parser
    from planner.predictor.calibration import load_calibration

    source = tmp_path / "a40.yaml"
    source.write_text((ROOT / "profiles/calibration/a40.yaml").read_text())
    before = source.read_text()
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({"feasible": True, "provenance": {}}))

    args = build_parser().parse_args([
        "measure-apply", "--plan", str(plan_yaml),
        "--input", "sim_error:A40/x", "--value", "-7.0", "--concurrency", "55.0",
        "--calibration", str(source), "--evidence", "tests", "--no-replan",
    ])
    assert args.func(args) == 0
    assert source.read_text() == before

    domain = load_calibration(source.with_suffix(".measured.yaml")).hardware["A40"].accuracy_domain
    assert domain is not None
    assert domain.outside_domain == "refuse"
    assert [p.conc for p in domain.points] == [55.0]
    assert domain.points[0].tpot_err_pct == pytest.approx(-7.0)
    assert domain.points[0].note == "tests"
    assert domain.tpot_margin_pct(55.0) == pytest.approx(7.5269, abs=1e-4)


def test_measure_apply_refuses_a_second_point_at_the_same_concurrency(tmp_path) -> None:
    from planner.__main__ import build_parser

    source = tmp_path / "rngd_card_edf.yaml"
    source.write_text((ROOT / "profiles/calibration/rngd_card_edf.yaml").read_text())
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({"feasible": True, "provenance": {}}))
    args = build_parser().parse_args([
        "measure-apply", "--plan", str(plan_yaml),
        "--input", "sim_error:RNGD-CARD/x", "--value", "-1.0", "--concurrency", "76.0",
        "--calibration", str(source), "--no-replan",
    ])
    assert args.func(args) == 1
