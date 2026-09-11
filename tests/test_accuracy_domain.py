"""The margin comes from the run's own operating point (STEP 4.2 / 4.3).

`docs/rps_aware_planning_design.md` §5. A hand-set `--tpot-margin-percent` is one
number for every candidate, and the simulator's error is not one number: it is
+11.6 % at served concurrency 3.9 and -18 % at 76 on the same device. So a manual
margin is either too loose at the top or too tight at the bottom, and D22's winner
is what "too loose at the top" looks like -- it passed a 50 ms SLO at a predicted
48.41 ms, and 48.41 x 1.18 = 57.1.

Three properties are asserted, each of which could plausibly have been built the
other way:

  * the margin is ONE-SIDED. Where the simulator is pessimistic the prediction is
    left alone rather than deflated, because deflating would make plans look
    better than the hardware measured -- the direction of the retraction.
  * hardware with no domain gets margin 0 and a note, never a margin borrowed
    from a different device (rule 3).
  * outside the measured points the margin GROWS with distance and never caps,
    so a candidate far outside is rejected by its own uncertainty.
"""

from __future__ import annotations

import pytest

from planner.predictor.calibration import AccuracyDomain, load_calibration

RNGD = "profiles/calibration/rngd_card_edf.yaml"
A40 = "profiles/calibration/a40.accuracy.yaml"


def _domain(**over) -> AccuracyDomain:
    d = {
        "fitted_at_concurrency": 10.0,
        "points": [
            {"conc": 10.0, "tpot_err_pct": -5.0},
            {"conc": 20.0, "tpot_err_pct": -15.0},
        ],
    }
    d.update(over)
    return AccuracyDomain.model_validate(d)


def test_the_margin_is_one_sided():
    """A pessimistic simulator is not 'corrected' downwards."""
    assert AccuracyDomain.margin_from_error(-18.0) == 18.0
    assert AccuracyDomain.margin_from_error(+11.6) == 0.0
    assert AccuracyDomain.margin_from_error(0.0) == 0.0
    assert AccuracyDomain.margin_from_error(None) == 0.0


def test_interpolates_between_measured_points():
    d = _domain()
    assert d.tpot_error_at(10.0) == pytest.approx(-5.0)
    assert d.tpot_error_at(15.0) == pytest.approx(-10.0)
    assert d.tpot_error_at(20.0) == pytest.approx(-15.0)


def test_outside_the_domain_the_margin_grows_and_does_not_cap():
    d = _domain()
    assert d.in_domain(15.0) and not d.in_domain(40.0)
    near, far = d.tpot_margin_pct(30.0), d.tpot_margin_pct(60.0)
    assert far > near > 15.0, (near, far)
    # |slope| is 1.0 %/unit, so 40 beyond the last point is 40 more percent
    assert d.tpot_margin_pct(60.0) == pytest.approx(55.0)


def test_a_single_point_domain_holds_flat_because_it_has_no_slope():
    d = _domain(points=[{"conc": 170.56, "tpot_err_pct": -1.42}])
    for c in (1.0, 170.56, 5000.0):
        assert d.tpot_margin_pct(c) == pytest.approx(1.42)
    assert d.in_domain(1.0) is False, "flat is not the same as in-domain"


def test_unsorted_points_are_rejected():
    with pytest.raises(Exception, match="sorted"):
        _domain(points=[{"conc": 20.0, "tpot_err_pct": -15.0},
                        {"conc": 10.0, "tpot_err_pct": -5.0}])


# --- the committed domains --------------------------------------------------

def test_the_rngd_domain_spans_the_measured_range():
    """Rebuilt at 300 requests on 2026-09-11 (deviations D32): nine points from
    1.020, not six from 1.09. The floor moved because the run it was measured on
    changed length, not because the hardware did."""
    d = load_calibration(RNGD).hardware["RNGD-CARD"].accuracy_domain
    assert d is not None
    assert d.conc_min == pytest.approx(1.020)
    assert d.conc_max == pytest.approx(76.0)
    assert d.fitted_at_concurrency == pytest.approx(16.6)
    assert len(d.points) == 9


def test_the_two_points_discarded_at_a_40pct_gap_are_back_and_changed_sign():
    """The D32 result, pinned.

    `rngd_card_edf.yaml` used to discard the c15.3 and c15.59 points because the
    simulator settled 40 % below the hardware, and attributed that to throughput
    error. At 300 requests -- same rates, same trace, only the run length
    different -- they land at -3.1 % and -2.4 %, so the gap was the drain tail.
    Their TPOT error also changes SIGN, from -1.22 / -1.97 % to +3.26 / +2.47 %,
    and only a negative error earns a margin: readmitting them REMOVES a margin
    around c15 rather than adding one.
    """
    d = load_calibration(RNGD).hardware["RNGD-CARD"].accuracy_domain
    assert d is not None
    concs = [p.conc for p in d.points]
    assert pytest.approx(14.832) in concs
    assert pytest.approx(15.212) in concs
    assert d.tpot_error_at(14.832) == pytest.approx(3.26, abs=0.01)
    assert d.tpot_error_at(15.212) == pytest.approx(2.47, abs=0.01)
    assert d.tpot_margin_pct(15.212) == 0.0


def test_the_two_points_above_29_stay_out():
    """Not everything the 300-request run touched is admissible. At the rates
    matching measured c59.2 and c107.2 the simulator settles at served 37.67 and
    44.47 even at 300 requests -- gaps of -36.4 % and -58.5 % -- so its TPOT
    belongs to a different operating point. That residual is the card model's
    real throughput ceiling, and it is what the 20-request runs could not
    separate from the drain tail."""
    d = load_calibration(RNGD).hardware["RNGD-CARD"].accuracy_domain
    assert d is not None
    concs = [p.conc for p in d.points]
    assert not any(30.0 < c < 76.0 for c in concs), (
        "only 25.181 may sit between the 16.6 and 76.0 anchors")


def test_the_simulator_is_pessimistic_at_low_load_and_optimistic_at_high():
    """Measured, and it is why extrapolating the 16.6 point downwards was wrong in
    SIGN as well as size (outputs/lowload_sim_error/)."""
    d = load_calibration(RNGD).hardware["RNGD-CARD"].accuracy_domain
    assert d is not None
    assert d.tpot_error_at(2.0) > 0, "pessimistic below ~7"
    assert d.tpot_error_at(76.0) < 0, "optimistic at the top"
    # so the one-sided margin is zero where the simulator already over-predicts
    assert d.tpot_margin_pct(2.0) == 0.0
    assert d.tpot_margin_pct(76.0) == pytest.approx(18.0)


def test_the_d22_winner_would_now_be_caught():
    """The concrete case §5 names: 48.41 ms predicted against a 50 ms SLO."""
    d = load_calibration(RNGD).hardware["RNGD-CARD"].accuracy_domain
    assert d is not None
    margin = d.tpot_margin_pct(76.0)
    robust = 48.41 * (1 + margin / 100.0)
    assert robust > 50.0, f"robust TPOT {robust:.2f} ms should breach the 50 ms SLO"
    assert robust == pytest.approx(57.1, abs=0.2)


def test_the_a40_domain_is_a_separate_file_from_the_default_calibration():
    """Keeping it out of a40.yaml is what makes the default path unchanged."""
    default = load_calibration("profiles/calibration/a40.yaml")
    assert default.hardware["A40"].accuracy_domain is None, (
        "a domain in a40.yaml would apply automatically and move the frozen output"
    )
    optin = load_calibration(A40).hardware["A40"].accuracy_domain
    assert optin is not None
    assert optin.fitted_at_concurrency == pytest.approx(170.56)


# --- 4.6: the energy definition must not move -------------------------------

def test_the_planner_does_not_compute_energy_from_the_power_model():
    """D29b. `power_model` is parsed and deliberately unused by the planner.

    Every tok/J in CLAIMS.md, PROJECT_REPORT.md and the sweeps was computed from
    the simulator's NODE-level power. Switching to a card-level curve would change
    what the metric means, and the whole historical comparison with it, for a
    reason that is not a change in the hardware or the plan. Asserted on the
    source so the coupling cannot appear by accident.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "planner"
    users = []
    for f in root.rglob("*.py"):
        text = f.read_text()
        code = "\n".join(
            ln for ln in text.splitlines()
            if "power_model" in ln and not ln.lstrip().startswith(("#", "*", '"'))
        )
        if "power_model" in code and f.name != "inventory.py":
            users.append(f.relative_to(root).as_posix())
    assert users == [], (
        f"only inventory.py may mention power_model in code; {users} would couple "
        f"the planner's energy definition to a card-level curve (D29b)"
    )


# --- D33: refuse is the default, and the domain says so -------------------

def test_refuse_is_the_default_and_the_committed_domains_opt_into_widening():
    """Deviations D33. A new domain refuses outside its points; the E5/E6 domains
    say `widen_error_bars` explicitly and keep extrapolating."""
    assert _domain().outside_domain == "refuse"
    for path, hw in ((RNGD, "RNGD-CARD"), (A40, "A40"),
                     ("profiles/calibration/rngd_perpe.yaml", "RNGD")):
        d = load_calibration(path).hardware[hw].accuracy_domain
        assert d is not None and d.outside_domain == "widen_error_bars"


def test_errors_at_is_none_outside_a_refusing_domain_and_a_pair_inside():
    d = _domain()
    assert d.errors_at(25.0) is None
    assert d.errors_at(5.0) is None
    inside = d.errors_at(15.0)
    assert inside is not None
    ttft, tpot = inside
    assert ttft is None, "no TTFT point anywhere in this domain"
    assert tpot == pytest.approx(-10.0)
    assert d.has_metric("tpot") and not d.has_metric("ttft")


def test_errors_at_widens_when_told_to():
    d = _domain(outside_domain="widen_error_bars")
    got = d.errors_at(25.0)
    assert got is not None and got[1] == pytest.approx(d.tpot_error_at(25.0))
    assert got[1] < -15.0


def test_basis_at_names_the_bracket_or_the_refusal():
    d = _domain()
    assert "interpolated between L=10 and L=20" in d.basis_at(15.0)
    assert "ttft: NO point" in d.basis_at(15.0)
    assert "refuse" in d.basis_at(25.0)
    assert "EXTRAPOLATED" in _domain(outside_domain="widen_error_bars").basis_at(25.0)


def test_a_domain_may_be_scoped_to_a_canonical_shape_only():
    assert _domain(workload_shape="in_lt1024-out_ge512").workload_shape == "in_lt1024-out_ge512"
    with pytest.raises(ValueError, match="canonical shape"):
        _domain(workload_shape="sharegpt")


def test_two_files_with_a_domain_for_one_hardware_are_refused(tmp_path):
    """The loader never picks silently between two measurements of one device."""
    from pathlib import Path

    from planner.predictor.calibration import load_accuracy_domains

    src = Path(RNGD).read_text()
    (tmp_path / "a.yaml").write_text(src)
    (tmp_path / "b.yaml").write_text(src)
    with pytest.raises(ValueError, match="two accuracy domains for RNGD-CARD"):
        load_accuracy_domains(paths=[tmp_path / "a.yaml", tmp_path / "b.yaml"])
    only = load_accuracy_domains(paths=[tmp_path / "a.yaml"])
    assert set(only) == {"RNGD-CARD"}


def test_the_committed_tree_loads_without_a_duplicate():
    from planner.predictor.calibration import load_accuracy_domains, load_calibrations

    domains = load_accuracy_domains(".")
    assert {"A40", "RNGD-CARD", "RNGD"} <= set(domains)
    merged = load_calibrations(".")
    assert {"A40", "RNGD-CARD", "RNGD"} <= set(merged.hardware)


# --- served concurrency (uncertainty work order A2) -----------------------

def test_served_concurrency_from_sim_is_total_time_over_the_window():
    """Four requests of 10 s each inside a 20 s window average 2 in flight."""
    from planner.predictor.accuracy_domain import served_concurrency_from_sim

    assert served_concurrency_from_sim([10.0, 10.0, 10.0, 10.0], 20.0) == pytest.approx(2.0)
    assert served_concurrency_from_sim([1.0], 0.0) == 0.0


def test_served_concurrency_little():
    """rps 2, W = 100 ms + 500 x 20 ms = 10.1 s, so L = 20.2."""
    from planner.predictor.accuracy_domain import served_concurrency_little

    assert served_concurrency_little(
        rps=2, ttft_ms=100, tpot_ms=20, mean_out_tokens=500
    ) == pytest.approx(20.2)


def test_served_concurrency_reproduces_the_committed_d22_envelope():
    """The one check that this is the same arithmetic D22 published.

    docs/deviations.md D22's table gives eff. conc 15.3 / 29.3 / 59.2 / 107.2
    for real_c{16,32,64,128}.json. Reproducing it from the raw per-request
    records is what says `served_concurrency_from_sim` is the D22 statistic and
    not the requested concurrency, which for these same runs reads 16 / 32 /
    64 / 128.
    """
    import json
    from pathlib import Path

    from planner.predictor.accuracy_domain import served_concurrency_from_sim

    root = Path(__file__).resolve().parents[1]
    expected = {16: 15.3, 32: 29.3, 64: 59.2, 128: 107.2}
    for requested, published in expected.items():
        path = root / f"outputs/rngd_envelope/edf/real_c{requested}.json"
        report = json.loads(path.read_text())
        latencies = [
            row["latency_ns"] / 1e9
            for row in report["per_request"]
            if not row.get("error") and row.get("latency_ns")
        ]
        served = served_concurrency_from_sim(latencies, report["wall_s"])
        assert served == pytest.approx(published, abs=0.05)
        assert report["concurrency"] == requested
        assert abs(served - requested) > 0.5, "requested and served must not be confused"


# --- golden: served_concurrency must not leak into default output ---------

def _plan_with_metrics(served: float | None):
    from planner.plan import (
        CandidateConfig,
        DeploymentPlan,
        IslandAssignment,
        PlannerOutput,
        PredictedMetrics,
        ScoredPlan,
    )
    from planner.spec import Objective

    metrics = PredictedMetrics(
        p50_ttft_ms=1.0, p95_ttft_ms=2.0, p99_ttft_ms=3.0,
        p50_tpot_ms=1.0, p95_tpot_ms=2.0, p99_tpot_ms=3.0,
        throughput_tps=10.0, slo_goodput_rps=1.0, slo_attainment=1.0,
        completed_requests=1, completed_tokens=1,
        served_concurrency=served,
    )
    plan = DeploymentPlan(
        plan_id="hp-1", model="m",
        candidate=CandidateConfig(
            id="c1", model="m", dtype="bfloat16",
            assignments=[IslandAssignment(island_id="i", tp_size=1)],
        ),
        predicted=metrics,
    )
    return PlannerOutput(
        feasible=True, service_model="m", cluster_id="c",
        recommended=ScoredPlan(plan=plan, objective=Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE,
                               value=1.0),
        alternatives=[ScoredPlan(plan=plan, objective=Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE,
                                 value=0.5)],
    )


def test_served_concurrency_is_absent_from_the_default_yaml(tmp_path):
    """Rule A4. The predictor fills this on every real run, so unlike
    `uncertain_inputs` it would appear WITH A VALUE, not as a null, in output
    that must not change."""
    import yaml as _yaml

    from planner.__main__ import _write_output

    path = tmp_path / "plan.yaml"
    _write_output(_plan_with_metrics(42.5), path)
    text = path.read_text()
    assert "served_concurrency" not in text
    loaded = _yaml.safe_load(text)
    assert "served_concurrency" not in loaded["recommended"]["plan"]["predicted"]
    assert "served_concurrency" not in loaded["alternatives"][0]["plan"]["predicted"]


def test_served_concurrency_is_kept_once_the_registry_is_present(tmp_path):
    import yaml as _yaml

    from planner.__main__ import _write_output
    from planner.uncertainty import UncertainInputRegistry

    output = _plan_with_metrics(42.5)
    output.uncertain_inputs = UncertainInputRegistry()
    path = tmp_path / "plan.yaml"
    _write_output(output, path)
    loaded = _yaml.safe_load(path.read_text())
    assert loaded["recommended"]["plan"]["predicted"]["served_concurrency"] == 42.5
