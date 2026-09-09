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
    d = load_calibration(RNGD).hardware["RNGD-CARD"].accuracy_domain
    assert d is not None
    assert d.conc_min == pytest.approx(1.09)
    assert d.conc_max == pytest.approx(76.0)
    assert d.fitted_at_concurrency == pytest.approx(16.6)


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
