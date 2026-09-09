"""Outside the measured calibration domain, refuse — do not fall back (STEP 2.3).

`WORK_ORDER_rps_aware.md` rev 2. `slab3d` shortens the allreduce's latency term,
which makes decode look 24.4 % (tp8) / 13.4 % (tp4) faster than it is unless dim 1
carries the measured correction. The table covers two splits at three bandwidths
(`docs/slab3d_calibration.md`); everything else is unmeasured.

The temptation is a factor of 1.0 outside the table, because it "changes nothing".
It changes everything: 1.0 is the uncorrected case, so the fallback would
reinstate the full 13-24 % optimism in exactly the region nobody has checked, and
report it as a normal result. So `factor_for` returns None and the compiler
refuses, into its own rejection bucket -- an epistemic category, distinct from
both a feasibility verdict and a crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.calibration import (
    CalibrationError,
    Slab3dCalibration,
    load_slab3d_calibration,
    slab3d_calibration,
)
from planner.plan import RejectionStage
from planner.predictor import SimOutcome

ROOT = Path(__file__).resolve().parents[1]


def test_the_measured_points_resolve():
    c = slab3d_calibration()
    assert c.factor_for(2, 16.0) == 1.0
    assert c.factor_for(8, 7.7) == 4.0
    assert c.factor_for(8, 12.6) == 4.0
    assert c.factor_for(8, 16.0) == 4.0
    assert c.factor_for(8, 35.2) == 4.0
    assert c.factor_for(8, 100.0) == 4.0
    assert c.factor_for(4, 16.0) == 2.0
    assert c.factor_for(4, 35.2) == 2.0
    assert c.factor_for(4, 100.0) == 2.0


@pytest.mark.parametrize("tp,bw", [
    (8, 50.0),      # measured split, unmeasured bandwidth
    (8, 35.1),      # near-miss bandwidth: near is not measured
    (16, 35.2),     # unmeasured split, even though tp/2 would predict 8
    (32, 16.0),
])
def test_unmeasured_points_return_none_not_one(tp, bw):
    """None means refuse. 1.0 would mean "uncorrected", silently."""
    assert slab3d_calibration().factor_for(tp, bw) is None


def test_the_split_label_matches_the_table_key():
    assert Slab3dCalibration.split_label(8) == "[4,2]"
    assert Slab3dCalibration.split_label(4) == "[2,2]"


def test_the_refusal_says_what_to_do_about_it():
    note = slab3d_calibration().domain_note(16, 35.2)
    assert "[8,2]" in note
    assert "outside the measured calibration domain" in note
    assert "slab3d_calibrate.py" in note, "a refusal should name its remedy"
    assert "13-24" in note, "and say what accepting the guess would cost"


def test_a_table_without_validity_is_an_error(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text("points:\n  - {split: '[4,2]', link_bw: 16.0, factor: 4}\n")
    with pytest.raises(CalibrationError, match="validity"):
        load_slab3d_calibration(bad)


def test_a_missing_table_is_an_error_not_an_empty_domain(tmp_path):
    """An absent table must not read as "nothing is calibrated, carry on"."""
    with pytest.raises(CalibrationError, match="no slab3d calibration table"):
        load_slab3d_calibration(tmp_path / "absent.yaml")


def test_a_widen_error_bars_policy_is_refused(tmp_path):
    """A6 allows the policy name; nothing implements it, and silently behaving as
    if it did would be the failure mode the rule exists to prevent."""
    bad = tmp_path / "t.yaml"
    bad.write_text(
        "points:\n  - {split: '[4,2]', link_bw: 16.0, factor: 4}\n"
        "validity:\n  splits: ['[4,2]']\n  link_bw_gbps: [16.0]\n"
        "  extrapolation: widen_error_bars\n"
    )
    c = load_slab3d_calibration(bad)
    with pytest.raises(CalibrationError, match="only 'refuse' is implemented"):
        c.factor_for(8, 16.0)


def test_the_refusal_has_its_own_bucket_end_to_end():
    """It must be neither a feasibility verdict nor a crash."""
    assert SimOutcome.OUTSIDE_CALIBRATION_DOMAIN.is_error
    assert SimOutcome.OUTSIDE_CALIBRATION_DOMAIN is not SimOutcome.CRASHED
    assert (RejectionStage.OUTSIDE_CALIBRATION_DOMAIN
            is not RejectionStage.SIM_ERROR)
    assert (RejectionStage.OUTSIDE_CALIBRATION_DOMAIN
            is not RejectionStage.SLO_VIOLATED)
    assert RejectionStage.OUTSIDE_CALIBRATION_DOMAIN.value == "outside_calibration_domain"
