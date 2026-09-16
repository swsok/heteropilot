"""The §5.4 two-stage procedure, on synthetic points whose answer is known.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V1. The real run
(`experiments/p2_evidence/v1_region.py` against the committed domain) registers
almost nothing, so it cannot tell a working implementation from one that rejects
everything. These five synthetic points can: each case isolates one condition and
says which way it must come out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/p2_evidence/v1_region.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("v1_region_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pt(mod, conc, e_pct, *, n=300, prediction=40.0, measured=None, mode=None):
    """One synthetic point. `measured` defaults to exactly what `e_pct` implies."""
    if measured is None:
        measured = prediction / (1.0 + e_pct / 100.0)
    return mod.Pt(
        conc=conc, e_pct=e_pct, r_pct=mod.r_from_e(e_pct),
        margin_pct=mod.margin_of(e_pct), n=n, prediction_ms=prediction,
        measured_ms=measured, mode=mode or ("saturated" if conc >= 44 else
                                            "unsaturated"),
        basis="synthetic",
    )


@pytest.fixture
def five(mod):
    """Five points a doubling apart, all optimistic, all well sampled.

    Spacing is x1.5 so every adjacent pair clears the ratio condition, which
    isolates the other conditions in the cases below.
    """
    return [_pt(mod, c, -5.0) for c in (10.0, 15.0, 22.5, 33.75, 40.0)]


def test_a_ratio_violation_leaves_the_interval_unregistered(mod):
    """Condition 1, isolated: only the spacing differs between the two cases."""
    a, near, far = _pt(mod, 10.0, -5.0), _pt(mod, 19.0, -5.0), _pt(mod, 31.0, -5.0)

    ok = mod._stage1(a, near)["checks"]           # x1.9
    assert ok["concurrency_ratio"]["pass"]

    bad = mod._stage1(a, far)["checks"]           # x3.1
    assert not bad["concurrency_ratio"]["pass"]
    # and nothing else is what rejected it
    assert bad["sample_count"]["pass"]
    assert bad["same_operating_mode"]["pass"]

    # End to end: the interval must not register either.
    region = mod.build_region("C", [a, _pt(mod, 20.0, -5.0), far])
    assert region["validation_region"] == []


def test_b_an_uncovered_validation_point_leaves_the_interval_unregistered(mod):
    """Condition 2. The margin is sized on the ends and the middle is worse.

    A validation point whose measurement exceeds `prediction x (1 + m)` must
    block registration -- that is the whole point of the second stage.
    """
    a, b = _pt(mod, 10.0, -5.0), _pt(mod, 15.0, -5.0)
    margin = mod._interval_margin(a, b, 0.0)
    # Interior point measured far worse than the end-sized margin covers.
    bad = _pt(mod, 12.0, -5.0, prediction=40.0, measured=40.0 * (1 + margin / 100) + 1)
    assert not mod._stage2(a, b, [bad], margin)["pass"]

    good = _pt(mod, 12.0, -5.0, prediction=40.0,
               measured=40.0 * (1 + margin / 100) - 1)
    assert mod._stage2(a, b, [good], margin)["pass"]


def test_c_all_calibration_registers_nothing(mod, five):
    """Placement (A). With no validation point, stage 2 has nothing to test.

    The work order asks for this to be confirmed rather than assumed: a
    procedure that registered an interval here would be registering on the
    strength of the points that defined the margin, which proves nothing.
    """
    region = mod.build_region("A", five)
    assert region["validation_points"] == []
    assert region["validation_region"] == []
    assert all(iv["stage2"]["vacuous"] for iv in region["intervals"])
    assert not any(iv["registered"] for iv in region["intervals"])


def test_d_the_region_is_exactly_the_union_of_registered_intervals(mod, five):
    """The region must be derived from the intervals, not tracked separately."""
    region = mod.build_region("C", five)
    registered = [[iv["lo"], iv["hi"]] for iv in region["intervals"]
                  if iv["registered"]]
    assert region["validation_region"] == registered
    assert region["covered"] == pytest.approx(
        sum(hi - lo for lo, hi in registered))


def test_e_a_mode_change_blocks_an_otherwise_admissible_interval(mod):
    """Condition 3, which is what separates the c25.181-to-c76 interval."""
    a = _pt(mod, 30.0, -5.0)
    b = _pt(mod, 45.0, -5.0)  # above SATURATION_CONC
    checks = mod._stage1(a, b)["checks"]
    assert checks["concurrency_ratio"]["pass"]
    assert not checks["same_operating_mode"]["pass"]


def test_f_a_pessimistic_predictor_earns_no_margin(mod):
    """One-sidedness, which the disclosure and `margin_from_error` both require."""
    assert mod.margin_of(+5.0) == 0.0
    assert mod.margin_of(-5.0) > 0.0
    # And the conversion is the D70 one, not `-e`.
    assert mod.r_from_e(-18.0) == pytest.approx(21.9512, abs=1e-3)


def test_g_a_thin_endpoint_blocks_the_interval(mod):
    """Condition 2 of stage 1 -- the five-sample bucket fit is why it exists."""
    a, b = _pt(mod, 10.0, -5.0, n=5), _pt(mod, 15.0, -5.0)
    checks = mod._stage1(a, b)["checks"]
    assert not checks["sample_count"]["pass"]
    assert checks["concurrency_ratio"]["pass"]
