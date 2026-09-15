"""§2.4's three false-positive categories, one synthetic case each.

`WORK_ORDER_uq_stage_b_plus.md` STEP C3. E-B2 counts a predicted flip that did
not materialise as a false positive, and the committed E-A1 run has 17 of them.
Only one of the three reasons a flip can fail to materialise is the detector
being wrong, so collapsing them into one precision figure understates the rule
and misdirects the fix.

The classifier reads `Sensitivity.grid`, which already records the argmax and
both objective values at every swept point -- so these cases are built as grids,
which is exactly what the real path hands it.
"""

from __future__ import annotations

import pytest

from experiments.uncertainty.eb2_flip_detection import (
    TIE_TOL,
    classify_false_positive,
)
from planner.uncertainty.sensitivity import GridPoint


def _pt(value: float, best: str, best_value: float, current: float) -> GridPoint:
    return GridPoint(value=value, best_plan_id=best,
                     best_value=best_value, current_value=current)


# nominal 35 (degraded), truth 13: restoring walks the range downwards.
NOMINAL, TRUTH = 35.0, 13.0


def test_delta_is_the_sweep_and_the_restore_disagreeing_at_one_point() -> None:
    """The crossing sits AT the truth value, and restoring to that same value
    changed nothing. Two computations of one point cannot disagree unless they
    are computing different functions -- so this is not a statement about where
    the crossing is. On E-A1 every false positive is this: the SIM_ERROR item
    declares its range as simulator error with truth at 0, while the harness
    restores it by putting the measured DOMAIN back, which is not zero error."""
    grid = [
        _pt(13.0, "rival", -70_000.0, -80_000.0),   # 13 is the truth value
        _pt(24.0, "incumbent", -80_000.0, -80_000.0),
        _pt(35.0, "incumbent", -80_000.0, -80_000.0),
    ]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "delta"


def test_delta_outranks_beta_when_both_could_apply() -> None:
    """A crossing at truth and another strictly inside the interval. The one at
    truth is a contradiction; the other is only evidence of misplacement, and a
    contradiction is the stronger signal about which component is at fault."""
    grid = [
        _pt(13.0, "rival", -70_000.0, -80_000.0),
        _pt(24.0, "rival", -70_000.0, -80_000.0),
        _pt(35.0, "incumbent", -80_000.0, -80_000.0),
    ]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "delta"


def test_alpha_is_a_real_crossing_truth_never_reaches() -> None:
    """The sweep is right that the decision changes somewhere in the range, and
    the change sits at 5 GB/s -- below the truth at 13, so restoring to 13 never
    crosses it. The plan's claim ("measuring this COULD change the decision") is
    true; scoring it against one truth value asks a different question."""
    grid = [
        _pt(5.0, "rival", -70_000.0, -80_000.0),    # rival strictly better here
        _pt(13.0, "incumbent", -80_000.0, -80_000.0),
        _pt(24.0, "incumbent", -80_000.0, -80_000.0),
        _pt(35.0, "incumbent", -80_000.0, -80_000.0),
    ]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "alpha"


def test_beta_is_a_crossing_the_rule_put_between_nominal_and_truth() -> None:
    """Same shape, but the claimed crossing is at 24 -- inside [13, 35], the
    interval restoring actually walks. Restoring to truth produced no change
    anyway, so the rule placed the crossing wrongly. This is the only one of the
    three that counts against the closed form."""
    grid = [
        _pt(5.0, "incumbent", -80_000.0, -80_000.0),
        _pt(13.0, "incumbent", -80_000.0, -80_000.0),
        _pt(24.0, "rival", -70_000.0, -80_000.0),
        _pt(35.0, "incumbent", -80_000.0, -80_000.0),
    ]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "beta"


def test_gamma_is_an_id_change_that_wins_nothing() -> None:
    """The argmax id moves to a candidate with the same outcome. Nothing
    changed but the name, and `collapse_equivalent` dissolves it."""
    grid = [
        _pt(5.0, "twin", -80_000.0, -80_000.0),
        _pt(13.0, "twin", -80_000.0, -80_000.0),
        _pt(24.0, "incumbent", -80_000.0, -80_000.0),
        _pt(35.0, "incumbent", -80_000.0, -80_000.0),
    ]
    assert classify_false_positive(
        grid, "incumbent", NOMINAL, TRUTH, {"twin"}) == "gamma"


def test_gamma_also_covers_a_pure_tie_without_a_declared_equivalent() -> None:
    """A rival that takes the argmax while winning nothing is a tie-break, not a
    flip -- whether or not `equivalent_candidates` happened to list it."""
    grid = [_pt(5.0, "rival", -80_000.0, -80_000.0),
            _pt(35.0, "incumbent", -80_000.0, -80_000.0)]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "gamma"


def test_a_grid_that_never_changes_the_argmax_is_beta() -> None:
    """The sweep reported a flip its own grid does not show. Only the rule can
    be responsible for that, so it must not land in the category that excuses
    the rule."""
    grid = [_pt(v, "incumbent", -80_000.0, -80_000.0) for v in (5.0, 13.0, 35.0)]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "beta"


def test_an_unfeasible_grid_point_is_not_read_as_a_flip() -> None:
    """`best_plan_id` is "" where nothing was feasible. Treating that empty
    string as a rival candidate would turn every infeasible sweep point into a
    spurious crossing -- and on F2, 42 % of degradation sets have no feasible
    plan at all (STEP C2)."""
    grid = [_pt(5.0, "", float("-inf"), -80_000.0),
            _pt(35.0, "incumbent", -80_000.0, -80_000.0)]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "beta"


@pytest.mark.parametrize("delta", [TIE_TOL / 2, 0.0, -TIE_TOL])
def test_a_win_below_the_tie_tolerance_is_not_a_win(delta: float) -> None:
    """`collapse_equivalent` rounds an outcome to six decimals before comparing,
    so a margin finer than that is not a distinction the planner makes."""
    grid = [_pt(5.0, "rival", -80_000.0 + delta, -80_000.0),
            _pt(35.0, "incumbent", -80_000.0, -80_000.0)]
    assert classify_false_positive(grid, "incumbent", NOMINAL, TRUTH, set()) == "gamma"


def test_the_ordering_of_nominal_and_truth_does_not_matter() -> None:
    """Restoring can walk upwards as easily as downwards; the interval is the
    same either way. A classifier that assumed nominal > truth would silently
    mislabel every PROFILE item, whose degraded value is ABOVE its truth."""
    grid = [
        _pt(1.0, "incumbent", -80_000.0, -80_000.0),
        _pt(1.2, "rival", -70_000.0, -80_000.0),
        _pt(1.39, "incumbent", -80_000.0, -80_000.0),
    ]
    assert classify_false_positive(grid, "incumbent", 1.39, 1.0, set()) == "beta"
    assert classify_false_positive(grid, "incumbent", 1.0, 1.39, set()) == "beta"
