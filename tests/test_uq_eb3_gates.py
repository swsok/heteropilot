"""E-B3's two identity gates and §2.5's refusal to compute a rank correlation.

`WORK_ORDER_uq_stage_b_plus.md` STEP C4. PR #86 committed two Spearman values of
1.000 that were evidence of nothing, and an identity control whose deviation was
reported but never gated on. Both are now rules rather than prose, and these are
the rules.

None of this needs a simulator: the gates read refinement records and control
rows, which is exactly why they were lifted out of `main`.
"""

from __future__ import annotations

from pathlib import Path

from experiments.uncertainty.eb1_regret_vs_budget import World
from experiments.uncertainty.eb3_closed_form_vs_resim import (
    MIN_ACTIVE_FOR_SPEARMAN,
    active_records,
    identity_gate,
    mirror_members,
    restrict,
    spearman_refusal,
)
from planner.uncertainty.perturb import JudgedMetrics, PerturbContext


class _Rec:
    """A `Refinement` in the two fields the gates read."""

    def __init__(self, input_id, closed, resim, skipped=""):
        self.input_id = input_id
        self.closed_form = closed
        self.resimulated = resim
        self.skipped = skipped


# ---------------------------------------------------------------------------
# §2.5: a correlation over ties measures the ties
# ---------------------------------------------------------------------------

def test_two_active_items_is_not_enough_for_a_rank_correlation() -> None:
    """F2's shape: `sim_error` carries the most regret and cannot be
    resimulated, leaving the two A40 profiles. Two points always correlate
    perfectly or perfectly inversely, so the number would carry no information
    about the closed form."""
    note = spearman_refusal(2, 4)
    assert note is not None
    assert "2 item(s)" in note and "4 checked" in note


def test_three_active_items_clears_the_gate() -> None:
    assert spearman_refusal(MIN_ACTIVE_FOR_SPEARMAN, 10) is None


def test_the_refusal_says_how_many_rather_than_just_refusing() -> None:
    """A null with no reason reads as "the correlation came out zero". The
    payload carries this string next to the null so it cannot."""
    assert "not computed" in (spearman_refusal(0, 0) or "")


def test_active_is_read_from_both_sides_not_just_the_closed_form() -> None:
    """An item the closed form calls inert and the simulator does not is the
    most interesting row E-B3 can produce. Counting `active` off the closed form
    alone would drop it, and with it the only evidence that the rule missed
    something."""
    records = [
        _Rec("a", 0.0, 0.0),
        _Rec("b", 0.0, 12_000.0),        # closed form says inert, resim says not
        _Rec("c", 9_000.0, 0.0),         # and the other way round
    ]
    assert [r.input_id for r in active_records(records)] == ["b", "c"]


def test_a_skipped_item_is_not_active_however_large_its_closed_form() -> None:
    """`sim_error` is skipped and carries F2's largest dR. Counting it would
    clear the gate on the strength of the one item that was never checked."""
    records = [_Rec("sim_error:domain", 41_118.0, 41_118.0, skipped="no rule")]
    assert active_records(records) == []


# ---------------------------------------------------------------------------
# The x1.0 identity control, as a gate
# ---------------------------------------------------------------------------

def _ctl(*deviants: str) -> dict:
    return {"deviating_candidates": list(deviants)}


def test_a_clean_control_passes() -> None:
    assert identity_gate([_ctl(), _ctl()], set())["passed"] is True


def test_deviations_that_are_all_known_mirrors_pass() -> None:
    """D40 is a recorded defect, not a new one: one cache entry serving two
    mirrored placements. A mirror still in the corpus is expected to deviate."""
    verdict = identity_gate([_ctl("mix(a+b)"), _ctl("mix(a+b)")], {"mix(a+b)"})
    assert verdict["passed"] is True
    assert verdict["deviating_candidates"] == ["mix(a+b)"]
    assert verdict["unexplained"] == []


def test_one_deviation_outside_the_mirror_list_fails_the_gate() -> None:
    """The case §2.5 says to stop on. A candidate reading a prediction that is
    not its own, for a reason D40 does not explain, contaminates every magnitude
    the run reports -- so it must not be discovered in the write-up."""
    verdict = identity_gate([_ctl("mix(a+b)", "pd(c+d)")], {"mix(a+b)"})
    assert verdict["passed"] is False
    assert verdict["unexplained"] == ["pd(c+d)"]


def test_the_gate_unions_across_controls_rather_than_taking_the_last() -> None:
    """Each refined item runs its own x1.0 control, and a candidate can deviate
    under one and not another. The E-A1 run's worst-deviation field showed one
    candidate per control; the list is what §2.5 compares."""
    verdict = identity_gate([_ctl("x"), _ctl("y"), _ctl()], set())
    assert verdict["unexplained"] == ["x", "y"]
    assert verdict["controls_run"] == 3


# ---------------------------------------------------------------------------
# --exclude: the mirror list, and restricting a world by it
# ---------------------------------------------------------------------------

def test_mirror_members_keeps_the_representative(tmp_path: Path) -> None:
    """The representative owns the cache entry the group shares, so it is the
    one placement whose prediction is its own. Dropping it too would remove the
    measurement along with its duplicates."""
    path = tmp_path / "mirror_pairs.json"
    path.write_text(
        '{"representative_to_members": {"a": ["a", "b"], "c": ["c", "d", "e"]}}'
    )
    assert mirror_members(path) == {"b", "d", "e"}


def _world(ids: list[str]) -> World:
    candidates = {i: object() for i in ids}
    context = PerturbContext(
        spec=None, cluster=None, islands={}, candidates=dict(candidates),
        topology=None, operating_points={i: {"served": 1.0} for i in ids},
    )
    return World(
        spec=None, cluster=None, islands={}, candidates=candidates,
        context=context, truth=JudgedMetrics.trusted({i: object() for i in ids}),
        island_hw={}, costs=None,
    )


def test_restrict_removes_from_corpus_truth_and_context_together() -> None:
    """All three or none. A candidate left in the context while gone from the
    corpus is still priced by `perturb`, and one left in the truth is still a
    rival for the incumbent -- either way the two sides of the comparison would
    be scored over different candidate sets."""
    world, removed = restrict(_world(["a", "b", "c"]), {"b"})
    assert removed == ["b"]
    assert set(world.candidates) == {"a", "c"}
    assert set(world.truth) == {"a", "c"}
    assert set(world.context.candidates) == {"a", "c"}
    assert set(world.context.operating_points) == {"a", "c"}


def test_restrict_is_a_no_op_when_nothing_to_drop() -> None:
    """F2's corpus already has C1's mirrors removed at build time, so passing
    --exclude on it changes nothing. The run must say so rather than appear to
    have done the exclusion itself."""
    world, removed = restrict(_world(["a", "b"]), {"zzz"})
    assert removed == []
    assert set(world.candidates) == {"a", "b"}


def test_restrict_keeps_the_truth_a_judged_metrics() -> None:
    """`perturb` refuses a plain dict, and the reason is a real failure: the
    envelope cache's raw output has no P/D transfer term. A filtered truth that
    came back as a dict would fail much later and somewhere else."""
    world, _ = restrict(_world(["a", "b"]), {"a"})
    assert isinstance(world.truth, JudgedMetrics)
