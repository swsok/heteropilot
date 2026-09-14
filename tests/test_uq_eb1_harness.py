"""The three things E-B1's harness gets wrong silently.

`WORK_ORDER_uq_stage_b_plus.md` STEP C2. Each of these has already happened once
and cost a run whose output looked ordinary:

* plain sampling let the six links crowd out every mixed degradation set, so the
  comparison between kinds that E-B1 exists to make was rarely drawn at all;
* a degraded state with nothing feasible produced dR None for every item, the
  ranking fell back to alphabetical order, and `ours` came last;
* the report path is shared with the committed E-A1 result, so a change made for
  F2 can move a number nobody re-reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.uncertainty import eb1_report
from experiments.uncertainty.eb1_regret_vs_budget import (
    Degraded,
    _best,
    require_judged_degraded,
    stratified_subsets,
)
from planner.spec import load_service_spec
from planner.uncertainty.registry import Grade, Range, UncertainInput, UncertainKind

EB1_RAW = Path("outputs/uncertainty/eb1/eb1_regret_vs_budget.json")
EB1_SUMMARY = Path("outputs/uncertainty/eb1/eb1_summary.json")


def _item(name: str, kind: UncertainKind) -> Degraded:
    return Degraded(
        item=UncertainInput(
            id=name, kind=kind, grade=Grade.PLACEHOLDER, nominal=1.0,
            range=Range(lo=0.0, hi=2.0, unit="fraction", source="test"),
        ),
        truth_value=0.0, degraded_value=1.0,
    )


@pytest.fixture
def pool() -> list[Degraded]:
    """This cluster's shape: links outnumber everything else."""
    return (
        [_item(f"link_bw:l{i}", UncertainKind.LINK_BW) for i in range(6)]
        + [_item(f"profile:p{i}", UncertainKind.PROFILE) for i in range(4)]
        + [_item("sim_error:domain", UncertainKind.SIM_ERROR)]
    )


# -- stratified sampling --------------------------------------------------

def test_no_stratified_set_is_single_kind(pool: list[Degraded]) -> None:
    """The work order's check: k=2 over a three-kind pool, and not one draw
    comes back with both items from the same kind."""
    subsets = stratified_subsets(pool, 2)
    assert subsets, "a three-kind pool has mixed pairs"
    assert all(len({d.item.kind for d in s}) >= 2 for s in subsets)
    # 100 draws, as asked. Drawing from the filtered enumeration rather than
    # rejecting after the fact is what makes this hold for every draw and not
    # merely for most of them.
    import random
    for seed in range(100):
        chosen = random.Random(seed).choice(subsets)
        assert len({d.item.kind for d in chosen}) >= 2


def test_stratification_is_what_removes_the_single_kind_sets(pool) -> None:
    """Guard against a filter that silently passes everything: unfiltered k=2
    really does contain link-only pairs, and they really are the majority
    problem the work order describes."""
    import itertools
    everything = [list(c) for c in itertools.combinations(pool, 2)]
    single = [s for s in everything if len({d.item.kind for d in s}) == 1]
    assert len(single) == 15 + 6, "6 links -> C(6,2)=15 pairs, 4 profiles -> 6"
    assert len(stratified_subsets(pool, 2)) == len(everything) - len(single)


def test_k1_is_returned_whole(pool: list[Degraded]) -> None:
    """A single-item set cannot mix kinds; filtering it would empty the k=1
    sweep, which the work order wants enumerated in full."""
    assert len(stratified_subsets(pool, 1)) == len(pool)


# -- the zero-feasible gate -----------------------------------------------

def test_a_degraded_state_that_judged_nothing_is_fatal() -> None:
    """Not dR None, not a warning. An empty corpus stands in for the real
    accident -- passing no domains at all, which made every candidate
    `unmeasured` and left the search with nothing to recommend."""
    spec = load_service_spec(
        Path("experiments/uncertainty/fixtures/llama31-8b-ttft8000.yaml"))
    world = SimpleNamespace(
        truth={}, policy_truth=None, candidates={}, spec=spec,
        context=None, island_hw={},
    )
    assert _best(world, world.truth) is None
    with pytest.raises(SystemExit, match="judged nothing"):
        require_judged_degraded(world, [])


def test_a_degraded_state_with_no_feasible_plan_is_NOT_fatal(monkeypatch) -> None:
    """The discriminator that makes the gate usable. On F2, degrading one A40
    profile SLO-rejects all 223 candidates -- a real verdict, and the state
    where a measurement is worth the most. Halting there would refuse to measure
    the most informative degradation in the sweep."""
    import experiments.uncertainty.eb1_regret_vs_budget as eb1

    monkeypatch.setattr(eb1, "_state", lambda w, d, r: (None, None))
    monkeypatch.setattr(eb1, "verdict_counts",
                        lambda w, m, p: {"rejected": 223})
    assert eb1.require_judged_degraded(SimpleNamespace(), []) == {"rejected": 223}

    monkeypatch.setattr(eb1, "verdict_counts",
                        lambda w, m, p: {"unmeasured": 223})
    with pytest.raises(SystemExit, match="judged nothing"):
        eb1.require_judged_degraded(SimpleNamespace(), [])


# -- the committed E-A1 result --------------------------------------------

@pytest.mark.skipif(not EB1_RAW.exists() or not EB1_SUMMARY.exists(),
                    reason="the committed E-B1 artifacts are not present")
def test_the_ea1_ours_curve_still_summarizes_to_the_committed_numbers() -> None:
    """`--fixture` and `--stratified` are additions, not changes: re-summarizing
    the committed raw curves must reproduce the committed summary. This is the
    half of the regression a test can hold without the simulator -- the raw
    curves themselves are pinned by the artifact being committed."""
    payload = json.loads(EB1_RAW.read_text())
    committed = json.loads(EB1_SUMMARY.read_text())
    grid = committed["budget_grid"]

    for key, expected in committed["summaries"].items():
        penalty, scope = key.split(", ", 1)
        rows = [
            r for r in payload["runs"]
            if f"penalty={r['penalty']}" == penalty
            and (scope == "all sets" or r["initial_regret"] > 1e-9)
        ]
        got = eb1_report.summarize(rows, grid)
        assert got["ours"]["n"] == expected["ours"]["n"], key
        assert got["ours"]["mean"] == pytest.approx(expected["ours"]["mean"]), key
        assert got["ours"]["mean_zero_budget_h"] == pytest.approx(
            expected["ours"]["mean_zero_budget_h"]), key


@pytest.mark.skipif(not EB1_RAW.exists(), reason="artifact not present")
def test_the_committed_ea1_run_has_no_pd_exclusions() -> None:
    """The default fixture enumerates no P/D candidate, so neither D40 nor D71
    can apply to it. If a future change starts excluding candidates from the
    E-A1 corpus, the committed result stops being the thing it claims to be."""
    payload = json.loads(EB1_RAW.read_text())
    assert payload.get("excluded_mirror", []) == []
    assert payload.get("excluded_uncached", []) == []
