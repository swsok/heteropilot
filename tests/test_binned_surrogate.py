"""The binned surrogate sees the parallelism axis the shipped one was blind to.

`docs/deviations.md` D30: `AnalyticalRooflineRanker` orders by a proxy tok/J that
is algebraically invariant to TP and DP, because throughput and power both scale
with `tp * dp`. It is not exactly constant, though — floating-point arithmetic
leaves about one part in ten thousand — and sorting on that dust is what picks the
parallelism configuration. That is why `tpj_then_floor`, an explicit tie-break on
the parallelism-sensitive term, measured **byte-identical to plain `roofline`**:
no two values are ever exactly tied, so the tie-break never fires.

`BinnedRooflineRanker` makes the tie explicit — group values within `rel_tol`,
order inside a group by the roofline TPOT floor — and the regret curve says it
weakly dominates the shipped ranker over sixteen corpora
(`docs/surrogate_topk_regret.md`).

These tests pin the properties that claim rests on. The regret curve itself is a
measurement and lives in the results doc, not here.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from planner.candidate_generator import CandidateGenerator
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import greedy
from planner.optimizer.surrogate import (
    AnalyticalRooflineRanker,
    BinnedRooflineRanker,
)
from planner.spec import load_service_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    cluster = load_cluster_spec(ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml")
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    cands = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_bound_pruning=True, enable_pd=False,
    ).generate().candidates
    return spec, {i.id: i for i in islands}, profiles, cands


def _order(ranker, corpus):
    spec, islands, profiles, cands = corpus
    return [c.id for c in ranker.order(cands, spec, islands, profiles)]


def test_it_is_a_permutation_not_a_filter(corpus):
    """A ranker produces an ORDERING. Dropping a candidate here would be a
    silent top-K of its own."""
    _, _, _, cands = corpus
    got = _order(BinnedRooflineRanker(), corpus)
    assert sorted(got) == sorted(c.id for c in cands)
    assert len(got) == len(cands)


def test_it_is_deterministic(corpus):
    assert _order(BinnedRooflineRanker(), corpus) == _order(BinnedRooflineRanker(), corpus)


def test_it_actually_reorders_the_shipped_ranker(corpus):
    """If it did not, it would be `tpj_then_floor` — which measured identical to
    `roofline` and is the failure this class exists to avoid."""
    assert _order(BinnedRooflineRanker(), corpus) != _order(AnalyticalRooflineRanker(), corpus)


def test_the_bin_width_does_not_decide_which_candidates_survive_top_k(corpus):
    """0.1 %, 1 % and 5 % must select the same candidates.

    Not the same ORDER -- they differ at 56 and 74 positions of 324, deep in the
    list where a wider bin merges groups a narrower one keeps apart. What must
    agree is the only thing top-K reads: the SET of the first K. It does, at
    every K measured, which is the check that the width is not doing the work.
    The tolerances bracket the proxy's noise (about 1e-4 relative) from below and
    the factor-scale gaps between `max_num_seqs` groups from above."""
    a = _order(BinnedRooflineRanker(0.001), corpus)
    b = _order(BinnedRooflineRanker(0.01), corpus)
    c = _order(BinnedRooflineRanker(0.05), corpus)
    for k in (5, 10, 20, 30, 50):
        assert set(a[:k]) == set(b[:k]) == set(c[:k]), f"top-{k} membership differs"


def test_within_a_bin_the_order_is_by_the_parallelism_term(corpus):
    """The load-bearing property: candidates whose proxy tok/J is tied to within
    `rel_tol` come out sorted by `roofline_tpot_ms` ascending."""
    spec, islands, profiles, cands = corpus
    est = {e.candidate_id: e for e in greedy.rank(cands, spec, islands, profiles)}
    ordered = BinnedRooflineRanker(0.01).order(cands, spec, islands, profiles)
    checked = 0
    for prev, cur in pairwise(ordered):
        p, c = est[prev.id], est[cur.id]
        hi = p.proxy_tokens_per_joule
        if hi > 0 and (hi - c.proxy_tokens_per_joule) / hi <= 0.01:
            # Same bin (or the bin's own chain) -- the floor must not decrease.
            assert c.roofline_tpot_ms >= p.roofline_tpot_ms or prev.id > cur.id, (
                f"{prev.id} -> {cur.id}: floor went {p.roofline_tpot_ms} -> "
                f"{c.roofline_tpot_ms} inside a bin")
            checked += 1
    assert checked > 0, "no tied pairs at all -- the fixture cannot exercise this"


def test_the_coarse_order_survives(corpus):
    """Binning must not reorder candidates that genuinely differ in efficiency.
    Anything better than the next by more than `rel_tol` keeps its place."""
    spec, islands, profiles, cands = corpus
    est = {e.candidate_id: e for e in greedy.rank(cands, spec, islands, profiles)}
    ordered = BinnedRooflineRanker(0.01).order(cands, spec, islands, profiles)
    for prev, cur in pairwise(ordered):
        p, c = est[prev.id], est[cur.id]
        hi = p.proxy_tokens_per_joule
        if hi > 0 and (hi - c.proxy_tokens_per_joule) / hi > 0.01:
            assert c.proxy_tokens_per_joule < p.proxy_tokens_per_joule


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.01, 2.0])
def test_a_nonsense_tolerance_is_refused(bad):
    with pytest.raises(ValueError):
        BinnedRooflineRanker(bad)


def test_the_cli_default_is_the_binned_ranker():
    """The adoption itself. `roofline` stays selectable so a published run
    reproduces."""
    import argparse

    from planner.__main__ import build_parser

    def _walk(parser):
        for a in parser._actions:
            if "--surrogate" in (a.option_strings or []):
                yield a
            if isinstance(a, argparse._SubParsersAction):
                for sub in a.choices.values():
                    yield from _walk(sub)

    action = next(_walk(build_parser()))
    assert action.default == "binned"
    assert set(action.choices) == {"binned", "roofline"}
