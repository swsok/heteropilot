"""Stage-6 surrogate ranker for top-K candidate selection (work order §5.4).

Candidate blow-up (P/D squares the pairing space) makes full simulation of every
candidate expensive. Stage 6 scores all candidates with a cheap surrogate, keeps
only the top-K, and lets the real simulator evaluate just those.

This is a HEURISTIC, deliberately different from the sound bound-pruning stages
4-5. Stages 4-5 are relaxations of feasibility - they may only reject a candidate
when the most optimistic arithmetic already misses a constraint, so they never
drop the optimum. A surrogate top-K CAN drop the optimum; that loss is *surrogate
error*, measured against the exhaustive oracle
(experiments/scripts/exp_surrogate.py), never a correctness bug. For that reason
a `SurrogateRanker` produces only an ORDERING - it never emits `PredictedMetrics`
and is not a `Predictor`, so it cannot inject un-simulated numbers into
feasibility or ranking (the MockPredictor-physics failure CLAUDE.md warns of).

The shipped ranker is analytical: it reuses `greedy.rank`, i.e. the same
memory-roofline goodput/J proxy the candidate generator uses at stage 5, so the
surrogate respects exactly the physics of the bounds. A learned (xgboost) ranker
is a documented follow-up, gated on a real training corpus that does not exist on
a single-node CUDA machine; building it now would risk an *asserted* accuracy that
rule 3 forbids (accuracy must be measured, per exp_surrogate.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from planner.inventory import AcceleratorProfile, ExecutionIsland
from planner.optimizer import greedy
from planner.plan import CandidateConfig
from planner.spec import ServiceSpec


class SurrogateRanker(ABC):
    """Orders candidates best-first for top-K selection. Deterministic. A
    HEURISTIC that may rank the optimum low - not a sound bound."""

    @abstractmethod
    def order(
        self,
        candidates: list[CandidateConfig],
        spec: ServiceSpec,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
        *,
        gpu_memory_utilization: float = 0.90,
    ) -> list[CandidateConfig]:
        """Return `candidates` reordered best-first. Same length, same members."""
        raise NotImplementedError


class AnalyticalRooflineRanker(SurrogateRanker):
    """Rank by the memory-roofline goodput/J proxy (`greedy.rank`), which mirrors
    the candidate generator's stage-5 physics. No simulation, no training,
    deterministic (candidate-id tie-break, inherited from greedy)."""

    def order(
        self,
        candidates: list[CandidateConfig],
        spec: ServiceSpec,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
        *,
        gpu_memory_utilization: float = 0.90,
    ) -> list[CandidateConfig]:
        ranked = greedy.rank(
            candidates, spec, islands, profiles,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        by_id = {c.id: c for c in candidates}
        return [by_id[e.candidate_id] for e in ranked]


class BinnedRooflineRanker(SurrogateRanker):
    """Rank by the roofline proxy tok/J to its MEANINGFUL precision, then by the
    roofline TPOT floor. The shipped default since 2026-09-11.

    `AnalyticalRooflineRanker` orders on a quantity that is algebraically
    invariant to TP and DP -- throughput and power both scale with `tp * dp`, so
    the ratio cancels (deviations D30). It is not quite constant across the axis,
    though: floating-point arithmetic leaves a spread of about one part in ten
    thousand, and sorting on that dust is what decides the parallelism
    configuration. The ranker does not so much ignore the axis as let rounding
    error pick for it -- which is why `tpj_then_floor`, an explicit tie-break on
    the floor, measured byte-identical to plain `roofline`: no two values are ever
    exactly tied, so the tie-break never fires.

    So group candidates whose proxy tok/J differs by less than `rel_tol`, treat a
    group as tied, and order INSIDE it by `roofline_tpot_ms` -- the one term in
    `greedy.estimate` that does vary with parallelism. The coarse ordering
    (accelerator, `max_num_seqs`) is untouched: those differ by factors, not by
    parts per ten thousand.

    Bins are cut where the relative gap between consecutive sorted values exceeds
    `rel_tol`, not rounded onto a fixed grid, so no boundary can split a group
    that is genuinely tied.

    **Measured, on sixteen corpora** (`docs/surrogate_topk_regret.md`): weakly
    dominates the shipped ranker -- 96 (corpus, K) cells, 11 strictly better, 0
    worse, 85 identical -- and cuts false-infeasibility at K=20 from 8 corpora to
    2. `rel_tol` is not fitted: 0.001, 0.01 and 0.05 give identical regret curves
    everywhere and select the identical SET of candidates at every K measured.
    They do not give the identical ORDER -- a wider bin merges groups a narrower
    one keeps apart, differing at 56 and 74 positions of 324 deep in the list --
    but top-K reads only the membership, and that is the check that the width is
    not doing the work.

    What it does NOT fix: at 20 rps every efficiency-ordered ranker is
    false-infeasible to K=50 on both fixtures, because there feasibility is
    decided by the floor alone. `--top-k` is still not a general cost lever.
    """

    #: Three orders of magnitude above the proxy's numerical noise (about 1e-4
    #: relative) and two below the factor-scale gaps it must keep apart.
    DEFAULT_REL_TOL = 0.01

    def __init__(self, rel_tol: float = DEFAULT_REL_TOL) -> None:
        if not 0.0 < rel_tol < 1.0:
            raise ValueError(f"rel_tol must be in (0, 1), got {rel_tol}")
        self.rel_tol = rel_tol

    def order(
        self,
        candidates: list[CandidateConfig],
        spec: ServiceSpec,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
        *,
        gpu_memory_utilization: float = 0.90,
    ) -> list[CandidateConfig]:
        est = {
            e.candidate_id: e
            for e in greedy.rank(candidates, spec, islands, profiles,
                                 gpu_memory_utilization=gpu_memory_utilization)
        }

        def _floor_key(c: CandidateConfig) -> tuple[float, str]:
            return (est[c.id].roofline_tpot_ms, c.id)

        by_tpj = sorted(
            candidates,
            key=lambda c: (-est[c.id].proxy_tokens_per_joule, c.id),
        )
        out: list[CandidateConfig] = []
        group: list[CandidateConfig] = []
        top: float | None = None
        for c in by_tpj:
            tpj = est[c.id].proxy_tokens_per_joule
            # A non-positive proxy cannot open or extend a relative-gap bin; it
            # gets its own, which keeps the walk total and order-preserving.
            same = bool(group) and top is not None and tpj > 0 and top > 0 and (
                (top - tpj) / top <= self.rel_tol)
            if not same:
                out.extend(sorted(group, key=_floor_key))
                group, top = [], tpj
            group.append(c)
        out.extend(sorted(group, key=_floor_key))
        return out
