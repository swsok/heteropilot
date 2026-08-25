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
