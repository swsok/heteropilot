"""Prediction interface (work order §5.5, §9).

`Predictor` is an ABC so the simulator subprocess can be mocked in tests, which
§9 requires. `SimResult` keeps failure modes distinct: a crashed run and a timed
out run are *not* infeasible configurations, and folding them into a feasibility
bucket silently biases the search (docs/deviations.md D12).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
from planner.plan import CandidateConfig, PredictedMetrics
from planner.spec import ServiceSpec


class SimOutcome(str, enum.Enum):
    OK = "ok"
    #: Non-zero exit. The configuration may well be servable in reality.
    CRASHED = "crashed"
    #: Exceeded the wall-clock budget. Attempt 1 of the D12 fix showed the
    #: simulator can hang rather than crash, so this must be its own outcome.
    TIMEOUT = "timeout"
    #: Output was produced but could not be parsed.
    UNPARSEABLE = "unparseable"

    @property
    def is_error(self) -> bool:
        return self is not SimOutcome.OK


@dataclass
class SimResult:
    candidate_id: str
    outcome: SimOutcome
    metrics: PredictedMetrics | None = None
    detail: str = ""
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome is SimOutcome.OK and self.metrics is not None


class Predictor(ABC):
    """Predicts the metrics of a candidate configuration."""

    @abstractmethod
    def predict(
        self,
        candidate: CandidateConfig,
        spec: ServiceSpec,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
    ) -> SimResult:
        """Evaluate one candidate. Must never raise for a simulation failure -
        return a SimResult carrying the outcome instead."""

    def close(self) -> None:  # noqa: B027 - optional hook, not every predictor has state
        """Release any resources (temp dirs, caches). Optional to override."""
