"""Kubernetes serving backend - permanent stub (work order §5.7).

The work order is explicit: "Kubernetes는 adapter 자리(stub)만 두고 구현하지
않는다" - leave an adapter seat and do not implement it. A Kubernetes operator is
listed under CLAUDE.md "Out of scope". This class exists so the backend registry
has a named seat; every method refuses.
"""

from __future__ import annotations

from planner.deploy.base import (
    DeploymentHandle,
    DeploymentMetrics,
    ServingBackend,
)
from planner.inventory import ClusterSpecV2, ExecutionIsland
from planner.plan import DeploymentPlan

_MESSAGE = (
    "KubernetesBackend is intentionally not implemented: a Kubernetes operator is out of "
    "scope for HeteroPilot (CLAUDE.md 'Out of scope', work order §5.7). Use a local/SSH "
    "backend instead."
)


class KubernetesBackend(ServingBackend):
    """Out-of-scope placeholder; never implemented."""

    name = "kubernetes"

    def validate(
        self,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> list[str]:
        return [_MESSAGE]

    def launch(
        self,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> DeploymentHandle:
        raise NotImplementedError(_MESSAGE)

    def stop(self, deployment_id: str) -> None:
        raise NotImplementedError(_MESSAGE)

    def metrics(self, deployment_id: str) -> DeploymentMetrics:
        raise NotImplementedError(_MESSAGE)
