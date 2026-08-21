"""Ascend NPU vLLM serving backend - stub (work order §5.7).

vLLM-Ascend would mirror `VllmCudaBackend`: assemble a ``vllm serve`` command
and set ``ASCEND_RT_VISIBLE_DEVICES`` (the NPU analogue of
``CUDA_VISIBLE_DEVICES``) from the island's device map, then poll ``npu-smi`` for
power instead of ``nvidia-smi``.

No Ascend hardware exists on this machine (see CLAUDE.md "This machine"), so
every operation raises with a clear message rather than pretending to work.
Absolute rule 3: we never emit numbers from hardware that is not present.
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
    "VllmAscendBackend is not implemented: no Ascend NPU hardware is present. "
    "When it exists, it will set ASCEND_RT_VISIBLE_DEVICES and poll npu-smi for power, "
    "mirroring VllmCudaBackend."
)


class VllmAscendBackend(ServingBackend):
    """Placeholder for the Ascend NPU backend."""

    name = "vllm-ascend"

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
