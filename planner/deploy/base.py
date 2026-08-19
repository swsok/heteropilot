"""ServingBackend abstraction and its data types (work order §5.7).

`ServingBackend` mirrors the `Predictor` ABC in `planner/predictor/__init__.py`:
the concrete backend that talks to real hardware sits behind an interface so a
test can substitute a mock, and the CLI never imports vLLM at all.

Field names on `DeploymentMetrics` deliberately match `PredictedMetrics`
(planner/plan.py) wherever the two overlap, so `predictor/calibration.py` can
diff a prediction against a measurement without translating names.

Device resolution: an execution island lists accelerator ids such as ``gpu0``
or ``npu1``. The runtime index a backend hands to ``CUDA_VISIBLE_DEVICES`` (or
``ASCEND_RT_VISIBLE_DEVICES``) is taken from the numeric suffix of that id
(``gpu0`` -> 0). This assumes the id suffix equals the driver's device index on
its node, which is true for the profiled machines but not guaranteed on every
host; callers may override it by passing a different ``index_of`` callable.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from planner.inventory import ClusterSpecV2, ExecutionIsland
from planner.plan import DeploymentPlan


class DeploymentError(RuntimeError):
    """Raised when a deployment cannot be launched, stopped or read.

    A *validation* problem is never raised - it is returned by
    ``ServingBackend.validate`` as a string in a list. This exception is for
    operations that were expected to succeed and did not (a dead process, a
    missing handle, an unreachable metrics endpoint).
    """


@dataclass(frozen=True)
class DeviceMap:
    """Which node an island lives on, and its per-node device indices."""

    node_id: str
    device_indices: list[int]

    @property
    def visible_devices(self) -> str:
        """The ``CUDA_VISIBLE_DEVICES`` / ``ASCEND_RT_VISIBLE_DEVICES`` value."""
        return ",".join(str(i) for i in self.device_indices)


def device_index_from_id(accelerator_id: str) -> int:
    """Derive a driver device index from the numeric suffix of an id.

    ``gpu0`` -> 0, ``npu13`` -> 13. See the module docstring for the assumption
    this encodes and how to override it.
    """
    match = re.search(r"(\d+)$", accelerator_id)
    if match is None:
        raise DeploymentError(
            f"cannot derive a device index from accelerator id '{accelerator_id}': "
            "it has no trailing number. Pass an explicit index_of resolver."
        )
    return int(match.group(1))


def resolve_devices(
    island: ExecutionIsland,
    *,
    index_of: Callable[[str], int] = device_index_from_id,
) -> DeviceMap:
    """Map an island to its (node, device-index list)."""
    return DeviceMap(
        node_id=island.node_id,
        device_indices=[index_of(a) for a in island.accelerator_ids],
    )


@dataclass
class DeploymentHandle:
    """A launched (or would-be-launched) serving instance.

    Persisted as JSON under the backend's deployment state directory so a later
    ``status`` / ``stop`` invocation - a different process - can find it again.
    ``pid`` is ``None`` for a dry run or a remote launch whose pid is not local.
    """

    deployment_id: str
    backend: str
    pid: int | None
    host: str
    port: int
    base_url: str
    plan_id: str
    started_at: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Measured runtime metrics for a deployment (work order §5.7).

    Latency/throughput come from the vLLM Prometheus endpoint; power comes from
    ``nvidia-smi`` polling integrated with the trapezoidal rule (see
    ``planner/monitor``). Names overlap with ``PredictedMetrics`` on purpose so
    calibration can diff the two directly.
    """

    deployment_id: str
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float
    throughput_tps: float
    completed_requests: int
    completed_tokens: int
    average_power_w: float | None = None
    peak_power_w: float | None = None
    total_energy_j: float | None = None
    tokens_per_joule: float | None = None
    sample_count: int = 0
    window_seconds: float = 0.0

    @property
    def has_energy(self) -> bool:
        return self.total_energy_j is not None


class ServingBackend(ABC):
    """Launches, stops and reads a `DeploymentPlan` on real hardware.

    Concrete backends: `VllmCudaBackend`, `VllmAscendBackend` (stub, no NPU
    hardware yet), `KubernetesBackend` (stub, out of scope).
    """

    #: Backend tag, e.g. ``"vllm-cuda"``. Matches the DeploymentPlan §3.4
    #: ``instances[].backend`` naming.
    name: str = "abstract"

    @abstractmethod
    def validate(
        self,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> list[str]:
        """Return the reasons this plan cannot be served here; empty means OK.

        Never raises for a plan problem - the caller decides whether to proceed.
        """

    @abstractmethod
    def launch(
        self,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> DeploymentHandle:
        """Start the serving instance(s) and return a persistable handle."""

    @abstractmethod
    def stop(self, deployment_id: str) -> None:
        """Terminate a running deployment. Idempotent where possible."""

    @abstractmethod
    def metrics(self, deployment_id: str) -> DeploymentMetrics:
        """Scrape live metrics for a running deployment."""
