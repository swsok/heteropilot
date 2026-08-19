"""Local CUDA vLLM serving backend (work order §5.7).

Assembles a ``vllm serve`` command from a `DeploymentPlan`, sets
``CUDA_VISIBLE_DEVICES`` from the island's device map, launches it as a detached
local subprocess, and reads it back through the Prometheus endpoint plus power
sampling.

Two deliberate boundaries for this Phase 4 increment:

* **Aggregated engines only.** Prefill/Decode split placement is Phase 5; a plan
  carrying a non-aggregated role is rejected by `validate` with a clear message.
* **Local execution only.** ``host`` other than ``"local"`` is an SSH hook point
  (see `launch`); the remote path is documented but not implemented here.

``build_serve_command`` is a pure function so the argv, environment and knob
flags can be unit-tested without launching anything.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from planner.deploy.base import (
    DeploymentError,
    DeploymentHandle,
    DeploymentMetrics,
    ServingBackend,
    device_index_from_id,
    resolve_devices,
)
from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland, compatibility
from planner.monitor.metrics import PowerSampler, parse_vllm_metrics
from planner.plan import DeploymentPlan, IslandAssignment, Role

BACKEND_CUDA = "cuda"
DEFAULT_PORT = 8000
#: How long ``metrics()`` samples power before integrating energy.
DEFAULT_POWER_WINDOW_S = 5.0


@dataclass(frozen=True)
class ServeCommand:
    """A ready-to-run ``vllm serve`` invocation and its environment overrides."""

    argv: list[str]
    env: dict[str, str]

    def as_shell(self) -> str:
        """A copy-pasteable representation for dry-run output."""
        env = " ".join(f"{k}={v}" for k, v in self.env.items())
        return (env + " " if env else "") + " ".join(self.argv)


def _flag(argv: list[str], enabled: bool, name: str) -> None:
    """Append vLLM's paired boolean flag, ``--name`` or ``--no-name``."""
    argv.append(f"--{name}" if enabled else f"--no-{name}")


def build_serve_command(
    plan: DeploymentPlan,
    assignment: IslandAssignment,
    island: ExecutionIsland,
    *,
    port: int = DEFAULT_PORT,
    host: str = "0.0.0.0",
    index_of: Callable[[str], int] = device_index_from_id,
) -> ServeCommand:
    """Build the ``vllm serve`` argv + env for one aggregated engine. Pure.

    ``CUDA_VISIBLE_DEVICES`` is set from the island's device indices; the engine
    then addresses them as 0..N-1, so ``--tensor-parallel-size`` is the island's
    TP degree and no per-device pinning is needed.
    """
    knobs = plan.candidate.knobs
    devices = resolve_devices(island, index_of=index_of)

    argv: list[str] = [
        "vllm",
        "serve",
        plan.model,
        "--tensor-parallel-size",
        str(assignment.tp_size),
    ]
    if assignment.pp_size > 1:
        argv += ["--pipeline-parallel-size", str(assignment.pp_size)]
    argv += [
        "--dtype",
        plan.candidate.dtype,
        "--kv-cache-dtype",
        knobs.kv_cache_dtype,
        "--block-size",
        str(knobs.block_size),
        "--max-num-seqs",
        str(knobs.max_num_seqs),
        "--max-num-batched-tokens",
        str(knobs.max_num_batched_tokens),
        "--host",
        host,
        "--port",
        str(port),
    ]
    _flag(argv, knobs.enable_prefix_caching, "enable-prefix-caching")
    _flag(argv, knobs.enable_chunked_prefill, "enable-chunked-prefill")

    env = {"CUDA_VISIBLE_DEVICES": devices.visible_devices}
    return ServeCommand(argv=argv, env=env)


class VllmCudaBackend(ServingBackend):
    """Runs a plan as one or more local CUDA vLLM ``serve`` processes."""

    name = "vllm-cuda"

    def __init__(
        self,
        *,
        root: str | Path = ".",
        host: str = "local",
        port: int = DEFAULT_PORT,
        profiles: dict[str, AcceleratorProfile] | None = None,
        index_of: Callable[[str], int] = device_index_from_id,
        power_window_s: float = DEFAULT_POWER_WINDOW_S,
    ) -> None:
        self.root = Path(root)
        self.host = host
        self.port = port
        #: Optional: when supplied, `validate` also checks model/dtype support.
        self.profiles = profiles or {}
        self.index_of = index_of
        self.power_window_s = power_window_s

    # -- state directory -------------------------------------------------

    def _deployment_dir(self, deployment_id: str) -> Path:
        return self.root / "outputs" / "deployments" / deployment_id

    def _handle_path(self, deployment_id: str) -> Path:
        return self._deployment_dir(deployment_id) / "handle.json"

    def _pidfile(self, deployment_id: str) -> Path:
        return self._deployment_dir(deployment_id) / "vllm.pid"

    def read_handle(self, deployment_id: str) -> DeploymentHandle:
        path = self._handle_path(deployment_id)
        if not path.is_file():
            raise DeploymentError(
                f"no deployment '{deployment_id}' under {self._deployment_dir(deployment_id)}"
            )
        return DeploymentHandle(**json.loads(path.read_text()))

    def is_running(self, deployment_id: str) -> bool:
        pidfile = self._pidfile(deployment_id)
        if not pidfile.is_file():
            return False
        try:
            pid = int(pidfile.read_text().strip())
        except ValueError:
            return False
        return _pid_alive(pid)

    # -- validation ------------------------------------------------------

    def validate(
        self,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> list[str]:
        problems: list[str] = []
        for assignment in plan.candidate.assignments:
            problems += self._validate_assignment(assignment, plan, cluster, islands)
        return problems

    def _validate_assignment(
        self,
        assignment: IslandAssignment,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> list[str]:
        problems: list[str] = []
        island = islands.get(assignment.island_id)
        if island is None:
            return [f"island '{assignment.island_id}' is not in this cluster's islands"]

        if island.backend != BACKEND_CUDA:
            # Absolute rule 2: this backend serves CUDA only. A non-CUDA island
            # (or a plan that mixed backends) must be refused, not coerced.
            problems.append(
                f"island '{island.id}' has backend '{island.backend}', not "
                f"'{BACKEND_CUDA}'; VllmCudaBackend cannot serve it"
            )

        if assignment.role is not Role.AGGREGATED:
            problems.append(
                f"island '{island.id}' role is '{assignment.role.value}'; this increment "
                "serves aggregated engines only (Prefill/Decode split is Phase 5)"
            )

        if island.size % assignment.tp_size != 0:
            problems.append(
                f"island '{island.id}': tp_size={assignment.tp_size} does not divide "
                f"island size {island.size}"
            )
        if assignment.total_devices > island.size:
            problems.append(
                f"island '{island.id}': plan wants {assignment.total_devices} devices but the "
                f"island has {island.size}"
            )

        problems += self._validate_devices(assignment, island, cluster)
        problems += self._validate_model(island, plan)
        return problems

    def _validate_devices(
        self,
        assignment: IslandAssignment,
        island: ExecutionIsland,
        cluster: ClusterSpecV2,
    ) -> list[str]:
        problems: list[str] = []
        for accel_id in island.accelerator_ids:
            try:
                accel = cluster.accelerator(island.node_id, accel_id)
            except KeyError:
                problems.append(
                    f"island '{island.id}': device '{accel_id}' is not on node "
                    f"'{island.node_id}'"
                )
                continue
            if not accel.is_free:
                problems.append(
                    f"island '{island.id}': device '{accel_id}' is {accel.state.value}, "
                    "not FREE; refusing to launch onto an occupied device"
                )
        return problems

    def _validate_model(self, island: ExecutionIsland, plan: DeploymentPlan) -> list[str]:
        profile = self.profiles.get(island.accelerator_model)
        if profile is None:
            return []  # cannot check without a profile; not this backend's job to guess
        if not compatibility(plan.model, plan.candidate.dtype, profile):
            return [
                f"island '{island.id}': profile '{profile.profile_id}' declares no support "
                f"for {plan.model} @ {plan.candidate.dtype}"
            ]
        return []

    # -- launch / stop ---------------------------------------------------

    def launch(
        self,
        plan: DeploymentPlan,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
    ) -> DeploymentHandle:
        problems = self.validate(plan, cluster, islands)
        if problems:
            raise DeploymentError(
                "cannot launch: " + "; ".join(problems)
            )
        if self.host != "local":
            # SSH hook point. A remote launch would wrap the same argv/env in an
            # `ssh <host> env CUDA_VISIBLE_DEVICES=... vllm serve ...` and record
            # the remote pid. Left unimplemented on purpose for this increment.
            raise NotImplementedError(
                f"remote launch to host '{self.host}' is not implemented; only "
                "host='local' is supported in this increment (SSH hook point)"
            )
        if len(plan.candidate.assignments) != 1:
            raise NotImplementedError(
                "multi-island launch (a plan with more than one assignment) needs a router "
                "and is out of scope for this increment; launch a single-island plan"
            )
        assignment = plan.candidate.assignments[0]
        if assignment.dp_replicas != 1 or assignment.pp_size != 1:
            raise NotImplementedError(
                "launching dp_replicas>1 or pp_size>1 needs multi-engine orchestration and is "
                "out of scope for this increment; the config is still buildable for a dry run"
            )
        island = islands[assignment.island_id]
        command = build_serve_command(
            plan, assignment, island, port=self.port, index_of=self.index_of
        )

        deployment_id = plan.plan_id
        # Refuse to clobber a live deployment: overwriting its pidfile would
        # orphan the running server (still holding the GPUs) with no way to stop
        # it through this CLI.
        if self.is_running(deployment_id):
            raise DeploymentError(
                f"deployment '{deployment_id}' is already running; stop it first"
            )
        dep_dir = self._deployment_dir(deployment_id)
        dep_dir.mkdir(parents=True, exist_ok=True)
        log_path = dep_dir / "vllm.log"

        env = {**os.environ, **command.env}
        with log_path.open("w") as log:
            # start_new_session detaches into its own process group so stop()
            # can signal the whole group and the server outlives this CLI call.
            # argv is a built list (never a shell string), so no shell injection.
            proc = subprocess.Popen(
                command.argv,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._pidfile(deployment_id).write_text(str(proc.pid))

        handle = DeploymentHandle(
            deployment_id=deployment_id,
            backend=self.name,
            pid=proc.pid,
            host=self.host,
            port=self.port,
            base_url=f"http://127.0.0.1:{self.port}",
            plan_id=plan.plan_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            extra={
                "island_id": island.id,
                "node_id": island.node_id,
                "device_indices": resolve_devices(island, index_of=self.index_of).device_indices,
                "cuda_visible_devices": command.env["CUDA_VISIBLE_DEVICES"],
                "argv": command.argv,
                "log": str(log_path),
            },
        )
        self._handle_path(deployment_id).write_text(json.dumps(asdict(handle), indent=2))
        return handle

    def stop(self, deployment_id: str) -> None:
        pidfile = self._pidfile(deployment_id)
        if not pidfile.is_file():
            raise DeploymentError(f"no pidfile for deployment '{deployment_id}'")
        try:
            pid = int(pidfile.read_text().strip())
        except ValueError as exc:
            raise DeploymentError(f"corrupt pidfile for '{deployment_id}'") from exc
        # Guard against a recycled pid: only signal if the process is both alive
        # and still looks like our vLLM server. Otherwise the OS may have reused
        # the pid for an unrelated process and killpg would hit its group.
        if _pid_alive(pid) and _pid_is_vllm(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # raced to exit between the check and the signal; idempotent
            except OSError as exc:
                raise DeploymentError(f"failed to stop '{deployment_id}': {exc}") from exc
        pidfile.unlink(missing_ok=True)

    # -- metrics ---------------------------------------------------------

    def metrics(self, deployment_id: str) -> DeploymentMetrics:
        handle = self.read_handle(deployment_id)

        average_power_w: float | None = None
        peak_power_w: float | None = None
        total_energy_j: float | None = None
        tokens_per_joule: float | None = None
        sample_count = 0
        window_seconds = 0.0
        throughput_tps = 0.0

        # vLLM token/request counters are cumulative since server start, so
        # throughput is a delta over a window, never a single scrape divided by
        # anything. When power sampling runs we bracket its window with two
        # scrapes; the closing scrape also supplies the reported percentiles.
        device_indices = [int(i) for i in handle.extra.get("device_indices", [])]
        sampler = PowerSampler(device_indices)
        if device_indices and sampler.available:
            # Bracket the whole measurement with two scrapes and the wall clock,
            # so the token delta, the throughput denominator, and the energy all
            # cover the same window. Using the power-sample span (last-first
            # sample) as the denominator undercounts the interval; using lifetime
            # counters over-counts the numerator.
            t0 = time.monotonic()
            start = parse_vllm_metrics(self._fetch_metrics_text(handle))
            series = sampler.collect(self.power_window_s)
            scrape = parse_vllm_metrics(self._fetch_metrics_text(handle))
            elapsed = time.monotonic() - t0
            average_power_w = series.average_power_w
            peak_power_w = series.peak_power_w
            total_energy_j = series.total_energy_j
            sample_count = len(series.samples)
            window_seconds = elapsed
            # Cumulative counters -> windowed delta. Floor at 0 so a server
            # restart mid-window cannot produce a negative rate.
            token_delta = max(0.0, scrape.completed_tokens - start.completed_tokens)
            if elapsed > 0:
                throughput_tps = token_delta / elapsed
            if total_energy_j and total_energy_j > 0:
                # Windowed tokens / windowed joules - never lifetime tokens,
                # which would inflate tokens/J by orders of magnitude on a
                # long-lived server (a headline core metric).
                tokens_per_joule = token_delta / total_energy_j
        else:
            scrape = parse_vllm_metrics(self._fetch_metrics_text(handle))

        return DeploymentMetrics(
            deployment_id=deployment_id,
            p50_ttft_ms=scrape.p50_ttft_ms,
            p95_ttft_ms=scrape.p95_ttft_ms,
            p99_ttft_ms=scrape.p99_ttft_ms,
            p50_tpot_ms=scrape.p50_tpot_ms,
            p95_tpot_ms=scrape.p95_tpot_ms,
            p99_tpot_ms=scrape.p99_tpot_ms,
            throughput_tps=throughput_tps,
            completed_requests=scrape.completed_requests,
            completed_tokens=scrape.completed_tokens,
            average_power_w=average_power_w,
            peak_power_w=peak_power_w,
            total_energy_j=total_energy_j,
            tokens_per_joule=tokens_per_joule,
            sample_count=sample_count,
            window_seconds=window_seconds,
        )

    def _fetch_metrics_text(self, handle: DeploymentHandle) -> str:
        import urllib.error
        import urllib.request

        url = f"{handle.base_url}/metrics"
        try:
            # Fixed http:// URL from our own handle; not user-controlled.
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            raise DeploymentError(
                f"could not scrape {url}: {exc}. Is deployment '{handle.deployment_id}' running?"
            ) from exc


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _pid_is_vllm(pid: int) -> bool:
    """True if pid's command line looks like our vLLM server.

    A best-effort identity check (Linux /proc) to avoid signalling a process
    that inherited a recycled pid. If /proc is unreadable we assume True rather
    than silently refuse to stop a real server on a non-Linux host.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", "replace"
        )
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return True
    return "vllm" in cmdline.lower()
