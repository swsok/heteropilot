"""LLMServingSim predictor: compile, run, parse (work order §5.5).

The planner reasons over `ClusterSpecV2` but the simulator only accepts the flat
`configs/cluster/*.json` schema, so every candidate is compiled down at
prediction time. That compilation is lossy by construction - the simulator has
no link graph (docs/deviations.md D3) - and the reduction is recorded rather
than hidden.

Column names, units and the stdout layout all come from real artifacts at the
pinned commit; see docs/phase0_formats.md. Nothing here was guessed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from planner.inventory import AcceleratorProfile, ClusterSpecV2, ExecutionIsland
from planner.plan import CandidateConfig, PredictedMetrics
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.spec import ServiceSpec
from planner.topology import TopologyGraph, TopologyReduction
from planner.util import memory as memutil
from planner.util.percentile import percentile
from planner.util.power_parse import PowerParseError, parse_power
from planner.util.workload import WorkloadTrace

# Reuse the simulator's own (pinned) dimension inference so the per-dimension
# link_bw list the Level-2 compile emits always matches the length ASTRA-Sim
# expects; replicating the logic here would risk drifting from upstream (D14).
from serving.core.config_builder import _compute_network_dims

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Per-request CSV columns, verified against real output. Names contain spaces.
COL_INSTANCE = "instance id"
COL_REQUEST = "request id"
COL_INPUT = "input"
COL_OUTPUT = "output"
COL_ARRIVAL = "arrival"
COL_END = "end_time"
COL_LATENCY = "latency"
COL_TTFT = "TTFT"
COL_TPOT = "TPOT"

NS_PER_MS = 1e6
NS_PER_S = 1e9


def _repo_relative(path: Path) -> str:
    """Path as the simulator needs it: relative to the repository root.

    `serving/__main__.py` chdir's into `astra-sim/` and `config_builder.py` then
    prepends `../`, so an absolute path is mangled into `../<abs>` and never
    found. Anything outside the repo cannot be expressed at all, which is why
    the predictor stages its files under `outputs/`.
    """
    resolved = Path(path).resolve()
    try:
        return os.path.relpath(resolved, REPO_ROOT)
    except ValueError as exc:
        raise CompileError(
            f"{resolved} is not reachable from the repository root; the simulator "
            f"only accepts repo-relative paths"
        ) from exc


class CompileError(ValueError):
    """Raised when a candidate cannot be expressed in the simulator's schema."""


def _node_power_block(
    node,
    instances: list[dict],
    profiles: dict[str, AcceleratorProfile],
    islands: dict[str, ExecutionIsland],
    candidate: CandidateConfig,
) -> dict | None:
    """Build the node's `power:` block, or None to run without energy output.

    Requires both halves: node-level components from `Node.power` and per-device
    figures from each accelerator profile. Emitting a partial block would make
    the simulator report an energy number built on defaults nobody chose, which
    is worse than reporting none - §5.6 then correctly refuses to evaluate the
    power constraints rather than silently passing them.
    """
    if node.power is None:
        return None

    npu: dict[str, dict] = {}
    for assignment in candidate.assignments:
        island = islands[assignment.island_id]
        if island.node_id != node.id:
            continue
        profile = profiles.get(island.accelerator_model)
        if profile is None or profile.power is None or profile.sim_hardware is None:
            return None
        npu[profile.sim_hardware] = {
            "idle_power": profile.power.idle_power,
            "standby_power": profile.power.standby_power,
            "active_power": profile.power.active_power,
            "standby_duration": profile.power.standby_duration,
        }
    if not npu:
        return None

    p = node.power
    return {
        "base_node_power": p.base_node_power,
        "npu": npu,
        "cpu": {
            "idle_power": p.cpu_idle_power,
            "active_power": p.cpu_active_power,
            "util": p.cpu_util,
        },
        "dram": {
            "dimm_size": p.dram_dimm_size,
            "idle_power": p.dram_idle_power,
            "energy_per_bit": p.dram_energy_per_bit,
        },
        "link": {
            "num_links": p.link_num_links,
            "idle_power": p.link_idle_power,
            "energy_per_bit": p.link_energy_per_bit,
        },
        "nic": {"num_nics": p.nic_num_nics, "idle_power": p.nic_idle_power},
        "storage": {
            "num_devices": p.storage_num_devices,
            "idle_power": p.storage_idle_power,
        },
    }


def compile_to_sim_config(
    candidate: CandidateConfig,
    cluster: ClusterSpecV2,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    *,
    topology: TopologyGraph,
    gpu_memory_utilization: float = 0.90,
    activation_reserve_gb: float = 0.0,
    topology_level: int = 1,
) -> tuple[dict, TopologyReduction]:
    """CandidateConfig + ClusterSpecV2 -> the simulator's cluster-config dict.

    `npu_mem.mem_size` is written **derated**, not nominal. The simulator computes
    `mem_for_kv = mem_size - weight` with no utilization factor and no activation
    reserve, so passing the raw device size hands it ~55-71% more KV than the
    hardware really has. Feeding the derated size instead cut mean absolute error
    on a measured A5000 run from 22.54% to 9.26% (docs/deviations.md D10). The
    derating is echoed into provenance so a result is never mistaken for one
    computed against nominal capacity.
    """
    selected = [islands[a.island_id] for a in candidate.assignments]
    reduction = topology.reduce_for_simulator(selected)

    by_node: dict[str, list[dict]] = {}
    for assignment in candidate.assignments:
        island = islands[assignment.island_id]
        profile = profiles.get(island.accelerator_model)
        if profile is None or profile.sim_hardware is None:
            raise CompileError(
                f"island {island.id}: no profile with a sim_hardware key; "
                f"there is no profiler/perf/<hardware>/ bundle to simulate against"
            )
        if assignment.total_devices > island.size:
            raise CompileError(
                f"island {island.id}: candidate wants {assignment.total_devices} devices "
                f"but the island has {island.size}"
            )

        per_device_gb = island.total_memory_gb / island.size
        effective_gb = per_device_gb * gpu_memory_utilization - activation_reserve_gb
        if effective_gb <= 0:
            raise CompileError(
                f"island {island.id}: derating leaves {effective_gb:.2f} GB of usable memory"
            )

        knobs = candidate.knobs
        for replica in range(assignment.dp_replicas):
            by_node.setdefault(island.node_id, []).append(
                {
                    "model_name": candidate.model,
                    "hardware": profile.sim_hardware,
                    "npu_mem": {
                        "mem_size": round(effective_gb, 3),
                        "mem_bw": profile.memory_bandwidth_gbps,
                        "mem_latency": 0,
                    },
                    "num_npus": assignment.devices_per_replica,
                    "tp_size": assignment.tp_size,
                    "pp_size": assignment.pp_size,
                    "pd_type": assignment.role.pd_type,
                    "max_num_seqs": knobs.max_num_seqs,
                    "max_num_batched_tokens": knobs.max_num_batched_tokens,
                    "enable_chunked_prefill": knobs.enable_chunked_prefill,
                    "enable_prefix_caching": knobs.enable_prefix_caching,
                    "prioritize_prefill": knobs.prioritize_prefill,
                    "block_size": knobs.block_size,
                    "kv_cache_dtype": knobs.kv_cache_dtype,
                }
            )
            del replica

    nodes = []
    for node in cluster.nodes:
        instances = by_node.get(node.id)
        if not instances:
            continue
        entry: dict = {
            "num_instances": len(instances),
            "cpu_mem": {
                "mem_size": node.cpu_memory_gb,
                "mem_bw": node.cpu_memory_bw_gbps,
                "mem_latency": node.cpu_memory_latency_ns,
            },
            "instances": instances,
        }
        power = _node_power_block(node, instances, profiles, islands, candidate)
        if power is not None:
            entry["power"] = power
        nodes.append(entry)

    if not nodes:
        raise CompileError(f"candidate {candidate.id} placed no instances on any node")

    # Level 1 (default): one scalar bottleneck serves every ASTRA-Sim dimension.
    # Level 2: emit a per-dimension list so intra-island (TP) collectives keep
    # their real bandwidth instead of being dragged down by a slow cross-instance
    # fabric. The list length must equal ASTRA-Sim's dimension count, so size it
    # from the pinned _compute_network_dims over the instances in the same order
    # config_builder flattens them (node order, then instance order).
    link_bw: float | list[float] = reduction.link_bw_gbps
    link_latency: float | list[float] = reduction.link_latency_ns
    if topology_level == 2:
        perdim = topology.reduce_for_simulator_perdim(selected)
        flattened = [inst for node in nodes for inst in node["instances"]]
        num_dims = len(_compute_network_dims(flattened))
        if perdim.cross_bw_gbps is not None and num_dims == 2:
            assert perdim.cross_lat_ns is not None  # set together with cross_bw
            link_bw = [perdim.intra_bw_gbps, perdim.cross_bw_gbps]
            link_latency = [perdim.intra_lat_ns, perdim.cross_lat_ns]
        else:
            # Single distinct island (DP replicas / same-island P/D), or a single
            # network dimension: intra serves everything and normalizes exactly
            # like the Level-1 scalar for this placement (byte-identical).
            link_bw = perdim.intra_bw_gbps
            link_latency = perdim.intra_lat_ns

    config = {
        "num_nodes": len(nodes),
        "link_bw": link_bw,
        "link_latency": link_latency,
        "nodes": nodes,
    }
    return config, reduction


class LLMServingSimPredictor(Predictor):
    """Runs `python -m serving` as a subprocess, one process per candidate."""

    def __init__(
        self,
        trace: WorkloadTrace,
        *,
        work_dir: Path | None = None,
        timeout_s: float = 900.0,
        gpu_memory_utilization: float = 0.90,
        activation_reserve_gb: float = 0.0,
        python: str | None = None,
        retry_once: bool = True,
        keep_artifacts: bool = False,
        topology_level: int = 1,
        routing_policy: str = "LOAD",
        run_id_prefix: str = "",
    ) -> None:
        self.trace = trace
        self.timeout_s = timeout_s
        self.gpu_memory_utilization = gpu_memory_utilization
        self.activation_reserve_gb = activation_reserve_gb
        # 1 = one scalar link_bw (default); 2 = per-dimension list (Level-2, D3).
        self.topology_level = topology_level
        # Request-routing policy passed to the simulator (RR/RAND/LOAD/CUSTOM).
        # Default LOAD keeps the Phase-2 behavior byte-identical; the §12 router
        # baseline varies it. Router choice only affects multi-replica candidates.
        self.routing_policy = routing_policy
        self.python = python or sys.executable
        self.retry_once = retry_once
        # The simulator stages ASTRA-Sim inputs under a --run-id derived from
        # the candidate id. Candidate ids are unique only within one search, so
        # two concurrent predictors simulating identically-named candidates
        # (e.g. ScenarioLab evaluating the same placement on two clusters)
        # would share - and corrupt - one input directory. A caller-supplied
        # prefix restores isolation; empty keeps existing ids byte-identical.
        self.run_id_prefix = run_id_prefix
        self.keep_artifacts = keep_artifacts
        self._owns_dir = work_dir is None
        # The simulator chdir's into astra-sim/ and then prepends "../" to every
        # path it was given (serving/__main__.py:199, config_builder.py:274), so
        # arguments must be **repo-relative**. Staging under the repo keeps that
        # possible; a temp dir on another filesystem would not be expressible.
        self.work_dir = (
            Path(work_dir)
            if work_dir
            else Path(tempfile.mkdtemp(prefix="hp-sim-", dir=REPO_ROOT / "outputs"))
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.last_reduction: TopologyReduction | None = None

    # -- Predictor ---------------------------------------------------------

    def predict(
        self,
        candidate: CandidateConfig,
        spec: ServiceSpec,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
        profiles: dict[str, AcceleratorProfile],
    ) -> SimResult:
        try:
            config, reduction = compile_to_sim_config(
                candidate,
                cluster,
                islands,
                profiles,
                topology=TopologyGraph(cluster),
                gpu_memory_utilization=self.gpu_memory_utilization,
                activation_reserve_gb=self.activation_reserve_gb,
                topology_level=self.topology_level,
            )
        except CompileError as exc:
            return SimResult(candidate.id, SimOutcome.CRASHED, detail=f"compile failed: {exc}")
        self.last_reduction = reduction

        # Candidate ids may contain characters that are awkward in paths
        # (mixed ids carry parentheses and plus signs); sanitize for the dir.
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in candidate.id)
        run_dir = self.work_dir / safe_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "cluster.json"
        config_path.write_text(json.dumps(config, indent=2))

        # Energy is trustworthy only when EVERY node in this deployment carries
        # a power block. Upstream itself disables power modeling when any node
        # lacks one (config_builder.py:326), so today this guard is defense in
        # depth: if that upstream behavior ever changes, a partially-covered
        # total would under-count energy and inflate tokens/J (deviations D14).
        power_complete = all("power" in node for node in config["nodes"])

        result = self._run_once(candidate, spec, config_path, run_dir, power_complete)
        if result.outcome is SimOutcome.CRASHED and self.retry_once:
            # §5.5 asks for one retry. A deterministic simulator rarely benefits,
            # but a transient failure (disk, port, ASTRA-Sim startup) can.
            retry = self._run_once(
                candidate, spec, config_path, run_dir, power_complete, attempt=2
            )
            if retry.ok:
                retry.warnings.append("succeeded on retry after a first-attempt failure")
                return retry
        return result

    def close(self) -> None:
        if self._owns_dir and not self.keep_artifacts and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

    # -- internals ---------------------------------------------------------

    def _run_once(
        self,
        candidate: CandidateConfig,
        spec: ServiceSpec,
        config_path: Path,
        run_dir: Path,
        power_complete: bool,
        attempt: int = 1,
    ) -> SimResult:
        csv_path = run_dir / f"sim{attempt}.csv"
        log_path = run_dir / f"sim{attempt}.log"
        run_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in f"{self.run_id_prefix}{candidate.id}-a{attempt}"
        )

        cmd = [
            self.python, "-m", "serving",
            "--cluster-config", _repo_relative(config_path),
            "--dataset", _repo_relative(self.trace.path),
            "--output", _repo_relative(csv_path),
            "--num-reqs", str(self.trace.num_requests),
            "--dtype", spec.service.dtype,
            "--kv-cache-dtype", spec.service.kv_cache_dtype,
            "--block-size", str(candidate.knobs.block_size),
            "--max-num-seqs", str(candidate.knobs.max_num_seqs),
            "--max-num-batched-tokens", str(candidate.knobs.max_num_batched_tokens),
            "--request-routing-policy", self.routing_policy,
            "--network-backend", "analytical",
            "--log-level", "WARNING",
            "--log-interval", "1.0",
            "--run-id", run_id,
        ]
        if not candidate.knobs.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if candidate.knobs.enable_chunked_prefill is False:
            cmd.append("--no-enable-chunked-prefill")
        if candidate.knobs.prioritize_prefill:
            cmd.append("--prioritize-prefill")

        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                timeout=self.timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            log_path.write_text(f"timed out after {self.timeout_s}s\ncommand: {' '.join(cmd)}\n")
            return SimResult(
                candidate.id,
                SimOutcome.TIMEOUT,
                detail=(
                    f"simulator exceeded {self.timeout_s:.0f}s. This is a distinct outcome from "
                    f"a crash: the simulator can deadlock when NPU memory saturates "
                    f"(docs/deviations.md D12)."
                ),
                artifacts={"log": str(log_path), "config": str(config_path)},
            )
        wall = time.monotonic() - started

        stdout = proc.stdout or ""
        log_path.write_text(stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))

        if proc.returncode != 0:
            tail = (proc.stderr or stdout).strip().splitlines()[-3:]
            return SimResult(
                candidate.id,
                SimOutcome.CRASHED,
                detail=f"exit {proc.returncode}: " + " | ".join(tail),
                artifacts={"log": str(log_path), "config": str(config_path)},
            )

        if not csv_path.exists():
            return SimResult(
                candidate.id,
                SimOutcome.UNPARSEABLE,
                detail="simulator exited 0 but wrote no per-request CSV",
                artifacts={"log": str(log_path)},
            )

        try:
            metrics, warnings = self._parse(csv_path, stdout, spec, wall, power_complete)
        except (ValueError, KeyError, PowerParseError) as exc:
            return SimResult(
                candidate.id,
                SimOutcome.UNPARSEABLE,
                detail=f"cannot parse simulator output: {exc}",
                artifacts={"log": str(log_path), "csv": str(csv_path)},
            )

        return SimResult(
            candidate.id,
            SimOutcome.OK,
            metrics=metrics,
            warnings=warnings,
            artifacts={"log": str(log_path), "csv": str(csv_path), "config": str(config_path)},
        )

    def _parse(
        self,
        csv_path: Path,
        stdout: str,
        spec: ServiceSpec,
        wall: float,
        power_complete: bool,
    ) -> tuple[PredictedMetrics, list[str]]:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("per-request CSV has no rows")
        missing = {COL_TTFT, COL_TPOT, COL_OUTPUT, COL_ARRIVAL, COL_END} - set(df.columns)
        if missing:
            raise KeyError(f"CSV is missing expected columns: {sorted(missing)}")

        warnings: list[str] = []

        ttft_ms = (df[COL_TTFT] / NS_PER_MS).tolist()
        tpot_ms = (df[COL_TPOT] / NS_PER_MS).tolist()

        # Percentiles come from our single util, never from the simulator's own
        # printed P99 - the two use different interpolation (§4).
        p50_ttft, p95_ttft, p99_ttft = (percentile(ttft_ms, p) for p in (50, 95, 99))
        p50_tpot, p95_tpot, p99_tpot = (percentile(tpot_ms, p) for p in (50, 95, 99))

        duration_s = float(df[COL_END].max() - df[COL_ARRIVAL].min()) / NS_PER_S
        completed_tokens = int(df[COL_OUTPUT].sum())
        total_tokens = int((df[COL_INPUT] + df[COL_OUTPUT]).sum())
        throughput = total_tokens / duration_s if duration_s > 0 else 0.0

        # SLO attainment: a request counts only if it meets BOTH targets (§4).
        meets = (df[COL_TTFT] / NS_PER_MS <= spec.slo.ttft.max_ms) & (
            df[COL_TPOT] / NS_PER_MS <= spec.slo.tpot.max_ms
        )
        attainment = float(meets.mean())
        goodput_rps = float(meets.sum()) / duration_s if duration_s > 0 else 0.0

        power = parse_power(stdout)
        warnings.extend(power.warnings)
        energy = avg_w = peak_w = tok_per_j = None
        if power.summary is not None and not power_complete:
            warnings.append(
                "power output covers only part of this deployment's nodes; energy "
                "metrics dropped rather than reported under-counted (deviations D2)"
            )
        elif power.summary is not None:
            energy = power.summary.total_energy_j
            avg_w = power.summary.average_power_w
            peak_w = power.summary.peak_power_w
            tok_per_j = power.summary.tokens_per_joule(completed_tokens)

        return (
            PredictedMetrics(
                p50_ttft_ms=p50_ttft,
                p95_ttft_ms=p95_ttft,
                p99_ttft_ms=p99_ttft,
                p50_tpot_ms=p50_tpot,
                p95_tpot_ms=p95_tpot,
                p99_tpot_ms=p99_tpot,
                throughput_tps=throughput,
                slo_goodput_rps=goodput_rps,
                slo_attainment=attainment,
                completed_requests=len(df),
                completed_tokens=completed_tokens,
                total_energy_j=energy,
                average_power_w=avg_w,
                peak_power_w=peak_w,
                tokens_per_joule=tok_per_j,
                sim_wall_seconds=round(wall, 2),
            ),
            warnings,
        )


def kv_matched_memory_gb(
    model: str,
    *,
    tp_size: int,
    dtype: str,
    measured_kv_gib: float,
) -> float:
    """`mem_size` that reproduces a measured vLLM KV budget exactly.

    Handy when a real measurement is available for the device: the simulator
    computes `mem_for_kv = mem_size - weight`, so setting
    `mem_size = weight + measured_kv` makes the two agree by construction. This
    is how the A5000 KV-matched configuration was derived (D10).
    """
    report = memutil.evaluate(
        model, tp_size=tp_size, device_memory_gb=1e6, dtype=dtype,
        gpu_memory_utilization=1.0,
    )
    return report.weight_bytes / (1024 ** 3) + measured_kv_gib
