"""M3 BatchRunner: the scenario matrix executor (DESIGN §6).

One scenario = one pure function call (FR-B2): (cluster YAML, service YAML,
scenario seed, policy) -> ScenarioResult. Workers share nothing; the DB is the
only truth for progress, which is what makes a batch resumable (FR-B4) and a
worker crash a contained event (FR-B5).
"""

from __future__ import annotations

import json
import tempfile
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from planner.inventory import (
    compatibility,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive, feasibility
from planner.plan import CandidateConfig, DeploymentPlan, IslandAssignment
from planner.predictor import Predictor
from planner.spec import load_service_spec
from planner.util import memory as memutil
from planner.util import provenance as prov
from planner.util.workload import generate_trace
from scenariolab.config import LabConfig
from scenariolab.generator.cluster_gen import ClusterSummary, generate_cluster
from scenariolab.generator.sampling import derive_seed, rng_for
from scenariolab.generator.slo_gen import ServiceSummary, generate_service
from scenariolab.runner.tiers import FIDELITY_SURROGATE, make_predictor
from scenariolab.store.db import ResultStore

BASELINE_ID = "baseline-fastest-maxtp"


@dataclass(frozen=True)
class ScenarioTask:
    """Everything one worker needs; deliberately picklable and path-based."""

    scenario_id: str
    batch_id: str
    cluster_id: str
    service_id: str
    cluster_yaml: str
    service_yaml: str
    seed: int
    num_requests: int
    enable_pd: bool
    predictor_kind: str
    root: str
    results_dir: str
    config_hash: str
    gpu_memory_utilization: float = 0.90
    activation_reserve_gb: float = 0.0


def _naive_baseline(
    spec: Any,
    cluster: Any,
    islands: list[Any],
    profiles: dict[str, Any],
    predictor: Predictor,
    *,
    gpu_memory_utilization: float,
) -> tuple[float | None, str]:
    """FR-B7 naive baseline: fastest-accelerator-class island, max TP that
    fits, all devices used. Returns (avg_power_w or None, note). The power is
    returned only when the baseline itself meets the SLO - a saving computed
    against an SLO-violating baseline would be meaningless."""
    compat = [
        i for i in islands
        if (p := profiles.get(i.accelerator_model)) is not None
        and compatibility(spec.model, spec.service.dtype, p)
    ]
    if not compat:
        return None, "no compatible island for the baseline policy"
    fastest_bw = max(profiles[i.accelerator_model].memory_bandwidth_gbps for i in compat)
    fast = [
        i for i in compat
        if profiles[i.accelerator_model].memory_bandwidth_gbps == fastest_bw
    ]
    island = sorted(fast, key=lambda i: (-i.size, i.id))[0]
    per_device_gb = island.total_memory_gb / island.size

    tp = None
    for t in sorted(island.max_tp_candidates, reverse=True):
        fits, _ = memutil.feasible(
            spec.model, tp_size=t, device_memory_gb=per_device_gb,
            dtype=spec.service.dtype, kv_cache_dtype=spec.service.kv_cache_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if fits:
            tp = t
            break
    if tp is None:
        return None, "baseline does not fit the fastest island at any TP"

    candidate = CandidateConfig(
        id=BASELINE_ID,
        model=spec.model,
        dtype=spec.service.dtype,
        assignments=[
            IslandAssignment(island_id=island.id, tp_size=tp, dp_replicas=island.size // tp)
        ],
    )
    result = predictor.predict(
        candidate, spec, cluster, {i.id: i for i in islands}, profiles
    )
    if not result.ok or result.metrics is None:
        return None, f"baseline prediction failed: {result.detail or result.outcome.value}"
    plan = DeploymentPlan(
        plan_id=BASELINE_ID, model=spec.model, candidate=candidate,
        predicted=result.metrics,
    )
    report = feasibility.evaluate(plan, spec)
    if not report.passed:
        return None, (
            "baseline violates: " + ", ".join(v.metric for v in report.violations)
        )
    if result.metrics.average_power_w is None:
        return None, "baseline produced no power figure"
    return result.metrics.average_power_w, "ok"


def run_scenario(
    task: ScenarioTask,
    predictor_factory: Callable[..., Predictor] | None = None,
) -> dict[str, Any]:
    """Pure scenario evaluation (FR-B2/B6/B7). Never raises: errors come back
    as {"ok": False, "error": ...} so the batch can isolate them (FR-B5)."""
    started = time.perf_counter()
    try:
        record = _run_scenario_inner(task, predictor_factory)
        record["ok"] = True
    except Exception:
        record = {"ok": False, "error": traceback.format_exc()}
    record["scenario_id"] = task.scenario_id
    record["elapsed_s"] = round(time.perf_counter() - started, 3)
    return record


def _run_scenario_inner(
    task: ScenarioTask,
    predictor_factory: Callable[..., Predictor] | None,
) -> dict[str, Any]:
    root = Path(task.root)
    spec = load_service_spec(task.service_yaml)
    cluster = load_cluster_spec(task.cluster_yaml)
    profiles = load_profiles_for(cluster, root)
    islands = detect_islands(cluster, profiles)

    with tempfile.TemporaryDirectory(prefix=f"slab-{task.scenario_id}-") as tmp:
        trace = generate_trace(
            spec, Path(tmp) / "workload.jsonl",
            num_requests=task.num_requests, seed=task.seed,
        )
        provenance = prov.collect(
            service_spec_path=task.service_yaml,
            cluster_spec_path=task.cluster_yaml,
            profile_paths=[
                root / a.profile
                for node in cluster.nodes for a in node.accelerators if a.profile
            ],
            dataset_path=trace.path,
            random_seed=task.seed,
            extra={
                "scenariolab": {
                    "scenario_id": task.scenario_id,
                    "batch_id": task.batch_id,
                    "lab_config_hash": task.config_hash,
                    "fidelity": task.predictor_kind,
                    # The trace lives in a per-run temp dir; recording that
                    # path would break byte-identical reruns. Identity is the
                    # dataset_hash provenance field; shape is recorded here.
                    "workload": {
                        k: v for k, v in trace.as_provenance().items() if k != "path"
                    },
                }
            },
        )

        if predictor_factory is not None:
            predictor = predictor_factory(trace)
        else:
            predictor = make_predictor(
                task.predictor_kind, trace,
                gpu_memory_utilization=task.gpu_memory_utilization,
                activation_reserve_gb=task.activation_reserve_gb,
            )
        try:
            output = exhaustive.search(
                spec, cluster, islands, profiles, predictor,
                enable_pd=task.enable_pd,
                gpu_memory_utilization=task.gpu_memory_utilization,
                activation_reserve_gb=task.activation_reserve_gb,
                max_workers=1,
                provenance=provenance,
            )
            baseline_power, baseline_note = _naive_baseline(
                spec, cluster, islands, profiles, predictor,
                gpu_memory_utilization=task.gpu_memory_utilization,
            )
        finally:
            predictor.close()

    fidelity = FIDELITY_SURROGATE if task.predictor_kind == FIDELITY_SURROGATE else (
        task.predictor_kind
    )
    summary: dict[str, Any] = {
        "feasible": output.feasible,
        "fidelity": fidelity,
        "calibrated": False,        # P2: calibration coverage lands with tiers
        "npu_extrapolated": False,  # P2: FR-T6 concurrency flag lands with tiers
        "violated": [v.model_dump(mode="json") for v in output.violated_constraints],
        "provenance": provenance,
        "baseline_power_w": baseline_power,
        "baseline_note": baseline_note,
        "power_saving_pct": None,
    }
    plan = output.recommended.plan if output.recommended is not None else None
    if plan is not None:
        m = plan.predicted
        summary.update(
            p99_ttft_ms=m.p99_ttft_ms,
            p99_tpot_ms=m.p99_tpot_ms,
            avg_power_w=m.average_power_w,
            peak_power_w=m.peak_power_w,
            tokens_per_joule=m.tokens_per_joule,
            slo_goodput=m.slo_goodput_rps,
            active_devices=plan.active_accelerators,
        )
        if baseline_power and m.average_power_w is not None and baseline_power > 0:
            summary["power_saving_pct"] = round(
                (1.0 - m.average_power_w / baseline_power) * 100.0, 3
            )

    results_dir = Path(task.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plan_path = results_dir / f"{task.scenario_id}.json"
    document = {
        "scenario_id": task.scenario_id,
        "batch_id": task.batch_id,
        "cluster_id": task.cluster_id,
        "service_id": task.service_id,
        "scenario_seed": task.seed,
        "fidelity": fidelity,
        "baseline": {"avg_power_w": baseline_power, "note": baseline_note},
        "planner_output": output.model_dump(mode="json"),
    }
    plan_path.write_text(json.dumps(document, indent=2, sort_keys=True))
    summary["plan_json_path"] = str(plan_path)
    return summary


class BatchRunner:
    def __init__(
        self,
        config: LabConfig,
        config_hash: str,
        config_text: str,
        root: str | Path = ".",
    ) -> None:
        self.config = config
        self.config_hash = config_hash
        self.config_text = config_text
        self.root = Path(root)

    # -- generation (M1 + M2) ------------------------------------------------ #

    def generate(self) -> tuple[list[ClusterSummary], list[ServiceSummary]]:
        cfg = self.config
        seed = cfg.lab.seed
        clusters = [
            generate_cluster(
                cfg.cluster_generator, i, derive_seed(seed, "cluster", i),
                self.root / cfg.store.clusters_dir, self.root, self.config_hash,
            )
            for i in range(cfg.cluster_generator.num_clusters)
        ]
        services = [
            generate_service(
                cfg.slo_generator, j, derive_seed(seed, "slo", j),
                self.root / cfg.store.services_dir, self.config_hash,
            )
            for j in range(cfg.slo_generator.num_specs)
        ]
        return clusters, services

    # -- matrix -------------------------------------------------------------- #

    def _pairs(self) -> list[tuple[int, int]]:
        cfg = self.config
        all_pairs = [
            (i, j)
            for i in range(cfg.cluster_generator.num_clusters)
            for j in range(cfg.slo_generator.num_specs)
        ]
        if cfg.pairing.mode == "cross":
            return all_pairs
        assert cfg.pairing.num_pairs is not None
        rng = rng_for(derive_seed(cfg.lab.seed, "pairing"))
        take = min(cfg.pairing.num_pairs, len(all_pairs))
        picked = rng.choice(len(all_pairs), size=take, replace=False)
        return sorted(all_pairs[int(k)] for k in picked)

    # -- execution ------------------------------------------------------------ #

    def run(
        self,
        store: ResultStore,
        *,
        predictor_factory: Callable[..., Predictor] | None = None,
        quiet: bool = False,
    ) -> dict[str, Any]:
        """Generate, register and execute the whole batch. Resumable: DONE
        scenarios are never re-run (FR-B4). Returns the batch summary."""
        cfg = self.config
        batch_id = cfg.lab.batch_name
        clusters, services = self.generate()

        store.register_batch(batch_id, self.config_text, self.config_hash, cfg.lab.seed)
        for c in clusters:
            store.upsert_cluster(c)
        for s in services:
            store.upsert_service(s)

        by_cluster = {c.cluster_id: c for c in clusters}
        by_service = {s.service_id: s for s in services}
        for i, j in self._pairs():
            store.register_scenario(
                f"sc{i:04d}x{j:04d}", batch_id, f"c{i:04d}", f"s{j:04d}",
                derive_seed(cfg.lab.seed, "scenario", i, j),
            )

        results_dir = self.root / cfg.store.results_dir / batch_id
        error_count = 0
        # Two passes: the main run, then one retry pass over fresh ERRORs
        # (FR-B5: automatic retry, exactly once per invocation).
        for _pass in range(2):
            pending = store.pending_scenarios(batch_id)
            if not pending:
                break
            tasks = []
            for row in pending:
                store.mark_running(row["scenario_id"])
                tasks.append(
                    ScenarioTask(
                        scenario_id=row["scenario_id"],
                        batch_id=batch_id,
                        cluster_id=row["cluster_id"],
                        service_id=row["service_id"],
                        cluster_yaml=str(by_cluster[row["cluster_id"]].yaml_path),
                        service_yaml=str(by_service[row["service_id"]].yaml_path),
                        seed=row["seed"],
                        num_requests=cfg.runner.num_requests,
                        enable_pd=cfg.runner.enable_pd,
                        predictor_kind=FIDELITY_SURROGATE,
                        root=str(self.root),
                        results_dir=str(results_dir),
                        config_hash=self.config_hash,
                    )
                )
            error_count = self._execute(
                tasks, store, predictor_factory=predictor_factory, quiet=quiet
            )

        summary = store.batch_summary(batch_id)
        summary["errors"] = error_count
        store.finish_batch(batch_id, "DONE" if error_count == 0 else "DONE_WITH_ERRORS")
        if not quiet:
            print(self._render_summary(batch_id, summary))
        return summary

    def _execute(
        self,
        tasks: list[ScenarioTask],
        store: ResultStore,
        *,
        predictor_factory: Callable[..., Predictor] | None,
        quiet: bool,
    ) -> int:
        total = len(tasks)
        done = 0
        errors = 0

        def _absorb(record: dict[str, Any]) -> None:
            nonlocal done, errors
            done += 1
            if record["ok"]:
                store.insert_result(
                    record["scenario_id"], record["elapsed_s"],
                    {k: v for k, v in record.items() if k not in ("ok", "elapsed_s")},
                )
            else:
                errors += 1
                store.mark_error(
                    record["scenario_id"], record["error"], record["elapsed_s"]
                )
            if not quiet:
                state = "ok" if record["ok"] else "ERROR"
                print(
                    f"  [{done}/{total}] {record['scenario_id']} {state} "
                    f"({record['elapsed_s']:.1f}s)"
                )

        # An injected factory cannot cross a process boundary; run inline.
        if predictor_factory is not None or self.config.runner.workers == 1:
            for task in tasks:
                _absorb(run_scenario(task, predictor_factory))
            return errors

        with ProcessPoolExecutor(max_workers=self.config.runner.workers) as pool:
            futures = [pool.submit(run_scenario, task) for task in tasks]
            for future in as_completed(futures):
                _absorb(future.result())
        return errors

    @staticmethod
    def _render_summary(batch_id: str, summary: dict[str, Any]) -> str:
        parts = [f"[batch {batch_id}] {summary['done']} done"]
        rate = summary.get("feasible_rate")
        if rate is not None:
            parts.append(f"feasible {rate * 100:.0f}%")
        median = summary.get("median_power_saving_pct")
        if median is not None:
            parts.append(f"median power saving {median:.1f}%")
        parts.append(f"errors {summary.get('errors', 0)}")
        return " · ".join(parts)


def scenario_task_fields() -> list[str]:
    """Stable field list, used by tests to detect accidental contract drift."""
    return list(ScenarioTask.__dataclass_fields__)


def task_as_dict(task: ScenarioTask) -> dict[str, Any]:
    return asdict(task)
