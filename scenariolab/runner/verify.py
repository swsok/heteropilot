"""Tier-3 verification pass (DESIGN §7.4, FR-B8): stratified full-sim
cross-checks of fast-path results.

A verification sample re-evaluates the recommended plan (and up to K Pareto
alternatives) with the real LLMServingSim, then records how wrong the fast
path was: per-metric error, feasibility flips, and whether re-ranking under
simulation would have picked a different plan (selection flip + energy
regret). Flip detection is limited to {recommended + simmed alternatives} -
simulating the whole candidate space is exactly what the tiers avoid, and the
oracle-agreement tests cover that path separately.

Simulated results are written into the shared envelope cache, so every
verified scenario makes later batches cheaper and more accurate (Tier-1).
"""

from __future__ import annotations

import json
import tempfile
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import feasibility
from planner.plan import DeploymentPlan
from planner.predictor import Predictor
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.util.percentile import percentile
from planner.util.workload import generate_trace
from scenariolab.config import LabConfig
from scenariolab.generator.sampling import derive_seed, rng_for
from scenariolab.runner.tiers import SharedEnvelope, make_predictor
from scenariolab.store.db import ResultStore


def cluster_size_bucket(num_accels: int) -> str:
    if num_accels <= 2:
        return "small"
    if num_accels <= 6:
        return "mid"
    return "large"


def stratified_sample(
    pool: list[dict[str, Any]],
    *,
    master_seed: int,
    fraction: float,
    min_count: int,
    stratify_by: list[str],
) -> list[str]:
    """Deterministic stratified sample of scenario ids (§7.4).

    Strata are the cross product of the requested axes; picks rotate across
    strata so every stratum is represented before any is drawn twice.
    """
    if not pool:
        return []
    target = min(len(pool), max(min_count, round(len(pool) * fraction)))
    if target == 0:
        return []

    def stratum_key(row: dict[str, Any]) -> tuple:
        parts: list[Any] = []
        for axis in stratify_by:
            if axis == "feasible":
                parts.append(bool(row["feasible"]))
            elif axis == "cluster_size_bucket":
                parts.append(cluster_size_bucket(int(row["num_accels"])))
            elif axis == "has_npu":
                parts.append(bool(row["has_npu"]))
            else:
                raise ValueError(f"unknown stratify_by axis '{axis}'")
        return tuple(parts)

    strata: dict[tuple, list[str]] = {}
    for row in sorted(pool, key=lambda r: r["scenario_id"]):
        strata.setdefault(stratum_key(row), []).append(row["scenario_id"])

    rng = rng_for(derive_seed(master_seed, "verify"))
    queues = [
        list(rng.permutation(ids))
        for _, ids in sorted(strata.items(), key=lambda kv: repr(kv[0]))
    ]
    picked: list[str] = []
    while len(picked) < target:
        progressed = False
        for queue in queues:
            if queue and len(picked) < target:
                picked.append(str(queue.pop(0)))
                progressed = True
        if not progressed:  # pragma: no cover - target <= len(pool) prevents this
            break
    return sorted(picked)


@dataclass(frozen=True)
class VerifyTask:
    """Everything one verification worker needs; picklable and path-based."""

    scenario_id: str
    cluster_yaml: str
    service_yaml: str
    seed: int
    num_requests: int
    root: str
    plan_json_path: str
    envelope_dir: str | None
    sim_timeout_s: float
    alternatives_k: int
    fast_feasible: bool
    fast_p99_ttft_ms: float | None
    fast_p99_tpot_ms: float | None
    fast_avg_power_w: float | None


def _err_pct(sim: float | None, fast: float | None) -> float | None:
    """Fast-path error relative to the simulator, (sim - fast) / sim (§7.4)."""
    if sim is None or fast is None or sim == 0:
        return None
    return round((sim - fast) / sim * 100.0, 3)


def run_verification(
    task: VerifyTask,
    predictor_factory: Callable[..., Predictor] | None = None,
) -> dict[str, Any]:
    """Verify one scenario. Never raises; errors come back in the record."""
    started = time.perf_counter()
    try:
        record = _verify_inner(task, predictor_factory)
        record["ok"] = True
    except Exception:
        record = {"ok": False, "error": traceback.format_exc()}
    record["scenario_id"] = task.scenario_id
    record["elapsed_s"] = round(time.perf_counter() - started, 3)
    return record


def _verify_inner(
    task: VerifyTask,
    predictor_factory: Callable[..., Predictor] | None,
) -> dict[str, Any]:
    document = json.loads(Path(task.plan_json_path).read_text())
    output = document["planner_output"]

    plans: list[DeploymentPlan] = []
    if output.get("recommended"):
        plans.append(DeploymentPlan.model_validate(output["recommended"]["plan"]))
        for alt in output.get("alternatives", [])[: task.alternatives_k]:
            plans.append(DeploymentPlan.model_validate(alt["plan"]))
    elif output.get("closest_plan"):
        plans.append(DeploymentPlan.model_validate(output["closest_plan"]))
    if not plans:
        return {"skipped": "no recommended or closest plan to verify"}

    root = Path(task.root)
    spec = load_service_spec(task.service_yaml)
    cluster = load_cluster_spec(task.cluster_yaml)
    profiles = load_profiles_for(cluster, root)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}

    with tempfile.TemporaryDirectory(prefix=f"slab-verify-{task.scenario_id}-") as tmp:
        trace = generate_trace(
            spec, Path(tmp) / "workload.jsonl",
            num_requests=task.num_requests, seed=task.seed,
        )
        cache = None
        if task.envelope_dir is not None:
            reduction = TopologyGraph(cluster).reduce_for_simulator(
                list(islands.values())
            )
            cache = SharedEnvelope(
                root / task.envelope_dir, spec,
                accelerator_of={
                    i.id: i.accelerator_model for i in islands.values()
                },
                link_bw_gbps=reduction.link_bw_gbps,
                readonly=False,  # verification results ARE simulations
            )
        if predictor_factory is not None:
            predictor = predictor_factory(trace)
        else:
            predictor = make_predictor(
                "sim", trace,
                work_dir=Path(tmp) / "sims", timeout_s=task.sim_timeout_s,
                run_id_prefix=f"v-{task.scenario_id}-",
            )
        sims: dict[str, Any] = {}
        try:
            for plan in plans:
                result = predictor.predict(
                    plan.candidate, spec, cluster, islands, profiles
                )
                sims[plan.candidate.id] = result
                if cache is not None and result.ok:
                    cache.put(plan.candidate, result)
        finally:
            predictor.close()

    recommended = plans[0]
    rec_sim = sims[recommended.candidate.id]
    if not rec_sim.ok or rec_sim.metrics is None:
        return {
            "skipped": (
                f"simulation of the recommended plan failed: "
                f"{rec_sim.outcome.value} {rec_sim.detail}"
            )
        }
    sim_metrics = rec_sim.metrics

    sim_plan = recommended.model_copy(update={"predicted": sim_metrics})
    sim_feasible = feasibility.evaluate(sim_plan, spec).passed

    # Selection flip among the simmed set, by simulated energy (the batch
    # objective is fixed to minimize_energy, FR-S5).
    selection_flipped = None
    regret_energy_pct = None
    if task.fast_feasible and sim_metrics.total_energy_j is not None:
        best_id = recommended.candidate.id
        best_energy = sim_metrics.total_energy_j
        for plan in plans[1:]:
            result = sims[plan.candidate.id]
            if not result.ok or result.metrics is None:
                continue
            if result.metrics.total_energy_j is None:
                continue
            resim = plan.model_copy(update={"predicted": result.metrics})
            if not feasibility.evaluate(resim, spec).passed:
                continue
            if result.metrics.total_energy_j < best_energy:
                best_id = plan.candidate.id
                best_energy = result.metrics.total_energy_j
        selection_flipped = best_id != recommended.candidate.id
        if selection_flipped and best_energy > 0:
            regret_energy_pct = round(
                (sim_metrics.total_energy_j - best_energy) / best_energy * 100.0, 3
            )
        elif not selection_flipped:
            regret_energy_pct = 0.0

    return {
        "sim_p99_ttft_ms": sim_metrics.p99_ttft_ms,
        "sim_p99_tpot_ms": sim_metrics.p99_tpot_ms,
        "sim_avg_power_w": sim_metrics.average_power_w,
        "err_ttft_pct": _err_pct(sim_metrics.p99_ttft_ms, task.fast_p99_ttft_ms),
        "err_tpot_pct": _err_pct(sim_metrics.p99_tpot_ms, task.fast_p99_tpot_ms),
        "err_power_pct": _err_pct(sim_metrics.average_power_w, task.fast_avg_power_w),
        "selection_flipped": selection_flipped,
        "feasibility_flipped": sim_feasible != task.fast_feasible,
        "regret_energy_pct": regret_energy_pct,
    }


def run_verification_pass(
    config: LabConfig,
    store: ResultStore,
    *,
    root: str | Path = ".",
    fraction: float | None = None,
    min_count: int | None = None,
    predictor_factory: Callable[..., Predictor] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Sample, simulate, record (FR-B8). Returns the verification summary."""
    batch_id = config.lab.batch_name
    verification = config.runner.verification
    fraction = verification.fraction if fraction is None else fraction
    min_count = verification.min_count if min_count is None else min_count

    pool = [dict(row) for row in store.verification_pool(batch_id)]
    sampled = stratified_sample(
        pool,
        master_seed=config.lab.seed,
        fraction=fraction,
        min_count=min_count,
        stratify_by=verification.stratify_by,
    )
    by_id = {row["scenario_id"]: row for row in pool}
    tasks = [
        VerifyTask(
            scenario_id=sid,
            cluster_yaml=str(Path(root) / by_id[sid]["cluster_yaml"]),
            service_yaml=str(Path(root) / by_id[sid]["service_yaml"]),
            seed=by_id[sid]["seed"],
            num_requests=config.runner.num_requests,
            root=str(root),
            plan_json_path=by_id[sid]["plan_json_path"],
            envelope_dir=(
                str(config.store.envelope_dir)
                if config.runner.tier_policy.envelope_cache else None
            ),
            sim_timeout_s=config.runner.tier_policy.sim_timeout_s,
            alternatives_k=config.runner.tier_policy.surrogate_top_k,
            fast_feasible=bool(by_id[sid]["feasible"]),
            fast_p99_ttft_ms=by_id[sid]["p99_ttft_ms"],
            fast_p99_tpot_ms=by_id[sid]["p99_tpot_ms"],
            fast_avg_power_w=by_id[sid]["avg_power_w"],
        )
        for sid in sampled
    ]

    records: list[dict[str, Any]] = []

    def _absorb(record: dict[str, Any]) -> None:
        records.append(record)
        if record["ok"] and "skipped" not in record:
            store.insert_verification(
                record["scenario_id"],
                {k: v for k, v in record.items() if k not in ("ok", "elapsed_s")},
            )
        if not quiet:
            state = (
                "ok" if record["ok"] and "skipped" not in record
                else record.get("skipped", "ERROR")
            )
            print(
                f"  [verify {len(records)}/{len(tasks)}] {record['scenario_id']} "
                f"{state} ({record['elapsed_s']:.1f}s)"
            )

    if predictor_factory is not None or verification.sim_workers == 1:
        for task in tasks:
            _absorb(run_verification(task, predictor_factory))
    else:
        workers = min(verification.sim_workers, max(1, len(tasks)))
        with ProcessPoolExecutor(max_workers=workers) as pool_exec:
            futures = [pool_exec.submit(run_verification, task) for task in tasks]
            for future in as_completed(futures):
                _absorb(future.result())

    return _summarize(batch_id, records, quiet=quiet)


def verify_workspace(
    store: Any,
    workspace_id: str,
    *,
    root: str | Path = ".",
    sim_timeout_s: float = 900.0,
    predictor_factory: Callable[..., Predictor] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Cross-check a workspace's PLACED plans with the full simulator (work
    order §9): each plan is re-simulated on the SAME occupancy view it was
    planned against (everything else PLACED marked ALLOCATED). Results go to
    <document>.verify.json - fast-path rows keep their fidelity labels; this
    measures their error, it never overwrites them."""
    from planner.plan import DeploymentPlan
    from scenariolab.runner.tiers import make_predictor
    from scenariolab.runner.workspace import cluster_overlay
    from scenariolab.store.db import devices_from_json

    workspace = store.get_workspace(workspace_id)
    if workspace is None:
        raise KeyError(f"no workspace '{workspace_id}'")
    cluster_row = store.get_cluster(workspace["cluster_id"])
    root = Path(root)
    cluster = load_cluster_spec(root / cluster_row["yaml_path"])
    placed = [
        row for row in store.workspace_placements(workspace_id, include_removed=False)
        if row["status"] == "PLACED"
    ]
    all_devices = store.placed_devices(workspace_id)
    records: list[dict[str, Any]] = []

    for row in placed:
        document = json.loads(Path(row["plan_json_path"]).read_text())
        output = document["planner_output"]
        if not output.get("recommended"):
            continue
        plan = DeploymentPlan.model_validate(output["recommended"]["plan"])
        own = devices_from_json(row["devices_json"])
        overlay = cluster_overlay(cluster, all_devices - own)
        profiles = load_profiles_for(overlay, root)
        islands = {i.id: i for i in detect_islands(overlay, profiles)}
        service_row = store.get_service(row["service_id"])
        spec = load_service_spec(root / service_row["yaml_path"])

        with tempfile.TemporaryDirectory(prefix=f"wsv-{row['placement_id']}-") as tmp:
            trace = generate_trace(
                spec, Path(tmp) / "workload.jsonl",
                num_requests=document.get("num_requests", 100),
                seed=document.get("seed", 42),
            )
            if predictor_factory is not None:
                predictor = predictor_factory(trace)
            else:
                predictor = make_predictor(
                    "sim", trace, work_dir=Path(tmp) / "sims",
                    timeout_s=sim_timeout_s,
                    run_id_prefix=f"wsv-{row['placement_id']}-",
                )
            try:
                result = predictor.predict(
                    plan.candidate, spec, overlay, islands, profiles
                )
            finally:
                predictor.close()

        record: dict[str, Any] = {"placement_id": row["placement_id"]}
        if result.ok and result.metrics is not None:
            metrics = result.metrics
            record.update(
                sim_p99_ttft_ms=metrics.p99_ttft_ms,
                sim_p99_tpot_ms=metrics.p99_tpot_ms,
                sim_avg_power_w=metrics.average_power_w,
                err_ttft_pct=_err_pct(metrics.p99_ttft_ms, row["p99_ttft_ms"]),
                err_tpot_pct=_err_pct(metrics.p99_tpot_ms, row["p99_tpot_ms"]),
                err_power_pct=_err_pct(metrics.average_power_w, row["avg_power_w"]),
                sim_slo_ttft_ok=metrics.p99_ttft_ms <= service_row["ttft_p99_ms"],
                sim_slo_tpot_ok=metrics.p99_tpot_ms <= service_row["tpot_p99_ms"],
            )
        else:
            record["skipped"] = f"{result.outcome.value} {result.detail}"
        out_path = Path(row["plan_json_path"]).with_suffix(".verify.json")
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        records.append(record)
        if not quiet:
            state = record.get("skipped", (
                f"err ttft {record.get('err_ttft_pct')}% "
                f"tpot {record.get('err_tpot_pct')}% "
                f"power {record.get('err_power_pct')}%"
            ))
            print(f"  [verify-ws] {row['placement_id']} {state}")

    summary = {
        "workspace_id": workspace_id,
        "verified": sum(1 for r in records if "skipped" not in r),
        "skipped": sum(1 for r in records if "skipped" in r),
        "records": records,
    }
    if not quiet:
        print(
            f"[verify {workspace_id}] {summary['verified']} verified · "
            f"{summary['skipped']} skipped"
        )
    return summary


def _summarize(
    batch_id: str, records: list[dict[str, Any]], *, quiet: bool
) -> dict[str, Any]:
    done = [r for r in records if r["ok"] and "skipped" not in r]
    errors = [r for r in records if not r["ok"]]
    skipped = [r for r in records if r.get("skipped")]
    summary: dict[str, Any] = {
        "sampled": len(records),
        "verified": len(done),
        "skipped": len(skipped),
        "errors": len(errors),
        "selection_flips": sum(1 for r in done if r.get("selection_flipped")),
        "feasibility_flips": sum(1 for r in done if r.get("feasibility_flipped")),
    }
    for metric in ("err_ttft_pct", "err_tpot_pct", "err_power_pct"):
        values = [abs(r[metric]) for r in done if r.get(metric) is not None]
        summary[f"{metric}_p50"] = round(percentile(values, 50), 3) if values else None
        summary[f"{metric}_p95"] = round(percentile(values, 95), 3) if values else None
    if not quiet:
        print(
            f"[verify {batch_id}] {summary['verified']} verified · "
            f"{summary['selection_flips']} selection flips · "
            f"{summary['feasibility_flips']} feasibility flips · "
            f"|err| p95: ttft {summary['err_ttft_pct_p95']}% "
            f"tpot {summary['err_tpot_pct_p95']}% "
            f"power {summary['err_power_pct_p95']}% · errors {summary['errors']}"
        )
    return summary
