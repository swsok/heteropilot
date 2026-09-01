"""Tier-3 verification tests (DESIGN §7.4/§7.5, FR-B8).

The "simulator" here is a scaled surrogate with a known bias, so the error
statistics the pass computes have exact expected values - no ASTRA-Sim needed.
"""

from __future__ import annotations

from pathlib import Path

from planner.predictor import Predictor, SimResult
from planner.util.workload import WorkloadTrace
from scenariolab.config import load_lab_config
from scenariolab.runner.batch import BatchRunner
from scenariolab.runner.tiers import SurrogatePredictor
from scenariolab.runner.verify import (
    VerifyTask,
    cluster_size_bucket,
    run_verification,
    run_verification_pass,
    stratified_sample,
)
from scenariolab.store.db import ResultStore
from tests.scenariolab.conftest import ROOT, write_lab_config


class ScaledSimPredictor(Predictor):
    """Surrogate physics with a fixed, known bias: latency x1.25, power x1.1.

    Fast-path error relative to this 'simulator' is then exactly
    (sim - fast) / sim = 1 - 1/1.25 = 20% on latency and ~9.09% on power.
    """

    def __init__(self, trace: WorkloadTrace) -> None:
        self.inner = SurrogatePredictor(trace)
        self.calls = 0

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        self.calls += 1
        result = self.inner.predict(candidate, spec, cluster, islands, profiles)
        if not result.ok or result.metrics is None:
            return result
        m = result.metrics
        scaled = m.model_copy(update={
            "p50_ttft_ms": m.p50_ttft_ms * 1.25,
            "p95_ttft_ms": m.p95_ttft_ms * 1.25,
            "p99_ttft_ms": m.p99_ttft_ms * 1.25,
            "p50_tpot_ms": m.p50_tpot_ms * 1.25,
            "p95_tpot_ms": m.p95_tpot_ms * 1.25,
            "p99_tpot_ms": m.p99_tpot_ms * 1.25,
            "average_power_w": (
                None if m.average_power_w is None else m.average_power_w * 1.1
            ),
            "total_energy_j": (
                None if m.total_energy_j is None else m.total_energy_j * 1.1
            ),
        })
        return SimResult(result.candidate_id, result.outcome, metrics=scaled)


class ScaledFactory:
    def __init__(self) -> None:
        self.predictors: list[ScaledSimPredictor] = []

    def __call__(self, trace: WorkloadTrace) -> Predictor:
        predictor = ScaledSimPredictor(trace)
        self.predictors.append(predictor)
        return predictor

    @property
    def total_calls(self) -> int:
        return sum(p.calls for p in self.predictors)


def _pool_row(i: int, *, feasible: bool, accels: int, npu: bool) -> dict:
    return {
        "scenario_id": f"sc{i:04d}x0000",
        "feasible": feasible,
        "num_accels": accels,
        "has_npu": npu,
    }


def test_stratified_sample_deterministic_and_covering() -> None:
    pool = [
        _pool_row(i, feasible=(i % 2 == 0), accels=(i % 3) * 3 + 1, npu=(i % 4 == 0))
        for i in range(40)
    ]
    kwargs = {
        "master_seed": 777,
        "fraction": 0.25,
        "min_count": 4,
        "stratify_by": ["feasible", "cluster_size_bucket", "has_npu"],
    }
    first = stratified_sample(pool, **kwargs)
    second = stratified_sample(pool, **kwargs)
    assert first == second
    assert len(first) == 10  # round(40 * 0.25)
    assert len(set(first)) == 10
    # Round-robin over strata: with 10 picks and fewer strata than picks,
    # every stratum contributes at least one sample.
    def key(row):
        return (row["feasible"], cluster_size_bucket(row["num_accels"]), row["has_npu"])
    by_id = {row["scenario_id"]: row for row in pool}
    sampled_strata = {key(by_id[sid]) for sid in first}
    assert sampled_strata == {key(r) for r in pool}


def test_stratified_sample_min_count_and_empty() -> None:
    pool = [_pool_row(i, feasible=True, accels=2, npu=False) for i in range(6)]
    assert len(stratified_sample(
        pool, master_seed=1, fraction=0.0, min_count=3,
        stratify_by=["feasible"],
    )) == 3
    assert stratified_sample(
        [], master_seed=1, fraction=1.0, min_count=5, stratify_by=["feasible"],
    ) == []


def test_verification_pass_records_known_error(tmp_path: Path) -> None:
    path = write_lab_config(tmp_path)
    config, digest = load_lab_config(path, ROOT)
    runner = BatchRunner(config, digest, path.read_text(), ROOT)
    factory = ScaledFactory()
    with ResultStore(config.store.db_path) as store:
        runner.run(store, quiet=True)
        summary = run_verification_pass(
            config, store,
            root=ROOT,
            fraction=1.0,
            min_count=0,
            predictor_factory=factory,
            quiet=True,
        )
        assert summary["sampled"] == 6
        assert summary["errors"] == 0
        # Latency bias is exactly 20%, power exactly 1 - 1/1.1 = 9.091%.
        assert abs(summary["err_ttft_pct_p50"] - 20.0) < 0.01
        assert abs(summary["err_tpot_pct_p50"] - 20.0) < 0.01
        assert abs(summary["err_power_pct_p50"] - 9.091) < 0.01

        rows = list(store._conn.execute("SELECT * FROM verifications"))
        assert len(rows) == summary["verified"] > 0
        statuses = store.scenario_counts("lab-test")
        assert statuses.get("VERIFIED", 0) == summary["verified"]
        # Envelope cache now holds genuine (here: fake-sim) results for reuse.
        envelope_files = list(Path(config.store.envelope_dir).glob("*.json"))
        assert envelope_files


def test_verification_skips_scenario_without_plans(tmp_path: Path) -> None:
    """A bound-pruned infeasible scenario has neither recommended nor
    closest_plan; verification must skip it with a reason, not crash."""
    from scenariolab.runner.batch import run_scenario
    from tests.scenariolab.test_batch import _direct_task

    record = run_scenario(_direct_task(tmp_path, ttft_ms=10_000, tpot_ms=1.0))
    assert record["ok"]
    task = VerifyTask(
        scenario_id="scX",
        cluster_yaml="unused",
        service_yaml="unused",
        seed=99,
        num_requests=20,
        root=str(ROOT),
        plan_json_path=record["plan_json_path"],
        envelope_dir=None,
        sim_timeout_s=10.0,
        alternatives_k=3,
        fast_feasible=False,
        fast_p99_ttft_ms=None,
        fast_p99_tpot_ms=None,
        fast_avg_power_w=None,
    )
    result = run_verification(task, predictor_factory=lambda trace: None)
    assert result["ok"]
    assert "no recommended or closest plan" in result["skipped"]


def test_selection_flip_fields_populated(tmp_path: Path) -> None:
    path = write_lab_config(tmp_path)
    config, digest = load_lab_config(path, ROOT)
    runner = BatchRunner(config, digest, path.read_text(), ROOT)
    with ResultStore(config.store.db_path) as store:
        runner.run(store, quiet=True)
        run_verification_pass(
            config, store, root=ROOT, fraction=1.0, min_count=0,
            predictor_factory=ScaledFactory(), quiet=True,
        )
        rows = [dict(r) for r in store._conn.execute(
            "SELECT v.*, r.feasible FROM verifications v "
            "JOIN results r ON r.scenario_id = v.scenario_id"
        )]
        assert rows
        for row in rows:
            # A uniform bias rescales every candidate identically, so the
            # ranking cannot flip; the field must still be filled for
            # feasible scenarios (0/None semantics per §7.4).
            if row["feasible"]:
                assert row["selection_flipped"] == 0
                assert row["regret_energy_pct"] == 0.0
            assert row["feasibility_flipped"] in (0, 1)
