"""M3 BatchRunner tests (DESIGN §6.5): mini E2E, resume, error isolation,
worker-count determinism, baseline semantics, golden batch."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scenariolab.config import load_lab_config
from scenariolab.generator.sampling import derive_seed
from scenariolab.runner.batch import BatchRunner, ScenarioTask, run_scenario
from scenariolab.store.db import DONE_STATES, ResultStore
from tests.scenariolab.conftest import ROOT, ExplodingFactory, write_lab_config

GOLDEN = Path(__file__).parent / "golden"


def _make_runner(tmp_path: Path, **overrides):
    path = write_lab_config(tmp_path, **overrides)
    config, digest = load_lab_config(path, ROOT)
    return BatchRunner(config, digest, path.read_text(), ROOT), config


def _normalize(dump: dict) -> dict:
    """Strip run-location specifics so dumps from different roots compare."""
    for row in dump["results"]:
        row["plan_json_path"] = Path(row["plan_json_path"]).name
    return dump


def test_mini_batch_e2e(tmp_path: Path, capsys) -> None:
    runner, config = _make_runner(tmp_path)
    with ResultStore(config.store.db_path) as store:
        summary = runner.run(store)
        assert summary["done"] == 6
        assert summary["errors"] == 0
        counts = store.scenario_counts("lab-test")
        assert sum(counts.get(s, 0) for s in DONE_STATES) == 6
        rows = store.query_results("lab-test")
        assert len(rows) == 6
        for row in rows:
            assert row["fidelity"] == "surrogate"
            plan = json.loads((Path(row["plan_json_path"])).read_text())
            assert plan["scenario_id"] == row["scenario_id"]
            assert plan["planner_output"]["provenance"]["scenariolab"]["fidelity"] == (
                "surrogate"
            )
    out = capsys.readouterr().out
    assert "[batch lab-test] 6 done" in out


def test_resume_skips_done(tmp_path: Path) -> None:
    runner, config = _make_runner(tmp_path)
    with ResultStore(config.store.db_path) as store:
        runner.run(store, quiet=True)
        before = _normalize(store.dump_for_golden("lab-test"))

        # Simulate an interrupted batch: two scenarios lose their results.
        redo = ["sc0000x0001", "sc0001x0002"]
        for sid in redo:
            store._conn.execute("DELETE FROM results WHERE scenario_id=?", (sid,))
            store._conn.execute(
                "UPDATE scenarios SET status='PENDING' WHERE scenario_id=?", (sid,)
            )
        store._conn.commit()

        factory = ExplodingFactory()
        runner.run(store, predictor_factory=factory, quiet=True)
        # Only the two missing scenarios ran again (FR-B4: DONE is untouched).
        assert len(factory.calls) == 2
        after = _normalize(store.dump_for_golden("lab-test"))
        assert after == before


def test_error_isolation_and_retry(tmp_path: Path) -> None:
    runner, config = _make_runner(tmp_path)
    bad_seed = derive_seed(777, "scenario", 0, 1)
    factory = ExplodingFactory(fail_seeds={bad_seed})
    with ResultStore(config.store.db_path) as store:
        summary = runner.run(store, predictor_factory=factory, quiet=True)
        counts = store.scenario_counts("lab-test")
        assert counts.get("ERROR") == 1
        assert sum(counts.get(s, 0) for s in DONE_STATES) == 5
        assert summary["errors"] == 1
        # The failed scenario was attempted twice (FR-B5: one automatic retry).
        assert factory.calls.count(bad_seed) == 2
        row = store._conn.execute(
            "SELECT error_text, attempts FROM scenarios WHERE scenario_id='sc0000x0001'"
        ).fetchone()
        assert "injected predictor failure" in row["error_text"]
        assert row["attempts"] == 2


def test_worker_count_does_not_change_results(tmp_path: Path) -> None:
    """FR-B3: workers=1 and workers=2 produce identical DB content."""
    dumps = []
    for workers in (1, 2):
        sub = tmp_path / f"w{workers}"
        sub.mkdir()
        runner, config = _make_runner(sub, **{"runner.workers": workers})
        with ResultStore(config.store.db_path) as store:
            runner.run(store, quiet=True)
            dumps.append(_normalize(store.dump_for_golden("lab-test")))
    assert dumps[0] == dumps[1]


def test_random_pairing_is_deterministic_and_bounded(tmp_path: Path) -> None:
    runner, _ = _make_runner(
        tmp_path, pairing={"mode": "random", "num_pairs": 4, "max_scenarios": 1500}
    )
    pairs_a = runner._pairs()
    pairs_b = runner._pairs()
    assert pairs_a == pairs_b
    assert len(pairs_a) == 4
    assert len(set(pairs_a)) == 4


def _direct_task(tmp_path: Path, ttft_ms: float, tpot_ms: float) -> ScenarioTask:
    """One handcrafted scenario on a generated single-A5000 cluster."""
    from scenariolab.config import ClusterGeneratorConfig
    from scenariolab.generator.cluster_gen import generate_cluster

    gen = ClusterGeneratorConfig.model_validate({
        "num_clusters": 1,
        "nodes_per_cluster": {"min": 1, "max": 1},
        "accelerators_per_node": {"min": 1, "max": 1},
        "accelerator_pool": ["a5000"],
        "internode_link_pool": ["ib_400g"],
        "free_ratio": {"min": 1.0, "max": 1.0},
    })
    summary = generate_cluster(gen, 0, 1234, tmp_path / "cl", ROOT, "h")

    service = {
        "service": {"model": "meta-llama/Llama-3.1-8B", "dtype": "bfloat16"},
        "traffic": {
            "arrival_rate_rps": 1.0,
            "input_tokens": {"p50": 256, "p95": 512},
            "output_tokens": {"p50": 64, "p95": 128},
        },
        "slo": {
            "ttft": {"percentile": 99, "max_ms": ttft_ms},
            "tpot": {"percentile": 99, "max_ms": tpot_ms},
            "max_cluster_power_w": 3000,
        },
        "objective": {"primary": "minimize_energy"},
    }
    service_path = tmp_path / f"svc-{ttft_ms}-{tpot_ms}.yaml"
    service_path.write_text(yaml.safe_dump(service))

    return ScenarioTask(
        scenario_id="scX", batch_id="bX", cluster_id="c0000", service_id="sX",
        cluster_yaml=str(summary.yaml_path), service_yaml=str(service_path),
        seed=99, num_requests=20, enable_pd=False, predictor_kind="surrogate",
        root=str(ROOT), results_dir=str(tmp_path / "res"), config_hash="h",
    )


def test_baseline_feasible_records_saving(tmp_path: Path) -> None:
    record = run_scenario(_direct_task(tmp_path, ttft_ms=10_000, tpot_ms=200))
    assert record["ok"], record.get("error")
    assert record["feasible"] is True
    assert record["baseline_note"] == "ok"
    assert record["baseline_power_w"] is not None
    # One island, so recommended == baseline family: saving is defined (>= 0).
    assert record["power_saving_pct"] is not None


def test_baseline_infeasible_yields_null_saving(tmp_path: Path) -> None:
    """FR-B7: an SLO nothing can meet -> infeasible diagnosis, NULL saving.

    tpot=1ms is below even the optimistic roofline, so every candidate dies at
    the sound stage-5 bound: nothing is evaluated, and the diagnosis is the
    rejection summary rather than violated_constraints."""
    record = run_scenario(_direct_task(tmp_path, ttft_ms=10_000, tpot_ms=1.0))
    assert record["ok"], record.get("error")
    assert record["feasible"] is False
    assert record["power_saving_pct"] is None
    assert record["baseline_note"].startswith("baseline violates")
    document = json.loads(Path(record["plan_json_path"]).read_text())
    output = document["planner_output"]
    assert output["feasible"] is False
    assert output["rejected_summary"].get("analytical_lower_bound", 0) > 0
    assert output["evaluated_candidates"] == 0


def test_slo_violation_after_prediction_diagnosed(tmp_path: Path) -> None:
    """A TPOT between the optimistic bound (~21ms on one A5000) and the
    surrogate prediction (~25ms) survives pruning but fails feasibility, so
    the FR-S6 diagnosis branch (violated_constraints + closest_plan) fires."""
    record = run_scenario(_direct_task(tmp_path, ttft_ms=10_000, tpot_ms=22.0))
    assert record["ok"], record.get("error")
    assert record["feasible"] is False
    assert record["power_saving_pct"] is None
    document = json.loads(Path(record["plan_json_path"]).read_text())
    output = document["planner_output"]
    assert output["violated_constraints"]
    assert output["closest_plan"] is not None
    assert any(
        v["metric"] == "p99_tpot_ms" for v in output["violated_constraints"]
    )


def test_reproducibility_byte_identical_modulo_timestamp(tmp_path: Path) -> None:
    task = _direct_task(tmp_path, ttft_ms=10_000, tpot_ms=200)
    docs = []
    for run_dir in ("r1", "r2"):
        record = run_scenario(
            ScenarioTask(**{**task.__dict__, "results_dir": str(tmp_path / run_dir)})
        )
        assert record["ok"], record.get("error")
        doc = json.loads(Path(record["plan_json_path"]).read_text())
        doc["planner_output"]["provenance"].pop("timestamp")
        docs.append(doc)
    assert docs[0] == docs[1]


def test_envelope_reuse_across_scenarios(tmp_path: Path) -> None:
    """Tier-1 end to end: a second scenario with the same placement/bucket is
    served from the envelope written by the first - zero predictor calls, and
    the result is relabelled fidelity=envelope (FR-T1/FR-T3)."""
    from planner.predictor import Predictor
    from scenariolab.runner.tiers import SurrogatePredictor

    calls: list[str] = []

    class CountingPredictor(Predictor):
        def __init__(self, trace) -> None:
            self.inner = SurrogatePredictor(trace)

        def predict(self, candidate, spec, cluster, islands, profiles):
            calls.append(candidate.id)
            return self.inner.predict(candidate, spec, cluster, islands, profiles)

    base = _direct_task(tmp_path, ttft_ms=10_000, tpot_ms=200)
    # full_sim=top_k marks the run as simulation-fidelity, which is what makes
    # the envelope writable; the counting predictor stands in for the sim.
    task1 = ScenarioTask(**{
        **base.__dict__, "full_sim": "top_k",
        "envelope_dir": str(tmp_path / "env"),
    })
    record1 = run_scenario(task1, predictor_factory=CountingPredictor)
    assert record1["ok"], record1.get("error")
    assert record1["fidelity"] == "sim"
    first_calls = len(calls)
    assert first_calls > 0

    task2 = ScenarioTask(**{
        **task1.__dict__, "scenario_id": "scY",
        "results_dir": str(tmp_path / "res2"),
    })
    record2 = run_scenario(task2, predictor_factory=CountingPredictor)
    assert record2["ok"], record2.get("error")
    assert len(calls) == first_calls  # everything, baseline included, was cached
    assert record2["fidelity"] == "envelope"
    assert record2["avg_power_w"] == record1["avg_power_w"]


def test_oracle_agreement_tiered_path(tmp_path: Path) -> None:
    """DESIGN §7.5 core regression: the Tier-3 path (full_sim=top_k, no K cap,
    same predictor) must recommend exactly what the pruning-disabled oracle
    recommends. If they disagree, the tier plumbing lost a candidate."""
    import tempfile

    from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
    from planner.optimizer import exhaustive
    from planner.spec import load_service_spec
    from planner.util.workload import generate_trace
    from scenariolab.config import ClusterGeneratorConfig
    from scenariolab.generator.cluster_gen import generate_cluster
    from scenariolab.runner.tiers import SurrogatePredictor

    gen = ClusterGeneratorConfig.model_validate({
        "num_clusters": 1,
        "nodes_per_cluster": {"min": 2, "max": 2},
        "accelerators_per_node": {"min": 2, "max": 2},
        "accelerator_pool": ["a5000", "furiosa_rngd_card"],
        "internode_link_pool": ["ib_400g"],
        "free_ratio": {"min": 1.0, "max": 1.0},
    })
    summary = generate_cluster(gen, 0, 777, tmp_path / "cl", ROOT, "h")

    base = _direct_task(tmp_path, ttft_ms=10_000, tpot_ms=200)
    task = ScenarioTask(**{
        **base.__dict__,
        "cluster_yaml": str(summary.yaml_path),
        "full_sim": "top_k",
        "top_k": None,  # no stage-6 cap: simulate every pruning survivor
    })
    record = run_scenario(task, predictor_factory=SurrogatePredictor)
    assert record["ok"], record.get("error")
    document = json.loads(Path(record["plan_json_path"]).read_text())
    tiered = document["planner_output"]

    spec = load_service_spec(task.service_yaml)
    cluster = load_cluster_spec(task.cluster_yaml)
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    with tempfile.TemporaryDirectory() as tmp:
        trace = generate_trace(
            spec, Path(tmp) / "w.jsonl", num_requests=task.num_requests, seed=task.seed
        )
        oracle = exhaustive.oracle(
            spec, cluster, islands, profiles, SurrogatePredictor(trace), max_workers=1
        )

    assert oracle.feasible == tiered["feasible"]
    assert oracle.recommended is not None
    assert tiered["recommended"] is not None
    assert (
        tiered["recommended"]["plan"]["candidate"]["id"]
        == oracle.recommended.plan.candidate.id
    )
    assert tiered["recommended"]["value"] == oracle.recommended.value


def test_golden_batch(tmp_path: Path) -> None:
    """Frozen DB image of the fixed-seed test batch (DESIGN §11).

    Regenerate deliberately with:
      pytest tests/scenariolab/test_batch.py::test_golden_batch --golden-update
    """
    runner, config = _make_runner(tmp_path)
    with ResultStore(config.store.db_path) as store:
        runner.run(store, quiet=True)
        dump = _normalize(store.dump_for_golden("lab-test"))
    golden_path = GOLDEN / "lab-test-db.json"
    if not golden_path.exists():  # first run: write and fail loudly
        GOLDEN.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(dump, indent=2, sort_keys=True))
        raise AssertionError(
            f"golden file created at {golden_path}; inspect and commit it"
        )
    assert dump == json.loads(golden_path.read_text())
