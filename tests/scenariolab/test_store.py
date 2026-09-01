"""M5 ResultStore tests (DESIGN §8.3)."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from scenariolab.generator.cluster_gen import ClusterSummary
from scenariolab.generator.slo_gen import ServiceSummary
from scenariolab.store.db import ResultStore, StoreError


def _cluster(i: int) -> ClusterSummary:
    return ClusterSummary(
        cluster_id=f"c{i:04d}", seed=i, yaml_path=Path(f"c{i:04d}.yaml"),
        num_nodes=2, num_accels=4, num_free_accels=3,
        classes=["a5000"], num_islands=2, has_npu=(i % 2 == 0),
    )


def _service(j: int) -> ServiceSummary:
    return ServiceSummary(
        service_id=f"s{j:04d}", seed=j, yaml_path=Path(f"s{j:04d}.yaml"),
        model="meta-llama/Llama-3.1-8B", rps=1.0 + j, input_p50=256, output_p50=64,
        ttft_p99_ms=1000.0, tpot_p99_ms=50.0, power_cap_w=2000.0,
    )


def _result(feasible: bool = True, saving: float | None = 10.0) -> dict:
    return {
        "feasible": feasible,
        "fidelity": "surrogate",
        "plan_json_path": "x.json",
        "p99_ttft_ms": 100.0,
        "p99_tpot_ms": 20.0,
        "avg_power_w": 200.0,
        "peak_power_w": 300.0,
        "tokens_per_joule": 5.0,
        "slo_goodput": 3.0,
        "active_devices": 2,
        "baseline_power_w": 250.0,
        "power_saving_pct": saving,
        "baseline_note": "ok",
        "violated": [],
        "provenance": {"git_commit": "abc"},
    }


def _seed_matrix(store: ResultStore, n_clusters: int = 2, n_services: int = 2) -> None:
    store.register_batch("b1", "yaml", "hash", 1)
    for i in range(n_clusters):
        store.upsert_cluster(_cluster(i))
    for j in range(n_services):
        store.upsert_service(_service(j))
    for i in range(n_clusters):
        for j in range(n_services):
            store.register_scenario(
                f"sc{i:04d}x{j:04d}", "b1", f"c{i:04d}", f"s{j:04d}", i * 10 + j
            )


def test_roundtrip(tmp_path: Path) -> None:
    with ResultStore(tmp_path / "t.sqlite") as store:
        _seed_matrix(store)
        store.insert_result("sc0000x0000", 1.5, _result())
        rows = store.query_results("b1")
        assert len(rows) == 1
        row = rows[0]
        assert row["feasible"] == 1
        assert row["fidelity"] == "surrogate"
        assert row["power_saving_pct"] == 10.0
        counts = store.scenario_counts("b1")
        assert counts["DONE_FEASIBLE"] == 1
        assert counts["PENDING"] == 3


def test_pending_includes_error_and_running(tmp_path: Path) -> None:
    with ResultStore(tmp_path / "t.sqlite") as store:
        _seed_matrix(store)
        store.mark_running("sc0000x0000")
        store.mark_error("sc0000x0001", "boom", 0.1)
        store.insert_result("sc0001x0000", 1.0, _result())
        pending = [row["scenario_id"] for row in store.pending_scenarios("b1")]
        assert pending == ["sc0000x0000", "sc0000x0001", "sc0001x0001"]


def test_filters_sort_pagination(tmp_path: Path) -> None:
    with ResultStore(tmp_path / "t.sqlite") as store:
        _seed_matrix(store, n_clusters=3, n_services=3)
        k = 0
        for i in range(3):
            for j in range(3):
                k += 1
                store.insert_result(
                    f"sc{i:04d}x{j:04d}", 0.1,
                    _result(feasible=(k % 2 == 0), saving=float(k)),
                )
        feasible_rows = store.query_results("b1", feasible=True)
        assert all(r["feasible"] == 1 for r in feasible_rows)
        assert len(feasible_rows) == 4
        npu_rows = store.query_results("b1", has_npu=True)
        assert {r["cluster_id"] for r in npu_rows} == {"c0000", "c0002"}
        high = store.query_results("b1", min_saving_pct=7.0)
        assert len(high) == 3
        ordered = store.query_results("b1", order_by="power_saving_pct", descending=True)
        savings = [r["power_saving_pct"] for r in ordered]
        assert savings == sorted(savings, reverse=True)
        page1 = store.query_results("b1", page=1, page_size=4)
        page2 = store.query_results("b1", page=2, page_size=4)
        page3 = store.query_results("b1", page=3, page_size=4)
        ids = [r["scenario_id"] for r in page1 + page2 + page3]
        assert len(ids) == 9 and len(set(ids)) == 9
        with pytest.raises(StoreError):
            store.query_results("b1", order_by="1; DROP TABLE results")


def test_schema_version_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "t.sqlite"
    ResultStore(db).close()
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="schema version"):
        ResultStore(db)


def _parallel_insert(args: tuple[str, int]) -> None:
    db, worker = args
    with ResultStore(db) as store:
        for k in range(10):
            sid = f"sc{worker:04d}x{k:04d}"
            store.insert_result(sid, 0.1, _result(saving=float(worker * 10 + k)))


def test_parallel_writes_lossless(tmp_path: Path) -> None:
    db = str(tmp_path / "t.sqlite")
    with ResultStore(db) as store:
        store.register_batch("b1", "y", "h", 1)
        for w in range(8):
            store.upsert_cluster(_cluster(w))
            store.upsert_service(_service(w))
            for k in range(10):
                store.register_scenario(
                    f"sc{w:04d}x{k:04d}", "b1", f"c{w:04d}", f"s{w:04d}", w * 100 + k
                )
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(_parallel_insert, [(db, w) for w in range(8)]))
    with ResultStore(db) as store:
        rows = store.query_results("b1", page_size=200)
        assert len(rows) == 80
        savings = sorted(r["power_saving_pct"] for r in rows)
        assert savings == [float(v) for v in range(80)]
