"""M6 Web API tests (DESIGN §9.4): full endpoint round-trips over a fixture
DB built by a real (surrogate-tier) mini batch, plus an OpenAPI snapshot as
the UI's contract golden."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenariolab.api.server import create_app
from scenariolab.config import load_lab_config
from scenariolab.runner.batch import BatchRunner
from scenariolab.runner.verify import run_verification_pass
from scenariolab.store.db import ResultStore
from tests.scenariolab.conftest import ROOT, write_lab_config
from tests.scenariolab.test_verify import ScaledFactory

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory) -> Path:
    """One 2x3 batch, fully verified with the scaled fake simulator."""
    tmp_path = tmp_path_factory.mktemp("apidb")
    path = write_lab_config(tmp_path)
    config, digest = load_lab_config(path, ROOT)
    runner = BatchRunner(config, digest, path.read_text(), ROOT)
    with ResultStore(config.store.db_path) as store:
        runner.run(store, quiet=True)
        run_verification_pass(
            config, store, root=ROOT, fraction=1.0, min_count=0,
            predictor_factory=ScaledFactory(), quiet=True,
        )
    return Path(config.store.db_path)


@pytest.fixture(scope="module")
def client(fixture_db: Path) -> TestClient:
    return TestClient(create_app(fixture_db, root=ROOT))


def test_summary(client: TestClient) -> None:
    data = client.get("/api/summary").json()
    assert data["selected_batch"] == "lab-test"
    batch = data["batches"][0]
    assert batch["done"] == 6
    assert batch["feasible_rate"] is not None
    assert data["charts"]["feasible_by_power_cap"]
    assert data["charts"]["saving_histogram_by_fidelity"]
    assert data["verification"]["verified"] > 0
    # Labels reach the dashboard (FR-U1 starts at the API).
    assert "npu_extrapolated_count" in data
    assert "uncalibrated_count" in data


def test_scenarios_filters_and_pagination(client: TestClient) -> None:
    everything = client.get("/api/scenarios", params={"batch_id": "lab-test"}).json()
    assert everything["total"] == 6
    assert len(everything["rows"]) == 6
    for row in everything["rows"]:
        assert row["fidelity"] in ("surrogate", "envelope", "sim")

    page = client.get(
        "/api/scenarios",
        params={"batch_id": "lab-test", "page_size": 4, "page": 2},
    ).json()
    assert page["total"] == 6
    assert len(page["rows"]) == 2

    feasible_only = client.get(
        "/api/scenarios", params={"batch_id": "lab-test", "feasible": True}
    ).json()
    assert all(r["feasible"] for r in feasible_only["rows"])
    assert feasible_only["total"] == len(feasible_only["rows"])

    ordered = client.get(
        "/api/scenarios",
        params={"batch_id": "lab-test", "sort": "avg_power_w", "descending": True},
    ).json()
    powers = [r["avg_power_w"] for r in ordered["rows"] if r["avg_power_w"] is not None]
    assert powers == sorted(powers, reverse=True)

    bad_sort = client.get("/api/scenarios", params={"sort": "1; DROP TABLE results"})
    assert bad_sort.status_code == 422


def test_scenario_detail(client: TestClient) -> None:
    detail = client.get("/api/scenarios/sc0000x0000").json()
    assert detail["row"]["scenario_id"] == "sc0000x0000"
    assert detail["service"]["model"] == "meta-llama/Llama-3.1-8B"
    assert detail["cluster"]["cluster_id"] == "c0000"
    graph = detail["graph"]
    assert graph["nodes"] and graph["links"]
    kinds = {n["kind"] for n in graph["nodes"]}
    assert kinds == {"accelerator", "nic"}
    if detail["row"]["feasible"]:
        # The recommended plan's devices are highlighted (FR-U3/FR-A5).
        assert any(n["in_plan"] for n in graph["nodes"])
        assert all(n["role"] for n in graph["nodes"] if n["in_plan"])
    assert detail["document"]["planner_output"]["provenance"]
    assert detail["verification"] is not None

    assert client.get("/api/scenarios/nope").status_code == 404


def test_clusters_and_services(client: TestClient) -> None:
    clusters = client.get("/api/clusters").json()
    assert [c["cluster_id"] for c in clusters] == ["c0000", "c0001"]
    detail = client.get("/api/clusters/c0000").json()
    assert detail["graph"]["nodes"]
    assert client.get("/api/clusters/c9999").status_code == 404

    services = client.get("/api/services").json()
    assert [s["service_id"] for s in services] == ["s0000", "s0001", "s0002"]
    assert client.get("/api/services/s0000").json()["rps"] > 0
    assert client.get("/api/services/s9999").status_code == 404


def test_verification_endpoint(client: TestClient) -> None:
    data = client.get("/api/verification", params={"batch_id": "lab-test"}).json()
    assert data["stats"]["verified"] > 0
    assert abs(data["stats"]["err_tpot_pct_p50"] - 20.0) < 0.01
    for point in data["points"]:
        assert point["sim_p99_tpot_ms"] is not None
        assert point["fast_p99_tpot_ms"] is not None


def test_progress_endpoint(client: TestClient) -> None:
    data = client.get("/api/batches/lab-test/progress").json()
    assert data["total"] == 6
    assert data["done"] == 6


def test_web_ui_served(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "ScenarioLab" in index.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_missing_db_is_503(tmp_path: Path) -> None:
    app = create_app(tmp_path / "nope.sqlite", root=ROOT)
    response = TestClient(app).get("/api/summary")
    assert response.status_code == 503


def test_readonly_store_rejects_writes(fixture_db: Path) -> None:
    """FR-A6: the API-side store cannot mutate results."""
    import sqlite3

    store = ResultStore(fixture_db, readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store._conn.execute("DELETE FROM results")
    finally:
        store.close()


def test_openapi_contract_golden(client: TestClient) -> None:
    """The UI depends only on this contract (DESIGN §10.4). Regenerate by
    deleting the golden file and re-running; inspect the diff deliberately."""
    spec = client.get("/openapi.json").json()
    surface = {
        path: sorted(methods.keys()) for path, methods in spec["paths"].items()
    }
    schemas = {
        name: sorted(schema.get("properties", {}).keys())
        for name, schema in spec["components"]["schemas"].items()
        if not name.startswith(("HTTP", "Validation"))
    }
    snapshot = {"paths": surface, "schemas": schemas}
    golden_path = GOLDEN / "openapi-surface.json"
    if not golden_path.exists():
        GOLDEN.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        raise AssertionError(f"golden created at {golden_path}; inspect and commit")
    assert snapshot == json.loads(golden_path.read_text())
