"""Workspace REST API tests - the work order §8 P1 gate scenario, end to end
over HTTP: build cluster -> workspace -> place random + user services ->
summary -> remove -> devices FREE again."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scenariolab.api.server import create_app
from scenariolab.store.db import ResultStore
from tests.scenariolab.conftest import ROOT

BUILD_REQUEST = {
    "name": "gate",
    "nodes": [
        {"class": "a40", "count_per_node": 8, "num_nodes": 2},
        {"class": "furiosa_rngd_card", "count_per_node": 4, "num_nodes": 1},
    ],
    "interconnect": {"inter_node": {"preset": "ib_100g"}},
}

RELAXED_SLO = {
    "rps": 2.0, "input_p50": 256, "output_p50": 64,
    "ttft_p99_ms": 20000, "tpot_p99_ms": 200, "power_cap_w": 3000,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    tmp = tmp_path_factory.mktemp("wsapi")
    db = tmp / "lab.sqlite"
    ResultStore(db).close()
    return TestClient(create_app(db, root=ROOT, calibration_dir=None))


def test_gate_scenario(client: TestClient) -> None:
    # 1. Build the cluster (a40 x16 + rngd x4, ib_100g).
    built = client.post("/api/clusters/build", json=BUILD_REQUEST)
    assert built.status_code == 200, built.text
    cluster = built.json()
    assert cluster["cluster"]["num_accels"] == 20
    assert cluster["cluster"]["origin"] == "custom"
    assert cluster["islands"]
    cluster_id = cluster["cluster"]["cluster_id"]

    # Idempotency (FR-CB4): the same request maps to the same cluster.
    again = client.post("/api/clusters/build", json=BUILD_REQUEST)
    assert again.json()["cluster"]["cluster_id"] == cluster_id
    assert again.json()["already_existed"] is True
    catalog = client.get("/api/clusters", params={"origin": "custom"}).json()
    assert [c["cluster_id"] for c in catalog] == [cluster_id]

    # 2. Start a workspace.
    workspace = client.post(
        "/api/workspaces", json={"cluster_id": cluster_id, "name": "gate"}
    ).json()
    workspace_id = workspace["workspace_id"]
    assert workspace["status"] == "ACTIVE"

    # 3a. Three random services in one shot (auto-confirmed sequentially).
    random_response = client.post(
        f"/api/workspaces/{workspace_id}/placements",
        json={"slo": "random", "count": 3, "seed": 123},
    )
    assert random_response.status_code == 200, random_response.text
    random_rows = random_response.json()["placements"]
    assert len(random_rows) == 3
    for row in random_rows:
        assert row["status"] in ("PLACED", "REJECTED")
        assert row["fidelity"] in ("surrogate", "envelope", None)

    # 3b. A user-typed impossible SLO -> REJECTED with a diagnosis.
    impossible = dict(RELAXED_SLO, tpot_p99_ms=1.0)
    rejected = client.post(
        f"/api/workspaces/{workspace_id}/placements",
        json={"slo": impossible, "confirm": True},
    ).json()["placements"][0]
    assert rejected["status"] == "REJECTED"
    assert rejected["rejected_reason"]["occupied_devices"] is not None
    assert rejected["service"]["origin"] == "user"

    # 3c. A user-typed feasible SLO with preview -> confirm.
    preview = client.post(
        f"/api/workspaces/{workspace_id}/placements",
        json={"slo": RELAXED_SLO},
    ).json()["placements"][0]
    assert preview["status"] == "PLANNING"
    confirmed = client.post(
        f"/api/workspaces/{workspace_id}/placements/"
        f"{preview['placement_id']}/confirm"
    ).json()
    assert confirmed["status"] == "PLACED"

    # 4. Summary: SLO verdicts, power sums, remaining resources, overlay.
    summary = client.get(f"/api/workspaces/{workspace_id}/summary").json()
    placed = [p for p in summary["placements"] if p["status"] == "PLACED"]
    assert placed
    for row in placed:
        assert row["slo_ttft_ok"] is not None
        assert row["slo_tpot_ok"] is not None
    devices_placed = {d for p in placed for d in p["devices"]}
    assert summary["resources"]["total_accels"] == 20
    assert summary["resources"]["free_accels"] == 20 - len(devices_placed)
    assert summary["power"]["sum_avg_w"] == pytest.approx(
        sum(p["avg_power_w"] for p in placed)
    )
    assert summary["power"]["sum_peak_w"] >= summary["power"]["sum_avg_w"]
    assert set(summary["topology_overlay"]) == devices_placed
    assert "interference" in summary["interference_notice"].lower() or (
        "sole use" in summary["interference_notice"].lower()
    )

    # 5. Remove one placement -> its devices return to FREE.
    victim = placed[0]
    free_before = summary["resources"]["free_accels"]
    deleted = client.delete(
        f"/api/workspaces/{workspace_id}/placements/{victim['placement_id']}"
    )
    assert deleted.status_code == 200
    after = client.get(f"/api/workspaces/{workspace_id}/summary").json()
    assert after["resources"]["free_accels"] == free_before + len(victim["devices"])
    removed_row = next(
        p for p in after["placements"]
        if p["placement_id"] == victim["placement_id"]
    )
    assert removed_row["status"] == "REMOVED"  # FR-W6: history retained


def test_workspace_404s(client: TestClient) -> None:
    assert client.post(
        "/api/workspaces", json={"cluster_id": "nope", "name": "x"}
    ).status_code == 404
    assert client.get("/api/workspaces/nope/summary").status_code == 404
    assert client.post(
        "/api/workspaces/nope/placements", json={"slo": RELAXED_SLO}
    ).status_code == 404


def test_build_rejects_bad_requests(client: TestClient) -> None:
    too_big = {
        "name": "big",
        "nodes": [{"class": "a40", "count_per_node": 8, "num_nodes": 9}],
        "interconnect": {"inter_node": {"preset": "ib_100g"}},
    }
    assert client.post("/api/clusters/build", json=too_big).status_code == 400

    both = {
        "name": "both",
        "nodes": [{"class": "a40", "count_per_node": 2, "num_nodes": 1}],
        "interconnect": {"inter_node": {
            "preset": "ib_100g",
            "custom": {"type": "ETHERNET", "bandwidth_gbps": 50, "latency_ns": 1},
        }},
    }
    assert client.post("/api/clusters/build", json=both).status_code == 422

    missing_traffic = {"slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50}}
    workspaces = client.get("/api/workspaces").json()
    if workspaces:
        response = client.post(
            f"/api/workspaces/{workspaces[0]['workspace_id']}/placements",
            json=missing_traffic,
        )
        assert response.status_code == 400
        assert "traffic is required" in response.json()["detail"]


def test_double_confirm_is_409(client: TestClient) -> None:
    workspaces = client.get("/api/workspaces").json()
    workspace_id = workspaces[0]["workspace_id"]
    preview = client.post(
        f"/api/workspaces/{workspace_id}/placements", json={"slo": RELAXED_SLO}
    ).json()["placements"][0]
    url = (
        f"/api/workspaces/{workspace_id}/placements/"
        f"{preview['placement_id']}/confirm"
    )
    assert client.post(url).status_code == 200
    assert client.post(url).status_code == 409  # not PLANNING any more


def test_placement_detail_endpoint(client: TestClient) -> None:
    """FR-W5: the UI reuses the detail view; the API serves row + document."""
    workspaces = client.get("/api/workspaces").json()
    workspace_id = workspaces[0]["workspace_id"]
    summary = client.get(f"/api/workspaces/{workspace_id}/summary").json()
    target = summary["placements"][0]
    detail = client.get(
        f"/api/workspaces/{workspace_id}/placements/{target['placement_id']}"
    ).json()
    assert detail["placement"]["placement_id"] == target["placement_id"]
    assert "planner_output" in detail["document"]
    missing = client.get(f"/api/workspaces/{workspace_id}/placements/nope")
    assert missing.status_code == 404
