"""F2/F3 workspace placement tests (workspace work order §5.6, §4.2)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from scenariolab.generator.cluster_builder import ClusterBuildRequest, build_cluster
from scenariolab.runner.workspace import (
    place_service,
    random_workspace_service,
    save_user_service,
)
from scenariolab.store.db import ResultStore, StoreError
from tests.scenariolab.conftest import ROOT

RELAXED = {
    "rps": 2.0, "input_p50": 256, "output_p50": 64,
    "ttft_p99_ms": 20000, "tpot_p99_ms": 200, "power_cap_w": 3000,
}


@pytest.fixture
def env(tmp_path: Path):
    """A store + an 8-GPU custom cluster + one workspace."""
    request = ClusterBuildRequest.model_validate({
        "name": "fixture-8gpu",
        "nodes": [{"class": "a40", "count_per_node": 4, "num_nodes": 2}],
        "interconnect": {"inter_node": {"preset": "ib_400g"}},
    })
    summary, _, _ = build_cluster(request, tmp_path / "clusters", ROOT)
    store = ResultStore(tmp_path / "lab.sqlite")
    store.upsert_cluster(summary)
    workspace_id = store.create_workspace(summary.cluster_id, "t")
    yield store, workspace_id, tmp_path
    store.close()


def _place(store, workspace_id, tmp_path, slo=None, confirm=True):
    seq = len(store.workspace_placements(workspace_id)) + 1
    summary, spec_obj = save_user_service(
        workspace_id, seq, dict(slo or RELAXED), tmp_path / "services"
    )
    store.upsert_service(summary, origin="user")
    return place_service(
        store, workspace_id, summary.service_id, spec_obj,
        root=ROOT, results_dir=tmp_path / "results",
        envelope_dir=None, calibration_dir=None, confirm=confirm,
    )


def test_sequential_placement_disjoint_devices(env) -> None:
    store, workspace_id, tmp_path = env
    first = _place(store, workspace_id, tmp_path)
    second = _place(store, workspace_id, tmp_path)
    assert first["status"] == "PLACED" and second["status"] == "PLACED"
    assert set(first["devices"]) & set(second["devices"]) == set()
    occupied = store.placed_devices(workspace_id)
    assert occupied == set(first["devices"]) | set(second["devices"])
    assert len(occupied) <= 8
    # SLO verdicts and labels recorded (FR-P6).
    row = store.get_placement(first["placement_id"])
    assert row["slo_ttft_ok"] == 1 and row["slo_tpot_ok"] == 1
    assert row["fidelity"] == "surrogate"


def test_exhaustion_rejected_with_occupancy_summary(env) -> None:
    store, workspace_id, tmp_path = env
    # Load heavy enough that each service needs multiple devices.
    heavy = dict(RELAXED, rps=30.0)
    outcomes = [
        _place(store, workspace_id, tmp_path, slo=heavy) for _ in range(9)
    ]
    statuses = [o["status"] for o in outcomes]
    assert "REJECTED" in statuses, statuses
    rejected = next(o for o in outcomes if o["status"] == "REJECTED")
    reason = rejected["record"]["rejected_reason"]
    assert reason["occupied_devices"]  # FR-P5: what was unavailable and why
    # The overlay is untouched by the rejection.
    placed = [o for o in outcomes if o["status"] == "PLACED"]
    assert store.placed_devices(workspace_id) == {
        d for o in placed for d in o["devices"]
    }


def test_remove_returns_devices_and_allows_replacement(env) -> None:
    store, workspace_id, tmp_path = env
    first = _place(store, workspace_id, tmp_path)
    before = store.placed_devices(workspace_id)
    assert before
    store.remove_placement(first["placement_id"])
    assert store.placed_devices(workspace_id) == set()
    again = _place(store, workspace_id, tmp_path)
    assert again["status"] == "PLACED"
    assert set(again["devices"]) == before  # deterministic pipeline, same pick


def test_preview_does_not_occupy(env) -> None:
    store, workspace_id, tmp_path = env
    preview = _place(store, workspace_id, tmp_path, confirm=False)
    assert preview["status"] == "PLANNING"
    assert store.placed_devices(workspace_id) == set()  # FR-W4
    store.confirm_placement(preview["placement_id"])
    assert store.placed_devices(workspace_id) == set(preview["devices"])


def test_confirm_atomicity_under_race(env) -> None:
    """FR-P2: two previews computed from the same FREE state can never both
    occupy the devices."""
    store, workspace_id, tmp_path = env
    a = _place(store, workspace_id, tmp_path, confirm=False)
    b = _place(store, workspace_id, tmp_path, confirm=False)
    assert set(a["devices"]) == set(b["devices"])  # same FREE state, same pick

    results: dict[str, Exception | None] = {}

    def confirm(pid: str) -> None:
        try:
            with ResultStore(store.db_path) as own:
                own.confirm_placement(pid)
            results[pid] = None
        except StoreError as exc:
            results[pid] = exc

    threads = [
        threading.Thread(target=confirm, args=(p["placement_id"],))
        for p in (a, b)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    outcomes = list(results.values())
    assert sum(1 for e in outcomes if e is None) == 1
    assert sum(1 for e in outcomes if e is not None) == 1
    assert "already occupied" in str(next(e for e in outcomes if e))
    assert store.placed_devices(workspace_id) == set(a["devices"])


def test_workspace_power_cap(env) -> None:
    store, workspace_id, _tmp = env
    row = store.get_workspace(workspace_id)
    capped = store.create_workspace(row["cluster_id"], "capped", total_power_cap_w=1.0)
    outcome = _place(store, capped, _tmp)
    assert outcome["status"] == "REJECTED"
    violated = outcome["record"]["rejected_reason"]["violated_constraints"]
    assert violated[0]["metric"] == "workspace_power_cap"


def test_workspaces_are_isolated(env) -> None:
    """FR-CAT2: placements in one workspace never affect another's overlay."""
    store, workspace_a, tmp_path = env
    row = store.get_workspace(workspace_a)
    workspace_b = store.create_workspace(row["cluster_id"], "b")
    placed_a = _place(store, workspace_a, tmp_path)
    assert placed_a["status"] == "PLACED"
    assert store.placed_devices(workspace_b) == set()
    placed_b = _place(store, workspace_b, tmp_path)
    # Same FREE view as A's first placement -> same deterministic pick.
    assert set(placed_b["devices"]) == set(placed_a["devices"])


def test_random_sequence_reproducible(env) -> None:
    """FR-P7: (seed, index) fully determines the random SLO sequence."""
    _store, workspace_id, tmp_path = env
    first = [
        random_workspace_service(workspace_id, k, 123, tmp_path / "svc-a")
        for k in range(3)
    ]
    second = [
        random_workspace_service(workspace_id, k, 123, tmp_path / "svc-b")
        for k in range(3)
    ]
    assert [s.rps for s in first] == [s.rps for s in second]
    assert [s.ttft_p99_ms for s in first] == [s.ttft_p99_ms for s in second]
    other = random_workspace_service(workspace_id, 0, 999, tmp_path / "svc-c")
    assert other.service_id != first[0].service_id  # seed embedded in the id


def test_devices_json_records_roles(env) -> None:
    store, workspace_id, tmp_path = env
    placed = _place(store, workspace_id, tmp_path)
    row = store.get_placement(placed["placement_id"])
    stored = json.loads(row["devices_json"])
    assert isinstance(stored, dict)
    assert set(stored.values()) <= {"aggregated", "prefill", "decode"}
