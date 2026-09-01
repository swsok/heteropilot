"""Interactive fast-path tests (DESIGN §2.2, FR-T5/FR-A3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenariolab.config import ClusterGeneratorConfig
from scenariolab.generator.cluster_gen import generate_cluster
from scenariolab.runner.interactive import (
    InteractivePlanError,
    build_service_spec,
    plan_interactive,
)
from tests.scenariolab.conftest import ROOT

RELAXED = {
    "rps": 2.0, "input_p50": 256, "output_p50": 64,
    "ttft_p99_ms": 20000, "tpot_p99_ms": 200, "power_cap_w": 3000,
}


@pytest.fixture(scope="module")
def cluster_yaml(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("iplan")
    gen = ClusterGeneratorConfig.model_validate({
        "num_clusters": 1,
        "nodes_per_cluster": {"min": 2, "max": 2},
        "accelerators_per_node": {"min": 2, "max": 2},
        "accelerator_pool": ["a5000", "furiosa_rngd_card"],
        "internode_link_pool": ["ib_400g"],
        "free_ratio": {"min": 1.0, "max": 1.0},
    })
    return generate_cluster(gen, 0, 777, tmp, ROOT, "h").yaml_path


def test_traffic_required(cluster_yaml: Path) -> None:
    slo = dict(RELAXED)
    slo.pop("rps")
    slo["rps"] = None
    with pytest.raises(InteractivePlanError, match="traffic is required"):
        build_service_spec(slo)


def test_fast_path_labels_and_determinism(cluster_yaml: Path) -> None:
    first = plan_interactive(cluster_yaml, dict(RELAXED), root=ROOT)
    second = plan_interactive(cluster_yaml, dict(RELAXED), root=ROOT)
    assert first["feasible"] is True
    # FR-A2: the honesty block is always present.
    for key in ("fidelity", "calibrated", "npu_extrapolated", "truncated",
                "elapsed_s", "seed"):
        assert key in first
    assert first["fidelity"] == "surrogate"
    rec1 = first["planner_output"]["recommended"]
    rec2 = second["planner_output"]["recommended"]
    assert rec1["plan"]["candidate"]["id"] == rec2["plan"]["candidate"]["id"]
    assert rec1["value"] == rec2["value"]


def test_top_k_truncation_flagged(cluster_yaml: Path) -> None:
    result = plan_interactive(cluster_yaml, dict(RELAXED), root=ROOT, top_k=1)
    assert result["truncated"] is True
    assert result["planner_output"]["rejected_summary"]["surrogate_pruned"] > 0


def test_infeasible_is_a_diagnosis(cluster_yaml: Path) -> None:
    slo = dict(RELAXED)
    slo["tpot_p99_ms"] = 1.0
    result = plan_interactive(cluster_yaml, slo, root=ROOT)
    assert result["feasible"] is False
    out = result["planner_output"]
    assert out["reason"]
    assert out["suggestions"]
