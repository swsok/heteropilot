"""Shared fixtures for the ScenarioLab suite.

Batches under test are tiny (2 clusters x 3 SLOs) and run on the surrogate
tier only, so the whole suite needs no simulator and no GPU. Store paths point
into tmp_path; profile reads go to the real repo root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from planner.predictor import Predictor
from planner.util.workload import WorkloadTrace
from scenariolab.config import LabConfig, load_lab_config
from scenariolab.runner.tiers import SurrogatePredictor

ROOT = Path(__file__).resolve().parents[2]


def lab_yaml_dict(tmp_path: Path, **overrides: Any) -> dict:
    """A small, valid LabConfig mapping with store paths under tmp_path."""
    raw: dict = {
        "lab": {"batch_name": "lab-test", "seed": 777},
        "cluster_generator": {
            "num_clusters": 2,
            "nodes_per_cluster": {"min": 1, "max": 2},
            "accelerators_per_node": {"min": 1, "max": 2},
            "accelerator_pool": ["a5000", "furiosa_rngd_card"],
            "same_class_per_node": True,
            "internode_link_pool": ["ib_400g"],
            "free_ratio": {"min": 0.5, "max": 1.0},
        },
        "slo_generator": {
            "num_specs": 3,
            "models": ["meta-llama/Llama-3.1-8B"],
            "arrival_rate_rps": {"dist": "loguniform", "min": 0.5, "max": 10},
            "input_tokens_p50": {"dist": "choice", "values": [256, 512]},
            "output_tokens_p50": {"dist": "choice", "values": [64, 128]},
            "ttft_p99_ms": {"dist": "loguniform", "min": 200, "max": 5000},
            "tpot_p99_ms": {"dist": "loguniform", "min": 30, "max": 300},
            "power_cap_w": {"dist": "uniform", "min": 400, "max": 4000},
            "min_tokens_per_joule": {"dist": "fixed", "value": 0.0},
            "objective": {
                "primary": "minimize_energy",
                "secondary": "minimize_active_accelerators",
            },
        },
        "pairing": {"mode": "cross", "max_scenarios": 1500},
        "runner": {
            "workers": 1,
            "num_requests": 20,
            "tier_policy": {"full_sim": "never"},
            "verification": {"fraction": 0.0, "min_count": 0},
        },
        "store": {
            "db_path": str(tmp_path / "lab.sqlite"),
            "results_dir": str(tmp_path / "results"),
            "clusters_dir": str(tmp_path / "clusters"),
            "services_dir": str(tmp_path / "services"),
            "envelope_dir": str(tmp_path / "envelope"),
        },
    }
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        if field:
            raw[section][field] = value
        else:
            raw[section] = value
    return raw


def write_lab_config(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "lab.yaml"
    path.write_text(yaml.safe_dump(lab_yaml_dict(tmp_path, **overrides)))
    return path


@pytest.fixture
def lab(tmp_path: Path) -> tuple[LabConfig, str, str]:
    path = write_lab_config(tmp_path)
    config, digest = load_lab_config(path, ROOT)
    return config, digest, path.read_text()


class ExplodingFactory:
    """Predictor factory that raises for chosen scenario seeds (error tests)."""

    def __init__(self, fail_seeds: set[int] | None = None) -> None:
        self.fail_seeds = fail_seeds or set()
        self.calls: list[int] = []

    def __call__(self, trace: WorkloadTrace) -> Predictor:
        self.calls.append(trace.seed)
        if trace.seed in self.fail_seeds:
            raise RuntimeError("injected predictor failure")
        return SurrogatePredictor(trace)
