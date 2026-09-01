"""M2 SLOGenerator tests (DESIGN §5.4)."""

from __future__ import annotations

from pathlib import Path

from planner.spec import load_service_spec
from scenariolab.config import SloGeneratorConfig
from scenariolab.generator.sampling import derive_seed
from scenariolab.generator.slo_gen import generate_service


def _slo_config(**overrides) -> SloGeneratorConfig:
    raw = {
        "num_specs": 5,
        "models": ["meta-llama/Llama-3.1-8B"],
        "arrival_rate_rps": {"dist": "loguniform", "min": 0.5, "max": 30},
        "input_tokens_p50": {"dist": "choice", "values": [256, 512, 1024]},
        "output_tokens_p50": {"dist": "choice", "values": [64, 128, 512]},
        "ttft_p99_ms": {"dist": "loguniform", "min": 200, "max": 5000},
        "tpot_p99_ms": {"dist": "loguniform", "min": 30, "max": 300},
        "power_cap_w": {"dist": "uniform", "min": 400, "max": 4000},
        "min_tokens_per_joule": {"dist": "fixed", "value": 0.0},
        "objective": {
            "primary": "minimize_energy",
            "secondary": "minimize_active_accelerators",
        },
    }
    raw.update(overrides)
    return SloGeneratorConfig.model_validate(raw)


def test_determinism_byte_identical(tmp_path: Path) -> None:
    seed = derive_seed(777, "slo", 0)
    a = generate_service(_slo_config(), 0, seed, tmp_path / "a", "h")
    b = generate_service(_slo_config(), 0, seed, tmp_path / "b", "h")
    assert a.yaml_path.read_bytes() == b.yaml_path.read_bytes()


def test_independence_from_batch_size(tmp_path: Path) -> None:
    small = [
        generate_service(
            _slo_config(), j, derive_seed(9, "slo", j), tmp_path / "s", "h"
        ).yaml_path.read_text()
        for j in range(3)
    ]
    large = [
        generate_service(
            _slo_config(), j, derive_seed(9, "slo", j), tmp_path / "l", "h"
        ).yaml_path.read_text()
        for j in range(6)
    ]
    assert small == large[:3]


def test_all_specs_valid_and_in_range(tmp_path: Path) -> None:
    config = _slo_config()
    for j in range(50):
        summary = generate_service(config, j, derive_seed(5, "slo", j), tmp_path, "h")
        spec = load_service_spec(summary.yaml_path)
        # Sampled fields respect the configured ranges.
        assert 0.5 <= spec.traffic.arrival_rate_rps <= 30
        assert spec.traffic.input_tokens.p50 in (256, 512, 1024)
        assert spec.traffic.output_tokens.p50 in (64, 128, 512)
        assert 200 <= spec.slo.ttft.max_ms <= 5000
        assert 30 <= spec.slo.tpot.max_ms <= 300
        assert spec.slo.max_cluster_power_w is not None
        assert 400 <= spec.slo.max_cluster_power_w <= 4000
        # min_tokens_per_joule fixed at 0.0 -> constraint omitted.
        assert spec.slo.min_tokens_per_joule is None
        # FR-S4: monotone token percentiles.
        for dist in (spec.traffic.input_tokens, spec.traffic.output_tokens):
            assert dist.p95 is not None and dist.p99 is not None
            assert dist.p50 <= dist.p95 <= dist.p99
        # FR-S5: objective is the batch-level control variable.
        assert spec.objective.primary.value == "minimize_energy"
        assert spec.objective.secondary is not None
        assert spec.objective.secondary.value == "minimize_active_accelerators"


def test_header_records_seed_and_hash(tmp_path: Path) -> None:
    seed = derive_seed(777, "slo", 2)
    summary = generate_service(_slo_config(), 2, seed, tmp_path, "cafebabe")
    head = summary.yaml_path.read_text().splitlines()[:4]
    assert head[0] == "# generated_by: scenariolab"
    assert f"# slo_seed: {seed}" in head
    assert "# lab_config_hash: cafebabe" in head
    assert head[3].startswith("# sampled: model=")
