"""ServiceSpec loading and validation (work order §5.1, §9)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from planner.spec import Objective, SpecError, load_service_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "service_specs"

VALID = """
service:
  model: Qwen/Qwen3-32B
  dtype: bfloat16
traffic:
  arrival_rate_rps: 15
  input_tokens: {p50: 512, p95: 4096, p99: 8192}
  output_tokens: {p50: 128, p95: 512}
slo:
  ttft: {percentile: 99, max_ms: 500}
  tpot: {percentile: 99, max_ms: 40}
objective:
  primary: maximize_slo_goodput_per_joule
  secondary: minimize_active_accelerators
"""


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_shipped_examples_load() -> None:
    for name in ("qwen3-32b.yaml", "llama31-8b.yaml"):
        spec = load_service_spec(EXAMPLES / name)
        assert spec.model
        assert spec.traffic.arrival_rate_rps > 0


def test_valid_spec_roundtrip(tmp_path: Path) -> None:
    spec = load_service_spec(write(tmp_path, VALID))
    assert spec.model == "Qwen/Qwen3-32B"
    assert spec.slo.ttft.max_ms == 500
    assert spec.objective.primary is Objective.MAXIMIZE_SLO_GOODPUT_PER_JOULE


def test_missing_traffic_is_rejected_with_an_explanation(tmp_path: Path) -> None:
    body = VALID.replace(
        """traffic:
  arrival_rate_rps: 15
  input_tokens: {p50: 512, p95: 4096, p99: 8192}
  output_tokens: {p50: 128, p95: 512}
""",
        "",
    )
    with pytest.raises(SpecError) as exc:
        load_service_spec(write(tmp_path, body))
    msg = str(exc.value)
    assert "traffic" in msg
    # The work order asks specifically that this failure explain *why*.
    assert "SLO" in msg and "arrival_rate_rps" in msg


@pytest.mark.parametrize("percentile", [0, 90, 98, 100, 999])
def test_only_p50_p95_p99_allowed(tmp_path: Path, percentile: int) -> None:
    body = VALID.replace("ttft: {percentile: 99", f"ttft: {{percentile: {percentile}")
    with pytest.raises(SpecError, match="percentile"):
        load_service_spec(write(tmp_path, body))


@pytest.mark.parametrize("percentile", [50, 95, 99])
def test_allowed_percentiles_accepted(tmp_path: Path, percentile: int) -> None:
    body = VALID.replace("ttft: {percentile: 99", f"ttft: {{percentile: {percentile}")
    assert load_service_spec(write(tmp_path, body)).slo.ttft.percentile == percentile


def test_identical_objectives_rejected(tmp_path: Path) -> None:
    body = VALID.replace("secondary: minimize_active_accelerators",
                         "secondary: maximize_slo_goodput_per_joule")
    with pytest.raises(SpecError, match="must differ"):
        load_service_spec(write(tmp_path, body))


def test_unknown_objective_rejected(tmp_path: Path) -> None:
    body = VALID.replace("primary: maximize_slo_goodput_per_joule", "primary: go_faster")
    with pytest.raises(SpecError):
        load_service_spec(write(tmp_path, body))


def test_typo_in_field_name_is_not_silently_ignored(tmp_path: Path) -> None:
    body = VALID.replace("arrival_rate_rps: 15", "arrival_rate_rp: 15")
    with pytest.raises(SpecError):
        load_service_spec(write(tmp_path, body))


def test_non_monotone_token_distribution_rejected(tmp_path: Path) -> None:
    body = VALID.replace("input_tokens: {p50: 512, p95: 4096, p99: 8192}",
                         "input_tokens: {p50: 512, p95: 256, p99: 8192}")
    with pytest.raises(SpecError, match="monotone"):
        load_service_spec(write(tmp_path, body))


def test_nonpositive_arrival_rate_rejected(tmp_path: Path) -> None:
    with pytest.raises(SpecError):
        load_service_spec(write(tmp_path, VALID.replace("arrival_rate_rps: 15",
                                                        "arrival_rate_rps: 0")))


def test_defaults_applied(tmp_path: Path) -> None:
    spec = load_service_spec(write(tmp_path, VALID))
    assert spec.traffic.burstiness == 1.0
    assert spec.traffic.prefix_share_ratio == 0.0
    assert spec.service.kv_cache_dtype == "auto"


def test_missing_file_is_a_spec_error(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not found"):
        load_service_spec(tmp_path / "nope.yaml")
