"""Calibration tests: linear fit, error stats, robust margins, persistence.

No GPU, no network. The bench-summary fitting path reads the committed real
summary (`outputs/phase0_bench/A40/vllm/validation-nominal/summary.txt`) so it
exercises real numbers without inventing any (absolute rule 3). That file is the
one `profiles/calibration/a40.yaml` records in its `fitted_from` provenance, and
refitting it reproduces the shipped coefficients exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    IslandAssignment,
    PredictedMetrics,
    Role,
)
from planner.predictor.calibration import (
    CalibrationModel,
    apply_robust_margins,
    compute_error_stats,
    fit_from_summaries,
    fit_linear,
    load_calibration,
    parse_validation_summary,
    robust_metric,
    save_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
A40_SUMMARY = ROOT / "outputs/phase0_bench/A40/vllm/validation-nominal/summary.txt"

_METRICS = PredictedMetrics(
    p50_ttft_ms=10.0, p95_ttft_ms=20.0, p99_ttft_ms=30.0,
    p50_tpot_ms=1.0, p95_tpot_ms=2.0, p99_tpot_ms=3.0,
    throughput_tps=100.0, slo_goodput_rps=5.0, slo_attainment=1.0,
    completed_requests=100, completed_tokens=1000,
)


def _plan() -> DeploymentPlan:
    candidate = CandidateConfig(
        id="cand-1",
        model="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
        assignments=[
            IslandAssignment(
                island_id="cuda-rtx-a5000-node0", role=Role.AGGREGATED, tp_size=1
            )
        ],
    )
    return DeploymentPlan(
        plan_id="hp-test", model=candidate.model, candidate=candidate, predicted=_METRICS
    )


# ---------------------------------------------------------------------------
# linear fit + error stats
# ---------------------------------------------------------------------------

def test_fit_linear_recovers_slope_and_intercept() -> None:
    sim = [1.0, 2.0, 3.0, 4.0]
    real = [3.0, 5.0, 7.0, 9.0]  # real = 2*sim + 1
    fit = fit_linear(sim, real)
    assert abs(fit.alpha - 2.0) < 1e-9
    assert abs(fit.beta - 1.0) < 1e-9
    assert fit.sample_count == 4
    assert fit.apply(10.0) == pytest.approx(21.0)


def test_fit_linear_insufficient_data_is_offset_only() -> None:
    fit = fit_linear([5.0], [8.0])
    assert fit.alpha == 1.0
    assert fit.beta == pytest.approx(3.0)
    assert fit.source == "insufficient_data"


def test_fit_linear_empty_is_identity() -> None:
    fit = fit_linear([], [])
    assert fit.alpha == 1.0 and fit.beta == 0.0
    assert fit.source == "identity"


def test_compute_error_stats() -> None:
    # sim under-predicts by 10% everywhere: real = 1.1 * sim.
    sim = [100.0, 200.0, 300.0]
    real = [110.0, 220.0, 330.0]
    stats = compute_error_stats(sim, real)
    assert stats.mean_error == pytest.approx(0.1)
    assert stats.p95_abs_error == pytest.approx(0.1)
    assert stats.sample_count == 3


def test_robust_metric() -> None:
    assert robust_metric(450.0, 0.08) == pytest.approx(486.0)


# ---------------------------------------------------------------------------
# identity model defaults
# ---------------------------------------------------------------------------

def test_identity_model_is_passthrough() -> None:
    model = CalibrationModel.identity()
    assert model.apply_ttft("A40", 123.0) == 123.0
    assert model.apply_tpot("A40", 45.0) == 45.0
    assert model.margins("A40", "any-bucket") == (0.0, 0.0)


def test_load_missing_calibration_is_identity(tmp_path: Path) -> None:
    model = load_calibration(tmp_path / "does-not-exist.yaml")
    assert model.hardware == {}
    assert model.apply_ttft("A40", 5.0) == 5.0


# ---------------------------------------------------------------------------
# bench summary ingestion
# ---------------------------------------------------------------------------

def test_parse_validation_summary_real_file() -> None:
    pairs = parse_validation_summary(A40_SUMMARY.read_text())
    assert len(pairs.ttft) == 5  # Mean, Median, P90, P95, P99
    assert len(pairs.tpot) == 5
    # each pair is (sim, real); the A40 sim slightly under the vLLM figure.
    sim, real = pairs.ttft[0]
    assert sim == pytest.approx(39789.2)
    assert real == pytest.approx(40552.4)


def test_fit_from_summaries_builds_hardware_entry() -> None:
    model = fit_from_summaries([(A40_SUMMARY, "A40", "in_lt1024-out_lt128-rps_lt5")])
    assert "A40" in model.hardware
    cal = model.hardware["A40"]
    assert cal.ttft.source == "measured"
    # A40 sim tracks vLLM closely, so the corrected value stays near the input.
    corrected = model.apply_ttft("A40", 39789.2)
    assert abs(corrected - 40572.3) < 5000.0
    assert model.provenance["fitted_from"]["A40"]


# ---------------------------------------------------------------------------
# robust margins are opt-in
# ---------------------------------------------------------------------------

def test_apply_robust_margins_identity_leaves_zero() -> None:
    plan = _plan()
    out = apply_robust_margins(plan, CalibrationModel.identity(), hardware="A40", bucket="b")
    assert out.robust_margin_ttft_percent == 0.0
    assert out.robust_margin_tpot_percent == 0.0
    # original plan is untouched
    assert plan.robust_margin_ttft_percent == 0.0


def test_apply_robust_margins_from_model() -> None:
    bucket = "in_lt1024-out_lt128-rps_lt5"
    model = fit_from_summaries([(A40_SUMMARY, "A40", bucket)])
    out = apply_robust_margins(_plan(), model, hardware="A40", bucket=bucket)
    assert out.robust_margin_ttft_percent >= 0.0
    assert out.robust_margin_tpot_percent >= 0.0


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model = fit_from_summaries([(A40_SUMMARY, "A40", "b")])
    path = tmp_path / "a40.yaml"
    save_calibration(model, path)
    reloaded = load_calibration(path)
    assert "A40" in reloaded.hardware
    assert reloaded.apply_ttft("A40", 100.0) == pytest.approx(model.apply_ttft("A40", 100.0))
