"""The SLO safety margins of `experiments/scripts/pd_slo_sweep.py` reach feasibility.

WORK_ORDER_consolidation.md STEP 1: the branch added `--tpot-margin-percent` and
`--ttft-margin-percent` to the sweep driver but no test, and the 18 % TPOT figure
those flags carry (deviations D22) is what overturned the loose-TTFT regime. Two
things therefore have to hold, and are asserted here:

1. omitting the flags leaves both margins at 0.0, so every sweep committed before
   D22 still means what its result file says it means;
2. passing 18 arrives at `feasibility.check_latency` as `robust = predicted * 1.18`.

No simulator runs: `exhaustive.search` is intercepted and its kwargs captured, and
the arithmetic in (2) is checked against the real feasibility function.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from planner.optimizer import exhaustive, feasibility
from planner.plan import PredictedMetrics

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/pd_slo_sweep.py"


def _load_sweep():
    """Import the driver by path -- experiments/scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("pd_slo_sweep_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubOutput:
    """The subset of PlannerOutput that main() reads when nothing is feasible."""

    feasible = False
    generated_candidates = 0
    evaluated_candidates = 0
    reason = "stubbed: search was intercepted"
    recommended = None


def _run_sweep(monkeypatch, tmp_path: Path, extra_argv: list[str]) -> list[dict]:
    """Run main() with `exhaustive.search` intercepted; return the captured kwargs."""
    sweep = _load_sweep()
    captured: list[dict] = []

    def fake_search(*args, **kwargs):
        captured.append(kwargs)
        return _StubOutput()

    monkeypatch.setattr(exhaustive, "search", fake_search)
    monkeypatch.setattr(sys, "argv", [
        "pd_slo_sweep.py",
        "--service", str(ROOT / "examples/service_specs/llama31-8b.yaml"),
        "--cluster", str(ROOT / "examples/clusters/heterogeneous-lab.yaml"),
        "--root", str(ROOT),
        # One SLO point and a tiny trace: this test is about argument plumbing,
        # not about the sweep's answer.
        "--ttft-ms", "1000",
        "--num-requests", "4",
        "--output-dir", str(tmp_path / "out"),
        *extra_argv,
    ])

    assert sweep.main() == 0
    assert len(captured) == 1, "one SLO point must call search exactly once"
    return captured


def test_margin_default_is_zero(monkeypatch, tmp_path):
    """No flags => both margins 0.0, so pre-D22 sweeps are unchanged."""
    kwargs = _run_sweep(monkeypatch, tmp_path, [])[0]
    assert kwargs["tpot_margin_percent"] == 0.0
    assert kwargs["ttft_margin_percent"] == 0.0


def test_margin_reaches_feasibility(monkeypatch, tmp_path):
    """--tpot-margin-percent 18 arrives as robust = predicted * 1.18."""
    kwargs = _run_sweep(monkeypatch, tmp_path, ["--tpot-margin-percent", "18"])[0]
    assert kwargs["tpot_margin_percent"] == 18.0
    assert kwargs["ttft_margin_percent"] == 0.0, "the TPOT flag must not move TTFT"

    # The value the sweep forwards is the one feasibility inflates by. 42.4 ms is
    # under a 50 ms TPOT SLO on its own and over it once the measured optimism is
    # charged -- exactly the reversal D22 reports.
    from planner.spec import load_service_spec

    spec = load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")
    spec.slo.tpot.max_ms = 50.0
    spec.slo.ttft.max_ms = 1_000_000.0
    metrics = PredictedMetrics(
        p50_ttft_ms=100.0, p95_ttft_ms=150.0, p99_ttft_ms=200.0,
        p50_tpot_ms=20.0, p95_tpot_ms=30.0, p99_tpot_ms=42.4,
        throughput_tps=1000.0, slo_goodput_rps=10.0, slo_attainment=1.0,
        completed_requests=100, completed_tokens=10_000,
    )
    assert spec.slo.tpot.percentile == 99, "the fixture must pin p99 for this check"

    assert feasibility.check_latency(metrics, spec) == []

    violations = feasibility.check_latency(
        metrics, spec, tpot_margin_percent=kwargs["tpot_margin_percent"]
    )
    assert [v.metric for v in violations] == ["p99_tpot_ms"]
    assert violations[0].predicted == pytest.approx(42.4 * 1.18)
