"""M4 (P1 increment) SurrogatePredictor tests (DESIGN §7.5 subset).

The envelope tier, calibration flags and full-sim verification are P2; here we
pin down what P1 ships: a deterministic, physics-consistent, honestly-labelled
analytic predictor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.plan import CandidateConfig, IslandAssignment
from planner.util.workload import generate_trace
from scenariolab.runner.tiers import SurrogatePredictor, make_predictor


@pytest.fixture
def trace(spec, tmp_path: Path):
    return generate_trace(spec, tmp_path / "w.jsonl", num_requests=20, seed=1)


def _a5000_candidate(islands) -> tuple[CandidateConfig, dict]:
    by_id = {i.id: i for i in islands}
    island = next(i for i in islands if i.accelerator_model == "RTX-A5000")
    candidate = CandidateConfig(
        id="t-a5000-tp1",
        model="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
        assignments=[IslandAssignment(island_id=island.id, tp_size=1, dp_replicas=1)],
    )
    return candidate, by_id


def test_deterministic_and_internally_consistent(
    spec, cluster, islands, profiles, trace
) -> None:
    candidate, by_id = _a5000_candidate(islands)
    predictor = SurrogatePredictor(trace)
    first = predictor.predict(candidate, spec, cluster, by_id, profiles)
    second = predictor.predict(candidate, spec, cluster, by_id, profiles)
    assert first.ok and second.ok
    assert first.metrics == second.metrics

    m = first.metrics
    assert m is not None
    assert m.p50_ttft_ms <= m.p95_ttft_ms <= m.p99_ttft_ms
    assert m.p50_tpot_ms <= m.p95_tpot_ms <= m.p99_tpot_ms
    assert m.throughput_tps > 0
    assert 0.0 <= m.slo_attainment <= 1.0


def test_power_comes_from_the_profile(spec, cluster, islands, profiles, trace) -> None:
    """One A5000 at TP=1: peak power must be exactly the profile's measured
    active_power - the surrogate copies hardware numbers, never invents them."""
    candidate, by_id = _a5000_candidate(islands)
    result = SurrogatePredictor(trace).predict(candidate, spec, cluster, by_id, profiles)
    m = result.metrics
    assert m is not None
    a5000_active = profiles["RTX-A5000"].power.active_power
    assert m.peak_power_w == pytest.approx(a5000_active)
    assert m.average_power_w is not None
    assert profiles["RTX-A5000"].power.idle_power <= m.average_power_w <= m.peak_power_w
    assert m.total_energy_j is not None and m.total_energy_j > 0
    assert m.tokens_per_joule == pytest.approx(m.completed_tokens / m.total_energy_j)


def test_energy_scales_with_devices(spec, cluster, islands, profiles, trace) -> None:
    by_id = {i.id: i for i in islands}
    island = next(i for i in islands if i.accelerator_model == "RTX-A5000")
    single = CandidateConfig(
        id="one", model=spec.model, dtype="bfloat16",
        assignments=[IslandAssignment(island_id=island.id, tp_size=1, dp_replicas=1)],
    )
    double = CandidateConfig(
        id="two", model=spec.model, dtype="bfloat16",
        assignments=[IslandAssignment(island_id=island.id, tp_size=1, dp_replicas=2)],
    )
    predictor = SurrogatePredictor(trace)
    m1 = predictor.predict(single, spec, cluster, by_id, profiles).metrics
    m2 = predictor.predict(double, spec, cluster, by_id, profiles).metrics
    assert m1 is not None and m2 is not None
    assert m2.peak_power_w == pytest.approx(2 * m1.peak_power_w)
    assert m2.throughput_tps > m1.throughput_tps


def test_make_predictor_rejects_unknown_kind(trace) -> None:
    with pytest.raises(ValueError, match="unknown predictor kind"):
        make_predictor("sim", trace)
