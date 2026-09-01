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


# ---------------------------------------------------------------------------
# P2: Tier-1 envelope, calibration coverage, FR-T6 NPU flag
# ---------------------------------------------------------------------------

def _shared_envelope(root, spec, islands, *, readonly: bool):
    from scenariolab.runner.tiers import SharedEnvelope

    return SharedEnvelope(
        root, spec,
        accelerator_of={i.id: i.accelerator_model for i in islands},
        link_bw_gbps=64.0,
        readonly=readonly,
    )


def test_envelope_tier1_hit_and_readonly(
    spec, cluster, islands, profiles, trace, tmp_path: Path
) -> None:
    candidate, by_id = _a5000_candidate(islands)
    result = SurrogatePredictor(trace).predict(candidate, spec, cluster, by_id, profiles)

    reader = _shared_envelope(tmp_path / "env", spec, islands, readonly=True)
    reader.put(candidate, result)  # must be a no-op
    assert reader.get(candidate) is None
    assert list((tmp_path / "env").glob("*.json")) == []

    writer = _shared_envelope(tmp_path / "env", spec, islands, readonly=False)
    writer.put(candidate, result)
    hit = reader.get(candidate)
    assert hit is not None and hit.ok
    assert hit.metrics == result.metrics
    assert any("envelope cache" in w for w in hit.warnings)


def test_envelope_key_ignores_trace(
    spec, cluster, islands, profiles, trace, tmp_path: Path
) -> None:
    """Tier-1 entries are bucket-level: a different trace (same workload
    bucket) must still hit, or cross-scenario reuse would never happen."""
    from planner.util.workload import generate_trace

    candidate, by_id = _a5000_candidate(islands)
    writer = _shared_envelope(tmp_path / "env", spec, islands, readonly=False)
    writer.put(
        candidate,
        SurrogatePredictor(trace).predict(candidate, spec, cluster, by_id, profiles),
    )
    other_trace = generate_trace(spec, tmp_path / "w2.jsonl", num_requests=30, seed=99)
    assert other_trace.path != trace.path
    reader = _shared_envelope(tmp_path / "env", spec, islands, readonly=True)
    assert reader.get(candidate) is not None


def _calibration_model(bucket: str, hardware: str = "A5000"):
    from planner.predictor.calibration import (
        BucketError,
        CalibrationModel,
        ErrorStats,
        HardwareCalibration,
    )

    return CalibrationModel(
        hardware={
            hardware: HardwareCalibration(
                hardware=hardware,
                errors={
                    bucket: BucketError(
                        workload_bucket=bucket,
                        ttft=ErrorStats(p95_abs_error=0.10, sample_count=5),
                        tpot=ErrorStats(p95_abs_error=0.05, sample_count=5),
                    )
                },
            )
        }
    )


def test_calibration_margins_covered_and_not(spec) -> None:
    from planner.envelope import workload_bucket
    from scenariolab.runner.tiers import calibration_margins

    bucket = workload_bucket(spec)
    model = _calibration_model(bucket)

    ttft, tpot, calibrated = calibration_margins(model, {"A5000"}, spec)
    assert (ttft, tpot, calibrated) == (10.0, 5.0, True)

    # One uncovered hardware class disqualifies the whole scenario (FR-T2).
    ttft, tpot, calibrated = calibration_margins(model, {"A5000", "RNGD-CARD"}, spec)
    assert (ttft, tpot, calibrated) == (0.0, 0.0, False)

    # Covered hardware but a different bucket: raw + calibrated false (§0.4).
    other = _calibration_model("some-other-bucket")
    ttft, tpot, calibrated = calibration_margins(other, {"A5000"}, spec)
    assert (ttft, tpot, calibrated) == (0.0, 0.0, False)


def _rngd_islands(tmp_path: Path):
    from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
    from scenariolab.config import ClusterGeneratorConfig
    from scenariolab.generator.cluster_gen import generate_cluster
    from tests.scenariolab.conftest import ROOT

    gen = ClusterGeneratorConfig.model_validate({
        "num_clusters": 1,
        "nodes_per_cluster": {"min": 1, "max": 1},
        "accelerators_per_node": {"min": 2, "max": 2},
        "accelerator_pool": ["furiosa_rngd_card"],
        "internode_link_pool": ["ib_400g"],
        "free_ratio": {"min": 1.0, "max": 1.0},
    })
    summary = generate_cluster(gen, 0, 4321, tmp_path / "rngd", ROOT, "h")
    cluster = load_cluster_spec(summary.yaml_path)
    profiles = load_profiles_for(cluster, ROOT)
    return cluster, profiles, detect_islands(cluster, profiles)


def test_npu_concurrency_flag(spec, tmp_path: Path) -> None:
    """FR-T6: >32 estimated concurrent sequences per RNGD card -> flagged."""
    from planner.plan import VllmKnobs
    from scenariolab.runner.tiers import npu_concurrency_extrapolated

    _, profiles, islands = _rngd_islands(tmp_path)
    by_id = {i.id: i for i in islands}
    island = islands[0]

    def cand(seqs: int) -> CandidateConfig:
        return CandidateConfig(
            id=f"rngd-s{seqs}",
            model="meta-llama/Llama-3.1-8B",
            dtype="bfloat16",
            assignments=[IslandAssignment(island_id=island.id, tp_size=1)],
            knobs=VllmKnobs(max_num_seqs=seqs),
        )

    assert npu_concurrency_extrapolated(cand(128), spec, by_id, profiles) is True
    assert npu_concurrency_extrapolated(cand(16), spec, by_id, profiles) is False


def test_npu_flag_false_on_gpu_only(spec, islands, profiles) -> None:
    from scenariolab.runner.tiers import npu_concurrency_extrapolated

    candidate, by_id = _a5000_candidate(islands)
    assert npu_concurrency_extrapolated(candidate, spec, by_id, profiles) is False


def test_make_predictor_kinds(trace, tmp_path: Path) -> None:
    from planner.predictor.llmservingsim import LLMServingSimPredictor

    with pytest.raises(ValueError, match="unknown predictor kind"):
        make_predictor("envelope", trace)  # envelope is a cache tier, not a predictor
    sim = make_predictor("sim", trace, work_dir=tmp_path / "sims")
    try:
        assert isinstance(sim, LLMServingSimPredictor)
    finally:
        sim.close()
    assert isinstance(make_predictor("surrogate", trace), SurrogatePredictor)


def test_sim_predictor_gets_scenario_scoped_run_ids(trace, tmp_path: Path) -> None:
    """Candidate ids repeat across scenarios; without a per-scenario run-id
    prefix, concurrent simulations of same-named candidates share (and
    corrupt) one ASTRA-Sim input root. Found by the P2 verification smoke."""
    sim = make_predictor(
        "sim", trace, work_dir=tmp_path / "sims", run_id_prefix="sc0001x0002-"
    )
    try:
        assert sim.run_id_prefix == "sc0001x0002-"
    finally:
        sim.close()
