"""PerformanceEnvelope cache (work order §3.6).

The distinctness tests exist because a missing key field is silent and
catastrophic: a wrong cache hit returns plausible metrics for a configuration
that was never simulated, and the planner then ranks on them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.envelope import EnvelopeCache, key_for, network_class, workload_bucket
from planner.plan import CandidateConfig, IslandAssignment, PredictedMetrics, Role, VllmKnobs
from planner.predictor import SimOutcome, SimResult

ACCEL = {"isl": "RTXPRO6000"}


def candidate(cid="c", *, tp=1, dp=1, pp=1, seqs=128, tokens=2048,
              role=Role.AGGREGATED) -> CandidateConfig:
    return CandidateConfig(
        id=cid, model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id="isl", role=role, tp_size=tp, pp_size=pp, dp_replicas=dp)
        ],
        knobs=VllmKnobs(max_num_seqs=seqs, max_num_batched_tokens=tokens),
    )


def result(cid="c", ttft=100.0) -> SimResult:
    return SimResult(cid, SimOutcome.OK, metrics=PredictedMetrics(
        p50_ttft_ms=ttft * 0.5, p95_ttft_ms=ttft * 0.9, p99_ttft_ms=ttft,
        p50_tpot_ms=8.0, p95_tpot_ms=9.0, p99_tpot_ms=10.0,
        throughput_tps=1000.0, slo_goodput_rps=10.0, slo_attainment=1.0,
        completed_requests=300, completed_tokens=20_000,
    ))


def cache(tmp_path: Path, spec, **kw) -> EnvelopeCache:
    return EnvelopeCache(tmp_path / "env", spec, accelerator_of=ACCEL,
                         link_bw_gbps=900.0, **kw)


# --- key distinctness -----------------------------------------------------

@pytest.mark.parametrize(
    "kw_a,kw_b,field",
    [
        ({"dp": 1}, {"dp": 2}, "dp_replicas"),
        ({"tp": 1}, {"tp": 2}, "tp_size"),
        ({"pp": 1}, {"pp": 2}, "pp_size"),
        ({"seqs": 32}, {"seqs": 256}, "max_num_seqs"),
        ({"tokens": 2048}, {"tokens": 8192}, "max_num_batched_tokens"),
        ({"role": Role.PREFILL}, {"role": Role.DECODE}, "role"),
    ],
)
def test_configurations_that_change_the_result_get_distinct_keys(spec, kw_a, kw_b, field) -> None:
    a = key_for(candidate(**kw_a), spec, accelerator="RTXPRO6000", link_bw_gbps=900.0)
    b = key_for(candidate(**kw_b), spec, accelerator="RTXPRO6000", link_bw_gbps=900.0)
    assert a.digest() != b.digest(), f"{field} does not affect the cache key"


def test_dp_collision_regression(tmp_path: Path, spec) -> None:
    """The exact bug this key field was added for.

    Without `dp` in the key, storing the dp=1 result made the dp=2 candidate a
    cache hit, so it was never simulated and inherited single-replica metrics.
    """
    c = cache(tmp_path, spec)
    c.put(candidate("dp1", dp=1), result("dp1", ttft=5000.0))
    assert c.get(candidate("dp2", dp=2)) is None


def test_identical_configuration_hits(tmp_path: Path, spec) -> None:
    c = cache(tmp_path, spec)
    c.put(candidate("a", tp=2, dp=2), result("a", ttft=1234.0))
    hit = c.get(candidate("b", tp=2, dp=2))
    assert hit is not None and hit.metrics is not None
    assert hit.metrics.p99_ttft_ms == pytest.approx(1234.0)
    assert any("cache" in w for w in hit.warnings)


def test_different_trace_does_not_hit(tmp_path: Path, spec) -> None:
    """A different workload trace is a different experiment."""
    a = cache(tmp_path, spec, trace_digest="aaa")
    b = cache(tmp_path, spec, trace_digest="bbb")
    a.put(candidate(), result())
    assert b.get(candidate()) is None


# --- bucketing ------------------------------------------------------------

def test_workload_bucket_boundaries(spec) -> None:
    lo = spec.model_copy(deep=True)
    lo.traffic.input_tokens.p50 = 500
    lo.traffic.output_tokens.p50 = 64
    lo.traffic.arrival_rate_rps = 1.0
    hi = spec.model_copy(deep=True)
    hi.traffic.input_tokens.p50 = 8000
    hi.traffic.output_tokens.p50 = 1000
    hi.traffic.arrival_rate_rps = 50.0
    assert workload_bucket(lo) != workload_bucket(hi)
    assert "in_lt1024" in workload_bucket(lo)
    assert "in_ge4096" in workload_bucket(hi)


@pytest.mark.parametrize(
    "bw,label", [(10, "lt25"), (50, "lt100"), (150, "lt200"), (300, "lt400"), (900, "ge400")]
)
def test_network_class_bands(bw, label) -> None:
    assert network_class(bw) == label


# --- robustness -----------------------------------------------------------

def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path, spec) -> None:
    """A cache that can break a planning run is worse than no cache."""
    c = cache(tmp_path, spec)
    c.put(candidate(), result())
    for f in (tmp_path / "env").glob("*.json"):
        f.write_text("{ this is not json")
    assert c.get(candidate()) is None


def test_disabled_cache_never_stores(tmp_path: Path, spec) -> None:
    c = cache(tmp_path, spec, enabled=False)
    c.put(candidate(), result())
    assert c.get(candidate()) is None


def test_failed_sims_are_not_cached(tmp_path: Path, spec) -> None:
    """Caching a crash would make the failure permanent across runs."""
    c = cache(tmp_path, spec)
    c.put(candidate(), SimResult("c", SimOutcome.CRASHED, detail="boom"))
    assert c.get(candidate()) is None


def test_hit_and_miss_counters(tmp_path: Path, spec) -> None:
    c = cache(tmp_path, spec)
    c.get(candidate())
    c.put(candidate(), result())
    c.get(candidate())
    assert c.stats() == {"hits": 1, "misses": 1}
