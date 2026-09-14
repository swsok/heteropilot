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

ACCEL = {"isl": "RTXPRO6000", "isl2": "RTX-A5000"}


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
    a = key_for(candidate(**kw_a), spec, accelerator_of=ACCEL, link_bw_gbps=900.0)
    b = key_for(candidate(**kw_b), spec, accelerator_of=ACCEL, link_bw_gbps=900.0)
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


def candidate_mixed(cid="m", *, tp=1, dp_a=1, dp_b=1, seqs=128, tokens=2048) -> CandidateConfig:
    return CandidateConfig(
        id=cid, model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id="isl", role=Role.AGGREGATED, tp_size=tp, dp_replicas=dp_a),
            IslandAssignment(island_id="isl2", role=Role.AGGREGATED, tp_size=tp, dp_replicas=dp_b),
        ],
        knobs=VllmKnobs(max_num_seqs=seqs, max_num_batched_tokens=tokens),
    )


def test_mixed_never_collides_with_single(tmp_path: Path, spec) -> None:
    """A two-island placement must not hit the entry of a single-island one
    that happens to share its first assignment - the mixed-generation version
    of the D13 collision."""
    c = cache(tmp_path, spec)
    c.put(candidate("single", tp=1, dp=1), result("single", ttft=100.0))
    assert c.get(candidate_mixed("mixed", tp=1, dp_a=1, dp_b=1)) is None


def test_mixed_key_is_assignment_order_independent(spec) -> None:
    a = candidate_mixed("m1")
    b = candidate_mixed("m2")
    b_rev = b.model_copy(update={"assignments": list(reversed(b.assignments))})
    ka = key_for(a, spec, accelerator_of=ACCEL, link_bw_gbps=900.0)
    kb = key_for(b_rev, spec, accelerator_of=ACCEL, link_bw_gbps=900.0)
    assert ka.digest() == kb.digest()


def test_mixed_dp_split_changes_the_key(spec) -> None:
    """1+2 replicas and 2+1 replicas on *different* islands are different
    deployments even though total replica count matches."""
    a = key_for(candidate_mixed("a", dp_a=1, dp_b=2), spec,
                accelerator_of=ACCEL, link_bw_gbps=900.0)
    b = key_for(candidate_mixed("b", dp_a=2, dp_b=1), spec,
                accelerator_of=ACCEL, link_bw_gbps=900.0)
    assert a.digest() != b.digest()


def test_unmapped_island_refuses_a_key(tmp_path: Path, spec) -> None:
    """Never build a key that silently ignores an assignment."""
    from planner.envelope import EnvelopeKeyError

    bad = CandidateConfig(
        id="x", model="m", dtype="bfloat16",
        assignments=[IslandAssignment(island_id="ghost", tp_size=1, dp_replicas=1)],
    )
    with pytest.raises(EnvelopeKeyError):
        key_for(bad, spec, accelerator_of=ACCEL, link_bw_gbps=900.0)
    # and the cache treats it as unkeyable, not as an error
    c = cache(tmp_path, spec)
    c.put(bad, result("x"))
    assert c.get(bad) is None


# --- the operating point must survive the cache ----------------------------

def test_the_operating_point_round_trips_through_the_cache(tmp_path, spec):
    """A cached run must still earn its accuracy-domain margin (STEP 4.3).

    Found the hard way: the margin was derived from `SimResult.artifacts["csv"]`,
    which a cache hit does not have, so a warm cache silently dropped every
    margin to zero and printed the plan as though nothing had changed. Nothing
    failed and nothing warned -- the numbers were just quietly less safe. The
    operating point is a property of the run, so it is cached with the metrics.
    """
    c = cache(tmp_path, spec)
    r = result()
    r.operating_point = {"A40": {"concurrency": 14.86, "phase": "total",
                                 "requests": 20, "wall_s": 27.4}}
    cand = candidate()
    c.put(cand, r)

    got = cache(tmp_path, spec).get(cand)
    assert got is not None and got.ok
    assert got.operating_point["A40"]["concurrency"] == pytest.approx(14.86)
    assert got.operating_point["A40"]["phase"] == "total"


def test_a_pre_existing_cache_entry_says_it_cannot_be_margined(tmp_path, spec):
    """Entries written before this existed must announce the gap rather than
    look like a run that simply had no load."""
    import json

    c = cache(tmp_path, spec)
    cand = candidate()
    c.put(cand, result())
    path = next((tmp_path / "env").glob("*.json"))
    payload = json.loads(path.read_text())
    payload.pop("operating_point", None)
    path.write_text(json.dumps(payload))

    got = cache(tmp_path, spec).get(cand)
    assert got is not None and got.operating_point == {}
    assert any("predates operating-point caching" in w for w in got.warnings)


def test_a_metrics_schema_change_invalidates_the_cache(tmp_path, spec) -> None:
    """A stale entry must miss, not validate (uncertainty STEP B2, D33).

    A new optional field defaults to None, so an entry written before it existed
    still parses - and E-A1 once re-ran to byte-identical numbers off a cache that
    predated `served_concurrency_per_island`, which reads as "the change had no
    effect" when it means "the change was never applied". The digest lives in the
    payload, not the file name, so the committed replay caches keep their names.
    """
    import json

    import planner.envelope as env
    from planner.plan import CandidateConfig, IslandAssignment, PredictedMetrics
    from planner.predictor import SimOutcome, SimResult

    candidate = CandidateConfig(
        id="c", model=spec.model, dtype="bfloat16",
        assignments=[IslandAssignment(island_id="i0", tp_size=1)],
    )
    metrics = PredictedMetrics(
        p50_ttft_ms=1.0, p95_ttft_ms=1.0, p99_ttft_ms=1.0,
        p50_tpot_ms=1.0, p95_tpot_ms=1.0, p99_tpot_ms=1.0,
        throughput_tps=1.0, slo_goodput_rps=1.0, slo_attainment=1.0,
        completed_requests=1, completed_tokens=1,
    )

    def make() -> env.EnvelopeCache:
        return env.EnvelopeCache(
            tmp_path, spec, accelerator_of={"i0": "A40"}, link_bw_gbps=100.0
        )

    make().put(candidate, SimResult("c", SimOutcome.OK, metrics=metrics))
    path = next(tmp_path.glob("*.json"))
    assert json.loads(path.read_text())["metrics_schema"] == env._METRICS_SCHEMA
    hit = make().get(candidate)
    assert hit is not None, "same schema, same entry"
    assert not any("no metrics-schema digest" in w for w in hit.warnings)

    original = env._METRICS_SCHEMA
    try:
        env._METRICS_SCHEMA = "pretend-a-field-was-added"
        assert make().get(candidate) is None, "a schema change must miss"
    finally:
        env._METRICS_SCHEMA = original
    assert make().get(candidate) is not None, "and the original is still there"

    # A legacy entry with no digest at all is served, and says so.
    payload = json.loads(path.read_text())
    payload.pop("metrics_schema")
    path.write_text(json.dumps(payload))
    legacy = make().get(candidate)
    assert legacy is not None
    assert any("no metrics-schema digest" in w for w in legacy.warnings)
