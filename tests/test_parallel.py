"""Concurrent candidate simulation helper (planner/util/parallel.py)."""

from __future__ import annotations

import time

from planner.plan import CandidateConfig, IslandAssignment, Role, ServingArch, VllmKnobs
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.util import parallel


class _SleepyPredictor(Predictor):
    """Returns a trivial ok result after a short sleep, to expose concurrency."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        time.sleep(self.delay)
        return SimResult(candidate_id=candidate.id, outcome=SimOutcome.OK)


def _cands(n: int) -> list[CandidateConfig]:
    return [
        CandidateConfig(
            id=f"c{i}", model="m", dtype="bfloat16",
            assignments=[IslandAssignment(island_id="isl", role=Role.AGGREGATED, tp_size=1)],
            serving_arch=ServingArch.AGGREGATED, knobs=VllmKnobs(),
        )
        for i in range(n)
    ]


def test_predict_all_returns_every_candidate_keyed_by_id():
    cands = _cands(8)
    out = parallel.predict_all(_SleepyPredictor(0.0), cands, None, None, {}, {}, max_workers=4)
    assert set(out) == {c.id for c in cands}
    assert all(out[c.id].candidate_id == c.id for c in cands)


def test_result_is_independent_of_worker_count():
    cands = _cands(6)
    seq = parallel.predict_all(_SleepyPredictor(0.0), cands, None, None, {}, {}, max_workers=1)
    par = parallel.predict_all(_SleepyPredictor(0.0), cands, None, None, {}, {}, max_workers=4)
    assert ({k: v.candidate_id for k, v in seq.items()}
            == {k: v.candidate_id for k, v in par.items()})


def test_parallel_is_faster_than_sequential():
    cands = _cands(8)
    t0 = time.monotonic()
    parallel.predict_all(_SleepyPredictor(0.1), cands, None, None, {}, {}, max_workers=1)
    seq_s = time.monotonic() - t0
    t0 = time.monotonic()
    parallel.predict_all(_SleepyPredictor(0.1), cands, None, None, {}, {}, max_workers=8)
    par_s = time.monotonic() - t0
    assert par_s < seq_s / 2  # 8 x 0.1s sequential ~0.8s; 8-wide ~0.1s


def test_empty_and_single_worker():
    assert parallel.predict_all(_SleepyPredictor(0.0), [], None, None, {}, {}) == {}
    cands = _cands(3)
    out = parallel.predict_all(_SleepyPredictor(0.0), cands, None, None, {}, {}, max_workers=1)
    assert set(out) == {c.id for c in cands}


def test_default_workers_is_sane():
    assert parallel.default_workers(1) == 1
    assert 1 <= parallel.default_workers(100) <= 32


def test_duplicate_ids_rejected():
    import pytest
    dup = _cands(1) * 2
    with pytest.raises(ValueError, match="unique candidate ids"):
        parallel.predict_all(_SleepyPredictor(0.0), dup, None, None, {}, {})


# --- cache dedup: same-key candidates must not be re-simulated (parallel) ------

def _two_a5000_islands():
    """Two single-A5000 nodes -> two islands of the SAME model, so a tp1 candidate
    on each collides on one envelope-cache key."""
    from planner.inventory import (
        Accelerator,
        AcceleratorProfile,
        AcceleratorType,
        ClusterSpecV2,
        Node,
        SupportedModel,
        detect_islands,
    )
    nodes = [
        Node(id=f"n{i}", accelerators=[Accelerator(
            id="gpu0", type=AcceleratorType.GPU, vendor="NVIDIA", model="RTX-A5000",
            backend="cuda", memory_gb=24.0)])
        for i in range(2)
    ]
    cluster = ClusterSpecV2(cluster_id="dup-key", nodes=nodes)
    profiles = {"RTX-A5000": AcceleratorProfile(
        profile_id="a5000", vendor="NVIDIA", model="RTX-A5000", backend="cuda",
        memory_gb=24.0, memory_bandwidth_gbps=768.0, sim_hardware="A5000",
        supported_models=[SupportedModel(pattern="*", dtypes=["bfloat16"])], max_tp_size=1)}
    return cluster, detect_islands(cluster, profiles), profiles


def test_cache_dedups_same_key_across_islands(spec, mock_predictor, tmp_path):
    """With a real EnvelopeCache, two same-model-island candidates share a key:
    the second must be served as a cache hit, not re-simulated - byte-identical to
    the sequential loop and without the redundant sim."""
    from planner.candidate_generator import CandidateGenerator
    from planner.envelope import EnvelopeCache
    from planner.optimizer import exhaustive

    cluster, islands, profiles = _two_a5000_islands()
    islands_by_id = {i.id: i for i in islands}
    cands = CandidateGenerator(spec, cluster, islands, profiles,
                               enable_prefix_caching=False).generate().candidates

    class _Counting(Predictor):
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def predict(self, candidate, s, c, i, p) -> SimResult:
            self.calls += 1
            return self.inner.predict(candidate, s, c, i, p)

    # distinct keys present? (two islands, same model -> at least one collision)
    keys = {}
    cache_probe = EnvelopeCache(tmp_path / "probe", spec,
                                accelerator_of={i.id: i.accelerator_model for i in islands},
                                link_bw_gbps=100.0)
    for cand in cands:
        keys.setdefault(cache_probe.cache_key(cand), []).append(cand.id)
    assert any(len(v) > 1 for v in keys.values()), "test needs a key collision"

    def _run(workers):
        pred = _Counting(mock_predictor)
        cache = EnvelopeCache(tmp_path / f"c{workers}", spec,
                              accelerator_of={i.id: i.accelerator_model for i in islands},
                              link_bw_gbps=100.0)
        ev = exhaustive.evaluate_candidates(cands, spec, cluster, islands_by_id, profiles,
                                            pred, cache=cache, max_workers=workers)
        return pred.calls, sorted(ev.cache_hits), ev.evaluated

    seq = _run(1)
    par = _run(4)
    assert seq == par                      # byte-identical accounting, sequential vs parallel
    assert seq[0] < len(cands)             # colliding candidate was NOT re-simulated
    assert len(seq[1]) >= 1                # at least one within-run cache hit recorded
