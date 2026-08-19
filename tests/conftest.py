"""Shared fixtures. The mock predictor is what makes the Phase 2 suite fast.

Work order §9: "시뮬레이터 subprocess는 테스트에서 mock 가능하게 predictor를
인터페이스 뒤에 둔다". A real simulation is ~60s; the oracle-agreement and
reproducibility tests need dozens of them, so they run against a deterministic
analytic stand-in instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.inventory import (
    AcceleratorProfile,
    ClusterSpecV2,
    ExecutionIsland,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.plan import CandidateConfig, PredictedMetrics
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.spec import ServiceSpec, load_service_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cluster() -> ClusterSpecV2:
    return load_cluster_spec(ROOT / "examples/clusters/heterogeneous-lab.yaml")


@pytest.fixture
def profiles(cluster: ClusterSpecV2) -> dict[str, AcceleratorProfile]:
    return load_profiles_for(cluster, ROOT)


@pytest.fixture
def islands(cluster, profiles) -> list[ExecutionIsland]:
    return detect_islands(cluster, profiles)


@pytest.fixture
def spec() -> ServiceSpec:
    return load_service_spec(ROOT / "examples/service_specs/llama31-8b.yaml")


#: How much worse than the memory roofline the mock pretends to be. Must be
#: >= 1.0: a predictor that beats the roofline would contradict the very bound
#: the oracle test is checking, and the test would fail for reasons that have
#: nothing to do with the planner.
MOCK_ROOFLINE_SLACK = 1.2


class MockPredictor(Predictor):
    """Deterministic predictor that respects the same physics as the bounds.

    An earlier version invented throughput independently of the memory roofline.
    That made the oracle-agreement test meaningless: the pruned search would
    reject a candidate on a real bandwidth bound while the mock cheerfully
    reported it as feasible, so the two disagreed by construction. A mock that
    contradicts the bounds cannot validate them.

    So latency here is derived from the real weight/KV sizes and the profile's
    memory bandwidth, then padded by a fixed slack. Absolute numbers are still
    fictional - only the ordering and the internal consistency matter.
    """

    def __init__(self, *, fail_ids: set[str] | None = None,
                 timeout_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.timeout_ids = timeout_ids or set()
        self.calls: list[str] = []

    def predict(self, candidate: CandidateConfig, spec, cluster, islands, profiles) -> SimResult:
        from planner.util import memory as memutil

        self.calls.append(candidate.id)
        if candidate.id in self.timeout_ids:
            return SimResult(candidate.id, SimOutcome.TIMEOUT, detail="mock timeout")
        if candidate.id in self.fail_ids:
            return SimResult(candidate.id, SimOutcome.CRASHED, detail="mock crash")

        seqs = candidate.knobs.max_num_seqs
        devices = candidate.total_devices
        median_len = spec.traffic.input_tokens.p50 + spec.traffic.output_tokens.p50

        from planner.plan import Role

        # Per-assignment roofline, then aggregate. Disaggregation is modelled
        # honestly: a prefill engine (Role.PREFILL) runs no decode steps, so its
        # roofline feeds TTFT, never TPOT, and it contributes no decode
        # throughput. TPOT comes from the decode/aggregated assignments only.
        # This mirrors the role-aware analytical bounds in candidate_generator -
        # a mock that charged prefill against TPOT would reject the very
        # slow-prefill/fast-decode candidates the bounds now (correctly) admit,
        # and the oracle-agreement test would validate a bug instead of catching
        # it. Aggregated/single/mixed carry only Role.AGGREGATED assignments, so
        # decode_tpot == max over all of them and behaviour is unchanged.
        decode_tpot: list[float] = []
        prefill_tpot: list[float] = []
        decode_tps = 0.0
        all_powered = True
        for a in candidate.assignments:
            island = islands[a.island_id]
            profile = profiles[island.accelerator_model]
            report = memutil.evaluate(
                candidate.model, tp_size=a.tp_size,
                device_memory_gb=island.total_memory_gb / island.size,
                dtype=candidate.dtype,
            )
            active = max(1, min(seqs, report.kv_tokens // max(1, median_len)))
            bytes_per_step = report.weight_bytes + active * report.kv_bytes_per_token
            roofline_ms = (bytes_per_step / (profile.memory_bandwidth_gbps * 1e9)) * 1e3
            latency_i = roofline_ms * MOCK_ROOFLINE_SLACK
            if a.role is Role.PREFILL:
                prefill_tpot.append(latency_i)
            else:
                decode_tpot.append(latency_i)
                decode_tps += (active / (latency_i / 1e3)) * a.dp_replicas
            if cluster.node(island.node_id).power is None:
                all_powered = False

        # A P/D candidate always carries exactly one decode assignment; fall back
        # to prefill only in the degenerate case of no decode assignment at all.
        tpot = max(decode_tpot) if decode_tpot else max(prefill_tpot)
        offered_tps = spec.traffic.arrival_rate_rps * spec.traffic.output_tokens.p50
        utilization = offered_tps / decode_tps if decode_tps > 0 else float("inf")
        attainment = min(1.0, 1.0 / utilization) if utilization > 0 else 1.0

        # Queueing: as offered load approaches capacity, waiting time runs away.
        # Without this an under-provisioned island looks merely slow instead of
        # infeasible, and the real simulator does show the blow-up (a saturated
        # A5000 measured 102s mean TTFT against 7s on an unsaturated card).
        queue_factor = 1.0 / max(0.02, 1.0 - min(utilization, 0.98))
        # TTFT is set by the prefill work; with disaggregation that is the
        # prefill engine's roofline, otherwise the (aggregated) decode roofline.
        prefill_ms = max(prefill_tpot) if prefill_tpot else tpot
        ttft = (prefill_ms * 8.0 + seqs * 0.5) * queue_factor

        tokens = 20_000
        # Mirror the real simulator: power modeling is disabled wholesale
        # unless every node has a `power:` block (config_builder.py:326), so a
        # partially-covered deployment yields no energy at all (deviations D14).
        energy = (1000.0 * devices + seqs * 2.0) if all_powered else None

        return SimResult(
            candidate.id,
            SimOutcome.OK,
            metrics=PredictedMetrics(
                p50_ttft_ms=ttft * 0.6,
                p95_ttft_ms=ttft * 0.9,
                p99_ttft_ms=ttft,
                p50_tpot_ms=tpot * 0.8,
                p95_tpot_ms=tpot * 0.95,
                p99_tpot_ms=tpot,
                throughput_tps=decode_tps,
                slo_goodput_rps=spec.traffic.arrival_rate_rps * attainment,
                slo_attainment=attainment,
                completed_requests=100,
                completed_tokens=tokens,
                total_energy_j=energy,
                average_power_w=None if energy is None else energy / 10.0,
                peak_power_w=None if energy is None else energy / 8.0,
                tokens_per_joule=None if energy is None else tokens / energy,
                sim_wall_seconds=0.0,
            ),
        )


@pytest.fixture
def mock_predictor() -> MockPredictor:
    return MockPredictor()
