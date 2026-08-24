"""Router baseline plumbing (work order §12).

The end-to-end sweep (experiments/scripts/exp_router.py) needs a real simulator;
these fast tests pin the two things that must hold without one:
- the predictor's routing policy defaults to LOAD (so Phase-2 output stays
  byte-identical) and is settable to RR/RAND/LOAD;
- the driver picks an aggregated, multi-replica, simulatable candidate
  deterministically (router choice is a no-op for single-replica deployments).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scripts"))

import exp_router as xr

from planner.candidate_generator import CandidateGenerator
from planner.plan import ServingArch
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.util.workload import WorkloadTrace


def _trace(tmp_path) -> WorkloadTrace:
    p = tmp_path / "wl.jsonl"
    p.write_text('{"input_toks": 8, "output_toks": 4, "arrival_time_ns": 0}\n')
    return WorkloadTrace(path=p, num_requests=1, seed=42,
                         total_input_tokens=8, total_output_tokens=4, horizon_s=1.0)


def test_routing_policy_defaults_to_load(tmp_path):
    pred = LLMServingSimPredictor(_trace(tmp_path), work_dir=tmp_path / "w")
    assert pred.routing_policy == "LOAD"  # byte-identical to pre-change behavior
    pred.close()


def test_routing_policy_is_settable(tmp_path):
    for policy in ("RR", "RAND", "LOAD"):
        pred = LLMServingSimPredictor(_trace(tmp_path), work_dir=tmp_path / f"w_{policy}",
                                      routing_policy=policy)
        assert pred.routing_policy == policy
        pred.close()


def test_pick_multi_replica_is_aggregated_and_deterministic(spec, cluster, islands, profiles):
    islands_by_id = {i.id: i for i in islands}
    gen = CandidateGenerator(
        spec, cluster, islands, profiles,
        max_num_seqs=(256,), max_num_batched_tokens=(2048,), enable_prefix_caching=False,
    ).generate()
    picks = [xr._pick_multi_replica(gen.candidates, islands_by_id, profiles) for _ in range(3)]
    assert picks[0] is not None
    assert picks[0].id == picks[1].id == picks[2].id            # deterministic
    assert picks[0].serving_arch is ServingArch.AGGREGATED
    assert sum(a.dp_replicas for a in picks[0].assignments) >= 2  # router has work to do
    # every touched island must be simulatable (placeholder profiles excluded)
    for a in picks[0].assignments:
        assert profiles[islands_by_id[a.island_id].accelerator_model].sim_hardware is not None


def test_pick_multi_replica_none_when_single_replica_only(spec, cluster, islands, profiles):
    """A single-device candidate set yields no router-relevant pick."""
    islands_by_id = {i.id: i for i in islands}
    singles = [
        c for c in CandidateGenerator(
            spec, cluster, islands, profiles,
            max_num_seqs=(256,), max_num_batched_tokens=(2048,), enable_prefix_caching=False,
        ).generate().candidates
        if sum(a.dp_replicas for a in c.assignments) == 1
    ]
    assert xr._pick_multi_replica(singles, islands_by_id, profiles) is None
