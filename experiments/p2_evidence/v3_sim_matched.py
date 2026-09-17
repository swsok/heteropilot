#!/usr/bin/env python
"""V3: the simulator's answer for P1 on EXACTLY the trace the hardware was given.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V3. P1's committed prediction comes
from E-A1, which ran a trace SYNTHESISED from the service spec's length
distributions. The hardware ran `workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl`.
The two are close -- input mean 875.4 against 857.5, output mean 636.4 against
652.5, offered rate 9.90 against 10.28 -- but "close" is not "the same", and V3's
headline is a 44 % TPOT error. This removes the confound by giving the simulator
the file the server was driven with.

Cache disabled, so the candidate is simulated on its own (D40 cannot apply).

Run:  .venv/bin/python experiments/p2_evidence/v3_sim_matched.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.candidate_generator import CandidateGenerator  # noqa: E402
from planner.inventory import (  # noqa: E402
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.predictor.llmservingsim import LLMServingSimPredictor  # noqa: E402
from planner.spec import load_service_spec  # noqa: E402
from planner.util.workload import WorkloadTrace  # noqa: E402

WORKLOAD = REPO / "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl"
CLUSTER = REPO / "experiments/configs/clusters/pd-rngd-gpu-card.yaml"
SERVICE = REPO / "examples/service_specs/llama31-8b.yaml"
WORK = REPO / "outputs/p2_evidence/v3_sim"
OUT = REPO / "experiments/p2_evidence/results/v3_sim_matched.json"
CANDIDATES = ("cuda-a40-node_a40a-tp4-dp1-s128-t2048",)


def main() -> int:
    spec = load_service_spec(SERVICE)
    cluster = load_cluster_spec(CLUSTER)
    profiles = load_profiles_for(cluster, REPO)
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}

    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines() if line.strip()]
    WORK.mkdir(parents=True, exist_ok=True)
    trace = WorkloadTrace(
        path=WORKLOAD, num_requests=len(rows), seed=42,
        total_input_tokens=sum(r["input_toks"] for r in rows),
        total_output_tokens=sum(r["output_toks"] for r in rows),
        horizon_s=max(r["arrival_time_ns"] for r in rows) / 1e9,
    )
    generated = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False,
    ).generate().candidates
    by_cid = {c.id: c for c in generated}

    report = {"trace": str(WORKLOAD.relative_to(REPO)), "cache_used": False,
              "candidates": []}
    for cid in CANDIDATES:
        candidate = by_cid.get(cid)
        if candidate is None:
            print(f"!! {cid} not generated")
            continue
        predictor = LLMServingSimPredictor(
            trace, work_dir=WORK / "matched", timeout_s=3600.0)
        print(f"--- simulating {cid} on the measured trace", flush=True)
        result = predictor.predict(candidate, spec, cluster, by_id, profiles)
        metrics = getattr(result, "metrics", None)
        m = asdict(metrics) if is_dataclass(metrics) else dict(metrics or {})
        outcome = getattr(result.outcome, "name", str(result.outcome))
        print(f"    outcome={outcome} L={m.get('served_concurrency')} "
              f"p99_tpot={m.get('p99_tpot_ms')} p99_ttft={m.get('p99_ttft_ms')}", flush=True)
        report["candidates"].append(
            {"candidate_id": cid, "outcome": outcome, "metrics": m})

    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
