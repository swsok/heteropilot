#!/usr/bin/env python
"""V2: the simulator's answer for the same arrival process the hardware was given.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V2. Claim 2 says the verification
information must be looked up at the PREDICTED served concurrency. Every
committed point makes that untestable: they are closed-loop, or arrival-rate
matched, so `L_pred ~ L_meas` holds by construction. Under an open loop the
arrival process is fixed and the concurrency is an outcome, so the two can
diverge -- by Little's law an optimistic predictor lands below the measurement.

This runs the simulator on **the same rescaled arrival times** the driver
offered, for the configuration that was actually deployed, and records `L_pred`
beside the measured `L_meas`.

**Run it only when the hardware ladder is finished.** ASTRA-Sim is CPU-bound and
the driver needs the event loop to fire within 20 ms of schedule; running both at
once measures the contention, not the server.

Run:  .venv/bin/python experiments/p2_evidence/v2_sim_side.py
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
CLUSTER = REPO / "experiments/configs/clusters/a40x8.yaml"
SERVICE = REPO / "examples/service_specs/llama31-8b.yaml"
MEASURED = REPO / "outputs/p2_evidence/v2"
WORK = REPO / "outputs/p2_evidence/v2_sim"
OUT = REPO / "experiments/p2_evidence/results/v2_sim_side.json"

#: What `deploy` launched: island cuda-a40-node0-0, TP=1, DP=1, and the plan's
#: knobs (max_num_seqs 128, max_num_batched_tokens 2048).
CANDIDATE_ID = "cuda-a40-node0-0-tp1-dp1-s128-t2048"


def rescale_trace(rates: list[float]) -> dict[float, Path]:
    """One trace per offered rate, rescaled exactly as the driver rescales it.

    The driver multiplies every arrival offset by `source / target`, which keeps
    a Poisson process Poisson and divides the rate exactly. The simulator has to
    be given the SAME schedule, not merely the same mean rate, or the comparison
    is between two different arrival processes.
    """
    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines() if line.strip()]
    offsets = [r["arrival_time_ns"] / 1e9 for r in rows]
    source = len(offsets) / max(offsets)
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for rate in rates:
        factor = source / rate
        path = WORK / f"trace_rps{rate}.jsonl"
        with path.open("w") as fh:
            for row, off in zip(rows, offsets, strict=True):
                copy = dict(row)
                copy["arrival_time_ns"] = int(off * factor * 1e9)
                fh.write(json.dumps(copy) + "\n")
        out[rate] = path
        print(f"  rps {rate:>5}: factor {factor:.4f}, span "
              f"{max(offsets) * factor:.1f}s -> {path.name}")
    return out


def measured_rates() -> list[float]:
    """The rates the hardware actually ran, read off the driver's own output."""
    rates = set()
    for path in sorted(MEASURED.glob("rps*_r*.json")):
        rates.add(float(json.loads(path.read_text())["offered_rps"]))
    return sorted(rates)


def main() -> int:
    rates = measured_rates()
    if not rates:
        print(f"no measured stages in {MEASURED}; run the ladder first")
        return 1
    print(f"rates measured on hardware: {rates}\nrescaling traces:")
    traces = rescale_trace(rates)

    spec = load_service_spec(SERVICE)
    cluster = load_cluster_spec(CLUSTER)
    profiles = load_profiles_for(cluster, REPO)
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}

    candidates = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False,
    ).generate().candidates
    candidate = next((c for c in candidates if c.id == CANDIDATE_ID), None)
    if candidate is None:
        print(f"!! {CANDIDATE_ID} not generated. Available on island 0:")
        for c in candidates[:15]:
            if c.id.startswith("cuda-a40-node0-0-"):
                print("   ", c.id)
        return 1

    report = {"candidate_id": CANDIDATE_ID, "cluster": str(CLUSTER.relative_to(REPO)),
              "service": str(SERVICE.relative_to(REPO)), "stages": []}
    for rate, path in traces.items():
        rows = path.read_text().splitlines()
        trace = WorkloadTrace(
            path=path, num_requests=len(rows), seed=42,
            total_input_tokens=sum(json.loads(r)["input_toks"] for r in rows),
            total_output_tokens=sum(json.loads(r)["output_toks"] for r in rows),
            horizon_s=max(json.loads(r)["arrival_time_ns"] for r in rows) / 1e9,
        )
        predictor = LLMServingSimPredictor(
            trace, work_dir=WORK / f"sim_rps{rate}", timeout_s=3600.0,
        )
        print(f"--- simulating rps {rate}", flush=True)
        result = predictor.predict(candidate, spec, cluster, by_id, profiles)
        metrics = getattr(result, "metrics", None)
        m = (asdict(metrics) if is_dataclass(metrics) else dict(metrics or {}))
        outcome = getattr(result.outcome, "name", str(result.outcome))
        print(f"    outcome={outcome} L_pred={m.get('served_concurrency')} "
              f"p99_ttft={m.get('p99_ttft_ms')} p99_tpot={m.get('p99_tpot_ms')}",
              flush=True)
        report["stages"].append({"offered_rps": rate, "outcome": outcome,
                                 "trace": str(path.relative_to(REPO)), "metrics": m})

    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
