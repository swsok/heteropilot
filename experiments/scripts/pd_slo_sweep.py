"""Sweep the TTFT SLO and find where a heterogeneous P/D split starts to win.

Answers the question the per-watt table cannot: RNGD beats every GPU here on both
GB/s per watt and GFLOP/s per watt, so pure efficiency says use RNGD for
everything and mixing strictly costs tokens/J. A mixed split can only win when a
CONSTRAINT binds -- and the binding one is TTFT, because single-request prefill
latency is capped by one island's compute and RNGD is far behind per device
(2048-token down_proj: 8513 us on one PE against 2010 us on an A40).

So the interesting quantity is the SLO threshold: below which TTFT does the
all-RNGD plan stop being feasible, forcing prefill onto the GPU? This driver
sweeps `slo.ttft_p99_ms` over a service spec, re-plans at each point, and reports
what the planner picks and what it costs.

Reuses the planner's own path -- generator, transfer cost, feasibility, ranking
(``exhaustive.evaluate_candidates``) -- so the answer is the one `plan` would
give, not a reimplementation. Candidates are simulated ONCE and cached, then
re-evaluated per SLO point: the SLO changes feasibility and ranking, not the
underlying performance.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/scripts/pd_slo_sweep.py \
        --service examples/service_specs/llama31-8b.yaml \
        --cluster experiments/configs/clusters/pd-rngd-gpu.yaml \
        --ttft-ms 500,1000,2000,4000,8000,16000,32000 \
        --output-dir outputs/.hp-pd-slo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from planner.envelope import EnvelopeCache
from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.predictor.calibration import load_accuracy_domains
from planner.predictor.llmservingsim import LLMServingSimPredictor
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.util import provenance as prov
from planner.util.workload import generate_trace

DEFAULT_SEED = 42
DEFAULT_NUM_REQUESTS = 300


def backend_mix(candidate, islands_by_id) -> str:
    """Label a candidate by the backends its prefill and decode roles land on."""
    roles: dict[str, set[str]] = {}
    for assignment in candidate.assignments:
        role = getattr(assignment.role, "value", str(assignment.role))
        island = islands_by_id.get(assignment.island_id)
        backend = island.backend if island is not None else "?"
        roles.setdefault(role, set()).add(f"{backend}:tp{assignment.tp_size}")

    def one(role: str) -> str:
        entries = roles.get(role) or set()
        return "+".join(sorted(entries)) if entries else "-"

    if "prefill" in roles or "decode" in roles:
        return f"P[{one('prefill')}] D[{one('decode')}]"
    return "agg[" + "+".join(sorted(e for s in roles.values() for e in s)) + "]"


def _knob_filter(spec_str: str):
    """`fixed-from:<json>` -> a predicate over candidates.

    A shape whose policy entry is `fixed` keeps only that knob pair. A shape left
    open -- every configuration with no feasible candidate at 10 rps -- keeps all
    of them, because which knob revives it at low load is the question E6 asks.
    A shape the table has never seen is kept, so a widened candidate space fails
    open rather than silently shrinking.
    """
    if not spec_str.startswith("fixed-from:"):
        raise SystemExit(f"--knob-policy must be 'fixed-from:<json>', got {spec_str!r}")
    table = json.loads(Path(spec_str.split(":", 1)[1]).read_text())["policy"]

    def shape_of(candidate) -> str:
        parts = sorted(f"{a.island_id.split('-', 1)[0]}:tp{a.tp_size}"
                       for a in candidate.assignments)
        return f"{candidate.serving_arch.value}[{'+'.join(parts)}]"

    def keep(candidate) -> bool:
        entry = table.get(shape_of(candidate))
        if entry is None or not entry.get("fixed"):
            return True
        want = tuple(entry["knob"])
        return (candidate.knobs.max_num_seqs,
                candidate.knobs.max_num_batched_tokens) == want

    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, type=Path)
    parser.add_argument("--cluster", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ttft-ms", default="500,1000,2000,4000,8000,16000,32000",
                        help="comma-separated TTFT SLO max_ms values to sweep; the "
                             "percentile stays whatever the service spec declares")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--tpot-margin-percent", type=float, default=0.0,
        help="inflate simulated p-TPOT by this %% before the feasibility check. "
             "Use it to encode a MEASURED model error rather than a guess: at the "
             "concurrency the card fixture runs at, the simulator is 18 %% "
             "optimistic on TPOT (deviations D22).")
    parser.add_argument(
        "--rps", default=None,
        help="comma-separated arrival rates. One sweep per rate; the trace is "
             "regenerated for each, which is also what separates their cache "
             "entries (the key carries the trace digest).",
    )
    parser.add_argument(
        "--knob-policy", default=None,
        help="fixed-from:<knob_policy.json> -- keep only the knob each shape won "
             "with at 10 rps. A heuristic; every result under it must carry the "
             "`knob: fixed@10rps` label.",
    )
    parser.add_argument(
        "--accuracy-domain", action="store_true",
        help="size each candidate's margin from its own operating point "
             "(STEP 4.3). E5 uses this INSTEAD of a manual margin: the question "
             "is whether the planner rejects D22's winner on its own.",
    )
    parser.add_argument(
        "--ttft-margin-percent", type=float, default=0.0,
        help="same, for TTFT.")
    # work-dir and cache-dir DERIVE from --output-dir unless given explicitly.
    # They used to default to literal outputs/.hp-pd-slo/{work,cache}, so passing a
    # different --output-dir moved only the summary JSON and left the simulations
    # and the envelope cache in the first run's directories. That was not a
    # correctness bug - the cache key includes the sim hardware name, so a
    # RNGD-CARD candidate cannot collide with an RNGD one, and the A40 candidates
    # SHOULD share entries because both fixtures declare identical A40 islands -
    # but it made a second sweep's per-candidate results unfindable, which is where
    # the interesting rows live.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/.hp-pd-slo"))
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="default: <output-dir>/work")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="default: <output-dir>/cache. Point two sweeps at one "
                             "cache deliberately to share candidates they have in "
                             "common; the key guards against false hits.")
    args = parser.parse_args()
    args.work_dir = args.work_dir or args.output_dir / "work"
    args.cache_dir = args.cache_dir or args.output_dir / "cache"

    ttft_points = sorted((float(v) for v in args.ttft_ms.split(",") if v), reverse=True)
    cluster = load_cluster_spec(args.cluster)
    islands = detect_islands(cluster)
    profiles = load_profiles_for(cluster, root=args.root)
    # The trace is fixed across the sweep on purpose: only the SLO moves, so the
    # workload must be byte-identical or the comparison is between two workloads.
    base_spec = load_service_spec(args.service)
    rps_points = ([float(v) for v in args.rps.split(",") if v.strip()]
                  if args.rps else [base_spec.traffic.arrival_rate_rps])
    knob_filter = _knob_filter(args.knob_policy) if args.knob_policy else None
    # Every measured accuracy domain under profiles/calibration/. Empty unless
    # --accuracy-domain, so the default sweep behaves exactly as before.
    accuracy_domains: dict = {}
    if args.accuracy_domain:
        accuracy_domains = load_accuracy_domains(args.root)
        print(f"accuracy domains: {sorted(accuracy_domains)}", file=sys.stderr)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # The cache key needs the reduction bandwidth the compiler will use, so the
    # sweep's cached entries match what `plan` would write.
    link_bw_gbps = TopologyGraph(cluster).reduce_for_simulator(islands).link_bw_gbps
    islands_by_id = {i.id: i for i in islands}
    print(f"{len(islands)} island(s): " + ", ".join(
        f"{i.id}({i.backend},{i.size})" for i in islands))
    print(f"sweeping slo.ttft.max_ms (p{base_spec.slo.ttft.percentile}) over {ttft_points} ms\n")

    rows = []
    for rps in rps_points:
      # A trace per RATE: the arrival times are what the rate changes, and the
      # digest goes into the cache key, so entries separate by rate on their own.
      rate_tag = str(rps).replace(".", "p")
      rate_spec = base_spec.model_copy(deep=True)
      rate_spec.traffic.arrival_rate_rps = rps
      trace = generate_trace(rate_spec, args.work_dir / f"trace_rps{rate_tag}.jsonl",
                             num_requests=args.num_requests, seed=args.seed)
      predictor = LLMServingSimPredictor(
          trace, work_dir=args.work_dir / f"rps{rate_tag}", timeout_s=args.timeout,
      )
      for ttft in ttft_points:
        spec = base_spec.model_copy(deep=True)
        spec.traffic.arrival_rate_rps = rps
        spec.slo.ttft.max_ms = ttft
        # One cache per SLO point would defeat the point; the key includes the
        # spec, so a shared root still reuses every simulation whose candidate is
        # unchanged. The SLO moves feasibility and ranking, not the metrics.
        cache = EnvelopeCache(
            args.cache_dir, spec,
            accelerator_of={i.id: i.accelerator_model for i in islands},
            link_bw_gbps=link_bw_gbps,
            trace_digest=prov.hash_file(trace.path),
        )
        started = time.monotonic()
        output = exhaustive.search(
            spec, cluster, islands, profiles, predictor,
            enable_pd=True, cache=cache, max_workers=args.workers,
            tpot_margin_percent=args.tpot_margin_percent,
            ttft_margin_percent=args.ttft_margin_percent,
            accuracy_domains=accuracy_domains,
            candidate_filter=knob_filter,
        )
        elapsed = time.monotonic() - started
        row = {
            "rps": rps,
            "wall_seconds": round(elapsed, 1),
            "knob_policy": ("fixed@10rps" if knob_filter else None),
            "ttft_slo_max_ms": ttft,
            "feasible": output.feasible,
            "generated": output.generated_candidates,
            "evaluated": output.evaluated_candidates,
            # Persisted because not persisting it is a recorded trap: this driver
            # printed INFEASIBLE identically whether candidates were REJECTED or
            # never evaluated, and on 2026-09-03 that hid 71 of 222 timeouts --
            # including the committed winner (docs/HANDOVER.md §3). A sweep row
            # without these counts is not a result.
            "rejected_summary": dict(output.rejected_summary),
            "cache_hits": output.provenance.get("envelope_cache_hits", 0),
            "recommended": None,
            "reason": output.reason,
        }
        best = output.recommended
        if best is not None:
            plan = best.plan
            flags = [r.in_calibration_domain for r in plan.operating_point]
            row["validity"] = (
                "unknown" if (not flags or any(f is None for f in flags))
                else "measured" if all(flags) else "extrapolated"
            )
            row["applied_tpot_margin_pct"] = plan.robust_margin_tpot_percent
            row["margin_source"] = plan.margin_source
            row["operating_point"] = [r.model_dump() for r in plan.operating_point]
        if best is not None:
            plan = best.plan
            cand = plan.candidate
            row["recommended"] = {
                "plan_id": plan.plan_id,
                "arch": getattr(cand.serving_arch, "value", str(cand.serving_arch)),
                "backend_mix": backend_mix(cand, islands_by_id),
                "accelerators": sum(
                    a.tp_size * a.pp_size * a.dp_replicas for a in cand.assignments),
                "tokens_per_joule": plan.predicted.tokens_per_joule,
                "slo_goodput_rps": plan.predicted.slo_goodput_rps,
                "p99_ttft_ms": plan.predicted.p99_ttft_ms,
                "p99_tpot_ms": plan.predicted.p99_tpot_ms,
                "average_power_w": plan.predicted.average_power_w,
                "slo_attainment": plan.predicted.slo_attainment,
            }
            r = row["recommended"]
            print(f"  TTFT <= {ttft:8.0f} ms : {r['backend_mix']:<26} "
                  f"{r['arch']:<12} n={r['accelerators']:<2} "
                  f"tok/J={r['tokens_per_joule']:8.3f} "
                  f"goodput={r['slo_goodput_rps']:6.2f} rps "
                  f"p99_ttft={r['p99_ttft_ms']:9.1f}")
        else:
            print(f"  TTFT <= {ttft:8.0f} ms : INFEASIBLE - {output.reason[:80]}")
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "pd_slo_sweep.json"
    out.write_text(json.dumps({
        "service": str(args.service), "cluster": str(args.cluster),
        "num_requests": args.num_requests, "seed": args.seed,
        "islands": [{"id": i.id, "backend": i.backend, "size": i.size} for i in islands],
        "sweep": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
