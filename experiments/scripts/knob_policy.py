"""Fix one knob combination per deployment shape, from the 10 rps results.

`WORK_ORDER_rps_aware.md` rev 2 STEP 5, E6a. The RPS axis costs 14.6x a single
10 rps point, so E6a needs the candidate set smaller and the work order allows
exactly one way to do it: for each `(arch, backend_mix)`, keep the knob
combination that was feasible at 10 rps with the highest tok/J, and drop the
other five.

**This is a heuristic and every result computed under it must say so.** It
assumes the knob that wins at 10 rps also wins at 1 -- which is precisely the kind
of load-independence assumption this whole work order exists to distrust. Two
things bound the damage:

  * **Shapes with no feasible candidate at 10 rps keep all six knobs.** Every
    RNGD-involving configuration is in that bucket, and which knob brings one back
    at low load IS E6's question; fixing it from a run where none worked would
    answer the question by assumption.
  * The regret is measured, not asserted: E6a re-runs the 1 rps point with every
    cuda knob and reports what the fixing cost.

    knob_policy.py --cache outputs/.hp-reval-.../cache --service ... --cluster ...
                   --out outputs/e6/knob_policy_<fixture>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.candidate_generator import CandidateGenerator  # noqa: E402
from planner.inventory import (  # noqa: E402
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive  # noqa: E402
from planner.plan import PredictedMetrics  # noqa: E402
from planner.predictor import Predictor, SimOutcome, SimResult  # noqa: E402
from planner.predictor.calibration import load_accuracy_domains  # noqa: E402
from planner.spec import load_service_spec  # noqa: E402


def shape_of(candidate) -> str:
    """`(arch, backend_mix)` as one string, matching the sweep's own labels."""
    parts = sorted(
        f"{a.island_id.split('-', 1)[0]}:tp{a.tp_size}" for a in candidate.assignments
    )
    return f"{candidate.serving_arch.value}[{'+'.join(parts)}]"


def knobs_of(candidate) -> tuple[int, int]:
    return (candidate.knobs.max_num_seqs, candidate.knobs.max_num_batched_tokens)


class _Replay(Predictor):
    def __init__(self, records: dict):
        self.records = records

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        rec = self.records.get(candidate.id)
        if rec is None:
            return SimResult(candidate.id, SimOutcome.CRASHED, detail="not simulated")
        return SimResult(candidate.id, SimOutcome.OK,
                         metrics=PredictedMetrics.model_validate(rec["metrics"]),
                         operating_point=rec.get("operating_point", {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--service", required=True, type=Path)
    ap.add_argument("--cluster", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--ttft-ms", type=float, default=64000.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    records = {}
    for f in sorted(args.cache.glob("*.json")):
        d = json.loads(f.read_text())
        records[d["candidate_id"]] = d

    spec = load_service_spec(args.service).model_copy(deep=True)
    spec.slo.ttft.max_ms = args.ttft_ms
    cluster = load_cluster_spec(args.cluster)
    profiles = load_profiles_for(cluster, args.root)
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}

    generated = CandidateGenerator(
        spec, cluster, islands, profiles,
        enable_prefix_caching=False, enable_bound_pruning=True, enable_pd=True,
    ).generate()
    result = exhaustive.evaluate_candidates(
        generated.candidates, spec, cluster, by_id, profiles, _Replay(records),
        accuracy_domains=load_accuracy_domains(args.root),
    )

    by_shape: dict[str, list] = {}
    for c in generated.candidates:
        by_shape.setdefault(shape_of(c), []).append(c)
    feasible_tokj: dict[str, float] = {}
    feasible_knob: dict[str, tuple[int, int]] = {}
    for plan in result.feasible_plans:
        shape = shape_of(plan.candidate)
        tokj = plan.predicted.tokens_per_joule
        if tokj is None:
            continue
        if tokj > feasible_tokj.get(shape, float("-inf")):
            feasible_tokj[shape] = tokj
            feasible_knob[shape] = knobs_of(plan.candidate)

    policy = {}
    for shape, cands in sorted(by_shape.items()):
        all_knobs = sorted({knobs_of(c) for c in cands})
        if shape in feasible_knob:
            policy[shape] = {
                "fixed": True, "knob": list(feasible_knob[shape]),
                "tok_per_j_at_10rps": feasible_tokj[shape],
                "dropped": len(all_knobs) - 1,
            }
        else:
            policy[shape] = {
                "fixed": False, "knob": None,
                "reason": "no feasible candidate at 10 rps; which knob revives it "
                          "at low load is what E6 asks, so all are kept",
                "kept": len(all_knobs),
            }

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "cache": str(args.cache), "service": str(args.service),
        "cluster": str(args.cluster), "ttft_ms": args.ttft_ms,
        "label": "knob: fixed@10rps",
        "policy": policy,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    fixed = sum(1 for v in policy.values() if v["fixed"])
    kept = sum(v.get("kept", 1) for v in policy.values() if not v["fixed"])
    dropped = sum(v.get("dropped", 0) for v in policy.values() if v["fixed"])
    print(f"{len(policy)} shapes: {fixed} fixed to one knob ({dropped} candidates "
          f"dropped), {len(policy) - fixed} left open ({kept} knob slots kept)",
          file=sys.stderr)
    for shape, v in sorted(policy.items()):
        mark = f"s{v['knob'][0]}-t{v['knob'][1]}" if v["fixed"] else "ALL (none feasible)"
        print(f"  {shape:<48} {mark}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
