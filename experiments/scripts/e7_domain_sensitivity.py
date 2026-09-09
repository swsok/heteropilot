"""E7: how much does the crossover move if the accuracy domain had fewer points?

`WORK_ORDER_rps_aware.md` rev 2 STEP 5, the optional sensitivity study, run
because E6 found a crossover. D11 quantified what profile-grid density costs
(~2.2 pp of end-to-end accuracy) by the same leave-one-out method; this asks the
analogous question of the *accuracy domain*, which is what sets every margin and
therefore every feasibility verdict.

The work order frames E7 against the ENVELOPE. E6a does not use the envelope --
`--envelope-prefilter` was off, because refusing candidates before simulating them
would have removed the very rows the switchover table is made of. What E6a does
use is the accuracy domain, so that is what is perturbed here. The substitution is
recorded rather than silent.

No simulation: every candidate's result is replayed from the sweep's own cache,
so the only thing that changes between runs is the domain.
"""

from __future__ import annotations

import argparse
import json
import re
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
from planner.optimizer import exhaustive, pareto  # noqa: E402
from planner.plan import PredictedMetrics  # noqa: E402
from planner.predictor import Predictor, SimOutcome, SimResult  # noqa: E402
from planner.predictor.calibration import (  # noqa: E402
    AccuracyDomain,
    load_accuracy_domains,
)
from planner.spec import load_service_spec  # noqa: E402


class _Replay(Predictor):
    def __init__(self, records: dict):
        self.records = records

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        rec = self.records.get(candidate.id)
        if rec is None:
            return SimResult(candidate.id, SimOutcome.CRASHED, detail="not in cache")
        return SimResult(candidate.id, SimOutcome.OK,
                         metrics=PredictedMetrics.model_validate(rec["metrics"]),
                         operating_point=rec.get("operating_point", {}))


def backends(mix: str) -> str:
    return "+".join(sorted(set(re.findall(r"([a-z0-9]+):tp\d+", mix))))


def winner(records, spec, cluster, by_id, profiles, domains, keep):
    cands = [c for c in CandidateGenerator(
        spec, cluster, list(by_id.values()), profiles, enable_prefix_caching=False,
        enable_bound_pruning=True, enable_pd=True).generate().candidates if keep(c)]
    res = exhaustive.evaluate_candidates(
        cands, spec, cluster, by_id, profiles, _Replay(records),
        accuracy_domains=domains)
    if not res.feasible_plans:
        return None
    best = pareto.rank(res.feasible_plans, spec.objective.primary)[0].plan
    return {
        "candidate": best.candidate.id,
        "backends": backends(" ".join(
            f"{a.island_id.split('-',1)[0]}:tp{a.tp_size}" for a in best.candidate.assignments)),
        "tok_per_j": best.predicted.tokens_per_joule,
        "margin": best.robust_margin_tpot_percent,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="pd-rngd-gpu-card")
    ap.add_argument("--hardware", default="RNGD-CARD",
                    help="which domain to thin out")
    ap.add_argument("--rps", default="1,3.3,10,20")
    ap.add_argument("--ttft-ms", type=float, default=64000.0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/e6/e7_domain_sensitivity.json")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "experiments/scripts"))
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "sw", ROOT / "experiments/scripts/pd_slo_sweep.py")
    sw = importlib.util.module_from_spec(sp)
    sys.modules["sw"] = sw
    sp.loader.exec_module(sw)
    keep = sw._knob_filter(
        f"fixed-from:{ROOT}/outputs/e6/knob_policy_{args.fixture}.json")

    cluster = load_cluster_spec(ROOT / f"experiments/configs/clusters/{args.fixture}.yaml")
    profiles = load_profiles_for(cluster, ROOT)
    by_id = {i.id: i for i in detect_islands(cluster, profiles)}
    base_domains = load_accuracy_domains(ROOT)
    full = base_domains[args.hardware]

    # one cache dir per rate is not how the sweep stored it; entries carry their
    # own trace digest, so split by it and match on the rate's trace.
    cache_dir = ROOT / f"outputs/e6/{args.fixture}/cache"
    work = ROOT / f"outputs/e6/{args.fixture}/work"
    from planner.util import provenance as prov
    by_digest: dict[str, dict] = {}
    for f in sorted(cache_dir.glob("*.json")):
        d = json.loads(f.read_text())
        by_digest.setdefault(d.get("trace_digest", ""), {})[d["candidate_id"]] = d

    variants: list[tuple[str, dict]] = [("full", base_domains)]
    for i, pt in enumerate(full.points):
        thinned = AccuracyDomain.model_validate({
            **full.model_dump(),
            "points": [p.model_dump() for j, p in enumerate(full.points) if j != i],
        })
        variants.append((f"without conc={pt.conc}", {**base_domains, args.hardware: thinned}))
    variants.append(("no domain at all",
                     {k: v for k, v in base_domains.items() if k != args.hardware}))

    rows = []
    for rate in [float(v) for v in args.rps.split(",")]:
        tag = str(rate).replace(".", "p")
        trace = work / f"trace_rps{tag}.jsonl"
        if not trace.exists():
            print(f"  no trace for rps {rate}; skipping", file=sys.stderr)
            continue
        records = by_digest.get(prov.hash_file(trace), {})
        spec = load_service_spec(
            ROOT / "examples/service_specs/llama31-8b.yaml").model_copy(deep=True)
        spec.traffic.arrival_rate_rps = rate
        spec.slo.ttft.max_ms = args.ttft_ms
        for label, domains in variants:
            w = winner(records, spec, cluster, by_id, profiles, domains, keep)
            rows.append({"rps": rate, "variant": label, "winner": w})
            b = ("INFEASIBLE" if w is None else
                 f"{w['backends']:<14} {w['tok_per_j']:.3f} tok/J  "
                 f"margin {w['margin']:.2f}%")
            print(f"  rps {rate:>5}  {label:<22} {b}", file=sys.stderr)

    base = {r["rps"]: r for r in rows if r["variant"] == "full"}
    flips = [r for r in rows if r["variant"] != "full" and (
        (r["winner"] is None) != (base[r["rps"]]["winner"] is None)
        or (r["winner"] and base[r["rps"]]["winner"]
            and r["winner"]["backends"] != base[r["rps"]]["winner"]["backends"]))]
    out = {"_note": __doc__.strip().splitlines()[0], "fixture": args.fixture,
           "hardware": args.hardware, "ttft_ms": args.ttft_ms,
           "rows": rows, "flips": flips}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{len(flips)} of {len(rows) - len(base)} thinned variants change the "
          f"winning backend", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
