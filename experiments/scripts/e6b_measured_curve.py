"""E6b: tokens/J against RPS from the MEASUREMENT, beside the simulator's answer.

`WORK_ORDER_rps_aware.md` rev 2 STEP 5. E6a is entirely simulation. E6b is the
half that is not: the STEP 3 RNGD curve gives tokens/J at five measured operating
points, and each point corresponds to an arrival rate the same way the solver
computes it (`throughput / mean output tokens`). Putting the two side by side asks
whether the simulator's error at each point stays inside what
`accuracy_domain` declares -- i.e. whether the domain is honest about itself.

**The asymmetry is the point and must survive into any sentence written from
this.** The RNGD side is measured silicon. The A40 side exists only in simulation,
because this node has no NVIDIA GPU (`scripts/whichnode.sh`), so a crossover
claim reads "RNGD **measured** against A40 **simulated**" and never "RNGD against
A40".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.perf_envelope import load_envelope  # noqa: E402
from planner.predictor.calibration import load_accuracy_domains  # noqa: E402

ENV = ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"
LOWLOAD = ROOT / "outputs/lowload_sim_error/lowload_sim_error.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/e6/e6b_measured.json")
    args = ap.parse_args()

    env = load_envelope(ENV)
    mean_out = env.measured_on_workload.output_tokens_mean
    assert mean_out is not None
    domain = load_accuracy_domains(ROOT)["RNGD-CARD"]
    sim_err = {round(r["conc_measured"], 2): r
               for r in json.loads(LOWLOAD.read_text()) if r.get("comparable")}

    rows = []
    for pt in env.points:
        if pt.power_w is None or pt.tok_per_j is None:
            continue                      # the D22 half has no power column
        rps = pt.tput_tok_s / mean_out
        rec = sim_err.get(round(pt.conc, 2))
        # The domain is indexed by SIMULATED served concurrency, because that is
        # what the planner knows when it consults the table. Querying it at the
        # MEASURED concurrency would be the wrong axis -- the two differ by up to
        # 19 % at these points, which is itself the throughput error.
        sim_conc = rec["conc_sim"] if rec else None
        declared = domain.tpot_error_at(sim_conc) if sim_conc is not None else None
        observed = rec["tpot_err_pct"] if rec else None
        rows.append({
            "measured_served_conc": pt.conc,
            "sim_served_conc": sim_conc,
            "rps_per_card": rps,
            "measured_tput_tok_s": pt.tput_tok_s,
            "measured_power_w": pt.power_w,
            "measured_tok_per_j": pt.tok_per_j,
            "measured_tpot_p50_ms": pt.tpot_p50,
            "domain_declared_tpot_err_pct": declared,
            "observed_sim_tpot_err_pct": observed,
            "agrees": (None if (observed is None or declared is None)
                       else abs(observed - declared) < 0.01),
        })

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "envelope": str(ENV.relative_to(ROOT)),
        "mean_output_tokens": mean_out,
        "unit": "one RNGD card, TP=8 internal",
        "measured_side": "RNGD: silicon (STEP 3). A40: SIMULATION ONLY -- no NVIDIA "
                         "GPU on this node, so no measured A40 curve exists.",
        "points": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'meas conc':>9} {'sim conc':>9} {'rps/card':>9} {'tok/J':>8} "
          f"{'declared':>10} {'observed':>10}  agrees", file=sys.stderr)
    for r in rows:
        def pct(v):
            return "-" if v is None else f"{v:+.2f}%"
        sc = "-" if r["sim_served_conc"] is None else f"{r['sim_served_conc']:.2f}"
        print(f"{r['measured_served_conc']:>9.2f} {sc:>9} {r['rps_per_card']:>9.4f} "
              f"{r['measured_tok_per_j']:>8.3f} "
              f"{pct(r['domain_declared_tpot_err_pct']):>10} "
              f"{pct(r['observed_sim_tpot_err_pct']):>10}  {r['agrees']}",
              file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
