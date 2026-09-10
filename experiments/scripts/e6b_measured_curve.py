"""E6b: tokens/J against RPS from the MEASUREMENT, beside the simulator's answer.

`WORK_ORDER_rps_aware.md` rev 2 STEP 5. E6a is entirely simulation. E6b is the
half that is not: the STEP 3 RNGD curve gives tokens/J at five measured operating
points, and each point corresponds to an arrival rate the same way the solver
computes it (`throughput / mean output tokens`). Putting the two side by side asks
whether the simulator's error at each point stays inside what
`accuracy_domain` declares -- i.e. whether the domain is honest about itself.

**The asymmetry was the point and it is now half closed.** The RNGD side is
measured silicon. The A40 side was simulation only, because the node this script
was written on has no NVIDIA GPU -- so a crossover claim had to read "RNGD
**measured** against A40 **simulated**". Since 2026-09-10 an A40 curve exists
too, measured on the A40 node (`docs/HANDOVER.md` §2.2), and `--hardware A40`
emits it.

**The two curves are not interchangeable and this script will not pretend they
are.** The RNGD curve is CLOSED-LOOP (a fixed number in flight); the A40 curve is
OPEN-LOOP (an arrival trace replayed), because that is how the A40 accuracy
domain's existing point was made and a domain must be internally consistent
before it is externally comparable. Every row carries `closed_loop`, and a
sentence putting the two devices on one axis has to say which protocol each side
was measured under. `scripts/whichnode.sh` still decides what may be claimed
from where: neither curve may be relabelled as the other node's.
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

#: Per hardware: the measured envelope, the simulator-error file to read beside
#: it, and what one unit of the curve IS. Adding a device is a row, not a branch.
FIXTURES = {
    "RNGD-CARD": {
        "envelope": "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml",
        "lowload": "outputs/lowload_sim_error/lowload_sim_error.json",
        "unit": "one RNGD card, TP=8 internal",
    },
    "A40": {
        "envelope": "profiles/envelopes/A40/meta-llama/Llama-3.1-8B/bf16/tp1.yaml",
        "lowload": "outputs/a40_lowload_sim_error/lowload_sim_error.json",
        "unit": "one A40, TP=1",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", choices=sorted(FIXTURES), default="RNGD-CARD")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: outputs/e6/e6b_measured[_<hw>].json")
    args = ap.parse_args()

    fixture = FIXTURES[args.hardware]
    env_path = ROOT / fixture["envelope"]
    lowload_path = ROOT / fixture["lowload"]
    if args.out is None:
        suffix = "" if args.hardware == "RNGD-CARD" else f"_{args.hardware.lower()}"
        args.out = ROOT / f"outputs/e6/e6b_measured{suffix}.json"

    env = load_envelope(env_path)
    mean_out = env.measured_on_workload.output_tokens_mean
    assert mean_out is not None
    domains = load_accuracy_domains(ROOT)
    if args.hardware not in domains:
        raise SystemExit(
            f"no accuracy domain for {args.hardware}: a measured curve without one "
            f"cannot be checked against what the predictor claims about itself. "
            f"Rule 3 -- borrowing another device's domain is not an option."
        )
    domain = domains[args.hardware]
    sim_err = {round(r["conc_measured"], 2): r
               for r in json.loads(lowload_path.read_text()) if r.get("comparable")}

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
            "closed_loop": env.closed_loop,
            "domain_declared_tpot_err_pct": declared,
            "observed_sim_tpot_err_pct": observed,
            "agrees": (None if (observed is None or declared is None)
                       else abs(observed - declared) < 0.01),
        })

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "hardware": args.hardware,
        "envelope": str(env_path.relative_to(ROOT)),
        "mean_output_tokens": mean_out,
        "unit": fixture["unit"],
        # Which protocol produced the curve. Two devices measured under different
        # protocols may be reported side by side only with this attached.
        "closed_loop": env.closed_loop,
        "measured_side": (
            f"{args.hardware}: measured silicon. Curves for different hardware in "
            f"this directory may have been taken under different load protocols "
            f"(closed_loop above) and on different nodes; neither may be "
            f"relabelled as the other's."),
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
