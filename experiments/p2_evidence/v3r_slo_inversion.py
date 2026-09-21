#!/usr/bin/env python
"""S7.4: at which TPOT SLO does the CHOICE OF DOMAIN flip the verdict?

`WORK_ORDER_domain_scoping.md` STEP S7.4 rev 4. No simulation and no card: a
candidate's predicted p99 TPOT and its margin under each domain are fixed, so
only the threshold moves. E-A1's 50 ms aggregate is **not** re-scored and
nothing under `outputs/uncertainty/ea1/` is touched — this is the post-hoc
sensitivity analysis V3 §6.2 established.

**What S7.3 changed, and why this script exists.** rev 4 was written expecting
the open-loop domain to make the sim L 37.965 candidate *infeasible* where the
closed-loop one made it feasible. It does not, at the fixture's 50 ms: the
margin does triple, 7.51 % → 21.40 %, but a predicted 35.332 ms times 1.2140 is
42.89 ms and 42.89 < 50. The inversion is real and the threshold is simply
elsewhere, so the question becomes *where*, and that is measurable here.

**The band.** A candidate inverts over exactly

    [ T x (1 + m_closed),  T x (1 + m_open) )

— the closed-loop domain passes it and the open-loop one does not. Below the
band both pass; above it both fail.

**rev 4's 3x rule decides whether a band is usable.** A verdict is only ranked
where both sides clear three times the run-to-run p99 spread AT THAT OPERATING
POINT; inside that margin the verdict recorded is "indistinguishable", which is
a result and not a failure to produce one. The spread is measured (S7.3) and is
not one number — it is larger at lower load — so it is looked up per point and
never assumed.

**A candidate the open-loop domain REFUSES is not an inversion.** Outside the
measured range the policy is `refuse`, so there is no margin to compare: the
verdict is a hold, and a hold is reported as such rather than as a margin of
zero.

Run::

    PYTHONPATH=$PWD .venv/bin/python experiments/p2_evidence/v3r_slo_inversion.py \\
        --closed outputs/s74/reselect_closedloop.json \\
        --open   outputs/s74/reselect_openloop.json \\
        --out    experiments/p2_evidence/results/v3r_slo_inversion.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: Measured run-to-run p99 TPOT spread, by offered rps (S7.3, three runs each).
#: Absent means the point was run once and its spread is UNKNOWN -- which is not
#: the same as zero, and disqualifies the point from being ranked.
MEASURED_SPREAD_MS = {1.75: 0.776, 2.0: 0.116}

RULE = "c_accuracy_domain"

#: Stages at which the policy declines to judge rather than judging.
REFUSAL_STAGES = {"outside_calibration_domain", "calibration_condition_mismatch"}


def _margin(row: dict) -> float | None:
    return row["tpot_margin_pct"][RULE]


def analyse(closed: dict, openl: dict, candidate: str,
            threshold: float) -> list[dict]:
    cl = {r["rps"]: r for r in closed["rows"] if r["candidate_id"].endswith(candidate)}
    op = {r["rps"]: r for r in openl["rows"] if r["candidate_id"].endswith(candidate)}
    out = []
    for rps in sorted(op):
        o, c = op[rps], cl.get(rps)
        if c is None:
            continue
        m_open, m_closed = _margin(o), _margin(c)
        rec: dict = {
            "offered_rps": rps,
            "conc_sim": o["served_concurrency"],
            "operating_point": o["operating_point"],
            "tpot_p99_ms": o["p99_tpot_ms"],
            "margin_closed_pct": m_closed,
            "margin_open_pct": m_open,
            "stage_open": o["stages"][RULE],
            "spread_ms": MEASURED_SPREAD_MS.get(rps),
        }
        # A REFUSAL is detected by the stage, not by the margin. The policy
        # records `tpot_percent: 0.0` when it declines to judge, so testing the
        # margin for None reads a hold as "a margin of zero" and then as "the
        # open-loop domain is the looser of the two" -- the exact opposite of
        # what happened.
        if m_open is None or o["stages"][RULE] in REFUSAL_STAGES:
            rec["verdict"] = (
                f"no inversion: the open-loop domain REFUSES this operating "
                f"point ({o['stages'][RULE]}), so there is no margin to compare")
            rec["inversion_band_ms"] = None
            out.append(rec)
            continue
        t = o["p99_tpot_ms"]
        lo, hi = t * (1 + m_closed / 100.0), t * (1 + m_open / 100.0)
        rec["robust_closed_ms"], rec["robust_open_ms"] = lo, hi
        if hi <= lo:
            rec["inversion_band_ms"] = None
            rec["verdict"] = ("no inversion in this direction: the open-loop "
                              "margin is the looser of the two here")
            out.append(rec)
            continue

        rec["inversion_band_ms"] = [lo, hi]
        rec["band_width_ms"] = hi - lo
        spread = rec["spread_ms"]
        if spread is None:
            rec["verdict"] = ("band exists but this point was run ONCE: its "
                              "spread is unknown, so no verdict may be ranked "
                              "here (rev 4)")
            out.append(rec)
            continue
        need = threshold * spread
        usable_lo, usable_hi = lo + need, hi - need
        rec["spread_threshold_ms"] = need
        if usable_hi <= usable_lo:
            rec["verdict"] = (
                f"INDISTINGUISHABLE: the band is {hi - lo:.3f} ms but "
                f"{threshold:g}x the spread is {need:.3f} ms on each side, which "
                f"consumes it. Recorded as a verdict, not as a failure to reach "
                f"one (rev 4)")
            rec["usable_slo_ms"] = None
        else:
            mid = (lo + hi) / 2.0
            rec["usable_slo_ms"] = [usable_lo, usable_hi]
            rec["best_slo_ms"] = mid
            rec["separation_at_best_ms"] = mid - lo
            rec["separation_in_spreads"] = (mid - lo) / spread
            rec["verdict"] = (
                f"RANKABLE at SLO {mid:.3f} ms: closed-loop passes by "
                f"{mid - lo:.3f} ms, open-loop fails by {hi - mid:.3f} ms, "
                f"each {(mid - lo) / spread:.1f}x the measured spread")
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closed", type=Path, required=True)
    ap.add_argument("--open", dest="openl", type=Path, required=True)
    ap.add_argument("--candidate", default="s128-t2048")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="multiples of the run-to-run spread a verdict must "
                         "clear to be ranked (rev 4 fixes this at 3)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from planner.util import provenance as prov

    rows = analyse(json.loads(args.closed.read_text()),
                   json.loads(args.openl.read_text()),
                   args.candidate, args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"closed": str(args.closed), "open": str(args.openl),
         "candidate": args.candidate, "spread_threshold": args.threshold,
         "measured_spread_ms": MEASURED_SPREAD_MS,
         "provenance": prov.collect(), "points": rows}, indent=2) + "\n")

    print(f"{'rps':>5} {'sim L':>8} {'m_cl':>7} {'m_op':>7} {'band (SLO ms)':>22} "
          f"{'spread':>7}  verdict")
    for r in rows:
        band = ("—" if not r.get("inversion_band_ms")
                else f"[{r['inversion_band_ms'][0]:.3f}, {r['inversion_band_ms'][1]:.3f})")
        mo = "refused" if r["margin_open_pct"] is None else f"{r['margin_open_pct']:.2f}%"
        sp = "—" if r["spread_ms"] is None else f"{r['spread_ms']:.3f}"
        print(f"{r['offered_rps']:5g} {r['conc_sim']:8.3f} "
              f"{r['margin_closed_pct']:6.2f}% {mo:>7} {band:>22} {sp:>7}  "
              f"{r['verdict'][:60]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
