#!/usr/bin/env python
"""V3: the four rules' verdicts for P1 across a sweep of TPOT SLOs.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V3, extension agreed 2026-09-16.

V3 can deploy only P1 -- the deploy backend refuses multi-island plans and
`dp_replicas > 1`, which blocks every P3 candidate and the informative P2 one.
One candidate under one SLO yields one cell, and four rules that all agree on it
yield nothing at all. Sweeping the TPOT SLO turns that single deployment into a
grid: the rules' robust values are fixed, the threshold moves, and the
measurement says which side of it the hardware actually landed on.

**The SLO is a post-hoc parameter here and the results file says so.** E-A1's
50 ms aggregate is NOT touched: this sweep asks what the same four rules would
have decided at other thresholds, which is a sensitivity analysis, not a
re-scoring of the experiment.

Why P1's TTFT never binds, which is what makes the sweep clean: its predicted
p99 TTFT is 16 159.5 ms against a 25 000 ms SLO, and the largest TTFT margin any
rule applies is the global 18 % -> 19 068 ms, still under. So TPOT alone decides
every cell.

Run:  .venv/bin/python experiments/p2_evidence/v3_slo_sweep.py [--measured MS]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_select_candidates import RULES, load, row_for  # noqa: E402

P1 = "cuda-a40-node_a40a-tp4-dp1-s128-t2048"
#: The thresholds swept. Chosen to bracket the rules' robust values (36.79 to
#: 43.42), which is where they can disagree; 50 is E-A1's own and anchors the
#: sweep to the committed experiment.
SLO_GRID_MS = (38.0, 40.0, 42.0, 44.0, 46.0, 50.0)
TTFT_SLO_MS = 25000.0
LABELS = {"a_margin0": "(a) no margin", "b_global18": "(b) global 18 %",
          "c_accuracy_domain": "(c) per-point", "d_refuse": "(d) per-point + refuse",
          "e_condition_refuse": "(e) + condition refuse"}

#: Rule (e) (domain-scoping S1/S4, D110) is NOT a fifth threshold comparison.
#: A condition mismatch is decided before any margin exists: no domain was
#: measured under this candidate's configuration, so none may be consulted, and
#: there is no robust value to compare against an SLO. Its cell is therefore the
#: same at every threshold, which is the point - the sweep's other four columns
#: move with the grid and this one cannot.
E_RULE = "e_condition_refuse"


def robust_values(candidate: str = P1, ea1_path=None) -> dict:
    ea1, cache = load(ea1_path)
    rules = RULES + ((E_RULE,) if E_RULE in ea1["conditions"] else ())
    row = row_for(candidate, ea1, cache, rules)
    out = {"candidate": candidate,
           "predicted_p99_tpot_ms": row["p99_tpot_ms"],
           "predicted_p99_ttft_ms": row["p99_ttft_ms"],
           "predicted_concurrency": row["served_concurrency"], "rules": {}}
    for rule in RULES:
        mt = row["tpot_margin_pct"][rule]
        mf = row["ttft_margin_pct"][rule]
        out["rules"][rule] = {
            "label": LABELS[rule],
            "tpot_margin_pct": mt, "ttft_margin_pct": mf,
            "robust_tpot_ms": row["p99_tpot_ms"] * (1 + mt / 100.0),
            "robust_ttft_ms": row["p99_ttft_ms"] * (1 + mf / 100.0),
        }
    if E_RULE in rules:
        out["held"] = {
            "label": LABELS[E_RULE],
            "stage": row["stages"][E_RULE],
            "reason": row["reasons"][E_RULE],
        }
    return out


def sweep(base: dict, measured_tpot_ms: float | None) -> list[dict]:
    """One row per SLO: each rule's verdict, and whether the hardware agreed."""
    rows = []
    for slo in SLO_GRID_MS:
        truth = None if measured_tpot_ms is None else measured_tpot_ms <= slo
        cells = {}
        for rule, r in base["rules"].items():
            # TTFT is checked too, and never binds for this candidate -- asserted
            # rather than assumed, so a future candidate cannot slip through.
            passes = r["robust_tpot_ms"] <= slo and r["robust_ttft_ms"] <= TTFT_SLO_MS
            cell = {"verdict": "feasible" if passes else "rejected",
                    "bound_by": ("ttft" if r["robust_ttft_ms"] > TTFT_SLO_MS
                                 else "tpot" if not passes else None)}
            if truth is not None:
                cell["outcome"] = (
                    "correct pass" if passes and truth else
                    "FALSE PASS" if passes and not truth else
                    "FALSE REJECTION" if not passes and truth else
                    "correct rejection")
            cells[rule] = cell
        if "held" in base:
            # Neither a pass nor a rejection: the candidate is UNMEASURED at its
            # own configuration, so the rule declines to answer and there is no
            # outcome to score against the measurement (S1, D110).
            cells[E_RULE] = {"verdict": "held", "bound_by": None,
                             "outcome": "held (condition mismatch)"}
        rows.append({"tpot_slo_ms": slo, "measured_satisfies": truth, "rules": cells})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured", type=float, default=None,
                    help="measured p99 TPOT in ms; omit to print the rule side only")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="where to write the record. The default is V3's own "
                         "COMMITTED artifact, so a re-run that asks a different "
                         "question must redirect it - the same trap "
                         "eb1_regret_vs_budget.py sprang on 2026-09-17, which "
                         "cost a `git checkout` to undo (docs/HANDOVER.md §3)")
    ap.add_argument("--ea1", type=Path, default=None,
                    help="an E-A1 record other than the committed one. S4 passes "
                         "its re-run, which reproduces (a)-(d) exactly and adds "
                         "rule (e); the committed file stays as V3 published it")
    args = ap.parse_args()

    if args.ea1 is not None and args.out_json is None:
        print("error: --ea1 asks a different question from the committed sweep, so "
              "--out-json is required; writing this run over "
              "experiments/p2_evidence/results/v3_slo_sweep.json would replace "
              "V3's published record with one its own text does not describe",
              file=sys.stderr)
        return 1

    base = robust_values(ea1_path=args.ea1)
    print(f"candidate      : {base['candidate']}")
    print(f"predicted p99  : TPOT {base['predicted_p99_tpot_ms']:.5f} ms, "
          f"TTFT {base['predicted_p99_ttft_ms']:.1f} ms, L {base['predicted_concurrency']:.4f}")
    print(f"TTFT SLO       : {TTFT_SLO_MS:.0f} ms (never binds for this candidate)\n")
    print(f"{'rule':24s} {'tpot margin':>12s} {'robust TPOT':>12s} {'robust TTFT':>12s}")
    for r in base["rules"].values():
        print(f"{r['label']:24s} {r['tpot_margin_pct']:11.4f}% {r['robust_tpot_ms']:12.5f} "
              f"{r['robust_ttft_ms']:12.1f}")

    if "held" in base:
        print(f"\n{base['held']['label']:24s} {base['held']['stage']}")
        print(f"  {base['held']['reason'][:150]}")

    rows = sweep(base, args.measured)
    shown = RULES + ((E_RULE,) if "held" in base else ())
    print(f"\n{'TPOT SLO':>9s} " + " ".join(f"{LABELS[r]:>22s}" for r in shown)
          + ("   measured satisfies?" if args.measured else ""))
    for row in rows:
        cells = " ".join(
            f"{(row['rules'][r].get('outcome') or row['rules'][r]['verdict']):>22s}"
            for r in shown)
        tail = ("" if row["measured_satisfies"] is None
                else f"   {'YES' if row['measured_satisfies'] else 'no'}")
        print(f"{row['tpot_slo_ms']:9.0f} {cells}{tail}")

    out = args.out_json or REPO / "experiments/p2_evidence/results/v3_slo_sweep.json"
    out.write_text(json.dumps(
        {"base": base, "ttft_slo_ms": TTFT_SLO_MS, "slo_grid_ms": list(SLO_GRID_MS),
         "measured_p99_tpot_ms": args.measured, "sweep": rows}, indent=2) + "\n")
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
