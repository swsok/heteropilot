#!/usr/bin/env python
"""V1: the disclosure's §5.4 two-stage validation-region procedure, on the measured points.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V1. The disclosure describes a
procedure and has never run it on data. This does, for three calibration /
validation placements and on two residual bases, and reports what registers.

**Nothing in `planner/` or `profiles/` is touched.** Whether the procedure should
move into the planner is a decision to take after reading the result.

The procedure, as §5.4 states it:

  Stage 1 (is the interval admissible at all?) -- an interval between two
  adjacent CALIBRATION points registers only if the concurrency ratio is at most
  `RATIO_MAX`, both endpoints carry at least `N_MIN` samples, and the two
  endpoints are in the same operating mode.

  Stage 2 (does the margin actually cover?) -- among the VALIDATION points inside
  the interval, the fraction whose measured latency exceeds the corrected
  prediction `prediction x (1 + m)` must not exceed `TOLERANCE`.

  The margin `m` is a non-negative upper quantile of the interval's calibration
  residuals plus slack.

Run:  .venv/bin/python experiments/p2_evidence/v1_region.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.util.percentile import percentile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_percentile_audit import audit  # noqa: E402

# ---------------------------------------------------------------------------
# Thresholds. Every one is a choice; the reasoning is in v1_validation_region.md
# and repeated here so the code is readable without it.
# ---------------------------------------------------------------------------

#: §5.4's adjacency condition. Two points more than a doubling apart cannot
#: support a linear claim about what happens between them.
RATIO_MAX = 2.0

#: Smallest measured sample count an endpoint may carry. The only committed
#: point below it is c16.6, a five-sample bucket fit -- five samples cannot
#: support a distributional claim, and every genuine comparison has n >= 128.
N_MIN = 100

#: §5.4's exceedance allowance. NOTE what this means at these sizes: with k
#: validation points inside an interval, 5 % permits floor(0.05 k) failures,
#: which is ZERO for every k < 20. No interval here has 20 validation points, so
#: the condition is exactly "no validation point may be left uncovered".
TOLERANCE = 0.05

#: Added to the margin, in percentage points. Zero is the primary case because
#: it is the STRICTEST: a larger margin raises the corrected latency and can only
#: reduce exceedances, so an interval that registers at slack 0 registers at any
#: larger slack. Failing intervals report the minimum slack that would register
#: them instead of being given one.
SLACK_PP = 0.0

#: Quantile of the interval's calibration residuals used as the margin. With two
#: endpoints this is the maximum, which is what §5.4's "upper quantile" reduces
#: to for an interval defined by its two ends.
MARGIN_QUANTILE = 100.0

#: The simulator's own throughput ceiling on this card: D32 measured it settling
#: at served 37.67 and 44.47 when offered the rates that put the hardware at 59.2
#: and 107.2. A point above it is in a different operating mode from one below.
SATURATION_CONC = 44.0


def r_from_e(e_pct: float) -> float:
    """Disclosure convention from file convention, `r = -e / (1 + e)`, percent."""
    return -(e_pct / 100.0) / (1.0 + e_pct / 100.0) * 100.0


def margin_of(e_pct: float) -> float:
    """One-sided: a pessimistic simulator earns no margin."""
    return max(0.0, r_from_e(e_pct))


@dataclass
class Pt:
    conc: float
    e_pct: float
    r_pct: float
    margin_pct: float
    n: int
    prediction_ms: float | None
    measured_ms: float | None
    mode: str
    basis: str
    #: Per-request measured TPOT, when the raw records are available. Stage 2's
    #: by-request exceedance needs the distribution, not just its p99.
    measured_requests: list[float] | None = None


def _mode(conc: float) -> str:
    return "saturated" if conc >= SATURATION_CONC else "unsaturated"


def load_points(basis: str) -> list[Pt]:
    """The nine RNGD-CARD points on one residual basis.

    `committed` is what the domain file stores. `p99` recomputes both sides on
    the percentile `feasibility.py` applies the margin to -- reported, never
    written back (rule A3).
    """
    pts: list[Pt] = []
    for row in audit():
        if basis == "committed":
            e = row["stored_e"]
        else:
            e = row["e_recomputed_p99"]
            if e is None:
                continue  # c16.6 has no per-request pair to recompute from
        pred = row.get("sim_p99") if basis == "p99" else row.get("sim_p50")
        meas = row.get("measured_p99") if basis == "p99" else row.get("measured_used")
        pts.append(Pt(
            conc=row["conc"], e_pct=e, r_pct=r_from_e(e), margin_pct=margin_of(e),
            n=row["n_measured"], prediction_ms=pred, measured_ms=meas,
            mode=_mode(row["conc"]), basis=basis,
        ))
    return sorted(pts, key=lambda p: p.conc)


# ---------------------------------------------------------------------------
# Placements
# ---------------------------------------------------------------------------

def placement(kind: str, pts: list[Pt]) -> tuple[list[Pt], list[Pt]]:
    """Split the points into (calibration, validation)."""
    if kind == "A":  # every point calibrates; §5.6's illustrative arrangement
        return pts, []
    if kind == "B":  # §5.7's variant: the two ends calibrate, the interior validates
        return [pts[0], pts[-1]], pts[1:-1]
    if kind == "C":  # leave-one-out: alternate along the curve
        return pts[0::2], pts[1::2]
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# The two stages
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    lo: float
    hi: float
    registered: bool
    stage1: dict
    stage2: dict
    margin_pct: float
    min_slack_to_register_pp: float | None = None


def _stage1(a: Pt, b: Pt) -> dict:
    ratio = b.conc / a.conc
    checks = {
        "concurrency_ratio": {"value": round(ratio, 4), "limit": RATIO_MAX,
                              "pass": ratio <= RATIO_MAX},
        "sample_count": {"value": min(a.n, b.n), "limit": N_MIN,
                         "pass": min(a.n, b.n) >= N_MIN},
        "same_operating_mode": {"value": f"{a.mode}/{b.mode}",
                                "pass": a.mode == b.mode},
    }
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def _interval_margin(a: Pt, b: Pt, slack_pp: float) -> float:
    """Non-negative upper quantile of the interval's calibration residuals + slack."""
    q = percentile([a.margin_pct, b.margin_pct], MARGIN_QUANTILE)
    return max(0.0, q + slack_pp)


def _stage2(a: Pt, b: Pt, validation: list[Pt], margin: float) -> dict:
    inside = [v for v in validation if a.conc < v.conc < b.conc]
    exceed = []
    for v in inside:
        if v.prediction_ms is None or v.measured_ms is None:
            continue
        corrected = v.prediction_ms * (1.0 + margin / 100.0)
        if v.measured_ms > corrected:
            exceed.append({"conc": v.conc, "measured_ms": round(v.measured_ms, 4),
                           "corrected_ms": round(corrected, 4),
                           "short_by_ms": round(v.measured_ms - corrected, 4)})
    rate = len(exceed) / len(inside) if inside else 0.0
    return {
        "validation_points": [v.conc for v in inside],
        "n_inside": len(inside),
        "exceedances": exceed,
        "exceedance_rate": round(rate, 4),
        "tolerance": TOLERANCE,
        # An interval with no interior validation point proves nothing. §5.4's
        # test is vacuous there, so it does NOT register -- which is the whole
        # reason placement (A) registers nothing.
        "pass": bool(inside) and rate <= TOLERANCE,
        "vacuous": not inside,
    }


def _min_slack(a: Pt, b: Pt, validation: list[Pt]) -> float | None:
    """Smallest slack, in pp, that makes every interior validation point covered."""
    inside = [v for v in validation if a.conc < v.conc < b.conc
              and v.prediction_ms and v.measured_ms]
    if not inside:
        return None
    base = _interval_margin(a, b, 0.0)
    need = max((v.measured_ms / v.prediction_ms - 1.0) * 100.0 for v in inside)
    return round(max(0.0, need - base), 4)


def build_region(kind: str, pts: list[Pt], slack_pp: float = SLACK_PP) -> dict:
    calib, valid = placement(kind, pts)
    intervals: list[Interval] = []
    for a, b in pairwise(calib):
        s1 = _stage1(a, b)
        margin = _interval_margin(a, b, slack_pp)
        s2 = _stage2(a, b, valid, margin)
        intervals.append(Interval(
            lo=a.conc, hi=b.conc, registered=s1["pass"] and s2["pass"],
            stage1=s1, stage2=s2, margin_pct=round(margin, 4),
            # Slack can only answer a stage-2 rate failure. It cannot fix a
            # ratio, a sample count or a change of operating mode, so those
            # report None rather than a number that would not help.
            min_slack_to_register_pp=(
                _min_slack(a, b, valid)
                if s1["pass"] and not s2["pass"] and not s2["vacuous"]
                else None),
        ))
    region = [[i.lo, i.hi] for i in intervals if i.registered]
    return {
        "placement": kind,
        "basis": pts[0].basis if pts else None,
        "calibration_points": [p.conc for p in calib],
        "validation_points": [p.conc for p in valid],
        "intervals": [asdict(i) for i in intervals],
        "validation_region": region,
        "covered": round(sum(hi - lo for lo, hi in region), 4),
    }


# ---------------------------------------------------------------------------
# §5.6's worked example, re-judged against a measured region
# ---------------------------------------------------------------------------

#: The disclosure's three illustrative candidates. TPOT SLO is 50 ms and the
#: predicted TPOT is the card winner's 48.41 for all three -- §5.6 varies only
#: the operating point, to show the same prediction judged three ways.
DISCLOSURE_CANDIDATES = [("A", 20.0), ("B", 76.0), ("C", 110.0)]
SLO_TPOT_MS = 50.0
PREDICTED_TPOT_MS = 48.4097


def judge(region: dict, pts: list[Pt]) -> list[dict]:
    """Verdict for each §5.6 candidate against a registered region."""
    out = []
    for name, conc in DISCLOSURE_CANDIDATES:
        interval = next((iv for iv in region["intervals"]
                         if iv["registered"] and iv["lo"] <= conc <= iv["hi"]), None)
        if interval is None:
            out.append({"candidate": name, "conc": conc, "verdict": "unmeasured",
                        "why": "no registered interval covers this operating point"})
            continue
        m = interval["margin_pct"]
        robust = PREDICTED_TPOT_MS * (1.0 + m / 100.0)
        out.append({
            "candidate": name, "conc": conc, "margin_pct": round(m, 4),
            "robust_tpot_ms": round(robust, 4),
            "verdict": "SLO_VIOLATED" if robust > SLO_TPOT_MS else "feasible",
            "why": f"registered interval [{interval['lo']}, {interval['hi']}]",
        })
    return out


def committed_domain_control() -> list[dict]:
    """What `plan --accuracy-domain` returns today at §5.6's three points.

    The control for the re-judgement: the committed domain interpolates and,
    where `widen_error_bars` applies, extrapolates, so it answers at every
    operating point. The registered-region procedure answers only inside a
    registered interval. The interesting rows are the ones where they disagree.
    """
    import yaml

    from planner.predictor.calibration import AccuracyDomain
    doc = yaml.safe_load((REPO / "profiles/calibration/rngd_card_edf.yaml").read_text())
    dom = AccuracyDomain(**doc["hardware"]["RNGD-CARD"]["accuracy_domain"])
    out = []
    for name, conc in DISCLOSURE_CANDIDATES:
        m = dom.tpot_margin_pct(conc)
        robust = PREDICTED_TPOT_MS * (1.0 + m / 100.0)
        out.append({"candidate": name, "conc": conc, "margin_pct": round(m, 4),
                    "robust_tpot_ms": round(robust, 4),
                    "verdict": "SLO_VIOLATED" if robust > SLO_TPOT_MS else "feasible",
                    "in_domain": dom.points[0].conc <= conc <= dom.points[-1].conc})
    return out


def main() -> None:
    result = {"thresholds": {"ratio_max": RATIO_MAX, "n_min": N_MIN,
                             "tolerance": TOLERANCE, "slack_pp": SLACK_PP,
                             "margin_quantile": MARGIN_QUANTILE,
                             "saturation_conc": SATURATION_CONC},
              "bases": {}}
    for basis in ("committed", "p99"):
        pts = load_points(basis)
        result["bases"][basis] = {
            "points": [asdict(p) for p in pts],
            "placements": {k: build_region(k, pts) for k in ("A", "B", "C")},
        }
        for k in ("A", "B", "C"):
            result["bases"][basis]["placements"][k]["disclosure_verdicts"] = judge(
                result["bases"][basis]["placements"][k], pts)

    result["committed_domain_control"] = committed_domain_control()

    out = REPO / "experiments/p2_evidence/results/v1_region.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    for basis, block in result["bases"].items():
        print(f"\n=== basis: {basis} ({len(block['points'])} points) ===")
        for k, reg in block["placements"].items():
            print(f"\n-- placement {k}: calib={reg['calibration_points']} "
                  f"valid={reg['validation_points']}")
            for iv in reg["intervals"]:
                why = []
                for name, c in iv["stage1"]["checks"].items():
                    if not c["pass"]:
                        why.append(f"stage1:{name}={c['value']}")
                if iv["stage1"]["pass"] and not iv["stage2"]["pass"]:
                    why.append("stage2:vacuous" if iv["stage2"]["vacuous"]
                               else f"stage2:rate={iv['stage2']['exceedance_rate']}")
                flag = "REGISTERED" if iv["registered"] else "rejected"
                slack = ("" if iv["min_slack_to_register_pp"] is None
                         else f" min_slack={iv['min_slack_to_register_pp']}pp")
                print(f"   [{iv['lo']:>7}, {iv['hi']:>6}] m={iv['margin_pct']:6.2f}% "
                      f"{flag:11s} {', '.join(why)}{slack}")
            print(f"   region={reg['validation_region']}")
            for v in reg["disclosure_verdicts"]:
                print(f"   §5.6 {v['candidate']} (L={v['conc']}): {v['verdict']}"
                      + (f" robust={v['robust_tpot_ms']}" if "robust_tpot_ms" in v else ""))
    print("\n=== control: what the committed domain answers today ===")
    for v in result["committed_domain_control"]:
        print(f"   §5.6 {v['candidate']} (L={v['conc']}): {v['verdict']} "
              f"margin={v['margin_pct']}% robust={v['robust_tpot_ms']} "
              f"in_domain={v['in_domain']}")
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
