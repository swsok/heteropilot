#!/usr/bin/env python
"""V1: which percentile pair is each RNGD-CARD accuracy-domain point actually made of?

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V1, percentile-pair audit (added
2026-09-16). The feasibility check applies the margin to a **p99**
(`planner/optimizer/feasibility.py:67`, `robust_tpot = p99_tpot * (1 + m)`), so a
residual fitted on any other statistic compares two distributions at different
points. This script walks every committed point back to its raw artifacts,
records the statistic each side actually used, and recomputes the pair on p99
where the raw data allows.

**It changes no stored value** (rule A3). The recomputed column is reported
beside the committed one, never in place of it.

Run:  .venv/bin/python experiments/p2_evidence/v1_percentile_audit.py
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.util.percentile import percentile  # noqa: E402

NS_PER_MS = 1e6


@dataclass
class Stat:
    n: int
    mean: float
    p50: float
    p99: float
    eff: float | None = None


def _sim_stat(csv_path: Path) -> Stat:
    """TPOT statistics from a simulator CSV, in ms. `TPOT` is nanoseconds."""
    tpot: list[float] = []
    lat: list[float] = []
    arrival: list[float] = []
    end: list[float] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            tpot.append(float(row["TPOT"]) / NS_PER_MS)
            lat.append(float(row["latency"]) / 1e9)
            arrival.append(float(row["arrival"]) / 1e9)
            end.append(float(row["end_time"]) / 1e9)
    span = max(end) - min(arrival)
    return Stat(len(tpot), sum(tpot) / len(tpot), percentile(tpot, 50),
                percentile(tpot, 99), sum(lat) / span if span > 0 else None)


def _bench_stat(paths: list[Path]) -> Stat:
    """TPOT statistics from raw bench records, in ms, pooled across repeats.

    Reported two ways because the envelope file aggregates repeats by averaging
    each repeat's own percentile rather than pooling the requests; both are given
    so a reader can see the committed number is the average-of-repeats one.
    """
    per_repeat: list[list[float]] = []
    pooled: list[float] = []
    eff_num = eff_den = 0.0
    for p in paths:
        d = json.loads(p.read_text())
        ok = [r for r in d["per_request"] if r.get("error") is None]
        t = [r["tpot_ns"] / NS_PER_MS for r in ok]
        per_repeat.append(t)
        pooled += t
        eff_num += sum(r["latency_ns"] / 1e9 for r in ok)
        eff_den += d["wall_s"]
    avg = lambda f: sum(f(t) for t in per_repeat) / len(per_repeat)  # noqa: E731
    return Stat(
        n=len(pooled),
        mean=avg(lambda t: sum(t) / len(t)),
        p50=avg(lambda t: percentile(t, 50)),
        p99=avg(lambda t: percentile(t, 99)),
        eff=eff_num / eff_den if eff_den else None,
    )


@dataclass
class Point:
    """One committed domain point and where each of its two sides came from."""

    conc: float
    stored_e: float
    measured_conc: str
    sim_csv: Path | None = None
    measured_raw: list[Path] = field(default_factory=list)
    #: The value the envelope file stores in its `tpot_p50` field for this point.
    envelope_tpot_p50: float | None = None
    note: str = ""


#: The nine RNGD-CARD points, each tied to the artifacts that produced it. The
#: mapping is read off the `note` field of `profiles/calibration/rngd_card_edf.yaml`
#: ("300-req sim at X rps vs measured cY") and the run manifests beside each CSV.
_LOW = REPO / "outputs/card_lowload_300"
_HIGH = REPO / "outputs/card_highload_300"
_ENVLOW = REPO / "outputs/rngd_envelope_lowload"
_ENV = REPO / "outputs/rngd_envelope/edf"

POINTS = [
    Point(1.020, 2.25, "c1.0", _LOW / "sim_c1p0.csv",
          [_ENVLOW / "bench_c1.json", _ENVLOW / "bench_c1_r1.json"], 15.71),
    Point(2.194, 11.00, "c1.99", _LOW / "sim_c1p99.csv",
          [_ENVLOW / "bench_c2.json", _ENVLOW / "bench_c2_r1.json"], 18.01),
    Point(4.311, 10.79, "c3.98", _LOW / "sim_c3p98.csv",
          [_ENVLOW / "bench_c4.json", _ENVLOW / "bench_c4_r1.json"], 19.48),
    Point(8.211, 7.33, "c7.88", _LOW / "sim_c7p88.csv",
          [_ENVLOW / "bench_c8.json", _ENVLOW / "bench_c8_r1.json"], 21.86),
    Point(14.832, 3.26, "c15.3", _LOW / "sim_c15p3.csv",
          [_ENV / "real_c16.json"], 25.71),
    Point(15.212, 2.47, "c15.59", _LOW / "sim_c15p59.csv",
          [_ENVLOW / "bench_c16.json", _ENVLOW / "bench_c16_r1.json"], 26.05),
    Point(16.6, -3.10, "(EDF bundle fit)", None, [], None,
          "bucket mean error, 5 samples; no per-request pair in this repo"),
    Point(25.181, -3.28, "c29.3", _HIGH / "sim_c29p3.csv",
          [_ENV / "real_c32.json"], 31.18),
    Point(76.0, -18.00, "interp c59.2/c107.2", None,
          [_ENV / "real_c64.json", _ENV / "real_c128.json"], None,
          "sim side is the card fixture winner, not a lowload CSV"),
]

#: The card fixture's committed winner -- the c76 point's simulated side (V0).
WINNER_CACHE = REPO / (
    "outputs/uncertainty/ea1/cache/"
    "58d6c79533907b0576b730d7a3550a1e81f2bf82279f7fc01446ccd6939c18cf.json"
)


def _which_statistic(stored: float | None, m: Stat) -> str:
    """Which raw statistic the envelope's `tpot_p50` field actually holds."""
    if stored is None:
        return "n/a"
    gaps = {"p50": abs(m.p50 - stored), "mean": abs(m.mean - stored),
            "p99": abs(m.p99 - stored)}
    best = min(gaps, key=gaps.get)
    return f"**{best}**" if gaps[best] < 0.02 else f"{best}?(Δ{gaps[best]:.2f})"


def _err(sim: float, meas: float) -> float:
    return (sim - meas) / meas * 100.0


def _r(e: float) -> float:
    return -(e / 100.0) / (1.0 + e / 100.0) * 100.0


def _interp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def audit() -> list[dict]:
    rows = []
    for pt in POINTS:
        row: dict = {"conc": pt.conc, "stored_e": pt.stored_e,
                     "measured_conc": pt.measured_conc, "note": pt.note}

        if pt.conc == 76.0:  # the interpolated point: both sides are special
            a = _bench_stat([pt.measured_raw[0]])
            b = _bench_stat([pt.measured_raw[1]])
            m = json.loads(WINNER_CACHE.read_text()).get("metrics", {})
            sim_p50, sim_p99 = m["p50_tpot_ms"], m["p99_tpot_ms"]
            meas = {k: _interp(76.0, a.eff, getattr(a, k), b.eff, getattr(b, k))
                    for k in ("mean", "p50", "p99")}
            row.update(
                sim_stat="p50", measured_stat="**mean**",
                sim_p50=sim_p50, sim_p99=sim_p99,
                measured_used=meas["mean"], measured_p99=meas["p99"],
                measured_p50=meas["p50"], n_sim=m["completed_requests"],
                n_measured=a.n + b.n,
                e_recomputed_p99=_err(sim_p99, meas["p99"]),
            )
        elif pt.sim_csv is None:  # c16.6, a bucket fit with no per-request pair
            row.update(sim_stat="(bucket mean)", measured_stat="(bucket mean)",
                       n_sim=5, n_measured=5, e_recomputed_p99=None)
        else:
            s = _sim_stat(pt.sim_csv)
            m = _bench_stat(pt.measured_raw)
            row.update(
                sim_stat="p50", measured_stat=_which_statistic(pt.envelope_tpot_p50, m),
                sim_p50=s.p50, sim_p99=s.p99,
                measured_used=pt.envelope_tpot_p50, measured_p99=m.p99,
                measured_p50=m.p50, n_sim=s.n, n_measured=m.n,
                e_recomputed_p99=_err(s.p99, m.p99),
            )
        rows.append(row)
    return rows


def main() -> None:
    rows = audit()
    print("| domain conc | measured point | sim stat | measured stat (what the "
          "`tpot_p50` field holds) | committed `e` % | p99-vs-p99 `e` % | "
          "p99 `r` % | n sim / meas |")
    print("| ---: | --- | --- | --- | ---: | ---: | ---: | --- |")
    for r in rows:
        e99 = r["e_recomputed_p99"]
        e99s = f"{e99:+.2f}" if e99 is not None else "—"
        r99s = f"{_r(e99):+.2f}" if e99 is not None else "—"
        print(f"| {r['conc']:g} | {r['measured_conc']} | {r['sim_stat']} | "
              f"{r['measured_stat']} | {r['stored_e']:+.2f} | {e99s} | {r99s} | "
              f"{r['n_sim']} / {r['n_measured']} |")

    print()
    print("Detail, ms (committed measured value is the envelope's `tpot_p50` field):")
    print("| conc | sim p50 | sim p99 | measured used | measured p50 | measured p99 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        if "sim_p50" not in r:
            continue
        print(f"| {r['conc']:g} | {r['sim_p50']:.4f} | {r['sim_p99']:.4f} | "
              f"{r['measured_used']:.4f} | {r['measured_p50']:.4f} | "
              f"{r['measured_p99']:.4f} |")

    pairs = [r["measured_stat"] for r in rows]
    print()
    print(f"p99-against-p99 points: **{sum('p99' in p for p in pairs)} of {len(rows)}**")


if __name__ == "__main__":
    main()
