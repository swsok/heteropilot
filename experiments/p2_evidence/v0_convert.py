#!/usr/bin/env python
"""V0: the two residual conventions side by side, and what aggregation each point used.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V0. Two jobs, both of which must be
computed rather than typed (the work order forbids hand arithmetic here because
these numbers go into a patent specification):

1. **The convention table.** Every committed accuracy-domain point in both
   conventions. The domain files declare `e = (sim - measured) / measured`; the
   invention disclosure (v5 §0.3) defines `r = (measured - sim) / sim`. They are
   related by `r = -e / (1 + e)`, and `r` is also the one-sided margin the
   planner applies, so `sim * (1 + r)` reconstructs the measurement while
   `sim * (1 - e)` does not (deviations D70).

2. **The c76 aggregation reconciliation.** The RNGD-CARD domain's c76 point is
   the only high-load measured evidence the disclosure has, and it pairs a
   SIMULATED p50 against a MEASURED MEAN. Every other point in both files pairs
   p50 against p50. This script recomputes that point like-for-like from the raw
   artifacts so the disclosure can quote one basis.

Run:  .venv/bin/python experiments/p2_evidence/v0_convert.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.util.percentile import percentile  # noqa: E402

RNGD_DOMAIN = REPO / "profiles/calibration/rngd_card_edf.yaml"
A40_DOMAIN = REPO / "profiles/calibration/a40.accuracy.yaml"
C64 = REPO / "outputs/rngd_envelope/edf/real_c64.json"
C128 = REPO / "outputs/rngd_envelope/edf/real_c128.json"
#: The card fixture's committed winner, whose p99 48.41 is the number the
#: disclosure cites. `pd_slo_sweep_margin.md` identifies it as `s256-t8192`;
#: this cache entry is that run (p99 TTFT 480.09 ms, goodput 5.5519 rps both
#: reproduce `pd_slo_sweep.md`).
WINNER_CACHE = REPO / (
    "outputs/uncertainty/ea1/cache/"
    "58d6c79533907b0576b730d7a3550a1e81f2bf82279f7fc01446ccd6939c18cf.json"
)


def r_from_e(e_pct: float) -> float:
    """Disclosure convention from file convention. `r = -e / (1 + e)`, in percent."""
    e = e_pct / 100.0
    if e <= -1.0:
        raise ValueError(f"e={e_pct} % implies a non-positive measurement")
    return -e / (1.0 + e) * 100.0


def margin_pct(e_pct: float) -> float:
    """The one-sided margin: `r` where the simulator is optimistic, else 0."""
    return 0.0 if e_pct >= 0.0 else r_from_e(e_pct)


def _points(path: Path, hardware: str) -> list[dict]:
    doc = yaml.safe_load(path.read_text())
    return doc["hardware"][hardware]["accuracy_domain"]["points"]


def _tpots(path: Path) -> tuple[list[float], float, int]:
    """Per-request TPOT in ms, effective concurrency (Little's law), and n."""
    d = json.loads(path.read_text())
    ok = [r for r in d["per_request"] if r.get("error") is None]
    tpot = [r["tpot_ns"] / 1e6 for r in ok]
    eff = sum(r["latency_ns"] / 1e9 for r in ok) / d["wall_s"]
    return tpot, eff, len(ok)


def _interp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def convention_table() -> str:
    rows = []
    for hw, path, src_of in (
        ("RNGD-CARD", RNGD_DOMAIN, _rngd_source),
        ("A40", A40_DOMAIN, _a40_source),
    ):
        for p in _points(path, hw):
            e = float(p["tpot_err_pct"])
            basis, source = src_of(float(p["conc"]))
            rows.append(
                f"| {hw} | {p['conc']:g} | {source} | {basis} | "
                f"{e:+.2f} | {r_from_e(e):+.2f} | {margin_pct(e):.2f} |"
            )
    head = (
        "| hardware | conc (predicted) | source | aggregation basis | "
        "file `e` % | disclosure `r` % | margin % |\n"
        "| --- | ---: | --- | --- | ---: | ---: | ---: |"
    )
    return head + "\n" + "\n".join(rows)


def _rngd_source(conc: float) -> tuple[str, str]:
    if conc == 76.0:
        return ("**sim p50 vs measured MEAN**", "envelope interpolation (c64/c128)")
    if conc == 16.6:
        return ("bucket mean error", "EDF bundle fit (PROJECT_REPORT §4.8.4)")
    return ("sim p50 vs measured p50", "300-req arrival-matched comparison")


def _a40_source(conc: float) -> tuple[str, str]:
    if conc == 170.56:
        return ("**p95 abs error, signed by mean diff**", "300-req sharegpt validation")
    return ("sim p50 vs measured p50", "300-req arrival-matched comparison")


def c76_reconciliation() -> str:
    t64, eff64, n64 = _tpots(C64)
    t128, eff128, n128 = _tpots(C128)
    winner = json.loads(WINNER_CACHE.read_text())
    m = winner.get("metrics", winner)
    sim_p50, sim_p99 = m["p50_tpot_ms"], m["p99_tpot_ms"]
    sim_conc = m["served_concurrency"]

    out = [
        f"Raw: `{C64.relative_to(REPO)}` (n={n64}, eff {eff64:.3f}), "
        f"`{C128.relative_to(REPO)}` (n={n128}, eff {eff128:.3f}).",
        f"Simulator: `{winner['candidate_id']}`, served concurrency {sim_conc:.3f}, "
        f"p50 {sim_p50:.4f} ms, p99 {sim_p99:.4f} ms.",
        "",
        "| measured aggregation | c64 | c128 | interpolated @76 | vs sim p50 "
        f"({sim_p50:.2f}) `e` % | disclosure `r` % | robust TPOT of {sim_p99:.2f} |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, agg in (
        ("mean (what 52.7 is)", lambda v: sum(v) / len(v)),
        ("p50 (what every other point uses)", lambda v: percentile(v, 50)),
        ("p95", lambda v: percentile(v, 95)),
        ("p99 (what the SLO check reads)", lambda v: percentile(v, 99)),
    ):
        a, b = agg(t64), agg(t128)
        ref = _interp(76.0, eff64, a, eff128, b)
        e = (sim_p50 - ref) / ref * 100.0
        r = r_from_e(e)
        robust = sim_p99 * (1.0 + margin_pct(e) / 100.0)
        out.append(
            f"| {label} | {a:.4f} | {b:.4f} | **{ref:.4f}** | {e:+.2f} | "
            f"{r:+.2f} | **{robust:.2f} ms** |"
        )

    # The p99-vs-p99 pairing is the one the feasibility check actually consumes.
    p99_ref = _interp(76.0, eff64, percentile(t64, 99), eff128, percentile(t128, 99))
    e99 = (sim_p99 - p99_ref) / p99_ref * 100.0
    out += [
        "",
        "Like-for-like on the metric the SLO check reads (sim p99 against measured "
        f"p99): `e` = {e99:+.2f} %, `r` = {r_from_e(e99):+.2f} %, and "
        f"{sim_p99:.2f} x {1 + margin_pct(e99) / 100:.4f} = "
        f"{sim_p99 * (1 + margin_pct(e99) / 100):.2f} ms -- which reconstructs the "
        f"measured p99 ({p99_ref:.2f} ms) by construction.",
    ]
    return "\n".join(out)


def main() -> None:
    print("## 12-point convention table\n")
    print(convention_table())
    print("\n## c76 aggregation reconciliation\n")
    print(c76_reconciliation())


if __name__ == "__main__":
    main()
