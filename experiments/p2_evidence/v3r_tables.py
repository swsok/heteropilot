#!/usr/bin/env python
"""Render the S7.0 sweep JSON into the tables `v3r_candidate_selection.md` quotes.

`WORK_ORDER_domain_scoping.md` STEP S7.0. Generated tables live here so the
authored analysis beside them cannot drift from the record -- the split E-A1
uses (`--out` generated, `results/*.md` authored).

Run:  .venv/bin/python experiments/p2_evidence/v3r_tables.py \
          --json experiments/p2_evidence/results/v3r_candidates.json \
          --out outputs/p2_evidence/v3r/v3r_tables.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.predictor.calibration import load_accuracy_domains  # noqa: E402

TPOT_SLO = 50.0
TTFT_SLO = 25000.0
GLOBAL_MARGIN = 18.0

SHORT = {"a_margin0": "a", "b_global18": "b", "c_accuracy_domain": "c",
         "d_refuse": "d", "e_condition_refuse": "e"}


def knob(cid: str) -> str:
    return cid.split("-tp1-dp1-")[-1]


def in_domain_of(hardware: str = "RNGD-CARD"):
    """Is a row's operating point inside the domain's MEASURED range?

    This is not decoration. Outside the range the margin is an extrapolation
    (`widen_error_bars`) or the candidate is held outright (`refuse`, the D33
    default and rules (d)/(e) here), so a row with a large separation and an
    operating point past the top measured point is not a candidate -- it is a
    candidate the policy declines to judge. T4 sorts on separation, which would
    otherwise put exactly those rows on top.
    """
    domain = load_accuracy_domains(str(REPO)).get(hardware)
    if domain is None:
        return lambda _conc: None
    return lambda conc: None if conc is None else domain.in_domain(conc)


def verdict_cell(row: dict, rule: str) -> str:
    v = row["verdicts"][rule]
    if v == "feasible":
        return "pass"
    stage = row["stages"].get(rule) or ""
    if stage == "outside_calibration_domain":
        return "held-range"
    if stage == "calibration_condition_mismatch":
        return "held-cond"
    return "fail"


def sweep_table(rows: list[dict]) -> str:
    inside = in_domain_of()
    out = ["| rps | knob | L | in domain | p99 TPOT | p99 TTFT | m(c) % "
           "| a | b | c | d | e | decided on | shape |",
           "| ---: | --- | ---: | --- | ---: | ---: | ---: "
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        m = r["tpot_margin_pct"].get("c_accuracy_domain")
        out.append(
            f"| {r['rps']:g} | {knob(r['candidate_id'])} | "
            f"{_n(r['served_concurrency'])} | "
            f"{_yn(inside(r['served_concurrency']))} | {_n(r['p99_tpot_ms'])} | "
            f"{_n(r['p99_ttft_ms'], 0)} | {_n(m, 2)} | "
            + " | ".join(verdict_cell(r, k) for k in SHORT)
            + f" | {r['deciding_metric'] or '—'} | {','.join(r['shapes']) or '—'} |")
    return "\n".join(out)


def shape_windows(rows: list[dict]) -> str:
    """For each row, the TPOT band each shape would need at ITS operating point."""
    out = ["| rps | knob | L | m(c) % | P2-shape needs TPOT | P3-shape needs TPOT "
           "| actual p99 TPOT | in window? |",
           "| ---: | --- | ---: | ---: | --- | --- | ---: | --- |"]
    for r in rows:
        m = r["tpot_margin_pct"].get("c_accuracy_domain")
        tpot = r["p99_tpot_ms"]
        if m is None or tpot is None:
            continue
        lo = TPOT_SLO / (1.0 + m / 100.0)
        p2 = f"({lo:.2f}, {TPOT_SLO:.0f}]"
        b_lo = TPOT_SLO / (1.0 + GLOBAL_MARGIN / 100.0)
        p3 = f"({b_lo:.2f}, {lo:.2f}]" if b_lo < lo else "none (m >= 18 %)"
        inside = ("P2" if lo < tpot <= TPOT_SLO else
                  "P3" if b_lo < tpot <= lo else "—")
        out.append(f"| {r['rps']:g} | {knob(r['candidate_id'])} | {_n(r['served_concurrency'])} | "
                   f"{_n(m, 2)} | {p2} | {p3} | {_n(tpot)} | {inside} |")
    return "\n".join(out)


def queueing_table(rows: list[dict]) -> str:
    """The evidence for what a large L means here: L against p99 TTFT.

    S7.B flagged a tension -- the domain's note says the card model saturates
    near served 44, yet E-A1's candidates report L = 140-178. If the large L
    figures come with a p99 TTFT in the tens of seconds, they are residency
    inflated by a growing queue, not work the card is doing.
    """
    out = ["| L (bucket) | rows | median p99 TTFT | p99 TTFT min-max "
           "| meets TTFT SLO |",
           "| --- | ---: | ---: | --- | ---: |"]
    buckets = [(0, 16), (16, 30), (30, 45), (45, 70), (70, 1e9)]
    for lo, hi in buckets:
        sel = [r for r in rows
               if r["served_concurrency"] is not None
               and lo <= r["served_concurrency"] < hi and r["p99_ttft_ms"] is not None]
        if not sel:
            continue
        ttfts = sorted(r["p99_ttft_ms"] for r in sel)
        med = ttfts[len(ttfts) // 2]
        ok = sum(1 for t in ttfts if t <= TTFT_SLO)
        label = f"[{lo:g}, {hi:g})" if hi < 1e9 else f"≥ {lo:g}"
        out.append(f"| {label} | {len(sel)} | {med:.0f} | {ttfts[0]:.0f} ~ {ttfts[-1]:.0f} | "
                   f"{ok}/{len(sel)} |")
    return "\n".join(out)


def slo_sweep(rows: list[dict]) -> str:
    """Which shape each row would exhibit at other TPOT SLOs, and how far apart.

    **The SLO is a post-hoc parameter here**, exactly as in V3 §6.2: E-A1's
    50 ms aggregate is NOT re-scored, and nothing under `outputs/uncertainty/ea1/`
    is touched. This asks what the same five rules would decide at other
    thresholds, which is a sensitivity analysis and is what turns one deployment
    into a grid.

    It needs no simulation. A row's predicted TPOT and its own margin are fixed;
    only the threshold moves, so every cell is arithmetic on the rows already
    simulated.

    The two separations are what decide whether a measurement can TELL:

        P2-shape   (a) passes by  SLO - T        (the headroom it is granted)
                   (c) rejects by (1+m)T - SLO   (the overshoot it is caught by)
        P3-shape   (b) rejects by 1.18T - SLO
                   (c) passes by  SLO - (1+m)T

    `min_sep` is the smaller of a shape's two separations, because the weaker one
    is what a p99 spread has to beat. It is the column to sort on.
    """
    inside = in_domain_of()
    out = ["| shape | rps | knob | L | in domain | m % | TPOT SLO | p99 TPOT "
           "| sep 1 | sep 2 | min sep |",
           "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: "
           "| ---: |"]
    best: dict[str, tuple] = {}
    for r in rows:
        m_pct = r["tpot_margin_pct"].get("c_accuracy_domain")
        tpot = r["p99_tpot_ms"]
        ttft = r["p99_ttft_ms"]
        if m_pct is None or tpot is None or ttft is None:
            continue
        # A shape is only testable where TPOT decides it. A row whose TTFT
        # breaches its own SLO would be measured for the wrong reason.
        if ttft > TTFT_SLO:
            continue
        m = m_pct / 100.0
        for slo10 in range(200, 1001, 5):          # 20.0 .. 100.0 ms, 0.5 ms grid
            slo = slo10 / 10.0
            robust_c = (1.0 + m) * tpot
            robust_b = (1.0 + GLOBAL_MARGIN / 100.0) * tpot
            if tpot <= slo < robust_c:             # (a) passes, (c) rejects
                seps = (slo - tpot, robust_c - slo)
                shape = "P2-shape"
            elif robust_c <= slo < robust_b:       # (b) rejects, (c) passes
                seps = (robust_b - slo, slo - robust_c)
                shape = "P3-shape"
            else:
                continue
            key = (shape, r["rps"], r["candidate_id"])
            score = min(seps)
            if key not in best or score > best[key][0]:
                best[key] = (score, slo, seps, r, m_pct)
    # Usable rows (operating point inside the measured range) first, then the
    # rest, each block by descending min separation.
    def order(kv):
        (shape, _rps, _cid), (score, _slo, _seps, r, _m) = kv
        ok = inside(r["served_concurrency"])
        return (shape, not ok, -score)

    for (shape, _rps, _cid), (score, slo, seps, r, m_pct) in sorted(
            best.items(), key=order):
        ok = inside(r["served_concurrency"])
        out.append(
            f"| {shape} | {r['rps']:g} | {knob(r['candidate_id'])} | "
            f"{_n(r['served_concurrency'])} | {_yn(ok)} | {_n(m_pct, 2)} | "
            f"{slo:g} | {_n(r['p99_tpot_ms'])} | {seps[0]:.3f} | {seps[1]:.3f} | "
            f"{'**' if ok else ''}{score:.3f}{'**' if ok else ''} |")
    return "\n".join(out)


def ea1_control(rows: list[dict], cache_dir: Path) -> str:
    """The sweep's rows at the fixture's OWN rate, against E-A1's committed cache.

    The sweep reaches E-A1's operating points when it reaches E-A1's arrival
    rate, so those rows are a control: if this driver, its per-rate spec copy and
    its own cache produce the committed values, the rest of the sweep is on the
    same axis as the record the disclosure quotes. A mismatch here would mean the
    sweep is measuring something else and every other row is suspect.
    """
    import glob

    committed = {}
    for path in glob.glob(str(cache_dir / "*.json")):
        entry = json.loads(Path(path).read_text())
        if "candidate_id" in entry:
            committed[entry["candidate_id"]] = entry.get("metrics", {})
    if not committed:
        return "_E-A1 cache not found; control not run._"

    base = None
    for r in rows:
        if r.get("_base_rate"):
            base = r["rps"]
    out = ["| knob | sweep L | E-A1 L | sweep p99 TPOT | E-A1 p99 TPOT | agrees |",
           "| --- | ---: | ---: | ---: | ---: | --- |"]
    checked = 0
    for r in sorted(rows, key=lambda r: r["candidate_id"]):
        if base is not None and r["rps"] != base:
            continue
        ref = committed.get(r["candidate_id"])
        if not ref:
            continue
        pairs = [(r["served_concurrency"], ref.get("served_concurrency")),
                 (r["p99_tpot_ms"], ref.get("p99_tpot_ms"))]
        agrees = all(a is not None and b is not None and abs(a - b) < 5e-3
                     for a, b in pairs)
        checked += 1
        out.append(f"| {knob(r['candidate_id'])} | {_n(r['served_concurrency'])} | "
                   f"{_n(ref.get('served_concurrency'))} | {_n(r['p99_tpot_ms'])} | "
                   f"{_n(ref.get('p99_tpot_ms'))} | {'yes' if agrees else '**NO**'} |")
    if not checked:
        return "_no sweep row at the fixture's own rate; control not run._"
    return "\n".join(out)


def _n(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _yn(value: bool | None) -> str:
    return "?" if value is None else ("yes" if value else "**NO**")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ea1-cache", type=Path,
                    default=Path("outputs/uncertainty/ea1/cache"),
                    help="E-A1's committed envelope cache, for the T6 control")
    args = ap.parse_args()
    report = json.loads(args.json.read_text())
    rows = sorted(report["rows"], key=lambda r: (r["rps"], r["candidate_id"]))
    base_rate = report["fixture"]["base_arrival_rate_rps"]
    for r in rows:
        r["_base_rate"] = (r["rps"] == base_rate)

    parts = [
        "<!-- GENERATED by experiments/p2_evidence/v3r_tables.py -- do not hand-edit. -->",
        "", "## T1 the whole sweep", "", sweep_table(rows), "",
        "## T2 the TPOT band each shape needs, at each candidate's own operating point",
        "", shape_windows(rows), "",
        "## T3 what a large L means here -- L against p99 TTFT", "",
        queueing_table(rows), "",
        "## T4 the best TPOT SLO for each shape, per row (post-hoc, no simulation)",
        "", slo_sweep(rows), "",
    ]
    mirror = report.get("mirror_control")
    if mirror:
        parts += ["## T5 the D40 mirror control (node_rngd1 vs node_rngd0, cache off)", "",
                  f"`rps {mirror['rps']:g}`, `s32-t2048`, identical: "
                  f"**{mirror.get('identical')}**", "",
                  "| metric | node_rngd0 | node_rngd1 | \\|delta\\| % |",
                  "| --- | ---: | ---: | ---: |"]
        for field, c in sorted(mirror.get("compared", {}).items()):
            parts.append(f"| {field} | {_n(c['swept'], 6)} | {_n(c['mirror'], 6)} | "
                         f"{_n(c['abs_pct'], 6)} |")
        parts.append("")
    parts += [f"## T6 the sweep at the fixture's own {base_rate:g} rps, against "
              "E-A1's committed cache", "", ea1_control(rows, args.ea1_cache), ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
