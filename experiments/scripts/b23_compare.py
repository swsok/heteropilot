"""B.2-3: one TP group flat [8] vs split across two dims [4,2].

WORK_ORDER_spikes.md STEP B.2-3. Prints the per-iteration cost difference and
the p50/p99 TTFT/TPOT difference, and judges the TPOT gap against the RNGD
profile's own 3.1 % error (D22).

Note the simulator's `iteration N finished, C cycles` field is a **cumulative**
simulated timestamp in ns, not that iteration's cost -- iteration 0 prints the
first request's arrival time. Per-iteration cost is the consecutive difference
within one NPU, which is what this computes.
"""
from __future__ import annotations

import csv
import itertools
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planner.util.percentile import percentile

ITER = re.compile(
    r"NPU\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles\."
)
LINK_LATENCY_NS = 20000.0   # the fixture's scalar; the prediction below is in these units
TOLERANCE_PCT = 3.1         # D22, rngd_concurrency_envelope.md


def iterations(log: Path) -> dict[tuple[int, int], tuple[int, int]]:
    out = {}
    for line in log.read_text(errors="replace").splitlines():
        m = ITER.search(line)
        if m:
            npu, it, cyc, exposed = (int(g) for g in m.groups())
            out[(npu, it)] = (cyc, exposed)
    return out


def per_iteration(cum: dict[tuple[int, int], tuple[int, int]]):
    """Consecutive differences within each NPU: the actual cost of each step."""
    by_npu: dict[int, list[int]] = {}
    for npu, it in cum:
        by_npu.setdefault(npu, []).append(it)
    out = {}
    for npu, its in by_npu.items():
        its.sort()
        for prev, cur in itertools.pairwise(its):
            out[(npu, cur)] = (
                cum[(npu, cur)][0] - cum[(npu, prev)][0],
                cum[(npu, cur)][1] - cum[(npu, prev)][1],
            )
    return out


def pct(a: float, b: float) -> float:
    return float("nan") if a == 0 else (b - a) / a * 100.0


def percentiles(csv_path: Path) -> dict[str, dict[str, float]]:
    cols = {"TTFT": [], "TPOT": []}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            for k in cols:
                if row.get(k):
                    cols[k].append(float(row[k]))
    res = {}
    for k, v in cols.items():
        if not v:
            raise ValueError(f"{csv_path} has no {k} values")
        # planner/util/percentile.py is the single implementation (CLAUDE.md);
        # the simulator's own printed P99 uses a different method.
        res[k] = {
            "n": len(v),
            "p50": percentile(v, 50),
            "p99": percentile(v, 99),
            "mean": statistics.fmean(v),
        }
    return res


def main() -> int:
    base = Path("outputs/d14")
    a_log, s_log = base / "b23_auto.log", base / "b23_split2.log"
    a_csv, s_csv = base / "b23_auto.csv", base / "b23_split2.csv"
    for p in (a_log, s_log, a_csv, s_csv):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2

    a_it, s_it = iterations(a_log), iterations(s_log)
    shared = sorted(set(a_it) & set(s_it))
    print(f"iterations logged: auto {len(a_it)}, split2 {len(s_it)}, shared {len(shared)}")
    if not shared:
        print("no shared (npu, iteration) keys -- the two runs did not line up", file=sys.stderr)

    for label, it in (("auto [8]", a_it), ("split2 [4,2]", s_it)):
        last = max(it)
        print(f"  {label:<14} final cumulative t = {it[last][0]:,} ns, "
              f"exposed comm = {it[last][1]:,} ns "
              f"({it[last][1] / it[last][0] * 100:.1f} % of it)")

    a_cost, s_cost = per_iteration(a_it), per_iteration(s_it)
    shared_cost = sorted(set(a_cost) & set(s_cost))
    if shared_cost:
        d = [s_cost[k][0] - a_cost[k][0] for k in shared_cost]
        de = [s_cost[k][1] - a_cost[k][1] for k in shared_cost]
        same = sum(1 for x in d if x == 0)
        a_tot = sum(a_cost[k][0] for k in shared_cost)
        s_tot = sum(s_cost[k][0] for k in shared_cost)
        print("\n=== per-iteration cost (consecutive diff), split2 minus auto ===")
        print(f"compared:   {len(d)} iteration steps")
        print(f"identical:  {same}/{len(d)} ({same / len(d) * 100:.1f} %)")
        print(f"cost ns:      min {min(d):,}  p50 {statistics.median(d):,.0f}  "
              f"max {max(d):,}  mean {statistics.fmean(d):,.1f}")
        print(f"exposed comm: min {min(de):,}  p50 {statistics.median(de):,.0f}  "
              f"max {max(de):,}  mean {statistics.fmean(de):,.1f}")
        print(f"summed cost:  auto {a_tot:,} ns   split2 {s_tot:,} ns   "
              f"({pct(a_tot, s_tot):+.2f} %)")
        print(f"prediction (work order): [4,2] faster by 2*3*link_latency = "
              f"{2 * 3 * LINK_LATENCY_NS:,.0f} ns per layer; "
              f"measured mean {abs(statistics.fmean(d)):,.0f} ns per iteration")

    print("\n=== end-to-end percentiles ===")
    a_p, s_p = percentiles(a_csv), percentiles(s_csv)
    print(f"{'metric':<12}{'auto [8]':>18}{'split2 [4,2]':>18}{'diff %':>10}")
    worst_tpot = 0.0
    for metric in ("TTFT", "TPOT"):
        for q in ("p50", "p99", "mean"):
            diff = pct(a_p[metric][q], s_p[metric][q])
            print(f"{metric + ' ' + q:<12}{a_p[metric][q]:>18.1f}"
                  f"{s_p[metric][q]:>18.1f}{diff:>10.3f}")
            if metric == "TPOT" and q in ("p50", "p99"):
                worst_tpot = max(worst_tpot, abs(diff))

    print(f"\nworst |TPOT diff| = {worst_tpot:.3f} % vs the {TOLERANCE_PCT} % "
          f"RNGD profile error (D22)")
    print("VERDICT:", "usable without correction" if worst_tpot < TOLERANCE_PCT
          else "correction needed -- propose a per-dim link_latency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
