"""B.3 consistency check: does asymmetric P/D land where its two halves do?

WORK_ORDER_spikes.md STEP B.3 step 2. TTFT is produced on the prefill hardware and
TPOT on the decode hardware, so each should sit in the same order of magnitude as
that hardware's own standalone 20-request run. This is a **consistency check, not
an accuracy claim** -- P/D adds a handoff and changes batching, so exact agreement
is not expected and would be suspicious.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planner.util.percentile import percentile

RUNS = [
    ("A40 tp4 colocated (prefill ref)", "outputs/d14/a40_tp4_ref.csv"),
    ("RNGD tp8 colocated (decode ref)", "outputs/d14/b23_auto.csv"),
    ("asym P/D A40 tp4 -> RNGD tp8", "outputs/d14/b3f_asym_raw.csv"),
    ("asym P/D, dim-1 latency 80000", "outputs/d14/b3f_asym_corr.csv"),
]


def stats(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    cols: dict[str, list[float]] = {"TTFT": [], "TPOT": []}
    with path.open() as f:
        for row in csv.DictReader(f):
            for k in cols:
                if row.get(k):
                    cols[k].append(float(row[k]))
    return {
        k: {"p50": percentile(v, 50), "p99": percentile(v, 99), "mean": statistics.fmean(v)}
        for k, v in cols.items()
        if v
    }


def main() -> int:
    print(f"{'run':<34}{'TTFT p50':>14}{'TTFT p99':>14}{'TPOT p50':>14}{'TPOT p99':>14}")
    got = {}
    for label, path in RUNS:
        s = stats(Path(path))
        if s is None:
            print(f"{label:<34}{'(missing)':>14}")
            continue
        got[label] = s
        print(f"{label:<34}{s['TTFT']['p50'] / 1e6:>13.1f}m{s['TTFT']['p99'] / 1e6:>13.1f}m"
              f"{s['TPOT']['p50'] / 1e6:>13.2f}m{s['TPOT']['p99'] / 1e6:>13.2f}m")
    print("\n(all values ms)")

    pre = got.get("A40 tp4 colocated (prefill ref)")
    dec = got.get("RNGD tp8 colocated (decode ref)")
    for label in ("asym P/D A40 tp4 -> RNGD tp8", "asym P/D, dim-1 latency 80000"):
        asym = got.get(label)
        if not (asym and pre and dec):
            continue
        rt = asym["TTFT"]["p50"] / pre["TTFT"]["p50"]
        rp = asym["TPOT"]["p50"] / dec["TPOT"]["p50"]
        print(f"\n{label}")
        print(f"  TTFT p50 / A40 tp4 standalone  = {rt:.2f}x")
        print(f"  TPOT p50 / RNGD tp8 standalone = {rp:.2f}x")
        ok = 0.1 <= rt <= 10 and 0.1 <= rp <= 10
        print(f"  same order of magnitude on both: {'YES' if ok else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
