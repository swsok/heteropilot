"""Read the per-execution attention cost out of EDF traces.

`WORK_ORDER_npu_exec_model_spike.md` STEP C.3 (E-N8). The bundle's `attention.csv`
carries a per-layer **total** over however many executions the runtime did
(`meta.yaml`: "total decode-attention device time over (forwards x 32)"), which is
why STEP A could not price the grouping without double-counting it (D91). This
script reads the other thing: what **one** attention execution costs, as a
function of how many sequences it covers and how long their context is.

**The workload has to be a probe, not traffic.** An execution's cost can only be
attributed to a group size if the trace says what that group size was, and it does
-- an EDF stage is named by its compiled bucket, `AttentionBucket { batch_size,
attention_size, kv_cache_size }`. So `batch_size` *is* the group size. Traffic
works too; the fixed-length probes exist so that a whole concurrency level lands
on one `attention_size` and the length axis can be moved independently.

**Cycles convert at 1.6 GHz**, the figure `meta.yaml` derives from total device
cycles over wall time on a saturated card (1,599.9 MHz).

Decode executions are selected as `kv_cache_size > 0 and attention_size - kv == 1`
-- the same rule `furiosa_llm/metadata/config_types.py` uses and STEP 0 committed.
Prefill and extend executions are reported separately rather than mixed in.

Usage::

    python experiments/scripts/npu_exec_attention_groups.py \\
        outputs/npu_spike/c3_bucket1024/edf outputs/npu_spike/c3_bucket2048/edf \\
        --csv profiler/perf/RNGD-CARD/.../attention_groups.csv \\
        --markdown experiments/results/npu_exec_attention_groups.md
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

#: meta.yaml: device cycles over wall time on a saturated card, 1,599.9 MHz.
CYCLES_PER_US = 1600.0

BUCKET_RE = re.compile(
    r"batch_size: (?P<bs>\d+), attention_size: (?P<attn>\d+), "
    r"kv_cache_size: (?P<kv>\d+)"
)
TAG_RE = re.compile(r"edf_c(?P<tag>[0-9p.]+)\.csv$")


def _conc(path: Path) -> float:
    m = TAG_RE.search(path.name)
    if not m:
        return 0.0
    try:
        return float(m.group("tag").replace("p", "."))
    except ValueError:
        return 0.0


def read_executions(csv_path: Path) -> dict:
    """Group every attention stage execution by (kind, batch_size, attention_size)."""
    decode: dict[tuple[int, int], list[int]] = defaultdict(list)
    prefill: dict[tuple[int, int], list[int]] = defaultdict(list)
    composed = tokenwise = total = 0
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            total += 1
            name = row["name"]
            if name.startswith("Composed"):
                composed += 1
                continue
            if name.startswith("Tokenwise"):
                tokenwise += 1
                continue
            if not name.startswith("Attention"):
                continue
            m = BUCKET_RE.search(name)
            if not m:
                continue
            bs, attn, kv = (int(m.group(k)) for k in ("bs", "attn", "kv"))
            cycles = int(row["cycle"])
            if kv > 0 and attn - kv == 1:
                decode[(bs, attn)].append(cycles)
            elif kv == 0:
                prefill[(bs, attn)].append(cycles)
    return {
        "total_stages": total,
        "composed": composed,
        "tokenwise": tokenwise,
        "decode": decode,
        "prefill": prefill,
    }


def fit_linear(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Least squares `us = fixed + slope * n`. Needs two distinct group sizes."""
    xs = [float(n) for n, _ in points]
    ys = [t for _, t in points]
    if len({*xs}) < 2:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
    return my - slope * mx, slope


def collect(dirs: list[Path], measured_on: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for d in dirs:
        for csv_path in sorted(d.glob("edf_c*.csv"), key=_conc):
            data = read_executions(csv_path)
            for (bs, attn), cycles in sorted(data["decode"].items()):
                rows.append({
                    # The card is a COLUMN, not a filename convention. This file
                    # lands beside a perf bundle built from a different card, and
                    # a per-execution cost is a card property -- so the one thing
                    # a reader must not have to infer is which card it is.
                    "measured_on": measured_on or "unknown",
                    "source": str(csv_path),
                    "concurrency": _conc(csv_path),
                    "kind": "decode",
                    "group_size": bs,
                    "attention_size": attn,
                    "executions": len(cycles),
                    "median_cycles": statistics.median(cycles),
                    "median_us": statistics.median(cycles) / CYCLES_PER_US,
                    "p05_us": sorted(cycles)[int(0.05 * len(cycles))] / CYCLES_PER_US,
                    "p95_us": sorted(cycles)[int(0.95 * len(cycles))] / CYCLES_PER_US,
                })
            print(f"  {csv_path.name}: {data['total_stages']} stages, "
                  f"{data['composed']} composed "
                  f"({data['composed'] / data['total_stages'] * 100:.0f} %), "
                  f"{len(data['decode'])} decode buckets", file=sys.stderr)
    return rows


def to_markdown(rows: list[dict], dirs: list[Path]) -> str:
    by_attn: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_attn[r["attention_size"]][r["group_size"]].append(r)

    out = [
        "# E-N8 — what one attention execution costs",
        "",
        "`WORK_ORDER_npu_exec_model_spike.md` STEP C.3. Read from EDF traces in "
        + ", ".join(f"`{d}`" for d in dirs)
        + ".",
        "",
        "**Measured on `npu2` (PCI `45:00.0`), not the `npu0` the committed bundle",
        "was built from** — `npu0` was held by another tenant's pod throughout. A",
        "per-execution cost is a property of the *card*, so these numbers must not be",
        "merged into the `npu0` bundle without a cross-card check.",
        "",
        "Decode executions only (`kv > 0` and `attention_size - kv == 1`). The EDF stage",
        "name carries the compiled bucket, so `batch_size` **is** the number of sequences",
        "that execution covered — the group size. Cycles convert at 1.6 GHz.",
        "",
    ]
    for attn in sorted(by_attn):
        out += [
            f"## attention_size = {attn}",
            "",
            "| group size | executions | median µs | p05 | p95 | at concurrency |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        pts = []
        for g in sorted(by_attn[attn]):
            rs = by_attn[attn][g]
            med = statistics.median([r["median_us"] for r in rs])
            pts.append((g, med))
            concs = ", ".join(f"c{r['concurrency']:g}" for r in rs)
            out.append(
                f"| {g} | {sum(r['executions'] for r in rs)} | {med:.1f} | "
                f"{min(r['p05_us'] for r in rs):.1f} | "
                f"{max(r['p95_us'] for r in rs):.1f} | {concs} |"
            )
        fit = fit_linear(pts)
        out.append("")
        if fit:
            fixed, slope = fit
            out.append(
                f"Least squares over the group sizes above: "
                f"**{fixed:.1f} µs fixed + {slope:.2f} µs per sequence**."
            )
            out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+", type=Path, help="directories holding edf_c*.csv")
    ap.add_argument("--measured-on", default=None,
                    help="the card these traces came from, e.g. npu2. Written as a "
                         "column, because the CSV sits beside a bundle built from "
                         "another card and the cost is a card property.")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    rows = collect(args.dirs, args.measured_on)
    if not rows:
        print("no decode attention executions found", file=sys.stderr)
        return 1

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.markdown:
        args.markdown.write_text(to_markdown(rows, args.dirs))
    if not args.csv and not args.markdown:
        print(to_markdown(rows, args.dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
