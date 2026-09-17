"""Compare two simulated runs' TTFT, point by point.

`WORK_ORDER_npu_exec_model_spike.md` STEP B.2 asks for TTFT in a table of its own,
and this is why it has to be its own table: the bench side is closed-loop and the
simulator replays an arrival process, so simulated TTFT cannot be scored against
the measured envelope (D19). It *can* be compared against another simulated run,
which is what the prototype question actually is -- does making prefill steps
exclusive and `bs = 1` move TTFT, and by how much.

So every number here is **sim against sim**. Nothing is compared to hardware and
no claim about real TTFT error follows from it. The one measured number in the
neighbourhood, D17's -32.6 %, is quoted for scale and is not what this scores.

Percentiles come from `planner/util/percentile.py`, so they interpolate the way
every other percentile in the repo does.

Usage::

    python experiments/scripts/npu_exec_ttft_compare.py \\
        --baseline outputs/npu_spike/a1_current \\
        --variant  outputs/npu_spike/b_aot_strict \\
        --label-baseline vllm --label-variant bucketed_aot/strict \\
        --markdown experiments/results/npu_exec_ttft.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.util.percentile import percentile  # noqa: E402


def _tag_order(tag: str) -> tuple:
    try:
        return (0, float(tag.lstrip("c").replace("p", ".")))
    except ValueError:
        return (1, 0.0)


def read_ttft_ms(csv_path: Path) -> list[float]:
    """The TTFT column, in milliseconds. The CSV records nanoseconds."""
    out: list[float] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("TTFT")
            if raw in (None, ""):
                continue
            out.append(float(raw) / 1e6)
    return out


def summarise(csv_path: Path) -> dict | None:
    vals = read_ttft_ms(csv_path)
    if not vals:
        return None
    return {
        "n": len(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }


def collect(run_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for csv_path in run_dir.glob("sim_*.csv"):
        tag = csv_path.stem[len("sim_"):]
        s = summarise(csv_path)
        if s:
            rows[tag] = s
    return rows


def to_markdown(pairs: list[tuple[str, dict, dict]], lb: str, lv: str,
                base_dir: Path, var_dir: Path) -> str:
    out = [
        "# STEP B.2 — TTFT under the prototype, simulator against simulator",
        "",
        f"`{base_dir}` (**{lb}**) against `{var_dir}` (**{lv}**).",
        "",
        "**Both columns are simulated.** The measured envelope's TTFT is closed-loop",
        "and the simulator replays an arrival process, so the two are not comparable",
        "(D19); this table scores the prototype against the untouched step policy and",
        "nothing else. D17's measured TTFT error of -32.6 % is quoted elsewhere for",
        "scale and is not what is being tested here.",
        "",
        f"| point | requests | {lb} p50 | {lv} p50 | Δ p50 | {lb} p95 | {lv} p95 | Δ p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, b, v in pairs:
        d50 = (v["p50"] / b["p50"] - 1) * 100 if b["p50"] else None
        d95 = (v["p95"] / b["p95"] - 1) * 100 if b["p95"] else None
        out.append(
            f"| {tag} | {b['n']} | {b['p50']:.1f} | {v['p50']:.1f} | "
            f"{'--' if d50 is None else f'{d50:+.1f} %'} | "
            f"{b['p95']:.1f} | {v['p95']:.1f} | "
            f"{'--' if d95 is None else f'{d95:+.1f} %'} |"
        )
    out.append("")
    out.append("Times in milliseconds.")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-variant", default="variant")
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    base, var = collect(args.baseline), collect(args.variant)
    shared = sorted(set(base) & set(var), key=_tag_order)
    if not shared:
        print("no points in common", file=sys.stderr)
        return 1
    for only, where in ((set(base) - set(var), args.baseline),
                        (set(var) - set(base), args.variant)):
        if only:
            print(f"  only in {where}: {sorted(only, key=_tag_order)}", file=sys.stderr)

    pairs = [(t, base[t], var[t]) for t in shared]
    for tag, b, v in pairs:
        print(f"  {tag}: p50 {b['p50']:.1f} -> {v['p50']:.1f} ms "
              f"({(v['p50'] / b['p50'] - 1) * 100:+.1f} %)  "
              f"p95 {b['p95']:.1f} -> {v['p95']:.1f} ms "
              f"({(v['p95'] / b['p95'] - 1) * 100:+.1f} %)", file=sys.stderr)

    md = to_markdown(pairs, args.label_baseline, args.label_variant,
                     args.baseline, args.variant)
    if args.markdown:
        args.markdown.write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
