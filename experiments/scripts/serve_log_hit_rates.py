"""Tabulate cache hit rates from the committed `serve_*.log` files.

Two serving stacks, two log formats, and -- the part that matters -- two
different definitions of "hit rate":

* furiosa-llm (`furiosa_llm/server/metrics.py:98-99`) prints
  ``DpMetrics.prefix_cache_hit_rate`` / ``wire_hit_rate`` straight from the
  native runtime. The throughput fields on the same line are interval diffs;
  these two are NOT -- they are gauges the Rust side maintains, and the Python
  layer documents nothing about their window.
* vLLM (`vllm/v1/metrics/loggers.py:259` -> `stats.py:107`) prints
  ``CachingMetrics.hit_rate``, an explicit sliding window over the most recent
  ``max_recent_requests`` requests (default 1000).

So the two columns are not the same statistic and must not be averaged
together. Within one run the per-line values are also not independent samples
of a rate, which is why this reports first/last/min/max and never a mean.

Usage::

    python3 serve_log_hit_rates.py outputs/rngd_envelope*/**/serve_*.log ...
    python3 serve_log_hit_rates.py --markdown <logs...>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FURIOSA_RE = re.compile(
    r"\[Engine (?P<engine>\d+)\].*?"
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s.*?"
    r"Avg generation throughput: (?P<gen>[\d.]+) tokens/s.*?"
    r"Running: (?P<running>\d+) reqs.*?"
    r"Waiting: (?P<waiting>\d+) reqs.*?"
    r"RNGD KV cache usage: (?P<kv>[\d.]+)%.*?"
    r"Prefix cache hit rate: (?P<prefix>[\d.]+)%.*?"
    r"Wire pipeline hit rate: (?P<wire>[\d.]+)%"
)
VLLM_RE = re.compile(
    r"Engine (?P<engine>\d+):.*?"
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s.*?"
    r"Avg generation throughput: (?P<gen>[\d.]+) tokens/s.*?"
    r"Running: (?P<running>\d+) reqs.*?"
    r"Waiting: (?P<waiting>\d+) reqs.*?"
    r"GPU KV cache usage: (?P<kv>[\d.]+)%.*?"
    r"Prefix cache hit rate: (?P<prefix>[\d.]+)%"
)


def parse(path: Path) -> dict:
    rows: list[dict] = []
    stack = None
    for line in path.read_text(errors="replace").splitlines():
        match = FURIOSA_RE.search(line)
        if match:
            stack = "furiosa-llm"
        else:
            match = VLLM_RE.search(line)
            if match:
                stack = "vllm"
        if match is None:
            continue
        groups = match.groupdict()
        rows.append(
            {
                "prefix": float(groups["prefix"]),
                "wire": float(groups["wire"]) if "wire" in groups else None,
                "kv": float(groups["kv"]),
                "running": int(groups["running"]),
                "waiting": int(groups["waiting"]),
            }
        )
    return {"path": path, "stack": stack, "rows": rows}


def span(values: list[float]) -> tuple[float, float, float, float]:
    return values[0], values[-1], min(values), max(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    results = [parse(p) for p in sorted(args.logs)]
    results = [r for r in results if r["rows"]]

    header = (
        "| log | stack | metric lines | prefix first | prefix last | prefix min | prefix max "
        "| wire last | wire min | wire max | KV max | max running |"
    )
    if args.markdown:
        print(header)
        print("| " + " | ".join(["---"] * 11) + " |")

    for result in results:
        rows = result["rows"]
        pf, pl, pmin, pmax = span([r["prefix"] for r in rows])
        wires = [r["wire"] for r in rows if r["wire"] is not None]
        if wires:
            _, wl, wmin, wmax = span(wires)
            wire_cells = [f"{wl:.1f}", f"{wmin:.1f}", f"{wmax:.1f}"]
        else:
            wire_cells = ["n/a", "n/a", "n/a"]
        kv_max = max(r["kv"] for r in rows)
        run_max = max(r["running"] for r in rows)
        cells = [
            str(result["path"]),
            result["stack"],
            str(len(rows)),
            f"{pf:.1f}",
            f"{pl:.1f}",
            f"{pmin:.1f}",
            f"{pmax:.1f}",
            *wire_cells,
            f"{kv_max:.1f}",
            str(run_max),
        ]
        if args.markdown:
            print("| " + " | ".join(cells) + " |")
        else:
            print("  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
