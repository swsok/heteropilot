#!/usr/bin/env python3
"""Put several `bench validate` summaries side by side.

`bench validate` writes one summary.txt per (sim run, bench run) pair. Reading
them one at a time hides the thing we actually care about - how the error moves
when one variable changes (hardware, profile grid density, memory accounting).

Usage:
    python experiments/scripts/compare_validations.py LABEL=path/to/summary.txt ...

Every summary should be against the same real measurement for the error columns
to be comparable; the script checks that and warns loudly when they are not.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROW = re.compile(r"^(\w+(?: \w+)*)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)%\s*$")


def parse(path: Path) -> dict[str, tuple[float, float, float]]:
    """metric -> (vllm, sim, diff_pct)."""
    out: dict[str, tuple[float, float, float]] = {}
    for line in path.read_text().splitlines():
        m = ROW.match(line.strip())
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    if not out:
        raise SystemExit(f"{path}: no metric rows parsed - is this a bench validate summary?")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)

    runs: dict[str, dict[str, tuple[float, float, float]]] = {}
    for arg in argv[1:]:
        if "=" not in arg:
            raise SystemExit(f"expected LABEL=path, got '{arg}'")
        label, _, path = arg.partition("=")
        runs[label] = parse(Path(path))

    labels = list(runs)
    metrics = list(runs[labels[0]])

    # The vLLM column must be identical across runs, otherwise the diffs are
    # measured against different ground truth and cannot be compared.
    shared_truth = True
    for metric in metrics:
        truths = {runs[lab][metric][0] for lab in labels if metric in runs[lab]}
        if len(truths) > 1:
            shared_truth = False
            break

    width = max(len(m) for m in metrics) + 2
    col = max(14, max(len(lab) for lab in labels) + 2)
    header = f"{'Metric':<{width}}{'vLLM':>11}" + "".join(f"{lab:>{col}}" for lab in labels)
    print(header)
    print("-" * len(header))
    for metric in metrics:
        vllm = runs[labels[0]][metric][0]
        row = f"{metric:<{width}}{vllm:>11.1f}"
        for label in labels:
            cell = runs[label].get(metric)
            row += f"{cell[2]:>+{col - 1}.1f}%" if cell else f"{'-':>{col}}"
        print(row)

    print("-" * len(header))
    row = f"{'mean |error|':<{width}}{'':>11}"
    for label in labels:
        vals = [abs(v[2]) for v in runs[label].values()]
        row += f"{sum(vals) / len(vals):>{col - 1}.2f}%"
    print(row)

    if len(labels) > 1:
        base = labels[0]
        print()
        # Compare |error|, not signed error. When one run over-predicts and the
        # other under-predicts, the difference of signed values reads backwards:
        # -32% -> -7.7% is a large improvement but a +24pp signed delta.
        print(f"Change in |error| vs '{base}' (negative = more accurate):")
        for label in labels[1:]:
            deltas = [
                abs(runs[label][m][2]) - abs(runs[base][m][2])
                for m in metrics
                if m in runs[label] and m in runs[base]
            ]
            worse = sum(1 for d in deltas if d > 0)
            print(f"  {label:<20} mean {sum(deltas) / len(deltas):+.2f}pp  "
                  f"range {min(deltas):+.2f} .. {max(deltas):+.2f}pp  "
                  f"({len(deltas) - worse}/{len(deltas)} metrics improved)")

    if not shared_truth:
        print()
        print("WARNING: the vLLM columns differ between runs. These summaries are")
        print("against different real measurements, so the error columns are NOT")
        print("directly comparable - only read each column against its own baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
