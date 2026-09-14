"""Turn E-B1's raw curves into the result table and figure (STEP B4).

Kept apart from the sweep so the expensive part runs once and the presentation
can be redone. Reads `outputs/uncertainty/eb1/eb1_regret_vs_budget.json`.

Two things here are worth stating rather than burying in the code.

**Averaging step functions.** A curve is `(hours spent, regret)` after each
purchase, and different degradation sets buy different items, so no two curves
share an x axis. Each is therefore read as the step function it is - the regret
still standing once a budget `B` has been spent is the regret after the last
purchase that fit in `B` - and sampled onto one common budget grid before
averaging. Interpolating between purchases would invent budgets at which a
half-bought measurement had half-paid off.

**Two aggregates, both reported.** On this fixture most degradation sets do not
move the recommendation at all, so their regret is zero for every strategy at
every budget and they pull all five means towards each other. The tables give
the mean over ALL sets and the mean over only those sets that started with
regret > 0 - the ones where a measurement was needed. Neither is the "real"
number; the first says what a random degradation costs you, the second says what
the ranking is worth when it matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW = Path("outputs/uncertainty/eb1/eb1_regret_vs_budget.json")


def regret_at(curve: list[list[float]], budget: float) -> float:
    """Regret still standing after spending at most `budget`."""
    out = curve[0][1] if curve else 0.0
    for spent, regret in curve:
        if spent <= budget + 1e-9:
            out = regret
    return out


def zero_budget(curve: list[list[float]]) -> float | None:
    """Least spend at which regret first reaches zero. None if it never does."""
    for spent, regret in curve:
        if regret <= 1e-9:
            return spent
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (mean, var ** 0.5)


def summarize(runs: list[dict], grid: list[float]) -> dict:
    strategies = sorted({r["strategy"] for r in runs})
    out: dict = {}
    for strategy in strategies:
        rows = [r for r in runs if r["strategy"] == strategy]
        curves = [r["curve"] for r in rows]
        series = []
        for b in grid:
            series.append(mean_std([regret_at(c, b) for c in curves]))
        zeros = [zero_budget(c) for c in curves]
        reached = [z for z in zeros if z is not None]
        out[strategy] = {
            "n": len(rows),
            "budget_grid": grid,
            "mean": [m for m, _s in series],
            "std": [s for _m, s in series],
            "mean_zero_budget_h": (sum(reached) / len(reached)) if reached else None,
            "never_reached_zero": len(zeros) - len(reached),
            "area_under_regret": sum(m for m, _s in series) * (
                (grid[-1] - grid[0]) / max(1, len(grid) - 1)
            ),
        }
    return out


#: Budgets the tables quote: nothing, one accuracy-domain operating point, one
#: link probe, one full profiling run, and the most any k=3 set can spend.
PICKS = (0.0, 0.041, 0.114, 2.1, 6.3)


def at_budgets(curves: list[list[list[float]]], budgets) -> list[tuple[float, float]]:
    """Mean and sd of the regret still standing at each budget.

    Read off the CURVES, never off the plotting grid. An earlier version looked
    the budget up in the 80-point grid the figure uses, which snaps 0.041 h and
    0.114 h to the same grid point as 0 h - so three columns of the table were
    silently the same column.
    """
    return [mean_std([regret_at(c, b) for c in curves]) for b in budgets]


def _table(summary: dict, curves_by_strategy: dict) -> list[str]:
    head = " | ".join(f"R({p:g} h)" for p in PICKS)
    lines = [
        f"| strategy | n | {head} | mean h to regret 0 | never reached 0 |",
        "|---|---:|" + "---:|" * len(PICKS) + "---:|---:|",
    ]
    for strategy, row in summary.items():
        sampled = at_budgets(curves_by_strategy[strategy], PICKS)
        cells = " | ".join(f"{m:,.1f}" for m, _s in sampled)
        zero = row["mean_zero_budget_h"]
        lines.append(
            f"| `{strategy}` | {row['n']} | {cells} | "
            f"{'—' if zero is None else f'{zero:.3f}'} | {row['never_reached_zero']} |"
        )
    lines.append("")
    return lines


def figure(summaries: dict, grid: list[float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(summaries)
    cols = 2 if len(keys) > 1 else 1
    rows = (len(keys) + cols - 1) // cols
    fig, grid_axes = plt.subplots(
        rows, cols, figsize=(5.6 * cols, 3.9 * rows), sharex=True,
    )
    axes = list(grid_axes.flat) if len(keys) > 1 else [grid_axes]
    for ax, key in zip(axes, keys, strict=False):
        for strategy, row in summaries[key].items():
            mean = row["mean"]
            std = row["std"]
            # `oracle` and `ours` coincide exactly at the default penalty, so the
            # oracle is drawn wide and dashed underneath rather than hidden.
            style: dict = {"linewidth": 3.2, "linestyle": "--",
                           "alpha": 0.55, "zorder": 1}
            if strategy != "oracle":
                style = {"linewidth": 1.7, "zorder": 2}
            ax.plot(grid, mean, label=strategy, **style)
            ax.fill_between(
                grid,
                # Regret is non-negative by construction (§2.5), so the band is
                # clipped at 0: an interval reaching -10 000 J would be an
                # artifact of a symmetric sd on a one-sided quantity.
                [max(0.0, m - s) for m, s in zip(mean, std, strict=True)],
                [m + s for m, s in zip(mean, std, strict=True)],
                alpha=0.10, zorder=0,
            )
        ax.set_title(key, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("measurement budget (server-hours)")
        ax.set_ylabel("decision regret (J, truth-evaluated)")
        ax.grid(alpha=0.25, linewidth=0.5)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/uncertainty/results/eb1_regret_vs_budget.md"))
    parser.add_argument("--figure", type=Path,
                        default=Path("experiments/uncertainty/results/figures/eb1_regret_vs_budget.png"))
    parser.add_argument("--points", type=int, default=80)
    parser.add_argument("--summary-out", type=Path,
                        default=Path("outputs/uncertainty/eb1/eb1_summary.json"))
    args = parser.parse_args()

    payload = json.loads(args.raw.read_text())
    runs = payload["runs"]
    top = max(
        (c[0] for r in runs for c in r["curve"]), default=1.0
    )
    grid = [top * i / (args.points - 1) for i in range(args.points)]

    summaries: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    for penalty in sorted({r["penalty"] for r in runs}):
        for scope, keep in (
            ("all sets", lambda r: True),
            ("sets that moved the plan", lambda r: r["initial_regret"] > 1e-9),
        ):
            rows = [r for r in runs if r["penalty"] == penalty and keep(r)]
            if rows:
                key = f"penalty={penalty}, {scope}"
                summaries[key] = summarize(rows, grid)
                curves[key] = {
                    strategy: [r["curve"] for r in rows if r["strategy"] == strategy]
                    for strategy in sorted({r["strategy"] for r in rows})
                }

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure(summaries, grid, args.figure)
    out = {
        "summaries": summaries,
        "at_budgets": {
            key: {
                strategy: {
                    "budgets": list(PICKS),
                    "mean": [m for m, _s in at_budgets(cs, PICKS)],
                    "std": [sd for _m, sd in at_budgets(cs, PICKS)],
                }
                for strategy, cs in by_strategy.items()
            }
            for key, by_strategy in curves.items()
        },
        "budget_grid": grid,
        "source": str(args.raw),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(out, indent=2))
    # `--out` used to be accepted and then never written: the tables went to
    # stdout only, so the "generated" half of the result document was whatever
    # had last been pasted into it by hand.
    lines = [
        "<!-- Generated by experiments/uncertainty/eb1_report.py; do not hand-edit. -->",
        f"<!-- source: {args.raw} -->",
        "",
    ]
    for key, summary in summaries.items():
        lines += [f"### {key}", ""]
        lines += _table(summary, curves[key])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    for key, summary in summaries.items():
        print(f"\n== {key}")
        for line in _table(summary, curves[key]):
            print(line)
    print(f"figure: {args.figure}")
    print(f"tables: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
