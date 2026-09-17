#!/usr/bin/env python
"""Figure 2, drawn from measurement: the residual curve the disclosure illustrates.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V4. The disclosure's figure 2 uses
four invented points. This draws the nine measured RNGD-CARD points instead, on
the basis the feasibility check actually reads -- **p99 against p99** (V1's
percentile audit, deviations D101) -- and marks the three features the figure
exists to show: the sign change, the span where a one-sided margin charges
nothing, and the interval V1's §5.4 procedure actually registers.

The committed domain's own values are NOT used here. They pair a simulated p50
against a measured p50 for five points and against a measured MEAN for three
(D101), so plotting them would draw a curve on a basis no verdict is taken on.
The p99 pairing is recomputed from the raw artifacts by `v1_percentile_audit.py`.

Run:  .venv/bin/python experiments/p2_evidence/v4_figure2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v1_percentile_audit import audit  # noqa: E402

OUT = REPO / "experiments/figures/patent2_fig2_measured.png"
#: V1's one registered interval, on the p99 basis, placement C.
REGISTERED = (14.832, 25.181)


def r_from_e(e: float) -> float:
    """Disclosure convention from file convention: r = -e / (1 + e), percent."""
    return -(e / 100.0) / (1 + e / 100.0) * 100.0


def main() -> int:
    rows = [r for r in audit() if r["e_recomputed_p99"] is not None]
    x = [r["conc"] for r in rows]
    r = [r_from_e(r["e_recomputed_p99"]) for r in rows]
    margin = [max(0.0, v) for v in r]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhspan(-8, 0, color="0.93", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.axvspan(*REGISTERED, color="#cfe3f7", alpha=0.75, zorder=0,
               label=f"registered validation region [{REGISTERED[0]}, {REGISTERED[1]}]")

    ax.plot(x, r, "o-", color="#1f4e79", lw=1.6, ms=6, zorder=3,
            label="residual $r=(measured-predicted)/predicted$, p99 vs p99")
    # Above the sign change the margin IS the residual, so it is drawn on top
    # with a long dash: a thinner line underneath would simply disappear and the
    # figure would read as if no margin were applied there.
    ax.plot(x, margin, "s", color="#c0504d", ms=5, zorder=4)
    ax.plot(x, margin, color="#c0504d", lw=1.6, ls=(0, (7, 4)), zorder=4,
            label="applied margin $\\max(0, r)$ — one-sided;\n"
                  "coincides with $r$ where $r>0$")

    # The sign change, which is the feature the figure exists to show.
    neg = [c for c, v in zip(x, r, strict=True) if v < 0]
    pos = [c for c, v in zip(x, r, strict=True) if v >= 0]
    cross = (max(neg) + min(pos)) / 2 if neg and pos else None
    if cross:
        ax.axvline(cross, color="#c0504d", ls=":", lw=1.2, zorder=1)
        ax.annotate("sign change:\nno margin below, margin above",
                    xy=(cross, 0), xytext=(cross * 0.34, -6.4), fontsize=8,
                    color="#c0504d",
                    arrowprops={"arrowstyle": "->", "color": "#c0504d", "lw": 0.9})

    # A label sits above its point, except where the curve dips -- at c15.212 the
    # residual falls below its neighbour and a label above would land on the
    # line. Those go below-right instead.
    for i, (c, v) in enumerate(zip(x, r, strict=True)):
        dips = i > 0 and v < r[i - 1]
        dx, dy, ha = (9, -11, "left") if dips and v >= 0 else (
            (0, 7, "center") if v >= 0 else (0, -13, "center"))
        ax.annotate(f"{v:+.1f}", xy=(c, v), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=7.5,
                    color="#1f4e79")

    ax.set_xscale("log")
    ax.set_xlabel("predicted served concurrency $L$  (log scale)")
    ax.set_ylabel("residual / margin  (%)")
    ax.set_title("RNGD-CARD: measured residual on the p99 basis the verdict reads\n"
                 "(8 points; c16.6 has no per-request pair to recompute from)",
                 fontsize=10)
    ax.set_ylim(-9, 25)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT.relative_to(REPO)}")

    print(f"\n{'L':>9} {'e (file) %':>11} {'r (disclosure) %':>17} {'margin %':>9}")
    for row, rv, mv in zip(rows, r, margin, strict=True):
        print(f"{row['conc']:9.3f} {row['e_recomputed_p99']:11.2f} {rv:17.2f} {mv:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
