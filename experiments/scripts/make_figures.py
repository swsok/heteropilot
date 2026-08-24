"""Regenerate publication figures from committed result JSONs (work order §12, M6).

Reads the structured result artifacts in experiments/results/ and writes PNGs to
experiments/figures/. Pure plotting, no simulation — reproducible from the
committed data with one command:

    experiments/scripts/make_figures.py            # all figures
    experiments/scripts/make_figures.py --only exp1

Every figure title/caption states that the numbers are LLMServingSim predictions
(absolute rule 3). Figures whose source JSON is absent are skipped with a note
rather than fabricated; the pd_network_sweep.png figure is produced by its own
driver (pd_network_sweep.py) and is not regenerated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display, deterministic file output
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "experiments" / "results"
FIGURES = REPO_ROOT / "experiments" / "figures"

CAPTION = "LLMServingSim prediction (not a live measurement)"
GROUP_COLORS = {
    "optimizer": "#1f77b4",
    "resource": "#ff7f0e",
    "architecture": "#2ca02c",
    "ablation": "#d62728",
}


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        print(f"skip: {path} not found", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def make_exp1(data: dict) -> Path:
    """TP scaling: latency percentiles (log) + energy efficiency vs TP."""
    rows = sorted(data["rows"], key=lambda r: r["tp"])
    tps = [r["tp"] for r in rows]
    x = list(range(len(tps)))
    ttft = [r["p99_ttft_ms"] for r in rows]
    tpot = [r["p99_tpot_ms"] for r in rows]
    tpj = [r["tokens_per_joule"] for r in rows]
    power = [r["average_power_w"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    w = 0.38
    ax1.bar([i - w / 2 for i in x], ttft, w, label="p99 TTFT", color="#4c72b0")
    ax1.bar([i + w / 2 for i in x], tpot, w, label="p99 TPOT", color="#dd8452")
    ax1.set_yscale("log")
    ax1.set_ylabel("latency (ms, log)")
    ax1.set_xlabel("tensor-parallel degree")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"TP={t}" for t in tps])
    ax1.set_title("(a) Latency vs TP")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", axis="y", alpha=0.3)

    ax2.bar(x, tpj, 0.5, color="#55a868", label="tokens/J")
    ax2.set_ylabel("tokens / J")
    ax2.set_xlabel("tensor-parallel degree")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"TP={t}" for t in tps])
    ax2.set_title("(b) Efficiency vs TP")
    axp = ax2.twinx()
    axp.plot(x, power, "o-", color="#c44e52", label="avg power (W)")
    axp.set_ylabel("avg power (W)")
    lines = ax2.get_legend_handles_labels()[0] + axp.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + axp.get_legend_handles_labels()[1]
    ax2.legend(lines, labels, fontsize=8, loc="upper left")

    fig.suptitle(f"Exp 1 - same-GPU (A40) TP sweep, Llama-3.1-8B  |  {CAPTION}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "exp1_tp_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_baselines(data: dict) -> Path:
    """Regret per strategy, grouped/colored; N/A strategies annotated."""
    rows = data["rows"]
    labels, regrets, colors, annot = [], [], [], []
    for r in rows:
        labels.append(r["strategy"])
        colors.append(GROUP_COLORS.get(r["group"], "#888888"))
        if r.get("regret") is None:
            regrets.append(0.0)
            annot.append("n/a")
        else:
            regrets.append(float(r["regret"]))
            annot.append("")

    fig, ax = plt.subplots(figsize=(9, 6))
    y = list(range(len(labels)))[::-1]  # top-to-bottom in listed order
    ax.barh(y, regrets, color=colors)
    for yi, a, rg in zip(y, annot, regrets, strict=True):
        ax.text((rg if a == "" else 0.0) + 0.008, yi,
                a if a else f"{rg:.3f}", va="center", fontsize=8,
                color="#555" if a else "black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("regret vs oracle  (0 = oracle optimum; 1 = infeasible pick)")
    oracle = data.get("planner_metrics", {}).get("oracle_goodput_per_joule")
    otxt = f"oracle goodput/J = {oracle:.3f}" if isinstance(oracle, (int, float)) else ""
    ax.set_title(f"Baselines + ablation regret  |  {otxt}  |  {CAPTION}", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, list(GROUP_COLORS.keys()), fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "baselines_regret.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_router(data: dict) -> Path:
    """Router policy comparison: tail latency + SLO goodput per policy."""
    rows = [r for r in data["rows"] if "p99_ttft_ms" in r]
    pols = [r["policy"] for r in rows]
    x = list(range(len(pols)))
    ttft = [r["p99_ttft_ms"] for r in rows]
    goodput = [r["slo_goodput_rps"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.5
    bars = ax.bar(x, ttft, w, color="#4c72b0", label="p99 TTFT (ms)")
    ax.set_ylabel("p99 TTFT (ms)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(pols)
    ax.set_xlabel("request-routing policy")
    for b, v in zip(bars, ttft, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center",
                va="bottom", fontsize=8)
    axg = ax.twinx()
    axg.plot(x, goodput, "o-", color="#c44e52", label="SLO goodput (rps)")
    axg.set_ylabel("SLO goodput (rps)", color="#c44e52")
    axg.set_ylim(bottom=0)
    ax.set_title(f"Router baselines (heterogeneous 4-replica)  |  {CAPTION}", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "router_baselines.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


FIGURE_SPECS = {
    "exp1": ("exp1_tp_sweep.json", make_exp1),
    "baselines": ("baselines.json", make_baselines),
    "router": ("router_baselines.json", make_router),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(FIGURE_SPECS), help="build one figure")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    todo = [args.only] if args.only else list(FIGURE_SPECS)
    made = []
    for key in todo:
        src, fn = FIGURE_SPECS[key]
        data = _load(src)
        if data is None:
            continue
        out = fn(data)
        made.append(out)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    if not made:
        print("no figures produced (missing source JSON)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
