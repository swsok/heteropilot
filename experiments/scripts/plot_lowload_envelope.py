"""Four panels of the RNGD low-load envelope (WORK_ORDER_rps_aware.md STEP 3.3).

Pure plotting from the committed envelope YAML -- no simulation, no hardware.
The point of the figure is the contrast between the panels: throughput and
tokens/J move by ~9.5x while power moves by 1.085x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: deterministic file output
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"
OUT = ROOT / "experiments/figures/rngd_lowload_envelope.png"


def main() -> int:
    env = yaml.safe_load(ENV.read_text())
    pts = sorted(env["points"], key=lambda p: p["conc"])
    lo = [p for p in pts if p["power_w"] is not None]      # 2026-09-08, powered
    hi = [p for p in pts if p["power_w"] is None]          # D22, no power column

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("RNGD card (TP=8 internal), Llama-3.1-8B bf16 — served concurrency 1 to 107",
                 fontsize=12)

    def curve(a, key, label, ylab, logy=False):
        a.plot([p["conc"] for p in lo], [p[key] for p in lo],
               "o-", color="#1f77b4", label="2026-09-08 (power instrumented)")
        vals_hi = [p[key] for p in hi if p.get(key) is not None]
        if vals_hi:
            a.plot([p["conc"] for p in hi if p.get(key) is not None], vals_hi,
                   "s--", color="#888888", label="2026-08-31 (D22)")
        a.set_xscale("log")
        if logy:
            a.set_yscale("log")
        a.set_xlabel("served concurrency (Little's law)")
        a.set_ylabel(ylab)
        a.set_title(label, fontsize=10)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)

    curve(ax[0][0], "tput_tok_s", "Throughput", "output tok/s")
    curve(ax[0][1], "tpot_p50", "TPOT p50", "ms")

    # power: only the instrumented half exists, and saying so is the point
    a = ax[1][0]
    a.plot([p["conc"] for p in lo], [p["power_w"] for p in lo], "o-", color="#d62728")
    a.axhline(env["idle"]["standby_measured_w"], ls=":", color="#666666",
              label=f"standby {env['idle']['standby_measured_w']} W (model resident)")
    a.set_xscale("log")
    a.set_ylim(0, 320)
    a.set_xlim(0.85, 130)
    # The limit is on the x axis, not the y: power was never sampled above
    # conc 15.59, so shade the concurrency range rather than a wattage band.
    a.axvspan(15.59, 130, color="#cccccc", alpha=0.4)
    a.text(19, 250, "power not measured\nabove conc 15.59\n(validity: refuse)",
           fontsize=8, color="#555555")
    a.axhline(290.93, ls="--", color="#aaaaaa", lw=1)
    a.text(0.95, 296, "profile active_power 290.93 W (all 8 PEs)",
           fontsize=7, color="#888888")
    a.set_xlabel("served concurrency (Little's law)")
    a.set_ylabel("sustained card power (W)")
    a.set_title("Power — 1.085x across a 9.5x throughput range", fontsize=10)
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=8, loc="lower right")

    a = ax[1][1]
    a.plot([p["conc"] for p in lo], [p["tok_per_j"] for p in lo], "o-", color="#2ca02c")
    a.set_xscale("log")
    a.set_xlabel("served concurrency (Little's law)")
    a.set_ylabel("output tokens / joule")
    a.set_title("Energy efficiency — 9.4x, and still rising at the top", fontsize=10)
    a.grid(alpha=0.3, which="both")
    for p in lo:
        a.annotate(f"{p['tok_per_j']:.2f}", (p["conc"], p["tok_per_j"]),
                   textcoords="offset points", xytext=(4, -10), fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
