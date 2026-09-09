"""tok/J against RPS, per backend, with the validity of each point shaded.

`WORK_ORDER_rps_aware.md` rev 2 STEP 5. Reads the committed sweep JSONs -- no
simulation. The shading is the part that matters: a point whose operating point
sat outside its hardware's measured accuracy domain is drawn hollow, because a
curve that does not distinguish measured from extrapolated is how D22 happened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BACKEND_COLOUR = {"furiosa": "#d62728", "cuda": "#1f77b4"}


def backend_of(mix: str | None) -> str:
    if not mix:
        return "?"
    inner = mix.split("[", 1)[-1].rstrip("]")
    kinds = sorted({p.split(":", 1)[0] for p in inner.replace("] ", "+").split("+") if p})
    return "+".join(k for k in kinds if k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", nargs="+", type=Path, required=True,
                    help="pd_slo_sweep.json files, one per fixture")
    ap.add_argument("--measured", type=Path,
                    default=ROOT / "outputs/e6/e6b_measured.json")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/figures/e6_rps_sweep.png")
    args = ap.parse_args()

    sweeps = [(p.parent.name, json.loads(p.read_text())) for p in args.sweep]
    ttfts = sorted({r["ttft_slo_max_ms"] for _, d in sweeps for r in d["sweep"]},
                   reverse=True)
    fig, axes = plt.subplots(1, len(ttfts), figsize=(6.2 * len(ttfts), 5.0),
                             squeeze=False)
    fig.suptitle("E6a — recommended plan's tok/J against arrival rate "
                 "(hollow = operating point outside the measured accuracy domain)",
                 fontsize=11)

    for col, ttft in enumerate(ttfts):
        ax = axes[0][col]
        for fixture, d in sweeps:
            rows = sorted((r for r in d["sweep"] if r["ttft_slo_max_ms"] == ttft),
                          key=lambda r: r["rps"])
            xs = [r["rps"] for r in rows if r.get("recommended")]
            ys = [r["recommended"]["tokens_per_joule"] for r in rows if r.get("recommended")]
            if not xs:
                continue
            ax.plot(xs, ys, "-", color="#444444", lw=1, alpha=0.5, zorder=1)
            for r in rows:
                rec = r.get("recommended")
                if not rec:
                    ax.scatter([r["rps"]], [0], marker="x", color="#999999", zorder=3)
                    continue
                b = backend_of(rec.get("backend_mix"))
                colour = BACKEND_COLOUR.get(b, "#7f7f7f")
                measured = r.get("validity") == "measured"
                ax.scatter([r["rps"]], [rec["tokens_per_joule"]],
                           facecolors=colour if measured else "none",
                           edgecolors=colour, s=90, zorder=4,
                           label=f"{fixture} / {b}")
                ax.annotate(rec.get("backend_mix", "")[:22],
                            (r["rps"], rec["tokens_per_joule"]),
                            textcoords="offset points", xytext=(6, 5), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("arrival rate (rps, whole fleet)")
        ax.set_ylabel("tok/J of the recommended plan")
        ax.set_title(f"TTFT SLO <= {ttft/1000:.0f} s", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        handles, labels = ax.get_legend_handles_labels()
        seen: dict = dict(zip(labels, handles, strict=False))
        ax.legend(seen.values(), seen.keys(), fontsize=7)

    if args.measured.exists():
        m = json.loads(args.measured.read_text())
        note = ("RNGD measured curve (one card): " + ", ".join(
            f"{p['rps_per_card']:.3g} rps -> {p['measured_tok_per_j']:.2f} tok/J"
            for p in m["points"]))
        fig.text(0.01, 0.01, note + "   [E6b, silicon; the A40 side is simulation only]",
                 fontsize=7, color="#555555")

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
