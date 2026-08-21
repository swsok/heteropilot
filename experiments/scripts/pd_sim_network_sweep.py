"""Simulator-level P/D network-bandwidth sweep (Phase 5, docs/deviations.md D15).

Companion to ``pd_network_sweep.py``. That driver sweeps bandwidth *at the
planning level* because, until this change, the simulator charged the
prefill->decode KV handoff as free (its docstring still says so). This driver
sweeps the same bandwidth *inside the simulator*: it runs ``python -m serving``
once per bandwidth with ``--pd-transfer-model bandwidth`` and shows the
simulator's own P/D metrics move.

What the simulator models now
-----------------------------
``--pd-transfer-model bandwidth`` delays a transferred request's decode
eligibility by ``link_latency + KV_bytes / cross_instance_link_bw``. Because
this simulator emits the first token on the *prefill* instance, the delay lands
in end-to-end latency, ITL[0] and (smeared) TPOT -- NOT in TTFT. This is the
sim-honest bucket chosen deliberately (docs/deviations.md D15); it differs from
the planner-side add-on in ``pd_network_sweep.py``, which charges TTFT.

Pass criteria (asserted at the end)
-----------------------------------
1. ``bandwidth`` mode: mean latency and mean TPOT increase monotonically as the
   cross-instance bandwidth drops.
2. ``bandwidth`` mode: mean TTFT stays flat (no systematic movement with bw).
3. ``none`` mode (control): latency/TPOT do not move with bw -- proving the
   effect in (1) comes from this model, not from the collective network sim.
   This control assumes a tp=1 P/D config (the default): with tp>1, overriding
   ``link_bw`` also slows the intra-instance TP AllReduce, so ``none`` mode would
   legitimately move and the none_flat check would FAIL for a real reason, not a
   code bug.

This script doubles as the end-to-end regression check for deviations D15
(byte-identical default is checked separately; here we pin that the simulator's
own P/D metrics respond to bandwidth). It needs a built simulator, so it is run
manually rather than in the fast pytest suite.

Honesty
-------
Every number is a simulator prediction. The transfer time is a hand-computed
delay from KV bytes and one cross-instance link bandwidth; it does not model
contention between the KV transfer and concurrent TP collectives on a shared
link (that would need a real send/recv in the Chakra/ASTRA graph -- see the
D15 "deferred" note). Do not over-read the absolute magnitudes: at realistic
P/D bandwidths (>=25 GB/s) and short prompts the transfer is sub-millisecond,
consistent with the increment-2 finding.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_once(cluster_cfg: Path, mode: str, bw: float, dataset: str,
              num_reqs: int, out_csv: Path, run_id: str, inputs_root: Path,
              python: str) -> None:
    """Run one simulation with link_bw overridden to ``bw`` and the given
    transfer model. Writes the per-request CSV to ``out_csv``."""
    base = json.loads(cluster_cfg.read_text())
    base["link_bw"] = bw
    # Write the per-bandwidth config next to the repo (relative path): the
    # simulator prepends "../" to a non-absolute cluster-config path and cannot
    # open an absolute path under /tmp, so it must live under the repo tree.
    cfg = REPO_ROOT / "outputs" / ".pd_sim_sweep" / f"cfg_{mode}_{bw}.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(base))
    cmd = [
        python, "-m", "serving",
        "--cluster-config", str(cfg.relative_to(REPO_ROOT)),
        "--dtype", "bfloat16", "--block-size", "16",
        "--dataset", dataset, "--num-reqs", str(num_reqs),
        "--output", str(out_csv),
        "--run-id", run_id, "--inputs-root", str(inputs_root),
        "--pd-transfer-model", mode, "--log-level", "WARNING",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not out_csv.exists():
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError(f"simulation failed: mode={mode} bw={bw}")


def _summarize(out_csv: Path) -> dict:
    """Mean TTFT / TPOT / latency (ns) over completed requests in a run CSV."""
    rows = list(csv.DictReader(out_csv.open()))
    cols = {k.lower(): k for k in rows[0]}

    def col(name: str) -> list[float]:
        return [float(r[cols[name]]) for r in rows if r[cols[name]] not in ("", "-1")]

    return {
        "n": len(rows),
        "ttft_mean_ns": statistics.mean(col("ttft")),
        "tpot_mean_ns": statistics.mean(col("tpot")),
        "latency_mean_ns": statistics.mean(col("latency")),
    }


def _is_monotonic_increasing(values: list[float], tol: float = 0.0) -> bool:
    """True if each value is >= the previous one (bandwidths passed high->low)."""
    return all(b >= a - tol for a, b in itertools.pairwise(values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-config",
                    default="configs/cluster/single_node_pd_instance.json",
                    help="P/D-split cluster config to sweep")
    ap.add_argument("--dataset", default="workloads/example_trace.jsonl")
    ap.add_argument("--num-reqs", type=int, default=20)
    ap.add_argument("--bandwidths", type=float, nargs="+",
                    default=[16.0, 4.0, 1.0],
                    help="cross-instance link_bw values (GB/s), high to low")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="experiments/results/pd_sim_network_sweep_table.md")
    args = ap.parse_args()

    cluster_cfg = REPO_ROOT / args.cluster_config
    bws = list(args.bandwidths)

    with tempfile.TemporaryDirectory(prefix="pd_sim_sweep_") as tmp:
        tmp_root = Path(tmp)
        bw_rows, none_rows = [], []
        for bw in bws:
            out = tmp_root / f"bw_{bw}.csv"
            _run_once(cluster_cfg, "bandwidth", bw, args.dataset, args.num_reqs,
                      out, f"pdsim_bw_{bw}", tmp_root / f"in_bw_{bw}", args.python)
            bw_rows.append((bw, _summarize(out)))
        for bw in (bws[0], bws[-1]):
            out = tmp_root / f"none_{bw}.csv"
            _run_once(cluster_cfg, "none", bw, args.dataset, args.num_reqs,
                      out, f"pdsim_none_{bw}", tmp_root / f"in_none_{bw}", args.python)
            none_rows.append((bw, _summarize(out)))

    lat = [s["latency_mean_ns"] for _, s in bw_rows]
    tpot = [s["tpot_mean_ns"] for _, s in bw_rows]
    ttft = [s["ttft_mean_ns"] for _, s in bw_rows]
    lat_mono = _is_monotonic_increasing(lat)
    tpot_mono = _is_monotonic_increasing(tpot)
    # TTFT must NOT move systematically: max spread stays under 0.5% of the mean.
    ttft_flat = (max(ttft) - min(ttft)) / statistics.mean(ttft) < 0.005
    # none-mode control: latency spread across the extreme bandwidths is tiny.
    none_lat = [s["latency_mean_ns"] for _, s in none_rows]
    none_flat = abs(none_lat[0] - none_lat[-1]) / statistics.mean(none_lat) < 0.005

    lines = [
        "# Simulator-level P/D network-bandwidth sweep",
        "",
        f"- cluster: `{args.cluster_config}`  dataset: `{args.dataset}`  "
        f"num_reqs: {args.num_reqs}",
        "- All numbers are simulator predictions (ms). Transfer time is a "
        "hand-computed KV_bytes/link_bw delay charged to latency/TPOT, not TTFT "
        "(docs/deviations.md D15).",
        "",
        "## `--pd-transfer-model bandwidth`",
        "",
        "| link_bw (GB/s) | n | TTFT mean (ms) | TPOT mean (ms) | latency mean (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for bw, s in bw_rows:
        lines.append(
            f"| {bw:g} | {s['n']} | {s['ttft_mean_ns']/1e6:.4f} | "
            f"{s['tpot_mean_ns']/1e6:.4f} | {s['latency_mean_ns']/1e6:.4f} |")
    lines += [
        "",
        "## `--pd-transfer-model none` (control)",
        "",
        "| link_bw (GB/s) | n | TTFT mean (ms) | TPOT mean (ms) | latency mean (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for bw, s in none_rows:
        lines.append(
            f"| {bw:g} | {s['n']} | {s['ttft_mean_ns']/1e6:.4f} | "
            f"{s['tpot_mean_ns']/1e6:.4f} | {s['latency_mean_ns']/1e6:.4f} |")
    lines += [
        "",
        "## Verdict",
        "",
        f"- latency monotonic increasing as bw drops: {'PASS' if lat_mono else 'FAIL'}",
        f"- TPOT monotonic increasing as bw drops: {'PASS' if tpot_mono else 'FAIL'}",
        f"- TTFT flat (<0.5% spread): {'PASS' if ttft_flat else 'FAIL'}",
        f"- none-mode latency flat across bw (control): {'PASS' if none_flat else 'FAIL'}",
    ]
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    ok = lat_mono and tpot_mono and ttft_flat and none_flat
    print(f"\nWrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
