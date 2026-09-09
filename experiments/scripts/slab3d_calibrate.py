"""Fit the dim-1 `link_latency` factor that makes a split allreduce match a flat one.

`WORK_ORDER_rps_aware.md` rev 2 STEP 2.2. `slab3d` turns a flat ring allreduce of
`tp` ranks into a hierarchical `tp/2 x 2`. The bandwidth term is identical in both
(`2 * (7/8) * size / bw` at tp=8: reduce-scatter + all-gather, hierarchical
3/4 + (1/4)(1/2) = 7/8), so only the LATENCY term moves -- 7 hops to 4 at tp=8.
The RNGD tp8 profile's `link_latency` was calibrated against a flat ring, so a
3-D encoding sees decode as faster than it is unless dim 1 is corrected.

The experiment holds everything constant except the split. One colocated instance,
same model, same trace, same bandwidth; `topology_mode: split2` gives dims
`[tp/2, 2]` with the allreduce spanning both, `auto` gives `[tp]` flat. For each
(split, link_bw) the dim-1 latency factor is swept over {1, 2, 4, 8} and the
factor minimising |TPOT p50 split - TPOT p50 flat| wins, with the remaining gap
recorded as the residual.

A factor is fitted PER (split, bw) and reported as a table. The spike found 4x at
[4,2] / bw 16 and it would be easy to call that a law; the work order forbids
that, and the validity block in the output refuses any combination not measured.

    experiments/scripts/slab3d_calibrate.py --out outputs/slab3d_calib
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.util.percentile import percentile  # noqa: E402

BASE_LATENCY_NS = 20000.0
FACTORS = (1, 2, 4, 8)
#: 16.0 reproduces the spike; 35.2 is the composed A40<->A40 value; 100.0 is an
#: NVLink-class upper bound. 12.6 was added after STEP 2.3's first end-to-end run:
#: it is the A40<->RNGD cross-vendor link in `pd-rngd-gpu.yaml`, i.e. the bandwidth
#: the asymmetric candidates D28 exists to enable actually run at, and the work
#: order's three did not cover it. The compiler refused every one of them, which
#: is the refusal working -- and the reason to measure rather than widen the
#: domain by fiat. 7.7 came from the same run: it is `fabric-rngd0-rngd1`, the
#: RNGD<->RNGD link, i.e. the asymmetric NPU P/D case D14/D16(b) was written
#: about.
BANDWIDTHS = (7.7, 12.6, 16.0, 35.2, 100.0)

#: (label, base fixture, tp) -- the two shapes the work order names.
#: tp2 gives the `[1,2]` split, which `plan --enable-pd` produces for an A40 tp1
#: prefill paired with a tp2 decode. Degenerate-looking but real, and refusing it
#: left a fifth of the fixture's asymmetric candidates unevaluated.
SHAPES = (
    ("tp8", "experiments/configs/clusters/rngd-llama31-8b-tp8.json", 8),
    ("tp4", "experiments/configs/clusters/colocated-tp4x2-auto.json", 4),
    ("tp2", "experiments/configs/clusters/colocated-tp4x2-auto.json", 2),
)


def _base_config(path: str, tp: int) -> dict:
    cfg = json.loads((ROOT / path).read_text())
    # One colocated instance only: split2 refuses more, and the calibration needs
    # exactly one TP group so the split is the only thing that differs.
    node = copy.deepcopy(cfg["nodes"][0])
    node["num_instances"] = 1
    node["instances"] = node["instances"][:1]
    # The base fixture supplies the hardware, model and power block; `tp` is the
    # experimental variable, so it is set here rather than requiring one committed
    # fixture per width.
    if node["instances"][0]["tp_size"] != tp:
        if tp > node["instances"][0]["tp_size"]:
            raise ValueError(
                f"{path} has tp{node['instances'][0]['tp_size']}; cannot widen to tp{tp} "
                f"without more devices than the fixture declares"
            )
        node["instances"][0]["tp_size"] = tp
        node["instances"][0]["num_npus"] = tp
    cfg["nodes"] = [node]
    cfg["num_nodes"] = 1
    return cfg


def write_config(out: Path, base: dict, *, bw: float, mode: str, tp: int,
                 dim1_factor: int | None) -> Path:
    cfg = copy.deepcopy(base)
    cfg["link_bw"] = bw
    if mode == "split2":
        cfg["topology_mode"] = "split2"
        # dims are [tp/2, 2]; dim 0 keeps the calibrated per-hop latency and dim 1
        # carries the correction under test.
        cfg["link_latency"] = [BASE_LATENCY_NS, BASE_LATENCY_NS * float(dim1_factor)]
    else:
        cfg.pop("topology_mode", None)
        cfg["link_latency"] = BASE_LATENCY_NS      # dims are [tp]; one value
    out.write_text(json.dumps(cfg, indent=4) + "\n")
    return out


def _rel(p: Path) -> str:
    """Repo-relative, because the frontend prefixes '../' to the paths it is given
    (it runs the ASTRA-Sim child from inside astra-sim/). An absolute path becomes
    '..//home/...' and fails to open -- which is exactly how the first run of this
    driver lost all 30 runs."""
    return str(Path(p).resolve().relative_to(ROOT))


def run_sim(cfg: Path, out_dir: Path, name: str, dataset: Path, num_reqs: int,
            timeout: float) -> dict:
    csv = out_dir / f"{name}.csv"
    cmd = [
        "experiments/scripts/livelock_watch.sh", "-n", "45", "-g", "600",
        "-s", "600", "-t", str(int(timeout)), "--",
        ".venv/bin/python", "-m", "serving",
        "--cluster-config", _rel(cfg), "--dataset", _rel(dataset),
        "--output", _rel(csv), "--num-reqs", str(num_reqs),
        "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--block-size", "16",
        "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
        "--request-routing-policy", "LOAD", "--network-backend", "analytical",
        "--log-level", "WARNING", "--log-interval", "1.0",
        "--no-enable-prefix-caching", "--run-id", f"calib-{name}",
    ]
    log = out_dir / f"{name}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=fh,
                            timeout=timeout + 120).returncode
    return {"name": name, "rc": rc, "csv": str(csv), "log": str(log),
            **_tpot_from_csv(csv)}


def _tpot_from_csv(csv: Path) -> dict:
    """p50 TPOT in ms from the simulator CSV's own TPOT column (ns).

    Read rather than re-derived from latency/TTFT/output: the CSV publishes TPOT
    directly, and recomputing it would introduce a second definition to disagree
    with the planner's. The percentile comes from `planner/util/percentile.py` for
    the same reason. The `ITL` column holds a bracketed list, so the row is split
    on the leading fields only.
    """
    if not csv.exists():
        return {"tpot_p50": None, "rows": 0}
    lines = csv.read_text().splitlines()
    if len(lines) < 2:
        return {"tpot_p50": None, "rows": max(0, len(lines) - 1)}
    header = [h.strip() for h in lines[0].split(",")]
    if "TPOT" not in header:
        return {"tpot_p50": None, "rows": len(lines) - 1, "note": f"header {header}"}
    idx = header.index("TPOT")
    tpots = []
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) <= idx:
            continue
        try:
            tpots.append(float(cells[idx]) / 1e6)      # ns -> ms
        except ValueError:
            continue
    return {"tpot_p50": percentile(tpots, 50) if tpots else None,
            "rows": len(lines) - 1, "n_tpot": len(tpots)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/slab3d_calib")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/d23/trace20.jsonl")
    ap.add_argument("--num-reqs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=1500)
    args = ap.parse_args()

    out = args.out
    (out / "configs").mkdir(parents=True, exist_ok=True)

    jobs = []
    for label, path, tp in SHAPES:
        base = _base_config(path, tp)
        for bw in BANDWIDTHS:
            bwt = str(bw).replace(".", "p")
            cfg = write_config(out / "configs" / f"{label}_bw{bwt}_flat.json",
                               base, bw=bw, mode="auto", tp=tp, dim1_factor=None)
            jobs.append((f"{label}_bw{bwt}_flat", cfg))
            for f in FACTORS:
                cfg = write_config(out / "configs" / f"{label}_bw{bwt}_split_f{f}.json",
                                   base, bw=bw, mode="split2", tp=tp, dim1_factor=f)
                jobs.append((f"{label}_bw{bwt}_split_f{f}", cfg))

    print(f"{len(jobs)} runs, {args.workers} workers", file=sys.stderr)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_sim, cfg, out, name, args.dataset, args.num_reqs,
                          args.timeout): name for name, cfg in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results[r["name"]] = r
            print(f"  [{i}/{len(jobs)}] {r['name']} rc={r['rc']} "
                  f"tpot_p50={r['tpot_p50']}", file=sys.stderr)

    (out / "runs.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    # --- fit ---
    fits = []
    for label, _, tp in SHAPES:
        for bw in BANDWIDTHS:
            bwt = str(bw).replace(".", "p")
            flat = results.get(f"{label}_bw{bwt}_flat", {}).get("tpot_p50")
            if flat is None:
                fits.append({"split": f"[{tp // 2},2]", "tp": tp, "link_bw": bw,
                             "factor": None, "residual_pct": None,
                             "note": "flat reference did not produce TPOT"})
                continue
            cand = []
            for f in FACTORS:
                sp = results.get(f"{label}_bw{bwt}_split_f{f}", {}).get("tpot_p50")
                if sp is None:
                    continue
                cand.append((abs(sp - flat) / flat * 100.0, f, sp))
            if not cand:
                fits.append({"split": f"[{tp // 2},2]", "tp": tp, "link_bw": bw,
                             "factor": None, "residual_pct": None,
                             "note": "no split run produced TPOT"})
                continue
            cand.sort()
            resid, best, sp = cand[0]
            fits.append({
                "split": f"[{tp // 2},2]", "tp": tp, "link_bw": bw,
                "factor": best, "residual_pct": round(resid, 4),
                "flat_tpot_p50_ms": flat, "split_tpot_p50_ms": sp,
                "all_factors": {str(f): results.get(f"{label}_bw{bwt}_split_f{f}", {})
                                .get("tpot_p50") for f in FACTORS},
            })
    (out / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")

    print("\nsplit      bw     factor  residual%   flat p50    split p50", file=sys.stderr)
    for f in fits:
        if f["factor"] is None:
            print(f"{f['split']:<10} {f['link_bw']:<6} --      {f.get('note')}",
                  file=sys.stderr)
            continue
        print(f"{f['split']:<10} {f['link_bw']:<6} {f['factor']:<7} "
              f"{f['residual_pct']:<11.4f} {f['flat_tpot_p50_ms']:<11.4f} "
              f"{f['split_tpot_p50_ms']:.4f}", file=sys.stderr)
    print(f"\nwrote {out/'fits.json'}", file=sys.stderr)

    fitted = [f for f in fits if f["factor"] is not None]
    failed = [name for name, r in results.items() if r["rc"] != 0]
    if failed:
        print(f"\n{len(failed)} of {len(results)} runs failed, e.g. {failed[:3]}",
              file=sys.stderr)
    if len(fitted) < len(fits):
        # Silence here is how the first attempt reported "done" with 30 dead runs.
        print(f"error: only {len(fitted)} of {len(fits)} points fitted",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
