"""Compare the layerwise-profile prediction against a real furiosa-llm run.

Closes the loop opened by ``profile_rngd.py``: the bundle it measured predicts
per-request TTFT/TPOT through LLMServingSim, and ``bench_furiosa_endpoint.py``
measures the same workload on the real serving stack. This script puts the two
side by side in the canonical ``summary.txt`` format (the same
``bench.core.plots.write_summary`` the A40 comparison used, so the artifacts are
directly comparable), then fits ``real = alpha * sim + beta`` with the existing
``planner.predictor.calibration`` code.

Runs in ``.venv``: it only reads files.

What makes this comparison honest, and where it is not apples to apples:

* Same workload, same request set, same TP degree -- required, and checked here.
* The simulator replays exact output token counts; the real server may stop
  early. The row counts and generated-token totals are reported so any drift is
  visible instead of quietly biasing TPOT.
* The perf bundle was measured on layer implementations written for the harness,
  not on furiosa-llm's own compiled graph, and the served weights are the
  Instruct variant of the model whose base weights were profiled (same
  architecture and dimensions). So a gap here is not purely simulator error --
  it also contains the difference between two software stacks.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/scripts/compare_rngd_sim_vs_real.py \
        --sim-csv outputs/envcheck/rngd_verify_tp8.csv \
        --real-json outputs/rngd_bench/real_tp8.json \
        --out-dir outputs/rngd_bench --prefix rngd-tp8
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from bench.core.plots import write_summary
from planner.predictor.calibration import (
    fit_from_summaries,
    parse_validation_summary,
    save_calibration,
)

NS_PER_MS = 1_000_000


def read_sim(path: Path) -> dict[str, list[float]]:
    """Per-request TTFT / TPOT / latency in milliseconds from the sim CSV."""
    def ms(row: dict, name: str) -> float | None:
        raw = (row.get(name) or "").strip()
        return float(raw) / NS_PER_MS if raw else None

    ttft, tpot, latency = [], [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            t, p, lat = ms(row, "TTFT"), ms(row, "TPOT"), ms(row, "latency")
            if t is not None:
                ttft.append(t)
            if p is not None and p > 0:
                tpot.append(p)
            if lat is not None:
                latency.append(lat)
    return {"ttft": ttft, "tpot": tpot, "latency": latency}


def read_real(path: Path) -> tuple[dict[str, list[float]], dict]:
    report = json.loads(path.read_text())
    ttft, tpot, latency = [], [], []
    for row in report["per_request"]:
        if row.get("error") or row.get("ttft_ns") is None:
            continue
        ttft.append(row["ttft_ns"] / NS_PER_MS)
        if row.get("tpot_ns"):
            tpot.append(row["tpot_ns"] / NS_PER_MS)
        latency.append(row["latency_ns"] / NS_PER_MS)
    return {"ttft": ttft, "tpot": tpot, "latency": latency}, report


def describe(label: str, values: list[float]) -> str:
    if not values:
        return f"{label:<22} (none)"
    ordered = sorted(values)

    def pct(q: float) -> float:
        # Same nearest-rank convention bench/core/plots uses for its table.
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[index]

    return (f"{label:<22} n={len(values):<4d} mean={statistics.fmean(values):10.1f} "
            f"p50={pct(0.5):10.1f} p90={pct(0.9):10.1f} p99={pct(0.99):10.1f}")


def _resolve_bucket(args) -> str:
    """The calibration entry's key, canonical or nothing (§2.4.1).

    A free string is not rejected outright - `split_bucket` files it under
    `label` and leaves the canonical key empty - but it is never usable as a
    lookup key, so this warns rather than letting the caller believe otherwise.
    """
    from planner.envelope import is_canonical_bucket, workload_bucket
    from planner.spec import load_service_spec

    if args.service and args.bucket:
        raise SystemExit("give at most one of --bucket and --service")
    if args.service:
        return workload_bucket(load_service_spec(args.service))
    if not args.bucket:
        raise SystemExit(
            "give --bucket (a canonical key) or --service (a spec to compute one from)"
        )
    if not is_canonical_bucket(args.bucket):
        print(
            f"WARNING: --bucket {args.bucket!r} is not a canonical key, so it is stored "
            f"as a LABEL and this calibration will be invisible to --accuracy-domain "
            f"lookup (uncertainty work order §2.4.1). Pass --service to place it.",
            file=sys.stderr,
        )
    return args.bucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-csv", required=True, type=Path)
    parser.add_argument("--real-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="rngd")
    parser.add_argument("--hardware", default="RNGD")
    parser.add_argument(
        "--bucket", default=None,
        help="Canonical bucket key (in_*-out_*-rps_*) - §2.4.1 makes it the only "
             "lookup key. Or give --service to compute it from a spec. Free-form "
             "names go in --label, which is never used for lookup.")
    parser.add_argument(
        "--service", default=None,
        help="Service spec whose workload_bucket() becomes the canonical key.")
    parser.add_argument(
        "--label", default="",
        help="Free-form name for this run. Never a lookup key.")
    parser.add_argument("--calibration-out", type=Path, default=None,
                        help="write profiles/calibration/<hw>.yaml here")
    args = parser.parse_args()

    sim = read_sim(args.sim_csv)
    real, report = read_real(args.real_json)

    print("=== coverage ===")
    print(f"sim  requests: {len(sim['ttft'])}")
    print(f"real requests: {len(real['ttft'])} ok, {report.get('failed')} failed")
    print(f"real generated chunks {report.get('generated_chunks_total')} vs "
          f"requested {report.get('requested_output_toks_total')} output tokens")
    if len(sim["ttft"]) != len(real["ttft"]):
        print("WARNING: request counts differ; the summary compares distributions, "
              "not paired requests, so an unequal set biases every statistic")

    print("\n=== distributions (ms) ===")
    for metric in ("ttft", "tpot", "latency"):
        print(describe(f"sim  {metric}", sim[metric]))
        print(describe(f"real {metric}", real[metric]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_summary(
        args.out_dir, args.prefix,
        bench_ttft=real["ttft"], sim_ttft=sim["ttft"],
        bench_tpot=real["tpot"], sim_tpot=sim["tpot"],
        bench_latency=real["latency"], sim_latency=sim["latency"],
    )
    print(f"\nwrote {summary_path}")
    print(summary_path.read_text())

    pairs = parse_validation_summary(summary_path.read_text())
    if not pairs.ttft or not pairs.tpot:
        raise SystemExit("summary produced no TTFT/TPOT rows to fit")
    bucket = _resolve_bucket(args)
    model = fit_from_summaries([(summary_path, args.hardware, bucket)])
    if args.label:
        entry = model.hardware[args.hardware].errors.get(bucket)
        if entry is not None:
            entry.label = args.label
    calibration = model.hardware[args.hardware]
    print("=== fitted real = alpha * sim + beta ===")
    for metric in ("ttft", "tpot"):
        fit = getattr(calibration, metric)
        print(f"  {metric}: alpha={fit.alpha:.6f} beta={fit.beta:.6f} "
              f"n={fit.sample_count}")
    errors = calibration.errors.get(args.bucket)
    if errors:
        for metric in ("ttft", "tpot"):
            stats = getattr(errors, metric)
            print(f"  {metric} error: mean={stats.mean_error:.4f} "
                  f"p95_abs={stats.p95_abs_error:.4f} worst={stats.worst_error:.4f}")

    if args.calibration_out:
        save_calibration(model, args.calibration_out)
        print(f"wrote {args.calibration_out}")


if __name__ == "__main__":
    main()
