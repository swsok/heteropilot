"""Measure an envelope point OPEN-LOOP, the way the existing A40 point was made.

`docs/HANDOVER.md` §2.2. The A40 accuracy domain has one point, at served
concurrency 170.56, and it came from `python -m bench run` -- an arrival-trace
replay, not a closed-loop client. This driver adds points to that domain by
running the same harness at a lower offered rate, so the second point sits on the
same axis as the first rather than beside it.

**Why not just point `measure_envelope.py --backend cuda` at the A40.** That one
is closed-loop: a fixed number of requests in flight, which is how the RNGD
envelope was measured and is a different occupancy regime from a queue fed at a
fixed rate. Both are legitimate and this repository now has both, but mixing them
inside ONE accuracy domain would put two definitions of "served concurrency 8" on
one interpolation axis. Two consequences follow and neither is optional:

  * the metrics come from `bench.core.validate`, upstream's own derivation, not a
    reimplementation. `_bench_latencies` picks `queued_ts` as the arrival anchor
    when vLLM records `arrival_time` in a different clock domain, and getting
    that wrong is worth tens of percent. Running it over the committed
    `outputs/phase0_bench/A40/vllm/requests.jsonl` reproduces **170.5619** against
    the committed 170.56, which is the check that this driver is on the same axis;
  * **TTFT is comparable here**, unlike the closed-loop path. Both sides replay
    the same arrival process, so D19 -- a burst measured against a spread arrival
    process, with the whole difference landing in TTFT -- does not apply. That is
    the reason to prefer this route for a *calibration* point.

The A5 rules carry over in substance, with one that cannot:

  (a) concurrency is SERVED, by Little's law, never the value asked for;
  (c) power is a sustained mean over the bench window with the utilisation from
      the same samples;
  (e) every point runs the same workload, re-spaced.

  (b) is a closed-loop rule -- there is no pool to be 4x anything. Its purpose,
      "did the point sustain the load it claims?", survives as the QUEUE DELAY
      SLOPE: d(scheduled_ts - queued_ts)/d(arrival), in seconds of waiting per
      second of elapsed arrival time. A queue that is keeping up has slope ~0; one
      whose offered rate exceeds capacity grows without bound. Measured, the two
      cases do not overlap: the committed 170.56 run reads **+4.15 s/s** and a
      20-request run at 0.5 rps reads **-0.0000**.

      The obvious metric -- completed requests per second against the offered rate
      -- was tried first and is WRONG, in the direction that would have quietly
      poisoned the domain. It divides by a wall that includes the drain tail, so a
      perfectly healthy 20-request point at 0.5 rps reports 0.64 and looks
      saturated. Short runs are exactly where a low-load calibration point lives.
      `completed_rps` is still recorded, because it is worth seeing; it is no
      longer what decides.

Usage:

    experiments/scripts/measure_envelope_openloop.py \\
        --model NousResearch/Meta-Llama-3.1-8B \\
        --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \\
        --rps 0.2,0.6 --card 0 --repeats 2 \\
        --out outputs/a40_envelope_openloop
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench.core.validate import _bench_arrival_ts, _bench_latencies  # noqa: E402
from planner.util.percentile import summary  # noqa: E402

#: Reuse the sampler parser and the power/utilisation window analysis rather than
#: writing a second one. `measure_envelope.py` is a script, not a module, so it is
#: loaded by path; the alternative is two copies of A5(c) drifting apart.
_spec = importlib.util.spec_from_file_location(
    "measure_envelope_lib", REPO_ROOT / "experiments/scripts/measure_envelope.py")
assert _spec is not None and _spec.loader is not None
_me = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _me
_spec.loader.exec_module(_me)

read_sampler_csv = _me.read_sampler_csv
window_stats = _me.window_stats
gpu_uuid = _me.gpu_uuid
Window = _me.Window

NS_PER_S = 1e9
#: Empty-card baseline taken before the bench process starts. Deliberately NOT
#: called "idle": the RNGD envelope's idle window is a LOADED but unoccupied card,
#: and `python -m bench run` loads and unloads the model inside its own process,
#: so no loaded-but-idle window exists to take. Recorded under its own name so the
#: two are never averaged together.
EMPTY_CARD_WINDOW_S = 60.0
#: A5(b)'s surviving purpose: did the point sustain the rate it offered? Slope of
#: queue delay against arrival time, in s/s. Zero means the queue keeps up. The
#: threshold sits two orders of magnitude below the saturated case measured on
#: this node (+4.15) and above the healthy one (-0.0000), so it is not a close call.
QUEUE_GROWTH_SLOPE = 0.05


def _rel(p: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    `--out outputs/...` arrives relative, and `Path.relative_to` RAISES when one
    side is relative and the other absolute -- so recording provenance crashed
    the run AFTER the measurement had been taken. Both forms name the same file.
    """
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def respace_trace(out: Path, dataset: Path, rps: float, n: int) -> Path:
    """The dataset's first `n` requests, re-spaced at a constant `rps`.

    Constant spacing rather than Poisson, for the same reason
    `lowload_sim_error.py` does it: the envelope point is a steady-state average,
    and a deterministic arrival process reaches the same mean occupancy without
    adding a second source of variance to explain. `tests/test_envelope_openloop`
    pins the two implementations to the same output.
    """
    rows = []
    with dataset.open() as fh:
        for i, line in enumerate(fh):
            if n and i >= n:
                break
            rec = json.loads(line)
            rec["arrival_time_ns"] = int(i / rps * NS_PER_S)
            rows.append(rec)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def queue_delay_slope(reqs: list[dict]) -> float | None:
    """Seconds of queue delay gained per second of elapsed arrival time.

    `queued_ts` is when the replayer handed the request to the engine -- its
    arrival -- and `scheduled_ts` is when the engine started it, so the gap is
    time spent waiting for capacity. Its trend is the definition of saturation:
    an offered rate below capacity holds the gap flat, one above it accumulates.

    Least squares rather than first-vs-last, so a single slow request cannot set
    the verdict.
    """
    pairs = sorted(
        (r["queued_ts"], r["scheduled_ts"] - r["queued_ts"])
        for r in reqs
        if r.get("queued_ts") is not None and r.get("scheduled_ts") is not None
    )
    if len(pairs) < 2:
        return None
    t0 = pairs[0][0]
    xs = [t - t0 for t, _ in pairs]
    ys = [d for _, d in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom


def _epoch(iso: str) -> float:
    """`meta.json` timestamps are ISO-8601 UTC with a trailing Z."""
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def summarise_openloop_point(meta: dict, reqs: list[dict], rows: list,
                             offered_rps: float,
                             device_sn: str | None = None,
                             empty_window: Window | None = None) -> dict:
    """Everything one open-loop point claims, from measurements only.

    The bench window comes from `meta.json`'s own `started_at`/`finished_at`
    rather than from wall-clock around the subprocess: `python -m bench run`
    loads the model in the same process, and a window that included the load
    would average a ~2-minute compute-and-copy phase into a serving figure.
    """
    ttft_ms, tpot_ms, lat_ms = _bench_latencies(reqs)
    anchors = [a for a, _ in (_bench_arrival_ts(r) for r in reqs) if a is not None]
    lasts = [r["last_token_ts"] for r in reqs if r.get("last_token_ts") is not None]

    # Little's law over the SAME anchor the latencies use, or the two disagree by
    # whatever the clock-domain fallback corrected for.
    wall_s = (max(lasts) - min(anchors)) if (lasts and anchors) else 0.0
    served = (sum(lat_ms) / 1000.0 / wall_s) if wall_s else 0.0
    completed_rps = (len(lat_ms) / wall_s) if wall_s else 0.0
    slope = queue_delay_slope(reqs)
    saturated = slope is not None and slope > QUEUE_GROWTH_SLOPE

    notes: list[str] = []
    if saturated:
        notes.append(
            f"queue delay grew at {slope:+.3f} s per second of arrivals "
            f"(> {QUEUE_GROWTH_SLOPE}) against an offered {offered_rps:.3f} rps: "
            f"the queue grew, so this point measures saturation and its "
            f"concurrency is a backlog depth, not the offered load"
        )
    if slope is None:
        notes.append("no usable queued/scheduled timestamps: saturation unknown")
    missing = len(reqs) - len(lat_ms)
    if missing:
        notes.append(f"{missing} request(s) had no usable timestamps and are excluded")

    out_toks = sum(int(r.get("output_toks") or 0) for r in reqs)
    bench_window = Window(_epoch(meta["started_at"]), _epoch(meta["finished_at"]))

    return {
        "offered_rps": offered_rps,
        "served_concurrency": served,
        # Recorded because it is worth seeing, NOT because it decides: over a
        # wall that includes the drain tail it reads low on any short run.
        "completed_rps": completed_rps,
        "queue_delay_slope_s_per_s": slope,
        # The open-loop analogue of `pool_binding`: reported, never dropped.
        "saturated": saturated,
        "requests_ok": len(lat_ms),
        "requests_total": len(reqs),
        "wall_s": wall_s,
        "throughput_tok_s": (out_toks / wall_s) if wall_s else None,
        "ttft_ms": summary(ttft_ms) if ttft_ms else None,
        "tpot_ms": summary(tpot_ms) if tpot_ms else None,
        "e2e_ms": summary(lat_ms) if lat_ms else None,
        "bench_window": window_stats(rows, bench_window, device_sn),
        "empty_card_window": (window_stats(rows, empty_window, device_sn)
                              if empty_window else None),
        "notes": notes,
    }


def run_point(args, rps: float, out_dir: Path, repeat: int = 0) -> dict | None:
    tag = f"r{rps:g}".replace(".", "p") + (f"_rep{repeat}" if repeat else "")
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler_csv = out_dir / f"power_{tag}.csv"
    log_path = out_dir / f"bench_{tag}.log"

    trace = respace_trace(out_dir / f"trace_{tag}.jsonl", args.dataset,
                          rps, args.num_reqs)

    sampler = subprocess.Popen(
        [str(REPO_ROOT / "experiments/scripts/power_sampler_nvidia.sh"),
         "--out", str(sampler_csv), "--devices", str(args.card)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        # Empty-card baseline first: the only unoccupied window this harness can
        # honestly offer, because the model lives and dies with the bench process.
        print(f"{tag}: empty-card baseline {EMPTY_CARD_WINDOW_S:.0f} s",
              file=sys.stderr)
        empty_start = time.time()
        time.sleep(EMPTY_CARD_WINDOW_S)
        empty_window = Window(empty_start, time.time())

        print(f"{tag}: offering {rps:g} rps over {args.num_reqs} requests "
              f"(~{args.num_reqs / rps / 60:.0f} min)", file=sys.stderr)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.card)}
        with log_path.open("w") as log:
            proc = subprocess.run(
                [args.bench_python, "-u", "-m", "bench", "run",
                 "--model", args.model,
                 "--dataset", str(trace),
                 "--output-dir", str(run_dir),
                 "--tensor-parallel-size", str(args.tp),
                 "--max-num-seqs", str(args.max_num_seqs),
                 "--max-num-batched-tokens", str(args.max_num_batched_tokens),
                 "--dtype", args.dtype, "--kv-cache-dtype", args.kv_cache_dtype,
                 "--seed", str(args.seed), "--num-reqs", str(args.num_reqs),
                 "--tick-seconds", "1.0"],
                cwd=REPO_ROOT, env=env, stdout=log, stderr=log,
                check=False, timeout=args.timeout,
            )
        if proc.returncode != 0:
            print(f"{tag}: bench run exited {proc.returncode}; see {log_path}",
                  file=sys.stderr)
            return None
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=15)
        except subprocess.TimeoutExpired:
            sampler.kill()

    meta_path, reqs_path = run_dir / "meta.json", run_dir / "requests.jsonl"
    if not (meta_path.exists() and reqs_path.exists()):
        print(f"{tag}: the bench wrote no report", file=sys.stderr)
        return None
    meta = json.loads(meta_path.read_text())
    reqs = [json.loads(line) for line in reqs_path.read_text().splitlines() if line]
    rows = read_sampler_csv(sampler_csv)
    point = summarise_openloop_point(meta, reqs, rows, rps,
                                     device_sn=args.device_sn,
                                     empty_window=empty_window)
    point["artifacts"] = {"run_dir": _rel(run_dir),
                          "sampler_csv": _rel(sampler_csv),
                          "trace": _rel(trace)}
    return point


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="HF id passed to vLLM. The A40 point at 170.56 used the "
                         "ungated NousResearch mirror; use the same one or the "
                         "comparison is between two weight sets.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rps", default="0.2,0.6",
                    help="offered arrival rates. Served concurrency is MEASURED, "
                         "never requested -- these are the loads, not the answer.")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num-reqs", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=2,
                    help="independent processes per point (A5 asks for 2)")
    ap.add_argument("--device-sn", default=None,
                    help="GPU UUID for the power analysis; resolved from --card "
                         "when omitted")
    ap.add_argument("--bench-python", default=".venv-vllm/bin/python",
                    help="the CUDA venv. `.venv` has no vLLM and must not get one.")
    # Engine knobs default to what produced the 170.56 point
    # (outputs/phase0_bench/A40/vllm/meta.json). Changing one changes what the
    # accuracy domain's two points have in common.
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=14400.0)
    args = ap.parse_args()

    if args.device_sn is None:
        args.device_sn = gpu_uuid(args.card)
        if args.device_sn is None:
            print("could not resolve the GPU UUID; the power analysis will average "
                  "every device the sampler recorded", file=sys.stderr)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = [float(v) for v in args.rps.split(",") if v]

    results = []
    for rps in rates:
        for rep in range(args.repeats):
            point = run_point(args, rps, out_dir, repeat=rep)
            if point is None:
                print(f"{rps:g} rps rep{rep}: no result", file=sys.stderr)
                continue
            point["repeat"] = rep
            results.append(point)
            tag = f"r{rps:g}".replace(".", "p") + (f"_rep{rep}" if rep else "")
            (out_dir / f"point_{tag}.json").write_text(
                json.dumps(point, indent=2) + "\n")
            pw = point["bench_window"].get("power_w")
            pstr = f" {pw['mean']:.1f} W" if pw else " (no power)"
            flag = "  SATURATED" if point["saturated"] else ""
            slope = point["queue_delay_slope_s_per_s"]
            sstr = f"{slope:+.3f}" if slope is not None else "n/a"
            print(f"{rps:g} rps rep{rep}: served {point['served_concurrency']:.2f} "
                  f"(queue slope {sstr} s/s){pstr}{flag}", file=sys.stderr)

    (out_dir / "envelope_openloop.json").write_text(json.dumps({
        "run": {
            "backend": "cuda",
            "harness": "python -m bench run (arrival-trace replay)",
            "closed_loop": False,
            "model": args.model,
            "dataset": str(args.dataset),
            "card": args.card,
            "tp": args.tp,
            "device_sn": args.device_sn,
            "num_reqs": args.num_reqs,
            "repeats": args.repeats,
            "engine": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "dtype": args.dtype, "kv_cache_dtype": args.kv_cache_dtype,
                "seed": args.seed,
            },
            "metrics_from": "bench.core.validate._bench_latencies",
        },
        "points": results,
    }, indent=2) + "\n")
    print(f"wrote {out_dir / 'envelope_openloop.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
