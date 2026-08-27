"""Measure the RNGD on-package all-reduce directly, instead of inferring it.

`experiments/results/rngd_card_vs_pe_model.md` bounded the intra-card collective
at <= 202 us per all-reduce by attributing a residual to it. That number is doing
real work — it explains ~45% of measured TP=8 decode TPOT, and it is the only
handle on the `ONPACKAGE` fabric that deviations D3/D16 leave as placeholder — so
it deserves to be measured rather than derived.

Method: the all-reduce in tensor parallelism follows the ROW-PARALLEL layers
(`o_proj`, `down_proj`). Each rank holds a slice of the input dimension and
produces a *partial sum* over the full output, so the partial sums must be
reduced across the group. Column-parallel layers (`qkv_proj`, `gate_up_proj`)
need no reduction, only a concatenation that is free. So for each group size N:

    SHARD. set_fusion(1); Linear(in/N, out) on ONE PE
           -> exactly one rank's compute, and no reduction, because there is no
              group to reduce across
    FULL.  set_fusion(N); Linear(in,  out) on the FUSED N-PE device
           -> the same per-rank compute done N-ways in parallel, PLUS the
              reduction the compiler must emit for a row-parallel layer

and `FULL - SHARD` is the reduction, measured.

BUG FIXED 2026-08-26, recorded because the first version's numbers looked
plausible and were not. It ran BOTH measurements on the fused N-PE device, so the
"per-rank" leg was a 1/N-sized layer that the compiler *also* sharded and *also*
reduced. The reduction therefore largely cancelled in the subtraction, and what
was left was mostly the extra weight traffic of an N-times-larger layer. The
fusion=1 control could not catch it: at N=1 the two legs are the same graph
whichever device they run on, so it returned ~0 for the wrong reason.

TWO validity checks, and both must pass before a reading means anything:
  * CONTROL, N=1: SHARD and FULL are the same graph on one PE, so the difference
    must be ~0. This bounds the noise floor.
  * SCALING: N x SHARD(N) must be ~= SHARD(1). This is the check the old design
    lacked - it verifies the SHARD leg really is one rank's worth of work, which
    is the only thing that makes the subtraction meaningful.

`furiosa.torch` exposes no all_reduce API — collectives live inside the compiled
EDF — so fusion is the only way to get the compiler to emit one. `set_fusion()`
is process-global and one-shot, hence one subprocess per (N, leg).

Device spans come from the same `RNGDProfiler` path the layerwise profiler uses,
so `time_us` here is the union of device spans, comparable to the perf bundle.

Usage::

    PYTHONPATH=$PWD /usr/bin/python3 experiments/scripts/rngd_collective_probe.py \
        --out outputs/rngd_profile/collective_probe.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Llama-3.1-8B bf16 row-parallel shapes, the two layers an all-reduce follows.
ROW_PARALLEL = {
    "down_proj": (14336, 4096),   # (in, out): in is sharded across the TP group
    "o_proj": (4096, 4096),
}

WORKER = r'''
import json, sys, collections, time
import torch
import torch.nn as nn

# furiosa.torch must be imported after torch (see experiments/scripts/profile_rngd.py)
from furiosa.torch import native_device as nd

fusion, in_dim, out_dim, tokens, reps = (int(v) for v in sys.argv[1:6])
leg = sys.argv[6]                          # "shard" (one PE) or "full" (N fused)

# THE WHOLE POINT: the shard leg must run on ONE PE so it carries no reduction,
# and the full leg on the fused N-PE device so it carries one. set_fusion() is
# process-global and one-shot, which is why each leg is its own subprocess.
nd.set_fusion(1 if leg == "shard" else fusion)

import furiosa.torch as ft
from furiosa.torch import config as fcfg
from furiosa.torch.profiler import RNGDProfiler

_, _, devices = nd.get_device_configuration()
device = devices[0]


def spans(prof, path):
    prof.export_chrome_trace(path)
    raw = json.loads(open(path).read())
    events = raw if isinstance(raw, list) else raw.get("traceEvents", [])
    by_name = collections.Counter()
    intervals = []
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        name = e.get("name", "")
        if name.startswith("Renegade::") or name.startswith("DMA") or name == "Task":
            by_name[name] += e.get("dur", 0)
            intervals.append((e["ts"], e["ts"] + e.get("dur", 0)))
    total = 0.0
    cur_s = cur_e = None
    for s, t in sorted(intervals):
        if cur_e is None:
            cur_s, cur_e = s, t
        elif s <= cur_e:
            cur_e = max(cur_e, t)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, t
    if cur_e is not None:
        total += cur_e - cur_s
    return total, dict(by_name)


def measure(in_d, out_d):
    module = nn.Linear(in_d, out_d, bias=False, dtype=torch.bfloat16).to(device)
    x = torch.randn(tokens, in_d, dtype=torch.bfloat16).to(device)
    compiled = torch.compile(module, backend=ft.backend)
    with torch.no_grad():
        compiled(x)                        # compile + warm
        prof = RNGDProfiler()
        with fcfg.profiler_context(prof):
            with prof:
                for _ in range(reps):
                    compiled(x)
    union, by_name = spans(prof, f"/tmp/collprobe_{leg}_{fusion}_{in_d}.json")
    del compiled, x
    module.to("cpu")
    torch._dynamo.reset()
    return union / reps, {k: v / reps for k, v in by_name.items()}

sharded_in = max(1, in_dim // fusion)
want_in = sharded_in if leg == "shard" else in_dim
try:
    us, spans = measure(want_in, out_dim)
    print("RESULT " + json.dumps({
        "fusion": fusion, "leg": leg, "device": device, "devices": len(devices),
        "tokens": tokens, "in_dim": in_dim, "out_dim": out_dim,
        "measured_in": want_in, "time_us": us, "spans": spans,
    }), flush=True)
except Exception as exc:
    print("ERROR " + json.dumps({
        "fusion": fusion, "leg": leg, "in_dim": in_dim,
        "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
    }), flush=True)
'''


def run(fusion: int, in_dim: int, out_dim: int, tokens: int, reps: int,
        timeout: float, python: str, leg: str,
        log_dir: Path | None = None) -> dict:
    """Run one worker subprocess and return its RESULT/ERROR payload.

    The worker needs the VENDOR interpreter: furiosa.torch lives in the user
    site-packages with torch 2.10, and the planner venv has neither, so
    `sys.executable` silently fails here (this is the same trap as
    `--bench-python` in rebuild_rngd_bundle_from_edf.py).

    STDERR IS KEPT. An earlier version discarded it and reported only "no RESULT
    line", which hid every real failure - a crash in the worker looked identical
    to a worker that ran and printed nothing.
    """
    proc = subprocess.run(
        [python, "-u", "-c", WORKER,
         str(fusion), str(in_dim), str(out_dim), str(tokens), str(reps), leg],
        capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
    )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"worker_{leg}_f{fusion}_in{in_dim}.log").write_text(
            f"$ {python} -c <WORKER> {fusion} {in_dim} {out_dim} {tokens} "
            f"{reps} {leg}\n"
            f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}\n"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(("RESULT ", "ERROR ")):
            return json.loads(line.split(" ", 1)[1])
    tail = [ln for ln in proc.stderr.splitlines() if ln.strip()][-4:]
    return {"fusion": fusion, "leg": leg, "in_dim": in_dim,
            "error": f"no RESULT line (exit {proc.returncode}); stderr tail: "
                     + " | ".join(tail or ["(stderr empty)"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusions", default="1,2,4,8")
    parser.add_argument("--layers", default="down_proj,o_proj")
    parser.add_argument("--tokens", type=int, default=1,
                        help="1 = decode shape, where the collective matters most")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/rngd_profile/collective_probe.json"))
    parser.add_argument("--python", default="/usr/bin/python3",
                        help="interpreter that can import furiosa.torch; the "
                             "planner venv cannot")
    parser.add_argument("--log-dir", type=Path,
                        default=Path("outputs/rngd_profile/collective_probe_logs"),
                        help="per-worker stdout/stderr, so a failure is diagnosable")
    args = parser.parse_args()

    fusions = [int(v) for v in args.fusions.split(",") if v]
    layers = [name for name in args.layers.split(",") if name in ROW_PARALLEL]

    legs: list[dict] = []
    rows: list[dict] = []
    print(f"{'layer':<12} {'N':>3} {'shard us':>10} {'full us':>10} "
          f"{'collective us':>14} {'Nxshard/shard1':>15}")
    for layer in layers:
        in_dim, out_dim = ROW_PARALLEL[layer]
        shard1 = None
        for fusion in fusions:
            pair = {}
            for leg in ("shard", "full"):
                res = run(fusion, in_dim, out_dim, args.tokens, args.reps,
                          args.timeout, args.python, leg, args.log_dir)
                res["layer"] = layer
                legs.append(res)
                pair[leg] = res
            if fusion == 1 and "error" not in pair["shard"]:
                shard1 = pair["shard"]["time_us"]
            if any("error" in r for r in pair.values()):
                bad = next(r for r in pair.values() if "error" in r)
                print(f"{layer:<12} {fusion:>3} {'FAILED':>10}  "
                      f"{bad['leg']}: {bad['error'][:60]}")
                rows.append({"layer": layer, "fusion": fusion,
                             "error": f"{bad['leg']}: {bad['error']}"})
                continue
            shard, full = pair["shard"]["time_us"], pair["full"]["time_us"]
            # Scaling check: N ranks' worth of work must add up to the whole
            # layer, or the shard leg is not one rank and the subtraction is
            # meaningless.
            ratio = (fusion * shard / shard1) if shard1 else float("nan")
            row = {"layer": layer, "fusion": fusion, "shard_us": shard,
                   "full_us": full, "collective_us": full - shard,
                   "scaling_ratio": ratio,
                   "shard_spans": pair["shard"]["spans"],
                   "full_spans": pair["full"]["spans"],
                   "devices_visible": pair["full"]["devices"]}
            rows.append(row)
            print(f"{layer:<12} {fusion:>3} {shard:>10.2f} {full:>10.2f} "
                  f"{full - shard:>14.2f} {ratio:>15.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tokens": args.tokens, "reps": args.reps,
        "method": (
            "For each group size N, two SEPARATE processes because set_fusion() is "
            "process-global: (SHARD) set_fusion(1), Linear(in/N, out) on ONE PE = "
            "exactly one rank's compute with no group to reduce across; (FULL) "
            "set_fusion(N), Linear(in, out) on the fused N-PE device = the same "
            "per-rank compute done N-ways in parallel PLUS the reduction the "
            "compiler must emit for a row-parallel layer. collective_us = FULL - "
            "SHARD. Validity: N=1 is the control and must give ~0 (noise floor), "
            "and scaling_ratio = N*SHARD(N)/SHARD(1) must be ~1 or the shard leg "
            "is not one rank's work and the subtraction means nothing. time_us is "
            "the union of device spans, as in the perf bundle."
        ),
        "rows": rows, "legs": legs,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    control = [r for r in rows if r.get("fusion") == 1 and "error" not in r]
    if control:
        worst = max(abs(r["collective_us"]) for r in control)
        print(f"control (N=1) |collective| = {worst:.2f} us — the noise floor; "
              f"readings below it mean nothing")
    bad_scale = [r for r in rows if "error" not in r and r["fusion"] > 1
                 and not (0.7 <= r["scaling_ratio"] <= 1.4)]
    for r in bad_scale:
        print(f"WARNING {r['layer']} N={r['fusion']}: scaling ratio "
              f"{r['scaling_ratio']:.3f} is outside [0.7, 1.4], so its shard leg "
              f"is not one rank's work — treat its collective as unusable")


if __name__ == "__main__":
    main()
