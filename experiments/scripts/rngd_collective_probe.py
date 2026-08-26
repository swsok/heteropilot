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
need no reduction, only a concatenation that is free. So:

    A. Linear(in/N, out) on ONE PE            -> per-rank compute, no reduction
    B. Linear(in,   out) on a FUSED N-PE device -> same compute, plus the reduction

and `B - A` is the reduction, measured. Run for N in {1,2,4,8}: N=1 is the
control (B and A are the same graph, so the difference must be ~0, which is how
this script proves it is measuring what it claims).

`furiosa.torch` exposes no all_reduce API — collectives live inside the compiled
EDF — so fusion is the only way to get the compiler to emit one. `set_fusion()`
is process-global and one-shot, hence one subprocess per N.

Device spans come from the same `RNGDProfiler` path the layerwise profiler uses,
so `time_us` here is the union of device spans, comparable to the perf bundle.

Usage::

    PYTHONPATH=$PWD python3 experiments/scripts/rngd_collective_probe.py \
        --out outputs/rngd_profile/collective_probe.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
nd.set_fusion(fusion)                      # process-global, one-shot

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
    union, by_name = spans(prof, f"/tmp/collprobe_{fusion}_{in_d}.json")
    del compiled, x
    module.to("cpu")
    torch._dynamo.reset()
    return union / reps, {k: v / reps for k, v in by_name.items()}

sharded_in = max(1, in_dim // fusion)
try:
    a_us, a_spans = measure(sharded_in, out_dim)     # per-rank compute, one PE's worth
    b_us, b_spans = measure(in_dim, out_dim)         # full layer on the fused device
    print("RESULT " + json.dumps({
        "fusion": fusion, "device": device, "tokens": tokens,
        "in_dim": in_dim, "out_dim": out_dim, "sharded_in": sharded_in,
        "per_rank_us": a_us, "fused_full_us": b_us,
        "collective_us": b_us - a_us,
        "per_rank_spans": a_spans, "fused_spans": b_spans,
    }), flush=True)
except Exception as exc:
    print("ERROR " + json.dumps({
        "fusion": fusion, "in_dim": in_dim,
        "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
    }), flush=True)
'''


def run(fusion: int, in_dim: int, out_dim: int, tokens: int, reps: int,
        timeout: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-u", "-c", WORKER,
         str(fusion), str(in_dim), str(out_dim), str(tokens), str(reps)],
        capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(("RESULT ", "ERROR ")):
            return json.loads(line.split(" ", 1)[1])
    return {"fusion": fusion, "in_dim": in_dim,
            "error": "no RESULT line; worker output tail: "
                     + " | ".join(proc.stdout.splitlines()[-3:] or ["(empty)"])}


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
    args = parser.parse_args()

    fusions = [int(v) for v in args.fusions.split(",") if v]
    layers = [name for name in args.layers.split(",") if name in ROW_PARALLEL]

    rows = []
    print(f"{'layer':<12} {'fusion':>6} {'per-rank us':>12} {'fused us':>10} "
          f"{'collective us':>14}")
    for layer in layers:
        in_dim, out_dim = ROW_PARALLEL[layer]
        for fusion in fusions:
            row = run(fusion, in_dim, out_dim, args.tokens, args.reps, args.timeout)
            row["layer"] = layer
            rows.append(row)
            if "error" in row:
                print(f"{layer:<12} {fusion:>6} {'FAILED':>12}  {row['error'][:70]}")
            else:
                print(f"{layer:<12} {fusion:>6} {row['per_rank_us']:>12.2f} "
                      f"{row['fused_full_us']:>10.2f} {row['collective_us']:>14.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tokens": args.tokens, "reps": args.reps,
        "method": (
            "For each fusion N: (A) Linear(in/N, out) on one PE = per-rank compute "
            "without reduction; (B) Linear(in, out) on the fused N-PE device = same "
            "compute plus the reduction the compiler must emit for a row-parallel "
            "layer. collective_us = B - A. N=1 is the control and must give ~0. "
            "time_us is the union of device spans, as in the perf bundle."
        ),
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    control = [r for r in rows if r.get("fusion") == 1 and "error" not in r]
    if control:
        worst = max(abs(r["collective_us"]) for r in control)
        print(f"control (fusion=1) |collective| = {worst:.2f} us — this is the "
              f"noise floor; readings below it mean nothing")


if __name__ == "__main__":
    main()
