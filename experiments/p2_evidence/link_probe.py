#!/usr/bin/env python
"""Measure what the TP=4 all-reduce actually gets, and where it loses it.

`WORK_ORDER_p2_regular_spec_evidence.md` follow-up to V4. V4 found that setting
`pcie-a40a-02` to about an eighth of its `vendor_spec` 64.0 gbps reproduces all
three of V3's measured metrics at once, and that the link is the one uncertain
input the registry flags for that island. This measures it.

Three probes, run under `torchrun`:

* **all_reduce across a group** at the message sizes a TP group actually sends.
  For Llama-3.1-8B, hidden 4096 in bf16, one all-reduce carries 8 KiB per token;
  a 128-way decode step is 1 MiB and a 2048-token prefill chunk is 16 MiB. The
  sweep spans 8 KiB to 64 MiB so both ends are bracketed.
* **the same all_reduce on two GPUs that share an NVLink pair (0,1) and on two
  that do not (0,2)**. That pair of numbers is the measurement the cluster spec
  guesses at: 112.5 for `nvlink-a40a-01` against 64.0 `vendor_spec` for
  `pcie-a40a-02`.
* **point-to-point device copies** between the same two pairs, which separates
  the link from NCCL's collective algorithm.

`busbw` is reported as well as `algbw` because a ring all-reduce moves
`2(n-1)/n` times the buffer over the wire; `busbw` is the figure comparable with
a link rate, and `algbw` is what the application sees.

Run through `experiments/p2_evidence/run_link_probe.sh`, which also runs each
case with `NCCL_P2P_DISABLE=1` so the peer-to-peer path can be priced.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

#: Llama-3.1-8B hidden size in bf16 -- one all-reduce element per hidden dim.
BYTES_PER_TOKEN = 4096 * 2
#: 8 KiB (one token) to 64 MiB, doubling. Brackets the decode step (1 MiB at 128
#: concurrent) and the prefill chunk (16 MiB at max_num_batched_tokens 2048).
SIZES_BYTES = [8 << 10 << i for i in range(14)]
WARMUP, ITERS = 5, 20


def _bench_allreduce(group, device, nbytes: int) -> float:
    """Median seconds per all_reduce over `ITERS`."""
    buf = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=device)
    for _ in range(WARMUP):
        dist.all_reduce(buf, group=group)
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(ITERS):
        dist.barrier(group=group)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        dist.all_reduce(buf, group=group)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def _bench_p2p(src: int, dst: int, nbytes: int) -> tuple[float, float]:
    """Median seconds for a device-to-device copy, and the empty-copy latency.

    Run on rank 0 only: a direct `Tensor.copy_` between two devices is the link
    without NCCL's algorithm on top, which is what separates "the link is slow"
    from "the collective is inefficient".
    """
    a = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=f"cuda:{src}")
    b = torch.empty_like(a, device=f"cuda:{dst}")
    tiny_a = torch.ones(1, dtype=torch.bfloat16, device=f"cuda:{src}")
    tiny_b = torch.empty_like(tiny_a, device=f"cuda:{dst}")
    for _ in range(WARMUP):
        b.copy_(a)
    torch.cuda.synchronize()
    big, small = [], []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        big.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        tiny_b.copy_(tiny_a)
        torch.cuda.synchronize()
        small.append(time.perf_counter() - t0)
    big.sort()
    small.sort()
    return big[len(big) // 2], small[len(small) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", default="0,1,2,3",
                    help="global ranks forming the collective group")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--p2p-pairs", default="0-1,0-2",
                    help="rank 0 also times direct copies between these device pairs")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    members = [int(r) for r in args.ranks.split(",")]
    group = dist.new_group(members)
    n = len(members)

    rows = []
    if rank in members:
        for nbytes in SIZES_BYTES:
            sec = _bench_allreduce(group, device, nbytes)
            algbw = nbytes / sec / 1e9                    # GB/s the caller sees
            busbw = algbw * 2 * (n - 1) / n               # GB/s over the wire
            if rank == members[0]:
                rows.append({"bytes": nbytes, "tokens": nbytes // BYTES_PER_TOKEN,
                             "seconds": sec, "algbw_gbps": algbw,
                             "busbw_gbps": busbw})
                print(f"  {nbytes/1024:10.0f} KiB  {sec*1e6:9.1f} us  "
                      f"algbw {algbw:7.2f} GB/s  busbw {busbw:7.2f} GB/s", flush=True)

    p2p = {}
    if rank == 0:
        for pair in args.p2p_pairs.split(","):
            s, d = (int(x) for x in pair.split("-"))
            if s >= torch.cuda.device_count() or d >= torch.cuda.device_count():
                continue
            big_s, small_s = _bench_p2p(s, d, 64 << 20)
            p2p[pair] = {"copy_64MiB_s": big_s,
                         "bandwidth_gbps": (64 << 20) / big_s / 1e9,
                         "latency_2B_us": small_s * 1e6,
                         "peer_access": torch.cuda.can_device_access_peer(s, d)}
            print(f"  p2p {pair}: {(64<<20)/big_s/1e9:7.2f} GB/s  "
                  f"lat {small_s*1e6:7.1f} us  peer_access="
                  f"{torch.cuda.can_device_access_peer(s, d)}", flush=True)

    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "label": args.label, "ranks": members, "world_size": world,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
            "torch": torch.__version__,
            "allreduce": rows, "p2p": p2p,
        }, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
