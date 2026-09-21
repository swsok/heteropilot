#!/usr/bin/env python
"""Price the cross-pair wire WITHOUT NCCL on top of it.

`WORK_ORDER_domain_scoping.md` S6(ii) asks for a figure from a vendor tool
beside the torch one. `nccl-tests` cannot supply it here -- the only build on
this host needs GLIBC 2.34 against Ubuntu 20.04's 2.31, and there is no `nvcc`
to rebuild it with -- and it would not have been much of a control anyway: it
calls the same NCCL 2.27.5 that `link_probe.py` calls through torch, so the two
would agree by construction rather than by corroboration.

The control that is actually independent is the one that does not go through
NCCL at all: a direct peer copy, which is `cudaMemcpyPeerAsync` under
`Tensor.copy_`. If the 8.8 GB/s the TP=4 all-reduce gets is the wire, a peer
copy over the same wire is bounded by it; if it is NCCL's ring on a four-rank
group, the copy is free of that and says so. `link_probe.py` already times a
64 MiB one-way copy as a side result; this widens it in the two directions that
matter for the question:

* **hundreds of MB rather than 64**, so the transfer is asymptotic and the
  fixed per-call cost cannot flatter it, and
* **round trip as well as one way**, because a ring all-reduce drives the link
  in both directions at once and a one-way number cannot be compared with it
  directly.

Both are reported as median plus spread over independent iterations, because
`docs/nodes/a40.md` records that a single parallel trial on this node is not
reproducible (two runs disagreed by 38 % on a 4-GPU host figure).

`peer_access` is reported per pair: a copy between devices that cannot reach
each other peer-to-peer is staged through host memory and is a measurement of
the host path, not of the wire.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

#: Hundreds of MB, per the control's whole point. 64 MiB is what link_probe.py
#: already reports and is kept as the overlap point between the two probes.
SIZES_MIB = [64, 256, 512]
WARMUP, ITERS = 5, 20


def _sync(*devs: int) -> None:
    for d in devs:
        torch.cuda.synchronize(d)


def _one_way(src: int, dst: int, nbytes: int) -> list[float]:
    """Seconds per src->dst copy, one sample per iteration."""
    a = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=f"cuda:{src}")
    b = torch.empty_like(a, device=f"cuda:{dst}")
    for _ in range(WARMUP):
        b.copy_(a)
    _sync(src, dst)
    out = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        b.copy_(a)
        _sync(src, dst)
        out.append(time.perf_counter() - t0)
    del a, b
    torch.cuda.empty_cache()
    return out


def _round_trip(src: int, dst: int, nbytes: int) -> list[float]:
    """Seconds per src->dst->src pair of copies.

    Issued back to back on the two devices' current streams and synchronised
    once at the end, so the two legs may overlap the way a collective's do. The
    reported bandwidth counts the bytes moved on BOTH legs.
    """
    a = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=f"cuda:{src}")
    b = torch.empty_like(a, device=f"cuda:{dst}")
    c = torch.empty_like(a, device=f"cuda:{src}")
    for _ in range(WARMUP):
        b.copy_(a)
        c.copy_(b)
    _sync(src, dst)
    out = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        b.copy_(a)
        c.copy_(b)
        _sync(src, dst)
        out.append(time.perf_counter() - t0)
    del a, b, c
    torch.cuda.empty_cache()
    return out


def _summarise(samples: list[float], nbytes: int, legs: int) -> dict:
    """Median bandwidth plus the spread, never a best-of-N.

    `docs/HANDOVER.md` §3 records that peak overstates a bulk copy by 25 % on
    one of this project's devices, so the headline is the median and min/max are
    carried beside it rather than instead of it.
    """
    samples = sorted(samples)
    med = statistics.median(samples)
    return {
        "median_s": med,
        "bandwidth_gbps": nbytes * legs / med / 1e9,
        "bandwidth_gbps_min": nbytes * legs / samples[-1] / 1e9,
        "bandwidth_gbps_max": nbytes * legs / samples[0] / 1e9,
        "spread_pct": (samples[-1] - samples[0]) / med * 100,
        "iters": len(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="0-1,0-2",
                    help="device pairs to time, e.g. '0-1,0-2' (within-pair, cross-pair)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ndev = torch.cuda.device_count()
    results = {}
    for spec in args.pairs.split(","):
        s, d = (int(x) for x in spec.split("-"))
        if s >= ndev or d >= ndev:
            print(f"  skip {spec}: only {ndev} devices visible", flush=True)
            continue
        peer = torch.cuda.can_device_access_peer(s, d)
        per_size = {}
        for mib in SIZES_MIB:
            nbytes = mib << 20
            one = _summarise(_one_way(s, d, nbytes), nbytes, legs=1)
            rt = _summarise(_round_trip(s, d, nbytes), nbytes, legs=2)
            per_size[f"{mib}MiB"] = {"one_way": one, "round_trip": rt}
            print(f"  {spec} {mib:4d} MiB  one-way {one['bandwidth_gbps']:6.2f}"
                  f" (spread {one['spread_pct']:5.2f} %)   round-trip"
                  f" {rt['bandwidth_gbps']:6.2f} (spread {rt['spread_pct']:5.2f} %)",
                  flush=True)
        results[spec] = {"peer_access": peer, "sizes": per_size,
                         "device_names": [torch.cuda.get_device_name(s),
                                          torch.cuda.get_device_name(d)]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "label": args.label,
        "torch": torch.__version__,
        "note": "direct peer copy; NCCL is not involved on any path here",
        "pairs": results,
    }, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
