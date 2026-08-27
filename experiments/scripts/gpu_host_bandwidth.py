"""Measure the GPU leg of the cross-vendor Prefill/Decode KV handoff.

A GPU-prefill / NPU-decode split has no device-to-device route between vendors,
so the KV cache travels **GPU -> host -> NPU**: two copies, and the slower leg
sets the ceiling. The NPU leg was measured on the NPU server
(``outputs/rngd_profile/host_bandwidth.json``, and the multi-stream aggregate
quoted in ``docs/HANDOVER_A40.md`` §1). This script measures the leg that only
an NVIDIA host can produce, so both P/D fixtures can stop carrying the NPU leg
as a stand-in for the whole path.

**The relevant direction is D2H.** The GPU is the prefill side, so its
contribution to the handoff is device-to-host. H2D is measured too, because the
pinned/pageable asymmetry differs by direction and the reverse split
(NPU prefill -> GPU decode) would need it.

Method is deliberately identical to ``measure_host_bandwidth()`` in
``experiments/scripts/rngd_device_facts.py`` -- contiguous bfloat16 buffer,
host-side ``perf_counter``, best and median of N reps after one warm copy,
sizes 1/4/16/64/256 MB -- because two legs measured differently are not
comparable. Three CUDA-specific departures, each forced and each recorded in
the emitted JSON:

* **Explicit ``torch.cuda.synchronize()`` inside the timed region.** CUDA copies
  can complete asynchronously with respect to the host; timing the enqueue
  instead of the transfer would report bandwidth the scheduler never gets. The
  RNGD ``.to(device)`` blocks, so synchronising here mirrors its semantics
  rather than diverging from them.
* **Pinned host memory is measured as well as pageable.** ``cudaHostAlloc``
  changes CUDA transfer rates several-fold and the NPU figures are pageable, so
  reporting one number without saying which would be meaningless. Both are
  emitted; the comparison in the fixtures must state which it used.
* **The pinned path reuses a pre-allocated destination** instead of calling
  ``.cpu()``. Pinned allocation is expensive and a real KV handoff would use a
  persistent staging buffer, never a fresh pinned buffer per transfer. The
  pageable path keeps ``.cpu()`` exactly as the RNGD script has it, so the
  pageable rows are the like-for-like comparison against the NPU leg.

Parallel groups exist because the NPU leg scaled to 88 % of ideal across 8 PEs,
and whether the GPU leg does the same is the open question. Groups are chosen
against ``nvidia-smi topo -m``: GPUs 0-3 sit on NUMA node 0 and 4-7 on node 1,
so a same-node group and a cross-node group of the same width separate PCIe
host-bridge contention from UPI traversal.

Usage::

    .venv-vllm/bin/python experiments/scripts/gpu_host_bandwidth.py \\
        --out outputs/a40_profile/host_bandwidth.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

BYTES_PER_ELEMENT = 2  # bfloat16, matching the RNGD measurement

# The NPU leg, host -> RNGD PE, from the NPU server. Single stream is the
# committed JSON's peak_h2d_gbps; the 8-stream aggregate is what a TP=8 decode
# island actually sustains when its KV is sharded across PEs.
NPU_LEG_SINGLE_GBPS = 5.06
NPU_LEG_PARALLEL_GBPS = 35.47


def gpu_facts() -> list[dict]:
    """Per-device identity, so a re-run on other silicon is not mistaken for this one."""
    facts = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        facts.append({
            "index": idx,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "pci_bus_id": torch.cuda.get_device_properties(idx).pci_bus_id
            if hasattr(props, "pci_bus_id") else None,
        })
    return facts


def topology() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60
        ).stdout.strip() or None
    except Exception:
        return None


def _transfer_once(host: torch.Tensor, device: str, dst: torch.Tensor | None
                   ) -> tuple[float, float]:
    """One H2D + one D2H, each timed host-side with the copy synchronised.

    ``dst`` pre-allocated => pinned path; ``None`` => pageable path via
    ``.cpu()``, which is what the RNGD script does.
    """
    start = time.perf_counter()
    on_device = host.to(device)
    torch.cuda.synchronize(device)
    h2d = time.perf_counter() - start

    start = time.perf_counter()
    if dst is None:
        on_device.cpu()
    else:
        dst.copy_(on_device)
    torch.cuda.synchronize(device)
    d2h = time.perf_counter() - start

    del on_device
    return h2d, d2h


def measure_host_bandwidth(device: str, sizes_mb: list[int], reps: int,
                           pinned: bool, log) -> dict:
    """Host<->GPU transfer bandwidth, both directions, one device.

    Mirrors ``rngd_device_facts.measure_host_bandwidth`` row for row so the two
    legs of the KV path can be composed.
    """
    rows = []
    for size_mb in sizes_mb:
        elements = size_mb * 1000 ** 2 // BYTES_PER_ELEMENT
        host = torch.empty(elements, dtype=torch.bfloat16, pin_memory=pinned)
        dst = (torch.empty(elements, dtype=torch.bfloat16, pin_memory=True)
               if pinned else None)
        try:
            _transfer_once(host, device, dst)   # first copy pays context setup
            h2d, d2h = [], []
            for _ in range(reps):
                one_h2d, one_d2h = _transfer_once(host, device, dst)
                h2d.append(one_h2d)
                d2h.append(one_d2h)
            del host, dst
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            log(f"  {size_mb:5d} MB FAILED {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:100]}")
            continue
        bytes_moved = elements * BYTES_PER_ELEMENT
        row = {
            "size_mb": size_mb,
            "h2d_gbps": round(bytes_moved / min(h2d) / 1e9, 2),
            "d2h_gbps": round(bytes_moved / min(d2h) / 1e9, 2),
            "h2d_median_gbps": round(bytes_moved / statistics.median(h2d) / 1e9, 2),
            "d2h_median_gbps": round(bytes_moved / statistics.median(d2h) / 1e9, 2),
            "reps": reps,
        }
        rows.append(row)
        log(f"  {size_mb:5d} MB  H2D {row['h2d_gbps']:7.2f} GB/s (best) "
            f"{row['h2d_median_gbps']:7.2f} (median)   "
            f"D2H {row['d2h_gbps']:7.2f} / {row['d2h_median_gbps']:7.2f}")

    return {
        "device": device,
        "host_memory": "pinned" if pinned else "pageable",
        "table": rows,
        "peak_h2d_gbps": max((r["h2d_gbps"] for r in rows), default=None),
        "peak_d2h_gbps": max((r["d2h_gbps"] for r in rows), default=None),
    }


def _parallel_trial(indices: list[int], elements: int, reps: int, pinned: bool
                    ) -> tuple[float, float]:
    """One trial: fresh host buffers, N barrier-synchronised concurrent reps.

    Returns the trial's best H2D and D2H aggregate. Buffers are allocated inside
    the trial because that is what varies between runs -- their NUMA placement is
    fixed at allocation and uncontrolled, so re-allocating is the only way to
    sample it.
    """
    width = len(indices)
    barrier = threading.Barrier(width)
    spans: dict[int, list[tuple[float, float, float, float]]] = {i: [] for i in indices}

    def worker(idx: int) -> None:
        torch.cuda.set_device(idx)
        device = f"cuda:{idx}"
        host = torch.empty(elements, dtype=torch.bfloat16, pin_memory=pinned)
        dst = (torch.empty(elements, dtype=torch.bfloat16, pin_memory=True)
               if pinned else None)
        _transfer_once(host, device, dst)       # warm
        for _ in range(reps):
            barrier.wait()
            h2d_start = time.perf_counter()
            on_device = host.to(device)
            torch.cuda.synchronize(device)
            h2d_end = time.perf_counter()

            barrier.wait()
            d2h_start = time.perf_counter()
            if dst is None:
                on_device.cpu()
            else:
                dst.copy_(on_device)
            torch.cuda.synchronize(device)
            d2h_end = time.perf_counter()
            del on_device
            spans[idx].append((h2d_start, h2d_end, d2h_start, d2h_end))
        del host, dst

    threads = [threading.Thread(target=worker, args=(i,)) for i in indices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    bytes_per_rep = elements * BYTES_PER_ELEMENT * width
    h2d_rates, d2h_rates = [], []
    for rep in range(reps):
        h2d_window = max(spans[i][rep][1] for i in indices) - \
            min(spans[i][rep][0] for i in indices)
        d2h_window = max(spans[i][rep][3] for i in indices) - \
            min(spans[i][rep][2] for i in indices)
        h2d_rates.append(bytes_per_rep / h2d_window / 1e9)
        d2h_rates.append(bytes_per_rep / d2h_window / 1e9)
    gc.collect()
    torch.cuda.empty_cache()
    return max(h2d_rates), max(d2h_rates)


def measure_parallel(indices: list[int], size_mb: int, reps: int, trials: int,
                     pinned: bool, log) -> dict:
    """Aggregate bandwidth with every listed GPU copying concurrently.

    Aggregate is total bytes across all workers divided by the wall time of the
    concurrent region (first start to last finish), which is the same definition
    the NPU 8-stream figure uses. Workers meet at a barrier before each rep so
    the region is genuinely overlapped rather than staggered.

    Repeated over ``trials`` independent buffer allocations, and the **median**
    trial is the headline. A single trial is not reproducible here: the host
    buffers are not NUMA-bound, their placement is decided once at allocation,
    and two runs of this script disagreed by 38 % on the 4-GPU pinned D2H figure
    (84.4 vs 60.9 GB/s) with the same-node / cross-node ordering reversing
    between them. The spread is reported so no one reads a single number as
    tighter than it is.
    """
    elements = size_mb * 1000 ** 2 // BYTES_PER_ELEMENT
    h2d_trials, d2h_trials = [], []
    for _ in range(trials):
        one_h2d, one_d2h = _parallel_trial(indices, elements, reps, pinned)
        h2d_trials.append(one_h2d)
        d2h_trials.append(one_d2h)

    result = {
        "gpus": indices,
        "streams": len(indices),
        "size_mb": size_mb,
        "reps": reps,
        "trials": trials,
        "host_memory": "pinned" if pinned else "pageable",
        "h2d_aggregate_gbps": round(statistics.median(h2d_trials), 2),
        "d2h_aggregate_gbps": round(statistics.median(d2h_trials), 2),
        "h2d_trial_min_gbps": round(min(h2d_trials), 2),
        "h2d_trial_max_gbps": round(max(h2d_trials), 2),
        "d2h_trial_min_gbps": round(min(d2h_trials), 2),
        "d2h_trial_max_gbps": round(max(d2h_trials), 2),
    }
    log(f"  {len(indices)} stream(s) {str(indices):24s} "
        f"H2D {result['h2d_aggregate_gbps']:7.2f} GB/s "
        f"[{result['h2d_trial_min_gbps']:.1f}-{result['h2d_trial_max_gbps']:.1f}]  "
        f"D2H {result['d2h_aggregate_gbps']:7.2f} GB/s "
        f"[{result['d2h_trial_min_gbps']:.1f}-{result['d2h_trial_max_gbps']:.1f}]")
    return result


def compose_fabric(gpu_leg_gbps: float, npu_leg_gbps: float) -> dict:
    """The two candidate fabric bandwidths for the P/D fixtures.

    Serialised: the handoff does GPU->host, then host->NPU, so the rates add as
    resistances. Pipelined: the two copies overlap and the slower one sets the
    rate. Which applies is an implementation property, so both are emitted and
    the fixture must say which it assumed.
    """
    return {
        "gpu_leg_gbps": round(gpu_leg_gbps, 2),
        "npu_leg_gbps": round(npu_leg_gbps, 2),
        "serialised_gbps": round(1 / (1 / gpu_leg_gbps + 1 / npu_leg_gbps), 2),
        "pipelined_gbps": round(min(gpu_leg_gbps, npu_leg_gbps), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mb", default="1,4,16,64,256",
                        help="transfer sizes; the RNGD sweep used these")
    parser.add_argument("--reps", type=int, default=7,
                        help="reps per size; the committed RNGD run used 7")
    parser.add_argument("--device-index", type=int, default=0,
                        help="GPU for the single-stream sweep")
    parser.add_argument("--parallel-size-mb", type=int, default=256,
                        help="transfer size for the parallel groups")
    parser.add_argument("--parallel-groups",
                        default="0,1|0,4|0,1,2,3|0,1,4,5|0,1,2,3,4,5,6,7",
                        help="'|'-separated comma lists of GPU indices")
    parser.add_argument("--skip-parallel", action="store_true")
    parser.add_argument("--parallel-trials", type=int, default=5,
                        help="independent buffer allocations per parallel group; "
                             "the median trial is the headline, because host "
                             "buffer NUMA placement is uncontrolled and varies "
                             "run to run")
    parser.add_argument("--fabric-size-mb", type=int, default=256,
                        help="single-stream row used to compose the fabric "
                             "bandwidth; must be one of --sizes-mb")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/a40_profile/host_bandwidth.json"))
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; this script must run on the GPU host")

    sizes = [int(v) for v in args.sizes_mb.split(",") if v]
    device = f"cuda:{args.device_index}"
    torch.cuda.set_device(args.device_index)

    log(f"=== GPU<->host bandwidth on {device} "
        f"({torch.cuda.get_device_name(args.device_index)}) ===")

    results: dict = {
        "host": platform.node(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0],
        "gpus": gpu_facts(),
        "topology": topology(),
        "single_stream": {},
        "parallel": [],
    }

    for pinned in (False, True):
        label = "pinned" if pinned else "pageable"
        log(f"-- single stream, {label} host memory --")
        results["single_stream"][label] = measure_host_bandwidth(
            device, sizes, args.reps, pinned, log)

    if not args.skip_parallel:
        groups = [[int(v) for v in group.split(",") if v]
                  for group in args.parallel_groups.split("|") if group]
        available = torch.cuda.device_count()
        for pinned in (False, True):
            label = "pinned" if pinned else "pageable"
            log(f"-- parallel groups at {args.parallel_size_mb} MB, {label} --")
            for group in groups:
                if any(i >= available for i in group):
                    log(f"  skipping {group}: only {available} GPUs visible")
                    continue
                results["parallel"].append(measure_parallel(
                    group, args.parallel_size_mb, args.reps,
                    args.parallel_trials, pinned, log))

    # The GPU leg of the KV path is D2H: the GPU is the prefill side.
    #
    # Composed at --fabric-size-mb rather than at the peak over all sizes. The
    # peak is the wrong statistic here: pageable D2H peaks at 16 MB and then
    # falls by 5x at 64 MB and above, because PyTorch's CPU allocator stops
    # reusing a cached block at that size and every rep pays fresh page faults.
    # A KV handoff is a bulk transfer -- Llama-3.1-8B is 128 KiB per token, so a
    # 2048-token prompt moves ~262 MB -- so the large-transfer row is the honest
    # one, and it is also where the NPU leg's own peak sits, keeping the two
    # legs like-for-like.
    for label, single in results["single_stream"].items():
        row = next((r for r in single["table"]
                    if r["size_mb"] == args.fabric_size_mb), None)
        if row is None:
            continue
        widest = max(
            (p for p in results["parallel"] if p["host_memory"] == label),
            key=lambda p: p["d2h_aggregate_gbps"], default=None)
        results.setdefault("kv_path", {})[label] = {
            "composed_at_size_mb": args.fabric_size_mb,
            "single_stream": compose_fabric(
                row["d2h_gbps"], NPU_LEG_SINGLE_GBPS),
            "parallel": compose_fabric(
                widest["d2h_aggregate_gbps"], NPU_LEG_PARALLEL_GBPS
            ) if widest else None,
            "parallel_gpu_group": widest["gpus"] if widest else None,
        }

    results["method"] = (
        "torch .to(device) / .cpu() on a contiguous bfloat16 buffer, host-side "
        "perf_counter with torch.cuda.synchronize() inside the timed region, best "
        "and median of N reps after one warm copy. Deliberately mirrors "
        "experiments/scripts/rngd_device_facts.py measure_host_bandwidth() so the "
        "two legs of the cross-vendor KV path compose; the synchronize() is added "
        "because CUDA copies can complete asynchronously and the RNGD .to(device) "
        "does not. Pageable rows are the like-for-like comparison against the NPU "
        "leg (which is pageable); the pinned path reuses a pre-allocated staging "
        "buffer, as a real handoff would. Parallel aggregate is total bytes over "
        "the wall time of the concurrent region, the same definition as the NPU "
        "8-stream figure. Host buffers are not NUMA-bound, so a cross-node group "
        "pays whatever the allocator happened to choose. KNOWN ARTIFACT: pageable "
        "D2H is allocator-bound, not link-bound -- .cpu() allocates a fresh "
        "destination per rep, PyTorch's CPU allocator reuses a cached block up to "
        "~16 MB and mmaps above it, so the pageable D2H column peaks at 16 MB and "
        "drops ~5x at 64 MB and beyond, with medians far below the best. Read the "
        "pinned rows for the link, and the pageable rows only for like-for-like "
        "comparison against the pageable NPU leg."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    log(f"wrote {args.out}")
    log(json.dumps(results.get("kv_path"), indent=2))


if __name__ == "__main__":
    main()
