"""Measure the RNGD device facts a profile needs: usable memory and power.

Fills the two ``profiles/accelerators/furiosa_rngd.yaml`` fields that the
layerwise sweep does not produce, so neither has to stay ``placeholder``:

* **usable memory per PE** -- ``furiosa-smi`` reports 47.5 GiB per *card* and
  says nothing about what one PE can address. Measured by bisecting the largest
  bf16 tensor that allocates.
* **power** -- ``idle_power`` / ``active_power`` / ``standby_power`` /
  ``standby_duration``, the simulator's field names, following the deviations
  D7 protocol used for the A5000 and A40: idle baseline with no context,
  active mean under sustained load, standby as the elevated draw right after
  the load stops.

Only the card holding ``--device`` is sampled: npu0/1/2 belong to someone
else's pods (``docs/hardware_roadmap.md``, "Who holds the NPUs"), so a
max-across-cards reading would report their draw, not ours.

``furiosa-smi`` prints board power as whole watts, so every figure here is
quantised to 1 W. That is coarser than ``nvidia-smi``'s reading and is recorded
as such.

Usage::

    PYTHONPATH=$PWD python3 experiments/scripts/rngd_device_facts.py \
        --device rngd:24 --out outputs/rngd_profile/device_facts.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

# isort: off
# furiosa.torch MUST be imported after torch. It imports torch itself, and
# torch's backend autoload then calls furiosa.torch._register while that module
# is still initialising, which fails with "partially initialized module
# 'furiosa.torch' has no attribute '_register' (most likely due to a circular
# import)". isort sorts furiosa before torch alphabetically, so the ordering has
# to be pinned here or `ruff check --fix` silently breaks every run.
import furiosa.torch as ft  # noqa: F401  (registers the rngd device)
# isort: on

BYTES_PER_ELEMENT = 2  # bfloat16


def card_of(device: str) -> str:
    return f"npu{int(device.split(':')[1]) // 8}"


def sample_power_w(card: str) -> float | None:
    """One board-power reading for ``card`` from ``furiosa-smi info``."""
    try:
        raw = subprocess.run(
            ["furiosa-smi", "info"], capture_output=True, text=True, timeout=60
        ).stdout
    except Exception:
        return None
    for line in raw.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if card not in cells:
            continue
        for cell in cells:
            if cell.endswith("W"):
                try:
                    return float(cell[:-1].strip())
                except ValueError:
                    pass
    return None


class PowerSampler(threading.Thread):
    """Poll board power in the background while a load runs."""

    def __init__(self, card: str, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.card = card
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            watts = sample_power_w(self.card)
            if watts is not None:
                self.samples.append((time.time(), watts))
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def between(self, start: float, end: float) -> list[float]:
        return [w for ts, w in self.samples if start <= ts <= end]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def can_allocate(device: str, n_bytes: int) -> bool:
    elements = n_bytes // BYTES_PER_ELEMENT
    try:
        tensor = torch.empty(elements, dtype=torch.bfloat16, device=device)
        del tensor
        return True
    except Exception:
        return False


def measure_usable_memory(device: str, log) -> dict:
    """Bisect the largest single bf16 allocation the PE accepts."""
    gb = 1000 ** 3
    low, high = 0, 1  # high grows until it fails
    while can_allocate(device, high * gb) and high < 64:
        low = high
        high *= 2
        log(f"  allocation of {low} GB ok, trying {high} GB")
    # Bisect between the last success and the first failure, to 0.25 GB.
    lo_b, hi_b = low * gb, high * gb
    while hi_b - lo_b > gb // 4:
        mid = (lo_b + hi_b) // 2
        if can_allocate(device, mid):
            lo_b = mid
        else:
            hi_b = mid
    log(f"  largest single bf16 allocation: {lo_b / gb:.2f} GB")
    return {
        "largest_single_alloc_gb": round(lo_b / gb, 2),
        "method": "bisected torch.empty(bfloat16) on one PE, 0.25 GB resolution",
        "note": (
            "one allocation, not the PE's total budget: the runtime may refuse a "
            "single block smaller than the sum of several"
        ),
    }


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

LOAD_WORKER = """
import sys, time, torch
import furiosa.torch as ft
device, load_s = sys.argv[1], float(sys.argv[2])
size = 4096
a = torch.randn(size, size, dtype=torch.bfloat16).to(device)
b = torch.randn(size, size, dtype=torch.bfloat16).to(device)
matmul = torch.compile(lambda x, y: x @ y, backend=ft.backend)
with torch.no_grad():
    matmul(a, b)
    print("READY", flush=True)
    start = time.time()
    while time.time() - start < load_s:
        matmul(a, b)
"""


def start_load(first_pe: int, n_pes: int, duration_s: float) -> list:
    """Spawn one load process per PE and wait until all are past compilation.

    Returning only once every worker has printed READY matters: otherwise the
    timed window measures concurrent compiles instead of the load.
    """
    workers = []
    for offset in range(n_pes):
        workers.append(subprocess.Popen(
            [sys.executable, "-u", "-c", LOAD_WORKER,
             f"rngd:{first_pe + offset}", str(duration_s)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ))
    for worker in workers:
        assert worker.stdout is not None
        while True:
            line = worker.stdout.readline()
            if not line or line.strip() == "READY":
                break
    return workers


def stop_load(workers: list) -> None:
    for worker in workers:
        worker.terminate()
    for worker in workers:
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()


def fit_line(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares ``y = base + slope * x`` with R^2."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx else 0.0
    base = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (base + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    return {
        "base_w": round(base, 2),
        "per_pe_w": round(slope, 2),
        "r_squared": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
        "n_points": n,
    }


def sweep_power(device: str, max_pes: int, load_s: float, idle_s: float,
                settle_s: float, log) -> dict:
    """Board power against the number of loaded PEs, 0..max_pes.

    This is what separates the card's fixed board cost from the marginal cost of
    a busy PE. The accelerator unit in the profile is one PE, but board power is
    only measurable per card, so without this split an even 8-way division is
    just an assumption -- and it is a bad one if the fixed term is large.
    """
    card = card_of(device)
    first_pe = (int(device.split(":")[1]) // 8) * 8
    sampler = PowerSampler(card)
    sampler.start()

    points = []
    log(f"  idle baseline for {idle_s:.0f}s (0 PEs loaded)")
    start = time.time()
    time.sleep(idle_s)
    idle_samples = sampler.between(start, time.time())
    points.append({"loaded_pes": 0, "samples": idle_samples})

    for n_pes in range(1, max_pes + 1):
        workers = start_load(first_pe, n_pes, load_s + 120)
        time.sleep(settle_s)          # let the board reach steady state
        start = time.time()
        time.sleep(load_s)
        samples = sampler.between(start, time.time())
        stop_load(workers)
        mean = statistics.fmean(samples) if samples else None
        log(f"  {n_pes} PE(s) loaded: mean {mean} W over {len(samples)} samples")
        points.append({"loaded_pes": n_pes, "samples": samples})
        time.sleep(settle_s)          # let it fall back before the next point

    sampler.stop()

    table = []
    for point in points:
        values = point["samples"]
        table.append({
            "loaded_pes": point["loaded_pes"],
            "mean_w": round(statistics.fmean(values), 2) if values else None,
            "min_w": min(values) if values else None,
            "max_w": max(values) if values else None,
            "n": len(values),
        })

    loaded = [row for row in table if row["loaded_pes"] > 0 and row["mean_w"] is not None]
    fit = fit_line([float(r["loaded_pes"]) for r in loaded],
                   [float(r["mean_w"]) for r in loaded]) if len(loaded) >= 2 else None
    return {
        "card": card,
        "first_pe": first_pe,
        "table": table,
        "fit_over_loaded_points": fit,
        "method": (
            "furiosa-smi board power for this card only, polled at 2 Hz. For each "
            "n in 0..max, n PEs run a sustained 4096x4096 bf16 matmul; the window "
            "starts after a settle delay and after every worker reports READY, so "
            "compilation is excluded. Least squares is fitted over n>=1 only, "
            "because the n=0 point has no compiled context loaded and sits below "
            "the loaded-state intercept."
        ),
    }


def measure_host_bandwidth(device: str, sizes_mb: list[int], reps: int, log) -> dict:
    """Host<->PE transfer bandwidth, both directions.

    This bounds the cross-vendor P/D KV path. A GPU-prefill / NPU-decode split
    has no direct device-to-device route, so the KV cache goes
    GPU -> host -> NPU: two copies, and the slower leg sets the ceiling. Only
    the NPU leg is measurable on this machine (there is no GPU here), so what
    this produces is an upper bound on the end-to-end handoff rate, not the
    handoff rate itself.

    Timed host-side with perf_counter, because wall time is what the handoff
    actually costs the scheduler.
    """
    rows = []
    for size_mb in sizes_mb:
        elements = size_mb * 1000 ** 2 // BYTES_PER_ELEMENT
        host = torch.empty(elements, dtype=torch.bfloat16)
        try:
            warm = host.to(device)      # first copy pays context setup
            warm.cpu()
            h2d, d2h = [], []
            for _ in range(reps):
                start = time.perf_counter()
                on_device = host.to(device)
                h2d.append(time.perf_counter() - start)
                start = time.perf_counter()
                on_device.cpu()
                d2h.append(time.perf_counter() - start)
                del on_device
            del warm
            gc.collect()
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

    best_h2d = max((r["h2d_gbps"] for r in rows), default=None)
    best_d2h = max((r["d2h_gbps"] for r in rows), default=None)
    return {
        "device": device,
        "table": rows,
        "peak_h2d_gbps": best_h2d,
        "peak_d2h_gbps": best_d2h,
        "method": (
            "torch .to(device) / .cpu() on a contiguous bfloat16 buffer, host-side "
            "perf_counter, best and median of N reps after one warm copy. Bounds "
            "the cross-vendor P/D KV handoff: GPU->host->NPU is two copies and the "
            "GPU leg is not measurable on this machine."
        ),
    }


# ---------------------------------------------------------------------------
# Parallel host -> PE bandwidth
# ---------------------------------------------------------------------------

#: One process per PE, because that is how this file already drives several PEs
#: at once (see ``start_load``). Threads are deliberately NOT used: nothing here
#: establishes that ``furiosa.torch`` lets one interpreter hold several PE
#: contexts concurrently, and the subprocess pattern is the one already proven
#: on this hardware.
#:
#: The parent synchronises the workers by sending an absolute ``time.time()``
#: instant on stdin after every worker has reported READY. ``time.time()`` is
#: comparable across processes; ``perf_counter`` is not, which is why the
#: single-stream path uses it and this one cannot.
BANDWIDTH_WORKER = """
import sys, time, torch
import furiosa.torch as ft  # noqa: F401

device, size_mb, duration_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
elements = size_mb * 1000 ** 2 // 2          # bfloat16
host = torch.empty(elements, dtype=torch.bfloat16)
warm = host.to(device)                        # first copy pays context setup
warm.cpu()
del warm
print("READY", flush=True)


def run(direction, start_at):
    while time.time() < start_at:
        pass                                  # spin: sleep granularity is too coarse
    first = time.time()
    moved = 0
    last = first
    if direction == "d2h":
        on_device = host.to(device)
    # NOTE: the d2h loop calls .cpu(), which allocates a fresh pageable
    # destination every iteration. Above ~16 MB PyTorch's CPU allocator stops
    # reusing a cached block and mmaps instead, so every copy pays page faults
    # and the figure becomes allocator-bound rather than link-bound -- measured
    # and documented on the GPU side of this same path
    # (experiments/results/gpu_host_bandwidth.md). H2D does NOT have this
    # problem: its host buffer is allocated once and reused. Treat h2d as the
    # measurement and d2h as indicative only.
    while time.time() - first < duration_s:
        if direction == "h2d":
            on_device = host.to(device)
            del on_device
        else:
            on_device.cpu()
        moved += 1
        last = time.time()
    print(f"RESULT {direction} {moved} {first!r} {last!r}", flush=True)


for _ in range(2):
    line = sys.stdin.readline().split()       # "GO <direction> <start_at>"
    run(line[1], float(line[2]))
"""


def _parallel_trial(first_pe: int, streams: int, size_mb: int, duration_s: float,
                    log) -> dict | None:
    """One trial: fresh worker processes, both directions, synchronised start.

    Workers are respawned per trial on purpose. Host buffer placement is decided
    once at allocation and is not bound to a NUMA node, so re-allocating is the
    only way to sample it -- on the GPU side of this same measurement two runs
    disagreed by 38 % on the 4-stream figure with the ordering reversing between
    them (experiments/results/gpu_host_bandwidth.md).
    """
    workers = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", BANDWIDTH_WORKER,
             f"rngd:{first_pe + offset}", str(size_mb), str(duration_s)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        for offset in range(streams)
    ]
    try:
        for worker in workers:
            assert worker.stdout is not None
            if worker.stdout.readline().strip() != "READY":
                log(f"  worker on rngd:{first_pe} never reported READY; trial dropped")
                return None

        result: dict = {}
        for direction in ("h2d", "d2h"):
            start_at = time.time() + 2.0      # every worker is already spinning
            for worker in workers:
                assert worker.stdin is not None
                worker.stdin.write(f"GO {direction} {start_at!r}\n")
                worker.stdin.flush()
            spans = []
            for worker in workers:
                assert worker.stdout is not None
                parts = worker.stdout.readline().split()
                if len(parts) != 5 or parts[0] != "RESULT":
                    log(f"  malformed worker line {parts!r}; trial dropped")
                    return None
                spans.append((int(parts[2]), float(parts[3]), float(parts[4])))
            moved = sum(s[0] for s in spans) * size_mb * 1000 ** 2
            window = max(s[2] for s in spans) - min(s[1] for s in spans)
            result[direction] = moved / window / 1e9
        return result
    finally:
        stop_load(workers)


def measure_parallel_bandwidth(device: str, streams: list[int], size_mb: int,
                               duration_s: float, trials: int, log) -> dict:
    """Aggregate host<->PE bandwidth with N PEs of one card transferring at once.

    **This is the measurement the repo has been quoting without committing.** The
    figures 10.39 / 19.10 / 35.47 GB/s at 2 / 4 / 8 streams appear in
    docs/HANDOVER_A40.md, docs/PROJECT_REPORT.md and both P/D fixture comments,
    but outputs/rngd_profile/host_bandwidth.json holds only the single-stream run.
    Every composed cross-vendor fabric number rests on them, so they need to be
    reproducible from committed code.

    Aggregate is total bytes across all workers over the wall time of the
    concurrent region (first start to last finish) -- the same definition the GPU
    leg uses in experiments/scripts/gpu_host_bandwidth.py, so the two compose.

    The statistic differs from the GPU script deliberately: this one measures
    **sustained** throughput over a fixed duration rather than best-of-N single
    transfers, because coordinating a per-transfer barrier across processes is
    not worth the failure modes. A KV handoff is a sustained bulk transfer, so
    sustained is the more honest statistic; the ``streams=1`` row exists to be
    cross-checked against the committed single-stream table, which validates the
    method against a number that is already trusted.
    """
    first_pe = (int(device.split(":")[1]) // 8) * 8
    rows = []
    for width in streams:
        h2d_trials, d2h_trials = [], []
        for _ in range(trials):
            one = _parallel_trial(first_pe, width, size_mb, duration_s, log)
            if one is None:
                continue
            h2d_trials.append(one["h2d"])
            d2h_trials.append(one["d2h"])
        if not h2d_trials:
            log(f"  {width} stream(s): every trial failed")
            continue
        row = {
            "streams": width,
            "first_pe": first_pe,
            "size_mb": size_mb,
            "duration_s": duration_s,
            "trials": len(h2d_trials),
            "h2d_aggregate_gbps": round(statistics.median(h2d_trials), 2),
            "d2h_aggregate_gbps": round(statistics.median(d2h_trials), 2),
            "h2d_trial_min_gbps": round(min(h2d_trials), 2),
            "h2d_trial_max_gbps": round(max(h2d_trials), 2),
            "d2h_trial_min_gbps": round(min(d2h_trials), 2),
            "d2h_trial_max_gbps": round(max(d2h_trials), 2),
        }
        rows.append(row)
        log(f"  {width} stream(s)  H2D {row['h2d_aggregate_gbps']:7.2f} GB/s "
            f"[{row['h2d_trial_min_gbps']:.1f}-{row['h2d_trial_max_gbps']:.1f}]  "
            f"D2H {row['d2h_aggregate_gbps']:7.2f} GB/s "
            f"[{row['d2h_trial_min_gbps']:.1f}-{row['d2h_trial_max_gbps']:.1f}]")

    single = next((r for r in rows if r["streams"] == 1), None)
    for row in rows:
        if single and single["h2d_aggregate_gbps"]:
            row["h2d_scaling_vs_ideal"] = round(
                row["h2d_aggregate_gbps"]
                / (row["streams"] * single["h2d_aggregate_gbps"]), 3)
    return {
        "card": card_of(device),
        "table": rows,
        "peak_h2d_aggregate_gbps": max(
            (r["h2d_aggregate_gbps"] for r in rows), default=None),
        "method": (
            "One process per PE (threads are not used: nothing establishes that "
            "furiosa.torch allows several PE contexts in one interpreter, and "
            "subprocess-per-PE is the pattern start_load already proves on this "
            "hardware). Workers allocate a contiguous pageable bfloat16 buffer, "
            "warm once, report READY, then spin until an absolute time.time() "
            "instant sent by the parent and copy back to back for duration_s. "
            "Aggregate is total bytes over the concurrent window, first start to "
            "last finish -- the same definition experiments/scripts/"
            "gpu_host_bandwidth.py uses for the GPU leg, so the two compose. "
            "Sustained throughput, not best-of-N: the streams=1 row cross-checks "
            "against the committed single-stream table. Repeated over independent "
            "trials because host buffer NUMA placement is uncontrolled. h2d is "
            "the measurement -- its host buffer is allocated once and reused. "
            "d2h is indicative only: .cpu() allocates a fresh destination per "
            "iteration and is allocator-bound above ~16 MB, the same artifact "
            "documented for the GPU leg."
        ),
    }


def measure_power(device: str, load_s: float, idle_s: float, load_pes: int, log) -> dict:
    """Idle / active / standby board power for one card.

    ``load_pes`` is what makes ``active`` comparable to the GPU profiles: a
    single PE busy leaves 7/8 of the card idle, so its draw is nowhere near the
    card's active power. Loading every PE is the analogue of the A40's
    whole-GPU vLLM bench.
    """
    card = card_of(device)
    first_pe = (int(device.split(":")[1]) // 8) * 8
    sampler = PowerSampler(card)
    sampler.start()

    log(f"  idle baseline for {idle_s:.0f}s (no device context yet)")
    idle_start = time.time()
    time.sleep(idle_s)
    idle_end = time.time()
    idle = sampler.between(idle_start, idle_end)

    log(f"  sustained load on {load_pes} PE(s) of {card} for {load_s:.0f}s")
    workers = start_load(first_pe, load_pes, load_s + 60)
    log("  all load workers ready")
    load_start = time.time()
    time.sleep(load_s)
    load_end = time.time()
    active = sampler.between(load_start + 1.0, load_end)
    stop_load(workers)

    # Standby: the elevated draw immediately after the load stops. The A5000
    # protocol (D7) uses a 2 s window; keep it so the three profiles compare.
    standby_window = 2.0
    time.sleep(standby_window)
    standby = sampler.between(load_end, load_end + standby_window)

    log(f"  idle n={len(idle)} active n={len(active)} standby n={len(standby)}")
    sampler.stop()

    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "n": 0}
        return {
            "mean": round(statistics.fmean(values), 2),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    return {
        "card": card,
        "load_pes": load_pes,
        "idle": stats(idle),
        "active": stats(active),
        "standby": stats(standby),
        "standby_duration_ns": int(standby_window * 1e9),
        "sampler_interval_s": sampler.interval,
        "resolution_w": 1.0,
        "method": (
            "furiosa-smi info board power for this card only, polled at 2 Hz; "
            "active excludes the first second of the load window; standby is "
            "the D7 2 s post-load window"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="rngd:24")
    parser.add_argument("--load-s", type=float, default=40.0)
    parser.add_argument("--idle-s", type=float, default=20.0)
    parser.add_argument("--load-pes", type=int, default=8,
                        help="PEs of the card to load; 8 = the whole card")
    parser.add_argument("--pe-sweep", action="store_true",
                        help="sweep 0..--load-pes loaded PEs and fit the per-PE cost")
    parser.add_argument("--settle-s", type=float, default=5.0,
                        help="pause before and after each sweep point")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--host-bandwidth", action="store_true",
                        help="measure host<->PE transfer bandwidth and nothing else")
    parser.add_argument("--sizes-mb", default="1,4,16,64,256",
                        help="transfer sizes for --host-bandwidth")
    parser.add_argument("--bw-reps", type=int, default=5)
    parser.add_argument("--parallel-bandwidth", action="store_true",
                        help="measure aggregate host<->PE bandwidth across N PEs "
                             "of one card and nothing else; this is what closes "
                             "the multi-stream provenance gap")
    parser.add_argument("--streams", default="1,2,4,8",
                        help="PE counts to sweep for --parallel-bandwidth")
    parser.add_argument("--parallel-size-mb", type=int, default=256,
                        help="transfer size for --parallel-bandwidth; 256 MB is "
                             "the KV footprint of a 2048-token Llama-3.1-8B "
                             "prompt and the size the GPU leg was composed at")
    parser.add_argument("--parallel-duration-s", type=float, default=5.0,
                        help="sustained-transfer window per direction per trial")
    parser.add_argument("--parallel-trials", type=int, default=3,
                        help="independent worker spawns per stream count; the "
                             "median trial is the headline")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/rngd_profile/device_facts.json"))
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    log(f"=== RNGD device facts on {args.device} ({card_of(args.device)}) ===")
    facts = {
        "device": args.device,
        "card": card_of(args.device),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if args.parallel_bandwidth:
        facts["parallel_bandwidth"] = measure_parallel_bandwidth(
            args.device, [int(v) for v in args.streams.split(",") if v],
            args.parallel_size_mb, args.parallel_duration_s,
            args.parallel_trials, log)
    elif args.host_bandwidth:
        facts["host_bandwidth"] = measure_host_bandwidth(
            args.device, [int(v) for v in args.sizes_mb.split(",") if v],
            args.bw_reps, log)
    elif args.pe_sweep:
        facts["pe_power_sweep"] = sweep_power(
            args.device, args.load_pes, args.load_s, args.idle_s, args.settle_s, log)
    else:
        facts["power_w"] = measure_power(
            args.device, args.load_s, args.idle_s, args.load_pes, log)
    if not args.skip_memory and not args.parallel_bandwidth:
        facts["memory"] = measure_usable_memory(args.device, log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2) + "\n")
    log(f"wrote {args.out}")
    log(json.dumps(facts.get("parallel_bandwidth") or facts.get("host_bandwidth")
                   or facts.get("pe_power_sweep") or facts.get("power_w"), indent=2))


if __name__ == "__main__":
    main()
