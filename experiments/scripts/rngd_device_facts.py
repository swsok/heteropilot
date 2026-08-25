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
    if args.pe_sweep:
        facts["pe_power_sweep"] = sweep_power(
            args.device, args.load_pes, args.load_s, args.idle_s, args.settle_s, log)
    else:
        facts["power_w"] = measure_power(
            args.device, args.load_s, args.idle_s, args.load_pes, log)
    if not args.skip_memory:
        facts["memory"] = measure_usable_memory(args.device, log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2) + "\n")
    log(f"wrote {args.out}")
    log(json.dumps(facts.get("pe_power_sweep") or facts.get("power_w"), indent=2))


if __name__ == "__main__":
    main()
