"""Measure the Rebellions ATOM device facts a profile needs: memory and power.

Fills the ``profiles/accelerators/rbln_atom.yaml`` fields that a layerwise sweep
does not produce, so neither has to stay ``placeholder``:

* **usable memory per card** -- ``rbln-smi`` reports 15.7 GiB of DRAM per card
  and says nothing about how much a compiled model can actually claim. Measured
  by bisecting the largest single bf16 weight a model can carry onto the device.
* **power** -- ``idle_power`` / ``active_power`` / ``standby_power`` /
  ``standby_duration``, the simulator's field names, following the deviations D7
  protocol used for the A5000, A40 and RNGD: idle baseline with no context,
  active mean under sustained load, standby as the elevated draw right after the
  load stops.

ONE ACCELERATOR HERE IS ONE CARD, unlike RNGD where it is one PE. ``rbln-smi``
reports power per card and a card is the unit vLLM binds to, so no per-unit
split is needed and the RNGD script's 0..8 PE regression has no analogue. The
card sweep still runs, because what it establishes -- that loading one card does
not move its neighbours -- is an assumption the profile would otherwise inherit
untested.

``rbln-smi`` reports ``card_power`` in microwatts, so these figures are NOT
quantised to 1 W the way the RNGD and A40 numbers are.

Usage::

    .venv-rbln/bin/python experiments/scripts/atom_device_facts.py \
        --device 0 --out outputs/atom_profile/device_facts.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

# isort: off
# rebel MUST be imported after torch: it imports torch itself and registers the
# RBLN backend during that import. Keep this ordering pinned -- the sibling RNGD
# script carries the same guard for furiosa.torch and commit 46f0c70 records
# what happens when `ruff check --fix` reorders it.
import rebel
# isort: on

BYTES_PER_ELEMENT = 2  # bfloat16


def smi_devices() -> list[dict]:
    """Per-card state from ``rbln-smi -j``.

    This is the authoritative view: the sysfs attributes are per-device but the
    npu index in them collapsed to 0 for every card while the driver sat in a
    partially initialised state (docs/hardware_roadmap.md, 2026-08-28 update).
    """
    try:
        raw = subprocess.run(
            ["rbln-smi", "-j"], capture_output=True, text=True, timeout=120
        ).stdout
        return json.loads(raw).get("devices", [])
    except Exception:
        return []


def sample_power_w(device: int) -> float | None:
    """One card-power reading for ``device``, in watts.

    ``card_power`` is reported in microwatts (e.g. ``"19341755uW"``), so this is
    finer-grained than the whole-watt readings furiosa-smi and nvidia-smi give.
    """
    for d in smi_devices():
        if d.get("device") != f"rbln{device}":
            continue
        raw = str(d.get("card_power", "")).strip()
        if raw.endswith("uW"):
            try:
                return float(raw[:-2]) / 1e6
            except ValueError:
                return None
    return None


def sample_power_util(device: int) -> tuple[float | None, float | None]:
    """Card power (W) and utilisation (%) from one ``rbln-smi`` call.

    Utilisation is recorded because a power figure taken below saturation is not
    active power: a one-Linear synchronous load leaves this card at ~36% util and
    reads 44 W, while a saturating one reaches ~94% and reads 67 W. The artifact
    carries the util so the reading can be judged rather than trusted.
    """
    for d in smi_devices():
        if d.get("device") != f"rbln{device}":
            continue
        raw = str(d.get("card_power", "")).strip()
        watts = None
        if raw.endswith("uW"):
            with contextlib.suppress(ValueError):
                watts = float(raw[:-2]) / 1e6
        try:
            util = float(d.get("util"))
        except (TypeError, ValueError):
            util = None
        return watts, util
    return None, None


class PowerSampler(threading.Thread):
    """Poll card power in the background while a load runs."""

    def __init__(self, device: int, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.interval = interval
        self.samples: list[tuple[float, float, float | None]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            watts, util = sample_power_util(self.device)
            if watts is not None:
                self.samples.append((time.time(), watts, util))
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def between(self, start: float, end: float) -> list[float]:
        return [w for ts, w, _ in self.samples if start <= ts <= end]

    def util_between(self, start: float, end: float) -> list[float]:
        return [u for ts, _, u in self.samples
                if start <= ts <= end and u is not None]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def device_used_gib(device: int) -> float | None:
    """Resident device memory for one card, from the driver's own counter."""
    for d in smi_devices():
        if d.get("device") == f"rbln{device}":
            try:
                return int(d["memory"]["used"]) / 1024 ** 3
            except Exception:
                return None
    return None


def _n_for_device_gib(gib: float) -> int:
    """Square Linear width whose ON-DEVICE weight is about ``gib``.

    The compiler places weights as 2 bytes, not the module's 4: a 16384-wide
    Linear is 1.00 GiB of fp32 on the host and lands as 0.516 GiB on the card.
    So size the probe from the device figure, not the torch one.
    """
    n = int(((gib * 1024 ** 3) / BYTES_PER_ELEMENT) ** 0.5)
    return max(n - n % 64, 64)


def measure_memory(device: int, log, lo_gb: float = 1.0, hi_gb: float = 15.5,
                   tol_gb: float = 0.5) -> dict:
    """Largest single model whose weights place and run on one card.

    Bisects the on-device weight size of a one-Linear model, and takes the
    driver's ``memory.used`` as ground truth rather than the compiler's
    ``get_total_device_alloc()`` -- those disagree, and only the former tracks
    what is actually resident (see the note in the returned ``method``).

    This asks the same question the RNGD script asks -- largest single block the
    device accepts -- so the two profiles' ``memory_gb`` are comparable. It is
    ONE allocation, not a total budget.
    """
    def fits(gib: float) -> tuple[bool, dict | None]:
        n = _n_for_device_gib(gib)
        rt = cm = mod = None
        t0 = time.time()
        try:
            mod = torch.nn.Linear(n, n, bias=False).eval()
            cm = rebel.compile_from_torch(mod, [("x", [1, n], "float32")])
            rt = cm.create_runtime(tensor_type="pt", device=device)
            rt.run(torch.zeros(1, n))
            obs = device_used_gib(device)
            log(f"    {gib:6.2f} GiB (n={n}) accepted; driver reports "
                f"{obs:.3f} GiB resident ({time.time() - t0:.0f}s)")
            return True, {"target_gib": round(gib, 3), "n": n,
                          "observed_resident_gib": round(obs, 3) if obs else None,
                          "compiler_total_device_alloc": cm.get_total_device_alloc()}
        except Exception as exc:
            log(f"    {gib:6.2f} GiB (n={n}) refused after {time.time() - t0:.0f}s: "
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
            return False, None
        finally:
            del rt, cm, mod

    log(f"  bisecting largest single on-device weight on rbln{device} "
        f"({lo_gb}-{hi_gb} GiB, {tol_gb} GiB resolution)")
    ok, best = fits(lo_gb)
    if not ok:
        log(f"  even {lo_gb} GiB was refused; aborting memory probe")
        return {"device": device, "largest_single_alloc_gb": None}
    lo, hi = lo_gb, hi_gb
    while hi - lo > tol_gb:
        mid = (lo + hi) / 2
        ok, row = fits(mid)
        if ok:
            lo, best = mid, row
        else:
            hi = mid
    total = next((int(d["memory"]["total"]) for d in smi_devices()
                  if d.get("device") == f"rbln{device}"), None)
    return {
        "device": device,
        "largest_single_alloc_gb": round(lo, 3),
        "at_best": best,
        "bisect_tolerance_gb": tol_gb,
        "dram_total_gb": round(total / 1024 ** 3, 3) if total else None,
        "method": (
            "largest single Linear weight that compiles, places and runs on one "
            "card, bisected to the stated tolerance, sized by ON-DEVICE bytes "
            "(the compiler places weights as 2 bytes, not the module's 4). "
            "Ground truth is the driver's memory.used, NOT the compiler's "
            "get_total_device_alloc(): the latter is a compile-time figure and "
            "over-reports residency by ~70x when several runtimes share one "
            "compiled model, because the weights upload once and each runtime "
            "adds only a small context. One allocation, not a total budget."
        ),
    }


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

#: One process per card, mirroring the RNGD script: a compiled runtime is bound
#: to a device and driving several from one interpreter is not established here.
LOAD_WORKER = """
import sys, time, torch, rebel

device, duration_s, n, batch = (int(sys.argv[1]), float(sys.argv[2]),
                               int(sys.argv[3]), int(sys.argv[4]))


class Deep(torch.nn.Module):
    # A DEEP stack, not one Linear. A single matmul leaves the card at ~36%
    # utilisation because every rt.run() is a synchronous host round trip, and
    # power tracks utilisation almost linearly here -- so a shallow load
    # under-reads active power by ~30%. Eight layers per graph amortise the
    # round trip over eight matmuls.
    def __init__(self):
        super().__init__()
        self.l = torch.nn.ModuleList(
            [torch.nn.Linear(n, n, bias=False) for _ in range(8)])

    def forward(self, x):
        for m in self.l:
            x = m(x)
        return x


mod = Deep().eval()
cm = rebel.compile_from_torch(mod, [("x", [batch, n], "float32")])
# Async, with several calls in flight: overlapping the host round trip with
# device work is worth ~10 utilisation points on top of the depth.
rt = cm.create_async_runtime(tensor_type="pt", device=device)
x = torch.randn(batch, n)
rt.run(x).wait()                              # warm: first run pays setup
print("READY", flush=True)
sys.stdin.readline()
end = time.time() + duration_s
inflight = []
while time.time() < end:
    inflight.append(rt.run(x))
    if len(inflight) >= 8:
        inflight.pop(0).wait()
for f in inflight:
    f.wait()
print("DONE", flush=True)
"""


def start_load(devices: list[int], duration_s: float, n: int = 8192,
               batch: int = 8192) -> list:
    """Spawn one saturating worker per card and wait until all are warm.

    Defaults reach ~86% device utilisation on one card; the single-card power
    measurement raises n further. Verify with the util figures the artifact
    records -- a power reading taken below saturation is not active power.
    """
    workers = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", LOAD_WORKER, str(d), str(duration_s),
             str(n), str(batch)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        for d in devices
    ]
    for w in workers:
        assert w.stdout is not None
        if w.stdout.readline().strip() != "READY":
            stop_load(workers)
            raise RuntimeError("a load worker never reported READY")
    for w in workers:
        assert w.stdin is not None
        w.stdin.write("GO\n")
        w.stdin.flush()
    return workers


def stop_load(workers: list) -> None:
    for w in workers:
        try:
            w.kill()
            w.wait(timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def measure_power(device: int, load_s: float, idle_s: float, log,
                  settle_s: float = 45.0, load_n: int = 12288,
                  load_batch: int = 8192) -> dict:
    """Idle / active / standby card power, deviations D7 protocol.

    ``load_n``/``load_batch`` default to the configuration measured to reach
    ~94% device utilisation. Do not lower them without re-reading the util the
    artifact records -- power here is close to linear in utilisation.
    """
    sampler = PowerSampler(device)
    sampler.start()
    active_util: list[float] = []
    try:
        # Settle first. A memory probe or an earlier load leaves the card
        # elevated for tens of seconds, and an idle window opened straight after
        # one reads high: the first run of this script sampled 19.0-36.7 W in a
        # window meant to be idle, because the 15 GiB bisect had just released.
        log(f"  settling {settle_s:.0f}s before the idle baseline")
        time.sleep(settle_s)
        log(f"  idle baseline on rbln{device} for {idle_s:.0f}s")
        t0 = time.time()
        time.sleep(idle_s)
        idle = sampler.between(t0, time.time())

        log(f"  sustained load on rbln{device} for {load_s:.0f}s")
        workers = start_load([device], load_s + 60, n=load_n, batch=load_batch)
        try:
            t1 = time.time() + 5.0          # let the draw settle
            time.sleep(load_s)
            t1e = time.time()
            active = sampler.between(t1, t1e)
            active_util = sampler.util_between(t1, t1e)
        finally:
            stop_load(workers)

        t2 = time.time()
        time.sleep(2.0)                      # the D7 window
        standby = sampler.between(t2, time.time())
    finally:
        sampler.stop()

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
                "min": round(min(xs), 3), "max": round(max(xs), 3)}

    return {
        "device": device,
        "idle_w": stats(idle),
        "active_w": stats(active),
        "active_util_pct": stats(active_util),
        "standby_w": stats(standby),
        "standby_duration_ns": 2_000_000_000,
        "method": (
            "rbln-smi -j card_power for this card only, polled at 2 Hz. Idle with "
            "no context, active mean over a sustained compiled-matmul load after a "
            "3 s settle, standby over the 2 s window right after the load stops "
            "(deviations D7). Reported in microwatts by the tool, so not quantised "
            "to 1 W the way the RNGD and A40 figures are."
        ),
    }


def measure_card_sweep(devices: list[int], load_s: float, log) -> dict:
    """Load 1..N cards and read EVERY card's power at each point.

    A card is the accelerator unit here, so unlike the RNGD PE sweep this is not
    needed to split a shared board cost. It answers a different question the
    profile would otherwise assume: whether loading one card moves its
    neighbours. If the off-diagonal stays at idle, per-card power is additive and
    a multi-card ClusterSpecV2 can simply sum it.
    """
    rows = []
    for k in range(0, len(devices) + 1):
        loaded = devices[:k]
        workers = start_load(loaded, load_s + 15) if loaded else []
        try:
            time.sleep(3.0)
            per_card = {}
            for _ in range(4):
                for d in devices:
                    w = sample_power_w(d)
                    if w is not None:
                        per_card.setdefault(d, []).append(w)
                time.sleep(0.5)
        finally:
            stop_load(workers)
        row = {
            "loaded_cards": k,
            "loaded": loaded,
            "per_card_w": {f"rbln{d}": round(statistics.mean(v), 3)
                           for d, v in sorted(per_card.items())},
        }
        row["total_w"] = round(sum(row["per_card_w"].values()), 3)
        rows.append(row)
        log(f"  {k} card(s) loaded  total {row['total_w']:7.3f} W  "
            + "  ".join(f"{k2}={v2:6.2f}" for k2, v2 in row["per_card_w"].items()))
        time.sleep(2.0)
    return {"table": rows,
            "method": ("one saturating process per loaded card; every card's "
                       "power averaged over 4 samples after a 3 s settle.")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0,
                        help="card index to sample (0..3)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--card-sweep", action="store_true",
                        help="load 1..N cards and read every card at each point")
    parser.add_argument("--load-s", type=float, default=20.0)
    parser.add_argument("--idle-s", type=float, default=10.0)
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    present = [d.get("device") for d in smi_devices()]
    log(f"=== ATOM device facts on rbln{args.device} "
        f"(cards present: {', '.join(present) or 'none'}) ===")
    if rebel.device_count() == 0:
        log("rebel.device_count() == 0 -- the runtime sees no ATOM. Check "
            "rbln-smi -j for a collapsed 'npu' index before suspecting packaging "
            "(docs/hardware_roadmap.md, 2026-08-28 update).")
        return 2

    facts: dict = {
        "device": args.device,
        "device_count": rebel.device_count(),
        "npu_name": rebel.get_npu_name(),
        "rebel_version": rebel.__version__,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
    }
    if not args.skip_memory:
        facts["memory"] = measure_memory(args.device, log)
    if not args.skip_power:
        facts["power"] = measure_power(args.device, args.load_s, args.idle_s, log)
    if args.card_sweep:
        facts["card_sweep"] = measure_card_sweep(
            list(range(rebel.device_count())), args.load_s, log)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2) + "\n")
    log(f"\nwrote {args.out}")
    log(json.dumps({k: v for k, v in facts.items() if k != "card_sweep"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
