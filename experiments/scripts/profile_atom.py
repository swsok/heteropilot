"""Layerwise profiler for Rebellions ATOM, emitting a §3.7 CSV bundle.

Why this exists
---------------
``python -m profiler`` cannot drive an ATOM, for a reason that is *not* the one
RNGD had. The ``rbln`` vLLM platform plugin exists and activates, but the
profiler's two non-negotiable engine settings each collide with one of
vllm-rbln's two execution paths (``profiler/core/config.py``
``HOST_ENGINE_DEFAULTS``, marked "should not be changed"):

* ``load_format: "dummy"`` -- the default *optimum* path AOT-compiles a **real**
  checkpoint through ``optimum-rbln``, so it demands actual safetensors and
  fails with ``no file named model.safetensors`` on the profiler's temp config
  directory.
* ``enforce_eager: True`` -- the vLLM-native path (``VLLM_RBLN_USE_VLLM_MODEL=1``)
  accepts dummy weights but rejects eager unless ``VLLM_RBLN_USE_DEVICE_TENSOR=1``,
  which needs a real torch device named ``rbln``. Nothing on this machine
  registers one (there is no ``torch_rbln``), unlike RNGD where ``furiosa.torch``
  registers ``rngd:`` as PrivateUse1. That path also registers only
  deepseek_v2 / gpt_oss / minimax_m2 / qwen2 / qwen3 -- **not llama**, which is
  the model RNGD was profiled with.

Working around either means editing ``profiler/``, which is upstream and
pristine until Phase 5 (absolute rule 1). So this script measures the layers
directly with ``rebel`` and writes CSVs for
``profiler.core.importer.CsvProfileImporter``, exactly as ``profile_rngd.py``
does -- the Phase 3 V1 route.

The layers themselves come from ``llama_layers``, shared with the RNGD harness,
so both NPU arms of Exp 4 measure literally the same mathematics.

How the timing works
--------------------
Unlike RNGD, there is no usable device-span profiler here. ``rebel._C.profiler``
does emit protobuf traces (with ``RBLN_PROFILER=1``) containing ``comp_cycle``
and ``transfer`` records, but the schema is undocumented and converting it to
microseconds would mean guessing a field layout and a clock rate. A wrong guess
would look like a measurement, so this script does not do it.

Instead each shape is timed as **best-of-N wall clock per forward, minus a
per-shot I/O baseline measured on the same input shapes**:

* every ``rt.run()`` is a host->device->host round trip whose cost scales
  steeply with the bytes crossing that boundary -- measured with the baseline
  graph itself: 6.4 us at 16 B, 56.3 us at 2 KB, 300.6 us at 2 MB, 999.7 us at
  8 MB;
* so the baseline is re-measured for **every shot** with the same input shapes.
  A single constant does not work: the first version of this script subtracted
  6.5 us (calibrated on a 1x8 tensor) and left every shot carrying its own
  transfer cost, which inflated the elementwise layers 8-25x against the RNGD
  bundle -- layernorm@1 read 52.4 us where RNGD's device span is 3.4 us;
* the statistic is the **minimum** over reps, since jitter only ever adds time.

Two flags in the sidecar mark rows where the subtraction is doing the work and
the figure is an upper bound rather than a measurement: ``baseline_dominated``
(the layer costs less than its own I/O) and ``output_larger_than_baseline`` (the
layer's output is much bigger than the baseline's, so output-side transfer is
under-subtracted). Read ``time_us`` together with them.

**This is not the same statistic as the RNGD bundle's**, which is a true device
span from a vendor profiler. These numbers are wall-clock differences, so they
still contain whatever host cost the subtraction does not cancel. Treat them as
upper bounds on device time, and see the results file before comparing the two
NPUs layer by layer.

Usage::

    .venv-rbln/bin/python experiments/scripts/profile_atom.py \\
        --device 0 --out profiler/perf --max-tokens 2048
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

# isort: off
# rebel MUST be imported after torch: it imports torch itself and registers the
# RBLN compiler during that import. The sibling RNGD harness carries the same
# guard for furiosa.torch; commit 46f0c70 records what breaks when it is lost.
import rebel
# isort: on

from llama_layers import (
    DENSE_LAYERS,
    PER_SEQUENCE_LAYERS,
    attention_case,
    attention_shots,
    dense_layer,
    kv_bytes,
    load_model_config,
    per_sequence_layer,
    write_csv,
    x2_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
#: One card addresses ~15.05 GiB (measured, atom_device_facts.py). Attention
#: shots are budgeted well below that so a large shot cannot poison the smaller
#: ones that follow it in the same process.
DEFAULT_KV_BUDGET_BYTES = 4 * 1024 ** 3

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
}


def input_info(tensors: tuple) -> list[tuple]:
    """``rebel.compile_from_torch`` input spec for a tuple of example tensors."""
    spec = []
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"non-tensor input at {i}: {type(t)}")
        name = _DTYPE_NAMES.get(t.dtype)
        if name is None:
            raise TypeError(f"unmapped dtype {t.dtype}")
        spec.append((f"x{i}", list(t.shape), name))
    return spec


class _IOBaseline(nn.Module):
    """Same inputs as the real layer, essentially no arithmetic.

    This is the per-shot I/O baseline, and it has to be per-shot rather than a
    constant: the cost of an ``rt.run()`` scales steeply with how many bytes
    cross the host boundary. Measured with this very module, a 16-byte input
    costs 6.4 us and an 8 MB one costs 999.7 us, with a ~50 us step already at
    2 KB. Subtracting a single small constant (the first version of this script
    used 6.5 us, calibrated on a 1x8 tensor) leaves each shot carrying its own
    transfer cost, which inflated the elementwise layers 8-25x against the RNGD
    bundle's device-span numbers -- layernorm@1 read 52.4 us where RNGD's device
    span is 3.4 us.

    ``xs[0] + 1`` forces a full read and write of the first input; the ``sum()``
    over the rest forces them to be transferred too. The output has the shape of
    ``xs[0]``, so the output-side transfer is subtracted correctly whenever the
    layer's output is that shape -- true for the norms, activations and rotary,
    and for the projections only up to the in/out width ratio. Where the real
    output is much larger than ``xs[0]`` (``lm_head``, ``embedding``) the
    baseline under-subtracts and the row stays an upper bound; those rows are
    flagged ``output_larger_than_baseline`` in the sidecar.
    """

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        out = xs[0] + 1
        for extra in xs[1:]:
            out = out + extra.sum()
        return out


class Measurer:
    """Compile one module per shape and time it on device.

    ``time_us`` is best-of-N wall clock minus a per-shot I/O baseline. See the
    module docstring for why there is no device-span path and why the baseline
    cannot be a constant.
    """

    def __init__(self, device: int, reps: int, log) -> None:
        self.device = device
        self.reps = reps
        self.log = log
        self.notes: list[dict] = []

    def _time(self, module: nn.Module, inputs: tuple,
              reps: int | None = None) -> dict | None:
        """Per-forward wall time, timing EACH rep and reporting the spread.

        The statistic is the **minimum**, not the mean: every rt.run() is a host
        round trip whose cost is inflated by scheduler jitter, page faults and
        other processes, all of which only ever ADD time. The fastest rep is the
        one least contaminated -- the same best-of-N reasoning the committed
        host-bandwidth measurement uses. Median and max travel with it in the
        sidecar so a noisy shot is visible rather than hidden.

        Card utilisation is deliberately NOT sampled per shot. A shot is a few
        hundred microseconds while one rbln-smi call costs ~200 ms; the reading
        would be ~0 everywhere and would suggest an idle card rather than a
        mismatched instrument. atom_device_facts.py samples utilisation because
        its loads run for tens of seconds.
        """
        reps = reps or self.reps
        rt = cm = None
        try:
            cm = rebel.compile_from_torch(module.eval(), input_info(inputs))
            rt = cm.create_runtime(tensor_type="pt", device=self.device)
            rt.run(*inputs)                       # warm: pays context setup
            per_rep = []
            for _ in range(reps):
                start = time.perf_counter()
                rt.run(*inputs)
                per_rep.append((time.perf_counter() - start) * 1e6)
            return {
                "min_us": min(per_rep),
                "median_us": statistics.median(per_rep),
                "max_us": max(per_rep),
                "reps": reps,
            }
        except Exception as exc:
            self.log(f"      FAILED {type(exc).__name__}: "
                     f"{str(exc).splitlines()[0][:100]}")
            return None
        finally:
            del rt, cm
            gc.collect()

    def measure(self, key: str, module: nn.Module, inputs: tuple,
                out_elems: int | None = None) -> float | None:
        """Device time for one shot: the layer, minus its own I/O baseline."""
        base = self._time(_IOBaseline(), inputs)
        real = self._time(module, inputs)
        if base is None or real is None:
            return None
        corrected = real["min_us"] - base["min_us"]
        in_elems = inputs[0].numel() if isinstance(inputs[0], torch.Tensor) else 0
        output_larger = bool(out_elems and in_elems and out_elems > in_elems * 1.5)
        # A layer cheaper than its own I/O baseline lands at or below zero. That
        # is a real outcome for the tiny elementwise shots, not a bug -- but the
        # subtraction is then doing all the work, so say so rather than emit a
        # confident number.
        baseline_dominated = corrected <= base["min_us"] * 0.05
        if corrected <= 0:
            corrected = 1e-3
            baseline_dominated = True
        self.notes.append({
            "key": key,
            "layer_min_us": round(real["min_us"], 3),
            "io_baseline_min_us": round(base["min_us"], 3),
            "time_us": round(corrected, 3),
            "layer_median_us": round(real["median_us"], 3),
            "layer_max_us": round(real["max_us"], 3),
            "jitter_ratio": round(real["median_us"] / real["min_us"], 3),
            "baseline_dominated": baseline_dominated,
            "output_larger_than_baseline": output_larger,
        })
        return corrected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "profiler" / "perf")
    parser.add_argument("--hardware", default="ATOM")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--max-kv", type=int, default=8192)
    parser.add_argument("--layers", default="", help="comma-separated subset, for smoke runs")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--kv-budget-gb", type=float, default=4.0)
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    if rebel.device_count() == 0:
        log("rebel.device_count() == 0 -- no ATOM visible. Check rbln-smi -j for "
            "a collapsed 'npu' index before suspecting packaging "
            "(docs/hardware_roadmap.md, 2026-08-28 update).")
        return 2

    cfg = load_model_config(args.model, args.tp)
    variant = "bf16"
    out_dir = args.out / args.hardware / args.model / variant / f"tp{args.tp}"
    log(f"=== ATOM layerwise profile on rbln{args.device} "
        f"({args.model}, tp{args.tp}) ===")
    log(f"  output: {out_dir}")

    measurer = Measurer(args.device, args.reps, log)

    wanted = {s for s in args.layers.split(",") if s}
    tokens_grid = x2_grid(1, args.max_tokens)
    seq_grid = x2_grid(1, args.max_seqs)

    dense_rows: list[dict] = []
    for name in DENSE_LAYERS:
        if wanted and name not in wanted:
            continue
        log(f"  dense {name}")
        for tokens in tokens_grid:
            try:
                module, factory = dense_layer(name, cfg)
                inputs = factory(tokens)
            except KeyError:
                break
            got = measurer.measure(f"dense/{name}/{tokens}", module, inputs)
            if got is not None:
                dense_rows.append({"layer": name, "tokens": tokens,
                                   "time_us": got})

    seq_rows: list[dict] = []
    for name in PER_SEQUENCE_LAYERS:
        if wanted and name not in wanted:
            continue
        log(f"  per_sequence {name}")
        for seqs in seq_grid:
            module, factory = per_sequence_layer(name, cfg)
            got = measurer.measure(f"per_sequence/{name}/{seqs}", module,
                                   factory(seqs))
            if got is not None:
                seq_rows.append({"layer": name, "sequences": seqs,
                                 "time_us": got})

    attn_rows: list[dict] = []
    if not args.skip_attention:
        budget = int(args.kv_budget_gb * 1024 ** 3)
        chunks = x2_grid(1, args.max_tokens)
        kv_prefills = [0, *x2_grid(128, args.max_kv)]
        n_decodes = x2_grid(1, args.max_seqs)
        kv_decodes = x2_grid(128, args.max_kv)
        log("  attention")
        for shot in attention_shots(chunks, kv_prefills, n_decodes, kv_decodes):
            if kv_bytes(cfg, shot) > budget:
                continue
            module, tensors = attention_case(cfg, shot)
            if not tensors:
                continue
            key = ("attention/"
                   f"{shot['prefill_chunk']}/{shot['kv_prefill']}/"
                   f"{shot['n_decode']}/{shot['kv_decode']}")
            got = measurer.measure(key, module, tensors)
            if got is not None:
                attn_rows.append({**shot, "time_us": got})

    out_dir.mkdir(parents=True, exist_ok=True)
    if dense_rows:
        write_csv(out_dir / "dense.csv", ("layer", "tokens", "time_us"), dense_rows)
    if seq_rows:
        write_csv(out_dir / "per_sequence.csv",
                  ("layer", "sequences", "time_us"), seq_rows)
    if attn_rows:
        write_csv(out_dir / "attention.csv",
                  ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode",
                   "time_us"), attn_rows)

    baseline_dominated = sum(1 for n in measurer.notes if n["baseline_dominated"])
    output_larger = sum(1 for n in measurer.notes if n["output_larger_than_baseline"])
    sidecar = out_dir / "measurement_notes.json"
    sidecar.write_text(json.dumps({
        "device": f"rbln{args.device}",
        "npu_name": rebel.get_npu_name(),
        "rebel_version": rebel.__version__,
        "reps": args.reps,
        "statistic": ("min over per-rep wall times, minus a PER-SHOT I/O "
                      "baseline measured on the same input shapes"),
        "shots": len(measurer.notes),
        "baseline_dominated_shots": baseline_dominated,
        "output_larger_than_baseline_shots": output_larger,
        "method": (
            "time_us is steady-state wall clock per rt.run() minus a measured "
            "its own I/O baseline. There is no device-span profiler on this "
            "stack: rebel._C.profiler emits protobuf traces whose schema is "
            "undocumented, and deriving microseconds from them would mean "
            "guessing a field layout and a clock rate. Shots flagged "
            "floor_dominated have a corrected time below the floor itself, "
            "where the subtraction dominates -- treat those as upper bounds. "
            "time_us is the layer's minimum per-rep wall time minus the minimum "
            "for an _IOBaseline graph run on the SAME input shapes. The baseline "
            "must be per-shot because rt.run() cost scales steeply with bytes "
            "crossing the host boundary -- 6.4 us at 16 B, 56.3 at 2 KB, 999.7 "
            "at 8 MB. An earlier version subtracted a single 6.5 us constant and "
            "inflated the elementwise layers 8-25x against RNGD's device spans. "
            "Rows flagged baseline_dominated are cheaper than their own I/O and "
            "are upper bounds; rows flagged output_larger_than_baseline have an "
            "output much bigger than the baseline's, so their output-side "
            "transfer is under-subtracted and they are also upper bounds. "
            "Utilisation is not sampled per shot -- see Measurer._time."
        ),
        "notes": measurer.notes,
    }, indent=2) + "\n")

    log(f"\n  dense {len(dense_rows)} rows, per_sequence {len(seq_rows)}, "
        f"attention {len(attn_rows)}")
    log(f"  {baseline_dominated}/{len(measurer.notes)} shots baseline-dominated, "
        f"{output_larger} with output larger than the baseline")
    log(f"  wrote {out_dir} (+ {sidecar.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
