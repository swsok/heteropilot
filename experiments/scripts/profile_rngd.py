"""Layerwise profiler for FuriosaAI RNGD, emitting a §3.7 CSV bundle.

Why this exists
---------------
RNGD does not support vLLM: the only ``vllm.platform_plugins`` entry point on
this machine is ``rbln``, and ``furiosa-llm`` is a separate stack (``build``
compiles a bucketed AOT artifact, ``serve`` serves it). So
``profiler/core/engine.py``'s ``from vllm import LLM`` can never drive an RNGD,
and ``python -m profiler`` is unusable for it.

But ``furiosa.torch`` exposes a torch.compile backend, a PrivateUse1 device
(``rngd:<pe>``) and a torch-profiler-compatible ``RNGDProfiler``, which is
enough to measure per-layer device time directly. This script does that and
writes CSVs for ``profiler.core.importer.CsvProfileImporter``, so ``profiler/``
itself stays untouched (Phase 3 V1, ``docs/hardware_roadmap.md``).

How the timing works
--------------------
Eager mode cannot be timed: ``furiosa.torch.backend.eager.run_aten_op``
JIT-compiles every aten op per call and calls ``run_by_rngd`` *without* a
``profiler=``, so no device spans exist and the wall time is the compiler's.
Only ``torch.compile(m, backend=furiosa.torch.backend)`` inside
``furiosa.torch.config.profiler_context(RNGDProfiler())`` records device spans
(``backend/torch_compile.py`` passes ``profiler=config.get_profiler()``).

Those spans are hardware-unit-level, not op-named: ``Renegade::TuExec`` (tensor
unit), ``DMA (n)``, ``Task``. They therefore cannot be attributed back to
individual layers inside one big graph. So this profiler compiles **one
canonical layer per graph** and sums that graph's spans -- which is exactly the
contract's ``layer, tokens, time_us``.

What "TP=N" means here
----------------------
Mirrors ``profiler/core/engine.py``: every TP degree is measured on a *single*
device with the model config's ``SHARD_FIELDS`` (intermediate_size,
num_attention_heads, num_key_value_heads, vocab_size) divided by N, so the
kernel shapes match what one rank of a real ``tp=N`` deployment sees.
Collectives are not measured -- ASTRA-Sim adds them analytically.

A rank is **one PE**, not one card: ``furiosa-llm build -tp`` counts PEs per
tensor-parallel group and defaults to 8, i.e. a full card is a TP-8 group.

Provenance (absolute rule 3)
----------------------------
These are real measurements on real RNGD silicon, but of **layer
implementations written for this harness**, not of vLLM's fused kernels the way
the A40/A5000/RTXPRO6000 bundles were. That difference is recorded in the
emitted ``notes`` and must survive into every comparison built on this bundle.

Usage::

    PYTHONPATH=$PWD python3 experiments/scripts/profile_rngd.py \
        --model meta-llama/Llama-3.1-8B --tp 1 --device rngd:24 \
        --out outputs/rngd_profile
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

# isort: off
# furiosa.torch MUST be imported after torch. It imports torch itself, and
# torch's backend autoload then calls furiosa.torch._register while that module
# is still initialising, which fails with "partially initialized module
# 'furiosa.torch' has no attribute '_register' (most likely due to a circular
# import)". isort sorts furiosa before torch alphabetically, so the ordering has
# to be pinned here or `ruff check --fix` silently breaks every run.
import furiosa.torch as ft
from furiosa.torch import config as furiosa_config
from furiosa.torch.profiler import RNGDProfiler
# isort: on

# The canonical layers, the model-config sharding and the CSV writers live in
# llama_layers so this harness and profile_atom.py measure the SAME layers --
# Exp 4 compares their outputs, so a drift between two copies would silently
# invalidate it.
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

#: Tensor-unit execution spans.
COMPUTE_SPAN = "Renegade::TuExec"

#: On-device data movement spans, ``DMA (n)``. These are genuine layer work,
#: not harness overhead: DMA bytes track what the op must actually read, which
#: is the whole weight for a matmul (~5.2 us/MB, ~200 GB/s: o_proj 32 MB ->
#: 173 us, down_proj 112 MB -> 555 us, gate_up_proj 224 MB -> 1206 us) but only
#: the gathered rows for an embedding (flat ~21 us at any token count, because
#: ``index_select`` never streams the 1 GB table).
DMA_SPAN_PREFIX = "DMA"

#: ``time_us`` is the **union** of the device spans above, i.e. wall time the
#: device was busy, NOT their sum. Summing overcounts twice over, as measured
#: on down_proj @256 tokens in one forward: TuExec sum 1603 us, DMA sum 964 us,
#: naive total 2567 us -- but the union is 1250 us, and the whole timeline from
#: first span start to last span end is also 1250 us. So the tensor units run
#: several at once (the union is smaller than the TuExec sum alone) and DMA
#: overlaps compute, while the device is busy end to end. The union is
#: therefore both the honest device latency and the right analogue of the
#: single-stream ``cuda_time_us`` the GPU bundles record. It also makes
#: embedding measurable without a special case: a DMA-only op still has a
#: non-empty union. ``breakdown_*.csv`` keeps the sums beside it.

#: Skip an attention shot whose K+V tensors would exceed this many bytes. One
#: PE gets roughly an eighth of the card's 47.5 GiB, and the harness holds the
#: whole KV densely rather than paged. Skips are logged, never silent.
DEFAULT_KV_BUDGET_BYTES = 4 * 1000**3


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

#: Divided by the TP degree, exactly as ``profiler/core/config.py`` does.
# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class Measurer:
    """Compile one module per shape and sum its device spans."""

    def __init__(self, device: str, reps: int, trace_dir: Path) -> None:
        self.device = device
        self.reps = reps
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def measure(self, module: nn.Module, inputs: tuple) -> dict[str, float]:
        """Return per-forward device microseconds for ``module(*inputs)``.

        Device tensors are released before returning. Without that, one process
        walking a grid accumulates every shot's inputs on the PE and the large
        shapes die with ``OutOfMemoryError: failed to allocate ... on DRAM``
        even when the shot itself would fit -- a PE addresses only ~6.25 GB
        (measured, ``rngd_device_facts.py``).
        """
        module = module.to(self.device)
        dev_inputs = tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t for t in inputs
        )
        compiled = torch.compile(module, backend=ft.backend)
        try:
            with torch.no_grad():
                compiled(*dev_inputs)  # compile + warm this shape
                prof = RNGDProfiler()
                with furiosa_config.profiler_context(prof), prof:
                    for _ in range(self.reps):
                        compiled(*dev_inputs)
        finally:
            del compiled, dev_inputs
            module.to("cpu")
            # Dropping the wrapper is not enough: dynamo caches the compiled
            # graph, which keeps the loaded EDF -- and therefore its device
            # buffers -- alive. Without the reset, a large shot that succeeds
            # leaves the PE too full for the smaller ones after it (measured:
            # n_decode=128 kv=8192 succeeded, then n_decode=8 kv=8192 failed to
            # allocate 134 MB).
            torch._dynamo.reset()
            gc.collect()
        self._n += 1
        path = self.trace_dir / f"trace_{self._n:05d}.json"
        prof.export_chrome_trace(str(path))
        spans = self._sum_spans(path)
        path.unlink(missing_ok=True)
        return spans

    def _sum_spans(self, path: Path) -> dict[str, float]:
        raw = json.loads(path.read_text())
        # RNGDProfiler.export_chrome_trace emits a bare list when it has device
        # spans to merge, and falls through to torch's standard
        # {"traceEvents": [...]} object when it has none. Handle both, then
        # refuse a shot that produced no device span at all -- recording 0 us
        # would look like a fast layer instead of a failed measurement.
        events = raw if isinstance(raw, list) else raw.get("traceEvents", [])
        compute_sum = dma_sum = 0.0
        intervals: list[tuple[float, float]] = []
        for event in events:
            if not isinstance(event, dict) or event.get("ph") != "X":
                continue
            name = event.get("name", "")
            duration = event.get("dur", 0)
            if name == COMPUTE_SPAN:
                compute_sum += duration
            elif name.startswith(DMA_SPAN_PREFIX):
                dma_sum += duration
            else:
                continue
            intervals.append((event["ts"], event["ts"] + duration))
        if not intervals:
            raise RuntimeError(
                f"no device span recorded (trace had {len(events)} events) -- "
                f"the graph probably fell back to CPU"
            )
        return {
            "compute_us": compute_sum / self.reps,
            "dma_us": dma_sum / self.reps,
            "time_us": _union_us(intervals) / self.reps,
        }


def _union_us(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of ``(start, end)`` span intervals.

    Overlapping spans are collapsed, so concurrent tensor units and
    compute/DMA overlap are counted once. Reps are separated in time, so the
    union over a whole trace is the sum of the per-rep unions.
    """
    total = 0.0
    current_start, current_end = None, None
    for start, end in sorted(intervals):
        if current_end is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_end is not None:
        total += current_end - current_start
    return total


def card_of(device: str) -> str:
    """``rngd:<pe>`` -> ``npu<card>``; 8 PEs per card."""
    return f"npu{int(device.split(':')[1]) // 8}"


def read_power_w(device: str) -> float | None:
    """Board power of the *profiled* card only.

    Reading the maximum across cards would pick up the co-tenant pods on
    npu0/1/2 (``docs/hardware_roadmap.md``, "Who holds the NPUs"), so the row
    is matched on the device name.
    """
    want = card_of(device)
    try:
        raw = subprocess.run(
            ["furiosa-smi", "info"], capture_output=True, text=True, timeout=60
        ).stdout
    except Exception:
        return None
    for line in raw.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if want not in cells:
            continue
        for cell in cells:
            if cell.endswith("W"):
                try:
                    return float(cell[:-1].strip())
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def task_cost(task: tuple) -> float:
    """Cheap proxy for how long a task will take, used only to balance shards."""
    kind, _layer, size = task
    if kind == "attention":
        chunk, kv_p = size["prefill_chunk"], size["kv_prefill"]
        n_dec, kv_d = size["n_decode"], size["kv_decode"]
        return float(chunk * (kv_p + chunk) + n_dec * kv_d)
    return float(size)


def build_tasks(tokens_grid, seq_grid, shots, dense_layers, per_seq_layers) -> list[tuple]:
    """One flat, deterministic work list, ordered so striding balances load.

    Sorted by descending cost proxy, so ``tasks[shard::num_shards]`` hands each
    worker one of the most expensive tasks and then progressively cheaper ones
    -- longest-processing-time-first, which is what keeps 24 workers finishing
    together.

    Do NOT go back to plain layer-major order: with the size axis innermost, a
    worker count that is a multiple of the grid size aligns the stride to the
    grid period, and one unlucky shard draws the largest shape of *every* layer.
    That happened with 24 workers over a 12-point token grid: shard 23 got every
    2048-token task and ran ~4x longer than the rest.
    """
    tasks: list[tuple] = []
    for layer in dense_layers:
        for tokens in tokens_grid:
            tasks.append(("dense", layer, tokens))
    for layer in per_seq_layers:
        for seqs in seq_grid:
            tasks.append(("per_sequence", layer, seqs))
    for shot in shots:
        tasks.append(("attention", None, shot))
    # Stable tie-break on the printable form keeps the order reproducible.
    tasks.sort(key=lambda task: (-task_cost(task), task[0], str(task[1]), str(task[2])))
    return tasks


def run_tasks(cfg, measurer, tasks, log, kv_budget=DEFAULT_KV_BUDGET_BYTES):
    """Measure every task in ``tasks``, reusing one module per layer."""
    dense_rows: list[dict] = []
    seq_rows: list[dict] = []
    attn_rows: list[dict] = []
    skipped: list[dict] = []
    cache: dict[tuple[str, str], tuple] = {}

    for kind, layer, size in tasks:
        started = time.time()
        if kind == "attention":
            needed = kv_bytes(cfg, size)
            if needed > kv_budget:
                skipped.append({**size, "kv_bytes": needed})
                log(f"  attn SKIP (KV {needed / 1000**3:.1f} GB > budget "
                    f"{kv_budget / 1000**3:.1f} GB) {size}")
                continue
            module, inputs = attention_case(cfg, size)
            label = (f"attn pc={size['prefill_chunk']:<6d} kvp={size['kv_prefill']:<6d} "
                     f"nd={size['n_decode']:<5d} kvd={size['kv_decode']:<6d}")
        else:
            key = (kind, layer)
            if key not in cache:
                cache[key] = (dense_layer(layer, cfg) if kind == "dense"
                              else per_sequence_layer(layer, cfg))
            module, make_inputs = cache[key]
            inputs = make_inputs(size)
            label = f"{kind:12s} {layer:16s} size={size:<6d}"

        try:
            spans = measurer.measure(module, inputs)
        except Exception as exc:  # a compiler refusal is data, not a crash
            log(f"  {label} FAILED {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:110]}")
            continue

        if kind == "dense":
            dense_rows.append({"layer": layer, "tokens": size, **spans})
        elif kind == "per_sequence":
            seq_rows.append({"layer": layer, "sequences": size, **spans})
        else:
            attn_rows.append({**size, **spans})
        log(f"  {label} compute={spans['compute_us']:9.2f}us "
            f"dma={spans['dma_us']:8.2f}us time={spans['time_us']:9.2f}us "
            f"({time.time() - started:.0f}s)")

    return dense_rows, seq_rows, attn_rows, skipped


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--device", default="rngd:24")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("outputs/rngd_profile"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--max-kv", type=int, default=8192)
    parser.add_argument("--kv-decodes", default="",
                        help="comma-separated kv_decode values; default 128,1024,max-kv")
    parser.add_argument("--layers", default="", help="comma-separated subset, for smoke runs")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--shard", type=int, default=0,
                        help="this worker's index into the work list")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="total workers; each takes every num-shards-th task")
    parser.add_argument("--kv-budget-gb", type=float,
                        default=DEFAULT_KV_BUDGET_BYTES / 1000 ** 3,
                        help="per-worker KV cap for attention shots")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} out of range for "
                         f"--num-shards {args.num_shards}")

    cfg = load_model_config(args.model, args.tp)
    out_dir = args.out / f"tp{args.tp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / f"profile_tp{args.tp}.log"
    log_handle = log_path.open("a")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    subset = [s for s in args.layers.split(",") if s] or None
    dense = [x for x in DENSE_LAYERS if subset is None or x in subset]
    per_seq = [x for x in PER_SEQUENCE_LAYERS if subset is None or x in subset]

    log(f"=== RNGD profile: {args.model} tp={args.tp} on {args.device} "
        f"(shard {args.shard}/{args.num_shards}) ===")
    log(f"sharded config: hidden={cfg['hidden_size']} heads={cfg['num_attention_heads']} "
        f"kv_heads={cfg['num_key_value_heads']} inter={cfg['intermediate_size']} "
        f"vocab={cfg['vocab_size']} head_dim={cfg['head_dim']}")
    idle_w = read_power_w(args.device)
    log(f"board power of {card_of(args.device)} before run: {idle_w} W")

    measurer = Measurer(args.device, args.reps, args.out / "traces")
    tokens_grid = x2_grid(1, args.max_tokens)
    seq_grid = x2_grid(1, args.max_seqs)

    shots = [] if args.skip_attention else list(attention_shots(
        prefill_chunks=x2_grid(16, args.max_tokens),
        kv_prefills=[0],
        n_decodes=x2_grid(1, args.max_seqs),
        kv_decodes=([int(v) for v in args.kv_decodes.split(",") if v]
                    or [128, 1024, args.max_kv]),
    ))
    all_tasks = build_tasks(tokens_grid, seq_grid, shots, dense, per_seq)
    tasks = all_tasks[args.shard::args.num_shards]
    log(f"work list: {len(all_tasks)} task(s) total, {len(tasks)} in this shard")

    started = time.time()
    dense_rows, seq_rows, attn_rows, attn_skipped = run_tasks(
        cfg, measurer, tasks, log, kv_budget=int(args.kv_budget_gb * 1000 ** 3))
    busy_w = read_power_w(args.device)
    elapsed = time.time() - started

    write_csv(out_dir / "dense.csv", ("layer", "tokens", "time_us"), dense_rows)
    write_csv(out_dir / "per_sequence.csv", ("layer", "sequences", "time_us"), seq_rows)
    write_csv(out_dir / "attention.csv",
              ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"), attn_rows)
    # Span breakdown kept OUTSIDE tp<N>/ so the bundle CsvProfileImporter sees
    # holds exactly the three contract CSVs, while the compute/DMA sums behind
    # every union-based time_us stay auditable.
    write_csv(args.out / f"breakdown_dense_tp{args.tp}.csv",
              ("layer", "tokens", "compute_us", "dma_us", "time_us"), dense_rows)
    write_csv(args.out / f"breakdown_per_sequence_tp{args.tp}.csv",
              ("layer", "sequences", "compute_us", "dma_us", "time_us"), seq_rows)
    write_csv(args.out / f"breakdown_attention_tp{args.tp}.csv",
              ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode",
               "compute_us", "dma_us", "time_us"), attn_rows)

    summary = {
        "model": args.model, "tp": args.tp, "device": args.device,
        "shard": args.shard, "num_shards": args.num_shards,
        "tasks_in_shard": len(tasks), "tasks_total": len(all_tasks),
        "reps": args.reps, "elapsed_s": round(elapsed, 1),
        "rows": {"dense": len(dense_rows), "per_sequence": len(seq_rows),
                 "attention": len(attn_rows)},
        "attention_shots_skipped_over_kv_budget": attn_skipped,
        "power_w": {"before": idle_w, "after": busy_w},
        "sharded_config": {k: cfg[k] for k in
                           ("hidden_size", "num_attention_heads", "num_key_value_heads",
                            "intermediate_size", "vocab_size", "head_dim")},
    }
    (args.out / f"summary_tp{args.tp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done in {elapsed:.0f}s -> {out_dir}")
    log(json.dumps(summary["rows"]))


if __name__ == "__main__":
    main()
