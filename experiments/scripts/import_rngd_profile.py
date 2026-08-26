"""Import the measured RNGD CSV bundle into ``profiler/perf/`` and report it.

Runs in the planner venv (``.venv``), not the furiosa one: it only reads CSVs,
so it needs neither the device nor the vendor runtime. ``profiler/`` is not
modified -- this drives the existing Phase 3 ``CsvProfileImporter``.

The measurement itself is ``experiments/scripts/profile_rngd.py``; this step
validates the bundle against ``profiler/CONTRACT.md`` and writes ``meta.yaml``
with attribution.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/scripts/import_rngd_profile.py \
        --src outputs/rngd_profile --tp 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiler.core.importer import (
    CsvProfileImporter,
    ImportProvenance,
    ProfileContractError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MEASUREMENT_METHOD = (
    "Layerwise on-device measurement via furiosa.torch, not vLLM (RNGD has no "
    "vLLM platform plugin). Each canonical layer of profiler/models/llama.yaml "
    "is compiled on its own through torch.compile(backend=furiosa.torch.backend) "
    "inside furiosa.torch.config.profiler_context(RNGDProfiler()), which records "
    "device spans (Renegade::TuExec and DMA). time_us is the UNION of those "
    "spans -- wall time the device was busy -- because the tensor units run "
    "concurrently and DMA overlaps compute, so summing them overcounts (measured "
    "on down_proj@256: TuExec sum 1603 us + DMA sum 964 us, but union 1250 us, "
    "equal to the full span of the timeline). TP=N is measured on a single PE "
    "with intermediate_size / num_attention_heads / num_key_value_heads / "
    "vocab_size divided by N, mirroring profiler/core/engine.py, so the shapes "
    "are one rank's; collectives are left to ASTRA-Sim."
)

NOTES = (
    "IMPORTANT provenance caveat: these are real measurements on real RNGD "
    "silicon, but of layer implementations written for this harness "
    "(experiments/scripts/profile_rngd.py), NOT of vLLM's fused kernels the way "
    "the A40 / A5000 / RTXPRO6000 bundles were measured. A GPU-vs-RNGD "
    "comparison built on this bundle compares the same mathematics executed by "
    "different software stacks, and must say so. Coverage gaps, all deliberate "
    "and none silent: skew.csv / skew_fit.csv are absent, so the simulator falls "
    "back to uniform-batch attention and under-represents ragged decode batches; "
    "dense.csv is missing act_fn@1 token and rotary_emb@2048 because the "
    "compiler keeps those two shapes on the CPU reproducibly (retried, same "
    "result), and the simulator clamps below a layer's lowest key; attention.csv "
    "is missing exactly one shot, n_decode=256 x kv_decode=8192, whose K+V is "
    "8.6 GB against the 6.25 GB a single PE can address -- a real device limit, "
    "not a harness cap. A rank here is one PE, not one card: furiosa-llm build "
    "-tp counts PEs per TP group and defaults to 8, so a full card is a TP-8 "
    "group."
)


CARD_LEVEL_NOTE = (
    " CARD-LEVEL MAPPING: this bundle's tp1/ IS the measured tp8/ of the per-PE "
    "bundle, because one accelerator here is one whole RNGD CARD running at the "
    "vendor's default -tp 8. The 8 PEs each execute 1/8 of every layer in "
    "parallel, so the card's per-layer LATENCY equals one PE's sharded latency - "
    "which is exactly what tp8/ measured. Why model it this way: TP=8 is the only "
    "configuration that both uses all 8 PEs and keeps the weights sharded once "
    "(1.87 GiB/PE), so it holds 246,079 KV tokens against 123,550 for tp4xdp2 on "
    "the same silicon, and tp2xdp4 does not fit at all. It is also the only "
    "deployable configuration: furiosa-llm build -tp defaults to 8 and the "
    "vendor's prebuilt artifact is tensor_parallel_size 8. "
    "THE COST, stated plainly: a tp1 instance has no TP group, so the simulator "
    "adds NO collective for the intra-card all-reduce after o_proj and down_proj. "
    "That communication is real and is NOT in these numbers; it is absorbed into "
    "whatever gap remains against the real furiosa-llm benchmark, and that gap is "
    "the honest bound on it. "
    "INPUT QUALITY: tp8 is the sparsest measured grid - 81/13/27 rows against "
    "107/18/35 at tp4 - because sharding the dims by 8 (intermediate 1792, 4 "
    "heads, 1 KV head) makes shapes small enough that the compiler stops using "
    "the tensor unit. Every layer still holds 8-10 of 12 token points, so "
    "interpolation works, but this bundle is coarser than the per-PE tp4 one."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("outputs/rngd_profile"))
    parser.add_argument("--hardware", default="RNGD")
    parser.add_argument("--card-level", action="store_true",
                        help="the bundle models a whole card as ONE accelerator at tp1; "
                             "adds the mapping and its cost to the notes")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--tp", type=int, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate only, write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing bundle; needed when adding TP degrees")
    args = parser.parse_args()

    tp_degrees = args.tp or [1]
    summary_path = args.src / f"summary_tp{tp_degrees[0]}.json"
    iterations = None
    device = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        iterations = summary.get("reps")
        # The parallel driver records every PE it used; a single-worker run
        # records one device.
        devices = summary.get("devices") or ([summary["device"]] if summary.get("device") else [])
        if devices:
            device = f"FuriosaAI RNGD, PEs {', '.join(devices)}"

    importer = CsvProfileImporter()
    try:
        report = importer.validate(
            args.src,
            hardware=args.hardware,
            model=args.model,
            variant=args.variant,
            tp_degrees=tp_degrees,
        )
    except ProfileContractError as exc:
        raise SystemExit(f"contract violation, nothing written:\n  {exc}") from exc

    print("validation passed")
    for tp, tp_report in sorted(report.tp_reports.items()):
        for file_report in tp_report.files.values():
            if not file_report.present:
                continue
            print(f"  tp{tp} {file_report.filename:20s} rows={file_report.rows}")
            for axis, span in sorted((file_report.coverage or {}).items()):
                print(f"        {axis:16s} min={span.get('min')} max={span.get('max')} "
                      f"n_unique={span.get('n_unique')}")

    if args.dry_run:
        print("dry run: nothing written")
        return

    provenance = ImportProvenance(
        measured_by=(
            "HeteroPilot, measured on this NPU server's FuriosaAI RNGD cards via "
            "experiments/scripts/run_rngd_profile.py (24 workers, one PE each)"
        ),
        source="measured",
        serving_stack="furiosa-llm 2026.2.0 / furiosa-torch 2026.2.0 (torch 2.10.0)",
        runtime_version="RNGD firmware 2026.3.0, furiosa-smi 2026.1.2",
        backend="furiosa",
        device=device or "FuriosaAI RNGD (rngd PE)",
        measurement_method=MEASUREMENT_METHOD,
        measurement_iterations=iterations,
        notes=NOTES + (CARD_LEVEL_NOTE if args.card_level else ""),
    )
    dest = importer.import_bundle(
        src=args.src,
        hardware=args.hardware,
        model=args.model,
        variant=args.variant,
        tp_degrees=tp_degrees,
        provenance=provenance,
        overwrite=args.overwrite,
    )
    print(f"imported -> {dest}")


if __name__ == "__main__":
    main()
