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


# ---------------------------------------------------------------------------
# The EDF bundle is a DIFFERENT INSTRUMENT and must not inherit the harness's
# provenance text. It is built by experiments/scripts/rebuild_rngd_bundle_from_edf.py
# from FuriosaAI's own profiler, on the graph furiosa-llm actually serves.
# ---------------------------------------------------------------------------

EDF_MEASUREMENT_METHOD = (
    "FuriosaAI's own EDF profiler on the real served graph, NOT a layerwise "
    "harness. `furiosa-llm serve` was run on the vendor's prebuilt "
    "furiosa-ai/Llama-3.1-8B-Instruct artifact (tensor_parallel_size 8) with "
    "EDF_PROFILER_OUTPUT_PATH / TUC_PROFILE_LEVEL=info / RUST_LOG=span::tuc=info, "
    "and driven with 24 sharegpt requests at each of six concurrencies "
    "(1, 2, 4, 8, 16, 32) - 1.74 M stage executions in total. The trace is CSV "
    "(leader_device, name, cycle); cycles are converted at 1.6 GHz, derived as "
    "total device cycles over wall time on a saturated card (1,599.9 MHz, 0.006% "
    "off a round figure). Each row is one compiler-emitted stage execution named "
    "by its compiled bucket, so these are the times the vendor's own kernels take "
    "on the vendor's own graph, WITH THE INTRA-CARD REDUCTION ALREADY INSIDE "
    "THEM - which is the whole point of the card-as-device mapping below. "
    "The runtime has TWO compiled plans and both are captured: a fully-fused "
    "`Composed` graph for batch-1 decode (9 segments partitioning the stack, each "
    "executing once per forward) and a per-layer `Tokenwise` + `Attention` plan "
    "for batch >= 2 (exactly 32 Tokenwise executions per forward, one per decoder "
    "layer). There is no `input_size: 1` Tokenwise bucket anywhere in the traces "
    "because at batch 1 the bucketed path is not used at all."
)

EDF_NOTES = (
    "WHAT IS MEASURED AND WHAT IS INHERITED - the one thing to read before using "
    "this bundle. (1) MEASURED: absolute per-decoder-layer latency, per token "
    "bucket, on the real graph; decode and prefill attention; the head. "
    "(2) INHERITED, NOT MEASURED: the split of a decoder layer's stage time "
    "across the canonical layer names (layernorm / qkv_proj / rotary_emb / "
    "o_proj / gate_up_proj / act_fn / down_proj). The compiler fuses a whole "
    "decoder layer into ONE stage and does not expose the pieces, so the "
    "magnitude comes from the vendor and the distribution is taken from the "
    "per-PE harness bundle's shares at the same token count, rescaled to sum to "
    "the vendor figure. The simulator only ever SUMS the per-layer lookups for an "
    "iteration, so the sum is what has to be right - but no single row here is a "
    "vendor measurement of that named layer. The vendor/harness ratio is 1.16x to "
    "1.65x depending on bucket and is recorded per bucket in "
    "outputs/rngd_edf_bundle/edf_vs_harness_dense.csv. "
    "(3) The tokens=1 row comes from the fused Composed graph, corrected three "
    "ways: its 9 segment times overlap (device cycles are 114.7% of wall at "
    "concurrency 1), so they are scaled so their sum equals the measured "
    "wall-clock forward; the terminal segment is moved to per_sequence.csv "
    "because it is the head, not a layer; and the measured batch-1 attention is "
    "subtracted so the simulator's separate attention charge is not counted "
    "twice. (4) DECODE ATTENTION IS CALIBRATED TO THIS TRAFFIC MIX. The runtime "
    "groups a decode batch by kv bucket, so per-layer attention depends on the "
    "batch's kv DIVERSITY (1.95 executions per layer at batch 2, 3.08 at batch "
    "29), which the 4D contract cannot express. Each concurrency therefore "
    "contributes one row whose time is total decode-attention device time over "
    "(forwards x 32), which makes the total close but ties the decode attention "
    "axis to sharegpt-like traffic at mean kv ~2200. "
    "(5) per_sequence.csv takes its magnitude at 1 sequence from the vendor's "
    "terminal segment and its SHAPE over sequence count from the harness, which "
    "is the only source for how the head scales with batch. "
    "(6) MIXED prefill+decode steps are not in the traces at all: every attention "
    "bucket is pure prefill or pure decode, so the 4D grid has data only on the "
    "two axis planes and the simulator's nearest-slice fallback approximates the "
    "interior, which under-counts continuous-batching steps that do both. "
    "(7) No skew.csv / skew_fit.csv, so ragged decode batches fall back to the "
    "uniform-mean attention path. "
    "SELF-CONSISTENCY, the check that this bundle passes and the synthetic one "
    "could not: predicted wall time from this bundle against measured wall time "
    "over a 32x concurrency range is +0.0% / +0.1% / +0.5% / -4.7% / -5.2% / "
    "+0.2% at concurrency 1 / 2 / 4 / 8 / 16 / 32."
)

EDF_CARD_LEVEL_NOTE = (
    " CARD-LEVEL MAPPING, and why it is legitimate HERE where it was not before. "
    "One accelerator is one whole RNGD card running at the vendor's default "
    "-tp 8, so the card is a tp1 instance and the simulator adds no collective "
    "for it. With the per-PE harness bundle that was a fatal omission - the "
    "intra-card all-reduce was simply missing, and decode came out 45.5% fast. "
    "Here it is correct BY CONSTRUCTION: an EDF stage time is what the card took "
    "to execute that stage, reduction included, so the measurement has already "
    "paid for the communication and charging it again would double-count. The "
    "granularity also now matches the hardware: the artifact's "
    "tensor_parallel_size 8 is realised as TWO FUSED 4-PE QUADS (leader_device is "
    "npu0pe0-3, and the serve log confirms DpId(0) -> [npu0pe0-3, npu0pe4-7]), "
    "not eight ranks, so a per-PE tp8 bundle modelled a rank granularity that "
    "does not exist. EDF reports only one leader device, and the two quads run "
    "concurrently, so these times are one card's. "
    "max_tp_size stays 1: TP ACROSS cards has never been built or served here."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("outputs/rngd_profile"))
    parser.add_argument("--hardware", default="RNGD")
    parser.add_argument("--edf", action="store_true",
                        help="the bundle was built from FuriosaAI's EDF profiler by "
                             "rebuild_rngd_bundle_from_edf.py, not from the layerwise "
                             "harness; swaps in the matching provenance")
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
            "HeteroPilot, built from FuriosaAI's EDF profiler traces of "
            "`furiosa-llm serve` on this NPU server's RNGD npu0, via "
            "experiments/scripts/rebuild_rngd_bundle_from_edf.py"
            if args.edf else
            "HeteroPilot, measured on this NPU server's FuriosaAI RNGD cards via "
            "experiments/scripts/run_rngd_profile.py (24 workers, one PE each)"
        ),
        source="measured",
        serving_stack="furiosa-llm 2026.2.0 / furiosa-torch 2026.2.0 (torch 2.10.0)",
        runtime_version="RNGD firmware 2026.3.0, furiosa-smi 2026.1.2",
        backend="furiosa",
        device=("FuriosaAI RNGD card npu0, two fused 4-PE quads "
                "(npu0pe0-3, npu0pe4-7) at tensor_parallel_size 8"
                if args.edf else device or "FuriosaAI RNGD (rngd PE)"),
        measurement_method=EDF_MEASUREMENT_METHOD if args.edf else MEASUREMENT_METHOD,
        measurement_iterations=iterations,
        notes=(EDF_NOTES + (EDF_CARD_LEVEL_NOTE if args.card_level else ""))
        if args.edf
        else NOTES + (CARD_LEVEL_NOTE if args.card_level else ""),
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
