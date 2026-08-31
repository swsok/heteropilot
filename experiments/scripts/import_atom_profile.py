"""Import the measured ATOM CSV bundle into ``profiler/perf/`` and report it.

Runs in the planner venv (``.venv``), not a vendor one: it only reads CSVs, so
it needs neither the device nor ``rebel``. ``profiler/`` is not modified -- this
drives the existing Phase 3 ``CsvProfileImporter``.

The measurement itself is ``experiments/scripts/profile_atom.py``; this step
validates the bundle against ``profiler/CONTRACT.md`` and writes ``meta.yaml``
with attribution.

Usage::

    PYTHONPATH=$PWD .venv/bin/python experiments/scripts/import_atom_profile.py \\
        --src outputs/atom_profile/bundle --tp 1
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
    "Layerwise on-device measurement via rebel (Rebellions' compiler/runtime), "
    "not vLLM. The rbln vLLM platform plugin exists and activates, but the "
    "profiler's two non-negotiable engine settings each collide with one of "
    "vllm-rbln's two execution paths: load_format='dummy' is rejected by the "
    "default optimum path, which AOT-compiles a REAL checkpoint and demands "
    "safetensors; and enforce_eager=True is rejected by the vLLM-native path "
    "unless VLLM_RBLN_USE_DEVICE_TENSOR=1, which needs a torch device named "
    "'rbln' that nothing on this machine registers (there is no torch_rbln, "
    "unlike furiosa.torch registering rngd: as PrivateUse1). That native path "
    "also registers only deepseek_v2/gpt_oss/minimax_m2/qwen2/qwen3 -- not "
    "llama. Working around either would mean editing profiler/, which is "
    "upstream and pristine until Phase 5. So each canonical layer of "
    "profiler/models/llama.yaml is compiled on its own with "
    "rebel.compile_from_torch and run on one card. "
    "time_us is the MINIMUM per-rep wall time minus a measured host round-trip "
    "floor (6.49 us, calibrated per run with a trivial graph). The minimum is "
    "used because every rt.run() is a host round trip whose cost is only ever "
    "inflated by jitter; the median and max travel with every row in "
    "measurement_notes.json so a noisy shot is visible. There is no device-span "
    "path: rebel._C.profiler emits protobuf traces (with RBLN_PROFILER=1) "
    "carrying comp_cycle and transfer records, but the schema is undocumented "
    "and deriving microseconds would mean guessing a field layout and a clock "
    "rate. TP=N would be measured on a single card with intermediate_size / "
    "num_attention_heads / num_key_value_heads / vocab_size divided by N, "
    "mirroring profiler/core/engine.py; only tp1 is measured so far."
)

NOTES = (
    "IMPORTANT provenance caveat, identical in kind to the RNGD bundle's: these "
    "are real measurements on real ATOM silicon, but of layer implementations "
    "written for this harness (experiments/scripts/profile_atom.py, sharing "
    "experiments/scripts/llama_layers.py with the RNGD harness), NOT of vLLM's "
    "fused kernels the way the A40 / A5000 / RTXPRO6000 bundles were measured. "
    "A GPU-vs-ATOM comparison built on this compares the same mathematics "
    "executed by different software stacks. ATOM-vs-RNGD is the cleaner "
    "comparison, because both NPU bundles come from the same shared layer "
    "definitions. "
    "SECOND CAVEAT: time_us carries whatever host cost survives the floor "
    "subtraction, so it is an upper bound on pure device time. One shot "
    "(embedding at 1 token) is flagged floor_dominated in measurement_notes.json "
    "-- its corrected time is below the floor itself, so the subtraction "
    "dominates and that row should be read as an upper bound rather than a "
    "measurement. "
    "THIRD: no skew.csv, so the simulator falls back to uniform-batch attention "
    "and under-represents ragged decode batches -- the same gap the RNGD bundle "
    "has. "
    "Unlike the RNGD bundle, there are NO missing dense points: act_fn@1 and "
    "rotary_emb@2048, which the Furiosa compiler kept on CPU reproducibly, both "
    "compile and run on ATOM across the whole grid."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=REPO_ROOT / "outputs" / "atom_profile" / "bundle")
    parser.add_argument("--hardware", default="ATOM")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--tp", type=int, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate only; write nothing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tp_degrees = args.tp or [1]

    notes_path = args.src / f"tp{tp_degrees[0]}" / "measurement_notes.json"
    iterations = None
    device = "Rebellions ATOM RBLN-CA22 (rbln0)"
    if notes_path.is_file():
        sidecar = json.loads(notes_path.read_text())
        iterations = sidecar.get("reps")
        device = (f"Rebellions {sidecar.get('npu_name', 'ATOM')} "
                  f"({sidecar.get('device', 'rbln0')})")

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
        measured_by=("HeteroPilot, measured on this NPU server's Rebellions ATOM "
                     "cards via experiments/scripts/profile_atom.py"),
        source="measured",
        serving_stack=("rebel-compiler 0.11.0 / optimum-rbln 0.11.0.post1 "
                       "(torch 2.9.1+cpu), in .venv-rbln"),
        runtime_version="ATOM KMD/firmware 3.0.0, rbln-smi",
        backend="rbln",
        device=device,
        measurement_method=MEASUREMENT_METHOD,
        measurement_iterations=iterations,
        notes=NOTES,
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
