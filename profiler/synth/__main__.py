"""CLI for synthetic (Tier 0/1) profile bundles.

    python -m profiler.synth emit \\
        --accelerator profiles/accelerators/ascend_target.yaml \\
        --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1,2 \\
        --hardware-label ASCEND_TARGET-t0 --out profiler/perf

Subcommands `calibrate` / `pick-anchors` / `diff` arrive in STEPs 8-9.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planner.inventory import load_accelerator_profile
from profiler.core.config import load_architecture
from profiler.synth.backend import AnalyticalProfileBackend
from profiler.synth.device import DeviceSpec
from profiler.synth.dims import ModelDims
from profiler.synth.emit import BundleEmitter, GridParams

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_emit_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("emit", help="Generate a Tier 0 bundle from datasheet values.")
    p.add_argument("--accelerator", required=True, type=Path,
                   help="profiles/accelerators/*.yaml with a datasheet: block")
    p.add_argument("--model", required=True,
                   help="HF id matching configs/model/<org>/<name>.json")
    p.add_argument("--variant", required=True, help="e.g. bf16, bf16-kvfp8")
    p.add_argument("--tp", default="1", help="comma-separated TP degrees, e.g. 1,2,4")
    p.add_argument("--hardware-label", required=True,
                   help="bundle directory label; MUST end in -t0 (A3)")
    p.add_argument("--mirror-keys", type=Path, default=None,
                   help="measured variant root whose exact key set is reproduced")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--attention-max-kv", type=int, default=16384)
    p.add_argument("--attention-chunk-factor", type=float, default=2.0)
    p.add_argument("--attention-kv-factor", type=float, default=2.0)
    p.add_argument("--bytes-mode", choices=("sum", "max"), default="sum")
    p.add_argument("--attn-mode", choices=("max", "sum"), default="max")
    p.add_argument("--model-config", type=Path, default=None,
                   help="HF config path; default configs/model/<model>.json")
    p.add_argument("--out", required=True, type=Path, help="perf root to write under")
    p.add_argument("--force", action="store_true",
                   help="replace an existing destination bundle")
    p.add_argument("--generated-at", default=None,
                   help="ISO timestamp override for reproducible meta.yaml")


def _cmd_emit(args: argparse.Namespace) -> int:
    profile = load_accelerator_profile(args.accelerator)
    config_path = args.model_config or (REPO_ROOT / "configs" / "model" / f"{args.model}.json")
    dims = ModelDims.from_hf_config(config_path, args.variant)
    arch = load_architecture(REPO_ROOT / "profiler" / "models" / f"{dims.model_type}.yaml")
    weight_dtype = args.variant.split("-")[0]
    device = DeviceSpec.from_profile(profile, weight_dtype)

    tp_degrees = sorted({int(t) for t in args.tp.split(",") if t.strip()})
    backends = {
        tp: AnalyticalProfileBackend(
            dims, arch, device, tp,
            bytes_mode=args.bytes_mode, attn_mode=args.attn_mode,
        )
        for tp in tp_degrees
    }
    ds = profile.datasheet
    assert ds is not None  # DeviceSpec.from_profile already enforced this
    emitter = BundleEmitter(
        dims=dims,
        arch=arch,
        backend_for_tp=backends,
        hardware_label=args.hardware_label,
        variant=args.variant,
        out_root=args.out,
        grid=GridParams(
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            attention_max_kv=args.attention_max_kv,
            attention_chunk_factor=args.attention_chunk_factor,
            attention_kv_factor=args.attention_kv_factor,
        ),
        mirror_root=args.mirror_keys,
        datasheet_source=ds.datasheet_source,
        efficiency={
            "flops": device.flops_efficiency,
            "mem": device.mem_efficiency,
            "family": device.family_efficiency,
            "bytes_mode": args.bytes_mode,
            "attn_mode": args.attn_mode,
        },
        force=args.force,
        generated_at=args.generated_at,
    )
    report = emitter.emit()
    print(f"wrote {report.out_root}")
    for tp, files in sorted(report.rows.items()):
        listing = ", ".join(f"{name}={n}" for name, n in sorted(files.items()))
        print(f"  {tp}: {listing}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m profiler.synth")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_emit_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "emit":
        return _cmd_emit(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
