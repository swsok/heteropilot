"""CLI for synthetic (Tier 0/1) profile bundles.

    python -m profiler.synth emit \\
        --accelerator profiles/accelerators/ascend_target.yaml \\
        --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1,2 \\
        --hardware-label ASCEND_TARGET-t0 --out profiler/perf

Subcommands: emit (Tier 0/1 bundles), diff, fit-efficiency (STEP 8),
calibrate + pick-anchors (Tier 1, STEP 9).
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
    p.add_argument("--attn-mode", choices=("max", "sum"), default="sum")
    p.add_argument("--model-config", type=Path, default=None,
                   help="HF config path; default configs/model/<model>.json")
    p.add_argument("--out", required=True, type=Path, help="perf root to write under")
    p.add_argument("--force", action="store_true",
                   help="replace an existing destination bundle")
    p.add_argument("--generated-at", default=None,
                   help="ISO timestamp override for reproducible meta.yaml")
    p.add_argument("--scaling", type=Path, default=None,
                   help="ScalingTable YAML from 'calibrate'; makes the bundle "
                        "tier: calibrated (label must end -t1)")
    p.add_argument("--efficiency", type=Path, default=None,
                   help="fit-efficiency YAML overlaying the datasheet's "
                        "efficiencies (families fitted > 1.0 are skipped: they "
                        "broke the bound and cannot be datasheet values)")


def _cmd_emit(args: argparse.Namespace) -> int:
    profile = load_accelerator_profile(args.accelerator)
    if args.efficiency is not None:
        import yaml

        eff = yaml.safe_load(args.efficiency.read_text())
        ds = profile.datasheet
        if ds is None:
            print(f"error: {args.accelerator} has no datasheet block", file=sys.stderr)
            return 2
        flops_eff, mem_eff = eff.get("flops_efficiency"), eff.get("mem_efficiency")
        if (flops_eff or 0) > 1.0 or (mem_eff or 0) > 1.0:
            # Never clamp a bound-breaking fit into a plausible number.
            print(
                f"error: {args.efficiency} carries a base efficiency > 1.0 "
                f"(bound violation); resolve it before emitting",
                file=sys.stderr,
            )
            return 2
        profile = profile.model_copy(update={"datasheet": ds.model_copy(update={
            "flops_efficiency": flops_eff,
            "mem_efficiency": mem_eff,
            "family_efficiency": {
                k: v for k, v in (eff.get("family_efficiency") or {}).items()
                if v <= 1.0  # >1.0 families skipped (recorded bound violations)
            },
        })})
    config_path = args.model_config or (REPO_ROOT / "configs" / "model" / f"{args.model}.json")
    dims = ModelDims.from_hf_config(config_path, args.variant)
    arch = load_architecture(REPO_ROOT / "profiler" / "models" / f"{dims.model_type}.yaml")
    weight_dtype = args.variant.split("-")[0]
    device = DeviceSpec.from_profile(profile, weight_dtype)

    tp_degrees = sorted({int(t) for t in args.tp.split(",") if t.strip()})
    scaling = None
    calibration_anchors = None
    backends: dict[int, AnalyticalProfileBackend]
    if args.scaling is not None:
        import yaml

        from profiler.synth.backend import CalibratedProfileBackend
        from profiler.synth.calibrate import ScalingTable

        scaling = ScalingTable.load(args.scaling)
        raw = yaml.safe_load(args.scaling.read_text())
        calibration_anchors = {
            "n_anchors": raw.get("n_anchors"),
            "anchors_per_family": raw.get("anchors_per_family"),
            "derived_from": raw.get("derived_from"),
            "scaling_table": str(args.scaling),
        }
        backends = {
            tp: CalibratedProfileBackend(
                dims, arch, device, tp,
                bytes_mode=args.bytes_mode, attn_mode=args.attn_mode,
                scaling=scaling,
            )
            for tp in tp_degrees
        }
    else:
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
        calibration_anchors=calibration_anchors,
        force=args.force,
        generated_at=args.generated_at,
    )
    report = emitter.emit()
    print(f"wrote {report.out_root}")
    for tp, files in sorted(report.rows.items()):
        listing = ", ".join(f"{name}={n}" for name, n in sorted(files.items()))
        print(f"  {tp}: {listing}")
    return 0


def _build_diff_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("diff", help="Compare a measured and a synthetic bundle.")
    p.add_argument("--measured", required=True, type=Path, help="variant root")
    p.add_argument("--synth", required=True, type=Path, help="variant root")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--format", choices=("table", "json"), default="table")


def _cmd_diff(args: argparse.Namespace) -> int:
    import json

    from profiler.synth.diff import diff_bundles

    report = diff_bundles(args.measured, args.synth, args.tp)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"wrote {args.out}")
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render_table())
    return 0


def _build_fit_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fit-efficiency",
        help="Fit per-family efficiencies against a measured bundle "
             "(writes a separate YAML; never edits accelerator profiles).",
    )
    p.add_argument("--accelerator", required=True, type=Path)
    p.add_argument("--measured", required=True, type=Path, help="variant root")
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="bf16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--bytes-mode", choices=("sum", "max"), default="sum")
    p.add_argument("--attn-mode", choices=("max", "sum"), default="sum")
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)


def _cmd_fit(args: argparse.Namespace) -> int:
    import datetime

    import yaml

    from planner.util import provenance as prov
    from profiler.synth.diff import fit_efficiency

    profile = load_accelerator_profile(args.accelerator)
    ds = profile.datasheet
    if ds is None:
        print(f"error: {args.accelerator} has no datasheet block", file=sys.stderr)
        return 2
    config_path = args.model_config or (
        REPO_ROOT / "configs" / "model" / f"{args.model}.json"
    )
    dims = ModelDims.from_hf_config(config_path, args.variant)
    arch = load_architecture(
        REPO_ROOT / "profiler" / "models" / f"{dims.model_type}.yaml"
    )
    weight_dtype = args.variant.split("-")[0]
    # Theoretical-lower-bound mode: eff=1.0 injected explicitly.
    lb_profile = profile.model_copy(
        update={
            "datasheet": ds.model_copy(
                update={"flops_efficiency": 1.0, "mem_efficiency": 1.0,
                        "family_efficiency": {}}
            )
        }
    )
    device = DeviceSpec.from_profile(lb_profile, weight_dtype)
    backend = AnalyticalProfileBackend(
        dims, arch, device, args.tp,
        bytes_mode=args.bytes_mode, attn_mode=args.attn_mode,
    )
    fit = fit_efficiency(
        args.measured, args.tp, backend,
        derived_from={
            "measured_bundle": str(args.measured),
            "git_commit": prov.git_commit() or "unknown",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "bytes_mode": args.bytes_mode,
            "attn_mode": args.attn_mode,
            "tp": str(args.tp),
            "model": args.model,
            "variant": args.variant,
        },
    )
    out = {
        # gemm derates the compute term, elementwise the memory term; the
        # full per-family table is authoritative.
        "flops_efficiency": fit.family_efficiency.get("gemm"),
        "mem_efficiency": fit.family_efficiency.get("elementwise"),
        "family_efficiency": fit.family_efficiency,
        "n_per_family": fit.n_per_family,
        "derived_from": fit.derived_from,
    }
    if fit.bound_violations:
        # Recorded as-is, never clamped: a >1.0 value means the measurement
        # beat the 'theoretical' bound (usually large-L2 caching of small
        # working sets) and must stay visible in the artifact.
        out["bound_violations"] = fit.bound_violations
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"wrote {args.out}")
    for family, eff in fit.family_efficiency.items():
        print(f"  {family:<12} eff={eff:.4f}  (n={fit.n_per_family[family]})")
    if fit.bound_violations:
        print(
            f"warning: fitted efficiency > 1.0 for {fit.bound_violations} - "
            f"the DRAM-bandwidth 'lower bound' was beaten (typically large-L2 "
            f"caching). Recorded unclamped; such a family cannot be merged "
            f"into a Datasheet (which requires (0, 1]).",
            file=sys.stderr,
        )
        return 3
    return 0


def _build_calibrate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "calibrate",
        help="Fit a Tier 1 ScalingTable from anchor measurements (STEP 9).",
    )
    p.add_argument("--anchors", required=True, type=Path,
                   help="variant root (or tp dir) holding anchor-subset CSVs")
    p.add_argument("--accelerator", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="bf16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--bytes-mode", choices=("sum", "max"), default="sum")
    p.add_argument("--attn-mode", choices=("max", "sum"), default="sum")
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import datetime

    from planner.util import provenance as prov
    from profiler.synth.calibrate import fit_from_anchors

    profile = load_accelerator_profile(args.accelerator)
    config_path = args.model_config or (
        REPO_ROOT / "configs" / "model" / f"{args.model}.json"
    )
    dims = ModelDims.from_hf_config(config_path, args.variant)
    arch = load_architecture(
        REPO_ROOT / "profiler" / "models" / f"{dims.model_type}.yaml"
    )
    device = DeviceSpec.from_profile(profile, args.variant.split("-")[0])
    backend = AnalyticalProfileBackend(
        dims, arch, device, args.tp,
        bytes_mode=args.bytes_mode, attn_mode=args.attn_mode,
    )
    table = fit_from_anchors(
        args.anchors, backend, tp=args.tp, bins=args.bins,
        derived_from={
            "anchors": str(args.anchors),
            "accelerator": str(args.accelerator),
            "git_commit": prov.git_commit() or "unknown",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "bytes_mode": args.bytes_mode,
            "attn_mode": args.attn_mode,
            "tp": str(args.tp),
        },
    )
    table.save(args.out)
    print(f"wrote {args.out}")
    for family, count in table.anchors_per_family().items():
        pieces = len(table.piecewise.get(family, ()))
        print(
            f"  {family:<12} scalar={table.scalars[family]:.4f}  anchors={count}"
            + (f"  piecewise bins={pieces}" if pieces else "")
        )
    return 0


def _build_pick_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "pick-anchors",
        help="Plan which grid keys to measure as Tier 1 anchors.",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="bf16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--attention-share", type=float, default=0.7,
                   help="fraction of the budget spent on attention keys "
                        "(default 0.7 - attention does not transfer across "
                        "devices, STEP 6 background)")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--attention-max-kv", type=int, default=16384)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)


def _cmd_pick(args: argparse.Namespace) -> int:
    import csv as _csv

    from profiler.synth.calibrate import pick_anchors
    from profiler.synth.emit import enumerate_grid_keys

    config_path = args.model_config or (
        REPO_ROOT / "configs" / "model" / f"{args.model}.json"
    )
    dims = ModelDims.from_hf_config(config_path, args.variant)
    arch = load_architecture(
        REPO_ROOT / "profiler" / "models" / f"{dims.model_type}.yaml"
    )
    grid = GridParams(
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        attention_max_kv=args.attention_max_kv,
    )
    files = ["dense.csv", "per_sequence.csv", "attention.csv"]
    if dims.is_moe:
        files.append("moe.csv")
    keys_by_file = {
        f: enumerate_grid_keys(dims, arch, grid, args.tp, f) for f in files
    }
    plan = pick_anchors(
        keys_by_file, args.budget, attention_share=args.attention_share
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["file", "layer", "tokens", "sequences",
               "prefill_chunk", "kv_prefill", "n_decode", "kv_decode",
               "activated_experts"]
    with args.out.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for filename, keys in plan.items():
            for key in keys:
                row = {"file": filename}
                if filename == "dense.csv":
                    row.update(layer=key[0], tokens=key[1])
                elif filename == "per_sequence.csv":
                    row.update(layer=key[0], sequences=key[1])
                elif filename == "attention.csv":
                    row.update(prefill_chunk=key[0], kv_prefill=key[1],
                               n_decode=key[2], kv_decode=key[3])
                elif filename == "moe.csv":
                    row.update(tokens=key[0], activated_experts=key[1])
                writer.writerow(row)
    total = sum(len(v) for v in plan.values())
    print(f"wrote {args.out} ({total} anchor keys)")
    for filename, keys in plan.items():
        print(f"  {filename:<18} {len(keys)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m profiler.synth")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_emit_parser(sub)
    _build_diff_parser(sub)
    _build_fit_parser(sub)
    _build_calibrate_parser(sub)
    _build_pick_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "emit":
        return _cmd_emit(args)
    if args.command == "diff":
        return _cmd_diff(args)
    if args.command == "fit-efficiency":
        return _cmd_fit(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "pick-anchors":
        return _cmd_pick(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
