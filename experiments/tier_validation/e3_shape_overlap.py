"""E3 — how much profiling work do models share? (STEP 11)

Counts, across every measured bundle in the repo:
  raw overlap        identical (layer, tokens) / (layer, sequences) /
                     attention 4-axis keys across models on one hardware
  normalized overlap the same keys after ShapeResolver reduces each dense /
                     per_sequence row to its (family, M, N, K) GEMM shape -
                     different models often hit the SAME shapes, which is
                     the ceiling a shape cache could save (Dooly reports
                     56.4% GPU-hour savings from this).

No simulator needed; CI can always run it.

Usage:
    python -m experiments.tier_validation.e3_shape_overlap \
        [--out outputs/tier_validation/e3]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from experiments.tier_validation.common import write_report
from profiler.contract import schema_for
from profiler.core.config import load_architecture
from profiler.synth.dims import ModelDims
from profiler.synth.shapes import ShapeResolver

REPO = Path(__file__).resolve().parents[2]
PERF = REPO / "profiler" / "perf"


def _measured_bundles(perf_root: Path) -> list[tuple[str, str, Path]]:
    """(hardware, model, tp1 dir) for every measured bundle with a tp1/."""
    out = []
    for hw_dir in sorted(perf_root.iterdir()):
        if not hw_dir.is_dir() or hw_dir.name.endswith(("-t0", "-t1")):
            continue  # synthetic bundles would trivially overlap themselves
        for meta in sorted(hw_dir.rglob("meta.yaml")):
            variant_root = meta.parent
            tp1 = variant_root / "tp1"
            if tp1.is_dir():
                model = "/".join(variant_root.relative_to(hw_dir).parts[:-1])
                out.append((hw_dir.name, model, tp1))
    return out


def _raw_keys(tp_dir: Path, model: str) -> set[tuple]:
    """Model-namespaced raw keys.

    A raw (layer, tokens) or attention grid key is only reusable within ONE
    model: ("qkv_proj", 64) names different GEMMs on Llama-3.1-8B and
    Qwen3-32B, and the attention grid axes carry no head counts at all.
    Without the namespace, cross-model raw overlap would be almost entirely
    coincidental grid collisions (observed 0.95 for A5000-Llama vs
    RTXPRO-Qwen3-30B) - overstating reuse that only shape NORMALIZATION can
    legitimately claim.
    """
    keys: set[tuple] = set()
    for filename in ("dense.csv", "per_sequence.csv", "attention.csv", "moe.csv"):
        path = tp_dir / filename
        if not path.exists():
            continue
        schema = schema_for(filename)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                keys.add((model, filename, *(row[c] for c in schema.key_columns)))
    return keys


def _normalized_keys(tp_dir: Path, model: str) -> set[tuple] | None:
    """(family, M, N, K)-normalized keys; None when the model config or
    architecture yaml is not in the repo (some imports have no config)."""
    config = REPO / "configs" / "model" / f"{model}.json"
    if not config.exists():
        return None
    dims = ModelDims.from_hf_config(config, "bf16")
    arch_path = REPO / "profiler" / "models" / f"{dims.model_type}.yaml"
    if not arch_path.exists():
        return None
    resolver = ShapeResolver(dims, load_architecture(arch_path), tp=1)
    keys: set[tuple] = set()
    for filename, fn, key_col in (
        ("dense.csv", resolver.dense, "tokens"),
        ("per_sequence.csv", resolver.per_sequence, "sequences"),
    ):
        path = tp_dir / filename
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                cost = fn(row["layer"], int(row[key_col]))
                # Normalize to the physical work signature: two models whose
                # rows share (family, flops, bytes) would produce the same
                # measurement on the same hardware.
                keys.add((cost.family, cost.flops, cost.bytes_moved))
    # Attention keys normalize to the model-reduced phase work (FLOPs, bytes).
    path = tp_dir / "attention.csv"
    if path.exists():
        from profiler.synth.attn import AttentionCostModel
        from profiler.synth.device import DeviceSpec

        acm = AttentionCostModel(
            dims, DeviceSpec("norm", 1e12, 1e12, 1.0, 1.0), tp=1
        )
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                fp, bp, fd, bd = acm._phase_costs(
                    int(row["prefill_chunk"]), int(row["kv_prefill"]),
                    int(row["n_decode"]), int(row["kv_decode"]),
                )
                keys.add(("attention", fp + fd, bp + bd))
    return keys


def _pairwise_overlap(keysets: dict[str, set[tuple]]) -> dict[str, Any]:
    names = sorted(k for k, v in keysets.items() if v)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = len(keysets[a] & keysets[b])
            union = len(keysets[a] | keysets[b])
            smaller = min(len(keysets[a]), len(keysets[b]))
            pairs[f"{a} vs {b}"] = {
                "intersection": inter,
                "jaccard": inter / union if union else 0.0,
                # Overlap coefficient: the fraction of the SMALLER bundle a
                # shape cache could serve from the other - the savings bound.
                "overlap_coefficient": inter / smaller if smaller else 0.0,
            }
    return pairs


def run(perf_root: Path = PERF) -> dict[str, Any]:
    bundles = _measured_bundles(perf_root)
    raw: dict[str, set[tuple]] = {}
    normalized: dict[str, set[tuple]] = {}
    for hw, model, tp1 in bundles:
        name = f"{hw}:{model}"
        raw[name] = _raw_keys(tp1, model)
        norm = _normalized_keys(tp1, model)
        if norm is not None:
            normalized[name] = norm
    return {
        "bundles": {n: len(k) for n, k in raw.items()},
        "raw_overlap": _pairwise_overlap(raw),
        "normalized_overlap": _pairwise_overlap(normalized),
    }


def render_table(result: dict[str, Any]) -> str:
    lines = ["bundles (raw key counts):"]
    for name, n in result["bundles"].items():
        lines.append(f"  {name:<50} {n}")
    for title in ("raw_overlap", "normalized_overlap"):
        lines.append("")
        lines.append(f"{title}:")
        for pair, stats in result[title].items():
            lines.append(
                f"  {pair:<70} inter={stats['intersection']:>6} "
                f"jaccard={stats['jaccard']:.3f} "
                f"overlap={stats['overlap_coefficient']:.3f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e3_shape_overlap")
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "tier_validation" / "e3"
    )
    args = parser.parse_args(argv)
    result = run()
    table = render_table(result)
    print(table)
    write_report(args.out, "e3_shape_overlap", result, table=table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
