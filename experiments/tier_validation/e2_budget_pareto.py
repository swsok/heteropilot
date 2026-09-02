"""E2 — where should a limited measurement budget go? (STEP 11)

Four bundle conditions on one hardware, priced against the fully measured
bundle (condition D):

    A  Tier 0, zero measurements
    B  Tier 0 + `budget` ANCHORS, all on attention (pick-anchors share=1.0)
    C  Tier 0 + `budget` anchors spread uniformly (share = attention's
       natural fraction of the grid)
    D  Tier 2 (every row measured)

Measurement score = anchor count (a GPU-hour proxy: every anchor is one
measured shot). Error = per-key bundle MAPE vs condition D over the keys
NOT used as anchors (hold-out), computed with the STEP 8 diff harness. This
tests KernelSight-LM's claim that one attention-focused calibration pass
buys most of the accuracy.

Usage:
    python -m experiments.tier_validation.e2_budget_pareto --dry-run
    python -m experiments.tier_validation.e2_budget_pareto \
        [--budget 200] [--out outputs/tier_validation/e2]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from experiments.tier_validation.common import write_report
from planner.inventory import load_accelerator_profile
from profiler.contract import schema_for
from profiler.core.config import load_architecture
from profiler.synth.backend import AnalyticalProfileBackend
from profiler.synth.calibrate import fit_from_anchors, pick_anchors
from profiler.synth.device import DeviceSpec
from profiler.synth.diff import LAYER_FAMILY
from profiler.synth.dims import ModelDims

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MEASURED = REPO / "profiler" / "perf" / "A40" / "meta-llama" / "Llama-3.1-8B" / "bf16"
DEFAULT_ACCEL = REPO / "profiles" / "accelerators" / "a40.yaml"
DEFAULT_EFFICIENCY = REPO / "profiles" / "accelerators" / "a40.efficiency.yaml"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"


def _load_backend(accel: Path, efficiency: Path, model: str, tp: int) -> AnalyticalProfileBackend:
    profile = load_accelerator_profile(accel)
    assert profile.datasheet is not None
    eff = yaml.safe_load(efficiency.read_text())
    fitted = profile.model_copy(update={"datasheet": profile.datasheet.model_copy(update={
        "flops_efficiency": eff["flops_efficiency"],
        "mem_efficiency": eff["mem_efficiency"],
        "family_efficiency": {
            k: v for k, v in eff["family_efficiency"].items() if v <= 1.0
        },
    })})
    dims = ModelDims.from_hf_config(REPO / "configs" / "model" / f"{model}.json", "bf16")
    arch = load_architecture(REPO / "profiler" / "models" / f"{dims.model_type}.yaml")
    return AnalyticalProfileBackend(dims, arch, DeviceSpec.from_profile(fitted, "bf16"), tp)


def _measured_rows(measured: Path, tp: int) -> list[tuple[str, tuple, float]]:
    rows: list[tuple[str, tuple, float]] = []
    for name in ("dense.csv", "per_sequence.csv", "attention.csv", "moe.csv"):
        path = measured / f"tp{tp}" / name
        if not path.exists():
            continue
        schema = schema_for(name)
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                key = tuple(
                    int(r[c]) if c in schema.int_columns else r[c]
                    for c in schema.key_columns
                )
                rows.append((name, key, float(r["time_us"])))
    return rows


def _family_of(name: str, key: tuple) -> str:
    if name == "attention.csv":
        return "attention"
    if name == "moe.csv":
        return "moe"
    return LAYER_FAMILY.get(str(key[0]), "unknown")


def run(
    *,
    measured: Path = DEFAULT_MEASURED,
    accelerator: Path = DEFAULT_ACCEL,
    efficiency: Path = DEFAULT_EFFICIENCY,
    model: str = DEFAULT_MODEL,
    tp: int = 1,
    budget: int = 200,
    dry_run: bool = False,
    backend: AnalyticalProfileBackend | None = None,
) -> dict[str, Any]:
    rows = _measured_rows(measured, tp)
    keys_by_file: dict[str, list[tuple]] = {}
    for name, key, _ in rows:
        keys_by_file.setdefault(name, []).append(key)

    if dry_run:
        return {
            "dry_run": True,
            "budget": budget,
            "total_measured_rows": len(rows),
            "conditions": {
                "A_tier0": 0, "B_attention_anchors": budget,
                "C_uniform_anchors": budget, "D_tier2": len(rows),
            },
        }

    be = backend or _load_backend(accelerator, efficiency, model, tp)
    measured_by_key = {(n, k): t for n, k, t in rows}
    est = {
        "dense.csv": lambda k: be.dense_us(str(k[0]), int(k[1])),
        "per_sequence.csv": lambda k: be.per_sequence_us(str(k[0]), int(k[1])),
        "attention.csv": lambda k: be.attention_us(*k),
        "moe.csv": lambda k: be.expert_us(int(k[0]), int(k[1])),
    }
    n_attn = len(keys_by_file.get("attention.csv", []))
    natural_share = n_attn / len(rows) if rows else 0.0

    def condition(name: str, attention_share: float | None) -> dict[str, Any]:
        """None share = condition A (no anchors)."""
        anchor_keys: set[tuple[str, tuple]] = set()
        table = None
        if attention_share is not None:
            plan = pick_anchors(keys_by_file, budget, attention_share=attention_share)
            anchor_keys = {(f, k) for f, keys in plan.items() for k in keys}
            # Write the anchors as a subset bundle and fit (STEP 9 pipeline).
            import tempfile

            anchor_root = Path(tempfile.mkdtemp(prefix="e2-anchors-"))
            tp_dir = anchor_root / f"tp{tp}"
            tp_dir.mkdir(parents=True)
            for filename, keys in plan.items():
                schema = schema_for(filename)
                with (tp_dir / filename).open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(schema.columns)
                    for k in keys:
                        w.writerow([*k, f"{measured_by_key[(filename, k)]:.6g}"])
            table = fit_from_anchors(anchor_root, be, tp=tp)

        errs = []
        for filename, key, measured_us in rows:
            if (filename, key) in anchor_keys:
                continue  # hold-out only
            t0 = est[filename](key)
            t = t0 * table.scale(_family_of(filename, key), t0) if table else t0
            errs.append(abs(t - measured_us) / measured_us)
        return {
            "condition": name,
            "measurement_score": len(anchor_keys),
            "holdout_rows": len(errs),
            "mape_vs_tier2": sum(errs) / len(errs) if errs else 0.0,
        }

    conditions = [
        condition("A_tier0", None),
        condition("B_attention_anchors", 1.0),
        condition("C_uniform_anchors", natural_share),
        # D: fully measured - zero error by definition, full measurement cost.
        {
            "condition": "D_tier2",
            "measurement_score": len(rows),
            "holdout_rows": 0,
            "mape_vs_tier2": 0.0,
        },
    ]
    return {
        "budget": budget,
        "natural_attention_share": natural_share,
        "conditions": conditions,
    }


def render_table(result: dict[str, Any]) -> str:
    if result.get("dry_run"):
        parts = ", ".join(f"{k}={v}" for k, v in result["conditions"].items())
        return f"dry-run: budget={result['budget']} -> measurement scores {parts}"
    lines = [f"{'condition':<24} {'measured pts':>12} {'MAPE vs tier2':>14}"]
    for c in result["conditions"]:
        lines.append(
            f"{c['condition']:<24} {c['measurement_score']:>12} "
            f"{c['mape_vs_tier2'] * 100:>13.1f}%"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e2_budget_pareto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "tier_validation" / "e2"
    )
    args = parser.parse_args(argv)
    result = run(budget=args.budget, dry_run=args.dry_run)
    table = render_table(result)
    print(table)
    if not args.dry_run:
        write_report(
            args.out, "e2_budget_pareto", result, table=table,
            provenance_extra={"budget": args.budget},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
