"""E4 — how robust are unowned-hardware conclusions? (STEP 11)

The Ascend datasheet is secondary-sourced (no official Huawei datasheet
exists - see profiles/accelerators/ascend_target.yaml), so any plan that
depends on it must survive a sensitivity sweep. Each parameter in
{peak_tflops, memory_bandwidth_gbps, flops_efficiency, mem_efficiency} is
swept +-30%; the metric is WHERE (if anywhere) the recommended plan flips.

Predictor: the datasheet-driven AnalyticalPredictor by default - a full
simulator sweep would be 4 params x 13 steps x ~18 candidates ~ 900 minutes,
and the flip THRESHOLD only needs a predictor that responds to the swept
parameters. Pass a simulator-backed predictor factory for a spot check.

Usage:
    python -m experiments.tier_validation.e4_sensitivity --dry-run
    python -m experiments.tier_validation.e4_sensitivity \
        [--out outputs/tier_validation/e4] [--steps 13]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.tier_validation.common import AnalyticalPredictor, write_report
from planner.inventory import (
    AcceleratorProfile,
    detect_islands,
    load_cluster_spec,
    load_profiles_for,
)
from planner.optimizer import exhaustive
from planner.predictor import Predictor
from planner.spec import load_service_spec

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CLUSTER = REPO / "examples" / "clusters" / "hetero-gpu-ascend.yaml"
DEFAULT_SERVICE = REPO / "examples" / "service_specs" / "qwen3-32b.yaml"
SWEPT_PARAMS = ("peak_tflops", "memory_bandwidth_gbps", "flops_efficiency", "mem_efficiency")
TARGET_MODEL = "ASCEND_TARGET"


def _scaled_profile(profile: AcceleratorProfile, param: str, factor: float) -> AcceleratorProfile:
    ds = profile.datasheet
    assert ds is not None
    if param == "peak_tflops":
        update: dict[str, Any] = {
            "peak_tflops": {k: v * factor for k, v in ds.peak_tflops.items()}
        }
    elif param == "memory_bandwidth_gbps":
        base = ds.memory_bandwidth_gbps or profile.memory_bandwidth_gbps
        update = {"memory_bandwidth_gbps": base * factor}
    else:
        base = getattr(ds, param)
        # Efficiencies are capped at 1.0 by schema; clamp the +30% side and
        # record the effective factor instead of failing the sweep.
        update = {param: min(1.0, base * factor)}
    return profile.model_copy(update={"datasheet": ds.model_copy(update=update)})


def sweep_grid(steps: int) -> list[float]:
    """Symmetric multiplier grid over [0.7, 1.3] with 1.0 included."""
    if steps < 3 or steps % 2 == 0:
        raise ValueError("steps must be an odd integer >= 3 so 1.0 is on the grid")
    return [0.7 + 0.6 * i / (steps - 1) for i in range(steps)]


def run(
    *,
    cluster_path: Path = DEFAULT_CLUSTER,
    service_path: Path = DEFAULT_SERVICE,
    steps: int = 13,
    predictor_factory: Callable[[], Predictor] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cluster = load_cluster_spec(cluster_path)
    profiles = load_profiles_for(cluster, REPO)
    islands = detect_islands(cluster, profiles)
    spec = load_service_spec(service_path)
    factors = sweep_grid(steps)

    if dry_run:
        return {
            "dry_run": True,
            "params": list(SWEPT_PARAMS),
            "factors": factors,
            "searches": len(SWEPT_PARAMS) * len(factors),
        }

    factory = predictor_factory or AnalyticalPredictor

    def recommended_at(param: str, factor: float) -> str | None:
        swept = dict(profiles)
        swept[TARGET_MODEL] = _scaled_profile(profiles[TARGET_MODEL], param, factor)
        output = exhaustive.search(spec, cluster, islands, swept, factory())
        if output.recommended is None:
            return None
        return output.recommended.plan.candidate.id

    results: dict[str, Any] = {"params": {}, "factors": factors}
    for param in SWEPT_PARAMS:
        picks = {f: recommended_at(param, f) for f in factors}
        baseline = picks[1.0] if 1.0 in picks else picks[factors[len(factors) // 2]]
        # The flip thresholds: nearest factors below/above 1.0 where the
        # recommendation differs from the nominal one.
        flip_low = next(
            (f for f in sorted((x for x in factors if x < 1.0), reverse=True)
             if picks[f] != baseline),
            None,
        )
        flip_high = next(
            (f for f in sorted(x for x in factors if x > 1.0) if picks[f] != baseline),
            None,
        )
        results["params"][param] = {
            "baseline_pick": baseline,
            "picks": {f"{f:.3f}": p for f, p in picks.items()},
            "flip_below": flip_low,
            "flip_above": flip_high,
            "stable": flip_low is None and flip_high is None,
        }
    return results


def render_table(result: dict[str, Any]) -> str:
    if result.get("dry_run"):
        return (
            f"dry-run: {len(result['params'])} params x {len(result['factors'])} "
            f"factors = {result['searches']} searches"
        )
    lines = [f"{'parameter':<24} {'baseline pick':<40} {'flips below':>12} {'flips above':>12}"]
    for param, r in result["params"].items():
        lines.append(
            f"{param:<24} {r['baseline_pick']!s:<40} "
            f"{r['flip_below']!s:>12} {r['flip_above']!s:>12}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e4_sensitivity")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=13)
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "tier_validation" / "e4"
    )
    args = parser.parse_args(argv)
    result = run(steps=args.steps, dry_run=args.dry_run)
    table = render_table(result)
    print(table)
    if not args.dry_run:
        write_report(
            args.out, "e4_sensitivity", result, table=table,
            provenance_extra={"predictor": "AnalyticalPredictor", "steps": args.steps},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
