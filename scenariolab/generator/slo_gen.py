"""M2 SLOGenerator: random, valid ServiceSpec instances (DESIGN §5).

Deliberately infeasible SLOs are allowed through (FR-S6): the planner's
infeasibility diagnosis (closest_plan, violated_constraints, suggestions) is
one of the things ScenarioLab exists to observe. The objective is fixed per
batch (FR-S5) - it is the experiment's control variable, never re-sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from planner.spec import ServiceSpec, load_service_spec
from scenariolab.config import SloGeneratorConfig
from scenariolab.generator.sampling import rng_for


@dataclass(frozen=True)
class ServiceSummary:
    service_id: str
    seed: int
    yaml_path: Path
    model: str
    rps: float
    input_p50: int
    output_p50: int
    ttft_p99_ms: float
    tpot_p99_ms: float
    power_cap_w: float


class SloGenError(ValueError):
    """Raised when a generated spec fails its own validation."""


def _token_block(p50: int, config: SloGeneratorConfig) -> dict:
    # FR-S4: p50 <= p95 <= p99 holds by construction via fixed multipliers.
    return {
        "p50": p50,
        "p95": int(p50 * config.token_p95_multiplier),
        "p99": int(p50 * config.token_p99_multiplier),
    }


def generate_service(
    config: SloGeneratorConfig,
    index: int,
    seed: int,
    out_dir: Path,
    lab_config_hash: str,
    service_id: str | None = None,
) -> ServiceSummary:
    """Generate, self-validate and write a random service (FR-S1..S7).

    `service_id` defaults to the batch convention s{index:04d}; the workspace
    placement engine passes its own ids so workspace services never collide
    with batch services in the shared `services` table."""
    service_id = service_id or f"s{index:04d}"
    rng = rng_for(seed)

    # Sampling order is part of the reproducibility contract; do not reorder.
    model = config.models[int(rng.integers(len(config.models)))]
    rps = config.arrival_rate_rps.sample(rng)
    input_p50 = int(config.input_tokens_p50.sample(rng))
    output_p50 = int(config.output_tokens_p50.sample(rng))
    ttft_ms = config.ttft_p99_ms.sample(rng)
    tpot_ms = config.tpot_p99_ms.sample(rng)
    power_cap = config.power_cap_w.sample(rng)
    min_tpj = config.min_tokens_per_joule.sample(rng)

    slo: dict = {
        "ttft": {"percentile": 99, "max_ms": round(ttft_ms, 3)},
        "tpot": {"percentile": 99, "max_ms": round(tpot_ms, 3)},
        "max_cluster_power_w": round(power_cap, 3),
    }
    if min_tpj > 0:
        slo["min_tokens_per_joule"] = round(min_tpj, 6)

    raw = {
        "service": {"model": model, "dtype": config.dtype, "kv_cache_dtype": "auto"},
        "traffic": {
            "arrival_rate_rps": round(rps, 6),
            "input_tokens": _token_block(input_p50, config),
            "output_tokens": _token_block(output_p50, config),
            "burstiness": config.burstiness,
        },
        "slo": slo,
        "objective": {
            "primary": config.objective.primary,
            **(
                {"secondary": config.objective.secondary}
                if config.objective.secondary else {}
            ),
        },
    }

    # Self-check in memory first so the error message carries the sampled values.
    try:
        ServiceSpec.model_validate(raw)
    except Exception as exc:
        raise SloGenError(f"{service_id}: generated spec is invalid - {exc}") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{service_id}.yaml"
    header = (
        "# generated_by: scenariolab\n"
        f"# slo_seed: {seed}\n"
        f"# lab_config_hash: {lab_config_hash}\n"
        f"# sampled: model={model} rps={rps:.3f} in_p50={input_p50} "
        f"out_p50={output_p50} ttft_p99={ttft_ms:.1f}ms tpot_p99={tpot_ms:.1f}ms "
        f"power_cap={power_cap:.0f}W\n"
    )
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False))

    # FR-S3: the file itself must round-trip through the planner loader.
    load_service_spec(path)

    return ServiceSummary(
        service_id=service_id,
        seed=seed,
        yaml_path=path,
        model=model,
        rps=rps,
        input_p50=input_p50,
        output_p50=output_p50,
        ttft_p99_ms=ttft_ms,
        tpot_p99_ms=tpot_ms,
        power_cap_w=power_cap,
    )
