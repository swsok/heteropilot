"""ServiceSpec: the user-facing request (work order §3.1).

Model, traffic distribution, SLOs and objective. Everything the planner needs
to decide how much hardware a service requires.
"""

from __future__ import annotations

import enum
import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ALLOWED_PERCENTILES = (50, 95, 99)


class Objective(str, enum.Enum):
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_SLO_GOODPUT_PER_JOULE = "maximize_slo_goodput_per_joule"
    MINIMIZE_ACTIVE_ACCELERATORS = "minimize_active_accelerators"


class _Strict(BaseModel):
    """Reject unknown keys so typos in a YAML spec fail loudly."""

    model_config = ConfigDict(extra="forbid")


class Service(_Strict):
    model: str
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"


class TokenDistribution(_Strict):
    p50: int = Field(gt=0)
    p95: int | None = Field(default=None, gt=0)
    p99: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _monotone(self) -> TokenDistribution:
        seen = [("p50", self.p50)]
        if self.p95 is not None:
            seen.append(("p95", self.p95))
        if self.p99 is not None:
            seen.append(("p99", self.p99))
        for (lo_name, lo), (hi_name, hi) in itertools.pairwise(seen):
            if hi < lo:
                raise ValueError(
                    f"token distribution is not monotone: {hi_name}={hi} < {lo_name}={lo}"
                )
        return self


class Traffic(_Strict):
    arrival_rate_rps: float = Field(gt=0)
    input_tokens: TokenDistribution
    output_tokens: TokenDistribution
    burstiness: float = Field(default=1.0, gt=0)
    prefix_share_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class LatencyTarget(_Strict):
    percentile: int
    max_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def _percentile_allowed(self) -> LatencyTarget:
        if self.percentile not in ALLOWED_PERCENTILES:
            raise ValueError(
                f"percentile must be one of {ALLOWED_PERCENTILES}, got {self.percentile}"
            )
        return self


class Slo(_Strict):
    ttft: LatencyTarget
    tpot: LatencyTarget
    max_cluster_power_w: float | None = Field(default=None, gt=0)
    min_tokens_per_joule: float | None = Field(default=None, gt=0)


class ObjectiveSpec(_Strict):
    primary: Objective
    secondary: Objective | None = None

    @model_validator(mode="after")
    def _distinct(self) -> ObjectiveSpec:
        if self.secondary is not None and self.secondary == self.primary:
            raise ValueError(
                f"objective.primary and objective.secondary are both "
                f"'{self.primary.value}'; the secondary objective must differ"
            )
        return self


class ServiceSpec(_Strict):
    service: Service
    traffic: Traffic
    slo: Slo
    objective: ObjectiveSpec

    @property
    def model(self) -> str:
        return self.service.model


class SpecError(ValueError):
    """Raised when a spec file cannot be loaded or fails validation."""


def _require_traffic(raw: dict[str, Any], path: Path) -> None:
    """Traffic is what sizes the deployment; SLOs alone cannot (work order §3.1)."""
    if "traffic" not in raw:
        raise SpecError(
            f"{path}: missing required 'traffic' block.\n"
            "Resource sizing cannot be derived from SLOs alone - an SLO says how fast "
            "each request must be, not how many arrive. Provide at least "
            "traffic.arrival_rate_rps, traffic.input_tokens.p50 and traffic.output_tokens.p50."
        )


def load_service_spec(path: str | Path) -> ServiceSpec:
    """Load and validate a ServiceSpec YAML file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise SpecError(f"{path}: file not found") from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML - {exc}") from exc

    if not isinstance(raw, dict):
        raise SpecError(f"{path}: expected a YAML mapping at the top level")

    _require_traffic(raw, path)

    try:
        return ServiceSpec.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise SpecError(f"{path}: {exc}") from exc
