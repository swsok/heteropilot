"""LabConfig: the single YAML that fully describes one batch (DESIGN §3.1).

All randomness derives from `lab.seed`. Loading validates the accelerator pool
against the profile files (placeholder profiles are rejected outright, DESIGN
§0.2/§3.1) and checks that every model has a usable perf bundle on every pool
accelerator, so a batch fails before it starts rather than mid-run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from planner.inventory import (
    AcceleratorProfile,
    Source,
    compatibility,
    load_accelerator_profile,
)
from scenariolab.generator.sampling import DistSpec, FloatRange, IntRange

#: Accelerator pool entries map to profiles/accelerators/<name>.yaml.
PROFILE_DIR = Path("profiles/accelerators")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabSection(_Strict):
    batch_name: str
    seed: int = Field(ge=0)


class ClusterGeneratorConfig(_Strict):
    num_clusters: int = Field(ge=1)
    nodes_per_cluster: IntRange
    accelerators_per_node: IntRange
    accelerator_pool: list[str] = Field(min_length=1)
    same_class_per_node: bool = True
    internode_link_pool: list[str] = Field(min_length=1)
    free_ratio: FloatRange

    @model_validator(mode="after")
    def _free_ratio_range(self) -> ClusterGeneratorConfig:
        if not (0.0 < self.free_ratio.min <= 1.0 and 0.0 < self.free_ratio.max <= 1.0):
            raise ValueError("free_ratio must lie in (0, 1]")
        return self


class ObjectiveConfig(_Strict):
    primary: str = "minimize_energy"
    secondary: str | None = "minimize_active_accelerators"


class SloGeneratorConfig(_Strict):
    num_specs: int = Field(ge=1)
    models: list[str] = Field(min_length=1)
    dtype: str = "bfloat16"
    arrival_rate_rps: DistSpec
    input_tokens_p50: DistSpec
    output_tokens_p50: DistSpec
    ttft_p99_ms: DistSpec
    tpot_p99_ms: DistSpec
    power_cap_w: DistSpec
    min_tokens_per_joule: DistSpec = Field(
        default_factory=lambda: DistSpec(dist="fixed", value=0.0)
    )
    #: FR-S4: p95/p99 are derived from the sampled p50 by fixed multipliers so
    #: the distribution is internally monotone by construction.
    token_p95_multiplier: float = Field(default=4.0, ge=1.0)
    token_p99_multiplier: float = Field(default=8.0, ge=1.0)
    burstiness: float = Field(default=1.0, gt=0)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)

    @model_validator(mode="after")
    def _multipliers_ordered(self) -> SloGeneratorConfig:
        if self.token_p99_multiplier < self.token_p95_multiplier:
            raise ValueError(
                f"token_p99_multiplier={self.token_p99_multiplier} < "
                f"token_p95_multiplier={self.token_p95_multiplier}"
            )
        return self


class PairingConfig(_Strict):
    mode: Literal["cross", "random"] = "cross"
    num_pairs: int | None = Field(default=None, ge=1)
    max_scenarios: int = Field(default=1500, ge=1)

    @model_validator(mode="after")
    def _random_needs_num_pairs(self) -> PairingConfig:
        if self.mode == "random" and self.num_pairs is None:
            raise ValueError("pairing.mode=random requires num_pairs")
        return self


class TierPolicy(_Strict):
    envelope_cache: bool = True
    surrogate_top_k: int = Field(default=5, ge=1)
    full_sim: Literal["verification_only", "top_k", "never"] = "verification_only"


class VerificationConfig(_Strict):
    fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    min_count: int = Field(default=10, ge=0)
    stratify_by: list[str] = Field(
        default_factory=lambda: ["feasible", "cluster_size_bucket", "has_npu"]
    )
    sim_workers: int = Field(default=32, ge=1)


class RunnerConfig(_Strict):
    workers: int = Field(default=16, ge=1)
    num_requests: int = Field(default=300, ge=1)
    enable_pd: bool = False
    tier_policy: TierPolicy = Field(default_factory=TierPolicy)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)


class StoreConfig(_Strict):
    db_path: Path = Path("outputs/scenariolab/lab.sqlite")
    results_dir: Path = Path("outputs/scenariolab/results")
    clusters_dir: Path = Path("outputs/scenariolab/clusters")
    services_dir: Path = Path("outputs/scenariolab/services")


class LabConfig(_Strict):
    lab: LabSection
    cluster_generator: ClusterGeneratorConfig
    slo_generator: SloGeneratorConfig
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)

    def num_scenarios(self) -> int:
        if self.pairing.mode == "cross":
            return self.cluster_generator.num_clusters * self.slo_generator.num_specs
        assert self.pairing.num_pairs is not None
        return self.pairing.num_pairs


class LabConfigError(ValueError):
    """Raised when a LabConfig cannot be loaded or fails pool validation."""


def config_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]


def profile_path_for(pool_entry: str) -> Path:
    return PROFILE_DIR / f"{pool_entry}.yaml"


def load_pool_profiles(
    config: LabConfig, root: Path
) -> dict[str, AcceleratorProfile]:
    """Load and validate the accelerator pool (DESIGN §3.1 validation rules).

    Placeholder profiles are rejected as an error, not a warning: admitting one
    would let invented hardware numbers into every scenario of the batch.
    """
    profiles: dict[str, AcceleratorProfile] = {}
    for entry in config.cluster_generator.accelerator_pool:
        path = root / profile_path_for(entry)
        if not path.exists():
            raise LabConfigError(
                f"accelerator_pool entry '{entry}': no profile file at {path}"
            )
        profile = load_accelerator_profile(path)
        if profile.source == Source.PLACEHOLDER:
            raise LabConfigError(
                f"accelerator_pool entry '{entry}' has source=placeholder; "
                "ScenarioLab only admits measured or vendor_spec profiles "
                "(DESIGN §0.2). Remove it from the pool."
            )
        profiles[entry] = profile

    missing: list[str] = []
    for model in config.slo_generator.models:
        for entry, profile in profiles.items():
            if not compatibility(model, config.slo_generator.dtype, profile):
                missing.append(
                    f"model '{model}' @ {config.slo_generator.dtype} is not in "
                    f"supported_models of accelerator '{entry}' ({profile.profile_id})"
                )
            elif profile.perf_data is None or profile.sim_hardware is None:
                missing.append(
                    f"accelerator '{entry}' ({profile.profile_id}) has no perf bundle "
                    f"(perf_data={profile.perf_data!r}, sim_hardware={profile.sim_hardware!r}); "
                    f"it cannot predict model '{model}'"
                )
    if missing:
        raise LabConfigError(
            "model/accelerator coverage check failed:\n  - " + "\n  - ".join(missing)
        )
    return profiles


def load_lab_config(path: str | Path, root: str | Path = ".") -> tuple[LabConfig, str]:
    """Load a LabConfig YAML. Returns (config, config_hash).

    Validates the accelerator pool, the model coverage, and the scenario-count
    ceiling before anything runs.
    """
    path = Path(path)
    root = Path(root)
    try:
        raw_text = path.read_text()
    except FileNotFoundError as exc:
        raise LabConfigError(f"{path}: file not found") from exc
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise LabConfigError(f"{path}: invalid YAML - {exc}") from exc
    if not isinstance(raw, dict):
        raise LabConfigError(f"{path}: expected a YAML mapping at the top level")

    try:
        config = LabConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise LabConfigError(f"{path}: {exc}") from exc

    load_pool_profiles(config, root)

    total = config.num_scenarios()
    if total > config.pairing.max_scenarios:
        raise LabConfigError(
            f"{path}: {total} scenarios exceed pairing.max_scenarios="
            f"{config.pairing.max_scenarios}. Reduce num_clusters/num_specs or "
            "switch to pairing.mode: random with num_pairs."
        )
    return config, config_hash(raw_text)
