"""Profile-tier resolution and propagation (WORK_ORDER_tiered_profiles.md STEP 2).

A plan is only as trustworthy as the least trustworthy profile bundle it was
simulated on. This module reads each bundle's tier from its ``meta.yaml``,
folds a set of tiers to the weakest one, and produces the fixed caveat
wording that must travel with any plan built on non-measured inputs
(absolute rule A1: analytical numbers are never presented as measured).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Default bundle root, relative to the repo (planner/util/tier.py -> repo root).
DEFAULT_PERF_ROOT = Path(__file__).resolve().parents[2] / "profiler" / "perf"

#: Mapping from runtime dtype spellings to the profiler's short names.
#: Copied from serving/core/trace_generator.py::_DTYPE_SHORT (the simulator is
#: run as a subprocess, so the planner does not import serving; see the
#: planner/util/memory.py precedent).
_DTYPE_SHORT = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "half": "fp16", "fp16": "fp16",
    "float32": "fp32", "float": "fp32", "fp32": "fp32",
    "fp8": "fp8", "fp8_e4m3": "fp8",
    "int8": "int8", "int4": "int4",
}


def resolve_variant(dtype: str, kv_cache_dtype: str = "auto") -> str:
    """The profiler's variant folder name for a (dtype, kv dtype) choice.

    Mirrors serving/core/trace_generator.py::resolve_variant.
    """
    parts = [_DTYPE_SHORT.get(str(dtype), str(dtype))]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        parts.append(f"kv{_DTYPE_SHORT.get(str(kv_cache_dtype), str(kv_cache_dtype))}")
    return "-".join(parts)


class ProfileTier(str, enum.Enum):
    """How trustworthy a profile bundle's numbers are."""

    MEASURED = "measured"        # tier 2 - measured in-house
    IMPORTED = "imported"        # tier 2 - external measurement, imported
    CALIBRATED = "calibrated"    # tier 1 - analytical + measured anchors
    ANALYTICAL = "analytical"    # tier 0 - datasheet-derived
    PLACEHOLDER = "placeholder"  # no bundle / numbers not trustworthy
    UNKNOWN = "unknown"          # meta.yaml records neither tier nor source

    @property
    def rank(self) -> int:
        """Trust ordering; lower is less trustworthy (min_tier picks the lowest).

        PLACEHOLDER < UNKNOWN: no data at all is worse than data of unknown
        provenance. IMPORTED < MEASURED: an external measurement is real but
        less verifiable than one produced in-house, so a mixed
        measured+imported plan reports ``imported``.
        """
        return {
            ProfileTier.PLACEHOLDER: 0,
            ProfileTier.UNKNOWN: 1,
            ProfileTier.ANALYTICAL: 2,
            ProfileTier.CALIBRATED: 3,
            ProfileTier.IMPORTED: 4,
            ProfileTier.MEASURED: 5,
        }[self]

    @property
    def is_measurement(self) -> bool:
        """True only for tiers whose numbers came from real hardware runs."""
        return self in (ProfileTier.MEASURED, ProfileTier.IMPORTED)


#: meta.yaml `tier`/`source` string -> ProfileTier. Unknown strings map to
#: UNKNOWN rather than raising: a malformed bundle must degrade trust, not
#: crash the plan.
_TIER_BY_NAME = {t.value: t for t in ProfileTier}

#: Hardware-label suffix -> tier the label claims (CONTRACT.md label rule).
_SUFFIX_TIER = {"-t0": ProfileTier.ANALYTICAL, "-t1": ProfileTier.CALIBRATED}


@dataclass(frozen=True)
class TierResolution:
    """Tier plus any label/metadata inconsistency warnings (never exceptions)."""

    tier: ProfileTier
    warnings: list[str] = field(default_factory=list)


def resolve_bundle_tier_report(
    perf_root: Path | None,
    hardware: str,
    model: str,
    variant: str,
) -> TierResolution:
    """Resolve a bundle's tier from its ``meta.yaml``.

    Priority (A2 - never guess upward):
      1. ``tier`` in meta.yaml.
      2. ``tier`` in a fork-owned ``tier.yaml`` sidecar next to meta.yaml -
         exists only for bundles whose meta.yaml is an upstream-tracked file
         this fork must not edit (absolute rule 1), e.g. RTXPRO6000.
      3. ``source`` in meta.yaml.
      4. Neither -> UNKNOWN. Never assumed MEASURED.
    A missing bundle directory is PLACEHOLDER. The hardware label's -t0/-t1
    suffix is a secondary signal only: a contradiction with meta.yaml yields
    a warning, never an exception.
    """
    root = Path(perf_root) if perf_root is not None else DEFAULT_PERF_ROOT
    variant_root = root / hardware / model / variant
    warnings: list[str] = []

    suffix_tier = next(
        (t for suffix, t in _SUFFIX_TIER.items() if hardware.endswith(suffix)), None
    )

    if not variant_root.is_dir():
        return TierResolution(ProfileTier.PLACEHOLDER, warnings)

    meta = _load_yaml_mapping(variant_root / "meta.yaml")
    declared: ProfileTier | None = None
    if "tier" in meta:
        raw = str(meta["tier"])
        declared = _TIER_BY_NAME.get(raw)
        if declared is None:
            warnings.append(
                f"{variant_root}/meta.yaml: unrecognized tier {raw!r}; treating as unknown"
            )
            declared = ProfileTier.UNKNOWN
    else:
        sidecar = _load_yaml_mapping(variant_root / "tier.yaml")
        if "tier" in sidecar:
            raw = str(sidecar["tier"])
            declared = _TIER_BY_NAME.get(raw)
            if declared is None:
                warnings.append(
                    f"{variant_root}/tier.yaml: unrecognized tier {raw!r}; treating as unknown"
                )
                declared = ProfileTier.UNKNOWN
        elif "source" in meta:
            raw = str(meta["source"])
            declared = _TIER_BY_NAME.get(raw)
            if declared is None:
                warnings.append(
                    f"{variant_root}/meta.yaml: unrecognized source {raw!r}; "
                    f"treating as unknown"
                )
                declared = ProfileTier.UNKNOWN

    if declared is None:
        declared = ProfileTier.UNKNOWN

    if suffix_tier is not None and declared is not suffix_tier:
        warnings.append(
            f"hardware label {hardware!r} claims tier {suffix_tier.value} by its "
            f"suffix but the bundle records {declared.value}; trusting the bundle "
            f"(meta.yaml is the single source of truth)"
        )

    return TierResolution(declared, warnings)


def resolve_bundle_tier(
    perf_root: Path | None,
    hardware: str,
    model: str,
    variant: str,
) -> ProfileTier:
    """Tier only; see resolve_bundle_tier_report for the warning-carrying form."""
    return resolve_bundle_tier_report(perf_root, hardware, model, variant).tier


def min_tier(tiers: Iterable[ProfileTier]) -> ProfileTier:
    """The least trustworthy tier of the set; UNKNOWN for an empty input."""
    lowest: ProfileTier | None = None
    for t in tiers:
        if lowest is None or t.rank < lowest.rank:
            lowest = t
    return lowest if lowest is not None else ProfileTier.UNKNOWN


def caveat_for(tier: ProfileTier, hardware: str) -> str | None:
    """The fixed caveat wording for a non-measurement tier; None for tier 2."""
    if tier is ProfileTier.ANALYTICAL:
        return (
            f"simulator-only (analytical inputs): {hardware} profile is "
            f"datasheet-derived, not measured"
        )
    if tier is ProfileTier.CALIBRATED:
        return (
            f"simulator-only (calibrated inputs): {hardware} profile is "
            f"analytical + limited anchors"
        )
    if tier is ProfileTier.PLACEHOLDER:
        return f"no profile bundle for {hardware}: this plan cannot be reported as a result"
    if tier is ProfileTier.UNKNOWN:
        return (
            f"profile provenance unknown for {hardware}: meta.yaml records "
            f"neither tier nor source"
        )
    return None


def _load_yaml_mapping(path: Path) -> dict:
    """Read a YAML mapping; a missing/invalid file is an empty mapping."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}
