"""ProfileTier resolution and plan propagation guarantees (STEP 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from planner.optimizer import exhaustive
from planner.render import render
from planner.util.tier import (
    ProfileTier,
    caveat_for,
    min_tier,
    resolve_bundle_tier,
    resolve_bundle_tier_report,
    resolve_variant,
)

from .conftest import MockPredictor

REPO = Path(__file__).resolve().parents[1]
PERF = REPO / "profiler" / "perf"
MODEL = "meta-llama/Llama-3.1-8B"


def _bundle(root: Path, hardware: str, meta: dict | None, *, model: str = MODEL,
            variant: str = "bf16") -> Path:
    variant_root = root / hardware / model / variant
    variant_root.mkdir(parents=True)
    if meta is not None:
        (variant_root / "meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return variant_root


# --- resolution -------------------------------------------------------------

def test_resolve_reads_tier_field(tmp_path):
    """meta.yaml with tier: analytical resolves to ANALYTICAL."""
    _bundle(tmp_path, "X-t0", {"tier": "analytical", "source": "measured"})
    assert resolve_bundle_tier(tmp_path, "X-t0", MODEL, "bf16") is ProfileTier.ANALYTICAL


def test_resolve_falls_back_to_source(tmp_path):
    """Without a tier field, source: measured resolves to MEASURED."""
    _bundle(tmp_path, "X", {"source": "measured"})
    assert resolve_bundle_tier(tmp_path, "X", MODEL, "bf16") is ProfileTier.MEASURED


def test_resolve_returns_unknown_when_no_provenance(tmp_path):
    """Neither tier nor source resolves to UNKNOWN, never MEASURED (A2)."""
    _bundle(tmp_path, "X", {"gpu": "NVIDIA A40"})
    assert resolve_bundle_tier(tmp_path, "X", MODEL, "bf16") is ProfileTier.UNKNOWN


def test_resolve_returns_unknown_when_meta_missing(tmp_path):
    """A bundle directory without meta.yaml has unknown provenance."""
    _bundle(tmp_path, "X", None)
    assert resolve_bundle_tier(tmp_path, "X", MODEL, "bf16") is ProfileTier.UNKNOWN


def test_resolve_returns_placeholder_when_bundle_missing(tmp_path):
    """No bundle directory at all resolves to PLACEHOLDER."""
    assert resolve_bundle_tier(tmp_path, "GHOST", MODEL, "bf16") is ProfileTier.PLACEHOLDER


def test_resolve_reads_sidecar_for_upstream_bundle(tmp_path):
    """A tier.yaml sidecar is consulted when meta.yaml has no tier/source."""
    variant_root = _bundle(tmp_path, "X", {"gpu": "RTX PRO 6000"})
    (variant_root / "tier.yaml").write_text("tier: measured\n", encoding="utf-8")
    assert resolve_bundle_tier(tmp_path, "X", MODEL, "bf16") is ProfileTier.MEASURED


def test_resolve_real_bundles_are_measured():
    """The repo-shipped RTXPRO6000 / A40 / RNGD-CARD bundles resolve MEASURED."""
    for hw in ("RTXPRO6000", "A40", "RNGD-CARD"):
        assert resolve_bundle_tier(PERF, hw, MODEL, "bf16") is ProfileTier.MEASURED, hw


def test_label_suffix_mismatch_warns_not_raises(tmp_path):
    """A -t0 label with meta tier: measured warns and trusts the bundle."""
    _bundle(tmp_path, "X-t0", {"tier": "measured"})
    res = resolve_bundle_tier_report(tmp_path, "X-t0", MODEL, "bf16")
    assert res.tier is ProfileTier.MEASURED
    assert any("claims tier analytical" in w for w in res.warnings)


def test_resolve_variant():
    """Variant naming matches serving/core/trace_generator.py::resolve_variant."""
    assert resolve_variant("bfloat16") == "bf16"
    assert resolve_variant("bfloat16", "auto") == "bf16"
    assert resolve_variant("bfloat16", "fp8") == "bf16-kvfp8"
    assert resolve_variant("float16", "auto") == "fp16"


# --- min tier ---------------------------------------------------------------

@pytest.mark.parametrize(
    "tiers,expected",
    [
        # ANALYTICAL undercuts MEASURED: any analytical input taints the plan.
        ([ProfileTier.MEASURED, ProfileTier.ANALYTICAL], ProfileTier.ANALYTICAL),
        # IMPORTED ranks below MEASURED: an external measurement is real but
        # less verifiable, so a mixed set reports imported (see ProfileTier.rank).
        ([ProfileTier.MEASURED, ProfileTier.IMPORTED], ProfileTier.IMPORTED),
        ([ProfileTier.CALIBRATED, ProfileTier.ANALYTICAL], ProfileTier.ANALYTICAL),
        ([ProfileTier.UNKNOWN, ProfileTier.PLACEHOLDER], ProfileTier.PLACEHOLDER),
        ([], ProfileTier.UNKNOWN),
    ],
)
def test_min_tier(tiers, expected):
    """The least trustworthy tier of a mixed set is selected."""
    assert min_tier(tiers) is expected


def test_is_measurement():
    """Only MEASURED and IMPORTED count as measurements."""
    assert ProfileTier.MEASURED.is_measurement
    assert ProfileTier.IMPORTED.is_measurement
    for t in (ProfileTier.CALIBRATED, ProfileTier.ANALYTICAL,
              ProfileTier.PLACEHOLDER, ProfileTier.UNKNOWN):
        assert not t.is_measurement


def test_caveat_wording():
    """Caveat strings are the exact fixed wording of the work order."""
    assert caveat_for(ProfileTier.ANALYTICAL, "HW") == (
        "simulator-only (analytical inputs): HW profile is datasheet-derived, not measured"
    )
    assert caveat_for(ProfileTier.CALIBRATED, "HW") == (
        "simulator-only (calibrated inputs): HW profile is analytical + limited anchors"
    )
    assert caveat_for(ProfileTier.PLACEHOLDER, "HW") == (
        "no profile bundle for HW: this plan cannot be reported as a result"
    )
    assert caveat_for(ProfileTier.UNKNOWN, "HW") == (
        "profile provenance unknown for HW: meta.yaml records neither tier nor source"
    )
    assert caveat_for(ProfileTier.MEASURED, "HW") is None
    assert caveat_for(ProfileTier.IMPORTED, "HW") is None


# --- propagation ------------------------------------------------------------

def _tier_perf_root(tmp_path: Path, tiers_by_hw: dict[str, str]) -> Path:
    """A perf root whose bundles carry the given tiers (meta.yaml only -
    tier resolution reads provenance, not CSVs)."""
    root = tmp_path / "perf"
    for hw, tier in tiers_by_hw.items():
        _bundle(root, hw, {"tier": tier})
    return root


def test_planner_output_carries_min_tier(spec, cluster, islands, profiles, tmp_path):
    """A heterogeneous plan (measured + analytical islands) reports analytical."""
    perf_root = _tier_perf_root(
        tmp_path, {"A5000": "analytical", "RTXPRO6000": "measured"}
    )
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), perf_root=perf_root
    )
    tiers = output.profile_tiers
    assert any(v == "analytical" for v in tiers.values())
    assert any(v == "measured" for v in tiers.values())
    # The reported plans span both island kinds in the example cluster, so the
    # weakest tier wins.
    assert output.profile_tier == "analytical"
    assert output.provenance["profile_tier"] == "analytical"


def test_caveat_present_for_analytical_plan(spec, cluster, islands, profiles, tmp_path):
    """An analytical-tainted plan carries the exact fixed caveat wording."""
    perf_root = _tier_perf_root(
        tmp_path, {"A5000": "analytical", "RTXPRO6000": "measured"}
    )
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), perf_root=perf_root
    )
    assert (
        "simulator-only (analytical inputs): A5000 profile is datasheet-derived, "
        "not measured"
    ) in output.caveats


def test_no_caveat_for_all_measured_plan(spec, cluster, islands, profiles):
    """All-measured plans gain no tier caveat and report measured (A4)."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.profile_tier == "measured"
    assert not any("simulator-only" in c for c in output.caveats)
    assert not any("provenance unknown" in c for c in output.caveats)


def test_render_shows_banner(spec, cluster, islands, profiles, tmp_path):
    """render() puts a tier banner above everything for a non-measured plan."""
    perf_root = _tier_perf_root(
        tmp_path, {"A5000": "analytical", "RTXPRO6000": "analytical"}
    )
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(), perf_root=perf_root
    )
    text = render(output)
    first_lines = text.splitlines()[:3]
    assert any("PROFILE TIER: ANALYTICAL" in line for line in first_lines)


def test_render_no_banner_for_measured(spec, cluster, islands, profiles):
    """Measured plans render exactly without a banner (A4)."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert "PROFILE TIER" not in render(output)


def test_provenance_block_includes_tier(spec, cluster, islands, profiles):
    """The plan's provenance carries profile_tier and per-island profile_tiers."""
    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert output.provenance["profile_tier"] == output.profile_tier
    assert output.provenance["profile_tiers"] == output.profile_tiers
    assert set(output.profile_tiers) == {i.id for i in islands}


def test_plan_yaml_roundtrip_preserves_tier(spec, cluster, islands, profiles):
    """profile_tier survives a model_dump -> model_validate round trip."""
    from planner.plan import PlannerOutput

    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    restored = PlannerOutput.model_validate(output.model_dump(mode="json"))
    assert restored.profile_tier == output.profile_tier
    assert restored.profile_tiers == output.profile_tiers
