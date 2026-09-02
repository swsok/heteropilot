"""Datasheet schema and validation rules (STEP 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from planner.inventory import AcceleratorProfile, Datasheet, load_accelerator_profile

REPO = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO / "profiles" / "accelerators"


def _minimal_profile(**overrides) -> dict:
    base = {
        "profile_id": "test",
        "vendor": "TestCorp",
        "model": "T1000",
        "backend": "cuda",
        "memory_gb": 24,
        "memory_bandwidth_gbps": 500,
    }
    base.update(overrides)
    return base


def test_all_shipped_profiles_still_load():
    """Every profiles/accelerators/*.yaml still parses (backward compat).

    *.efficiency.yaml files are STEP 8 fit artifacts, not profiles.
    """
    files = sorted(
        p for p in PROFILE_DIR.glob("*.yaml") if not p.name.endswith(".efficiency.yaml")
    )
    assert len(files) >= 7
    for path in files:
        load_accelerator_profile(path)


def test_datasheet_optional():
    """A profile without a datasheet block is valid."""
    profile = AcceleratorProfile.model_validate(_minimal_profile())
    assert profile.datasheet is None


def test_t0_sim_hardware_requires_datasheet():
    """sim_hardware ending in -t0 without a datasheet is rejected (A2)."""
    with pytest.raises(ValidationError, match="datasheet"):
        AcceleratorProfile.model_validate(_minimal_profile(sim_hardware="T1000-t0"))
    with pytest.raises(ValidationError, match="datasheet"):
        AcceleratorProfile.model_validate(_minimal_profile(sim_hardware="T1000-t1"))


def test_t0_sim_hardware_with_datasheet_accepted():
    """The -t0 label + a sourced datasheet is the intended Tier 0 shape."""
    profile = AcceleratorProfile.model_validate(
        _minimal_profile(
            sim_hardware="T1000-t0",
            datasheet={"peak_tflops": {"bf16": 100.0}, "datasheet_source": "vendor doc"},
        )
    )
    assert profile.datasheet is not None
    assert profile.datasheet.peak_tflops["bf16"] == 100.0


def test_datasheet_requires_source():
    """A datasheet without datasheet_source is rejected (rule 3)."""
    with pytest.raises(ValidationError, match="datasheet_source"):
        AcceleratorProfile.model_validate(
            _minimal_profile(datasheet={"peak_tflops": {"bf16": 100.0}})
        )


def test_efficiency_bounds():
    """flops/mem efficiency accept only (0, 1]."""
    for field in ("flops_efficiency", "mem_efficiency"):
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValidationError):
                Datasheet.model_validate({field: bad, "datasheet_source": "x"})
        ok = Datasheet.model_validate({field: 1.0, "datasheet_source": "x"})
        assert getattr(ok, field) == 1.0


def test_family_efficiency_bounds():
    """family_efficiency values outside (0, 1] are rejected."""
    with pytest.raises(ValidationError, match="family_efficiency"):
        Datasheet.model_validate(
            {"family_efficiency": {"gemm": 1.2}, "datasheet_source": "x"}
        )


def test_a40_datasheet_has_source_urls():
    """a40.yaml's datasheet values are attributed, with URLs in the YAML."""
    profile = load_accelerator_profile(PROFILE_DIR / "a40.yaml")
    assert profile.datasheet is not None
    assert profile.datasheet.datasheet_source.strip()
    # 149.7 is the DENSE BF16 Tensor Core rate (299.4 is the sparsity figure).
    # Cross-checked in STEP 5: the measured A40 bundle sustains 81.3 TFLOP/s,
    # which would breach a 74.8 half-rate misreading.
    assert profile.datasheet.peak_tflops.get("bf16") == 149.7
    assert profile.datasheet.compute_units == 84
    raw = (PROFILE_DIR / "a40.yaml").read_text(encoding="utf-8")
    assert "https://" in raw


def test_a40_efficiency_not_prefilled():
    """A1: A40 efficiencies stay empty until the STEP 8 fit exists.

    When STEP 8 merges fitted values into a40.yaml, update this test and say
    so in that PR.
    """
    profile = load_accelerator_profile(PROFILE_DIR / "a40.yaml")
    assert profile.datasheet is not None
    assert profile.datasheet.flops_efficiency is None
    assert profile.datasheet.mem_efficiency is None


def test_ascend_target_still_placeholder():
    """ascend_target.yaml is untouched by STEP 3 (datasheet arrives in STEP 10)."""
    raw = yaml.safe_load((PROFILE_DIR / "ascend_target.yaml").read_text())
    assert raw.get("sim_hardware") is None
    assert "datasheet" not in raw
