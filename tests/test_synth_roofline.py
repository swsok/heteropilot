"""RooflineModel numeric and boundary behavior (STEP 5)."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

from planner.inventory import load_accelerator_profile
from profiler.core.config import load_architecture
from profiler.synth.device import (
    GBPS_TO_BPS,
    S_TO_US,
    TFLOPS_TO_FLOPS,
    DeviceSpec,
    DeviceSpecError,
)
from profiler.synth.dims import ModelDims
from profiler.synth.roofline import RooflineModel
from profiler.synth.shapes import OpCost, ShapeResolver

REPO = Path(__file__).resolve().parents[1]
A40_YAML = REPO / "profiles" / "accelerators" / "a40.yaml"
A40_TP1 = REPO / "profiler" / "perf" / "A40" / "meta-llama" / "Llama-3.1-8B" / "bf16" / "tp1"
LLAMA_CFG = REPO / "configs" / "model" / "meta-llama" / "Llama-3.1-8B.json"
LLAMA_ARCH = REPO / "profiler" / "models" / "llama.yaml"


def _device(**overrides) -> DeviceSpec:
    values = {
        "label": "TEST",
        "peak_flops": 100e12,
        "mem_bandwidth_bytes": 1e12,
        "flops_efficiency": 1.0,
        "mem_efficiency": 1.0,
        "kernel_launch_us": 2.0,
    }
    values.update(overrides)
    return DeviceSpec(**values)


def test_compute_bound_branch():
    """High arithmetic intensity selects the compute term."""
    # 1e12 FLOPs at 100 TFLOP/s = 10 ms; 1e6 bytes at 1 TB/s = 1 us.
    t = RooflineModel(_device()).estimate_us(OpCost(1e12, 1e6, "gemm"))
    assert t == pytest.approx(10_000.0)  # 10 ms in us, exact arithmetic


def test_memory_bound_branch():
    """Low arithmetic intensity selects the memory term."""
    # 1e6 FLOPs = 0.01 us of compute; 1e9 bytes at 1 TB/s = 1000 us.
    t = RooflineModel(_device()).estimate_us(OpCost(1e6, 1e9, "gemm"))
    assert t == pytest.approx(1000.0)


def test_launch_floor_dominates_at_tiny_size():
    """Near-zero work returns the kernel-launch floor."""
    t = RooflineModel(_device()).estimate_us(OpCost(1.0, 1.0, "elementwise"))
    assert t == pytest.approx(2.0)  # the injected kernel_launch_us


def test_unit_conversions():
    """GB/s, TFLOP/s and us conversion constants are exact."""
    assert TFLOPS_TO_FLOPS == 1e12
    assert GBPS_TO_BPS == 1e9
    assert S_TO_US == 1e6
    # peak 1 TFLOP/s, 2e12 FLOPs, eff=1.0 -> 2.0 s = 2e6 us.
    dev = _device(peak_flops=1 * TFLOPS_TO_FLOPS, kernel_launch_us=0.0)
    t = RooflineModel(dev).estimate_us(OpCost(2e12, 1.0, "gemm"))
    assert t == pytest.approx(2e6)


def test_efficiency_scales_linearly():
    """Halving an efficiency doubles that term's time."""
    base = RooflineModel(_device(kernel_launch_us=0.0)).estimate_us(OpCost(1e12, 1.0, "gemm"))
    half = RooflineModel(
        _device(flops_efficiency=0.5, kernel_launch_us=0.0)
    ).estimate_us(OpCost(1e12, 1.0, "gemm"))
    assert half == pytest.approx(2 * base)


def test_family_efficiency_override():
    """family_efficiency['attention'] overrides the default efficiencies."""
    dev = _device(
        flops_efficiency=1.0,
        mem_efficiency=1.0,
        family_efficiency={"attention": 0.5},
        kernel_launch_us=0.0,
    )
    gemm = RooflineModel(dev).estimate_us(OpCost(1e12, 1.0, "gemm"))
    attn = RooflineModel(dev).estimate_us(OpCost(1e12, 1.0, "attention"))
    assert attn == pytest.approx(2 * gemm)
    # A separate memory-side override is honored independently.
    assert dev.efficiency("attention") == (0.5, 0.5)
    dev2 = _device(family_efficiency={"attention": 0.5, "attention_mem": 0.25})
    assert dev2.efficiency("attention") == (0.5, 0.25)


def test_missing_datasheet_raises():
    """A profile without a datasheet cannot make a DeviceSpec (A2)."""
    from planner.inventory import AcceleratorProfile

    profile = AcceleratorProfile.model_validate({
        "profile_id": "p", "vendor": "v", "model": "m", "backend": "cuda",
        "memory_gb": 24, "memory_bandwidth_gbps": 500,
    })
    with pytest.raises(DeviceSpecError, match="no datasheet"):
        DeviceSpec.from_profile(profile, "bf16")


def test_missing_efficiency_raises():
    """flops_efficiency=None raises; 1.0 is never assumed (A2)."""
    profile = load_accelerator_profile(A40_YAML)
    assert profile.datasheet is not None and profile.datasheet.flops_efficiency is None
    with pytest.raises(DeviceSpecError, match="efficiency"):
        DeviceSpec.from_profile(profile, "bf16")


def test_unknown_dtype_raises():
    """A dtype absent from peak_tflops raises."""
    profile = load_accelerator_profile(A40_YAML)
    with pytest.raises(DeviceSpecError, match="fp8"):
        DeviceSpec.from_profile(profile, "fp8")


def test_scaling_table_none_is_identity():
    """No ScalingTable means multiplier 1.0."""
    cost = OpCost(1e12, 1e6, "gemm")
    dev = _device()

    class Doubler:
        def scale(self, family: str, feature: float) -> float:
            return 2.0

    assert RooflineModel(dev, None).estimate_us(cost) * 2 == pytest.approx(
        RooflineModel(dev, Doubler()).estimate_us(cost)
    )


def test_monotonic_in_cost():
    """More flops or more bytes never yields less time."""
    model = RooflineModel(_device())
    prev = 0.0
    for scale in (1, 4, 16, 64, 256):
        t = model.estimate_us(OpCost(1e9 * scale, 1e6 * scale, "gemm"))
        assert t >= prev
        prev = t


# --- measured-bundle comparison (the reason this STEP exists) ---------------

def test_order_of_magnitude_against_real_bundle():
    """Theoretical lower bound (eff=1.0) vs the measured A40 bundle.

    With eff=1.0 injected, the estimate must not exceed the measurement
    except within a 6% tolerance: small elementwise activations (< 6 MB) fit
    the A40's L2 and beat the 696 GB/s DRAM figure - worst observed +5.3%
    on rotary_emb at T=240; 99% of points satisfy the strict bound. The
    median est/measured ratio must land in [0.05, 1.0]: a breach means a
    shape formula or a unit conversion is wrong by an order of magnitude.
    """
    dims = ModelDims.from_hf_config(LLAMA_CFG, "bf16")
    arch = load_architecture(LLAMA_ARCH)
    profile = load_accelerator_profile(A40_YAML)
    assert profile.datasheet is not None
    # eff=1.0 injected explicitly - the theoretical-lower-bound mode. The
    # peak and bandwidth come from the sourced datasheet values.
    device = DeviceSpec(
        label="A40",
        peak_flops=profile.datasheet.peak_tflops["bf16"] * TFLOPS_TO_FLOPS,
        mem_bandwidth_bytes=profile.memory_bandwidth_gbps * GBPS_TO_BPS,
        flops_efficiency=1.0,
        mem_efficiency=1.0,
        kernel_launch_us=0.0,
    )
    resolver = ShapeResolver(dims, arch, tp=1, bytes_mode="sum")
    model = RooflineModel(device)

    ratios: list[float] = []
    strict_violations = 0
    for filename, cost_fn, key in (
        ("dense.csv", resolver.dense, "tokens"),
        ("per_sequence.csv", resolver.per_sequence, "sequences"),
    ):
        with (A40_TP1 / filename).open() as f:
            for row in csv.DictReader(f):
                est = model.estimate_us(cost_fn(row["layer"], int(row[key])))
                measured = float(row["time_us"])
                ratio = est / measured
                ratios.append(ratio)
                if ratio > 1.0:
                    strict_violations += 1
                assert ratio <= 1.06, (
                    f"{filename} {row['layer']} @{row[key]}: est {est:.2f}us > "
                    f"measured {measured:.2f}us by more than the 6% L2 allowance"
                )

    assert len(ratios) > 1000  # the bundle is dense; a thin read means a bug
    assert strict_violations / len(ratios) < 0.01
    med = statistics.median(ratios)
    assert 0.05 <= med <= 1.0, f"median est/measured {med:.4f} outside [0.05, 1.0]"
