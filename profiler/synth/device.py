"""Normalized device parameters for the Tier 0 cost model (STEP 5).

``DeviceSpec`` turns an ``AcceleratorProfile``'s datasheet block into the
values the roofline actually computes with, converting units exactly once
(datasheet TFLOP/s and GB/s -> FLOP/s and B/s). Missing required values
raise - a Tier 0 estimate built on invented numbers would violate absolute
rule A2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # planner imports are annotation-only; synth stays lean
    from planner.inventory import AcceleratorProfile

#: Unit conversions, fixed by test (STEP 5 instruction 4).
TFLOPS_TO_FLOPS = 1e12   # datasheet peak_tflops [TFLOP/s] -> FLOP/s
GBPS_TO_BPS = 1e9        # memory_bandwidth_gbps [GB/s]   -> B/s
S_TO_US = 1e6            # seconds -> the CSVs' time_us


class DeviceSpecError(ValueError):
    """The profile cannot supply the values Tier 0 needs (A2: no defaults)."""


@dataclass(frozen=True)
class DeviceSpec:
    """Datasheet values normalized for computation."""

    label: str
    peak_flops: float           # FLOP/s for the chosen dtype
    mem_bandwidth_bytes: float  # B/s
    flops_efficiency: float
    mem_efficiency: float
    family_efficiency: dict[str, float] = field(default_factory=dict)
    #: Kernel-launch floor in us. 0.0 is the identity (no floor), which is
    #: what a datasheet without a sourced launch-overhead figure yields (A2:
    #: missing data is identity/null, never an invented positive number).
    kernel_launch_us: float = 0.0
    source: str = ""

    @classmethod
    def from_profile(cls, profile: AcceleratorProfile, dtype: str) -> DeviceSpec:
        """Build a DeviceSpec from a profile's datasheet; raise on gaps (A2)."""
        ds = profile.datasheet
        if ds is None:
            raise DeviceSpecError(
                f"profile {profile.profile_id}: no datasheet block - Tier 0 "
                f"generation is impossible without datasheet values (A2)"
            )
        if dtype not in ds.peak_tflops:
            raise DeviceSpecError(
                f"profile {profile.profile_id}: datasheet has no peak_tflops "
                f"entry for dtype {dtype!r} (available: {sorted(ds.peak_tflops)})"
            )
        if ds.flops_efficiency is None or ds.mem_efficiency is None:
            raise DeviceSpecError(
                f"profile {profile.profile_id}: datasheet lacks "
                f"flops_efficiency/mem_efficiency - they must be fitted from "
                f"measurements, never assumed 1.0 (A2)"
            )
        bandwidth_gbps = ds.memory_bandwidth_gbps or profile.memory_bandwidth_gbps
        return cls(
            label=profile.sim_hardware or profile.model,
            peak_flops=ds.peak_tflops[dtype] * TFLOPS_TO_FLOPS,
            mem_bandwidth_bytes=bandwidth_gbps * GBPS_TO_BPS,
            flops_efficiency=ds.flops_efficiency,
            mem_efficiency=ds.mem_efficiency,
            family_efficiency=dict(ds.family_efficiency),
            kernel_launch_us=ds.kernel_launch_us or 0.0,
            source=ds.datasheet_source,
        )

    def efficiency(self, family: str) -> tuple[float, float]:
        """(flops_eff, mem_eff) with per-family overrides applied.

        ``family_efficiency[family]`` overrides both terms;
        ``family_efficiency[f"{family}_mem"]`` additionally overrides the
        memory term alone (attention's compute and bandwidth deratings can
        differ - STEP 6 background).
        """
        flops_eff = self.family_efficiency.get(family, self.flops_efficiency)
        mem_eff = self.family_efficiency.get(
            f"{family}_mem", self.family_efficiency.get(family, self.mem_efficiency)
        )
        return flops_eff, mem_eff
