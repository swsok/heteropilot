"""Roofline-as-constraint cost model (STEP 5) - the heart of Tier 0.

    t = max( flops / (peak_flops * eff_c),
             bytes / (bw * eff_m),
             kernel_launch_us * 1e-6 )

Design rationale: the roofline is used as a CONSTRAINT with an efficiency
derating inside it, not as a predictor. NeuSight (ASPLOS'25), KernelSight-LM
(2026 preprint) and GenZ arrived at this formulation independently. The
``kernel_launch_us`` floor is KernelSight-LM's t_0 term and dominates at
tiny sizes.
"""

from __future__ import annotations

from typing import Protocol

from profiler.synth.device import S_TO_US, DeviceSpec
from profiler.synth.shapes import OpCost


class ScalingTable(Protocol):
    """Tier 1's per-kernel-family multiplier table (filled in STEP 9).

    ``feature`` is a size proxy for the op - RooflineModel passes
    ``max(flops, bytes_moved)`` - so a piecewise Tier 1 fit can scale small
    ops differently from large ones. Tier 0 passes None and gets 1.0.
    """

    def scale(self, family: str, feature: float) -> float: ...


class RooflineModel:
    """OpCost + DeviceSpec -> time_us, optionally rescaled by a Tier 1 table."""

    def __init__(self, device: DeviceSpec, scaling: ScalingTable | None = None) -> None:
        self.device = device
        self.scaling = scaling

    def estimate_us(self, cost: OpCost) -> float:
        flops_eff, mem_eff = self.device.efficiency(cost.family)
        compute_s = cost.flops / (self.device.peak_flops * flops_eff)
        memory_s = cost.bytes_moved / (self.device.mem_bandwidth_bytes * mem_eff)
        launch_s = self.device.kernel_launch_us / S_TO_US
        t_us = max(compute_s, memory_s, launch_s) * S_TO_US
        if self.scaling is not None:
            t_us *= self.scaling.scale(cost.family, max(cost.flops, cost.bytes_moved))
        return t_us
