"""Profile backends: what turns a bundle key into a time (STEP 7/9).

``ProfileBackend`` is the seam between key enumeration (``emit.py``) and the
cost model. ``AnalyticalProfileBackend`` is Tier 0 (pure roofline);
``CalibratedProfileBackend`` is Tier 1 (the same roofline rescaled by a
per-kernel-family ScalingTable fitted from anchors in STEP 9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from profiler.core.config import Architecture
from profiler.synth.attn import AttentionCostModel
from profiler.synth.device import DeviceSpec
from profiler.synth.dims import ModelDims
from profiler.synth.roofline import RooflineModel, ScalingTable
from profiler.synth.shapes import ShapeResolver


class ProfileBackend(ABC):
    """Times for every key kind a bundle contains."""

    #: The meta.yaml tier/source value bundles from this backend carry.
    tier: str

    @abstractmethod
    def dense_us(self, layer: str, tokens: int) -> float: ...

    @abstractmethod
    def per_sequence_us(self, layer: str, sequences: int) -> float: ...

    @abstractmethod
    def attention_us(
        self, prefill_chunk: int, kv_prefill: int, n_decode: int, kv_decode: int
    ) -> float: ...

    @abstractmethod
    def expert_us(self, tokens: int, activated_experts: int) -> float: ...


class AnalyticalProfileBackend(ProfileBackend):
    """Tier 0: datasheet roofline, zero target-hardware measurements."""

    tier = "analytical"

    def __init__(
        self,
        dims: ModelDims,
        arch: Architecture,
        device: DeviceSpec,
        tp: int,
        *,
        bytes_mode: str = "sum",
        attn_mode: str = "max",
        scaling: ScalingTable | None = None,
    ) -> None:
        self.dims = dims
        self.device = device
        self.tp = tp
        self.bytes_mode = bytes_mode
        self.attn_mode = attn_mode
        self.resolver = ShapeResolver(dims, arch, tp, bytes_mode=bytes_mode)
        self.roofline = RooflineModel(device, scaling)
        self.attention = AttentionCostModel(
            dims, device, tp, mode=attn_mode, scaling=scaling
        )

    def dense_us(self, layer: str, tokens: int) -> float:
        return self.roofline.estimate_us(self.resolver.dense(layer, tokens))

    def per_sequence_us(self, layer: str, sequences: int) -> float:
        return self.roofline.estimate_us(self.resolver.per_sequence(layer, sequences))

    def attention_us(
        self, prefill_chunk: int, kv_prefill: int, n_decode: int, kv_decode: int
    ) -> float:
        return self.attention.estimate_key_us(
            prefill_chunk, kv_prefill, n_decode, kv_decode
        )

    def expert_us(self, tokens: int, activated_experts: int) -> float:
        return self.roofline.estimate_us(
            self.resolver.expert(tokens, activated_experts)
        )


class CalibratedProfileBackend(AnalyticalProfileBackend):
    """Tier 1: the analytical roofline rescaled by measured anchors.

    The ScalingTable is mandatory here; it reaches the GEMM/elementwise
    paths through RooflineModel and the attention path through
    AttentionCostModel, both using the same feature convention.
    """

    tier = "calibrated"

    def __init__(self, *args, scaling: ScalingTable, **kwargs) -> None:
        super().__init__(*args, scaling=scaling, **kwargs)
