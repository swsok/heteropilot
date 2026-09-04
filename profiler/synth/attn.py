"""Tier 0 cost model for the 4-axis attention grid (STEP 6).

Attention gets its own model because it is the one kernel family that does
not transfer across devices (KernelSight-LM reports up to 3.8x cross-device
efficiency variance; LLMCompass sees 2x+ error on softmax-adjacent ops;
Vidur buckets attention separately by design). It also dominates bundle row
counts, so if only one thing is ever measured per hardware, it should be
attention (the basis for STEP 9's anchor budget split).
"""

from __future__ import annotations

from typing import Protocol

from profiler.synth.device import S_TO_US, DeviceSpec
from profiler.synth.dims import ModelDims

#: How the fused-step time combines prefill and decode work. "max" applies
#: one roofline to the summed FLOPs/bytes; "sum" prices the two phases as
#: separate kernels and adds them. STEP 8's measured diff chose "sum" as the
#: default: variant V3 (bytes=sum, attn=sum) gives the lowest overall MAPE on
#: both fitted hardware (A40 38.9%% vs 40.3%%, RTXPRO6000 33.0%% vs 37.1%%);
#: see docs/tier0_calibration.md.
ATTN_MODES = ("max", "sum")


class AttnError(ValueError):
    """An attention key the model cannot price."""


class AttentionKey(Protocol):
    """Structural view of one attention.csv key (profiler AttentionPoint
    satisfies this without synth importing the torch-tainted categories
    module)."""

    prefill_chunk: int
    kv_prefill: int
    n_decode: int
    kv_decode: int


class AttentionCostModel:
    """Attention cost of one mixed (chunked-prefill + decode) step.

    A step may carry a prefill chunk and decode sequences at once. Two
    combination modes exist: "max" prices one fused roofline over the summed
    FLOPs/bytes; "sum" prices the phases separately and adds them. STEP 8's
    measured diff adopted "sum" as the default (see ATTN_MODES above).

    Degenerate keys: the measured attention.csv contains no row with
    prefill_chunk == 0 and n_decode == 0 (verified 2026-09-02 on the A40
    tp1 bundle: zero such rows; prefill_chunk == 0 always means a
    decode-only step per CONTRACT.md). An all-zero key therefore raises.
    """

    def __init__(
        self,
        dims: ModelDims,
        device: DeviceSpec,
        tp: int,
        *,
        mode: str = "sum",
        scaling=None,
    ) -> None:
        if tp < 1:
            raise AttnError(f"tp must be >= 1, got {tp}")
        if mode not in ATTN_MODES:
            raise AttnError(f"mode {mode!r} not in {ATTN_MODES}")
        self.dims = dims
        self.device = device
        self.tp = tp
        self.mode = mode
        #: Optional Tier 1 ScalingTable (see roofline.ScalingTable); applied
        #: with the same feature convention as RooflineModel.
        self.scaling = scaling

    # ------------------------------------------------------------------

    def _phase_costs(self, pc: int, kv_prefill: int, n_decode: int,
                     kv_decode: int) -> tuple[float, float, float, float]:
        """(flops_prefill, bytes_prefill, flops_decode, bytes_decode)."""
        d = self.dims
        n_q_l = d.num_attention_heads / self.tp
        n_kv_l = d.num_key_value_heads / self.tp

        # Prefill: causal attention over the chunk. 4 = QK^T (2) + PV (2);
        # the chunk sees the full kv_prefill history plus, on average, half
        # of itself (causal mask).
        flops_prefill = 4.0 * n_q_l * d.head_dim * pc * (kv_prefill + pc / 2)
        # QKV activation read + KV write for the chunk (Appendix A draft).
        bytes_prefill = pc * (n_q_l + 2 * n_kv_l) * d.head_dim * d.dtype_bytes
        if pc > 0 and kv_prefill > 0:
            # Reading the existing prefix KV: GQA reads n_kv heads, in the
            # KV-cache dtype.
            bytes_prefill += kv_prefill * 2 * n_kv_l * d.head_dim * d.kv_dtype_bytes

        # Decode: memory-bound KV streaming. GQA: bytes scale with the KV
        # heads (n_kv), not the query heads - pricing with n_q would
        # overcharge Llama-3.1-8B by exactly 4x (n_q=32, n_kv=8).
        flops_decode = 4.0 * n_q_l * d.head_dim * n_decode * kv_decode
        bytes_decode = (
            n_decode * kv_decode * 2 * n_kv_l * d.head_dim * d.kv_dtype_bytes
        )
        return flops_prefill, bytes_prefill, flops_decode, bytes_decode

    def _roofline_us(self, flops: float, bytes_moved: float) -> float:
        flops_eff, mem_eff = self.device.efficiency("attention")
        compute_s = flops / (self.device.peak_flops * flops_eff)
        memory_s = bytes_moved / (self.device.mem_bandwidth_bytes * mem_eff)
        launch_s = self.device.kernel_launch_us / S_TO_US
        return max(compute_s, memory_s, launch_s) * S_TO_US

    def estimate_key_us(self, prefill_chunk: int, kv_prefill: int,
                        n_decode: int, kv_decode: int) -> float:
        """time_us for one raw 4-axis key."""
        if min(prefill_chunk, kv_prefill, n_decode, kv_decode) < 0:
            raise AttnError("attention key axes must be non-negative")
        if prefill_chunk == 0 and n_decode == 0:
            raise AttnError(
                "degenerate attention key (prefill_chunk=0, n_decode=0): no "
                "such row exists in measured bundles (verified 2026-09-02) "
                "and a zero-work step has no meaning"
            )
        fp, bp, fd, bd = self._phase_costs(
            prefill_chunk, kv_prefill, n_decode, kv_decode
        )
        if self.mode == "max":
            t_us = self._roofline_us(fp + fd, bp + bd)
        else:
            prefill_us = self._roofline_us(fp, bp) if prefill_chunk > 0 else 0.0
            decode_us = self._roofline_us(fd, bd) if n_decode > 0 else 0.0
            t_us = max(prefill_us + decode_us, self.device.kernel_launch_us)
        if self.scaling is not None:
            # feature = the unscaled Tier 0 time (see roofline.ScalingTable).
            t_us *= self.scaling.scale("attention", t_us)
        return t_us

    def estimate_us(self, point: AttentionKey) -> float:
        """time_us for one profiler AttentionPoint-compatible key."""
        return self.estimate_key_us(
            point.prefill_chunk, point.kv_prefill, point.n_decode, point.kv_decode
        )
