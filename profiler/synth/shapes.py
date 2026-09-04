"""Per-layer FLOPs / moved-bytes resolution (STEP 4).

``ShapeResolver`` maps ``(layer name, tokens-or-sequences)`` to an ``OpCost``
using the shape formulas of WORK_ORDER_tiered_profiles.md Appendix A. Layer
names and TP behavior come from the architecture yaml
(``profiler/models/<model_type>.yaml``), never from guesses; an unknown name
raises instead of silently costing zero (a free layer would corrupt every
downstream estimate without a trace).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiler.core.config import Architecture
from profiler.synth.dims import ModelDims

#: How GEMM bytes_moved treats weight reuse. "sum" charges weights plus
#: activations every call (Appendix A variant V1, the default until STEP 8
#: decides otherwise); "max" assumes the larger of the two streams hides the
#: other (variant V2 - at large T weights are cache-resident/reused, so
#: charging them every call misclassifies large-T GEMMs as memory-bound).
BYTES_MODES = ("sum", "max")


class ShapeError(ValueError):
    """A layer name or category the resolver cannot cost."""


@dataclass(frozen=True)
class OpCost:
    flops: float
    bytes_moved: float
    family: str  # 'gemm' | 'elementwise' | 'gather' | 'attention' | 'moe'


class ShapeResolver:
    """Resolves dense / per_sequence / expert layer costs for one (model, TP)."""

    def __init__(
        self,
        dims: ModelDims,
        arch: Architecture,
        tp: int,
        *,
        bytes_mode: str = "sum",
    ) -> None:
        if tp < 1:
            raise ShapeError(f"tp must be >= 1, got {tp}")
        if bytes_mode not in BYTES_MODES:
            raise ShapeError(f"bytes_mode {bytes_mode!r} not in {BYTES_MODES}")
        self.dims = dims
        self.arch = arch
        self.tp = tp
        self.bytes_mode = bytes_mode

    # ------------------------------------------------------------------
    # Catalog access
    # ------------------------------------------------------------------

    def layers(self) -> list[str]:
        """Dense-layer names from the architecture yaml catalog."""
        return list(self.arch.catalog.dense)

    def per_sequence_layers(self) -> list[str]:
        return list(self.arch.catalog.per_sequence)

    def _tp_for(self, group: str, layer: str) -> int:
        entry = getattr(self.arch.catalog, group).get(layer)
        if entry is None:
            raise ShapeError(
                f"layer {layer!r} is not in the {group} catalog of "
                f"{self.dims.model_type}; refusing to cost it as zero"
            )
        return 1 if entry.tp_stable else self.tp

    # ------------------------------------------------------------------
    # Cost formulas (Appendix A)
    # ------------------------------------------------------------------

    def _gemm(self, m: float, k: float, n: float) -> OpCost:
        flops = 2.0 * m * n * k
        b = self.dims.dtype_bytes
        weight = k * n * b
        activation = (m * k + m * n) * b
        bytes_moved = (
            weight + activation if self.bytes_mode == "sum" else max(weight, activation)
        )
        return OpCost(flops=flops, bytes_moved=bytes_moved, family="gemm")

    def _elementwise(self, flops_per_elem: float, elems_in: float,
                     elems_out: float) -> OpCost:
        b = self.dims.dtype_bytes
        return OpCost(
            flops=flops_per_elem * elems_out,
            bytes_moved=(elems_in + elems_out) * b,
            family="elementwise",
        )

    def dense(self, layer: str, tokens: int) -> OpCost:
        """Cost of one token-parameterized (dense.csv) layer at T tokens."""
        if tokens < 1:
            raise ShapeError(f"tokens must be >= 1, got {tokens}")
        d = self.dims
        tp = self._tp_for("dense", layer)
        t = float(tokens)

        if layer == "qkv_proj":
            n_out = (d.num_attention_heads + 2 * d.num_key_value_heads) * d.head_dim / tp
            return self._gemm(t, d.hidden_size, n_out)
        if layer == "o_proj":
            return self._gemm(t, d.num_attention_heads * d.head_dim / tp, d.hidden_size)
        if layer == "gate_up_proj":
            return self._gemm(t, d.hidden_size, 2 * d.intermediate_size / tp)
        if layer == "down_proj":
            return self._gemm(t, d.intermediate_size / tp, d.hidden_size)
        if layer in ("layernorm", "final_layernorm"):
            # RMSNorm: square + reduce + rsqrt-normalize + scale ~= 5 ops/elem
            # (draft coefficient, revisited by the STEP 8 diff).
            elems = t * d.hidden_size / tp
            return self._elementwise(5.0, elems, elems)
        if layer == "qk_norm":
            # Qwen3-family per-head RMSNorm applied to q and k after the QKV
            # projection; same 5 ops/elem draft coefficient as RMSNorm. The
            # catalog marks it tp_stable, so tp is already 1 here.
            elems = t * (d.num_attention_heads + d.num_key_value_heads) * d.head_dim / tp
            return self._elementwise(5.0, elems, elems)
        if layer == "act_fn":
            # SiluAndMul: in = 2*d_ff/TP per token, out = d_ff/TP;
            # silu(x)*y ~= 3 ops per OUTPUT element (sigmoid+mul+mul, draft).
            elems_in = t * 2 * d.intermediate_size / tp
            return self._elementwise(3.0, elems_in, elems_in / 2)
        if layer == "rotary_emb":
            # Rotate q and k: sin/cos multiply-add pairs ~= 6 ops/elem (draft).
            elems = t * (d.num_attention_heads + d.num_key_value_heads) * d.head_dim / tp
            return self._elementwise(6.0, elems, elems)
        if layer == "embedding":
            # Gather: reads one full d_model row per token; negligible
            # arithmetic (Appendix A: flops ~= 0, the launch/memory floor
            # dominates). VocabParallelEmbedding shards the VOCAB axis, not
            # the row width, so bytes do not divide by TP.
            return OpCost(
                flops=0.0,
                bytes_moved=t * d.hidden_size * d.dtype_bytes,
                family="gather",
            )

        # Reaching here means the catalog names a dense layer this resolver
        # has no formula for - fail loud so Tier 0 cannot silently under-cost.
        self._tp_for("dense", layer)  # raises for names not in the catalog
        raise ShapeError(
            f"dense layer {layer!r} is in the catalog but ShapeResolver has "
            f"no shape formula for it - add one before emitting Tier 0"
        )

    def per_sequence(self, layer: str, sequences: int) -> OpCost:
        """Cost of one sequence-parameterized (per_sequence.csv) layer."""
        if sequences < 1:
            raise ShapeError(f"sequences must be >= 1, got {sequences}")
        d = self.dims
        tp = self._tp_for("per_sequence", layer)
        s = float(sequences)

        if layer == "lm_head":
            return self._gemm(s, d.hidden_size, d.vocab_size / tp)
        if layer == "sampler":
            # Reads the S x vocab logits, emits S token ids: ~3 ops/elem
            # (max + exp + compare class of work, draft coefficient).
            elems = s * d.vocab_size / tp
            return self._elementwise(3.0, elems, s)

        self._tp_for("per_sequence", layer)
        raise ShapeError(
            f"per_sequence layer {layer!r} has no shape formula - add one "
            f"before emitting Tier 0"
        )

    def expert(self, tokens: int, activated_experts: int) -> OpCost:
        """MoE expert-path cost for one step (moe.csv key).

        FLOPs follow Appendix A: every token is processed by exactly
        ``experts_per_token`` experts, so compute is 2*(T*k)*N*K for the
        gate_up GEMM plus the down GEMM - independent of how many DISTINCT
        experts the batch activates. ``activated_experts`` instead scales the
        weight traffic: each distinct expert's weights must be brought in
        once. (The work order's draft test sketch says "FLOPs linear in
        activated_experts"; that contradicts its own Appendix A formula and
        the grouped-GEMM physics, so bytes - not FLOPs - carry the
        activated_experts dependence. Recorded in the STEP 4 PR.)
        """
        d = self.dims
        if not d.is_moe:
            raise ShapeError(f"{d.model} is not a MoE model")
        if tokens < 1 or activated_experts < 1:
            raise ShapeError("tokens and activated_experts must be >= 1")
        assert d.experts_per_token is not None and d.moe_intermediate_size is not None
        tp = self.tp
        m = float(tokens) * d.experts_per_token
        n_up = 2 * d.moe_intermediate_size / tp
        k_up = d.hidden_size
        n_down = d.hidden_size
        k_down = d.moe_intermediate_size / tp

        flops = 2.0 * m * (n_up * k_up + n_down * k_down)
        b = d.dtype_bytes
        weight_per_expert = (k_up * n_up + k_down * n_down) * b
        weights = activated_experts * weight_per_expert
        activations = (m * k_up + m * n_up + m * k_down + m * n_down) * b
        if self.bytes_mode == "sum":
            bytes_moved = weights + activations
        else:
            bytes_moved = max(weights, activations)
        return OpCost(flops=flops, bytes_moved=bytes_moved, family="moe")
