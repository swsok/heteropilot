"""Model dimensions from HF configs (WORK_ORDER_tiered_profiles.md STEP 4).

Reads ``configs/model/<org>/<model>.json`` (a HuggingFace config) plus the
variant string into the handful of architecture constants every dense-layer
shape is determined by. Missing required fields raise - values are never
defaulted (absolute rule A2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Pure-Python helper from the profiler config module (no torch/vllm there).
from profiler.core.config import probe_moe_params

#: Weight/activation bytes per element for each variant dtype token. int4 is
#: sub-byte and not representable as an int; it raises (out of scope, work
#: order STEP 5 item 5: unknown dtypes are an error, not a guess).
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "int8": 1}

_REQUIRED_FIELDS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
)


class DimsError(ValueError):
    """A HF config or variant string cannot supply the required dimensions."""


def parse_variant(variant: str) -> tuple[int, int]:
    """(dtype_bytes, kv_dtype_bytes) for a CONTRACT.md variant string.

    The variant grammar is ``<dtype>[-kv<dtype>]`` (CONTRACT.md bundle
    layout: ``bf16``, ``bf16-kvfp8``, ...). No kv part means the KV cache
    uses the weight dtype.
    """
    parts = variant.split("-")
    weight = parts[0]
    if weight not in DTYPE_BYTES:
        raise DimsError(f"variant {variant!r}: unknown weight dtype {weight!r}")
    kv = weight
    for part in parts[1:]:
        if not part.startswith("kv"):
            raise DimsError(f"variant {variant!r}: unrecognized component {part!r}")
        kv = part[2:]
        if kv not in DTYPE_BYTES:
            raise DimsError(f"variant {variant!r}: unknown KV dtype {kv!r}")
    return DTYPE_BYTES[weight], DTYPE_BYTES[kv]


@dataclass(frozen=True)
class ModelDims:
    """Architecture constants that determine every dense-layer shape."""

    model: str
    model_type: str          # HF model_type; selects profiler/models/<type>.yaml
    num_hidden_layers: int
    hidden_size: int         # d_model
    intermediate_size: int   # d_ff
    num_attention_heads: int  # n_q
    num_key_value_heads: int  # n_kv
    head_dim: int
    vocab_size: int
    dtype_bytes: int         # weight/activation dtype
    kv_dtype_bytes: int      # KV-cache dtype (fp8 KV variants)
    # MoE (None for dense models)
    num_experts: int | None = None
    experts_per_token: int | None = None
    moe_intermediate_size: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None

    @classmethod
    def from_hf_config(cls, path: Path, variant: str) -> ModelDims:
        """Build ModelDims from a configs/model/<org>/<model>.json file.

        head_dim rule (verified 2026-09-02): Llama-3.1-8B.json and
        Llama-3.1-70B.json carry NO ``head_dim`` key, so it is derived as
        ``hidden_size // num_attention_heads``. When the config DOES carry
        ``head_dim`` it takes precedence - Qwen3-30B-A3B-Instruct-2507
        declares head_dim=128 while hidden_size/num_attention_heads would
        give 2048/32 = 64, so deriving there would be wrong by 2x.
        """
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DimsError(f"{path}: HF config not found") from exc
        except json.JSONDecodeError as exc:
            raise DimsError(f"{path}: invalid JSON - {exc}") from exc

        missing = [f for f in _REQUIRED_FIELDS if f not in raw]
        if missing:
            raise DimsError(
                f"{path}: required HF config fields missing: {missing} - "
                f"dimensions are never defaulted (absolute rule A2)"
            )

        dtype_bytes, kv_dtype_bytes = parse_variant(variant)

        head_dim = raw.get("head_dim")
        if head_dim is None:
            head_dim = raw["hidden_size"] // raw["num_attention_heads"]

        moe = probe_moe_params(raw)
        num_experts, experts_per_token = (moe if moe is not None else (None, None))
        moe_intermediate = raw.get("moe_intermediate_size")
        if moe is not None and moe_intermediate is None:
            raise DimsError(
                f"{path}: MoE model (num_experts={num_experts}) without "
                f"moe_intermediate_size - cannot size the expert GEMM (A2)"
            )

        return cls(
            model=f"{path.parent.name}/{path.stem}",
            model_type=str(raw["model_type"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(head_dim),
            vocab_size=int(raw["vocab_size"]),
            dtype_bytes=dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            moe_intermediate_size=(
                int(moe_intermediate) if moe_intermediate is not None else None
            ),
        )
