"""ModelDims reads HF configs exactly (STEP 4)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from profiler.synth.dims import DimsError, ModelDims, parse_variant

REPO = Path(__file__).resolve().parents[1]
LLAMA = REPO / "configs" / "model" / "meta-llama" / "Llama-3.1-8B.json"
QWEN_MOE = REPO / "configs" / "model" / "Qwen" / "Qwen3-30B-A3B-Instruct-2507.json"


def test_llama31_8b_dims():
    """Llama-3.1-8B dimensions match its HF config (read, not hardcoded)."""
    raw = json.loads(LLAMA.read_text())
    dims = ModelDims.from_hf_config(LLAMA, "bf16")
    assert dims.model_type == raw["model_type"]
    assert dims.hidden_size == raw["hidden_size"]
    assert dims.intermediate_size == raw["intermediate_size"]
    assert dims.num_attention_heads == raw["num_attention_heads"]
    assert dims.num_key_value_heads == raw["num_key_value_heads"]
    assert dims.num_hidden_layers == raw["num_hidden_layers"]
    assert dims.vocab_size == raw["vocab_size"]
    # No head_dim in this config (verified 2026-09-02): derived.
    assert "head_dim" not in raw
    assert dims.head_dim == raw["hidden_size"] // raw["num_attention_heads"]
    assert not dims.is_moe


def test_qwen3_moe_dims():
    """Qwen3-30B-A3B MoE fields are read, and its explicit head_dim wins."""
    raw = json.loads(QWEN_MOE.read_text())
    dims = ModelDims.from_hf_config(QWEN_MOE, "bf16")
    assert dims.num_experts == raw["num_experts"]
    assert dims.experts_per_token == raw["num_experts_per_tok"]
    assert dims.moe_intermediate_size == raw["moe_intermediate_size"]
    # The config declares head_dim=128 while hidden/heads = 64: the explicit
    # key must take precedence or attention shapes are wrong by 2x.
    assert raw["head_dim"] != raw["hidden_size"] // raw["num_attention_heads"]
    assert dims.head_dim == raw["head_dim"]
    assert dims.is_moe


def test_kv_dtype_from_variant():
    """variant 'bf16-kvfp8' gives weight bytes 2 and KV bytes 1."""
    dims = ModelDims.from_hf_config(LLAMA, "bf16-kvfp8")
    assert dims.dtype_bytes == 2
    assert dims.kv_dtype_bytes == 1
    plain = ModelDims.from_hf_config(LLAMA, "bf16")
    assert plain.kv_dtype_bytes == 2


def test_unknown_variant_raises():
    """Unknown dtype tokens in the variant raise (no modeling guess)."""
    with pytest.raises(DimsError):
        parse_variant("int4")
    with pytest.raises(DimsError):
        parse_variant("bf16-kvint4")
    with pytest.raises(DimsError):
        parse_variant("bf16-extra")


def test_missing_field_raises(tmp_path):
    """A config without a required field raises; never defaulted (A2)."""
    raw = json.loads(LLAMA.read_text())
    del raw["intermediate_size"]
    bad = tmp_path / "Broken.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(DimsError, match="intermediate_size"):
        ModelDims.from_hf_config(bad, "bf16")


def test_moe_without_intermediate_raises(tmp_path):
    """A MoE config without moe_intermediate_size raises (A2)."""
    raw = json.loads(QWEN_MOE.read_text())
    del raw["moe_intermediate_size"]
    bad = tmp_path / "BrokenMoe.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(DimsError, match="moe_intermediate_size"):
        ModelDims.from_hf_config(bad, "bf16")


def test_head_dim_derivation_documented():
    """The head_dim derivation rule is recorded in the docstring."""
    doc = inspect.getdoc(ModelDims.from_hf_config)
    assert doc is not None
    assert "head_dim" in doc
    assert "hidden_size // num_attention_heads" in doc
    assert "2026-09-02" in doc
