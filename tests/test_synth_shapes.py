"""ShapeResolver FLOPs / bytes formulas (STEP 4)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from profiler.core.config import load_architecture
from profiler.synth.dims import ModelDims
from profiler.synth.shapes import ShapeError, ShapeResolver

REPO = Path(__file__).resolve().parents[1]
LLAMA_CFG = REPO / "configs" / "model" / "meta-llama" / "Llama-3.1-8B.json"
QWEN_MOE_CFG = REPO / "configs" / "model" / "Qwen" / "Qwen3-30B-A3B-Instruct-2507.json"
LLAMA_ARCH = REPO / "profiler" / "models" / "llama.yaml"
QWEN_MOE_ARCH = REPO / "profiler" / "models" / "qwen3_moe.yaml"
A40_TP1 = REPO / "profiler" / "perf" / "A40" / "meta-llama" / "Llama-3.1-8B" / "bf16" / "tp1"
MOE_TP1 = (
    REPO / "profiler" / "perf" / "RTXPRO6000" / "Qwen"
    / "Qwen3-30B-A3B-Instruct-2507" / "bf16" / "tp1"
)


@pytest.fixture(scope="module")
def llama_resolver() -> ShapeResolver:
    dims = ModelDims.from_hf_config(LLAMA_CFG, "bf16")
    return ShapeResolver(dims, load_architecture(LLAMA_ARCH), tp=1)


def _resolver(tp: int, bytes_mode: str = "sum") -> ShapeResolver:
    dims = ModelDims.from_hf_config(LLAMA_CFG, "bf16")
    return ShapeResolver(dims, load_architecture(LLAMA_ARCH), tp=tp, bytes_mode=bytes_mode)


def _csv_column(path: Path, column: str) -> list[str]:
    with path.open() as f:
        return sorted({row[column] for row in csv.DictReader(f)})


def test_layer_list_matches_architecture_yaml(llama_resolver):
    """layers() equals the catalog.dense keys of profiler/models/llama.yaml."""
    raw = yaml.safe_load(LLAMA_ARCH.read_text())
    assert llama_resolver.layers() == list(raw["catalog"]["dense"])
    assert llama_resolver.per_sequence_layers() == list(raw["catalog"]["per_sequence"])


def test_dense_csv_layers_are_all_resolvable(llama_resolver):
    """Every layer in the measured dense/per_sequence/moe CSVs is costable."""
    for layer in _csv_column(A40_TP1 / "dense.csv", "layer"):
        cost = llama_resolver.dense(layer, tokens=64)
        assert cost.bytes_moved > 0
    for layer in _csv_column(A40_TP1 / "per_sequence.csv", "layer"):
        cost = llama_resolver.per_sequence(layer, sequences=4)
        assert cost.bytes_moved > 0
    # The Qwen3-MoE bundle exercises catalog layers Llama lacks (qk_norm).
    moe_dims = ModelDims.from_hf_config(QWEN_MOE_CFG, "bf16")
    moe_resolver = ShapeResolver(moe_dims, load_architecture(QWEN_MOE_ARCH), tp=1)
    for layer in _csv_column(MOE_TP1 / "dense.csv", "layer"):
        assert moe_resolver.dense(layer, tokens=64).bytes_moved > 0
    for layer in _csv_column(MOE_TP1 / "per_sequence.csv", "layer"):
        assert moe_resolver.per_sequence(layer, sequences=4).bytes_moved > 0
    # moe.csv has no layer column; its key is (tokens, activated_experts).
    with (MOE_TP1 / "moe.csv").open() as f:
        for row in list(csv.DictReader(f))[:32]:
            cost = moe_resolver.expert(int(row["tokens"]), int(row["activated_experts"]))
            assert cost.flops > 0 and cost.bytes_moved > 0


def test_qkv_proj_flops_formula():
    """qkv_proj FLOPs = 2*T*d_model*((n_q+2*n_kv)*d_head/TP), hand-checked."""
    # Llama-3.1-8B: d_model=4096, n_q=32, n_kv=8, d_head=128.
    t = 100
    expected_tp1 = 2 * t * 4096 * ((32 + 2 * 8) * 128 / 1)
    assert _resolver(1).dense("qkv_proj", t).flops == pytest.approx(expected_tp1)
    expected_tp2 = 2 * t * 4096 * ((32 + 2 * 8) * 128 / 2)
    assert _resolver(2).dense("qkv_proj", t).flops == pytest.approx(expected_tp2)


def test_row_parallel_layers_shard_input_not_output():
    """o_proj / down_proj FLOPs halve exactly from TP=1 to TP=2."""
    for layer in ("o_proj", "down_proj"):
        f1 = _resolver(1).dense(layer, 64).flops
        f2 = _resolver(2).dense(layer, 64).flops
        assert f2 == pytest.approx(f1 / 2)


def test_column_parallel_layers_shard_output():
    """qkv_proj / gate_up_proj FLOPs halve from TP=1 to TP=2."""
    for layer in ("qkv_proj", "gate_up_proj"):
        f1 = _resolver(1).dense(layer, 64).flops
        f2 = _resolver(2).dense(layer, 64).flops
        assert f2 == pytest.approx(f1 / 2)


def test_tp_stable_layers_do_not_shard():
    """tp_stable layers (layernorm, sampler) are TP-invariant."""
    assert _resolver(1).dense("layernorm", 64) == _resolver(4).dense("layernorm", 64)
    assert (
        _resolver(1).per_sequence("sampler", 8)
        == _resolver(4).per_sequence("sampler", 8)
    )


def test_lm_head_scales_with_sequences_not_tokens(llama_resolver):
    """per_sequence layers are linear in sequences."""
    f1 = llama_resolver.per_sequence("lm_head", 1).flops
    f8 = llama_resolver.per_sequence("lm_head", 8).flops
    assert f8 == pytest.approx(8 * f1)


def test_moe_flops_scale_with_tokens_bytes_with_experts():
    """MoE FLOPs are linear in tokens; weight bytes grow with activated experts.

    Appendix A prices MoE compute as 2*(T*experts_per_token)*N*K - every token
    is processed by exactly top-k experts, so FLOPs do NOT depend on how many
    distinct experts the batch activates; the distinct-expert count scales the
    weight traffic instead. (The work order's draft test sketch says "FLOPs
    linear in activated_experts", which contradicts its own Appendix A formula;
    the appendix + grouped-GEMM physics win, recorded in the STEP 4 PR.)
    """
    dims = ModelDims.from_hf_config(QWEN_MOE_CFG, "bf16")
    r = ShapeResolver(dims, load_architecture(QWEN_MOE_ARCH), tp=1)
    assert r.expert(64, 8).flops == pytest.approx(2 * r.expert(32, 8).flops)
    assert r.expert(64, 8).flops == pytest.approx(r.expert(64, 16).flops)
    assert r.expert(64, 16).bytes_moved > r.expert(64, 8).bytes_moved


def test_unknown_layer_raises(llama_resolver):
    """A layer name outside the catalog raises; zero-cost is forbidden."""
    with pytest.raises(ShapeError):
        llama_resolver.dense("mystery_proj", 64)
    with pytest.raises(ShapeError):
        llama_resolver.per_sequence("mystery_head", 4)


def test_moe_on_dense_model_raises(llama_resolver):
    """expert() on a dense model raises."""
    with pytest.raises(ShapeError):
        llama_resolver.expert(64, 8)


def test_flops_monotonic_in_tokens(llama_resolver):
    """FLOPs and bytes never decrease as tokens grow, for every dense layer."""
    for layer in llama_resolver.layers():
        prev = None
        for tokens in (1, 2, 4, 16, 64, 256, 1024, 4096):
            cost = llama_resolver.dense(layer, tokens)
            if prev is not None:
                assert cost.flops >= prev.flops, layer
                assert cost.bytes_moved > prev.bytes_moved, layer
            prev = cost


def test_weight_reuse_variants_differ():
    """The sum and max bytes_moved variants disagree at large T."""
    big_t = 4096
    for layer in ("qkv_proj", "down_proj"):
        b_sum = _resolver(1, "sum").dense(layer, big_t).bytes_moved
        b_max = _resolver(1, "max").dense(layer, big_t).bytes_moved
        assert b_sum > b_max, layer
        # And they must agree on FLOPs - only the byte accounting differs.
        assert (
            _resolver(1, "sum").dense(layer, big_t).flops
            == _resolver(1, "max").dense(layer, big_t).flops
        )


def test_families_assigned(llama_resolver):
    """Kernel families are what the Tier 1 calibration expects."""
    assert llama_resolver.dense("qkv_proj", 4).family == "gemm"
    assert llama_resolver.dense("layernorm", 4).family == "elementwise"
    assert llama_resolver.dense("embedding", 4).family == "gather"
    assert llama_resolver.per_sequence("lm_head", 4).family == "gemm"
