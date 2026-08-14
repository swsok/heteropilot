"""Memory feasibility, including the D10 derating (work order §5.4 stage 2)."""

from __future__ import annotations

import pytest

from planner.util import memory as memutil

GIB = 1024 ** 3
LLAMA = "meta-llama/Llama-3.1-8B"

# Ground truth captured from a real vLLM 0.19.0 boot on one RTX A5000
# (enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.9):
#   "Available KV cache memory: 5.29 GiB" / "GPU KV cache size: 43,296 tokens"
MEASURED_KV_GIB = 5.29
MEASURED_KV_TOKENS = 43_296
A5000_GB = 24.0


def test_weight_and_kv_come_from_the_simulator_model() -> None:
    """Sanity-check the numbers we get by calling upstream, not reimplementing."""
    r = memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=80.0)
    # Llama-3.1-8B bf16: 8.03e9 params x 2 B ~= 14.96 GiB
    assert 14.5 < r.weight_bytes / GIB < 15.5
    # KV/token = 2 (K,V) x 32 layers x 8 kv-heads x 128 head-dim x 2 B = 128 KiB
    assert r.kv_bytes_per_token == 128 * 1024


def test_tensor_parallel_shards_weight_and_kv() -> None:
    tp1 = memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=80.0)
    tp2 = memutil.evaluate(LLAMA, tp_size=2, device_memory_gb=80.0)
    assert tp2.weight_bytes == pytest.approx(tp1.weight_bytes / 2, rel=0.01)
    assert tp2.kv_bytes_per_token == tp1.kv_bytes_per_token // 2


def test_derating_moves_the_estimate_toward_measured_vllm() -> None:
    """The whole point of D10: the raw model is far off, derating closes most of it."""
    r = memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                         gpu_memory_utilization=0.90)
    naive_gib = (r.device_memory_bytes - r.weight_bytes) / GIB

    # The raw simulator model claims 24 GiB - 14.96 GiB = 9.04 GiB of KV against
    # a measured 5.29 GiB: 74,075 tokens vs 43,296, a +71% over-estimate.
    assert naive_gib == pytest.approx(9.04, abs=0.1)
    assert r.naive_kv_tokens == pytest.approx(74_075, rel=0.01)
    assert r.naive_kv_tokens / MEASURED_KV_TOKENS == pytest.approx(1.71, abs=0.02)

    # With utilization applied the gap shrinks a lot but does not vanish - the
    # remainder is the activation peak, which is a separate input.
    assert r.usable_kv_bytes / GIB == pytest.approx(6.64, abs=0.1)
    assert r.kv_tokens / MEASURED_KV_TOKENS == pytest.approx(1.26, abs=0.02)


def test_measured_activation_reserve_reproduces_vllm() -> None:
    """With the measured reserve supplied, we land on vLLM's real KV budget."""
    r = memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                         gpu_memory_utilization=0.90, activation_reserve_gb=1.34)
    assert r.usable_kv_bytes / GIB == pytest.approx(MEASURED_KV_GIB, abs=0.05)
    assert r.kv_tokens == pytest.approx(MEASURED_KV_TOKENS, rel=0.01)


def test_utilization_is_not_hardcoded() -> None:
    """A caller must be able to model a runtime that reserves differently."""
    full = memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                            gpu_memory_utilization=1.0)
    derated = memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                               gpu_memory_utilization=0.5)
    assert full.usable_kv_bytes > derated.usable_kv_bytes


def test_model_too_large_for_device_is_infeasible() -> None:
    """Qwen3-32B bf16 needs ~61 GiB of weights; it cannot sit on one 24 GB card."""
    fits, report = memutil.feasible("Qwen/Qwen3-32B", tp_size=1,
                                    device_memory_gb=A5000_GB)
    assert fits is False
    assert report.usable_kv_bytes == 0
    assert report.kv_tokens == 0


def test_min_kv_tokens_gate() -> None:
    """A placement with room for a token but not a request is not usable."""
    fits_1, report = memutil.feasible(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                                      min_kv_tokens=1)
    fits_huge, _ = memutil.feasible(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                                    min_kv_tokens=report.kv_tokens + 1)
    assert fits_1 is True
    assert fits_huge is False


@pytest.mark.parametrize("dtype,bits", [("bfloat16", 16), ("float16", 16),
                                        ("float32", 32), ("fp8", 8)])
def test_dtype_bits_mapping(dtype: str, bits: int) -> None:
    assert memutil.dtype_bits(dtype) == bits


def test_unknown_dtype_rejected() -> None:
    with pytest.raises(memutil.MemoryError_, match="unknown dtype"):
        memutil.dtype_bits("float13")


def test_invalid_utilization_rejected() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(memutil.MemoryError_, match="gpu_memory_utilization"):
            memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=A5000_GB,
                             gpu_memory_utilization=bad)


def test_evaluation_is_deterministic() -> None:
    runs = [memutil.evaluate(LLAMA, tp_size=1, device_memory_gb=A5000_GB) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
