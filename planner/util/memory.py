"""Memory feasibility (work order §5.4, pruning stage 2).

Weight and KV sizes are obtained by *calling* the simulator's own
`serving/core/memory_model.py`, not by reimplementing it - the work order is
explicit that the formulas must not be duplicated.

What we add on top is a derating step. The simulator computes
`mem_for_kv = npu_mem - weight` with no reserve of any kind, while real vLLM
first takes `gpu_memory_utilization` of VRAM and then also loses the activation
peak and CUDA-graph capture. Measured on an RTX A5000 running Llama-3.1-8B the
simulator over-estimates usable KV by +71%. See docs/deviations.md D10.
"""

from __future__ import annotations

from dataclasses import dataclass

from serving.core.memory_model import GB_TO_BYTE, MemoryModel
from serving.core.utils import get_config

#: dtype label -> bits, matching the simulator's `fp` argument (bits, not bytes).
DTYPE_BITS: dict[str, int] = {
    "float32": 32,
    "fp32": 32,
    "bfloat16": 16,
    "float16": 16,
    "fp16": 16,
    "bf16": 16,
    "fp8": 8,
    "int8": 8,
}


class MemoryError_(ValueError):
    """Raised when memory cannot be evaluated for a (model, dtype) pair."""


def model_config(model: str) -> dict:
    """The raw HF config the simulator resolves for `model`.

    Goes through upstream's `get_config` so the planner and the simulator can
    never disagree about which `configs/model/<org>/<name>.json` is in play.
    """
    try:
        return get_config(model)
    except Exception as exc:
        raise MemoryError_(f"cannot load model config for '{model}': {exc}") from None


def dtype_bits(dtype: str) -> int:
    try:
        return DTYPE_BITS[dtype.lower()]
    except KeyError:
        raise MemoryError_(
            f"unknown dtype '{dtype}'; expected one of {sorted(set(DTYPE_BITS))}"
        ) from None


@dataclass(frozen=True)
class MemoryReport:
    """Per-GPU memory breakdown for one candidate placement."""

    model: str
    tp_size: int
    device_memory_bytes: int
    weight_bytes: int
    kv_bytes_per_token: int
    usable_kv_bytes: int
    gpu_memory_utilization: float
    activation_reserve_bytes: int

    @property
    def kv_tokens(self) -> int:
        if self.kv_bytes_per_token <= 0:
            return 0
        return self.usable_kv_bytes // self.kv_bytes_per_token

    @property
    def fits(self) -> bool:
        return self.usable_kv_bytes > 0

    @property
    def naive_kv_tokens(self) -> int:
        """What the raw simulator model would report, for D10 comparisons."""
        naive = self.device_memory_bytes - self.weight_bytes
        if self.kv_bytes_per_token <= 0 or naive <= 0:
            return 0
        return naive // self.kv_bytes_per_token

    def summary(self) -> str:
        gib = GB_TO_BYTE
        return (
            f"weight {self.weight_bytes / gib:.2f} GiB/gpu, "
            f"usable KV {self.usable_kv_bytes / gib:.2f} GiB "
            f"({self.kv_tokens:,} tokens @ {self.kv_bytes_per_token / 1024:.0f} KiB/token)"
        )


def _probe(model: str, tp_size: int, bits: int, kv_cache_dtype: str) -> tuple[int, int]:
    """Ask the simulator's memory model for (weight_bytes, kv_bytes_per_token).

    `npu_mem` is deliberately set absurdly high: MemoryModel raises when weights
    exceed it, and here we only want the sizes, not its verdict.
    """
    try:
        mm = MemoryModel(
            model=model,
            instance_id=0,
            node_id=0,
            num_npus=tp_size,
            tp_size=tp_size,
            npu_mem=1 << 20,  # GB; effectively unbounded for a size query
            cpu_mem=0,
            block_size=16,
            fp=bits,
            enable_prefix_caching=False,
            enable_prefix_sharing=False,
            prefix_pool=None,
            prefix_storage=None,
            kv_cache_dtype=kv_cache_dtype,
        )
    except Exception as exc:
        raise MemoryError_(f"cannot size model '{model}' at tp={tp_size}: {exc}") from exc
    return int(mm.weight), int(mm.get_kv(1))


def evaluate(
    model: str,
    *,
    tp_size: int,
    device_memory_gb: float,
    dtype: str = "bfloat16",
    kv_cache_dtype: str = "auto",
    gpu_memory_utilization: float = 0.90,
    activation_reserve_gb: float = 0.0,
) -> MemoryReport:
    """Per-GPU memory report for placing `model` at `tp_size` on one device.

    `gpu_memory_utilization` and `activation_reserve_gb` are explicit inputs
    rather than baked-in constants: the right values depend on the runtime and
    belong in ClusterSpecV2, and silently hard-coding vLLM's 0.9 would hide the
    very accounting error D10 documents.
    """
    if tp_size < 1:
        raise MemoryError_(f"tp_size must be >= 1, got {tp_size}")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise MemoryError_(
            f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        )

    weight, kv_per_token = _probe(model, tp_size, dtype_bits(dtype), kv_cache_dtype)
    device_bytes = int(device_memory_gb * GB_TO_BYTE)
    reserve = int(activation_reserve_gb * GB_TO_BYTE)
    usable = int(device_bytes * gpu_memory_utilization) - weight - reserve

    return MemoryReport(
        model=model,
        tp_size=tp_size,
        device_memory_bytes=device_bytes,
        weight_bytes=weight,
        kv_bytes_per_token=kv_per_token,
        usable_kv_bytes=max(0, usable),
        gpu_memory_utilization=gpu_memory_utilization,
        activation_reserve_bytes=reserve,
    )


def feasible(
    model: str,
    *,
    tp_size: int,
    device_memory_gb: float,
    min_kv_tokens: int = 1,
    **kwargs: object,
) -> tuple[bool, MemoryReport]:
    """Pruning-stage-2 predicate: does this placement leave usable KV space?

    `min_kv_tokens` lets a caller demand headroom for at least one full request
    rather than a single token, which is the sizing question that actually
    matters once a workload is known.
    """
    report = evaluate(model, tp_size=tp_size, device_memory_gb=device_memory_gb, **kwargs)  # type: ignore[arg-type]
    return report.kv_tokens >= min_kv_tokens, report
