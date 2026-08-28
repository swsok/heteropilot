"""Backend-agnostic pieces of the layerwise NPU profilers.

Extracted from ``profile_rngd.py`` so the RNGD and ATOM harnesses measure
**literally the same layers**. That matters more than the duplication it saves:
Exp 4 compares the two NPUs against each other and against the A40, and if the
two harnesses' layer definitions drifted apart the comparison would quietly stop
meaning anything. There is one definition and both import it.

Everything here is pure ``torch`` -- no vendor runtime -- so it imports in any
of the three venvs. The vendor-specific half (how a module is compiled, placed
and timed on a device) stays in each harness.

Canonical layer names and the pipeline they belong to come from
``profiler/models/llama.yaml``; the CSV shapes they feed come from
``profiler/CONTRACT.md`` §3.7.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]


SHARD_FIELDS = ("intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size")


def load_model_config(model: str, tp: int) -> dict[str, Any]:
    """Read ``configs/model/<model>.json`` and shard it for TP degree ``tp``.

    No HF hub access: the simulator ships the config we need.
    """
    path = REPO_ROOT / "configs" / "model" / f"{model}.json"
    if not path.is_file():
        raise SystemExit(f"model config not found: {path}")
    cfg = json.loads(path.read_text())
    # head_dim is derived from the UNSHARDED head count, then the shard
    # divides the number of heads, not their width.
    cfg["head_dim"] = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    for field in SHARD_FIELDS:
        value = cfg.get(field)
        if value is None:
            continue
        if value % tp != 0:
            raise SystemExit(f"{field}={value} is not divisible by tp={tp}")
        cfg[field] = value // tp
    cfg["_tp"] = tp
    return cfg


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = (x.float() * x.float()).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class SiluAndMul(nn.Module):
    """vLLM's ``SiluAndMul``: silu of the first half times the second half."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        gate = x[..., :half]
        up = x[..., half:]
        return torch.nn.functional.silu(gate) * up


class RotaryEmbedding(nn.Module):
    """Rotate-half rotary applied to the query and key projections."""

    def __init__(self, head_dim: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        def rotate(t: torch.Tensor) -> torch.Tensor:
            half = t.shape[-1] // 2
            left = t[..., :half]
            right = t[..., half:]
            rotated = torch.cat((-right, left), dim=-1)
            return t * cos + rotated * sin

        return rotate(q).sum() + rotate(k).sum()


class Attention(nn.Module):
    """One scheduler step's attention: a prefill chunk plus decode rows.

    Both parts are computed in a single graph because the simulator's
    ``attention.csv`` is keyed on the whole step, not on its halves.
    """

    def __init__(self, head_dim: int, scale: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        total = None
        # tensors arrive as (q, k, v) triples; each triple is one part of the
        # step (prefill chunk and/or decode rows).
        for i in range(0, len(tensors), 3):
            q, k, v = tensors[i], tensors[i + 1], tensors[i + 2]
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            out = torch.matmul(torch.softmax(scores.float(), dim=-1).to(q.dtype), v)
            total = out.sum() if total is None else total + out.sum()
        return total


def dense_layer(name: str, cfg: dict[str, Any]) -> tuple[nn.Module, Callable[[int], tuple]]:
    """Build one canonical dense layer and its ``tokens -> inputs`` factory."""
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    bf16 = torch.bfloat16

    def h_in(tokens: int) -> tuple:
        return (torch.randn(tokens, hidden, dtype=bf16),)

    if name == "embedding":
        module = nn.Embedding(vocab, hidden, dtype=bf16)
        return module, lambda t: (torch.randint(0, vocab, (t,)),)
    if name in ("layernorm", "final_layernorm"):
        return RMSNorm(hidden, cfg.get("rms_norm_eps", 1e-5)), h_in
    if name == "qkv_proj":
        out = (n_heads + 2 * n_kv) * head_dim
        return nn.Linear(hidden, out, bias=False, dtype=bf16), h_in
    if name == "rotary_emb":
        module = RotaryEmbedding(head_dim, n_heads, n_kv)

        def rotary_in(tokens: int) -> tuple:
            return (
                torch.randn(tokens, n_heads * head_dim, dtype=bf16),
                torch.randn(tokens, n_kv * head_dim, dtype=bf16),
                torch.randn(tokens, 1, dtype=bf16),
                torch.randn(tokens, 1, dtype=bf16),
            )

        return module, rotary_in
    if name == "o_proj":
        return nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=bf16), \
            lambda t: (torch.randn(t, n_heads * head_dim, dtype=bf16),)
    if name == "gate_up_proj":
        return nn.Linear(hidden, 2 * inter, bias=False, dtype=bf16), h_in
    if name == "act_fn":
        return SiluAndMul(), lambda t: (torch.randn(t, 2 * inter, dtype=bf16),)
    if name == "down_proj":
        return nn.Linear(inter, hidden, bias=False, dtype=bf16), \
            lambda t: (torch.randn(t, inter, dtype=bf16),)
    raise KeyError(f"unknown dense layer: {name}")


def per_sequence_layer(name: str, cfg: dict[str, Any]) -> tuple[nn.Module, Callable[[int], tuple]]:
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    bf16 = torch.bfloat16

    def h_in(seqs: int) -> tuple:
        return (torch.randn(seqs, hidden, dtype=bf16),)

    if name == "lm_head":
        return nn.Linear(hidden, vocab, bias=False, dtype=bf16), h_in

    if name == "sampler":
        class Sampler(nn.Module):
            def forward(self, logits: torch.Tensor) -> torch.Tensor:
                probs = torch.softmax(logits.float(), dim=-1)
                return probs.argmax(dim=-1)

        return Sampler(), lambda s: (torch.randn(s, vocab, dtype=bf16),)
    raise KeyError(f"unknown per-sequence layer: {name}")


DENSE_LAYERS = (
    "embedding", "layernorm", "qkv_proj", "rotary_emb",
    "o_proj", "gate_up_proj", "act_fn", "down_proj", "final_layernorm",
)
PER_SEQUENCE_LAYERS = ("lm_head", "sampler")


def x2_grid(start: int, stop: int) -> list[int]:
    """``start`` doubling up to ``stop`` inclusive, the contract's x2 style."""
    out, value = [], start
    while value <= stop:
        out.append(value)
        value *= 2
    return out


def attention_shots(prefill_chunks, kv_prefills, n_decodes, kv_decodes):
    """Prefill-only and decode-only shots, the contract's two pure regimes."""
    for chunk in prefill_chunks:
        for kv in kv_prefills:
            yield {"prefill_chunk": chunk, "kv_prefill": kv, "n_decode": 0, "kv_decode": 0}
    for n in n_decodes:
        for kv in kv_decodes:
            yield {"prefill_chunk": 0, "kv_prefill": 0, "n_decode": n, "kv_decode": kv}


def kv_bytes(cfg, shot) -> int:
    """Bytes the K and V tensors of one shot would occupy on the PE."""
    head_dim, n_kv = cfg["head_dim"], cfg["num_key_value_heads"]
    total = 0
    if shot["prefill_chunk"]:
        ctx = shot["kv_prefill"] + shot["prefill_chunk"]
        total += 2 * n_kv * ctx * head_dim
    if shot["n_decode"]:
        total += 2 * shot["n_decode"] * n_kv * shot["kv_decode"] * head_dim
    return total * 2  # bfloat16


def attention_case(cfg, shot) -> tuple[nn.Module, tuple]:
    """Build the attention module and one shot's (q, k, v) tensors."""
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    bf16 = torch.bfloat16
    module = Attention(head_dim, head_dim ** -0.5)
    tensors: list[torch.Tensor] = []
    chunk, kv_p = shot["prefill_chunk"], shot["kv_prefill"]
    n_dec, kv_d = shot["n_decode"], shot["kv_decode"]
    # Grouped-query attention: batch over KV heads and fold each KV head's
    # query group into the query rows. Same FLOPs as expanding K/V to n_heads,
    # but K/V keep their true (unexpanded) size, which is what the KV cache
    # actually holds and what kv_bytes budgets.
    group = max(1, n_heads // n_kv)
    if chunk:
        ctx = kv_p + chunk
        tensors += [
            torch.randn(n_kv, group * chunk, head_dim, dtype=bf16),
            torch.randn(n_kv, ctx, head_dim, dtype=bf16),
            torch.randn(n_kv, ctx, head_dim, dtype=bf16),
        ]
    if n_dec:
        tensors += [
            torch.randn(n_dec * n_kv, group, head_dim, dtype=bf16),
            torch.randn(n_dec * n_kv, kv_d, head_dim, dtype=bf16),
            torch.randn(n_dec * n_kv, kv_d, head_dim, dtype=bf16),
        ]
    return module, tuple(tensors)


def write_csv(path: Path, columns: Iterable[str], rows: list[dict]) -> None:
    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})
