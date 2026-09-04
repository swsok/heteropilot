"""AttentionCostModel 4-axis scaling and measured-bundle comparison (STEP 6)."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest

from profiler.synth.attn import AttentionCostModel, AttnError
from profiler.synth.device import DeviceSpec
from profiler.synth.dims import ModelDims

REPO = Path(__file__).resolve().parents[1]
LLAMA_CFG = REPO / "configs" / "model" / "meta-llama" / "Llama-3.1-8B.json"
A40_ATTN = (
    REPO / "profiler" / "perf" / "A40" / "meta-llama" / "Llama-3.1-8B"
    / "bf16" / "tp1" / "attention.csv"
)

#: A40 datasheet values (STEP 3/5): dense bf16 149.7 TFLOP/s, 696 GB/s.
DEVICE = DeviceSpec(
    label="A40",
    peak_flops=149.7e12,
    mem_bandwidth_bytes=696e9,
    flops_efficiency=1.0,
    mem_efficiency=1.0,
)


def _model(tp: int = 1, mode: str = "max") -> AttentionCostModel:
    dims = ModelDims.from_hf_config(LLAMA_CFG, "bf16")
    return AttentionCostModel(dims, DEVICE, tp=tp, mode=mode)


def test_decode_only_point():
    """prefill_chunk=0 leaves only the decode terms."""
    m = _model()
    fp, bp, fd, bd = m._phase_costs(0, 0, 4, 1024)
    assert fp == 0 and bp == 0
    assert fd > 0 and bd > 0


def test_prefill_only_point():
    """n_decode=0 leaves only the prefill terms."""
    m = _model()
    fp, bp, fd, bd = m._phase_costs(256, 512, 0, 0)
    assert fd == 0 and bd == 0
    assert fp > 0 and bp > 0


def test_prefill_quadratic_in_chunk():
    """With kv_prefill=0, doubling the chunk quadruples prefill FLOPs.

    Causal: flops ~ pc*(0 + pc/2) = pc^2/2, so exactly 4x.
    """
    m = _model()
    f1, _, _, _ = m._phase_costs(256, 0, 0, 0)
    f2, _, _, _ = m._phase_costs(512, 0, 0, 0)
    assert f2 == pytest.approx(4 * f1)


def test_prefill_linear_in_kv_prefill():
    """With the chunk fixed, prefill FLOPs are linear in kv_prefill for
    kv_prefill >> pc (the pc/2 causal term becomes negligible)."""
    m = _model()
    pc = 16
    f_lo, *_ = m._phase_costs(pc, 8192, 0, 0)
    f_hi, *_ = m._phase_costs(pc, 16384, 0, 0)
    # (16384 + 8) / (8192 + 8) within 0.1%
    assert f_hi / f_lo == pytest.approx((16384 + pc / 2) / (8192 + pc / 2), rel=1e-6)
    assert f_hi / f_lo == pytest.approx(2.0, rel=1e-3)


def test_decode_linear_in_n_decode():
    """Decode FLOPs and bytes are linear in n_decode."""
    m = _model()
    _, _, f1, b1 = m._phase_costs(0, 0, 2, 1024)
    _, _, f2, b2 = m._phase_costs(0, 0, 4, 1024)
    assert f2 == pytest.approx(2 * f1)
    assert b2 == pytest.approx(2 * b1)


def test_decode_linear_in_kv_decode():
    """Decode FLOPs and bytes are linear in kv_decode."""
    m = _model()
    _, _, f1, b1 = m._phase_costs(0, 0, 2, 1024)
    _, _, f2, b2 = m._phase_costs(0, 0, 2, 2048)
    assert f2 == pytest.approx(2 * f1)
    assert b2 == pytest.approx(2 * b1)


def test_gqa_uses_kv_heads_for_bytes():
    """Decode bytes are priced with n_kv, not n_q (4x apart on Llama-3.1-8B)."""
    dims = ModelDims.from_hf_config(LLAMA_CFG, "bf16")
    assert dims.num_attention_heads == 32 and dims.num_key_value_heads == 8
    m = _model()
    _, _, _, bd = m._phase_costs(0, 0, 1, 1024)
    expected = 1 * 1024 * 2 * dims.num_key_value_heads * dims.head_dim * dims.kv_dtype_bytes
    assert bd == pytest.approx(expected)
    # Pricing with n_q would be exactly 4x this; catch that class of bug.
    assert bd * 4 == pytest.approx(
        1 * 1024 * 2 * dims.num_attention_heads * dims.head_dim * dims.kv_dtype_bytes
    )


def test_kv_dtype_scales_decode_bytes():
    """An fp8 KV cache halves decode bytes relative to bf16."""
    dims8 = ModelDims.from_hf_config(LLAMA_CFG, "bf16-kvfp8")
    m8 = AttentionCostModel(dims8, DEVICE, tp=1)
    _, _, _, bd8 = m8._phase_costs(0, 0, 1, 1024)
    _, _, _, bd16 = _model()._phase_costs(0, 0, 1, 1024)
    assert bd8 == pytest.approx(bd16 / 2)


def test_tp_shards_heads():
    """TP=2 halves the local heads, so the estimate drops."""
    big = (512, 8192, 8, 4096)
    assert _model(tp=2).estimate_key_us(*big) < _model(tp=1).estimate_key_us(*big)


def test_degenerate_point_handling():
    """prefill_chunk=0 and n_decode=0 raises.

    Investigated 2026-09-02: the measured A40 tp1 attention.csv contains zero
    rows with both axes 0 (prefill_chunk=0 means decode-only per CONTRACT.md),
    so the model refuses the meaningless key instead of pricing it.
    """
    with pytest.raises(AttnError, match="degenerate"):
        _model().estimate_key_us(0, 0, 0, 0)
    with pytest.raises(AttnError, match="degenerate"):
        _model().estimate_key_us(0, 512, 0, 512)


def test_mode_sum_vs_max():
    """The sum variant is never below the fused-max variant on mixed keys."""
    key = (256, 4096, 8, 2048)
    assert _model(mode="sum").estimate_key_us(*key) >= _model(mode="max").estimate_key_us(*key)


# --- measured comparison -----------------------------------------------------

def _read_attention_rows() -> list[tuple[int, int, int, int, float]]:
    with A40_ATTN.open() as f:
        return [
            (
                int(r["prefill_chunk"]), int(r["kv_prefill"]),
                int(r["n_decode"]), int(r["kv_decode"]), float(r["time_us"]),
            )
            for r in csv.DictReader(f)
        ]


def _spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation with average ranks for ties (no scipy dependency)."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return num / den


def test_lower_bound_property_on_real_attention_csv():
    """eff=1.0 estimates lower-bound every measured attention.csv row.

    Verified over all 8643 A40 tp1 rows with zero violations at the time of
    writing; any violating row is printed so the broken axis is obvious.
    """
    m = _model()
    violations = []
    for pc, kvp, nd, kvd, measured in _read_attention_rows():
        est = m.estimate_key_us(pc, kvp, nd, kvd)
        if est > measured:
            violations.append((pc, kvp, nd, kvd, round(est, 2), measured))
    assert not violations, f"estimate exceeded measurement: {violations[:10]}"


def test_relative_shape_correlation():
    """Spearman rank correlation of estimate vs measurement >= 0.9.

    Absolute error is fixable by the STEP 9 efficiency calibration; a broken
    ORDER is a structural formula error and must stop the work here.
    (Observed 0.975 at the time of writing.)
    """
    m = _model()
    rows = _read_attention_rows()
    est = [m.estimate_key_us(pc, kvp, nd, kvd) for pc, kvp, nd, kvd, _ in rows]
    measured = [t for *_, t in rows]
    rho = _spearman(est, measured)
    assert rho >= 0.9, f"spearman {rho:.4f} < 0.9: structurally wrong formula"


def test_median_ratio_documented():
    """The eff=1.0 median est/measured ratio stays a sane lower bound."""
    m = _model()
    ratios = [
        m.estimate_key_us(pc, kvp, nd, kvd) / t
        for pc, kvp, nd, kvd, t in _read_attention_rows()
    ]
    med = statistics.median(ratios)
    assert 0.05 <= med <= 1.0, f"median ratio {med:.4f} outside [0.05, 1.0]"
