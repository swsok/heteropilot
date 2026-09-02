"""Tier 1 calibration correctness and discipline (STEP 9)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest
import yaml

from planner.util.tier import ProfileTier, resolve_bundle_tier
from profiler.contract import ProfileContractError, schema_for
from profiler.core.config import load_architecture
from profiler.synth.__main__ import main as synth_main
from profiler.synth.backend import AnalyticalProfileBackend, CalibratedProfileBackend
from profiler.synth.calibrate import ScalingTable, fit_from_anchors, pick_anchors
from profiler.synth.device import DeviceSpec
from profiler.synth.diff import LAYER_FAMILY
from profiler.synth.dims import ModelDims

REPO = Path(__file__).resolve().parents[1]
LLAMA = "meta-llama/Llama-3.1-8B"
A40 = REPO / "profiler" / "perf" / "A40" / LLAMA / "bf16"

DEVICE = DeviceSpec(
    label="A40-t1", peak_flops=149.7e12, mem_bandwidth_bytes=696e9,
    flops_efficiency=0.7468390409816932, mem_efficiency=0.5319523056625672,
    family_efficiency={
        # The committed A40 V3 fit (profiles/accelerators/a40.efficiency.yaml).
        "attention": 0.26187364905355237, "elementwise": 0.5319523056625672,
        "gather": 0.7961560162297642, "gemm": 0.7468390409816932,
    },
)


@pytest.fixture(scope="module")
def backend() -> AnalyticalProfileBackend:
    dims = ModelDims.from_hf_config(REPO / "configs" / "model" / f"{LLAMA}.json", "bf16")
    arch = load_architecture(REPO / "profiler" / "models" / "llama.yaml")
    return AnalyticalProfileBackend(dims, arch, DEVICE, 1)


def _write_anchor_csv(dest_tp: Path, filename: str, rows: list[tuple]) -> None:
    schema = schema_for(filename)
    dest_tp.mkdir(parents=True, exist_ok=True)
    with (dest_tp / filename).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(schema.columns)
        writer.writerows(rows)


def _synthetic_anchors(tmp_path: Path, backend, *, gemm_scale: float,
                       elementwise_scale: float) -> Path:
    """Anchors whose measurements are exactly scaled Tier 0 outputs."""
    tp_dir = tmp_path / "anchors" / "tp1"
    dense_rows = []
    for layer, scale in (
        ("qkv_proj", gemm_scale), ("o_proj", gemm_scale),
        ("down_proj", gemm_scale), ("gate_up_proj", gemm_scale),
        ("layernorm", elementwise_scale), ("act_fn", elementwise_scale),
        ("rotary_emb", elementwise_scale),
    ):
        for tokens in (1, 4, 16, 64, 256, 1024):
            dense_rows.append(
                (layer, tokens, f"{backend.dense_us(layer, tokens) * scale:.6g}")
            )
    _write_anchor_csv(tp_dir, "dense.csv", dense_rows)
    return tmp_path / "anchors"


def test_fit_recovers_known_scale(tmp_path, backend):
    """Anchors at exactly 2x (gemm) / 3x (elementwise) Tier 0 recover both."""
    anchors = _synthetic_anchors(tmp_path, backend, gemm_scale=2.0, elementwise_scale=3.0)
    table = fit_from_anchors(anchors, backend, tp=1)
    assert table.scalars["gemm"] == pytest.approx(2.0, rel=1e-6)
    assert table.scalars["elementwise"] == pytest.approx(3.0, rel=1e-6)


def test_fit_is_per_family_not_global(tmp_path, backend):
    """Family multipliers do not bleed into each other."""
    anchors = _synthetic_anchors(tmp_path, backend, gemm_scale=2.0, elementwise_scale=3.0)
    table = fit_from_anchors(anchors, backend, tp=1)
    assert table.scale("gemm", 100.0) == pytest.approx(2.0, rel=1e-6)
    assert table.scale("elementwise", 100.0) == pytest.approx(3.0, rel=1e-6)
    # A family with no anchors at all stays at identity (A2).
    assert table.scale("attention", 100.0) == 1.0
    assert table.scale("moe", 100.0) == 1.0


def test_empty_anchors_yields_identity(tmp_path, backend):
    """No anchor rows at all produce the identity table (A2)."""
    (tmp_path / "anchors" / "tp1").mkdir(parents=True)
    table = fit_from_anchors(tmp_path / "anchors", backend, tp=1)
    assert table.scalars == {}
    assert table.piecewise == {}
    for family in ("gemm", "elementwise", "attention", "moe"):
        assert table.scale(family, 1e6) == 1.0


def test_too_few_family_anchors_stay_identity(tmp_path, backend):
    """A family below min_family_anchors is identity, not a 1-point scalar.

    The hold-out experiment showed a single launch-floor embedding anchor
    fitting gather at 36x and wrecking the whole family."""
    tp_dir = tmp_path / "anchors" / "tp1"
    _write_anchor_csv(tp_dir, "dense.csv", [
        ("embedding", 1, f"{backend.dense_us('embedding', 1) * 36.0:.6g}"),
    ])
    table = fit_from_anchors(tmp_path / "anchors", backend, tp=1)
    assert table.scale("gather", 1.0) == 1.0


def test_anchor_csv_validated_by_contract(tmp_path, backend):
    """A malformed anchor CSV raises ProfileContractError."""
    tp_dir = tmp_path / "anchors" / "tp1"
    tp_dir.mkdir(parents=True)
    (tp_dir / "dense.csv").write_text("layer,ntokens,time_us\nqkv_proj,16,1.5\n")
    with pytest.raises(ProfileContractError, match="header"):
        fit_from_anchors(tmp_path / "anchors", backend, tp=1)


def test_partial_anchor_csv_accepted(tmp_path, backend):
    """A small SUBSET of the grid is a valid anchor file (no new format)."""
    tp_dir = tmp_path / "anchors" / "tp1"
    rows = [("qkv_proj", t, f"{backend.dense_us('qkv_proj', t) * 1.5:.6g}")
            for t in (1, 2, 4, 8, 16, 32, 64, 128)]
    _write_anchor_csv(tp_dir, "dense.csv", rows)
    table = fit_from_anchors(tmp_path / "anchors", backend, tp=1)
    assert table.scalars["gemm"] == pytest.approx(1.5, rel=1e-4)  # 6-sig-fig CSV rounding


def test_tier1_improves_on_tier0_with_real_anchors(tmp_path, backend):
    """Hold-out: a 5% random anchor subset lowers MAPE on the other 95%.

    Split is deterministic (random.Random(42)); the improvement magnitude is
    recorded in docs/tier0_calibration.md, not asserted here.
    """
    rows: list[tuple[str, tuple, float]] = []
    for name in ("dense.csv", "per_sequence.csv", "attention.csv"):
        schema = schema_for(name)
        with (A40 / "tp1" / name).open() as f:
            for r in csv.DictReader(f):
                key = tuple(
                    int(r[c]) if c in schema.int_columns else r[c]
                    for c in schema.key_columns
                )
                rows.append((name, key, float(r["time_us"])))
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_anchor = int(len(rows) * 0.05)
    anchor_rows = [rows[i] for i in idx[:n_anchor]]
    holdout = [rows[i] for i in idx[n_anchor:]]

    by_file: dict[str, list[tuple]] = {}
    for name, key, t in anchor_rows:
        by_file.setdefault(name, []).append((*key, f"{t:.6g}"))
    for name, file_rows in by_file.items():
        _write_anchor_csv(tmp_path / "anchors" / "tp1", name, file_rows)

    table = fit_from_anchors(tmp_path / "anchors", backend, tp=1)

    est = {
        "dense.csv": lambda k: backend.dense_us(str(k[0]), int(k[1])),
        "per_sequence.csv": lambda k: backend.per_sequence_us(str(k[0]), int(k[1])),
        "attention.csv": lambda k: backend.attention_us(*k),
    }

    def family_of(name: str, key: tuple) -> str:
        if name == "attention.csv":
            return "attention"
        return LAYER_FAMILY.get(str(key[0]), "unknown")

    def mape(with_table: bool) -> float:
        errs = []
        for name, key, measured in holdout:
            t0 = est[name](key)
            t = t0 * table.scale(family_of(name, key), t0) if with_table else t0
            errs.append(abs(t - measured) / measured)
        return sum(errs) / len(errs)

    tier0, tier1 = mape(False), mape(True)
    assert tier1 < tier0, f"Tier 1 ({tier1:.3f}) must beat Tier 0 ({tier0:.3f})"


def test_attention_share_respected():
    """pick_anchors with attention_share=0.7 spends ~70% on attention keys."""
    keys = {
        "attention.csv": [(i, 0, 1, 16) for i in range(1000)],
        "dense.csv": [("qkv_proj", t) for t in range(500)],
        "per_sequence.csv": [("lm_head", s) for s in range(80)],
    }
    plan = pick_anchors(keys, budget=200, attention_share=0.7)
    n_attn = len(plan["attention.csv"])
    assert n_attn == 140  # round(200 * 0.7)
    assert sum(len(v) for v in plan.values()) == 200


def test_pick_anchors_covers_feature_range():
    """Picked anchors span the whole sweep, not one end of it."""
    tokens = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 2048]
    keys = {"dense.csv": [("qkv_proj", t) for t in tokens]}
    plan = pick_anchors(keys, budget=5, attention_share=0.0)
    picked = [k[1] for k in plan["dense.csv"]]
    assert picked[0] == 1  # smallest
    assert picked[-1] == 2048  # largest
    assert len(set(picked)) == 5


def test_scaling_table_yaml_roundtrip(tmp_path, backend):
    """save/load preserve scalars and piecewise rows."""
    anchors = _synthetic_anchors(tmp_path, backend, gemm_scale=2.0, elementwise_scale=3.0)
    table = fit_from_anchors(anchors, backend, tp=1, derived_from={"x": "y"})
    table.save(tmp_path / "scaling.yaml")
    loaded = ScalingTable.load(tmp_path / "scaling.yaml")
    assert loaded.scalars == pytest.approx(table.scalars)
    assert loaded.piecewise.keys() == table.piecewise.keys()
    assert loaded.derived_from == {"x": "y"}


def test_calibrated_bundle_tier_is_calibrated(tmp_path, backend):
    """--scaling emits tier: calibrated under a -t1 label; STEP 2 reads it."""
    anchors = _synthetic_anchors(tmp_path, backend, gemm_scale=2.0, elementwise_scale=3.0)
    table = fit_from_anchors(anchors, backend, tp=1)
    table.save(tmp_path / "scaling.yaml")

    accel = tmp_path / "a40_t1.yaml"
    accel.write_text(yaml.safe_dump({
        "profile_id": "a40-t1-test", "vendor": "NVIDIA", "model": "A40",
        "backend": "cuda", "memory_gb": 45, "memory_bandwidth_gbps": 696,
        "sim_hardware": "A40-t1", "source": "vendor_spec",
        "datasheet": {
            "peak_tflops": {"bf16": 149.7},
            "flops_efficiency": 0.75, "mem_efficiency": 0.53,
            "datasheet_source": "NVIDIA A40 datasheet (test fixture)",
        },
    }))
    rc = synth_main([
        "emit", "--accelerator", str(accel), "--model", LLAMA,
        "--variant", "bf16", "--tp", "1", "--hardware-label", "A40-t1",
        "--scaling", str(tmp_path / "scaling.yaml"),
        "--max-num-batched-tokens", "32", "--max-num-seqs", "4",
        "--attention-max-kv", "32", "--out", str(tmp_path / "perf"),
    ])
    assert rc == 0
    assert (
        resolve_bundle_tier(tmp_path / "perf", "A40-t1", LLAMA, "bf16")
        is ProfileTier.CALIBRATED
    )
    meta = yaml.safe_load(
        (tmp_path / "perf" / "A40-t1" / LLAMA / "bf16" / "meta.yaml").read_text()
    )
    assert meta["tier"] == "calibrated"
    assert meta["source"] == "calibrated"
    assert meta["calibration_anchors"]["n_anchors"] == 42  # 7 layers x 6 tokens
    assert meta["calibration_anchors"]["anchors_per_family"]["gemm"] == 24


def test_calibrated_backend_applies_scaling(backend):
    """CalibratedProfileBackend multiplies every family, attention included."""
    dims = ModelDims.from_hf_config(REPO / "configs" / "model" / f"{LLAMA}.json", "bf16")
    arch = load_architecture(REPO / "profiler" / "models" / "llama.yaml")
    table = ScalingTable(scalars={"gemm": 2.0, "attention": 3.0})
    cal = CalibratedProfileBackend(dims, arch, DEVICE, 1, scaling=table)
    assert cal.dense_us("qkv_proj", 64) == pytest.approx(
        2.0 * backend.dense_us("qkv_proj", 64)
    )
    assert cal.attention_us(0, 0, 4, 1024) == pytest.approx(
        3.0 * backend.attention_us(0, 0, 4, 1024)
    )
    assert cal.tier == "calibrated"
