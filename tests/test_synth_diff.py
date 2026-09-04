"""diff harness correctness (STEP 8)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from profiler.contract import ProfileContractError, schema_for
from profiler.synth.diff import DiffError, diff_bundles, fit_efficiency, spearman

REPO = Path(__file__).resolve().parents[1]
A40 = REPO / "profiler" / "perf" / "A40" / "meta-llama" / "Llama-3.1-8B" / "bf16"


def _copy_bundle(dst: Path, scale: float = 1.0, tp: int = 1) -> Path:
    """Copy the measured A40 tp1 keyed CSVs, scaling every time_us."""
    tp_dir = dst / f"tp{tp}"
    tp_dir.mkdir(parents=True)
    for name in ("dense.csv", "per_sequence.csv", "attention.csv"):
        schema = schema_for(name)
        src_path = A40 / f"tp{tp}" / name
        with src_path.open() as f, (tp_dir / name).open("w", newline="") as out:
            reader = csv.DictReader(f)
            writer = csv.DictWriter(out, fieldnames=list(schema.columns))
            writer.writeheader()
            for row in reader:
                row["time_us"] = f"{float(row['time_us']) * scale:.6g}"
                writer.writerow(row)
    return dst


def test_identical_bundles_have_zero_error(tmp_path):
    """Diffing a bundle against a copy of itself gives MAPE=0, ratio=1, rho=1."""
    copy = _copy_bundle(tmp_path / "copy")
    report = diff_bundles(A40, copy, tp=1)
    assert report.n_keys_only_measured == 0
    assert report.n_keys_only_synth == 0
    for d in report.per_file.values():
        # 6-sig-fig reformatting of the copied values costs < 0.01%.
        assert d.mape == pytest.approx(0.0, abs=1e-4)
        assert d.median_ratio == pytest.approx(1.0, abs=1e-4)
        assert d.spearman == pytest.approx(1.0, abs=1e-6)


def test_scaled_bundle_reports_known_error(tmp_path):
    """A 1.5x-scaled bundle reports MAPE=50%, median_ratio=1.5, rho=1."""
    scaled = _copy_bundle(tmp_path / "scaled", scale=1.5)
    report = diff_bundles(A40, scaled, tp=1)
    for d in report.per_file.values():
        assert d.mape == pytest.approx(0.5, abs=1e-3)
        assert d.median_ratio == pytest.approx(1.5, abs=1e-3)
        assert d.spearman == pytest.approx(1.0, abs=1e-6)
    assert report.overall_mape == pytest.approx(0.5, abs=1e-3)


def test_key_mismatch_is_reported_not_silently_dropped(tmp_path):
    """Keys existing on one side only appear in n_keys_only_*."""
    partial = _copy_bundle(tmp_path / "partial")
    # Remove the last 10 data rows of dense.csv on the synth side.
    dense = partial / "tp1" / "dense.csv"
    lines = dense.read_text().splitlines()
    dense.write_text("\n".join(lines[:-10]) + "\n")
    report = diff_bundles(A40, partial, tp=1)
    assert report.n_keys_only_measured == 10
    assert report.per_file["dense.csv"].n_only_measured == 10
    assert report.n_keys_only_synth == 0


def test_no_shared_keys_raises(tmp_path):
    """Two disjoint bundles cannot be compared."""
    empty = tmp_path / "empty" / "tp1"
    empty.mkdir(parents=True)
    for name in ("dense.csv", "per_sequence.csv", "attention.csv"):
        (empty / name).write_text(
            ",".join(schema_for(name).columns) + "\n"
        )
    with pytest.raises((DiffError, ProfileContractError)):
        diff_bundles(A40, tmp_path / "empty", tp=1)


def test_spearman_implementation():
    """The scipy-free rank correlation is exact on known cases."""
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Known middle case: one swapped pair among four -> rho = 0.8.
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)


def test_per_layer_and_per_family_breakdown(tmp_path):
    """dense.csv layers and kernel families are all aggregated."""
    copy = _copy_bundle(tmp_path / "copy")
    report = diff_bundles(A40, copy, tp=1)
    assert {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "layernorm",
            "act_fn", "rotary_emb", "embedding", "final_layernorm",
            "lm_head", "sampler"} <= set(report.per_layer)
    assert {"gemm", "elementwise", "gather", "attention"} <= set(report.per_family)
    for d in report.per_layer.values():
        assert d.n > 0


def test_fit_efficiency_writes_provenance():
    """The committed fit artifacts carry derived_from (path/commit/date)."""
    for name in ("a40.efficiency.yaml", "rtxpro6000.efficiency.yaml"):
        path = REPO / "profiles" / "accelerators" / name
        data = yaml.safe_load(path.read_text())
        derived = data["derived_from"]
        assert derived["measured_bundle"]
        assert derived["git_commit"]
        assert derived["date"]
        assert set(data["family_efficiency"]) >= {"gemm", "elementwise", "attention"}


def test_fit_efficiency_does_not_touch_accelerator_yaml():
    """The fit output stays separate: a40.yaml efficiencies remain empty."""
    from planner.inventory import load_accelerator_profile

    for name in ("a40.yaml", "rtxpro6000.yaml"):
        profile = load_accelerator_profile(REPO / "profiles" / "accelerators" / name)
        assert profile.datasheet is not None
        assert profile.datasheet.flops_efficiency is None
        assert profile.datasheet.mem_efficiency is None
        assert profile.datasheet.family_efficiency == {}


def test_fit_efficiency_bounds(tmp_path):
    """A fitted efficiency > 1.0 is reported as a violation, never clamped."""

    class HalfSpeedBackend:
        """Pretends the 'lower bound' is 2x the measurement (bound broken)."""

        tier = "analytical"

        def dense_us(self, layer, tokens):
            return 1e9

        def per_sequence_us(self, layer, sequences):
            return 1e9

        def attention_us(self, pc, kvp, nd, kvd):
            return 1e9

        def expert_us(self, tokens, activated):
            return 1e9

    copy = _copy_bundle(tmp_path / "copy")
    fit = fit_efficiency(copy, 1, HalfSpeedBackend(), derived_from={"x": "y"})
    assert fit.bound_violations  # every family blew the bound
    for family, eff in fit.bound_violations.items():
        assert eff > 1.0
        # Not clamped: the recorded table carries the same >1 value.
        assert fit.family_efficiency[family] == eff


def test_fit_efficiency_on_real_bundle_matches_committed():
    """Re-fitting A40 V3 reproduces the committed efficiencies (determinism)."""
    from planner.inventory import load_accelerator_profile
    from profiler.core.config import load_architecture
    from profiler.synth.backend import AnalyticalProfileBackend
    from profiler.synth.device import DeviceSpec
    from profiler.synth.dims import ModelDims

    profile = load_accelerator_profile(REPO / "profiles" / "accelerators" / "a40.yaml")
    assert profile.datasheet is not None
    lb = profile.model_copy(update={"datasheet": profile.datasheet.model_copy(
        update={"flops_efficiency": 1.0, "mem_efficiency": 1.0}
    )})
    dims = ModelDims.from_hf_config(
        REPO / "configs" / "model" / "meta-llama" / "Llama-3.1-8B.json", "bf16"
    )
    arch = load_architecture(REPO / "profiler" / "models" / "llama.yaml")
    backend = AnalyticalProfileBackend(dims, arch, DeviceSpec.from_profile(lb, "bf16"), 1)
    fit = fit_efficiency(A40, 1, backend, derived_from={})
    committed = yaml.safe_load(
        (REPO / "profiles" / "accelerators" / "a40.efficiency.yaml").read_text()
    )
    for family, eff in committed["family_efficiency"].items():
        assert fit.family_efficiency[family] == pytest.approx(eff, rel=1e-9), family
    assert not fit.bound_violations


def test_render_table_smoke(tmp_path):
    """The human-readable table renders every group."""
    copy = _copy_bundle(tmp_path / "copy")
    text = diff_bundles(A40, copy, tp=1).render_table()
    assert "overall MAPE" in text
    assert "family:attention" in text
    assert "layer:qkv_proj" in text
