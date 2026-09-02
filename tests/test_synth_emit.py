"""BundleEmitter produces contract-satisfying bundles (STEP 7)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from planner.util.tier import ProfileTier, resolve_bundle_tier
from profiler.contract import validate_bundle
from profiler.core.config import load_architecture
from profiler.core.importer import CsvProfileImporter, ImportProvenance
from profiler.synth.__main__ import main as synth_main
from profiler.synth.backend import AnalyticalProfileBackend
from profiler.synth.device import DeviceSpec
from profiler.synth.dims import ModelDims
from profiler.synth.emit import BundleEmitter, EmitError, GridParams

REPO = Path(__file__).resolve().parents[1]
LLAMA = "meta-llama/Llama-3.1-8B"
QWEN_MOE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
A40_MEASURED = REPO / "profiler" / "perf" / "A40" / LLAMA / "bf16"

#: Small grid so the suite stays fast; mirror tests use the real bundle.
SMALL_GRID = GridParams(max_num_batched_tokens=64, max_num_seqs=8, attention_max_kv=64)

DEVICE = DeviceSpec(
    label="A40-t0",
    peak_flops=149.7e12,
    mem_bandwidth_bytes=696e9,
    flops_efficiency=0.55,
    mem_efficiency=0.75,
)


def _emitter(model: str = LLAMA, *, tps=(1,), label="A40-t0", grid=SMALL_GRID,
             mirror=None, force=False, out_root: Path) -> BundleEmitter:
    dims = ModelDims.from_hf_config(REPO / "configs" / "model" / f"{model}.json", "bf16")
    arch = load_architecture(REPO / "profiler" / "models" / f"{dims.model_type}.yaml")
    backends = {
        tp: AnalyticalProfileBackend(dims, arch, DEVICE, tp) for tp in tps
    }
    return BundleEmitter(
        dims=dims, arch=arch, backend_for_tp=backends, hardware_label=label,
        variant="bf16", out_root=out_root, grid=grid, mirror_root=mirror,
        datasheet_source="test fixture", efficiency={"flops": 0.55, "mem": 0.75},
        force=force, generated_at="2026-09-02T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def emitted(tmp_path_factory) -> Path:
    """One small-grid Llama bundle shared by the read-only assertions."""
    out = tmp_path_factory.mktemp("perf")
    report = _emitter(out_root=out, tps=(1, 2)).emit()
    return report.out_root


def test_emitted_bundle_passes_contract_validation(emitted):
    """The synthetic bundle passes contract.validate_bundle."""
    validate_bundle(emitted, [1, 2], is_moe=False)


def test_emitted_bundle_passes_csv_importer(emitted, tmp_path):
    """CsvProfileImporter's validation mode accepts the synthetic bundle,
    and source: analytical is an accepted provenance value (STEP 1)."""
    importer = CsvProfileImporter(perf_root=tmp_path)
    report = importer.validate(
        emitted, hardware="A40-t0", model=LLAMA, variant="bf16"
    )
    assert report.tp_degrees == [1, 2]
    assert not report.has_skew
    ImportProvenance(measured_by="synthetic", source="analytical").validate()


def test_headers_match_real_bundle_byte_for_byte(emitted):
    """Synthetic CSV headers equal the measured bundle's headers exactly."""
    for name in ("dense.csv", "per_sequence.csv", "attention.csv"):
        synth_header = (emitted / "tp1" / name).read_text().splitlines()[0]
        real_header = (A40_MEASURED / "tp1" / name).read_text().splitlines()[0]
        assert synth_header == real_header, name


def test_all_times_positive(emitted):
    """Every emitted time_us is strictly positive."""
    for name in ("dense.csv", "per_sequence.csv", "attention.csv"):
        with (emitted / "tp1" / name).open() as f:
            for row in csv.DictReader(f):
                assert float(row["time_us"]) > 0, (name, row)


def test_mirror_keys_reproduces_exact_key_set(tmp_path):
    """--mirror-keys reproduces the measured key set exactly (STEP 8 precondition)."""
    report = _emitter(out_root=tmp_path, mirror=A40_MEASURED).emit()

    def keys(path: Path, cols: tuple[str, ...]) -> set[tuple]:
        with path.open() as f:
            return {tuple(r[c] for c in cols) for r in csv.DictReader(f)}

    for name, cols in (
        ("dense.csv", ("layer", "tokens")),
        ("per_sequence.csv", ("layer", "sequences")),
        ("attention.csv", ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode")),
    ):
        assert keys(report.out_root / "tp1" / name, cols) == keys(
            A40_MEASURED / "tp1" / name, cols
        ), name


def test_default_grid_keys_match_categories(emitted):
    """Grid-mode dense keys equal categories.py's _token_grid enumeration."""
    from profiler.core.categories import _token_grid

    with (emitted / "tp1" / "dense.csv").open() as f:
        tokens = sorted({int(r["tokens"]) for r in csv.DictReader(f)})
    assert tokens == _token_grid(SMALL_GRID.max_num_batched_tokens)


def test_moe_bundle_emits_moe_csv_tp1_only(tmp_path):
    """A MoE model gets moe.csv in tp1/ only; dense models get none."""
    report = _emitter(QWEN_MOE, tps=(1, 2), out_root=tmp_path).emit()
    assert (report.out_root / "tp1" / "moe.csv").exists()
    assert not (report.out_root / "tp2" / "moe.csv").exists()
    validate_bundle(report.out_root, [1, 2], is_moe=True)

    dense = _emitter(out_root=tmp_path / "dense").emit()
    assert not (dense.out_root / "tp1" / "moe.csv").exists()


def test_meta_yaml_records_tier_and_nulls(emitted):
    """meta.yaml carries tier/source/cost_model/efficiency and null_reason (A2)."""
    meta = yaml.safe_load((emitted / "meta.yaml").read_text())
    assert meta["tier"] == "analytical"
    assert meta["source"] == "analytical"
    assert meta["cost_model"] == "roofline-v1"
    assert meta["efficiency"]["flops"] == 0.55
    assert meta["datasheet_source"] == "test fixture"
    assert meta["vllm_version"] is None
    assert meta["cuda_version"] is None
    assert meta["gpu"] is None
    assert "synthetic bundle" in meta["null_reason"]
    assert meta["grid"]["max_num_batched_tokens"] == 64


def test_skew_omitted_and_declared(emitted):
    """No skew*.csv is written and meta.yaml says so."""
    assert not (emitted / "tp1" / "skew.csv").exists()
    assert not (emitted / "tp1" / "skew_fit.csv").exists()
    meta = yaml.safe_load((emitted / "meta.yaml").read_text())
    assert meta["skew"].startswith("omitted")


def test_label_without_suffix_rejected(tmp_path):
    """A hardware label without -t0/-t1 is refused (A3)."""
    with pytest.raises(EmitError, match="-t0"):
        _emitter(label="A40", out_root=tmp_path)


def test_label_tier_mismatch_rejected(tmp_path):
    """An analytical backend cannot write under a -t1 label."""
    with pytest.raises(EmitError, match="does not match tier"):
        _emitter(label="A40-t1", out_root=tmp_path)


def test_existing_output_not_overwritten(tmp_path):
    """Re-emitting without force is refused; force replaces."""
    _emitter(out_root=tmp_path).emit()
    with pytest.raises(EmitError, match="already exists"):
        _emitter(out_root=tmp_path).emit()
    _emitter(out_root=tmp_path, force=True).emit()  # must not raise


def test_partial_bundle_cleaned_on_validation_failure(tmp_path, monkeypatch):
    """A failed self-validation leaves no files behind."""
    import profiler.synth.emit as emit_mod

    def boom(*args, **kwargs):
        raise emit_mod.contract.ProfileContractError("forced failure")

    monkeypatch.setattr(emit_mod.contract, "validate_bundle", boom)
    emitter = _emitter(out_root=tmp_path)
    with pytest.raises(emit_mod.contract.ProfileContractError):
        emitter.emit()
    assert not (tmp_path / "A40-t0").exists()


def test_resolve_bundle_tier_reads_analytical(emitted):
    """STEP 2's resolver reads the emitted bundle as ANALYTICAL (integration)."""
    # emitted = <perf root>/A40-t0/meta-llama/Llama-3.1-8B/bf16
    perf_root = emitted.parents[3]
    assert (
        resolve_bundle_tier(perf_root, "A40-t0", LLAMA, "bf16")
        is ProfileTier.ANALYTICAL
    )


def test_cli_smoke(tmp_path, capsys):
    """python -m profiler.synth emit exits 0 and writes a bundle."""
    accel = tmp_path / "a40_t0.yaml"
    accel.write_text(yaml.safe_dump({
        "profile_id": "a40-t0-test", "vendor": "NVIDIA", "model": "A40",
        "backend": "cuda", "memory_gb": 45, "memory_bandwidth_gbps": 696,
        "sim_hardware": "A40-t0", "source": "vendor_spec",
        "datasheet": {
            "peak_tflops": {"bf16": 149.7}, "compute_units": 84,
            "flops_efficiency": 0.55, "mem_efficiency": 0.75,
            "datasheet_source": "NVIDIA A40 datasheet (test fixture)",
        },
    }))
    rc = synth_main([
        "emit", "--accelerator", str(accel), "--model", LLAMA,
        "--variant", "bf16", "--tp", "1", "--hardware-label", "A40-t0",
        "--max-num-batched-tokens", "32", "--max-num-seqs", "4",
        "--attention-max-kv", "32", "--out", str(tmp_path / "perf"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wrote" in out
    assert (tmp_path / "perf" / "A40-t0" / LLAMA / "bf16" / "meta.yaml").exists()
