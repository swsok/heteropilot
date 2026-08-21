"""Unit tests for the Phase 3 external-CSV profile importer.

Covers the ``profiler/CONTRACT.md`` importer checklist: valid bundles import
cleanly and land in the perf layout with a ``meta.yaml`` carrying source
attribution; malformed bundles (missing file, missing / renamed column, bad
type, duplicate key, non-positive time, missing attribution) each fail loudly
with :class:`ProfileContractError`.

Everything runs on tiny synthetic CSVs — no GPU, no real model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from profiler.core.importer import (
    CsvProfileImporter,
    ImportProvenance,
    ProfileContractError,
)

# ---------------------------------------------------------------------------
# Synthetic-bundle fixtures
# ---------------------------------------------------------------------------

_DENSE = [
    ["layer", "tokens", "time_us"],
    ["qkv_proj", "1", "4.20"],
    ["qkv_proj", "2", "5.10"],
    ["act_fn", "1", "1.11"],
]
_PER_SEQ = [
    ["layer", "sequences", "time_us"],
    ["lm_head", "1", "1476.26"],
    ["lm_head", "2", "1487.80"],
]
_ATTENTION = [
    ["prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"],
    ["0", "0", "1", "16", "12.51"],
    ["0", "0", "1", "32", "12.56"],
    ["16", "0", "0", "0", "20.00"],
]
_SKEW = [
    ["regime", "n", "nb", "ratio", "skew", "pc", "kp", "kvs",
     "kv_big", "kv_mean", "t_mean_us", "t_max_us", "t_skew_us", "alpha"],
    ["pure", "2", "1", "0.5", "4.0", "0", "0", "128",
     "512", "320", "71.3", "88.5", "72.0", "0.0426"],
    # A legitimately negative alpha — must not be rejected.
    ["pure", "2", "1", "0.5", "4.0", "0", "0", "256",
     "1024", "640", "93.1", "113.0", "86.0", "-0.3548"],
]
_SKEW_FIT = [
    ["pc", "n_label", "skew_rate_label", "kv_big_label", "kp_label", "alpha", "n_samples"],
    ["0", "n<=128", "sr<=15%", "kvB<=16k", "kp=0", "0.0068", "4"],
]
_MOE = [
    ["tokens", "activated_experts", "time_us"],
    ["1", "2", "30.0"],
    ["2", "4", "55.0"],
]


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _make_bundle(root: Path, *, tps=(1,), skew=True, moe=False) -> Path:
    """Write a valid synthetic variant-root bundle under ``root``."""
    for tp in tps:
        tp_dir = root / f"tp{tp}"
        _write_csv(tp_dir / "dense.csv", _DENSE)
        _write_csv(tp_dir / "per_sequence.csv", _PER_SEQ)
        _write_csv(tp_dir / "attention.csv", _ATTENTION)
        if skew:
            _write_csv(tp_dir / "skew.csv", _SKEW)
            _write_csv(tp_dir / "skew_fit.csv", _SKEW_FIT)
        if moe:
            _write_csv(tp_dir / "moe.csv", _MOE)
    return root


def _provenance(**over) -> ImportProvenance:
    kwargs = {
        "measured_by": "ETRI internal benchmark 2026-08",
        "source": "imported",
        "serving_stack": "vllm-rbln 0.7.0",
        "runtime_version": "rbln-driver 1.2",
        "backend": "rbln",
        "device": "Rebellions ATOM",
        "measurement_method": "3-iteration averaged forward",
        "measurement_iterations": 3,
    }
    kwargs.update(over)
    return ImportProvenance(**kwargs)


@pytest.fixture
def importer(tmp_path: Path) -> CsvProfileImporter:
    return CsvProfileImporter(perf_root=tmp_path / "perf")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_bundle_validates(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    report = importer.validate(
        src, hardware="ATOM", model="meta-llama/Llama-3.1-8B", variant="bf16",
    )
    assert report.tp_degrees == [1]
    assert report.has_skew and report.has_skew_fit
    dense = report.tp_reports[1].files["dense.csv"]
    assert dense.rows == 3
    assert dense.coverage["tokens"] == {"min": 1, "max": 2, "n_unique": 2}


def test_valid_bundle_imports_and_writes_meta(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src", tps=(1, 2))
    dest = importer.import_bundle(
        src, hardware="ATOM", model="meta-llama/Llama-3.1-8B", variant="bf16",
        provenance=_provenance(),
    )
    # Layout: perf/<hw>/<model>/<variant>/{meta.yaml, tp1/, tp2/}
    assert dest == importer.perf_root / "ATOM" / "meta-llama/Llama-3.1-8B" / "bf16"
    for tp in (1, 2):
        for fname in ("dense.csv", "per_sequence.csv", "attention.csv",
                      "skew.csv", "skew_fit.csv"):
            assert (dest / f"tp{tp}" / fname).is_file()

    meta = yaml.safe_load((dest / "meta.yaml").read_text())
    assert meta["source"] == "imported"
    assert meta["measured_by"] == "ETRI internal benchmark 2026-08"
    assert meta["backend"] == "rbln"
    assert meta["hardware"] == "ATOM"
    assert meta["tp_degrees"] == [1, 2]
    assert meta["skew_profile"]["enabled"] is True
    assert meta["measurement_iterations"] == 3
    # Coverage derived from CSV keys, not declared factors.
    assert meta["measured_coverage"]["tp1"]["dense.csv"]["rows"] == 3


def test_import_copies_verbatim(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    dest = importer.import_bundle(
        src, hardware="ATOM", model="m", variant="bf16", provenance=_provenance(),
    )
    got = (dest / "tp1" / "dense.csv").read_text()
    want = (src / "tp1" / "dense.csv").read_text()
    assert got == want


def test_moe_bundle(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src", moe=True)
    report = importer.validate(src, hardware="ATOM", model="m", variant="bf16", is_moe=True)
    assert report.tp_reports[1].files["moe.csv"].rows == 2


def test_tp_degrees_autodiscovered(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src", tps=(1, 2, 4))
    report = importer.validate(src, hardware="ATOM", model="m", variant="bf16")
    assert report.tp_degrees == [1, 2, 4]


def test_skew_optional(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src", skew=False)
    report = importer.validate(src, hardware="ATOM", model="m", variant="bf16")
    assert not report.has_skew
    dest = importer.import_bundle(
        src, hardware="ATOM", model="m", variant="bf16", provenance=_provenance(),
    )
    meta = yaml.safe_load((dest / "meta.yaml").read_text())
    assert meta["skew_profile"]["enabled"] is False
    assert not (dest / "tp1" / "skew.csv").exists()


# ---------------------------------------------------------------------------
# Loud-failure paths
# ---------------------------------------------------------------------------

def test_missing_required_file(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    (src / "tp1" / "attention.csv").unlink()
    with pytest.raises(ProfileContractError, match=r"required file missing.*attention\.csv"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_missing_moe_file_when_moe(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")  # no moe.csv
    with pytest.raises(ProfileContractError, match=r"moe\.csv"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16", is_moe=True)


def test_missing_column(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [["layer", "time_us"], ["qkv_proj", "4.2"]]  # 'tokens' dropped
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="header does not match"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_renamed_column(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [["layer", "tokens", "us"], ["qkv_proj", "1", "4.2"]]  # time_us -> us
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="header does not match"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_bad_int_type(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [["layer", "tokens", "time_us"], ["qkv_proj", "one", "4.2"]]
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="bad value 'one'"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_bad_float_type(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [["layer", "tokens", "time_us"], ["qkv_proj", "1", "fast"]]
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="bad value 'fast'"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_non_positive_time(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [["layer", "tokens", "time_us"], ["qkv_proj", "1", "0.0"]]
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="must be positive"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_duplicate_key(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [
        ["layer", "tokens", "time_us"],
        ["qkv_proj", "1", "4.2"],
        ["qkv_proj", "1", "4.9"],  # same (layer, tokens)
    ]
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="duplicate key"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_negative_alpha_in_skew_is_allowed(tmp_path: Path, importer: CsvProfileImporter) -> None:
    # _SKEW already contains a -0.3548 alpha row; validation must accept it.
    src = _make_bundle(tmp_path / "src")
    report = importer.validate(src, hardware="ATOM", model="m", variant="bf16")
    assert report.tp_reports[1].files["skew.csv"].rows == 2


def test_empty_data_rows(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    _write_csv(src / "tp1" / "dense.csv", [["layer", "tokens", "time_us"]])
    with pytest.raises(ProfileContractError, match="no data rows"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_wrong_column_count_row(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    bad = [["layer", "tokens", "time_us"], ["qkv_proj", "1"]]  # short row
    _write_csv(src / "tp1" / "dense.csv", bad)
    with pytest.raises(ProfileContractError, match="expected 3 columns"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_no_tp_dirs(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(ProfileContractError, match="no tp<N>/"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


def test_requested_tp_missing(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src", tps=(1,))
    with pytest.raises(ProfileContractError, match="tp2 does not exist"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16", tp_degrees=[1, 2])


# ---------------------------------------------------------------------------
# Provenance discipline
# ---------------------------------------------------------------------------

def test_missing_attribution_rejected(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    with pytest.raises(ProfileContractError, match="measured_by is required"):
        importer.import_bundle(
            src, hardware="ATOM", model="m", variant="bf16",
            provenance=ImportProvenance(measured_by="   "),
        )


def test_bad_source_rejected(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    with pytest.raises(ProfileContractError, match="source='synthesized' is invalid"):
        importer.import_bundle(
            src, hardware="ATOM", model="m", variant="bf16",
            provenance=_provenance(source="synthesized"),
        )


# ---------------------------------------------------------------------------
# Overwrite semantics
# ---------------------------------------------------------------------------

def test_overwrite_guard(tmp_path: Path, importer: CsvProfileImporter) -> None:
    src = _make_bundle(tmp_path / "src")
    importer.import_bundle(
        src, hardware="ATOM", model="m", variant="bf16", provenance=_provenance(),
    )
    with pytest.raises(ProfileContractError, match="destination already exists"):
        importer.import_bundle(
            src, hardware="ATOM", model="m", variant="bf16", provenance=_provenance(),
        )
    # With overwrite=True it succeeds.
    dest = importer.import_bundle(
        src, hardware="ATOM", model="m", variant="bf16", provenance=_provenance(),
        overwrite=True,
    )
    assert (dest / "meta.yaml").is_file()


# ---------------------------------------------------------------------------
# Skew sign-exemption applies to ALL skew time columns, not just alpha
# ---------------------------------------------------------------------------

def test_negative_skew_time_columns_are_allowed(
    tmp_path: Path, importer: CsvProfileImporter
) -> None:
    # t_mean_us / t_max_us / t_skew_us are float, sign-exempt (time_column=None
    # for skew.csv). A negative delta must validate, not just a negative alpha.
    src = _make_bundle(tmp_path / "src")
    rows = [
        _SKEW[0],
        ["pure", "2", "1", "0.5", "4.0", "0", "0", "128",
         "512", "320", "-5.0", "-3.0", "-4.0", "-0.10"],
    ]
    _write_csv(src / "tp1" / "skew.csv", rows)
    report = importer.validate(src, hardware="ATOM", model="m", variant="bf16")
    assert report.tp_reports[1].files["skew.csv"].rows == 1


def test_float_in_skew_int_column_rejected(
    tmp_path: Path, importer: CsvProfileImporter
) -> None:
    # kv_mean is an int column even in skew.csv; a float there is a contract
    # violation despite skew's sign/time exemptions.
    src = _make_bundle(tmp_path / "src")
    bad = [row[:] for row in _SKEW]
    bad[1][9] = "320.5"  # kv_mean -> float
    _write_csv(src / "tp1" / "skew.csv", bad)
    with pytest.raises(ProfileContractError, match=r"bad value '320\.5'"):
        importer.validate(src, hardware="ATOM", model="m", variant="bf16")


# ---------------------------------------------------------------------------
# extra_meta may not override validated provenance keys (absolute rule 3)
# ---------------------------------------------------------------------------

def test_extra_meta_cannot_override_reserved_keys(
    tmp_path: Path, importer: CsvProfileImporter
) -> None:
    src = _make_bundle(tmp_path / "src")
    with pytest.raises(ProfileContractError, match="may not override reserved"):
        importer.import_bundle(
            src, hardware="ATOM", model="m", variant="bf16",
            provenance=_provenance(),
            extra_meta={"source": "measured"},
        )


def test_extra_meta_new_keys_are_kept(
    tmp_path: Path, importer: CsvProfileImporter
) -> None:
    src = _make_bundle(tmp_path / "src")
    dest = importer.import_bundle(
        src, hardware="ATOM", model="m", variant="bf16",
        provenance=_provenance(),
        extra_meta={"external_ticket": "JIRA-1234"},
    )
    meta = yaml.safe_load((dest / "meta.yaml").read_text())
    assert meta["external_ticket"] == "JIRA-1234"
    assert meta["source"] == "imported"  # provenance still wins
