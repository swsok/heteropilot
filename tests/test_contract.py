"""Guarantees profiler.contract is the single source of the CSV contract (STEP 1)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from profiler.contract import (
    ALLOWED_SOURCES,
    SCHEMAS,
    ProfileContractError,
    schema_for,
    validate_bundle,
    validate_csv,
)

REPO = Path(__file__).resolve().parent.parent
CONTRACT_MD = REPO / "profiler" / "CONTRACT.md"
PERF = REPO / "profiler" / "perf"
RTX_TP1 = PERF / "RTXPRO6000" / "meta-llama" / "Llama-3.1-8B" / "bf16" / "tp1"

#: The three repo-shipped measured bundles STEP 1 uses as ground truth.
REAL_VARIANT_ROOTS = [
    PERF / "RTXPRO6000" / "meta-llama" / "Llama-3.1-8B" / "bf16",
    PERF / "A40" / "meta-llama" / "Llama-3.1-8B" / "bf16",
    PERF / "RNGD-CARD" / "meta-llama" / "Llama-3.1-8B" / "bf16",
]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_schemas_cover_all_contract_files():
    """SCHEMAS covers exactly the CSV files documented in CONTRACT.md."""
    text = CONTRACT_MD.read_text(encoding="utf-8")
    documented = set(re.findall(r"^### (\S+\.csv)", text, flags=re.MULTILINE))
    assert documented == {s.filename for s in SCHEMAS}


def test_columns_match_real_bundle_headers():
    """SCHEMAS columns match the shipped measured bundle headers byte-for-byte."""
    checked = 0
    for schema in SCHEMAS:
        path = RTX_TP1 / schema.filename
        if not path.exists():
            continue  # moe.csv is absent for a dense model
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == ",".join(schema.columns), schema.filename
        checked += 1
    assert checked >= 5  # dense, per_sequence, attention, skew, skew_fit


def test_allowed_sources_extended():
    """analytical / calibrated are allowed source values (STEP 1)."""
    assert "analytical" in ALLOWED_SOURCES
    assert "calibrated" in ALLOWED_SOURCES
    # The pre-existing values survive.
    assert {"imported", "measured", "placeholder"} <= set(ALLOWED_SOURCES)


def test_validate_csv_accepts_real_bundle():
    """Every CSV of the three shipped measured bundles passes validate_csv."""
    for variant_root in REAL_VARIANT_ROOTS:
        assert variant_root.is_dir(), variant_root
        tp_degrees = sorted(
            int(d.name[2:]) for d in variant_root.iterdir()
            if d.is_dir() and d.name.startswith("tp")
        )
        assert tp_degrees, variant_root
        for tp in tp_degrees:
            tp_dir = variant_root / f"tp{tp}"
            for schema in SCHEMAS:
                path = tp_dir / schema.filename
                if not path.exists():
                    # RNGD-CARD ships no skew*.csv; those are optional.
                    assert not schema.required or schema.moe_only, path
                    continue
                validate_csv(path, schema)
        # And the bundle-level check agrees.
        is_moe = (variant_root / f"tp{tp_degrees[0]}" / "moe.csv").exists()
        validate_bundle(variant_root, tp_degrees, is_moe)


def test_validate_csv_rejects_renamed_column(tmp_path):
    """A renamed header column raises ProfileContractError."""
    p = _write(tmp_path / "dense.csv", "layer,ntokens,time_us\nqkv_proj,16,1.5\n")
    with pytest.raises(ProfileContractError, match="header does not match"):
        validate_csv(p, schema_for("dense.csv"))


def test_validate_csv_rejects_duplicate_key(tmp_path):
    """A repeated measurement key raises ProfileContractError."""
    p = _write(
        tmp_path / "dense.csv",
        "layer,tokens,time_us\nqkv_proj,16,1.5\nqkv_proj,16,1.6\n",
    )
    with pytest.raises(ProfileContractError, match="duplicate key"):
        validate_csv(p, schema_for("dense.csv"))


def test_validate_csv_rejects_nonpositive_time(tmp_path):
    """time_us <= 0 raises ProfileContractError."""
    for bad in ("0.0", "-1.0"):
        p = _write(tmp_path / "dense.csv", f"layer,tokens,time_us\nqkv_proj,16,{bad}\n")
        with pytest.raises(ProfileContractError, match="must be positive"):
            validate_csv(p, schema_for("dense.csv"))


def test_validate_csv_allows_negative_alpha_in_skew(tmp_path):
    """skew.csv's alpha / t_*_us may be negative (importer behavior preserved)."""
    header = (
        "regime,n,nb,ratio,skew,pc,kp,kvs,kv_big,kv_mean,"
        "t_mean_us,t_max_us,t_skew_us,alpha"
    )
    row = "pure,4,2,0.5,0.1,0,0,128,256,128,10.0,12.0,-2.0,-0.3"
    p = _write(tmp_path / "skew.csv", f"{header}\n{row}\n")
    validate_csv(p, schema_for("skew.csv"))  # must not raise


def test_validate_csv_allows_empty_alpha_in_skew_only(tmp_path):
    """skew.csv's alpha may be empty (undefined fit, skew.py:353); time_us may not."""
    header = (
        "regime,n,nb,ratio,skew,pc,kp,kvs,kv_big,kv_mean,"
        "t_mean_us,t_max_us,t_skew_us,alpha"
    )
    row = "mixed,2,1,0.5,4.0,16,8192,1024,4096,2560,553.6,553.1,553.6,"
    p = _write(tmp_path / "skew.csv", f"{header}\n{row}\n")
    validate_csv(p, schema_for("skew.csv"))  # must not raise

    q = _write(tmp_path / "dense.csv", "layer,tokens,time_us\nqkv_proj,16,\n")
    with pytest.raises(ProfileContractError, match="bad value"):
        validate_csv(q, schema_for("dense.csv"))


def test_validate_bundle_rejects_missing_required_file(tmp_path):
    """A tp dir without dense.csv fails bundle validation."""
    tp_dir = tmp_path / "tp1"
    tp_dir.mkdir()
    _write(tp_dir / "per_sequence.csv", "layer,sequences,time_us\nlm_head,1,2.0\n")
    _write(
        tp_dir / "attention.csv",
        "prefill_chunk,kv_prefill,n_decode,kv_decode,time_us\n0,0,1,16,3.0\n",
    )
    with pytest.raises(ProfileContractError, match="required file missing"):
        validate_bundle(tmp_path, [1], is_moe=False)


def test_schema_for_unknown_filename():
    """schema_for rejects non-contract filenames."""
    with pytest.raises(ProfileContractError):
        schema_for("timings.csv")


def test_importer_aliases_point_at_contract():
    """importer's historical private names alias the contract symbols."""
    from profiler.core import importer

    assert importer._SCHEMAS is SCHEMAS
    assert importer._ALLOWED_SOURCES is ALLOWED_SOURCES
    assert importer.ProfileContractError is ProfileContractError
