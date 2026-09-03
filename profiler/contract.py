"""Single source of truth for the profile-bundle CSV contract.

Promoted out of ``profiler/core/importer.py`` (STEP 1 of
WORK_ORDER_tiered_profiles.md) so that the importer and the synthetic
bundle generator (``profiler.synth``) validate against one schema table
instead of two copies that can drift. ``profiler/CONTRACT.md`` is the
human-readable form of the same contract.

Location note: the work order places this file at
``profiler/core/contract.py`` and asks to investigate whether ruff's
``extend-exclude`` supports ``!``-negation to lint just that one file
inside the excluded ``profiler/core``. Verified 2026-09-02 with ruff
0.16.1: a ``"!profiler/core/contract.py"`` entry has no effect under
``ruff check .`` (the file stays excluded), so the work order's stated
fallback applies — the module lives at ``profiler/contract.py``, outside
the excluded subpackage, where ruff and mypy actually check it.

Import hygiene: standard library + ``yaml`` only. No torch / vllm /
pandas — this module must import in the CPU-only planner venv.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_SOURCES",
    "SCHEMAS",
    "TP_DIR_RE",
    "CsvSchema",
    "ProfileContractError",
    "as_float",
    "as_int",
    "cast_value",
    "schema_for",
    "validate_bundle",
    "validate_csv",
]


class ProfileContractError(ValueError):
    """A bundle violates ``profiler/CONTRACT.md``.

    Raised for a missing required file, a header that does not match the
    contract byte-for-byte, an un-parseable / wrong-type value, a duplicate
    measurement key, or a non-positive ``time_us``.
    """


@dataclass(frozen=True)
class CsvSchema:
    """Exact column contract for one CSV file in a bundle."""

    filename: str
    #: Header columns, in order. Compared byte-for-byte against the file.
    columns: tuple[str, ...]
    int_columns: frozenset[str]
    float_columns: frozenset[str]
    str_columns: frozenset[str]
    #: Columns whose parsed tuple must be unique across rows. Empty tuple
    #: disables the uniqueness check (skew tables are not simple keyed
    #: measurements).
    key_columns: tuple[str, ...]
    #: Column that must be strictly positive, or ``None`` to skip. Only the
    #: four keyed measurement files carry ``time_us``; skew columns
    #: (``alpha``, ``t_*_us``) legitimately go negative and must not be
    #: sign-checked.
    time_column: str | None
    #: ``True`` when the file must exist for every profiled TP degree.
    required: bool
    #: Present only for MoE models; skipped otherwise.
    moe_only: bool = False
    #: Float columns that may be empty. The profiler writes an empty cell for
    #: skew.csv's ``alpha`` when the fit is undefined (NaN guard at
    #: profiler/core/skew.py:353); every shipped measured bundle contains such
    #: rows, so the contract must accept them (real artifacts win).
    nullable_columns: frozenset[str] = frozenset()


# dtype casters keyed by the type sets above.
def as_int(value: str) -> int:
    return int(value)


def as_float(value: str) -> float:
    return float(value)


# Schemas, in a deterministic order (validated + written in this order).
SCHEMAS: tuple[CsvSchema, ...] = (
    CsvSchema(
        filename="dense.csv",
        columns=("layer", "tokens", "time_us"),
        int_columns=frozenset({"tokens"}),
        float_columns=frozenset({"time_us"}),
        str_columns=frozenset({"layer"}),
        key_columns=("layer", "tokens"),
        time_column="time_us",
        required=True,
    ),
    CsvSchema(
        filename="per_sequence.csv",
        columns=("layer", "sequences", "time_us"),
        int_columns=frozenset({"sequences"}),
        float_columns=frozenset({"time_us"}),
        str_columns=frozenset({"layer"}),
        key_columns=("layer", "sequences"),
        time_column="time_us",
        required=True,
    ),
    CsvSchema(
        filename="attention.csv",
        columns=("prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"),
        int_columns=frozenset({"prefill_chunk", "kv_prefill", "n_decode", "kv_decode"}),
        float_columns=frozenset({"time_us"}),
        str_columns=frozenset(),
        key_columns=("prefill_chunk", "kv_prefill", "n_decode", "kv_decode"),
        time_column="time_us",
        required=True,
    ),
    CsvSchema(
        filename="moe.csv",
        columns=("tokens", "activated_experts", "time_us"),
        int_columns=frozenset({"tokens", "activated_experts"}),
        float_columns=frozenset({"time_us"}),
        str_columns=frozenset(),
        key_columns=("tokens", "activated_experts"),
        time_column="time_us",
        required=True,
        moe_only=True,
    ),
    CsvSchema(
        filename="skew.csv",
        columns=(
            "regime", "n", "nb", "ratio", "skew", "pc", "kp", "kvs",
            "kv_big", "kv_mean", "t_mean_us", "t_max_us", "t_skew_us", "alpha",
        ),
        int_columns=frozenset({"n", "nb", "pc", "kp", "kvs", "kv_big", "kv_mean"}),
        float_columns=frozenset({"ratio", "skew", "t_mean_us", "t_max_us", "t_skew_us", "alpha"}),
        str_columns=frozenset({"regime"}),
        key_columns=(),
        time_column=None,
        required=False,
        nullable_columns=frozenset({"alpha"}),
    ),
    CsvSchema(
        filename="skew_fit.csv",
        columns=(
            "pc", "n_label", "skew_rate_label", "kv_big_label",
            "kp_label", "alpha", "n_samples",
        ),
        int_columns=frozenset({"pc", "n_samples"}),
        float_columns=frozenset({"alpha"}),
        str_columns=frozenset({"n_label", "skew_rate_label", "kv_big_label", "kp_label"}),
        key_columns=(),
        time_column=None,
        required=False,
    ),
)

TP_DIR_RE = re.compile(r"^tp(\d+)$")

#: Allowed values for ``meta.yaml``'s ``source`` (and, since the tiered
#: profile work, ``tier``) fields. ``analytical`` (Tier 0, datasheet-derived)
#: and ``calibrated`` (Tier 1, analytical + measured anchors) were added in
#: STEP 1 of WORK_ORDER_tiered_profiles.md.
ALLOWED_SOURCES = ("imported", "measured", "placeholder", "analytical", "calibrated")


def cast_value(
    path: Path, lineno: int, column: str, value: str, caster: Callable[[str], Any]
) -> Any:
    """Cast one CSV cell, raising ProfileContractError on failure.

    Error text matches the importer's historical ``_cast`` exactly (the type
    name is derived by stripping the caster-function prefix).
    """
    try:
        return caster(value)
    except (TypeError, ValueError):
        expected = caster.__name__.removeprefix("_as_").removeprefix("as_")
        raise ProfileContractError(
            f"{path}:{lineno}: column {column!r} has a bad value {value!r} "
            f"(expected {expected})."
        ) from None


def schema_for(filename: str) -> CsvSchema:
    """Return the schema for one contract CSV filename."""
    for schema in SCHEMAS:
        if schema.filename == filename:
            return schema
    raise ProfileContractError(
        f"{filename!r} is not a contract CSV; expected one of "
        f"{[s.filename for s in SCHEMAS]}."
    )


def validate_csv(path: Path, schema: CsvSchema) -> None:
    """Validate one CSV against its schema; raise ProfileContractError on breach.

    Checks: header byte-for-byte, column count, cell types, non-empty
    strings, key uniqueness, and strictly positive ``time_column``. This is
    the pure-validation core of ``CsvProfileImporter._validate_csv`` (which
    additionally builds coverage reports); both consult the same SCHEMAS.
    """
    # utf-8-sig so a stray BOM is stripped rather than corrupting the first
    # header cell into a confusing "header does not match".
    try:
        f = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ProfileContractError(f"{path}: cannot open - {exc}") from exc
    with f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ProfileContractError(f"{path} is empty (no header row).") from None

        expected = list(schema.columns)
        if header != expected:
            raise ProfileContractError(
                f"{path}: header does not match the contract.\n"
                f"  expected: {expected}\n"
                f"  found:    {header}"
            )

        int_idx = [i for i, c in enumerate(schema.columns) if c in schema.int_columns]
        float_idx = [i for i, c in enumerate(schema.columns) if c in schema.float_columns]
        str_idx = [i for i, c in enumerate(schema.columns) if c in schema.str_columns]
        key_idx = [schema.columns.index(c) for c in schema.key_columns]
        time_idx = (
            schema.columns.index(schema.time_column)
            if schema.time_column is not None
            else None
        )

        seen_keys: set[tuple[Any, ...]] = set()
        rows = 0
        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(schema.columns):
                raise ProfileContractError(
                    f"{path}:{lineno}: expected {len(schema.columns)} "
                    f"columns, found {len(row)}: {row}"
                )
            typed: list[Any] = list(row)
            for i in int_idx:
                typed[i] = cast_value(path, lineno, schema.columns[i], row[i], as_int)
            for i in float_idx:
                if row[i] == "" and schema.columns[i] in schema.nullable_columns:
                    typed[i] = None
                    continue
                typed[i] = cast_value(path, lineno, schema.columns[i], row[i], as_float)
            for i in str_idx:
                if not str(typed[i]).strip():
                    raise ProfileContractError(
                        f"{path}:{lineno}: column {schema.columns[i]!r} "
                        f"must be a non-empty string."
                    )
            if time_idx is not None:
                t = typed[time_idx]
                if not (isinstance(t, float) and t > 0.0):
                    raise ProfileContractError(
                        f"{path}:{lineno}: {schema.time_column} must be "
                        f"positive, got {t!r}."
                    )
            if key_idx:
                key = tuple(typed[i] for i in key_idx)
                if key in seen_keys:
                    cols = ", ".join(schema.key_columns)
                    raise ProfileContractError(
                        f"{path}:{lineno}: duplicate key ({cols})={key}. "
                        f"Importers must emit unique keys."
                    )
                seen_keys.add(key)
            rows += 1

    if rows == 0:
        raise ProfileContractError(f"{path}: header present but no data rows.")


def validate_bundle(variant_root: Path, tp_degrees: list[int], is_moe: bool) -> None:
    """Validate a whole variant-root bundle (tp<N>/ dirs of CSVs).

    Shared by the importer path and the synthetic generator's self-check.
    Raises ProfileContractError on the first violation. Mirrors the
    importer's rules: every requested tp<N>/ exists, required files present
    and valid, optional files valid when present, and skew coverage is
    all-or-nothing across TP degrees.
    """
    variant_root = Path(variant_root)
    if not variant_root.is_dir():
        raise ProfileContractError(f"bundle source is not a directory: {variant_root}")
    if not tp_degrees:
        raise ProfileContractError(
            f"no TP degrees given for {variant_root}; a bundle must contain "
            f"at least one tp<N>/ directory."
        )

    skew_present: set[int] = set()
    for tp in sorted(tp_degrees):
        tp_dir = variant_root / f"tp{tp}"
        if not tp_dir.is_dir():
            raise ProfileContractError(
                f"requested TP degree {tp} but {tp_dir} does not exist."
            )
        for schema in SCHEMAS:
            if schema.moe_only and not is_moe:
                continue
            # moe.csv lives in tp1/ only: the profiler measures MoE once at
            # TP=1 and the simulator scales per-expert time by ep_size
            # (categories_for; verified 2026-09-02 - the shipped Qwen3-30B
            # bundle has moe.csv in tp1/ and not tp2/). Real artifacts win.
            required = schema.required and not (schema.moe_only and tp != 1)
            path = tp_dir / schema.filename
            if not path.exists():
                if required:
                    raise ProfileContractError(
                        f"required file missing: {path} "
                        f"(contract requires {schema.filename} in every tp<N>/"
                        + (" for MoE models)" if schema.moe_only else ")")
                    )
                continue
            validate_csv(path, schema)
            if schema.filename == "skew.csv":
                skew_present.add(tp)

    if skew_present and skew_present != set(tp_degrees):
        missing = sorted(set(tp_degrees) - skew_present)
        raise ProfileContractError(
            f"inconsistent skew coverage: skew.csv present for TP "
            f"{sorted(skew_present)} but missing for TP {missing}. Skew "
            f"must be present for every profiled TP or none."
        )
