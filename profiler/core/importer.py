"""External-CSV profile importer (``CsvProfileImporter``).

Phase 3, NPU bring-up **V1**: import externally-measured latency CSVs (e.g.
vLLM-Ascend / vllm-rbln / furiosa-llm benchmark dumps, or a collaborator's
measurements) into the ``profiler/perf/<hardware>/<model>/<variant>/tp<N>/``
bundle layout the simulator's ``trace_generator.py`` consumes, *before* any
native NPU profiling adapter exists. See ``profiler/CONTRACT.md`` (the §3.7
CSV contract) and ``docs/hardware_roadmap.md`` step 4.

The work-order pseudocode (§3.7) names a ``HardwareProfilerBackend`` base with
``profile(self, spec)``. There is no such base class in this repository yet
(``CudaVllmProfiler`` is the ``python -m profiler`` CLI, not a class), and an
importer *imports* rather than *profiles*, so forcing a ``profile()`` method
here would be a misfit. This module is therefore a standalone, well-typed
class; if/when the ABC is introduced it can subclass it trivially.

Design contract (``profiler/CONTRACT.md`` importer checklist):
  1. Validate the header of every CSV byte-for-byte against the contract.
  2. Reject non-unique keys and non-positive ``time_us``.
  3. Emit ``meta.yaml`` with source attribution (``source: imported`` plus who
     measured the numbers, with what tool / from what publication).

Failure discipline (Risk-1 compatibility matrix, absolute rule 3): a malformed
bundle is **never** silently accepted. Every missing file, missing/renamed
column, un-parseable value, duplicate key, or non-positive time raises
``ProfileContractError`` with a message that names the file, row, and column.
Measured numbers are copied verbatim — the importer validates but never
transforms them.
"""

from __future__ import annotations

import csv
import datetime
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from profiler import __version__ as profiler_version

# The CSV schema table and its validation primitives were promoted to
# profiler/contract.py (WORK_ORDER_tiered_profiles.md STEP 1) so the importer
# and the synthetic bundle generator share one source of truth. The historical
# private names are kept as aliases for backward compatibility.
from profiler.contract import (
    ALLOWED_SOURCES,
    SCHEMAS,
    TP_DIR_RE,
    CsvSchema,
    ProfileContractError,
    as_float,
    as_int,
    cast_value,
)

__all__ = [
    "BundleReport",
    "CsvProfileImporter",
    "FileReport",
    "ImportProvenance",
    "ProfileContractError",
    "TpReport",
]

# Backward-compatible aliases for the pre-promotion private names.
_CsvSchema = CsvSchema
_SCHEMAS = SCHEMAS
_as_int = as_int
_as_float = as_float
_cast = cast_value
_TP_DIR_RE = TP_DIR_RE
_ALLOWED_SOURCES = ALLOWED_SOURCES


@dataclass
class ImportProvenance:
    """Where the imported numbers came from (§3.8 provenance discipline).

    ``measured_by`` is mandatory: absolute rule 3 forbids unattributed
    hardware numbers, so the importer refuses a bundle that cannot say who
    produced the measurements or which publication they were taken from.
    """

    #: Non-empty attribution: person/lab/tool that measured, or the
    #: publication being imported. Required.
    measured_by: str
    #: One of ``imported`` (external measurement, the default), ``measured``
    #: (measured on our own hardware but ingested via CSV), or
    #: ``placeholder``.
    source: str = "imported"
    #: Serving stack name + version that produced the numbers, e.g.
    #: ``"vllm-rbln 0.7.0"`` / ``"furiosa-llm 2025.4"``.
    serving_stack: str | None = None
    #: Backend runtime / driver version string.
    runtime_version: str | None = None
    #: Backend identifier (``cuda`` / ``ascend`` / ``rbln`` / ``furiosa``).
    backend: str | None = None
    #: Device string as the machine reports it (goes into meta ``gpu``).
    device: str | None = None
    #: Free-text description of how the numbers were measured.
    measurement_method: str | None = None
    #: Timed iterations averaged per shot, if known.
    measurement_iterations: int | None = None
    #: ISO-8601 timestamp; defaults to import time when omitted.
    profiled_at: str | None = None
    #: Any extra caveats worth carrying into the bundle.
    notes: str | None = None

    def validate(self) -> None:
        if not self.measured_by or not self.measured_by.strip():
            raise ProfileContractError(
                "ImportProvenance.measured_by is required and must be "
                "non-empty (absolute rule 3: no unattributed hardware "
                "numbers). Name the lab/tool/publication that produced "
                "these measurements."
            )
        if self.source not in _ALLOWED_SOURCES:
            raise ProfileContractError(
                f"ImportProvenance.source={self.source!r} is invalid; "
                f"expected one of {_ALLOWED_SOURCES}."
            )


# ---------------------------------------------------------------------------
# Validation reports
# ---------------------------------------------------------------------------

@dataclass
class FileReport:
    filename: str
    present: bool
    rows: int = 0
    #: Per-axis measured coverage: column -> {"min", "max", "n_unique"}.
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TpReport:
    tp: int
    files: dict[str, FileReport]

    @property
    def has_skew(self) -> bool:
        f = self.files.get("skew.csv")
        return bool(f and f.present)

    @property
    def has_skew_fit(self) -> bool:
        f = self.files.get("skew_fit.csv")
        return bool(f and f.present)


@dataclass
class BundleReport:
    hardware: str
    model: str
    variant: str
    is_moe: bool
    tp_reports: dict[int, TpReport]

    @property
    def tp_degrees(self) -> list[int]:
        return sorted(self.tp_reports)

    @property
    def has_skew(self) -> bool:
        # Skew data is only usable if present for every profiled TP.
        return bool(self.tp_reports) and all(
            r.has_skew for r in self.tp_reports.values()
        )

    @property
    def has_skew_fit(self) -> bool:
        return bool(self.tp_reports) and all(
            r.has_skew_fit for r in self.tp_reports.values()
        )


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class CsvProfileImporter:
    """Validate and import an externally-measured CSV bundle.

    Usage::

        importer = CsvProfileImporter(perf_root=Path("profiler/perf"))
        dest = importer.import_bundle(
            src=Path("/data/atom_llama31_8b"),   # holds tp1/, tp2/, ...
            hardware="ATOM",
            model="meta-llama/Llama-3.1-8B",
            variant="bf16",
            provenance=ImportProvenance(
                measured_by="Rebellions internal benchmark 2026-08",
                serving_stack="vllm-rbln 0.7.0",
                backend="rbln",
                device="Rebellions ATOM",
            ),
        )

    ``src`` is the *variant root*: a directory containing ``tp<N>/``
    subdirectories, each holding the per-TP CSVs. ``import_bundle`` first
    validates the whole bundle (raising :class:`ProfileContractError` on any
    contract breach and writing nothing), then copies every CSV verbatim into
    ``perf_root/<hardware>/<model>/<variant>/tp<N>/`` and writes a derived
    ``meta.yaml`` at the variant root.
    """

    def __init__(self, perf_root: Path | None = None) -> None:
        if perf_root is None:
            # profiler/core/importer.py -> profiler/perf
            perf_root = Path(__file__).resolve().parent.parent / "perf"
        self.perf_root = Path(perf_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        src: Path,
        *,
        hardware: str,
        model: str,
        variant: str,
        tp_degrees: list[int] | None = None,
        is_moe: bool = False,
    ) -> BundleReport:
        """Validate ``src`` against the contract without writing anything.

        Returns a :class:`BundleReport`. Raises :class:`ProfileContractError`
        on the first contract violation found.
        """
        src = Path(src)
        if not src.is_dir():
            raise ProfileContractError(f"bundle source is not a directory: {src}")

        tps = tp_degrees if tp_degrees is not None else self._discover_tp_degrees(src)
        if not tps:
            raise ProfileContractError(
                f"no tp<N>/ subdirectories found under {src}; a bundle must "
                f"contain at least one (e.g. tp1/)."
            )

        tp_reports: dict[int, TpReport] = {}
        for tp in sorted(tps):
            tp_dir = src / f"tp{tp}"
            if not tp_dir.is_dir():
                raise ProfileContractError(
                    f"requested TP degree {tp} but {tp_dir} does not exist."
                )
            tp_reports[tp] = self._validate_tp_dir(tp_dir, is_moe=is_moe)

        # Skew must be all-or-nothing across TP degrees: a bundle with skew for
        # some TPs but not others would write skew_profile.enabled=false into
        # meta.yaml while still shipping the partial skew CSVs, so meta would
        # contradict its own contents. Fail loud instead.
        skew_present = {tp for tp, r in tp_reports.items() if r.has_skew}
        if skew_present and skew_present != set(tp_reports):
            missing = sorted(set(tp_reports) - skew_present)
            raise ProfileContractError(
                f"inconsistent skew coverage: skew.csv present for TP "
                f"{sorted(skew_present)} but missing for TP {missing}. Skew "
                f"must be present for every profiled TP or none."
            )

        return BundleReport(
            hardware=hardware,
            model=model,
            variant=variant,
            is_moe=is_moe,
            tp_reports=tp_reports,
        )

    def import_bundle(
        self,
        src: Path,
        *,
        hardware: str,
        model: str,
        variant: str,
        provenance: ImportProvenance,
        tp_degrees: list[int] | None = None,
        is_moe: bool = False,
        overwrite: bool = False,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Validate then import a bundle into the perf tree.

        Returns the destination variant-root directory.
        """
        provenance.validate()
        report = self.validate(
            src,
            hardware=hardware,
            model=model,
            variant=variant,
            tp_degrees=tp_degrees,
            is_moe=is_moe,
        )

        dest_root = self.perf_root / hardware / model / variant
        if dest_root.exists() and not overwrite:
            raise ProfileContractError(
                f"destination already exists: {dest_root} (pass "
                f"overwrite=True to replace it)."
            )
        if dest_root.exists():
            shutil.rmtree(dest_root)

        src = Path(src)
        for tp in report.tp_degrees:
            src_tp = src / f"tp{tp}"
            dst_tp = dest_root / f"tp{tp}"
            dst_tp.mkdir(parents=True, exist_ok=True)
            for fr in report.tp_reports[tp].files.values():
                if fr.present:
                    shutil.copyfile(src_tp / fr.filename, dst_tp / fr.filename)

        self._write_meta(dest_root, report, provenance, extra_meta)
        return dest_root

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_tp_degrees(src: Path) -> list[int]:
        tps: list[int] = []
        for child in src.iterdir():
            if child.is_dir():
                m = _TP_DIR_RE.match(child.name)
                if m:
                    tps.append(int(m.group(1)))
        return sorted(tps)

    def _validate_tp_dir(self, tp_dir: Path, *, is_moe: bool) -> TpReport:
        files: dict[str, FileReport] = {}
        for schema in _SCHEMAS:
            if schema.moe_only and not is_moe:
                continue
            path = tp_dir / schema.filename
            required = schema.required
            if not path.exists():
                if required:
                    raise ProfileContractError(
                        f"required file missing: {path} "
                        f"(contract requires {schema.filename} in every tp<N>/"
                        + (" for MoE models)" if schema.moe_only else ")")
                    )
                files[schema.filename] = FileReport(schema.filename, present=False)
                continue
            files[schema.filename] = self._validate_csv(path, schema)
        return TpReport(tp=_tp_of(tp_dir), files=files)

    def _validate_csv(self, path: Path, schema: _CsvSchema) -> FileReport:
        # utf-8-sig so a stray BOM is stripped rather than corrupting the first
        # header cell into a confusing "header does not match". Any non-CSV
        # filesystem error (e.g. path is a directory) is wrapped so every bundle
        # defect surfaces as a ProfileContractError, per this module's contract.
        try:
            f = path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            raise ProfileContractError(f"{path}: cannot open - {exc}") from exc
        with f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ProfileContractError(
                    f"{path} is empty (no header row)."
                ) from None

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
            # Track measured coverage (min/max/n_unique) for numeric key axes.
            axis_values: dict[str, set[Any]] = {c: set() for c in schema.key_columns}
            rows = 0

            for lineno, row in enumerate(reader, start=2):
                if len(row) != len(schema.columns):
                    raise ProfileContractError(
                        f"{path}:{lineno}: expected {len(schema.columns)} "
                        f"columns, found {len(row)}: {row}"
                    )
                # Type sanity.
                for i in int_idx:
                    row[i] = _cast(path, lineno, schema.columns[i], row[i], _as_int)  # type: ignore[assignment]
                for i in float_idx:
                    row[i] = _cast(path, lineno, schema.columns[i], row[i], _as_float)  # type: ignore[assignment]
                for i in str_idx:
                    if not str(row[i]).strip():
                        raise ProfileContractError(
                            f"{path}:{lineno}: column {schema.columns[i]!r} "
                            f"must be a non-empty string."
                        )
                # Positive time.
                if time_idx is not None:
                    t = row[time_idx]
                    if not (isinstance(t, float) and t > 0.0):
                        raise ProfileContractError(
                            f"{path}:{lineno}: {schema.time_column} must be "
                            f"positive, got {t!r}."
                        )
                # Unique key.
                if key_idx:
                    key = tuple(row[i] for i in key_idx)
                    if key in seen_keys:
                        cols = ", ".join(schema.key_columns)
                        raise ProfileContractError(
                            f"{path}:{lineno}: duplicate key ({cols})={key}. "
                            f"Importers must emit unique keys."
                        )
                    seen_keys.add(key)
                    for c in schema.key_columns:
                        axis_values[c].add(row[schema.columns.index(c)])
                rows += 1

        if rows == 0:
            raise ProfileContractError(
                f"{path}: header present but no data rows."
            )

        coverage: dict[str, dict[str, Any]] = {}
        for c, vals in axis_values.items():
            numeric = [v for v in vals if isinstance(v, (int, float))]
            if numeric:
                coverage[c] = {
                    "min": min(numeric),
                    "max": max(numeric),
                    "n_unique": len(vals),
                }
        return FileReport(schema.filename, present=True, rows=rows, coverage=coverage)

    def _write_meta(
        self,
        dest_root: Path,
        report: BundleReport,
        provenance: ImportProvenance,
        extra_meta: dict[str, Any] | None,
    ) -> None:
        measured_coverage: dict[str, Any] = {}
        for tp in report.tp_degrees:
            tp_rep = report.tp_reports[tp]
            per_tp: dict[str, Any] = {}
            for fname, fr in tp_rep.files.items():
                if not fr.present:
                    continue
                entry: dict[str, Any] = {"rows": fr.rows}
                if fr.coverage:
                    entry["axes"] = fr.coverage
                per_tp[fname] = entry
            measured_coverage[f"tp{tp}"] = per_tp

        meta: dict[str, Any] = {
            "profiler_version": profiler_version,
            "source": provenance.source,
            "imported_by": "profiler.core.importer.CsvProfileImporter",
            "measured_by": provenance.measured_by,
            "serving_stack": provenance.serving_stack or "unknown",
            "runtime_version": provenance.runtime_version or "unknown",
            "backend": provenance.backend or "unknown",
            # 'gpu' is the device-string key the profiler's own meta.yaml uses;
            # keep it so downstream readers stay uniform across measured and
            # imported bundles.
            "gpu": provenance.device or "unknown",
            "hardware": report.hardware,
            "profiled_at": provenance.profiled_at or _utcnow_iso(),
            "model": report.model,
            "variant": report.variant,
            "tp_degrees": report.tp_degrees,
            "measurement_method": provenance.measurement_method or "unknown",
            "measurement_iterations": provenance.measurement_iterations,
            # Coverage is read off the CSV keys, never from declared factors
            # (CONTRACT.md / deviations D11: the CSVs are the source of truth).
            "measured_coverage": measured_coverage,
            "skew_profile": {"enabled": report.has_skew},
            "skew_fit": {"enabled": report.has_skew_fit},
        }
        if provenance.notes:
            meta["notes"] = provenance.notes
        if extra_meta:
            # extra_meta may only ADD caveats, never override validated
            # provenance keys - otherwise a caller could smuggle
            # source="measured"/empty measured_by past ImportProvenance.validate
            # (absolute rule 3). Collisions are a hard error.
            clash = set(meta) & set(extra_meta)
            if clash:
                raise ProfileContractError(
                    f"extra_meta may not override reserved meta keys "
                    f"{sorted(clash)}; these carry validated provenance "
                    f"(absolute rule 3). Use different keys for extra caveats."
                )
            meta.update(extra_meta)

        dest_root.mkdir(parents=True, exist_ok=True)
        out = dest_root / "meta.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(meta, f, Dumper=_CompactDumper, sort_keys=False)


# ---------------------------------------------------------------------------
# YAML dumper that keeps short scalar lists (e.g. tp_degrees) on one line.
# A local copy of profiler.core.writer._CompactDumper — importing writer would
# pull in the torch-dependent category/engine chain, and the importer must run
# in the torch-free simulator venv.
# ---------------------------------------------------------------------------

class _CompactDumper(yaml.SafeDumper):
    pass


def _represent_list(dumper: yaml.SafeDumper, data: list) -> Any:
    flow = all(
        isinstance(x, (int, float, str, bool, type(None))) for x in data
    )
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_CompactDumper.add_representer(list, _represent_list)
_CompactDumper.add_representer(tuple, _represent_list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tp_of(tp_dir: Path) -> int:
    m = _TP_DIR_RE.match(tp_dir.name)
    if not m:  # pragma: no cover - callers only pass tp<N> dirs
        raise ProfileContractError(f"{tp_dir} is not a tp<N> directory.")
    return int(m.group(1))


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
