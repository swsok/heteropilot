"""BundleEmitter: write a complete Tier 0/1 bundle and self-validate (STEP 7).

Key enumeration reuses ``profiler.core.categories``' grid generators - the
only way to guarantee key compatibility with measured bundles (work order
§2.1). ``profiler.core.writer`` is NOT used: it needs the live GPU stack at
meta-writing time; instead CSVs are written directly and the result is
checked against ``profiler.contract`` before the bundle is allowed to exist.
"""

from __future__ import annotations

import csv
import datetime
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from profiler import contract
from profiler.core.categories import (
    AttentionCategory,
    DenseCategory,
    ExpertCategory,
    SequenceCategory,
    categories_for,
)
from profiler.core.config import Architecture, ProfileArgs
from profiler.synth.backend import ProfileBackend
from profiler.synth.dims import ModelDims

#: Version stamp for meta.yaml's generator_version.
GENERATOR_VERSION = "synth-0.1.0"
COST_MODEL = "roofline-v1"

#: A cache-token budget high enough that it never prunes the grid; recorded
#: in meta.yaml's grid block. A synthetic bundle has no live engine whose
#: real cache size could apply.
UNBOUNDED = 2**31


class EmitError(ValueError):
    """The emitter refuses to write this bundle."""


@dataclass(frozen=True)
class GridParams:
    """Grid parameters, mirrored into meta.yaml (work order STEP 7)."""

    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256
    attention_max_kv: int = 16384
    attention_chunk_factor: float = 2.0
    attention_kv_factor: float = 2.0
    max_model_len: int = UNBOUNDED
    num_cache_tokens: int = UNBOUNDED


@dataclass
class _GridLimits:
    """Duck-typed stand-in for profiler.core.engine.RuntimeLimits.

    RuntimeLimits lives in engine.py, which imports torch at module level;
    the grid generators only read these attributes (their RuntimeLimits
    annotation is PEP 563 text), so a plain dataclass with the same fields
    keeps profiler.synth importable in the GPU-free venv.
    """

    max_num_batched_tokens: int
    max_num_seqs: int
    num_cache_tokens: int
    max_model_len: int
    num_experts: int | None = None
    top_k: int | None = None


@dataclass
class BundleReport:
    out_root: Path
    hardware: str
    model: str
    variant: str
    tp_degrees: list[int]
    rows: dict[str, dict[str, int]] = field(default_factory=dict)  # tp -> file -> rows


def _format_time_us(v: float) -> str:
    """6 sig figs, no scientific notation for typical values.

    Replicated from profiler/core/writer.py::_format_time_us (writer is
    unusable without the GPU stack, so the two-line rule is copied with this
    provenance note instead of imported).
    """
    return f"{v:.6g}"


def _mirror_keys(mirror_root: Path, tp: int, filename: str) -> list[tuple]:
    """Exact key tuples from a measured bundle's CSV (STEP 8 precondition)."""
    path = mirror_root / f"tp{tp}" / filename
    if not path.exists():
        return []
    schema = contract.schema_for(filename)
    keys: list[tuple] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(
                int(row[c]) if c in schema.int_columns else row[c]
                for c in schema.key_columns
            )
            keys.append(key)
    return keys


class BundleEmitter:
    """Write one synthetic bundle under out_root/<label>/<model>/<variant>/."""

    def __init__(
        self,
        *,
        dims: ModelDims,
        arch: Architecture,
        backend_for_tp: dict[int, ProfileBackend] | dict,
        hardware_label: str,
        variant: str,
        out_root: Path,
        grid: GridParams | None = None,
        mirror_root: Path | None = None,
        datasheet_source: str = "",
        efficiency: dict[str, Any] | None = None,
        calibration_anchors: dict[str, Any] | None = None,
        force: bool = False,
        generated_at: str | None = None,
    ) -> None:
        if not (hardware_label.endswith("-t0") or hardware_label.endswith("-t1")):
            raise EmitError(
                f"hardware label {hardware_label!r} must end in -t0 (analytical) "
                f"or -t1 (calibrated): synthetic bundles may never share a label "
                f"with measured ones (absolute rule A3)"
            )
        if not backend_for_tp:
            raise EmitError("at least one TP degree is required")
        tiers = {b.tier for b in backend_for_tp.values()}
        if len(tiers) != 1:
            raise EmitError(f"mixed backend tiers in one bundle: {sorted(tiers)}")
        self.tier = tiers.pop()
        expected_suffix = "-t0" if self.tier == "analytical" else "-t1"
        if not hardware_label.endswith(expected_suffix):
            raise EmitError(
                f"hardware label {hardware_label!r} does not match tier "
                f"{self.tier!r} (expected suffix {expected_suffix!r})"
            )
        self.dims = dims
        self.arch = arch
        self.backends = dict(backend_for_tp)
        self.hardware_label = hardware_label
        self.variant = variant
        self.out_root = Path(out_root)
        self.grid = grid or GridParams()
        self.mirror_root = Path(mirror_root) if mirror_root is not None else None
        self.datasheet_source = datasheet_source
        self.efficiency = efficiency or {}
        self.calibration_anchors = calibration_anchors
        self.force = force
        self.generated_at = generated_at

    # ------------------------------------------------------------------

    def emit(self) -> BundleReport:
        variant = self.variant
        dest = self.out_root / self.hardware_label / self.dims.model / variant
        if dest.exists():
            if not self.force:
                raise EmitError(
                    f"destination already exists: {dest} (pass force=True / "
                    f"--force to replace it)"
                )
            shutil.rmtree(dest)

        tp_degrees = sorted(self.backends)
        report = BundleReport(
            out_root=dest,
            hardware=self.hardware_label,
            model=self.dims.model,
            variant=variant,
            tp_degrees=tp_degrees,
        )
        try:
            for tp in tp_degrees:
                report.rows[f"tp{tp}"] = self._emit_tp(dest, tp)
            self._write_meta(dest, report)
            contract.validate_bundle(dest, tp_degrees, is_moe=self.dims.is_moe)
        except Exception:
            # Never leave a half-written bundle behind - including the
            # hardware-label directories a failed first emit created.
            if dest.exists():
                shutil.rmtree(dest)
            parent = dest.parent
            while parent != self.out_root and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
            raise
        return report

    # ------------------------------------------------------------------

    def _keys_for(self, tp: int, filename: str) -> list[tuple]:
        if self.mirror_root is not None:
            keys = _mirror_keys(self.mirror_root, tp, filename)
            if not keys and filename in ("dense.csv", "per_sequence.csv", "attention.csv"):
                raise EmitError(
                    f"--mirror-keys: {self.mirror_root}/tp{tp}/{filename} is "
                    f"missing or empty; cannot mirror a bundle that lacks it"
                )
            return keys
        return self._grid_keys(tp, filename)

    def _grid_keys(self, tp: int, filename: str) -> list[tuple]:
        g = self.grid
        args = ProfileArgs(
            architecture=self.dims.model_type,
            model=self.dims.model,
            hardware=self.hardware_label,
            tp_degrees=[tp],
            attention_max_kv=g.attention_max_kv,
            attention_chunk_factor=g.attention_chunk_factor,
            attention_kv_factor=g.attention_kv_factor,
        )
        limits = _GridLimits(
            max_num_batched_tokens=g.max_num_batched_tokens,
            max_num_seqs=g.max_num_seqs,
            num_cache_tokens=g.num_cache_tokens,
            max_model_len=g.max_model_len,
            num_experts=self.dims.num_experts,
            top_k=self.dims.experts_per_token,
        )
        by_file = {
            "dense.csv": DenseCategory,
            "per_sequence.csv": SequenceCategory,
            "attention.csv": AttentionCategory,
            "moe.csv": ExpertCategory,
        }
        wanted = by_file[filename]
        keys: list[tuple] = []
        for cat in categories_for(self.arch, tp):
            if not isinstance(cat, wanted):
                continue
            for shot in cat.compose_shots(self.arch, args, limits, tp):
                keys.append(cat.shot_key(shot))
        if filename == "dense.csv":
            layers = list(self.arch.catalog.dense)
            return [(layer, k[0]) for k in keys for layer in layers]
        if filename == "per_sequence.csv":
            layers = list(self.arch.catalog.per_sequence)
            return [(layer, k[0]) for k in keys for layer in layers]
        return keys

    def _emit_tp(self, dest: Path, tp: int) -> dict[str, int]:
        backend = self.backends[tp]
        tp_dir = dest / f"tp{tp}"
        tp_dir.mkdir(parents=True, exist_ok=True)
        rows_written: dict[str, int] = {}

        emitters = {
            "dense.csv": lambda key: backend.dense_us(key[0], key[1]),
            "per_sequence.csv": lambda key: backend.per_sequence_us(key[0], key[1]),
            "attention.csv": lambda key: backend.attention_us(*key),
            "moe.csv": lambda key: backend.expert_us(key[0], key[1]),
        }
        for schema in contract.SCHEMAS:
            if schema.filename not in emitters:
                continue  # skew*.csv: intentionally omitted (no measurement)
            if schema.moe_only and (not self.dims.is_moe or tp != 1):
                continue
            keys = self._keys_for(tp, schema.filename)
            if not keys:
                if schema.moe_only:
                    continue
                raise EmitError(
                    f"no keys enumerated for {schema.filename} at tp{tp} - "
                    f"the grid parameters exclude everything"
                )
            time_fn = emitters[schema.filename]
            out = tp_dir / schema.filename
            with out.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(schema.columns)
                for key in keys:
                    t_us = time_fn(key)
                    if not (t_us > 0):
                        raise EmitError(
                            f"{schema.filename} key {key}: non-positive time "
                            f"{t_us!r} - contract violation, refusing to write"
                        )
                    writer.writerow([*key, _format_time_us(t_us)])
            rows_written[schema.filename] = len(keys)
        return rows_written

    def _write_meta(self, dest: Path, report: BundleReport) -> None:
        generated_at = self.generated_at or datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds")
        meta: dict[str, Any] = {
            "tier": self.tier,
            "source": self.tier,
            "cost_model": COST_MODEL,
            "generator_version": GENERATOR_VERSION,
            "generated_at": generated_at,
            "hardware": report.hardware,
            "model": report.model,
            "variant": report.variant,
            "tp_degrees": report.tp_degrees,
            "datasheet_source": self.datasheet_source,
            "efficiency": self.efficiency,
            # A synthetic bundle has no serving-stack provenance; recording
            # null with a reason beats inventing one (absolute rule A2).
            "vllm_version": None,
            "cuda_version": None,
            "gpu": None,
            "null_reason": (
                "synthetic bundle: no vLLM/CUDA run produced these numbers "
                "(tier " + self.tier + ")"
            ),
            "skew": (
                "omitted (analytical bundle has no heterogeneous-decode "
                "measurement; the simulator falls back to the pooled "
                "constant alpha)"
            ),
            "grid": (
                {"mirrored_from": str(self.mirror_root)}
                if self.mirror_root is not None
                else {
                    "max_num_batched_tokens": self.grid.max_num_batched_tokens,
                    "max_num_seqs": self.grid.max_num_seqs,
                    "attention_max_kv": self.grid.attention_max_kv,
                    "attention_chunk_factor": self.grid.attention_chunk_factor,
                    "attention_kv_factor": self.grid.attention_kv_factor,
                    "max_model_len": self.grid.max_model_len,
                    "num_cache_tokens": self.grid.num_cache_tokens,
                }
            ),
            "rows": report.rows,
        }
        if self.calibration_anchors is not None:
            meta["calibration_anchors"] = self.calibration_anchors
        with (dest / "meta.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
