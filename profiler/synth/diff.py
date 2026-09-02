"""Bundle-vs-bundle comparison and efficiency fitting (STEP 8).

``diff_bundles`` quantifies, in OUR environment rather than from literature
numbers, how far a synthetic bundle sits from a measured one - per file, per
layer and per kernel family. ``fit_efficiency`` derives the only legitimate
efficiency values a datasheet may carry: the median measured-vs-theoretical
ratio per family, written to a SEPARATE provenance-carrying YAML (never into
the accelerator profile itself - merging is a human decision).
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profiler import contract
from profiler.synth.backend import ProfileBackend

#: Layer -> kernel family, matching ShapeResolver's OpCost.family values.
#: attention.csv rows are family "attention", moe.csv rows family "moe".
LAYER_FAMILY = {
    "qkv_proj": "gemm",
    "o_proj": "gemm",
    "gate_up_proj": "gemm",
    "down_proj": "gemm",
    "lm_head": "gemm",
    "layernorm": "elementwise",
    "final_layernorm": "elementwise",
    "act_fn": "elementwise",
    "rotary_emb": "elementwise",
    "sampler": "elementwise",
    "qk_norm": "elementwise",
    "embedding": "gather",
}

#: The keyed measurement files a diff covers (skew tables are not keyed).
_KEYED_FILES = ("dense.csv", "per_sequence.csv", "attention.csv", "moe.csv")


class DiffError(ValueError):
    """The two bundles cannot be compared."""


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation with average ranks for ties (no scipy dependency)."""
    if len(x) != len(y) or len(x) < 2:
        raise DiffError("spearman needs two equal-length lists of >= 2 values")

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
    if den == 0:
        return 1.0  # constant series: identical ordering by convention
    return num / den


@dataclass(frozen=True)
class LayerDiff:
    """Error statistics over one group of joined (measured, synth) rows."""

    n: int
    #: mean(|synth - measured| / measured), as a FRACTION (0.5 = 50%).
    mape: float
    #: median(synth / measured).
    median_ratio: float
    #: 95th percentile of |synth - measured| in us.
    p95_abs_error: float
    #: rank correlation of synth vs measured (1.0 when n < 2 varies).
    spearman: float
    #: (key, measured_us, synth_us) of the worst relative error.
    max_over_key: tuple


@dataclass(frozen=True)
class FileDiff(LayerDiff):
    """Per-file statistics plus the keys that exist on only one side."""

    n_only_measured: int = 0
    n_only_synth: int = 0


@dataclass(frozen=True)
class DiffReport:
    """Keyed comparison of a measured and a synthetic bundle at one TP."""

    measured: str
    synth: str
    tp: int
    n_keys_compared: int
    n_keys_only_measured: int
    n_keys_only_synth: int
    per_file: dict[str, FileDiff]
    per_layer: dict[str, LayerDiff]
    per_family: dict[str, LayerDiff]

    @property
    def overall_mape(self) -> float:
        total = sum(d.n for d in self.per_file.values())
        if total == 0:
            return 0.0
        return sum(d.mape * d.n for d in self.per_file.values()) / total

    def to_dict(self) -> dict[str, Any]:
        def layer_dict(d: LayerDiff) -> dict[str, Any]:
            out = {
                "n": d.n, "mape": d.mape, "median_ratio": d.median_ratio,
                "p95_abs_error_us": d.p95_abs_error, "spearman": d.spearman,
                "max_over_key": list(d.max_over_key),
            }
            if isinstance(d, FileDiff):
                out["n_only_measured"] = d.n_only_measured
                out["n_only_synth"] = d.n_only_synth
            return out

        return {
            "measured": self.measured,
            "synth": self.synth,
            "tp": self.tp,
            "n_keys_compared": self.n_keys_compared,
            "n_keys_only_measured": self.n_keys_only_measured,
            "n_keys_only_synth": self.n_keys_only_synth,
            "overall_mape": self.overall_mape,
            "per_file": {k: layer_dict(v) for k, v in self.per_file.items()},
            "per_layer": {k: layer_dict(v) for k, v in self.per_layer.items()},
            "per_family": {k: layer_dict(v) for k, v in self.per_family.items()},
        }

    def render_table(self) -> str:
        lines = [
            f"measured : {self.measured}",
            f"synth    : {self.synth}   (tp{self.tp})",
            f"keys     : {self.n_keys_compared} compared, "
            f"{self.n_keys_only_measured} measured-only, "
            f"{self.n_keys_only_synth} synth-only",
            "",
            f"{'group':<22} {'n':>6} {'MAPE%':>8} {'med ratio':>10} "
            f"{'p95 |err| us':>13} {'spearman':>9}",
        ]
        for title, groups in (
            ("file", self.per_file), ("family", self.per_family),
            ("layer", self.per_layer),
        ):
            for name in sorted(groups):
                d = groups[name]
                lines.append(
                    f"{title}:{name:<16} {d.n:>6} {d.mape * 100:>8.1f} "
                    f"{d.median_ratio:>10.3f} {d.p95_abs_error:>13.2f} "
                    f"{d.spearman:>9.3f}"
                )
        lines.append("")
        lines.append(f"overall MAPE: {self.overall_mape * 100:.1f}%")
        return "\n".join(lines)


def _read_keyed(path: Path, schema: contract.CsvSchema) -> dict[tuple, float]:
    rows: dict[tuple, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(
                int(row[c]) if c in schema.int_columns else row[c]
                for c in schema.key_columns
            )
            rows[key] = float(row[schema.time_column or "time_us"])
    return rows


def _stats(joined: list[tuple[tuple, float, float]]) -> LayerDiff:
    """joined = [(key, measured_us, synth_us), ...] with measured > 0."""
    n = len(joined)
    if n == 0:
        return LayerDiff(0, 0.0, 0.0, 0.0, 1.0, ())
    rel_errors = [abs(s - m) / m for _, m, s in joined]
    ratios = [s / m for _, m, s in joined]
    abs_errors = sorted(abs(s - m) for _, m, s in joined)
    p95 = abs_errors[min(n - 1, int(0.95 * (n - 1)))]
    worst_i = max(range(n), key=rel_errors.__getitem__)
    rho = (
        spearman([s for *_, s in joined], [m for _, m, _ in joined])
        if n >= 2 else 1.0
    )
    key, m, s = joined[worst_i]
    return LayerDiff(
        n=n,
        mape=sum(rel_errors) / n,
        median_ratio=statistics.median(ratios),
        p95_abs_error=p95,
        spearman=rho,
        max_over_key=(key, m, s),
    )


def diff_bundles(measured: Path, synth: Path, tp: int) -> DiffReport:
    """Compare two variant-root bundles at one TP degree, key by key."""
    measured, synth = Path(measured), Path(synth)
    per_file: dict[str, FileDiff] = {}
    per_layer_rows: dict[str, list] = {}
    per_family_rows: dict[str, list] = {}
    n_compared = n_only_m = n_only_s = 0

    for filename in _KEYED_FILES:
        m_path = measured / f"tp{tp}" / filename
        s_path = synth / f"tp{tp}" / filename
        if not m_path.exists() and not s_path.exists():
            continue
        schema = contract.schema_for(filename)
        m_rows = _read_keyed(m_path, schema) if m_path.exists() else {}
        s_rows = _read_keyed(s_path, schema) if s_path.exists() else {}
        shared = sorted(set(m_rows) & set(s_rows))
        only_m = len(set(m_rows) - set(s_rows))
        only_s = len(set(s_rows) - set(m_rows))
        joined = [(k, m_rows[k], s_rows[k]) for k in shared]
        base = _stats(joined)
        per_file[filename] = FileDiff(
            **base.__dict__, n_only_measured=only_m, n_only_synth=only_s
        )
        n_compared += len(shared)
        n_only_m += only_m
        n_only_s += only_s

        if filename in ("dense.csv", "per_sequence.csv"):
            for k, m, s in joined:
                layer = str(k[0])
                per_layer_rows.setdefault(layer, []).append((k, m, s))
                family = LAYER_FAMILY.get(layer, "unknown")
                per_family_rows.setdefault(family, []).append((k, m, s))
        elif filename == "attention.csv":
            per_family_rows.setdefault("attention", []).extend(joined)
        elif filename == "moe.csv":
            per_family_rows.setdefault("moe", []).extend(joined)

    if n_compared == 0:
        raise DiffError(
            f"no shared keys between {measured} and {synth} at tp{tp} - "
            f"regenerate the synth bundle with --mirror-keys"
        )

    return DiffReport(
        measured=str(measured),
        synth=str(synth),
        tp=tp,
        n_keys_compared=n_compared,
        n_keys_only_measured=n_only_m,
        n_keys_only_synth=n_only_s,
        per_file=per_file,
        per_layer={k: _stats(v) for k, v in sorted(per_layer_rows.items())},
        per_family={k: _stats(v) for k, v in sorted(per_family_rows.items())},
    )


# ---------------------------------------------------------------------------
# Efficiency fitting (the only legitimate way to fill the empty efficiencies)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EfficiencyFit:
    """Per-family efficiencies with the provenance the output YAML records."""

    family_efficiency: dict[str, float]
    n_per_family: dict[str, int]
    derived_from: dict[str, str]
    #: Families whose fitted value exceeded 1.0 (theoretical bound broken).
    bound_violations: dict[str, float] = field(default_factory=dict)


def fit_efficiency(
    measured_root: Path,
    tp: int,
    lower_bound_backend: ProfileBackend,
    *,
    derived_from: dict[str, str],
) -> EfficiencyFit:
    """Fit per-family efficiencies against a measured bundle.

    efficiency(family) = median(theoretical_lower_bound / measured) over the
    family's keys, computed with an eff=1.0 backend. A value > 1.0 means the
    'lower bound' beat the measurement - the bound is broken - and is
    reported as a violation, NOT clamped (clamping would hide a shape or
    unit error behind a plausible number).
    """
    ratios: dict[str, list[float]] = {}

    def _collect(filename: str, key_to_est) -> None:
        path = Path(measured_root) / f"tp{tp}" / filename
        if not path.exists():
            return
        schema = contract.schema_for(filename)
        for key, measured_us in _read_keyed(path, schema).items():
            family, est = key_to_est(key)
            ratios.setdefault(family, []).append(est / measured_us)

    be = lower_bound_backend
    _collect("dense.csv", lambda k: (
        LAYER_FAMILY.get(str(k[0]), "unknown"), be.dense_us(str(k[0]), int(k[1]))
    ))
    _collect("per_sequence.csv", lambda k: (
        LAYER_FAMILY.get(str(k[0]), "unknown"),
        be.per_sequence_us(str(k[0]), int(k[1])),
    ))
    _collect("attention.csv", lambda k: ("attention", be.attention_us(*k)))
    _collect("moe.csv", lambda k: ("moe", be.expert_us(int(k[0]), int(k[1]))))

    if not ratios:
        raise DiffError(f"no measured rows found under {measured_root}/tp{tp}")

    fitted = {f: statistics.median(v) for f, v in sorted(ratios.items())}
    violations = {f: e for f, e in fitted.items() if e > 1.0}
    return EfficiencyFit(
        family_efficiency=fitted,
        n_per_family={f: len(v) for f, v in sorted(ratios.items())},
        derived_from=dict(derived_from),
        bound_violations=violations,
    )
