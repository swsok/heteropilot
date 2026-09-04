"""Tier 1 calibration: anchors -> per-kernel-family scaling (STEP 9).

    t_tier1 = t_tier0 * scale(family, feature)

The simplest working structure comes first: one scalar per family, plus an
optional per-family piecewise table over the feature axis (feature = the
unscaled Tier 0 estimate in us, the convention shared with RooflineModel /
AttentionCostModel). KernelSight-LM describes the same architecture as a
"measured efficiency grid + interpolation" with a fallback at extrapolation
boundaries; here the fallback is the family scalar, and with no anchors at
all the table is the identity (absolute rule A2: no data means identity).

Anchor input format is a SUBSET of the measured-bundle CSVs - a dense.csv
with 100 rows is a valid anchor file - validated by profiler.contract. No
new format exists.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from profiler import contract
from profiler.synth.backend import ProfileBackend
from profiler.synth.diff import LAYER_FAMILY

#: Anchor-file names the fit accepts (keyed measurement files only).
ANCHOR_FILES = ("dense.csv", "per_sequence.csv", "attention.csv", "moe.csv")


class CalibrationError(ValueError):
    """The anchors cannot produce a scaling table."""


@dataclass(frozen=True)
class AnchorRecord:
    """One measured anchor point and what Tier 0 said about it."""

    file: str
    key: tuple
    family: str
    measured_us: float
    tier0_us: float

    @property
    def ratio(self) -> float:
        return self.measured_us / self.tier0_us


@dataclass(frozen=True)
class ScalingTable:
    """Per-kernel-family multipliers (Tier 1's product).

    ``scalars`` maps family -> multiplier; ``piecewise`` optionally maps
    family -> [(feature_lo, feature_hi, scale)] rows over the feature axis.
    Lookup order: piecewise bin containing the feature, else the family
    scalar, else 1.0 (identity - A2).
    """

    scalars: dict[str, float] = field(default_factory=dict)
    piecewise: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
    anchors: list[AnchorRecord] = field(default_factory=list)
    derived_from: dict[str, str] = field(default_factory=dict)

    def scale(self, family: str, feature: float) -> float:
        for lo, hi, s in self.piecewise.get(family, ()):
            if lo <= feature < hi:
                return s
        return self.scalars.get(family, 1.0)

    # ------------------------------------------------------------------
    # Serialization (yaml round trip for the emit CLI's --scaling)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "scalars": dict(self.scalars),
            "piecewise": {
                f: [list(row) for row in rows] for f, rows in self.piecewise.items()
            },
            "n_anchors": len(self.anchors),
            "anchors_per_family": self.anchors_per_family(),
            "derived_from": dict(self.derived_from),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> ScalingTable:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise CalibrationError(f"{path}: not a scaling-table mapping")
        return cls(
            scalars={str(k): float(v) for k, v in (raw.get("scalars") or {}).items()},
            piecewise={
                str(f): [
                    (float(row[0]), float(row[1]), float(row[2])) for row in rows
                ]
                for f, rows in (raw.get("piecewise") or {}).items()
            },
            derived_from={
                str(k): str(v) for k, v in (raw.get("derived_from") or {}).items()
            },
        )

    def anchors_per_family(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.anchors:
            out[a.family] = out.get(a.family, 0) + 1
        return dict(sorted(out.items()))


def _family_of(filename: str, key: tuple) -> str:
    if filename == "attention.csv":
        return "attention"
    if filename == "moe.csv":
        return "moe"
    return LAYER_FAMILY.get(str(key[0]), "unknown")


def read_anchors(anchor_root: Path, tp: int, backend: ProfileBackend) -> list[AnchorRecord]:
    """Read + contract-validate anchor CSVs and price them with Tier 0."""
    anchor_root = Path(anchor_root)
    tp_dir = anchor_root / f"tp{tp}"
    if not tp_dir.is_dir():
        # Also accept a bare directory of CSVs (a tp dir handed directly).
        tp_dir = anchor_root
    records: list[AnchorRecord] = []
    estimators = {
        "dense.csv": lambda k: backend.dense_us(str(k[0]), int(k[1])),
        "per_sequence.csv": lambda k: backend.per_sequence_us(str(k[0]), int(k[1])),
        "attention.csv": lambda k: backend.attention_us(*(int(x) for x in k)),
        "moe.csv": lambda k: backend.expert_us(int(k[0]), int(k[1])),
    }
    for filename in ANCHOR_FILES:
        path = tp_dir / filename
        if not path.exists():
            continue
        schema = contract.schema_for(filename)
        contract.validate_csv(path, schema)  # header/type/key/time discipline
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = tuple(
                    int(row[c]) if c in schema.int_columns else row[c]
                    for c in schema.key_columns
                )
                measured = float(row["time_us"])
                records.append(AnchorRecord(
                    file=filename,
                    key=key,
                    family=_family_of(filename, key),
                    measured_us=measured,
                    tier0_us=estimators[filename](key),
                ))
    return records


def fit_from_anchors(
    anchors: Path,
    backend: ProfileBackend,
    *,
    tp: int = 1,
    bins: int = 8,
    min_bin_anchors: int = 4,
    min_family_anchors: int = 8,
    derived_from: dict[str, str] | None = None,
) -> ScalingTable:
    """Fit a ScalingTable from anchor measurements.

    Per family: the scalar is median(measured / tier0); when a family has
    enough anchors, log-spaced feature bins get their own median (piecewise),
    with the scalar as the out-of-bin fallback. No anchors at all yields the
    identity table (A2). A family with fewer than ``min_family_anchors``
    anchors also stays at the identity: the hold-out experiment showed a
    scalar fitted from 1-4 anchors can land on a launch-floor key and scale
    the whole family by 36x (docs/tier0_calibration.md, STEP 9) - too little
    data is treated as no data, never as a confident multiplier.
    """
    records = read_anchors(anchors, tp, backend)
    by_family: dict[str, list[AnchorRecord]] = {}
    for r in records:
        by_family.setdefault(r.family, []).append(r)

    scalars: dict[str, float] = {}
    piecewise: dict[str, list[tuple[float, float, float]]] = {}
    for family, recs in sorted(by_family.items()):
        if len(recs) < min_family_anchors:
            continue  # identity via ScalingTable.scale's 1.0 default (A2)
        ratios = [r.ratio for r in recs]
        scalars[family] = statistics.median(ratios)
        # Piecewise refinement over log(feature): only where enough anchors
        # land per bin; sparse bins fall back to the scalar.
        feats = [r.tier0_us for r in recs]
        lo, hi = min(feats), max(feats)
        if len(recs) >= bins * min_bin_anchors and hi > lo > 0:
            edges = [
                lo * (hi / lo) ** (i / bins) for i in range(bins + 1)
            ]
            edges[-1] = math.nextafter(hi, math.inf)  # include the max point
            rows: list[tuple[float, float, float]] = []
            for i in range(bins):
                in_bin = [
                    r.ratio for r in recs if edges[i] <= r.tier0_us < edges[i + 1]
                ]
                if len(in_bin) >= min_bin_anchors:
                    rows.append((edges[i], edges[i + 1], statistics.median(in_bin)))
            if rows:
                piecewise[family] = rows

    return ScalingTable(
        scalars=scalars,
        piecewise=piecewise,
        anchors=records,
        derived_from=dict(derived_from or {}),
    )


# ---------------------------------------------------------------------------
# Anchor planning (what to measure, before any measurement exists)
# ---------------------------------------------------------------------------

def pick_anchors(
    keys_by_file: dict[str, list[tuple]],
    budget: int,
    *,
    attention_share: float = 0.7,
) -> dict[str, list[tuple]]:
    """Choose which grid keys to measure, spread evenly over each sweep.

    ``attention_share`` of the budget goes to attention keys (STEP 6
    background: attention is the family that does not transfer across
    devices), the rest is split evenly over the other present files. Keys
    are taken at evenly spaced positions of each file's enumeration order,
    which is monotone in the swept axes - so the picks cover the full range
    (log-scaled axes included) instead of clustering at one end.
    """
    if budget < 1:
        raise CalibrationError(f"budget must be >= 1, got {budget}")
    if not 0 <= attention_share <= 1:
        raise CalibrationError("attention_share must be within [0, 1]")

    plan: dict[str, list[tuple]] = {}
    others = [f for f in keys_by_file if f != "attention.csv" and keys_by_file[f]]
    n_attn = round(budget * attention_share) if "attention.csv" in keys_by_file else 0
    n_other_total = budget - n_attn

    def spread(keys: list[tuple], n: int) -> list[tuple]:
        n = min(n, len(keys))
        if n <= 0:
            return []
        if n == 1:
            return [keys[len(keys) // 2]]
        step = (len(keys) - 1) / (n - 1)
        return [keys[round(i * step)] for i in range(n)]

    if n_attn:
        plan["attention.csv"] = spread(keys_by_file["attention.csv"], n_attn)
    for i, filename in enumerate(others):
        share = n_other_total // len(others) + (
            1 if i < n_other_total % len(others) else 0
        )
        picked = spread(keys_by_file[filename], share)
        if picked:
            plan[filename] = picked
    return plan
