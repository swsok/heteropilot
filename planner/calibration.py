"""Measured calibration tables, and the refusal that guards them.

`WORK_ORDER_rps_aware.md` rev 2 STEP 2.2/2.3. `topology_mode: slab3d` encodes a
`tp` allreduce as `tp/2 x 2`, which shortens the ring's latency term from
`2(tp-1)` hops to `2(tp/2-1) + 2`. A profile whose `link_latency` was fitted
against a flat ring therefore sees decode as faster than it is -- 24.4 % at tp8,
13.4 % at tp4, measured. Multiplying dim 1's latency by a fitted factor removes
it.

The table is a lookup, never a formula. Both measured factors equal `tp/2` and
hop counting derives that, but `docs/slab3d_calibration.md` records why it is not
shipped as a rule: the factor varies with the split, and the work order requires a
table in that case. So `factor_for` returns None outside the measured domain and
the caller must refuse the candidate rather than guess -- A6's
`extrapolation: refuse`, applied to a calibration rather than an envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TABLE = (
    Path(__file__).resolve().parents[1] / "profiles/calibration/slab3d_latency.yaml"
)


class CalibrationError(ValueError):
    """The table itself is unusable -- missing, malformed, or without validity."""


@dataclass(frozen=True)
class Slab3dCalibration:
    """dim-1 `link_latency` factors, indexed by (split, link_bw)."""

    #: {(split, link_bw_gbps): factor}
    factors: dict[tuple[str, float], float]
    valid_splits: frozenset[str]
    valid_link_bw: tuple[float, ...]
    extrapolation: str
    base_link_latency_ns: float

    @staticmethod
    def split_label(tp: int) -> str:
        """The table's key for a `tp`-wide group split as `[tp/2, 2]`."""
        return f"[{tp // 2},2]"

    def factor_for(self, tp: int, link_bw_gbps: float) -> float | None:
        """The dim-1 multiplier, or None when this point was never measured.

        None means refuse, not "use 1.0". Falling back to 1.0 would silently
        reinstate the 24.4 % error the table exists to remove, and would do it
        precisely where nobody has checked the number.
        """
        if self.extrapolation != "refuse":
            raise CalibrationError(
                f"unsupported extrapolation policy {self.extrapolation!r}; only "
                f"'refuse' is implemented, and widening error bars silently is "
                f"exactly what A6 forbids"
            )
        return self.factors.get((self.split_label(tp), float(link_bw_gbps)))

    def domain_note(self, tp: int, link_bw_gbps: float) -> str:
        """Why a candidate was refused, in terms a reader can act on."""
        return (
            f"slab3d split {self.split_label(tp)} at link_bw {link_bw_gbps} Gbps is "
            f"outside the measured calibration domain "
            f"(splits {sorted(self.valid_splits)}, link_bw {list(self.valid_link_bw)}); "
            f"measure it with experiments/scripts/slab3d_calibrate.py rather than "
            f"assuming a factor -- uncorrected, slab3d under-predicts TPOT by "
            f"13-24 % (docs/slab3d_calibration.md)"
        )


def _require(raw: dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise CalibrationError(f"{path}: missing required key '{key}'")
    return raw[key]


def load_slab3d_calibration(path: Path | str = DEFAULT_TABLE) -> Slab3dCalibration:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise CalibrationError(
            f"{path}: no slab3d calibration table; a slab3d candidate cannot be "
            f"compiled without one (STEP 2.2)"
        ) from exc
    if not isinstance(raw, dict):
        raise CalibrationError(f"{path}: expected a mapping at the top level")

    validity = _require(raw, "validity", path)
    points = _require(raw, "points", path)
    factors: dict[tuple[str, float], float] = {}
    for p in points:
        for key in ("split", "link_bw", "factor"):
            if key not in p:
                raise CalibrationError(f"{path}: a point is missing '{key}': {p}")
        factors[(str(p["split"]), float(p["link_bw"]))] = float(p["factor"])
    if not factors:
        raise CalibrationError(f"{path}: the table has no points")

    return Slab3dCalibration(
        factors=factors,
        valid_splits=frozenset(str(s) for s in validity.get("splits", [])),
        valid_link_bw=tuple(float(b) for b in validity.get("link_bw_gbps", [])),
        extrapolation=str(validity.get("extrapolation", "refuse")),
        base_link_latency_ns=float(raw.get("base_link_latency_ns", 0.0)),
    )


@lru_cache(maxsize=4)
def slab3d_calibration(path: Path | str = DEFAULT_TABLE) -> Slab3dCalibration:
    """Cached loader. The table is a committed artifact and does not change
    within a run; re-reading it per candidate would be wasted I/O in a sweep."""
    return load_slab3d_calibration(path)
