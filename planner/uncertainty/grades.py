"""Loaders for ``profiles/uncertainty/grades.yaml`` and ``costs.yaml`` (§2.2, §2.6).

These two files are the only place an error range or a measurement cost may
come from. Absolute rule A1 is enforced mechanically here rather than trusted:

* a rule that carries a number must carry a non-empty ``source``; the loader
  rejects the table otherwise, so an unattributed range cannot reach a plan;
* a (kind, grade) combination the table does not mention is ``unbounded`` -
  never a guessed default (§2.2's closing rule).

The *numbers* live in the YAML; only the *formulas* live in code.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Repo-root-relative defaults (planner/uncertainty/grades.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADES_PATH = _REPO_ROOT / "profiles" / "uncertainty" / "grades.yaml"
DEFAULT_COSTS_PATH = _REPO_ROOT / "profiles" / "uncertainty" / "costs.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RangeRule(str, enum.Enum):
    """How a grade turns a nominal value into a range.

    Formulas, not numbers: each rule reads its magnitude from the YAML row.
    """

    #: No sourced width exists. The item is recorded with an open range and is
    #: reported as "cannot be decided before measuring" (§2.5).
    UNBOUNDED = "unbounded"
    #: ``[nominal * (1 - value), nominal * (1 + value)]``; ``value`` is a fraction.
    SYMMETRIC_FRACTION = "symmetric_fraction"
    #: ``[nominal * r_min, nominal]`` - a spec value is an upper bound and the
    #: worst measured/spec ratio in the repo is how far below it reality sat.
    RATIO_FLOOR = "ratio_floor"
    #: Width comes from the calibration store, not from this table: the bucket's
    #: ``p95_abs_error - |mean_error|`` added either side of the nominal error
    #: (§2.2, sim_error/measured). The YAML row only names the source file.
    CALIBRATION_DERIVED = "calibration_derived"


class GradeRule(_Strict):
    """One row of ``grades.yaml``: what a (kind, grade) pair's range is."""

    kind: str
    grade: str
    rule: RangeRule
    #: Half-width, as a fraction. Required by SYMMETRIC_FRACTION only.
    value: float | None = Field(default=None, gt=0)
    #: Worst measured/spec ratio. Required by RATIO_FLOOR only.
    r_min: float | None = Field(default=None, gt=0, le=1)
    unit: str = "fraction"
    source: str = ""
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.kind}/{self.grade}"

    @model_validator(mode="after")
    def _numbers_need_a_source(self) -> GradeRule:
        # Absolute rule A1, enforced rather than trusted. A row may be empty
        # (unbounded) or sourced, never numeric-and-anonymous.
        has_number = self.value is not None or self.r_min is not None
        if has_number and not self.source.strip():
            raise ValueError(
                f"grades.yaml row {self.id}: carries a number but no source - "
                f"an unattributed error range is forbidden (absolute rule A1)"
            )
        if self.rule is RangeRule.SYMMETRIC_FRACTION and self.value is None:
            raise ValueError(f"grades.yaml row {self.id}: rule symmetric_fraction needs `value`")
        if self.rule is RangeRule.RATIO_FLOOR and self.r_min is None:
            raise ValueError(f"grades.yaml row {self.id}: rule ratio_floor needs `r_min`")
        if self.rule is RangeRule.UNBOUNDED and has_number:
            raise ValueError(
                f"grades.yaml row {self.id}: rule unbounded must carry no number "
                f"(found value/r_min) - say what the number is or say unbounded"
            )
        return self


class GradesTable(_Strict):
    """The whole of ``grades.yaml``, indexed by (kind, grade)."""

    rules: list[GradeRule] = Field(default_factory=list)
    #: sha256 of the file this was read from; empty for a synthesised table.
    digest: str = ""
    path: str = ""

    @model_validator(mode="after")
    def _unique_rows(self) -> GradesTable:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"grades.yaml: duplicate row {rule.id}")
            seen.add(rule.id)
        return self

    def rule_for(self, kind: str, grade: str) -> GradeRule:
        """The row for a (kind, grade), or an unbounded stand-in.

        §2.2: a combination the table does not mention is unbounded. That is a
        lookup miss, not an error - the point of the registry is to surface
        exactly these.
        """
        for rule in self.rules:
            if rule.kind == kind and rule.grade == grade:
                return rule
        return GradeRule(
            kind=kind,
            grade=grade,
            rule=RangeRule.UNBOUNDED,
            source="",
            note="no row in grades.yaml for this (kind, grade); unbounded by §2.2",
        )


class CostRule(_Strict):
    """One row of ``costs.yaml``: what measuring this kind costs."""

    kind: str
    method: str
    #: Wall-clock hours. None when no timestamped evidence exists (§2.6): the
    #: honest value, which sorts such items last rather than inventing a cost.
    hours: float | None = Field(default=None, gt=0)
    #: True when the measurement owns the device/server for its duration, so a
    #: budget can count occupancy separately from elapsed time.
    exclusive: bool = False
    source: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _numbers_need_a_source(self) -> CostRule:
        if self.hours is not None and not self.source.strip():
            raise ValueError(
                f"costs.yaml row {self.kind}: carries `hours` but no source - "
                f"an unattributed cost is forbidden (absolute rule A1)"
            )
        return self


class CostsTable(_Strict):
    """The whole of ``costs.yaml``, indexed by kind."""

    costs: list[CostRule] = Field(default_factory=list)
    digest: str = ""
    path: str = ""

    @model_validator(mode="after")
    def _unique_rows(self) -> CostsTable:
        seen: set[str] = set()
        for cost in self.costs:
            if cost.kind in seen:
                raise ValueError(f"costs.yaml: duplicate row {cost.kind}")
            seen.add(cost.kind)
        return self

    def cost_for(self, kind: str) -> CostRule | None:
        """The row for a kind, or None when the table does not mention it."""
        for cost in self.costs:
            if cost.kind == kind:
                return cost
        return None


def _read_mapping(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{what}: expected a mapping at the top level, got {type(raw).__name__}")
    return raw


def load_grades(path: str | Path | None = None) -> GradesTable:
    """Load ``grades.yaml``. Raises on a malformed or unattributed table."""
    from planner.util import provenance as prov

    path = Path(path) if path is not None else DEFAULT_GRADES_PATH
    raw = _read_mapping(path, "grades.yaml")
    table = GradesTable.model_validate(
        {"rules": raw.get("rules", []), "digest": prov.hash_file(path) or "", "path": str(path)}
    )
    return table


def load_costs(path: str | Path | None = None) -> CostsTable:
    """Load ``costs.yaml``. Raises on a malformed or unattributed table."""
    from planner.util import provenance as prov

    path = Path(path) if path is not None else DEFAULT_COSTS_PATH
    raw = _read_mapping(path, "costs.yaml")
    return CostsTable.model_validate(
        {"costs": raw.get("costs", []), "digest": prov.hash_file(path) or "", "path": str(path)}
    )
