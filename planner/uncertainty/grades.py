"""Loaders for ``profiles/uncertainty/grades.yaml`` and ``costs.yaml`` (§2.2, §2.6).

These two files are the only place an error range or a measurement cost may
come from. Absolute rule A1 is enforced mechanically here rather than trusted:

* a rule that carries a number must carry a non-empty ``source``; the loader
  rejects the table otherwise, so an unattributed range cannot reach a plan;
* a (kind, grade) combination the table does not mention has no ROW-level
  width; what it falls back to is the grade's *default* range, and a grade with
  no default is ``unbounded`` (§2.2's closing rule, as amended by the
  domain-scoping work order S2 / deviations D111).

The *numbers* live in the YAML; only the *formulas* live in code.

**Defaults are a second layer, never a merged one.** A row in ``defaults:``
says how wide an input of a given grade is assumed to be when nothing measured
its own width. Every default carries its own ``source`` exactly as a rule does,
and a range built from one is tagged ``range_source: default`` all the way to
the rendered measurement plan, so a policy assumption is never read as this
input's own measurement (D111).
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


class DefaultRule(str, enum.Enum):
    """How a *default* turns a nominal value into a range.

    Two shapes, because the registry holds two shapes of quantity. A bandwidth,
    a latency or a profile multiplier is a SCALE - what is uncertain is the
    factor reality sits at, so its default is relative. A ``sim_error`` item is
    an ERROR FRACTION - an additive quantity that can be zero or negative, on
    which a relative width is meaningless (and, at a nominal of zero, empty).
    """

    #: ``[nominal * lo, nominal * hi]``, endpoints ordered afterwards so a
    #: negative nominal does not invert the interval.
    RELATIVE = "relative"
    #: ``[nominal - half_width, nominal + half_width]`` in the item's own unit.
    ABSOLUTE = "absolute"


class GradeDefault(_Strict):
    """One row of ``defaults:``: how wide an input of a grade is ASSUMED to be.

    Reached only when the (kind, grade) rule produces no width of its own. A
    row may name a ``kind`` to override the grade-level default for that kind -
    which `link_lat` needs, because a spec bandwidth is an upper bound on
    reality while a spec latency is a lower one.

    Absolute rule A1 still holds, in the form the domain-scoping work order S2
    sets: a default is a stated policy with a cited basis, never an invented
    number, and it is labelled as a default wherever it is used (D111).
    """

    grade: str
    #: Empty means "any kind"; a value narrows the row to that kind and wins
    #: over the grade-level row.
    kind: str = ""
    rule: DefaultRule = DefaultRule.RELATIVE
    #: RELATIVE only: multipliers on the nominal. `lo` may exceed 1 - a spec
    #: latency is a lower bound, so reality is at or above it.
    lo: float | None = Field(default=None, gt=0)
    hi: float | None = Field(default=None, gt=0)
    #: ABSOLUTE only: half-width in the item's own unit.
    half_width: float | None = Field(default=None, gt=0)
    source: str = ""
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.kind or '*'}/{self.grade}"

    @model_validator(mode="after")
    def _shape_and_source(self) -> GradeDefault:
        if not self.source.strip():
            raise ValueError(
                f"grades.yaml default {self.id}: carries a width but no source - "
                f"an unattributed default range is forbidden (absolute rule A1). "
                f"A grade with no defensible default has NO ROW, which is how it "
                f"stays undecidable."
            )
        if self.rule is DefaultRule.RELATIVE:
            if self.lo is None or self.hi is None:
                raise ValueError(f"grades.yaml default {self.id}: rule relative needs lo and hi")
            if self.hi <= self.lo:
                raise ValueError(
                    f"grades.yaml default {self.id}: relative range needs hi > lo, "
                    f"got [{self.lo}, {self.hi}]"
                )
            if self.half_width is not None:
                raise ValueError(
                    f"grades.yaml default {self.id}: rule relative takes no half_width"
                )
        else:
            if self.half_width is None:
                raise ValueError(f"grades.yaml default {self.id}: rule absolute needs half_width")
            if self.lo is not None or self.hi is not None:
                raise ValueError(f"grades.yaml default {self.id}: rule absolute takes no lo/hi")
        return self


class GradesTable(_Strict):
    """The whole of ``grades.yaml``, indexed by (kind, grade)."""

    rules: list[GradeRule] = Field(default_factory=list)
    #: The fallback layer (S2, D111): what an input of a grade is assumed to be
    #: worth when its own row sources no width. A grade absent from here has no
    #: default, which is what keeps "cannot be decided before measuring" a real
    #: category rather than a formality.
    defaults: list[GradeDefault] = Field(default_factory=list)
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
        seen_defaults: set[str] = set()
        for default in self.defaults:
            if default.id in seen_defaults:
                raise ValueError(f"grades.yaml: duplicate default {default.id}")
            seen_defaults.add(default.id)
        return self

    def without_defaults(self) -> GradesTable:
        """The same table with the defaults layer removed (pre-S2 behaviour).

        For the one caller that needs both answers from one file: an experiment
        comparing what the default ranges changed. Planning always uses the
        table as loaded.
        """
        return self.model_copy(update={"defaults": []})

    def default_for(self, kind: str, grade: str) -> GradeDefault | None:
        """The default range for a (kind, grade), or None when none is defined.

        A kind-specific row wins over the grade-level one; None is the answer
        that keeps an input undecidable, and it is returned rather than a
        zero-width stand-in so the caller cannot accidentally treat "no policy"
        as "no uncertainty".
        """
        fallback: GradeDefault | None = None
        for default in self.defaults:
            if default.grade != grade:
                continue
            if default.kind == kind:
                return default
            if not default.kind and fallback is None:
                fallback = default
        return fallback

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
        {
            "rules": raw.get("rules", []),
            "defaults": raw.get("defaults", []),
            "digest": prov.hash_file(path) or "",
            "path": str(path),
        }
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
