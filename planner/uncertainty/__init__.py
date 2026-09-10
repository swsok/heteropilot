"""Uncertainty-aware planning (WORK_ORDER_uncertainty_planner.md, Stage A/B).

The planner's inputs carry uncertainty from three unrelated places - simulator
prediction error, profile-bundle trust tier, and link bandwidth/latency
provenance - and until now each was handled differently and none of them
reached the verdict. This package collects all three into one registry so a
later stage can perturb each entry and say which measurement would change the
recommendation.

Absolute rule A1 governs everything here: an error range with no source is
recorded as ``unbounded``, never as a plausible default.
"""

from planner.uncertainty.grades import (
    GradeRule,
    GradesTable,
    RangeRule,
    load_costs,
    load_grades,
)
from planner.uncertainty.registry import (
    Grade,
    MeasurementCost,
    Range,
    UncertainInput,
    UncertainInputRegistry,
    UncertainKind,
    build_registry,
)

__all__ = [
    "Grade",
    "GradeRule",
    "GradesTable",
    "MeasurementCost",
    "Range",
    "RangeRule",
    "UncertainInput",
    "UncertainInputRegistry",
    "UncertainKind",
    "build_registry",
    "load_costs",
    "load_grades",
]
