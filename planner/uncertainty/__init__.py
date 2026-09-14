"""Uncertainty-aware planning (WORK_ORDER_uncertainty_planner.md, Stage A/B).

The planner's inputs carry uncertainty from three unrelated places - simulator
prediction error, profile-bundle trust tier, and link bandwidth/latency
provenance - and until now each was handled differently and none of them
reached the verdict. This package collects all three into one registry so a
later stage can perturb each entry and say which measurement would change the
recommendation.

Absolute rule A1 governs everything here: an error range with no source is
recorded as ``unbounded``, never as a plausible default.

`perturb` is deliberately NOT re-exported here. `planner.plan` imports
`registry` for the `uncertain_inputs` field, so this package's ``__init__`` runs
while `planner.plan` is still initialising; `perturb` needs `PredictedMetrics` at
runtime (it is a pydantic field type, not just an annotation), which would close
the cycle. Import it as ``from planner.uncertainty.perturb import perturb``.
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
