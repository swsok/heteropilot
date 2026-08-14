"""Parse the simulator's power output from stdout (docs/deviations.md D2).

The per-request CSV carries no power, energy or memory column - energy exists
only as console text. Upstream does the same thing in
`bench/core/validate.py::_load_sim_log`, so log parsing is the established
contract for metrics absent from the CSV, not a workaround invented here.

Every regex lives in this module so a format change fails in one place with a
golden-fixture test, instead of silently producing wrong energy numbers.
Redirected output is plain text (Rich detects the non-TTY); the only non-ASCII is
stable box drawing in the component tree, which these patterns never rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOTAL_ENERGY_KJ = re.compile(r"^Total energy consumption \(kJ\):\s*([\d.]+)\s*$", re.M)
_NODE_ENERGY_KJ = re.compile(r"^Node (\d+) total energy consumption \(kJ\):\s*([\d.]+)\s*$", re.M)
_COMPONENT_J = re.compile(r"([A-Za-z][A-Za-z +]*?) energy consumption \(J\):\s*([\d.]+)")
_POWER_SERIES = re.compile(r"^Power per ([\d.]+) sec \(W\):\s*\[([^\]]*)\]\s*$", re.M)


class PowerParseError(ValueError):
    """Raised when a power block is present but cannot be read."""


@dataclass(frozen=True)
class PowerSummary:
    total_energy_j: float
    per_node_energy_j: dict[int, float]
    per_component_j: dict[str, float]
    power_series_w: list[float]
    interval_s: float

    @property
    def average_power_w(self) -> float:
        if not self.power_series_w:
            return 0.0
        return sum(self.power_series_w) / len(self.power_series_w)

    @property
    def peak_power_w(self) -> float:
        """Highest sampled interval average.

        Resolution equals the run's `--log-interval`, so this is an interval
        mean, not a true instantaneous peak. A coarse interval understates it;
        enforcing a power cap against this figure is optimistic by that much.
        """
        return max(self.power_series_w) if self.power_series_w else 0.0

    def tokens_per_joule(self, completed_tokens: int) -> float | None:
        if self.total_energy_j <= 0:
            return None
        return completed_tokens / self.total_energy_j

    def as_provenance(self) -> dict[str, object]:
        return {
            "total_energy_j": self.total_energy_j,
            "average_power_w": round(self.average_power_w, 3),
            "peak_power_w": self.peak_power_w,
            "power_sample_interval_s": self.interval_s,
            "power_samples": len(self.power_series_w),
            "peak_is_interval_average": True,
        }


@dataclass
class PowerParseResult:
    summary: PowerSummary | None
    warnings: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return self.summary is not None


def parse_power(stdout: str) -> PowerParseResult:
    """Extract the power block, or report its absence.

    Absence is normal and not an error: the simulator only emits power when the
    cluster config carries a `power:` block.
    """
    warnings: list[str] = []

    total_match = _TOTAL_ENERGY_KJ.search(stdout)
    series_match = _POWER_SERIES.search(stdout)
    if total_match is None and series_match is None:
        return PowerParseResult(None, ["no power block in simulator output"])

    if total_match is None:
        raise PowerParseError(
            "found a 'Power per N sec (W)' series but no 'Total energy consumption (kJ)' "
            "line; the output format has changed"
        )
    total_energy_j = float(total_match.group(1)) * 1000.0

    per_node = {
        int(node): float(kj) * 1000.0 for node, kj in _NODE_ENERGY_KJ.findall(stdout)
    }
    if not per_node:
        warnings.append("no per-node energy lines found")

    per_component: dict[str, float] = {}
    for label, joules in _COMPONENT_J.findall(stdout):
        per_component[label.strip()] = per_component.get(label.strip(), 0.0) + float(joules)

    interval_s = 0.0
    series: list[float] = []
    if series_match is None:
        warnings.append(
            "no 'Power per N sec (W)' series; average and peak power are unavailable"
        )
    else:
        interval_s = float(series_match.group(1))
        body = series_match.group(2).strip()
        if body:
            try:
                series = [float(x) for x in body.split(",")]
            except ValueError as exc:
                raise PowerParseError(f"cannot parse power series {body!r}: {exc}") from None

    return PowerParseResult(
        PowerSummary(
            total_energy_j=total_energy_j,
            per_node_energy_j=per_node,
            per_component_j=per_component,
            power_series_w=series,
            interval_s=interval_s,
        ),
        warnings,
    )
