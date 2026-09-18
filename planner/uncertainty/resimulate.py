"""Exact endpoints for an uncertain input, by actually simulating them (§2.7).

B1's perturbation rules are post-processing: they take the cached prediction and
move it by arithmetic. Two of them are exact (`LINK_BW`, `LINK_LAT` re-price a
transfer term the planner adds itself) and the rest are first-order stand-ins.
`--resimulate-top N` is the escape hatch the work order leaves for the stand-ins:
for the N inputs the sweep values most, simulate the range's two ENDPOINTS for
real and recompute their regret from those predictions instead.

Two points, not the whole grid. That is the work order's choice and it is the
right one: a resimulation costs a full corpus each, so N inputs cost 2N corpora,
and the endpoints are where a monotone first-order rule is furthest from the
truth. The comparison E-B3 draws is therefore between two ΔR values computed on
the SAME two-point grid, differing only in where the metrics came from.

What "simulate this endpoint for real" means is different for every kind, and
for one kind it means nothing at all:

* `PROFILE` - write a copy of the hardware's perf bundle with every `time_us`
  scaled, point a copy of the profile at it, re-evaluate. This is the kind the
  feature exists for: a profile error is an error in per-operator time, and
  scaling latency percentiles by the same factor ignores that a slower engine
  also holds requests longer.

  One asymmetry to know about. A registry PROFILE item names ISLANDS, but a
  perf bundle belongs to an accelerator MODEL, and a cluster usually has several
  identical islands of one model. Only the candidates that use an affected
  island are re-evaluated, so a candidate on an untouched twin keeps its
  measured prediction - but a candidate spanning two islands of the same model
  gets both of them slowed, where the closed form applies one multiplier to the
  candidate as a whole. The two agree on "this candidate became ~x slower" and
  differ on how it was apportioned inside the candidate.
* `POWER` - scale the profile's power block and re-evaluate. No files: the power
  model is a handful of fields on the profile the planner already loaded.
* `LINK_BW` / `LINK_LAT` - rewrite the link in a copy of the cluster spec and
  re-evaluate. For a `p2p` LINK_BW item and for LINK_LAT the closed form is
  already exact, so E-B3 expects no rank movement and any movement is a bug in
  the rule. For an `all_reduce` LINK_BW item this module is not a cross-check
  but the ONLY way to price it: the value enters through the simulator's own
  `link_bw`, which `island_interconnect` reduces from these links, and no
  arithmetic outside the simulator reproduces a collective cost model. Those
  items arrive here with no closed-form dR to compare against - they are the
  `needs_resimulation` list, not a discrepancy (S3, D112). An earlier version of
  this docstring said their closed form was already exact, which was true only
  of the handoff half.
* `SIM_ERROR` - **there is nothing to simulate.** A prediction error is a claim
  about the margin, not about the prediction, so moving it changes no simulator
  input. Its closed form is exact by construction and this module refuses rather
  than inventing a run.

The cache is deliberately bypassed. `EnvelopeCache` keys on the accelerator
MODEL, not on the perf bundle behind it, so a scaled bundle would collide with
the measured one and silently return the unscaled prediction - a stale-cache
failure of exactly the shape that nearly put a wrong E-A1 into the record.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from planner.uncertainty.registry import (
    UncertainInput,
    UncertainKind,
    parse_link_item_id,
)

if TYPE_CHECKING:
    from planner.inventory import AcceleratorProfile, ExecutionIsland
    from planner.plan import CandidateConfig
    from planner.spec import ServiceSpec
    from planner.uncertainty.perturb import JudgedMetrics

PERF_ROOT = Path("profiler/perf")


class NotResimulable(Exception):
    """This kind has no simulator input to move, and saying so is the answer."""


@dataclass
class Endpoint:
    """One endpoint's exact metrics, and what it cost to get them."""

    value: float
    metrics: JudgedMetrics
    seconds: float
    simulated: int


@dataclass
class Resimulation:
    input_id: str
    kind: str
    lo: Endpoint
    hi: Endpoint

    @property
    def seconds(self) -> float:
        return self.lo.seconds + self.hi.seconds


# ---------------------------------------------------------------------------
# Perf bundle scaling
# ---------------------------------------------------------------------------

def scale_perf_bundle(hardware: str, factor: float, label: str) -> Path:
    """Copy `profiler/perf/<hardware>` with every `time_us` multiplied.

    Returns the new bundle's root. The caller owns it and is expected to remove
    it; `scaled_bundle` below does that with a context manager.

    Only the `time_us` column moves. Shapes, token counts and kv sizes are the
    experiment's independent variables, and meta.yaml records how the bundle was
    profiled - rewriting either would make the copy a different measurement
    rather than the same one at a different speed.
    """
    source = PERF_ROOT / hardware
    if not source.is_dir():
        raise NotResimulable(f"no perf bundle at {source}")
    dest = PERF_ROOT / label
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    for path in sorted(dest.rglob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "time_us" not in fields:
                continue
            rows = list(reader)
        for row in rows:
            try:
                row["time_us"] = f"{float(row['time_us']) * factor:.6f}"
            except (TypeError, ValueError):
                continue
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return dest


# ---------------------------------------------------------------------------
# Endpoint evaluation
# ---------------------------------------------------------------------------

def _affected_islands(item: UncertainInput) -> set[str]:
    return set(item.affects)


def _touched(item: UncertainInput, candidates: list[CandidateConfig]) -> list:
    """Which candidates this endpoint could possibly move.

    Structural, not the closed form's opinion: asking `perturb` which candidates
    it moved and then simulating only those would make the resimulation unable
    to contradict the rule it exists to check. For a PROFILE or POWER item that
    is every candidate placing work on an affected island; for a link it is the
    P/D candidates, because a KV transfer across a link is the only way a link
    enters a prediction at all (§2.7) - an aggregated candidate never crosses
    one.
    """
    from planner.plan import ServingArch

    if item.kind in (UncertainKind.LINK_BW, UncertainKind.LINK_LAT):
        return [c for c in candidates if c.serving_arch is ServingArch.PD_SPLIT]
    islands = _affected_islands(item)
    return [c for c in candidates if islands.intersection(c.islands)]


def endpoints_for(item: UncertainInput) -> tuple[float, float]:
    rng = item.range
    if rng is None or rng.lo is None or rng.hi is None:
        raise NotResimulable(
            f"{item.id}: unbounded range - there are no endpoints to simulate, "
            f"which is the same reason it carries no regret"
        )
    return (rng.lo, rng.hi)


def resimulate(
    item: UncertainInput,
    *,
    baseline: JudgedMetrics,
    spec: ServiceSpec,
    cluster,
    islands: dict[str, ExecutionIsland],
    profiles: dict[str, AcceleratorProfile],
    candidates: list[CandidateConfig],
    predictor,
    island_hw: dict[str, str],
    max_workers: int = 4,
) -> Resimulation:
    """Exact metrics at the item's `lo` and `hi`.

    Raises `NotResimulable` when the kind moves no simulator input. The caller
    reports that rather than substituting the closed form silently: "we could
    not check this one" and "we checked it and it agreed" are different claims.
    """
    if item.kind is UncertainKind.SIM_ERROR:
        raise NotResimulable(
            f"{item.id}: a SIM_ERROR moves the margin, not the prediction (§2.7), "
            f"so no simulator input changes and its closed form is exact by "
            f"construction - there is nothing a resimulation could disagree with"
        )
    lo, hi = endpoints_for(item)
    return Resimulation(
        input_id=item.id, kind=item.kind.value,
        lo=_endpoint(item, lo, baseline=baseline, spec=spec, cluster=cluster,
                     islands=islands, profiles=profiles, candidates=candidates,
                     predictor=predictor, island_hw=island_hw,
                     max_workers=max_workers),
        hi=_endpoint(item, hi, baseline=baseline, spec=spec, cluster=cluster,
                     islands=islands, profiles=profiles, candidates=candidates,
                     predictor=predictor, island_hw=island_hw,
                     max_workers=max_workers),
    )


def _endpoint(
    item: UncertainInput, value: float, *, baseline, spec, cluster, islands,
    profiles, candidates, predictor, island_hw, max_workers,
) -> Endpoint:
    """One endpoint, as a COMPLETE metric set.

    Only the candidates this input can touch are simulated - the rest cannot
    have moved - but what comes back is the baseline with those replaced, never
    the touched ones on their own. `perturb` returns a full set for the same
    reason, and the two have to agree: a partial set drops the incumbent
    whenever the input does not touch it, and the sweep then reads a missing
    incumbent as one that missed by a whole unit. That is how a profile on an
    island the recommendation does not use came back worth an entire objective.
    """
    import time

    from planner.optimizer import exhaustive
    from planner.uncertainty.perturb import JudgedMetrics, judged_metrics

    scaled_cluster = cluster
    scaled_profiles = dict(profiles)
    bundle: Path | None = None
    try:
        if item.kind is UncertainKind.PROFILE:
            bundle, scaled_profiles = _scaled_profiles(
                item, value, islands, profiles
            )
        elif item.kind is UncertainKind.POWER:
            scaled_profiles = _powered_profiles(item, value, islands, profiles)
        elif item.kind in (UncertainKind.LINK_BW, UncertainKind.LINK_LAT):
            scaled_cluster = _relinked_cluster(item, value, cluster)
        else:
            raise NotResimulable(f"{item.id}: no resimulation rule for {item.kind}")

        touched = _touched(item, candidates)
        if not touched:
            raise NotResimulable(
                f"{item.id}: no candidate in this corpus uses it, so there is "
                f"nothing to simulate at either endpoint"
            )
        started = time.monotonic()
        evaluation = exhaustive.evaluate_candidates(
            touched, spec, scaled_cluster, islands, scaled_profiles, predictor,
            cache=None, island_hw=island_hw, max_workers=max_workers,
        )
        elapsed = time.monotonic() - started
        merged = dict(baseline)
        merged.update(judged_metrics(evaluation))
    finally:
        if bundle is not None and bundle.exists():
            shutil.rmtree(bundle)
    return Endpoint(
        value=value, metrics=JudgedMetrics.trusted(merged), seconds=elapsed,
        simulated=len(touched),
    )


def _scaled_profiles(item, value, islands, profiles):
    hardware = {
        islands[i].accelerator_model for i in _affected_islands(item) if i in islands
    }
    if len(hardware) != 1:
        raise NotResimulable(
            f"{item.id}: affects {len(hardware)} accelerator models; a scaled "
            f"bundle is per hardware, so this item cannot be resimulated as one"
        )
    model = hardware.pop()
    profile = profiles.get(model)
    if profile is None or profile.sim_hardware is None:
        raise NotResimulable(f"{item.id}: {model} has no sim_hardware bundle")
    # The factor is `value` itself, NOT `value / nominal`. A PROFILE item's unit
    # is a multiplier on the MEASURED bundle (`registry.build_registry` gives it
    # nominal 1.0), and the bundle on disk is that measured one - so `value` is
    # already the number to scale it by. Dividing by nominal would be right only
    # if the bundle already carried the nominal's slowdown, which it never does:
    # a degraded belief lives in the registry, not in `profiler/perf/`.
    label = f"{profile.sim_hardware}-resim{value:.4f}".replace(".", "_")
    bundle = scale_perf_bundle(profile.sim_hardware, value, label)
    out = dict(profiles)
    out[model] = profile.model_copy(update={"sim_hardware": label})
    return bundle, out


def _powered_profiles(item, value, islands, profiles):
    # As with PROFILE: `value` is a multiplier on the profile's measured power
    # block, and that block is what the loaded profile carries.
    factor = value
    out = dict(profiles)
    for island_id in _affected_islands(item):
        island = islands.get(island_id)
        if island is None:
            continue
        profile = profiles.get(island.accelerator_model)
        if profile is None or profile.power is None:
            continue
        power = profile.power
        out[island.accelerator_model] = profile.model_copy(update={
            "power": power.model_copy(update={
                "idle_power": power.idle_power * factor,
                "standby_power": power.standby_power * factor,
                "active_power": power.active_power * factor,
            })
        })
    return out


def _relinked_cluster(item, value, cluster):
    link_id = parse_link_item_id(item.id).link_id
    field = (
        "bandwidth_gbps" if item.kind is UncertainKind.LINK_BW else "latency_ns"
    )
    links = [
        link.model_copy(update={field: value}) if link.id == link_id else link
        for link in cluster.links
    ]
    if all(link.id != link_id for link in cluster.links):
        raise NotResimulable(f"{item.id}: no link {link_id!r} in this cluster")
    return cluster.model_copy(update={"links": links})
