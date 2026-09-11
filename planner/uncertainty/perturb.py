"""Closed-form perturbation of cached predictions (§2.7, STEP B1).

Stage B asks what would change if an uncertain input turned out to be a
different value. Re-simulating every candidate at every grid point of every
registry entry is the obvious way and the wrong one: that is
`items x grid x candidates` simulations, and E-A1 needed 24 minutes for 162.
Instead each kind of uncertainty has a closed form that post-processes the
metrics already predicted, so the whole sweep is arithmetic.

Every function here is pure. Inputs are never mutated - a perturbed metric set
is always a `model_copy`, which is also what lets B2 evaluate grid points in any
order without them contaminating each other.

**Where a rule is exact and where it is not.** The `LINK_BW` and `LINK_LAT`
rules are EXACT, which the work order expected to have to approximate. The
reason is that the simulator does not model the P/D handoff at all - it
reallocates KV in zero simulated time - so the transfer term is not something
the simulator buried inside a queueing model. The planner adds it afterwards, by
plain addition per TTFT percentile, in `exhaustive.apply_pd_transfer_cost`.
Perturbing it is therefore recomputing one addend with the same helper that
produced it (`kv_transfer.transfer_ms`), not approximating a simulated effect.
`PROFILE` and `POWER` are first-order scalings and say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from planner.plan import PredictedMetrics, Role, ServingArch
from planner.topology import TopologyError, TopologyGraph
from planner.uncertainty.registry import UncertainInput, UncertainKind
from planner.util import kv_transfer

if TYPE_CHECKING:
    from planner.inventory import ClusterSpecV2, ExecutionIsland
    from planner.plan import CandidateConfig
    from planner.predictor import SimResult
    from planner.spec import ServiceSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PerturbContext:
    """Everything the rules need beyond the metrics themselves.

    Built once per analysis and reused across every item and grid point, because
    the topology and the candidate set do not move while a registry entry does.
    """

    spec: ServiceSpec
    cluster: ClusterSpecV2
    islands: dict[str, ExecutionIsland]
    candidates: dict[str, CandidateConfig]
    topology: TopologyGraph
    #: candidate id -> the run's per-hardware operating point
    #: (`SearchResult.operating_points`). A margin policy reads the operating
    #: point off the RUN, not off the metrics, so a re-judged grid point needs it
    #: beside the perturbed metrics. Empty for a predictor with no per-request
    #: records, which an accuracy-domain policy correctly reports as unmeasured.
    operating_points: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        spec: ServiceSpec,
        cluster: ClusterSpecV2,
        islands: dict[str, ExecutionIsland],
        candidates: dict[str, CandidateConfig],
        operating_points: dict[str, dict] | None = None,
    ) -> PerturbContext:
        return cls(spec, cluster, islands, candidates, TopologyGraph(cluster),
                   dict(operating_points or {}))

    def sim_for(self, candidate_id: str, metrics: PredictedMetrics) -> SimResult:
        """A `SimResult` a margin policy can read: the perturbed metrics beside
        the candidate's recorded operating point."""
        from planner.predictor import SimOutcome, SimResult

        return SimResult(
            candidate_id, SimOutcome.OK, metrics=metrics,
            operating_point=dict(self.operating_points.get(candidate_id, {})),
        )


class JudgedMetrics(dict):
    """Metrics as the planner JUDGED them - the only input `perturb` accepts.

    §2.7 asks for this to be a type and not a docstring, and the reason is a
    real trap. The envelope cache stores the RAW simulator output, from before
    `exhaustive.apply_pd_transfer_cost` adds the prefill->decode transfer term.
    The LINK_BW rule re-prices that term by adding the DIFFERENCE between two
    transfer times, so handing it raw metrics adds a difference to a number that
    never contained the original - silently, and only for P/D candidates, which
    is the worst way for it to be wrong.

    Build it with `judged_metrics(evaluation)` from a `SearchResult`, or with
    `JudgedMetrics.trusted(mapping)` when the caller has already applied the
    adjustment and can say so.
    """

    @classmethod
    def trusted(cls, mapping: dict[str, PredictedMetrics]) -> JudgedMetrics:
        """Wrap a mapping the caller asserts is post-adjustment."""
        return cls(mapping)


def judged_metrics(evaluation) -> JudgedMetrics:
    """Post-adjustment metrics for every candidate a `SearchResult` judged."""
    out: dict[str, PredictedMetrics] = {}
    for plan in evaluation.feasible_plans:
        out[plan.candidate.id] = plan.predicted
    for plan, _report in evaluation.infeasible_plans:
        out[plan.candidate.id] = plan.predicted
    out.update(evaluation.unmeasured_metrics)
    return JudgedMetrics(out)


class PerturbResult(_Strict):
    """What one (item, grid point) does to a whole metric set."""

    metrics: dict[str, PredictedMetrics]
    #: True when the rule is a first-order stand-in rather than the same
    #: arithmetic the planner itself uses. The measurement plan marks these.
    approximation: bool = False
    #: Candidate ids whose metrics actually moved.
    affected: list[str] = Field(default_factory=list)
    #: For SIM_ERROR only: the error fraction to apply INSTEAD of the accuracy
    #: domain's interpolation. The metrics are untouched - a prediction error is
    #: a claim about the margin, not about the prediction (§2.7).
    sim_error_override: float | None = None
    note: str = ""


def perturb(
    item: UncertainInput,
    value: float,
    metrics: JudgedMetrics,
    context: PerturbContext,
) -> PerturbResult:
    """Re-price every candidate as if `item` really had this value.

    `metrics` must be a `JudgedMetrics` - see that class for why a plain dict is
    refused rather than warned about.

    `value` is a point in the item's own units - the grid of §2.5 walks
    `item.range`, so a LINK_BW point is GB/s and a PROFILE point is the
    multiplier whose nominal is 1.0. Rules derive whatever relative change they
    need from `value` against `item.nominal`, so callers never have to know
    which convention a kind uses.
    """
    if not isinstance(metrics, JudgedMetrics):
        raise TypeError(
            "perturb() needs JudgedMetrics - metrics as the planner judged them, "
            "after apply_pd_transfer_cost. A plain dict is most likely the envelope "
            "cache's raw simulator output, which has no transfer term for the "
            "LINK_BW rule to re-price. Build one with judged_metrics(evaluation), or "
            "JudgedMetrics.trusted(mapping) if it is already post-adjustment."
        )
    if item.kind is UncertainKind.SIM_ERROR:
        return PerturbResult(
            metrics=dict(metrics), sim_error_override=value,
            note=f"{item.id}: margin moved to {value:+.4f}; predictions unchanged",
        )
    if item.kind is UncertainKind.PROFILE:
        return _scale_profile(item, value, metrics, context)
    if item.kind is UncertainKind.POWER:
        return _scale_power(item, value, metrics, context)
    if item.kind in (UncertainKind.LINK_BW, UncertainKind.LINK_LAT):
        return _reprice_transfer(item, value, metrics, context)
    raise ValueError(f"no perturbation rule for kind {item.kind}")


# ---------------------------------------------------------------------------
# PROFILE and POWER: first-order scalings
# ---------------------------------------------------------------------------

def _relative(value: float, nominal: float) -> float:
    """`delta` such that value == nominal * (1 + delta)."""
    if nominal == 0:
        raise ValueError("a multiplicative rule needs a non-zero nominal")
    return value / nominal - 1.0


def _candidates_using(item: UncertainInput, context: PerturbContext) -> set[str]:
    """Candidate ids that place work on any island this item affects."""
    islands = set(item.affects)
    return {
        cid for cid, candidate in context.candidates.items()
        if islands.intersection(candidate.islands)
    }


def _scale_profile(item, value, metrics, context) -> PerturbResult:
    """A profile that is `(1+delta)` slower makes its candidates that much slower.

    Latencies scale up, rates scale down, energy does not move: the power model
    is a separate uncertain input (POWER) and double-counting it here would let
    one measurement appear to settle two.

    APPROXIMATE, and the reason is worth stating. A profile error is an error in
    per-operator time; TTFT and TPOT are not linear in it once a queue forms,
    because a slower engine also holds requests longer. This scales them as if
    they were, which is right to first order and conservative in the direction
    that matters - it never makes a candidate look better than the resimulation
    would.
    """
    delta = _relative(value, item.nominal)
    touched = _candidates_using(item, context)
    scale = 1.0 + delta
    if scale <= 0:
        raise ValueError(f"{item.id}: a profile multiplier must stay positive, got {scale}")

    out = dict(metrics)
    affected: list[str] = []
    for cid in sorted(touched & set(metrics)):
        m = metrics[cid]
        out[cid] = m.model_copy(update={
            "p50_ttft_ms": m.p50_ttft_ms * scale,
            "p95_ttft_ms": m.p95_ttft_ms * scale,
            "p99_ttft_ms": m.p99_ttft_ms * scale,
            "p50_tpot_ms": m.p50_tpot_ms * scale,
            "p95_tpot_ms": m.p95_tpot_ms * scale,
            "p99_tpot_ms": m.p99_tpot_ms * scale,
            "throughput_tps": m.throughput_tps / scale,
            "slo_goodput_rps": m.slo_goodput_rps / scale,
        })
        affected.append(cid)
    return PerturbResult(
        metrics=out, approximation=True, affected=affected,
        note=(
            f"{item.id}: latencies x{scale:.4g}, rates /{scale:.4g} on "
            f"{len(affected)} candidate(s); energy unchanged (POWER owns it). "
            f"First-order: latency is not linear in operator time once a queue forms"
        ),
    )


def _scale_power(item, value, metrics, context) -> PerturbResult:
    """A power model that is `(1+delta)` high makes its candidates' energy so.

    Latency does not move: drawing more watts for the same work changes joules,
    not milliseconds. `tokens_per_joule` is recomputed from the new energy rather
    than scaled, so it stays consistent with `completed_tokens`.
    """
    delta = _relative(value, item.nominal)
    scale = 1.0 + delta
    if scale <= 0:
        raise ValueError(f"{item.id}: a power multiplier must stay positive, got {scale}")
    touched = _candidates_using(item, context)

    out = dict(metrics)
    affected: list[str] = []
    for cid in sorted(touched & set(metrics)):
        m = metrics[cid]
        if m.total_energy_j is None:
            # No energy was simulated for this candidate (deviations D2/D14), so
            # there is nothing to scale. Left alone rather than invented.
            continue
        new_energy = m.total_energy_j * scale
        out[cid] = m.model_copy(update={
            "total_energy_j": new_energy,
            "average_power_w": (
                None if m.average_power_w is None else m.average_power_w * scale
            ),
            "peak_power_w": None if m.peak_power_w is None else m.peak_power_w * scale,
            "tokens_per_joule": (
                m.completed_tokens / new_energy if new_energy > 0 else None
            ),
        })
        affected.append(cid)
    return PerturbResult(
        metrics=out, approximation=True, affected=affected,
        note=(
            f"{item.id}: energy x{scale:.4g} on {len(affected)} candidate(s); "
            f"latency unchanged. First-order: a constant factor on average power"
        ),
    )


# ---------------------------------------------------------------------------
# LINK_BW / LINK_LAT: exact, because the planner owns the term
# ---------------------------------------------------------------------------

def _link_by_id(context: PerturbContext, link_id: str):
    for link in context.cluster.links:
        if link.id == link_id:
            return link
    return None


def _pd_endpoints(candidate, context) -> tuple[str, str] | None:
    """(prefill endpoint, decode endpoint) for a P/D candidate, else None."""
    if candidate.serving_arch is not ServingArch.PD_SPLIT:
        return None
    prefills = [a for a in candidate.assignments if a.role is Role.PREFILL]
    decodes = [a for a in candidate.assignments if a.role is Role.DECODE]
    if len(prefills) != 1 or len(decodes) != 1:
        return None
    pf = context.islands.get(prefills[0].island_id)
    dc = context.islands.get(decodes[0].island_id)
    if pf is None or dc is None:
        return None
    return (
        f"{pf.node_id}/{pf.accelerator_ids[0]}",
        f"{dc.node_id}/{dc.accelerator_ids[0]}",
    )


def _reprice_transfer(item, value, metrics, context) -> PerturbResult:
    """Recompute the prefill->decode transfer at the perturbed link and re-add it.

    EXACT. `apply_pd_transfer_cost` adds `transfer_ms(...)` to each TTFT
    percentile and nothing else touches it, so the difference between the old
    and new transfer time is exactly what the planner would have computed had
    the link really carried this value. Both call `kv_transfer.transfer_ms`.

    Only P/D candidates whose path actually crosses this link move. Everything
    else - a single-island candidate, a P/D routed elsewhere - is returned
    untouched, which is the honest answer and not an omission: the transfer term
    is the only place a link bandwidth reaches a predicted metric today.
    """
    link_id = item.id.split(":", 1)[1] if ":" in item.id else item.id
    link = _link_by_id(context, link_id)
    if link is None:
        return PerturbResult(
            metrics=dict(metrics),
            note=f"{item.id}: no link {link_id!r} in the cluster; nothing to perturb",
        )

    tok = context.spec.traffic.input_tokens
    prompts = (
        tok.p50,
        tok.p95 if tok.p95 is not None else tok.p50,
        tok.p99 if tok.p99 is not None else tok.p50,
    )
    kv_per_token = kv_transfer._kv_bytes_per_token(
        context.spec.model, context.spec.service.dtype, context.spec.service.kv_cache_dtype
    )

    out = dict(metrics)
    affected: list[str] = []
    for cid in sorted(set(metrics)):
        candidate = context.candidates.get(cid)
        if candidate is None:
            continue
        endpoints = _pd_endpoints(candidate, context)
        if endpoints is None:
            continue
        try:
            path = context.topology.path(*endpoints)
        except TopologyError:
            # No declared path: `apply_pd_transfer_cost` charges the class
            # default, which this link is not part of, so nothing moves.
            continue
        if not any(hop.id == link.id for hop in path):
            continue

        before_bw = TopologyGraph.effective_bandwidth_gbps(path)
        before_lat = TopologyGraph.path_latency_ns(path)
        if item.kind is UncertainKind.LINK_BW:
            moved = [
                hop.model_copy(update={"bandwidth_gbps": value}) if hop.id == link.id
                else hop for hop in path
            ]
        else:
            moved = [
                hop.model_copy(update={"latency_ns": value}) if hop.id == link.id
                else hop for hop in path
            ]
        after_bw = TopologyGraph.effective_bandwidth_gbps(moved)
        after_lat = TopologyGraph.path_latency_ns(moved)

        m = metrics[cid]
        updates: dict[str, object] = {}
        for prompt, name in zip(prompts, ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms"),
                                strict=True):
            kv_bytes = kv_per_token * prompt
            shift = (
                kv_transfer.transfer_ms(
                    kv_bytes, bandwidth_gbps=after_bw, latency_ns=after_lat)
                - kv_transfer.transfer_ms(
                    kv_bytes, bandwidth_gbps=before_bw, latency_ns=before_lat)
            )
            updates[name] = getattr(m, name) + shift
        out[cid] = m.model_copy(update=updates)
        affected.append(cid)

    unit = "GB/s" if item.kind is UncertainKind.LINK_BW else "ns"
    return PerturbResult(
        metrics=out, approximation=False, affected=affected,
        note=(
            f"{item.id}: {link_id} {item.nominal:g} -> {value:g} {unit}; "
            f"P/D transfer re-priced on {len(affected)} candidate(s) with the same "
            f"helper apply_pd_transfer_cost uses. TPOT unchanged (the transfer is "
            f"one-time, before decode). The S4' pipeline lower bound also reads link "
            f"bandwidth, but it is a BOUND, not a metric, and §1.4 forbids changing it"
        ),
    )
