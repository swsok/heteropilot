"""Closed-form perturbation (WORK_ORDER_uncertainty_planner.md STEP B1).

The LINK_BW case is checked against `apply_pd_transfer_cost` itself rather than
against a formula retyped here: the whole claim of the rule is that it is the
same arithmetic, so a test that re-derived it would pass even if the two drifted
apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    PredictedMetrics,
    Role,
    ServingArch,
)
from planner.spec import load_service_spec
from planner.topology import TopologyGraph
from planner.uncertainty.perturb import JudgedMetrics, PerturbContext, perturb
from planner.uncertainty.registry import Grade, Range, UncertainInput, UncertainKind
from planner.util import kv_transfer

ROOT = Path(__file__).resolve().parents[1]
PD_FIXTURE = ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml"
SERVICE = ROOT / "examples/service_specs/llama31-8b.yaml"
SRC = "tests/test_perturb.py synthetic fixture"


@pytest.fixture
def world():
    spec = load_service_spec(SERVICE)
    cluster = load_cluster_spec(PD_FIXTURE)
    profiles = load_profiles_for(cluster, ROOT)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    return spec, cluster, profiles, islands


def _metrics(**over) -> PredictedMetrics:
    base: dict[str, object] = {
        "p50_ttft_ms": 100.0, "p95_ttft_ms": 200.0, "p99_ttft_ms": 300.0,
        "p50_tpot_ms": 10.0, "p95_tpot_ms": 20.0, "p99_tpot_ms": 30.0,
        "throughput_tps": 1000.0, "slo_goodput_rps": 5.0, "slo_attainment": 1.0,
        "completed_requests": 300, "completed_tokens": 60_000,
        "total_energy_j": 1000.0, "average_power_w": 100.0, "peak_power_w": 150.0,
        "tokens_per_joule": 60.0, "served_concurrency": 42.0,
    }
    base.update(over)
    return PredictedMetrics(**base)


def _single(cid: str, island_id: str) -> CandidateConfig:
    return CandidateConfig(
        id=cid, model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[IslandAssignment(island_id=island_id, tp_size=1)],
    )


def _pd(cid: str, prefill: str, decode: str, tp: int = 4) -> CandidateConfig:
    return CandidateConfig(
        id=cid, model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        serving_arch=ServingArch.PD_SPLIT,
        assignments=[
            IslandAssignment(island_id=prefill, role=Role.PREFILL, tp_size=tp),
            IslandAssignment(island_id=decode, role=Role.DECODE, tp_size=tp),
        ],
    )


def _item(kind: UncertainKind, ident: str, nominal: float, affects=()) -> UncertainInput:
    return UncertainInput(
        id=ident, kind=kind, grade=Grade.PLACEHOLDER, nominal=nominal,
        range=Range(unit="fraction"), affects=list(affects),
    )


# --- the identity ---------------------------------------------------------

@pytest.mark.parametrize(
    ("kind", "ident", "nominal"),
    [
        (UncertainKind.PROFILE, "profile:i0", 1.0),
        (UncertainKind.POWER, "power:i0", 1.0),
        (UncertainKind.SIM_ERROR, "sim_error:HW/b", 0.0),
    ],
)
def test_the_nominal_value_changes_nothing(world, kind, ident, nominal) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"c": _single("c", island_id)}
    metrics = JudgedMetrics.trusted({"c": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)

    result = perturb(_item(kind, ident, nominal, [island_id]), nominal, metrics, context)
    assert result.metrics["c"].model_dump() == metrics["c"].model_dump()


# --- PROFILE --------------------------------------------------------------

def test_a_slower_profile_scales_latency_up_and_rates_down(world) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"c": _single("c", island_id)}
    metrics = JudgedMetrics.trusted({"c": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)

    result = perturb(
        _item(UncertainKind.PROFILE, "profile:i0", 1.0, [island_id]), 1.1, metrics, context
    )
    m, before = result.metrics["c"], metrics["c"]
    assert m.p50_ttft_ms == pytest.approx(before.p50_ttft_ms * 1.1)
    assert m.p99_ttft_ms == pytest.approx(before.p99_ttft_ms * 1.1)
    assert m.p95_tpot_ms == pytest.approx(before.p95_tpot_ms * 1.1)
    assert m.throughput_tps == pytest.approx(before.throughput_tps / 1.1)
    assert m.slo_goodput_rps == pytest.approx(before.slo_goodput_rps / 1.1)
    assert result.approximation is True


def test_a_slower_profile_burns_proportionally_more_energy(world) -> None:
    """Energy is watts x seconds, and a PROFILE error moves the seconds (D34).

    This asserted the opposite until D34: `_scale_profile` held energy fixed on
    the grounds that energy belonged to POWER. The simulator says otherwise -
    `serving/core/power_model.py` accumulates active energy as
    `(active_power - idle_power) * latency_s` - and holding it fixed made every
    PROFILE item score `delta_regret == 0` under a `minimize_energy` objective
    unless it happened to cross an SLO boundary.
    """
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"c": _single("c", island_id)}
    metrics = JudgedMetrics.trusted({"c": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)

    result = perturb(
        _item(UncertainKind.PROFILE, "profile:i0", 1.0, [island_id]), 1.1, metrics, context
    )
    m, before = result.metrics["c"], metrics["c"]
    assert m.total_energy_j == pytest.approx(before.total_energy_j * 1.1)
    # Recomputed from the new energy, not scaled - so it stays consistent with
    # `completed_tokens`, which a profile perturbation does not change.
    assert m.completed_tokens == before.completed_tokens
    assert m.tokens_per_joule == pytest.approx(m.completed_tokens / m.total_energy_j)
    assert m.tokens_per_joule == pytest.approx(before.tokens_per_joule / 1.1)
    # WATTS are POWER's axis and must not move: a slower engine draws the same
    # power for longer. If these scaled too, PROFILE and POWER would double-count.
    assert m.average_power_w == before.average_power_w
    assert m.peak_power_w == before.peak_power_w


def test_a_profile_with_no_simulated_energy_is_left_alone(world) -> None:
    """Deviations D2/D14: absent energy is absent, never invented."""
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"c": _single("c", island_id)}
    metrics = JudgedMetrics.trusted({
        "c": _metrics(total_energy_j=None, tokens_per_joule=None,
                      average_power_w=None, peak_power_w=None)
    })
    context = PerturbContext.build(spec, cluster, islands, candidates)

    result = perturb(
        _item(UncertainKind.PROFILE, "profile:i0", 1.0, [island_id]), 1.4, metrics, context
    )
    m = result.metrics["c"]
    assert m.total_energy_j is None and m.tokens_per_joule is None
    # The latency half still applies - only the energy term had nothing to move.
    assert m.p99_ttft_ms == pytest.approx(metrics["c"].p99_ttft_ms * 1.4)


def test_profile_and_power_compose_without_double_counting(world) -> None:
    """PROFILE moves seconds, POWER moves watts; energy is their product.

    D34's rationale, as an assertion. Applying both in either order must give
    energy `x (1+dp)(1+dw)` exactly once, latency `x (1+dp)` only, and average
    watts `x (1+dw)` only. If either rule claimed the other's factor, the two
    orders would still agree but the magnitude would be wrong - so the test
    checks the magnitude against the product, not merely order-independence.
    """
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"c": _single("c", island_id)}
    base = JudgedMetrics.trusted({"c": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)
    prof = _item(UncertainKind.PROFILE, "profile:i0", 1.0, [island_id])
    powr = _item(UncertainKind.POWER, "power:i0", 1.0, [island_id])
    dp, dw = 1.25, 1.60

    first = JudgedMetrics.trusted(perturb(prof, dp, base, context).metrics)
    both = perturb(powr, dw, first, context).metrics["c"]
    other = JudgedMetrics.trusted(perturb(powr, dw, base, context).metrics)
    reversed_ = perturb(prof, dp, other, context).metrics["c"]

    before = base["c"]
    assert both.total_energy_j == pytest.approx(before.total_energy_j * dp * dw)
    assert reversed_.total_energy_j == pytest.approx(both.total_energy_j)
    # Each factor moves its own axis and only its own.
    assert both.p99_ttft_ms == pytest.approx(before.p99_ttft_ms * dp)
    assert both.average_power_w == pytest.approx(before.average_power_w * dw)
    assert both.tokens_per_joule == pytest.approx(
        before.completed_tokens / both.total_energy_j
    )


def test_a_profile_only_touches_candidates_on_its_own_islands(world) -> None:
    spec, cluster, _profiles, islands = world
    ids = list(islands)
    candidates = {"on": _single("on", ids[0]), "off": _single("off", ids[1])}
    metrics = JudgedMetrics.trusted({"on": _metrics(), "off": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)

    result = perturb(
        _item(UncertainKind.PROFILE, "profile:i0", 1.0, [ids[0]]), 1.5, metrics, context
    )
    assert result.affected == ["on"]
    assert result.metrics["off"].model_dump() == metrics["off"].model_dump()


def test_a_non_positive_profile_multiplier_is_refused(world) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    context = PerturbContext.build(spec, cluster, islands, {"c": _single("c", island_id)})
    with pytest.raises(ValueError, match="stay positive"):
        perturb(
            _item(UncertainKind.PROFILE, "profile:i0", 1.0, [island_id]),
            0.0, JudgedMetrics.trusted({"c": _metrics()}), context,
        )


# --- POWER ----------------------------------------------------------------

def test_power_moves_energy_and_not_latency(world) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"c": _single("c", island_id)}
    metrics = JudgedMetrics.trusted({"c": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)

    result = perturb(
        _item(UncertainKind.POWER, "power:i0", 1.0, [island_id]), 1.25, metrics, context
    )
    m, before = result.metrics["c"], metrics["c"]
    assert m.total_energy_j == pytest.approx(before.total_energy_j * 1.25)
    assert m.average_power_w == pytest.approx(before.average_power_w * 1.25)
    assert m.p50_ttft_ms == before.p50_ttft_ms
    # Recomputed from the new energy, not scaled, so it stays consistent.
    assert m.tokens_per_joule == pytest.approx(before.completed_tokens / m.total_energy_j)


def test_power_leaves_a_candidate_with_no_simulated_energy_alone(world) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    metrics = JudgedMetrics.trusted({"c": _metrics(
        total_energy_j=None, tokens_per_joule=None,
        average_power_w=None, peak_power_w=None)})
    context = PerturbContext.build(spec, cluster, islands, {"c": _single("c", island_id)})
    result = perturb(
        _item(UncertainKind.POWER, "power:i0", 1.0, [island_id]), 2.0, metrics, context
    )
    assert result.affected == []
    assert result.metrics["c"].total_energy_j is None


# --- SIM_ERROR ------------------------------------------------------------

def test_sim_error_moves_the_margin_and_not_the_prediction(world) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    metrics = JudgedMetrics.trusted({"c": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, {"c": _single("c", island_id)})

    result = perturb(_item(UncertainKind.SIM_ERROR, "sim_error:HW/b", 0.02), 0.31,
                     metrics, context)
    assert result.sim_error_override == pytest.approx(0.31)
    assert result.metrics["c"].model_dump() == metrics["c"].model_dump()
    assert result.approximation is False


# --- LINK_BW / LINK_LAT: checked against the planner's own term ------------

class _PdWorld(NamedTuple):
    spec: object
    cluster: object
    islands: dict
    candidate: object
    context: PerturbContext
    path: list


def _pd_world(world) -> _PdWorld:
    """A P/D candidate whose prefill->decode path crosses a known fabric link."""
    spec, cluster, _profiles, islands = world
    rngd = next(i for i in islands if i.startswith("furiosa-rngd-node_rngd0"))
    a40 = next(i for i in islands if i.startswith("cuda-a40-node_a40a"))
    candidate = _pd("pd", rngd, a40)
    context = PerturbContext.build(spec, cluster, islands, {"pd": candidate})
    endpoints = (
        f"{islands[rngd].node_id}/{islands[rngd].accelerator_ids[0]}",
        f"{islands[a40].node_id}/{islands[a40].accelerator_ids[0]}",
    )
    path = context.topology.path(*endpoints)
    return _PdWorld(spec, cluster, islands, candidate, context, path)


def test_link_bandwidth_reprices_the_transfer_exactly(world) -> None:
    """The shift must equal what `apply_pd_transfer_cost` would have added."""
    w = _pd_world(world)
    link = next(hop for hop in w.path if hop.id.startswith("fabric-"))

    metrics = JudgedMetrics.trusted({"pd": _metrics()})
    result = perturb(
        _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps),
        13.0, metrics, w.context,
    )
    assert result.approximation is False, "the planner owns this term; it is exact"
    assert result.affected == ["pd"]

    kv_per_token = kv_transfer._kv_bytes_per_token(
        w.spec.model, w.spec.service.dtype, w.spec.service.kv_cache_dtype
    )
    moved = [
        hop.model_copy(update={"bandwidth_gbps": 13.0}) if hop.id == link.id else hop
        for hop in w.path
    ]
    tok = w.spec.traffic.input_tokens
    prompts = (tok.p50,
               tok.p95 if tok.p95 is not None else tok.p50,
               tok.p99 if tok.p99 is not None else tok.p50)
    for prompt, field in zip(prompts, ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms"),
                             strict=True):
        expected = (
            kv_transfer.transfer_ms(
                kv_per_token * prompt,
                bandwidth_gbps=TopologyGraph.effective_bandwidth_gbps(moved),
                latency_ns=TopologyGraph.path_latency_ns(moved))
            - kv_transfer.transfer_ms(
                kv_per_token * prompt,
                bandwidth_gbps=TopologyGraph.effective_bandwidth_gbps(w.path),
                latency_ns=TopologyGraph.path_latency_ns(w.path))
        )
        assert getattr(result.metrics["pd"], field) == pytest.approx(
            getattr(metrics["pd"], field) + expected
        )


def test_a_slower_link_can_only_raise_ttft_and_never_tpot(world) -> None:
    w = _pd_world(world)
    link = next(hop for hop in w.path if hop.id.startswith("fabric-"))
    metrics = JudgedMetrics.trusted({"pd": _metrics()})

    slower = perturb(
        _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps),
        link.bandwidth_gbps / 3.0, metrics, w.context,
    ).metrics["pd"]
    faster = perturb(
        _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps),
        link.bandwidth_gbps * 3.0, metrics, w.context,
    ).metrics["pd"]

    assert slower.p99_ttft_ms > metrics["pd"].p99_ttft_ms > faster.p99_ttft_ms
    # The transfer happens once, before decode begins.
    assert slower.p99_tpot_ms == metrics["pd"].p99_tpot_ms
    assert slower.total_energy_j == metrics["pd"].total_energy_j


def test_link_latency_adds_its_change_per_traversal(world) -> None:
    w = _pd_world(world)
    link = next(hop for hop in w.path if hop.id.startswith("fabric-"))
    metrics = JudgedMetrics.trusted({"pd": _metrics()})
    bump_ns = 5_000.0

    result = perturb(
        _item(UncertainKind.LINK_LAT, f"link_lat:{link.id}", link.latency_ns),
        link.latency_ns + bump_ns, metrics, w.context,
    )
    hops = sum(1 for hop in w.path if hop.id == link.id)
    expected = bump_ns * hops / 1e6
    for field in ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms"):
        assert getattr(result.metrics["pd"], field) == pytest.approx(
            getattr(metrics["pd"], field) + expected
        )


def test_a_single_island_candidate_has_no_transfer_to_reprice(world) -> None:
    """No P/D handoff, so a link bandwidth reaches none of its metrics."""
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    candidates = {"solo": _single("solo", island_id)}
    metrics = JudgedMetrics.trusted({"solo": _metrics()})
    context = PerturbContext.build(spec, cluster, islands, candidates)
    link = cluster.links[0]

    result = perturb(
        _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps),
        1.0, metrics, context,
    )
    assert result.affected == []
    assert result.metrics["solo"].model_dump() == metrics["solo"].model_dump()


def test_a_pd_candidate_routed_off_this_link_is_untouched(world) -> None:
    w = _pd_world(world)
    off_path = next(
        link for link in w.cluster.links if all(hop.id != link.id for hop in w.path)
    )
    metrics = JudgedMetrics.trusted({"pd": _metrics()})
    result = perturb(
        _item(UncertainKind.LINK_BW, f"link_bw:{off_path.id}", off_path.bandwidth_gbps),
        1.0, metrics, w.context,
    )
    assert result.affected == []


def test_an_unknown_link_is_reported_not_guessed(world) -> None:
    spec, cluster, _profiles, islands = world
    island_id = next(iter(islands))
    context = PerturbContext.build(spec, cluster, islands, {"c": _single("c", island_id)})
    result = perturb(
        _item(UncertainKind.LINK_BW, "link_bw:no-such-link", 35.0),
        13.0, JudgedMetrics.trusted({"c": _metrics()}), context,
    )
    assert result.affected == []
    assert "no link" in result.note


# --- purity ---------------------------------------------------------------

def test_perturbation_never_mutates_its_input(world) -> None:
    w = _pd_world(world)
    link = next(hop for hop in w.path if hop.id.startswith("fabric-"))
    original = _metrics()
    metrics = JudgedMetrics.trusted({"pd": original})
    before = original.model_dump()
    before_link_bw = link.bandwidth_gbps

    for item, value in (
        (_item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps), 1.0),
        (_item(UncertainKind.PROFILE, "profile:x", 1.0, list(w.islands)), 2.0),
        (_item(UncertainKind.POWER, "power:x", 1.0, list(w.islands)), 2.0),
    ):
        result = perturb(item, value, metrics, w.context)
        assert result.metrics is not metrics, "a new mapping, not the caller's"
        assert original.model_dump() == before, "the metrics object was mutated"
        assert metrics["pd"] is original, "the caller's mapping was rebound"

    # The w.cluster the w.context holds must also be untouched: the rules copy links.
    assert link.bandwidth_gbps == before_link_bw


def test_a_plain_dict_is_refused_rather_than_silently_mispriced(world) -> None:
    """§2.7 asks for a type, not a docstring, and this is why.

    The envelope cache holds the RAW simulator output, from before the P/D
    transfer term is added. The LINK_BW rule adds the DIFFERENCE between two
    transfer times, so raw metrics would get a difference added to a number that
    never contained the original - silently, and only for P/D candidates.
    """
    w = _pd_world(world)
    link = next(hop for hop in w.path if hop.id.startswith("fabric-"))
    item = _item(UncertainKind.LINK_BW, f"link_bw:{link.id}", link.bandwidth_gbps)

    with pytest.raises(TypeError, match="needs JudgedMetrics"):
        perturb(item, 13.0, {"pd": _metrics()}, w.context)


def test_judged_metrics_is_built_from_what_the_planner_judged(spec, cluster,
                                                              islands, profiles) -> None:
    """The supported way in: every candidate a SearchResult reached, and only those."""
    from planner.optimizer import exhaustive
    from planner.uncertainty.perturb import judged_metrics

    from .conftest import MockPredictor

    generation = __import__(
        "planner.candidate_generator", fromlist=["CandidateGenerator"]
    ).CandidateGenerator(spec, cluster, islands, profiles).generate()
    evaluation = exhaustive.evaluate_candidates(
        generation.candidates, spec, cluster, {i.id: i for i in islands}, profiles,
        MockPredictor(),
    )
    metrics = judged_metrics(evaluation)
    assert isinstance(metrics, JudgedMetrics)
    judged = (
        {p.candidate.id for p in evaluation.feasible_plans}
        | {p.candidate.id for p, _ in evaluation.infeasible_plans}
        | set(evaluation.unmeasured_metrics)
    )
    assert set(metrics) == judged and metrics
