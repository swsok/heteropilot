"""`--resimulate-top`: exact endpoints instead of a first-order rule (§2.7, B4).

The expensive half - actually launching the simulator - is not exercised here.
What is exercised is everything around it: which kinds have a resimulation at
all, that a refusal is recorded rather than swallowed, that the refined ΔR is
scored with the same arithmetic as the closed form, and that a scaled perf
bundle is a real copy with only `time_us` moved.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer.margin import GlobalMargin
from planner.plan import (
    CandidateConfig,
    DeploymentPlan,
    IslandAssignment,
    PlannerOutput,
    PredictedMetrics,
    ScoredPlan,
)
from planner.spec import load_service_spec
from planner.uncertainty import resimulate as resim
from planner.uncertainty.perturb import JudgedMetrics, PerturbContext
from planner.uncertainty.registry import (
    Grade,
    MeasurementCost,
    Range,
    UncertainInput,
    UncertainInputRegistry,
    UncertainKind,
)
from planner.uncertainty.sensitivity import analyze, refine

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml"
SERVICE = ROOT / "examples/service_specs/llama31-8b.yaml"


def _metrics(*, energy: float, ttft: float = 500.0) -> PredictedMetrics:
    return PredictedMetrics(
        p50_ttft_ms=ttft * 0.6, p95_ttft_ms=ttft * 0.9, p99_ttft_ms=ttft,
        p50_tpot_ms=10.0, p95_tpot_ms=12.0, p99_tpot_ms=14.0,
        throughput_tps=1000.0, slo_goodput_rps=10.0, slo_attainment=1.0,
        completed_requests=300, completed_tokens=60_000,
        total_energy_j=energy, average_power_w=100.0, peak_power_w=150.0,
        tokens_per_joule=60_000 / energy, served_concurrency=42.0,
    )


class _World(NamedTuple):
    spec: object
    cluster: object
    islands: dict
    candidates: dict
    context: PerturbContext
    metrics: JudgedMetrics
    island_hw: dict
    a40: str


@pytest.fixture
def world() -> _World:
    spec = load_service_spec(SERVICE)
    cluster = load_cluster_spec(CLUSTER)
    profiles = load_profiles_for(cluster, ROOT)
    islands = {i.id: i for i in detect_islands(cluster, profiles)}
    a40 = next(i for i in islands if i.startswith("cuda-a40-node_a40a"))
    b40 = next(i for i in islands if i.startswith("cuda-a40-node_a40b"))
    candidates = {
        "A": CandidateConfig(
            id="A", model=spec.model, dtype="bfloat16",
            assignments=[IslandAssignment(island_id=a40, tp_size=4)],
        ),
        "B": CandidateConfig(
            id="B", model=spec.model, dtype="bfloat16",
            assignments=[IslandAssignment(island_id=b40, tp_size=4)],
        ),
    }
    metrics = JudgedMetrics.trusted({
        "A": _metrics(energy=2000.0),
        "B": _metrics(energy=1000.0),
    })
    return _World(
        spec, cluster, islands, candidates,
        PerturbContext.build(spec, cluster, islands, candidates),
        metrics, dict.fromkeys(islands, "A40"), a40,
    )


def _profile_item(world: _World) -> UncertainInput:
    return UncertainInput(
        id=f"profile:{world.a40}", kind=UncertainKind.PROFILE,
        grade=Grade.ANALYTICAL, nominal=1.0,
        range=Range(lo=1.0, hi=1.4, unit="fraction", source="tests"),
        affects=[world.a40],
        cost=MeasurementCost(method="m", hours=2.1, source="tests"),
    )


def _output(world: _World, recommended: str) -> PlannerOutput:
    plan = DeploymentPlan(
        plan_id="hp-1", model=world.spec.model,
        candidate=world.candidates[recommended],
        predicted=world.metrics[recommended],
    )
    return PlannerOutput(
        feasible=True, service_model=world.spec.model, cluster_id="c",
        recommended=ScoredPlan(
            plan=plan, objective=world.spec.objective.primary, value=0.0),
    )


# --- which kinds can be resimulated at all --------------------------------

def test_a_sim_error_has_nothing_to_simulate(world: _World) -> None:
    """Its closed form is exact by construction, so a run would add nothing."""
    item = UncertainInput(
        id="sim_error:x", kind=UncertainKind.SIM_ERROR, grade=Grade.MEASURED,
        nominal=0.1, range=Range(lo=0.0, hi=0.4, unit="fraction", source="tests"),
    )
    with pytest.raises(resim.NotResimulable, match="margin, not the prediction"):
        resim.resimulate(
            item, spec=world.spec, cluster=world.cluster, islands=world.islands,
            profiles={}, candidates=[], predictor=None, island_hw=world.island_hw,
        )


def test_an_unbounded_range_has_no_endpoints(world: _World) -> None:
    item = UncertainInput(
        id="link_bw:x", kind=UncertainKind.LINK_BW, grade=Grade.PLACEHOLDER,
        nominal=35.0, range=Range(unit="gbps", source="no sourced limit"),
    )
    with pytest.raises(resim.NotResimulable, match="unbounded"):
        resim.endpoints_for(item)


def test_a_link_endpoint_rewrites_only_that_link(world: _World) -> None:
    item = UncertainInput(
        id="link_bw:fabric-rngd0-a40a", kind=UncertainKind.LINK_BW,
        grade=Grade.PLACEHOLDER, nominal=35.0,
        range=Range(lo=13.0, hi=35.0, unit="gbps", source="tests"),
    )
    before = {link.id: link.bandwidth_gbps for link in world.cluster.links}
    moved = resim._relinked_cluster(item, 13.0, world.cluster)
    changed = {
        link.id: link.bandwidth_gbps for link in moved.links
        if link.bandwidth_gbps != before[link.id]
    }
    assert changed == {"fabric-rngd0-a40a": 13.0}


def test_an_unknown_link_is_refused_not_ignored(world: _World) -> None:
    item = UncertainInput(
        id="link_bw:no-such-link", kind=UncertainKind.LINK_BW,
        grade=Grade.PLACEHOLDER, nominal=35.0,
        range=Range(lo=13.0, hi=35.0, unit="gbps", source="tests"),
    )
    with pytest.raises(resim.NotResimulable, match="no link"):
        resim._relinked_cluster(item, 13.0, world.cluster)


# --- the scaled perf bundle ----------------------------------------------

def test_a_scaled_bundle_moves_time_us_and_nothing_else(tmp_path) -> None:
    source = resim.PERF_ROOT / "A40"
    if not source.is_dir():
        pytest.skip("no A40 perf bundle on this checkout")
    label = "A40-test-resim"
    dest = resim.scale_perf_bundle("A40", 2.0, label)
    try:
        original = next(source.rglob("dense.csv"))
        copy = dest / original.relative_to(source)
        with original.open() as a, copy.open() as b:
            before, after = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(before) == len(after)
        for old, new in zip(before[:50], after[:50], strict=True):
            assert new["layer"] == old["layer"]
            assert new["tokens"] == old["tokens"]
            assert float(new["time_us"]) == pytest.approx(
                float(old["time_us"]) * 2.0, rel=1e-6)
        # meta.yaml records HOW the bundle was profiled and must be carried over
        # untouched - a rewritten one would be a different measurement.
        metas = list(dest.rglob("meta.yaml"))
        assert metas
        for meta in metas:
            twin = source / meta.relative_to(dest)
            assert meta.read_bytes() == twin.read_bytes()
    finally:
        if dest.exists():
            import shutil
            shutil.rmtree(dest)


# --- refine() -------------------------------------------------------------

def test_a_refusal_is_recorded_and_does_not_use_a_slot(world: _World) -> None:
    """"could not check" and "checked, agreed" must not look the same."""
    item = _profile_item(world)
    registry = UncertainInputRegistry(items=[item])
    policy = GlobalMargin()
    sensitivities = analyze(
        _output(world, "B"), registry, world.metrics, policy, world.spec,
        world.context, world.island_hw, slo_penalty=2000.0,
    )

    def boom(_item):
        raise resim.NotResimulable("no bundle in this test")

    refined, records = refine(
        sensitivities, registry, boom, world.metrics, policy, world.spec,
        world.context, world.island_hw, "B", top=3, slo_penalty=2000.0,
    )
    assert [r.skipped for r in records] == ["no bundle in this test"]
    # Untouched: the closed form still stands, and it is still marked approximate.
    assert refined[0].delta_regret == sensitivities[0].delta_regret
    assert refined[0].approximation is True


def test_a_refined_item_takes_the_simulated_regret_and_drops_approximate(
    world: _World,
) -> None:
    """The stand-in is replaced, and the item stops claiming to be one."""
    item = _profile_item(world)
    registry = UncertainInputRegistry(items=[item])
    policy = GlobalMargin()
    sensitivities = analyze(
        _output(world, "B"), registry, world.metrics, policy, world.spec,
        world.context, world.island_hw, slo_penalty=2000.0,
    )

    class _Endpoint(NamedTuple):
        value: float
        metrics: JudgedMetrics
        seconds: float
        simulated: int

    class _Result(NamedTuple):
        lo: _Endpoint
        hi: _Endpoint

    # At `hi` the real simulator says A - the island this item affects - got so
    # slow that B wins by a mile; the first-order rule never moves energy at all.
    def fake(_item):
        return _Result(
            lo=_Endpoint(1.0, world.metrics, 1.0, 2),
            hi=_Endpoint(1.4, JudgedMetrics.trusted({
                "A": _metrics(energy=9000.0),
                "B": _metrics(energy=1000.0),
            }), 2.0, 2),
        )

    refined, records = refine(
        sensitivities, registry, fake, world.metrics, policy, world.spec,
        world.context, world.island_hw, "B", top=1, slo_penalty=2000.0,
    )
    assert len(records) == 1
    record = records[0]
    assert record.skipped == ""
    assert record.seconds == pytest.approx(3.0)
    assert record.simulated == 4
    assert refined[0].approximation is False
    assert refined[0].delta_regret == pytest.approx(record.resimulated)
    assert "RESIMULATED" in refined[0].note


def test_top_bounds_how_many_items_are_simulated(world: _World) -> None:
    a = _profile_item(world)
    b = a.model_copy(update={"id": "profile:other"})
    registry = UncertainInputRegistry(items=[a, b])
    policy = GlobalMargin()
    sensitivities = analyze(
        _output(world, "B"), registry, world.metrics, policy, world.spec,
        world.context, world.island_hw, slo_penalty=2000.0,
    )
    seen: list[str] = []

    class _E(NamedTuple):
        value: float
        metrics: JudgedMetrics
        seconds: float
        simulated: int

    class _R(NamedTuple):
        lo: _E
        hi: _E

    def fake(item):
        seen.append(item.id)
        return _R(_E(1.0, world.metrics, 0.0, 1), _E(1.4, world.metrics, 0.0, 1))

    refine(
        sensitivities, registry, fake, world.metrics, policy, world.spec,
        world.context, world.island_hw, "B", top=1, slo_penalty=2000.0,
    )
    assert len(seen) == 1
