"""Application conditions: a domain may not answer for a configuration it was
never measured under (domain-scoping work order S1, docs/deviations.md D110).

The case this exists for is V3's P1. An A40 island running tp=4 was margined by
`a40.accuracy.yaml`, a domain fitted at tp=1, which contributed **1.13 %** where
the measured simulator error turned out to be **-44.6 %**. Nothing was wrong
with the margin arithmetic and nothing was outside the domain's load axis: the
domain simply did not apply, and the code had no way to say so.

Two refusals, deliberately distinct, because they ask for different experiments:

  * `CALIBRATION_CONDITION_MISMATCH` - no domain was measured under THIS
    candidate's conditions, so none may be consulted. Measure at that
    configuration.
  * `OUTSIDE_CALIBRATION_DOMAIN` - the right domain exists and the operating
    point ran past the end of its measured load axis. Measure further along it.

Both are unmeasured, never infeasible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.optimizer.margin import AccuracyDomainMargin
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    PredictedMetrics,
    RejectionStage,
)
from planner.predictor import SimOutcome, SimResult
from planner.predictor.calibration import (
    AccuracyDomain,
    CandidateConditions,
    build_domain_index,
    load_accuracy_domains,
    load_domain_index,
)

ROOT = Path(__file__).resolve().parents[1]
HW = "A40"
SHAPE = "in_lt1024-out_ge512"

#: Two points, so the domain has a slope and a genuine interior.
POINTS = ((4.0, -1.0), (50.0, -18.0))


def _domain(**over) -> AccuracyDomain:
    d = {
        "fitted_at_concurrency": POINTS[0][0],
        "points": [
            {"conc": c, "tpot_err_pct": e, "ttft_err_pct": e} for c, e in POINTS
        ],
        "parallelism": {"tp": 1, "pp": 1, "dp": 1},
        "placement": {"islands": 1, "device_binding": "unknown"},
    }
    d.update(over)
    return AccuracyDomain.model_validate(d)


def _metrics() -> PredictedMetrics:
    return PredictedMetrics(
        p50_ttft_ms=100.0, p95_ttft_ms=200.0, p99_ttft_ms=300.0,
        p50_tpot_ms=30.0, p95_tpot_ms=30.0, p99_tpot_ms=30.0,
        throughput_tps=1000.0, slo_goodput_rps=5.0, slo_attainment=1.0,
        completed_requests=100, completed_tokens=20_000,
        total_energy_j=1000.0, tokens_per_joule=20.0,
    )


def _candidate(tp: int = 1, *, islands=("i0",), dp: int = 1) -> CandidateConfig:
    return CandidateConfig(
        id="c", model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id=i, tp_size=tp, dp_replicas=dp) for i in islands
        ],
    )


def _decide(policy, candidate, conc: float = 20.0):
    sim = SimResult(
        "c", SimOutcome.OK, metrics=_metrics(),
        operating_point={
            HW: {"concurrency": conc, "phase": "total", "requests": 100, "wall_s": 10.0}
        },
    )
    island_hw = {a.island_id: HW for a in candidate.assignments}
    return policy.decide(candidate, sim, sim.metrics, island_hw)


# --- (i) the V3 case: tp differs, the coordinate is fine ---------------------


def test_a_tp1_domain_refuses_a_tp4_candidate_even_well_inside_the_load_axis():
    """The refusal V3 needed and did not have.

    Concurrency 20 sits squarely inside [4, 50], so the operating-point test
    passes and would have handed back a margin. It is the CONFIGURATION that
    does not match, and that is checked first.
    """
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=4), conc=20.0)

    assert decision.is_unmeasured
    assert decision.refusal == "condition_mismatch"
    assert decision.mismatch_fields == ["tp"]
    assert decision.required_measurement == {
        "hardware": HW, "tp": 4, "pp": 1, "dp": 1, "islands": 1, "binding": "unknown",
    }
    # In domain on the load axis - so this is not that refusal wearing a new name.
    assert _domain().in_domain(20.0)
    assert "tp" in decision.basis and "domain 1" in decision.basis


def test_the_two_refusals_are_different_buckets():
    """End-to-end routing is covered by tests/test_cli_accuracy_domain.py; this
    pins the pair apart so neither can be quietly folded into the other."""
    assert (
        RejectionStage.CALIBRATION_CONDITION_MISMATCH
        is not RejectionStage.OUTSIDE_CALIBRATION_DOMAIN
    )
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    assert _decide(policy, _candidate(tp=4)).refusal == "condition_mismatch"
    assert _decide(policy, _candidate(tp=1), conc=500.0).refusal == "outside_domain"


# --- (ii) conditions match, the coordinate does not --------------------------


def test_a_matching_configuration_past_the_load_axis_keeps_the_old_reason():
    policy = AccuracyDomainMargin(
        {HW: _domain(outside_domain="refuse")}, shape=SHAPE
    )
    decision = _decide(policy, _candidate(tp=1), conc=500.0)

    assert decision.is_unmeasured
    assert decision.refusal == "outside_domain"
    assert decision.mismatch_fields == []
    assert decision.required_measurement is None
    assert "outside its accuracy domain" in decision.basis


# --- (iii) conditions match, coordinate inside: unchanged behaviour ----------


def test_a_matching_candidate_inside_the_domain_is_margined_as_before():
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=1), conc=20.0)

    assert not decision.is_unmeasured
    assert decision.status == "in_domain"
    assert decision.refusal == ""
    assert decision.mismatch_fields == []
    # The margin is the one the domain's own arithmetic gives, untouched by S1.
    err = _domain().tpot_error_at(20.0)
    assert decision.tpot_percent == pytest.approx(
        AccuracyDomain.margin_from_error(err)
    )


# --- (iv) an unstated condition is skipped and flagged, never guessed --------


def test_an_unknown_device_binding_is_not_checked_but_is_reported():
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=1))

    assert not decision.is_unmeasured
    assert "device_binding" not in decision.mismatch_fields
    assert any("device_binding" in w for w in decision.condition_warnings)
    assert "applied on trust" in decision.basis


def test_a_stated_binding_on_both_sides_IS_checked():
    """The other half of the same rule: stated on both sides, it is a condition.

    Worth 1.93x of throughput on the A40 node's second NUMA group
    (v3_verdict_accuracy.md A.3), so skipping it when it IS known would be the
    original bug in a quieter place.
    """
    domain = _domain(placement={"islands": 1, "device_binding": "numa_pinned"})
    policy = AccuracyDomainMargin(
        {HW: domain}, shape=SHAPE, device_binding={"i0": "unpinned"}
    )
    decision = _decide(policy, _candidate(tp=1))

    assert decision.refusal == "condition_mismatch"
    assert decision.mismatch_fields == ["device_binding"]
    assert decision.required_measurement["binding"] == "unpinned"


def test_islands_and_dp_are_compared_together():
    """A two-island candidate does not match a one-island domain.

    Kept separate from the D40 mirror problem on purpose: that is about two
    candidates the CACHE cannot tell apart; this is about a candidate and a
    MEASUREMENT that differ in how many islands were lit up.
    """
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=1, islands=("i0", "i1"), dp=2))

    assert decision.refusal == "condition_mismatch"
    assert decision.mismatch_fields == ["dp", "islands"]
    assert decision.required_measurement["islands"] == 2


def test_islands_of_one_hardware_at_different_parallelism_report_both():
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    candidate = CandidateConfig(
        id="c", model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[
            IslandAssignment(island_id="i0", tp_size=1),
            IslandAssignment(island_id="i1", tp_size=4),
        ],
    )
    decision = _decide(policy, candidate)

    assert decision.refusal == "condition_mismatch"
    required = decision.required_measurement
    assert required["per_island"] == [
        {"tp": 1, "pp": 1, "dp": 1}, {"tp": 4, "pp": 1, "dp": 1},
    ]
    # No single tp describes this hardware's share, so none is stated.
    assert "tp" not in required


# --- (v) the `warn` policy: through, but recorded ---------------------------


def test_warn_applies_the_domain_across_the_mismatch_and_says_so():
    policy = AccuracyDomainMargin(
        {HW: _domain()}, shape=SHAPE, condition_mismatch="warn"
    )
    decision = _decide(policy, _candidate(tp=4), conc=20.0)

    assert not decision.is_unmeasured
    assert decision.status == "in_domain"
    assert decision.mismatch_fields == ["tp"]
    assert "APPLIED ACROSS A CONDITION MISMATCH" in decision.basis
    # It really is the pre-S1 number, which is what makes `warn` a measurement
    # of the refusal's cost rather than a second policy.
    unscoped = AccuracyDomainMargin({HW: _domain(parallelism=None)}, shape=SHAPE)
    assert decision.tpot_percent == _decide(unscoped, _candidate(tp=4)).tpot_percent


# --- (vi) backward compatibility --------------------------------------------


def test_a_domain_with_no_condition_fields_still_loads_and_checks_nothing():
    """Every domain written before 2026-09-17 reads as "not stated"."""
    old = AccuracyDomain.model_validate({
        "fitted_at_concurrency": 4.0,
        "points": [{"conc": c, "tpot_err_pct": e} for c, e in POINTS],
    })
    assert old.parallelism is None and old.placement is None

    check = old.check_conditions(
        CandidateConditions(hardware=HW, tp=8, pp=2, dp=4, islands=3)
    )
    assert check.matched
    assert set(check.skipped) >= {"tp", "pp", "dp", "islands", "device_binding"}

    policy = AccuracyDomainMargin({HW: old}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=8), conc=20.0)
    assert not decision.is_unmeasured
    assert any("unchecked application conditions" in w
               for w in decision.condition_warnings)


def test_no_attribution_means_not_stated_rather_than_assumed():
    """With no island->hardware map the parallelism is unknown, not 1."""
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    candidate = _candidate(tp=4)
    sim = SimResult(
        "c", SimOutcome.OK, metrics=_metrics(),
        operating_point={HW: {"concurrency": 20.0, "phase": "total",
                              "requests": 100, "wall_s": 10.0}},
    )
    decision = policy.decide(candidate, sim, sim.metrics, {})
    assert not decision.is_unmeasured
    assert any("tp" in w for w in decision.condition_warnings)


# --- the committed domains and the registry ---------------------------------


def test_the_committed_domains_state_their_parallelism():
    """What S1 recorded, read back off the files rather than asserted in prose."""
    domains = load_accuracy_domains(ROOT)
    expected = {"A40": 1, "RNGD-CARD": 1, "RNGD": 8}
    for hardware, tp in expected.items():
        domain = domains[hardware]
        assert domain.parallelism is not None, hardware
        assert (domain.parallelism.tp, domain.parallelism.pp, domain.parallelism.dp) \
            == (tp, 1, 1), hardware
        assert domain.placement is not None and domain.placement.islands == 1

    # The A40 binding is `unpinned` on evidence, the NPU ones are `unknown` on
    # the absence of it. The difference is the point: one was established, the
    # other was not, and neither is a guess.
    assert domains["A40"].placement.device_binding == "unpinned"
    assert domains["RNGD-CARD"].placement.device_binding == "unknown"


def test_the_index_matches_the_domain_files():
    """`index.yaml` is a copy, and this is what stops it becoming a second truth."""
    stored = load_domain_index(ROOT)
    rebuilt = build_domain_index(ROOT)
    assert [e.path for e in stored.domains] == [e.path for e in rebuilt.domains]
    for a, b in zip(stored.domains, rebuilt.domains, strict=True):
        assert a.model_dump(exclude={"note"}) == b.model_dump(exclude={"note"})
        assert a.note, f"{a.path} has no note saying what it is"


def test_a_domain_filed_under_the_wrong_hardware_is_refused(tmp_path):
    from planner.predictor.calibration import (
        CalibrationModel,
        HardwareCalibration,
        save_calibration,
    )

    model = CalibrationModel(hardware={"A40": HardwareCalibration(
        hardware="A40", accuracy_domain=_domain(hardware="RNGD-CARD"),
    )})
    path = tmp_path / "wrong.yaml"
    save_calibration(model, path)
    with pytest.raises(ValueError, match="filed under"):
        load_accuracy_domains(tmp_path, [path])
