"""Per-candidate margins and the unmeasured verdict (uncertainty work order A3).

The margin policy reads `calibration.AccuracyDomain` - one curve per hardware,
the rps STEP 4 schema - at the served concurrency `SimResult.operating_point`
reports for that hardware, and returns ``unmeasured`` where a domain under
`refuse` does not reach (deviations D33).

The D22 reproduction near the bottom is the evidence behind the patent's §5.7
worked example, so it is written against the numbers docs/deviations.md
actually records rather than against round figures. Signs follow
`AccuracyPoint`: `(sim - measured) / measured * 100`, NEGATIVE where the
simulator is optimistic, and `margin_from_error` charges only that direction.
"""

from __future__ import annotations

import pytest

from planner.optimizer import exhaustive
from planner.optimizer.margin import AccuracyDomainMargin, GlobalMargin
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    PredictedMetrics,
    RejectionStage,
)
from planner.predictor import SimOutcome, SimResult
from planner.predictor.calibration import (
    AccuracyDomain,
    BucketError,
    CalibrationModel,
    ErrorStats,
    HardwareCalibration,
)

BUCKET = "in_lt1024-out_ge512-rps_lt20"   # canonical (§2.4.1)
SHAPE = "in_lt1024-out_ge512"
HW = "RNGD-CARD"
OUTSIDE = RejectionStage.OUTSIDE_CALIBRATION_DOMAIN.value

#: The two points D22 pins down: near-agreement where the profile was fitted,
#: and 18 % optimism at the load the card fixture actually uses.
D22_POINTS = ((16.6, -3.1), (76.0, -18.0))


def _domain(points=D22_POINTS, *, ttft=True, **over) -> AccuracyDomain:
    d = {
        "fitted_at_concurrency": points[0][0],
        "points": [
            {"conc": c, "tpot_err_pct": e, **({"ttft_err_pct": e} if ttft else {})}
            for c, e in points
        ],
    }
    d.update(over)
    return AccuracyDomain.model_validate(d)


def _candidate(cid: str, islands=("i0",)) -> CandidateConfig:
    return CandidateConfig(
        id=cid, model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[IslandAssignment(island_id=i, tp_size=1) for i in islands],
    )


def _metrics(*, tpot: float = 30.0) -> PredictedMetrics:
    return PredictedMetrics(
        p50_ttft_ms=100.0, p95_ttft_ms=200.0, p99_ttft_ms=300.0,
        p50_tpot_ms=tpot, p95_tpot_ms=tpot, p99_tpot_ms=tpot,
        throughput_tps=1000.0, slo_goodput_rps=5.0, slo_attainment=1.0,
        completed_requests=100, completed_tokens=20_000,
        total_energy_j=1000.0, tokens_per_joule=20.0,
    )


def _sim(points: dict[str, float] | None, *, phases: dict[str, str] | None = None) -> SimResult:
    """A SimResult whose operating point puts each hardware at a concurrency."""
    op = {} if points is None else {
        hw: {"concurrency": c, "phase": (phases or {}).get(hw, "total"),
             "requests": 100, "wall_s": 10.0}
        for hw, c in points.items()
    }
    return SimResult("c", SimOutcome.OK, metrics=_metrics(), operating_point=op)


def _decide(policy, points, *, islands=("i0",), phases=None, metrics=None):
    sim = _sim(points, phases=phases)
    return policy.decide(_candidate("c", islands), sim, metrics or sim.metrics, {})


# --- the three-candidate case from the work order -------------------------

CONCURRENCIES = (20.0, 60.0, 110.0)


def test_each_candidate_gets_the_margin_measured_at_its_own_operating_point() -> None:
    policy = AccuracyDomainMargin({HW: _domain()})
    decisions = [_decide(policy, {HW: c}) for c in CONCURRENCIES]

    assert [d.status for d in decisions] == ["in_domain", "in_domain", "unmeasured"]
    # The whole point: the first two differ, and the heavier one is harsher.
    assert decisions[0].tpot_percent != decisions[1].tpot_percent
    assert decisions[1].tpot_percent > decisions[0].tpot_percent

    # Linear between (16.6, -3.1) and (76.0, -18.0), charged one-sided.
    (lo_c, lo_e), (hi_c, hi_e) = D22_POINTS
    for decision, conc in zip(decisions[:2], CONCURRENCIES[:2], strict=True):
        weight = (conc - lo_c) / (hi_c - lo_c)
        assert decision.tpot_percent == pytest.approx(-(lo_e + weight * (hi_e - lo_e)))
    assert all(d.source == "accuracy_domain" for d in decisions[:2])
    assert decisions[0].operating_point[0].in_calibration_domain is True


def test_outside_the_domain_names_the_operating_point_and_the_range() -> None:
    decision = _decide(AccuracyDomainMargin({HW: _domain()}), {HW: 110.0})

    assert decision.is_unmeasured
    assert decision.concurrency == 110.0
    assert decision.ttft_percent == 0.0 and decision.tpot_percent == 0.0
    for fragment in ("110", "outside", HW, "16.6", "76", "refuse", "measure the envelope"):
        assert fragment in decision.basis


def test_widen_error_bars_extrapolates_instead_of_refusing() -> None:
    """The committed E5/E6 domains opt into this; the margin grows with distance."""
    policy = AccuracyDomainMargin({HW: _domain(outside_domain="widen_error_bars")})
    near = _decide(policy, {HW: 80.0})
    far = _decide(policy, {HW: 110.0})
    assert near.status == far.status == "extrapolated"
    assert far.tpot_percent > near.tpot_percent > 18.0
    assert far.extrapolated == {HW: 110.0}
    assert far.operating_point[0].in_calibration_domain is False
    assert "EXTRAPOLATED" in far.basis


def test_the_global_policy_reproduces_the_old_behaviour_exactly() -> None:
    policy = GlobalMargin(18.0, 18.0)
    for conc in CONCURRENCIES:
        decision = _decide(policy, {HW: conc})
        assert (decision.ttft_percent, decision.tpot_percent) == (18.0, 18.0)
        assert decision.status == "scalar"
        assert decision.source == "manual"
        assert not decision.is_unmeasured


def test_a_manual_floor_binds_where_it_exceeds_the_measured_margin() -> None:
    """rps STEP 4.3: the LARGER of floor and measured wins, and the source says which."""
    policy = AccuracyDomainMargin({HW: _domain()}, tpot_floor=10.0, ttft_floor=0.0)
    light = _decide(policy, {HW: 20.0})    # measured ~4 %, floor 10 % binds
    heavy = _decide(policy, {HW: 70.0})    # measured ~16.5 %, above the floor
    assert light.tpot_percent == 10.0 and light.source == "manual"
    assert "manual floor" in light.basis
    assert heavy.tpot_percent > 10.0 and heavy.source == "accuracy_domain"


# --- multi-hardware composition -------------------------------------------

def test_a_pd_candidate_charges_each_phase_to_the_hardware_that_owns_it() -> None:
    """Prefill hardware owns TTFT, decode hardware owns TPOT."""
    a40 = _domain(((10.0, -1.0), (120.0, -1.0)))
    policy = AccuracyDomainMargin({HW: _domain(), "A40": a40})
    decision = _decide(
        policy, {"A40": 60.0, HW: 60.0}, islands=("i0", "i1"),
        phases={"A40": "prefill", HW: "decode"},
    )
    assert decision.status == "in_domain"
    solo = _decide(AccuracyDomainMargin({HW: _domain()}), {HW: 60.0})
    assert decision.tpot_percent == pytest.approx(solo.tpot_percent)   # RNGD's TPOT
    assert decision.ttft_percent == pytest.approx(1.0)                  # A40's TTFT
    assert "A40[prefill]" in decision.basis and f"{HW}[decode]" in decision.basis
    assert len(decision.operating_point) == 2


def test_one_hardware_outside_its_domain_makes_the_whole_candidate_unmeasured() -> None:
    a40 = _domain(((10.0, -1.0), (20.0, -1.0)))
    policy = AccuracyDomainMargin({HW: _domain(), "A40": a40})
    decision = _decide(policy, {"A40": 60.0, HW: 60.0}, islands=("i0", "i1"))
    assert decision.is_unmeasured
    assert "A40" in decision.basis


# --- the gaps that used to pass as a zero margin --------------------------

def test_hardware_with_no_calibration_is_unmeasured_not_a_free_pass() -> None:
    """A hardware with no domain and no fitted bucket used to get margin 0,
    indistinguishable from "the simulator is exact". It is a missing measurement."""
    decision = _decide(AccuracyDomainMargin({HW: _domain()}), {"NEVER_FITTED": 30.0})
    assert decision.is_unmeasured
    assert "no accuracy domain for NEVER_FITTED" in decision.basis


def test_a_run_without_an_operating_point_is_unmeasured() -> None:
    decision = _decide(AccuracyDomainMargin({HW: _domain()}), None)
    assert decision.is_unmeasured and decision.unreadable
    assert decision.concurrency is None


def test_a_scalar_bucket_still_yields_a_margin_and_says_so() -> None:
    """Hardware with a fitted bucket but no domain falls back to the bucket."""
    calibration = CalibrationModel(hardware={HW: HardwareCalibration(
        hardware=HW,
        errors={BUCKET: BucketError(workload_bucket=BUCKET, label="synthetic",
                                    ttft=ErrorStats(mean_error=0.05),
                                    tpot=ErrorStats(mean_error=0.02))},
    )})
    policy = AccuracyDomainMargin({}, calibration=calibration, bucket=BUCKET)
    decision = _decide(policy, {HW: 10_000.0})
    assert decision.status == "scalar"
    assert decision.tpot_percent == pytest.approx(2.0)
    assert decision.ttft_percent == pytest.approx(5.0)
    assert "scalar mode" in decision.basis
    assert decision.no_domain == [HW]


def test_a_missing_bucket_lists_what_the_hardware_does_carry() -> None:
    calibration = CalibrationModel(hardware={HW: HardwareCalibration(
        hardware=HW,
        errors={BUCKET: BucketError(workload_bucket=BUCKET, tpot=ErrorStats(mean_error=0.02))},
    )})
    policy = AccuracyDomainMargin(
        {}, calibration=calibration, bucket="in_ge4096-out_lt128-rps_ge20"
    )
    decision = _decide(policy, {HW: 30.0})
    assert decision.is_unmeasured
    assert BUCKET in decision.basis


def test_a_pessimistic_simulator_earns_no_negative_margin() -> None:
    """The margin is one-sided: it only ever hardens a prediction."""
    policy = AccuracyDomainMargin({HW: _domain(((10.0, 30.0), (50.0, 10.0)))})
    decision = _decide(policy, {HW: 30.0})
    assert decision.status == "in_domain"
    assert (decision.ttft_percent, decision.tpot_percent) == (0.0, 0.0)


# --- §2.4.1: a domain is scoped to a token mix ----------------------------

def test_a_domain_measured_on_a_different_token_mix_is_refused() -> None:
    scoped = _domain(workload_shape="in_ge4096-out_lt128")
    decision = _decide(AccuracyDomainMargin({HW: scoped}, shape=SHAPE), {HW: 60.0})
    assert decision.is_unmeasured
    assert "in_ge4096-out_lt128" in decision.basis and SHAPE in decision.basis


def test_an_unscoped_domain_applies_to_any_shape_and_a_matching_one_is_used() -> None:
    unscoped = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    assert _decide(unscoped, {HW: 60.0}).status == "in_domain"
    scoped = AccuracyDomainMargin({HW: _domain(workload_shape=SHAPE)}, shape=SHAPE)
    assert _decide(scoped, {HW: 60.0}).status == "in_domain"


def test_a_non_canonical_shape_or_bucket_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="canonical"):
        AccuracyDomainMargin({}, shape="sharegpt")
    with pytest.raises(ValueError, match="canonical"):
        AccuracyDomainMargin({}, bucket="sharegpt-llama31-8b-20")


# --- partial coverage -----------------------------------------------------

def test_a_metric_with_no_point_is_unmeasured_not_zero() -> None:
    """A TPOT-only domain (the committed RNGD ones) margins TPOT and SAYS TTFT
    has no margin, instead of reading a 0 % TTFT error as 'the simulator is perfect'."""
    decision = _decide(AccuracyDomainMargin({HW: _domain(ttft=False)}), {HW: 60.0})
    assert decision.tpot_status == "in_domain"
    assert decision.ttft_status == "unmeasured"
    assert decision.unmeasured_metrics == ["ttft"]
    assert decision.is_partial and not decision.is_unmeasured
    assert "NO margin for ttft" in decision.basis


def test_a_closed_loop_domain_says_its_ttft_does_not_transfer() -> None:
    domain = _domain(ttft=False, arrival_process="closed_loop")
    decision = _decide(AccuracyDomainMargin({HW: domain}), {HW: 60.0})
    assert "closed-loop" in decision.basis and "D19" in decision.basis


def test_a_decode_leg_with_no_ttft_point_is_not_a_ttft_gap() -> None:
    """A metric is unmeasured only if a hardware that OWNS it lacks the point."""
    a40 = _domain(((10.0, -1.0), (120.0, -1.0)))            # has TTFT
    rngd = _domain(ttft=False)                               # TPOT only
    decision = _decide(
        AccuracyDomainMargin({HW: rngd, "A40": a40}), {"A40": 60.0, HW: 60.0},
        islands=("i0", "i1"), phases={"A40": "prefill", HW: "decode"},
    )
    assert decision.unmeasured_metrics == []


# --- D22 reproduction: the patent's §5.7 worked example --------------------

D22_PREDICTED_TPOT_MS = 48.41
D22_TPOT_SLO_MS = 50.0


def _spec_with_tpot_slo(max_ms: float):
    from planner.spec import load_service_spec

    spec = load_service_spec("examples/service_specs/llama31-8b.yaml")
    return spec.model_copy(
        update={
            "slo": spec.slo.model_copy(
                update={"tpot": spec.slo.tpot.model_copy(update={"max_ms": max_ms})}
            )
        }
    )


@pytest.mark.parametrize(
    ("concurrency", "expected_percent", "expected_robust_ms", "passes"),
    [
        (76.0, 18.0, 57.12, False),
        (16.6, 3.1, 49.91, True),
    ],
)
def test_d22_the_same_prediction_passes_or_fails_on_its_operating_point(
    concurrency: float, expected_percent: float, expected_robust_ms: float, passes: bool
) -> None:
    """One prediction, one SLO, two operating points, two verdicts.

    48.41 ms of predicted TPOT against a 50 ms SLO. At the concurrency the RNGD
    profile was fitted at the simulator is within ~3 %, so the plan holds; at the
    load the card fixture actually runs, it is 18 % optimistic and the same plan
    breaks the SLO. A single scalar margin cannot express both, which is the
    whole argument for indexing error by operating point.
    """
    from planner.optimizer import feasibility
    from planner.plan import DeploymentPlan

    metrics = _metrics(tpot=D22_PREDICTED_TPOT_MS)
    decision = _decide(AccuracyDomainMargin({HW: _domain()}), {HW: concurrency}, metrics=metrics)

    assert decision.status == "in_domain"
    assert decision.tpot_percent == pytest.approx(expected_percent, abs=0.05)

    robust = D22_PREDICTED_TPOT_MS * (1 + decision.tpot_percent / 100.0)
    assert robust == pytest.approx(expected_robust_ms, abs=0.02)

    plan = DeploymentPlan(
        plan_id="hp-0001", model="meta-llama/Llama-3.1-8B",
        candidate=_candidate("c"), predicted=metrics,
        robust_margin_tpot_percent=decision.tpot_percent,
    )
    report = feasibility.evaluate(
        plan, _spec_with_tpot_slo(D22_TPOT_SLO_MS),
        ttft_margin_percent=decision.ttft_percent,
        tpot_margin_percent=decision.tpot_percent,
    )
    tpot_violations = [v for v in report.violations if "tpot" in v.metric]
    if passes:
        assert not tpot_violations
    else:
        assert tpot_violations
        assert report.stage is RejectionStage.SLO_VIOLATED


def test_d22_a_scalar_margin_cannot_express_both_verdicts() -> None:
    """The counterfactual: no single percentage reproduces both rows above."""
    from planner.optimizer import feasibility
    from planner.plan import DeploymentPlan

    spec = _spec_with_tpot_slo(D22_TPOT_SLO_MS)

    def breaks_slo(percent: float) -> bool:
        plan = DeploymentPlan(
            plan_id="hp-0001", model="meta-llama/Llama-3.1-8B",
            candidate=_candidate("c"), predicted=_metrics(tpot=D22_PREDICTED_TPOT_MS),
        )
        report = feasibility.evaluate(
            plan, spec, ttft_margin_percent=0.0, tpot_margin_percent=percent
        )
        return any("tpot" in v.metric for v in report.violations)

    # 18 % rejects the light candidate too; 3.1 % lets the heavy one through.
    assert breaks_slo(18.0)
    assert not breaks_slo(3.1)


# --- end to end through the search ----------------------------------------

def _domains_for_every_island(
    domain: AccuracyDomain, islands, profiles
) -> dict[str, AccuracyDomain]:
    """The same domain under every hardware label the cluster actually uses.

    Without this a search-level test measures the wrong thing: an island whose
    hardware has no calibration is unmeasured for that reason alone, and the
    verdict rule under test never runs.
    """
    labels = {
        p.sim_hardware for island in islands
        if (p := profiles.get(island.accelerator_model)) and p.sim_hardware
    }
    return dict.fromkeys(labels, domain)


def test_search_charges_undecidable_candidates_to_outside_calibration_domain(
    spec, cluster, islands, profiles
) -> None:
    """Unmeasured must never be folded into SLO_VIOLATED.

    The mock predictor reports no operating point at all, so under an
    accuracy-domain policy every candidate is undecidable - which is the
    correct answer, and exactly the shape the stage exists to express.
    """
    from .conftest import MockPredictor

    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(),
        margin_policy=AccuracyDomainMargin({HW: _domain()}),
    )
    summary = output.rejected_summary
    assert summary.get(OUTSIDE, 0) > 0
    assert summary.get(RejectionStage.SLO_VIOLATED.value, 0) == 0
    assert not output.feasible
    assert output.closest_plan is None, "an undecidable candidate is not a near miss"
    assert any("could not be judged" in s for s in output.suggestions)
    assert any("undecidable, not infeasible" in s for s in output.suggestions)


def test_accuracy_domains_alone_build_the_policy(spec, cluster, islands, profiles) -> None:
    """The rps STEP 4 entry point - `accuracy_domains=` with no explicit policy -
    goes through the same decision, with the scalars as floors."""
    from .conftest import MockPredictor

    wide = _domain(((1.0, -0.5), (10_000.0, -0.6)))
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(served_concurrency=50.0),
        accuracy_domains=_domains_for_every_island(wide, islands, profiles),
        tpot_margin_percent=2.0,
    )
    assert output.recommended is not None
    plan = output.recommended.plan
    assert plan.robust_margin_tpot_percent == 2.0      # floor binds over ~0.5 %
    assert plan.margin_source == "manual"
    assert plan.margin_basis and "manual floor" in plan.margin_basis
    assert plan.operating_point and plan.operating_point[0].in_calibration_domain


def test_search_without_a_policy_is_unchanged(spec, cluster, islands, profiles) -> None:
    """Rule A4 at the search level: no policy, no new stage, no new field."""
    from .conftest import MockPredictor

    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert OUTSIDE not in output.rejected_summary
    assert output.recommended is not None
    assert output.recommended.plan.margin_basis is None
    assert output.recommended.plan.margin_source == "manual"


def test_margin_basis_is_recorded_once_a_policy_is_given(
    spec, cluster, islands, profiles
) -> None:
    from .conftest import MockPredictor

    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(),
        margin_policy=GlobalMargin(0.0, 0.0),
    )
    assert output.recommended is not None
    assert output.recommended.plan.margin_basis is not None
    assert "global margin" in output.recommended.plan.margin_basis


def test_margin_basis_is_absent_from_the_default_yaml(tmp_path, spec, cluster, islands,
                                                      profiles) -> None:
    import yaml

    from planner.__main__ import _write_output

    from .conftest import MockPredictor

    output = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    path = tmp_path / "plan.yaml"
    _write_output(output, path)
    text = path.read_text()
    assert "margin_basis" not in text
    assert "served_concurrency" not in text
    assert yaml.safe_load(text)["recommended"]["plan"]["plan_id"]


def test_an_all_undecidable_search_does_not_claim_nothing_was_simulated(
    spec, cluster, islands, profiles
) -> None:
    """The diagnosis must match what happened.

    `infeasible_plans` is empty in this case for the same reason it is empty
    when generation produced nothing, but the two are opposite situations: here
    candidates were generated AND simulated, and the missing thing is evidence,
    not a configuration.
    """
    from .conftest import MockPredictor

    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(),
        margin_policy=AccuracyDomainMargin({HW: _domain()}),
    )
    assert output.evaluated_candidates > 0
    assert "nothing was simulated" not in output.reason
    assert "undecidable" in output.reason
    assert str(output.rejected_summary[OUTSIDE]) in output.reason


def test_a_pass_that_rests_on_an_unmeasured_metric_is_kept_with_a_caveat(
    spec, cluster, islands, profiles
) -> None:
    """D33: partial coverage is a caveat, not a rejection.

    The committed domains are TPOT-only by construction (a closed-loop bench
    carries no comparable TTFT, D19). Rejecting every pass that lacks a TTFT
    margin would empty every search on them; the pass is kept and the search
    says which check ran unmargined.
    """
    from .conftest import MockPredictor

    entry = _domain(((1.0, -0.01), (10_000.0, -0.02)), ttft=False)
    output = exhaustive.search(
        spec, cluster, islands, profiles, MockPredictor(served_concurrency=50.0),
        margin_policy=AccuracyDomainMargin(_domains_for_every_island(entry, islands, profiles)),
    )
    assert output.feasible
    assert output.rejected_summary.get(OUTSIDE, 0) == 0
    assert any("NO measured ttft margin" in c for c in output.caveats)


def test_a_violation_is_a_real_verdict_whatever_the_coverage(
    spec, cluster, islands, profiles
) -> None:
    """Margins only inflate, so a violation needs no coverage to be trusted.

    With an impossible SLO every candidate violates it, and those must be
    charged to SLO_VIOLATED even though TTFT has no margin here - not swept into
    the unmeasured stage, which would hide a real infeasibility behind a
    missing measurement.
    """
    from .conftest import MockPredictor

    tight = spec.model_copy(
        update={
            "slo": spec.slo.model_copy(
                update={"ttft": spec.slo.ttft.model_copy(update={"max_ms": 0.001})}
            )
        }
    )
    entry = _domain(((1.0, -0.01), (10_000.0, -0.02)), ttft=False)
    output = exhaustive.search(
        tight, cluster, islands, profiles, MockPredictor(served_concurrency=50.0),
        margin_policy=AccuracyDomainMargin(_domains_for_every_island(entry, islands, profiles)),
        enable_bound_pruning=False,
    )
    assert not output.feasible
    assert output.rejected_summary.get(RejectionStage.SLO_VIOLATED.value, 0) > 0
    assert output.rejected_summary.get(OUTSIDE, 0) == 0
