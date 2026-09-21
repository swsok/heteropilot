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


# --- compared_metric (S7.3) ------------------------------------------------
#
# `WORK_ORDER_domain_scoping.md` S7.3. The margin is applied to a p99 (D101)
# while every domain committed before 2026-09-18 was fitted on a p50 and says
# nothing about it. A domain that DECLARES p50 gets a warning; silence does not,
# because warning on silence would fire on every run and tell nobody anything.


def _domain_with(**kw):
    from planner.predictor.calibration import AccuracyDomain

    base = {
        "fitted_at_concurrency": 10.0,
        "points": [{"conc": 5.0, "tpot_err_pct": -10.0},
                   {"conc": 50.0, "tpot_err_pct": -20.0}],
        "outside_domain": "widen_error_bars",
        "hardware": "RNGD-CARD",
    }
    return AccuracyDomain(**{**base, **kw})


def test_a_domain_may_record_the_percentile_it_was_fitted_on():
    d = _domain_with(compared_metric="tpot_p99")
    assert d.compared_metric == "tpot_p99"
    # Unstated is "" and NOT p50: stating it for the committed domains is a
    # migration with its own measurement question, not a field default.
    assert _domain_with().compared_metric == ""


def test_a_point_may_carry_the_p50_error_beside_the_fitted_one():
    """Recorded, never consulted: `tpot_err_pct` is what gets interpolated."""
    d = _domain_with(points=[{"conc": 5.0, "tpot_err_pct": -18.0,
                              "tpot_err_pct_p50": -11.0},
                             {"conc": 50.0, "tpot_err_pct": -20.0}])
    assert d.points[0].tpot_err_pct_p50 == -11.0
    assert d.points[1].tpot_err_pct_p50 is None
    # The interpolation reads tpot_err_pct, whatever the p50 column says.
    assert d.tpot_error_at(5.0) == -18.0


def test_an_unknown_domain_field_is_still_refused():
    """`extra=forbid` is what makes a typo in a committed yaml loud, and adding
    two fields must not have loosened it."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _domain_with(comparedmetric="tpot_p99")


def test_a_domain_that_declares_p50_warns_but_is_still_consulted():
    """D101's mismatch, made visible where a reader will see it.

    The margin is applied to a p99. A domain fitted on a p50 is being charged on
    a basis it was not measured on -- but it IS a measurement of this hardware at
    this operating point, so refusing it would throw away real information. The
    warning goes in the basis, beside the closed-loop one it is modelled on.
    """
    policy = AccuracyDomainMargin(
        {HW: _domain(compared_metric="tpot_p50")}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=1), conc=20.0)

    assert not decision.is_unmeasured, "a basis mismatch is a warning, not a refusal"
    assert decision.tpot_percent > 0.0
    assert "fitted on a p50" in decision.basis
    assert "D101" in decision.basis


def test_a_domain_fitted_on_p99_does_not_warn():
    policy = AccuracyDomainMargin(
        {HW: _domain(compared_metric="tpot_p99")}, shape=SHAPE)
    assert "fitted on a p50" not in _decide(
        policy, _candidate(tp=1), conc=20.0).basis


def test_silence_about_the_basis_does_not_warn():
    """Every committed domain is p50 in fact and says nothing. Warning on silence
    would fire on every run while telling nobody anything new, and stating it for
    them is a migration with its own measurement question."""
    policy = AccuracyDomainMargin({HW: _domain()}, shape=SHAPE)
    assert "fitted on a p50" not in _decide(
        policy, _candidate(tp=1), conc=20.0).basis


def test_the_index_copies_the_basis_rather_than_defaulting_it():
    """A field the entry carries but `of()` never fills is worse than no field:
    the drift test cannot see it, because stored and rebuilt are both empty."""
    from planner.predictor.calibration import DomainIndexEntry

    entry = DomainIndexEntry.of("p.yaml", HW, _domain(compared_metric="tpot_p99"))
    assert entry.compared_metric == "tpot_p99"
    assert "compared_metric=tpot_p99" in entry.conditions_summary()
    # Unstated reads as unstated in the summary too, not as a silent p50.
    assert "compared_metric=not stated" in (
        DomainIndexEntry.of("p.yaml", HW, _domain()).conditions_summary())


def test_the_index_regenerator_refuses_a_domain_it_has_no_note_for(tmp_path):
    """The note is the one field a generator cannot derive.

    `index.yaml` says "Regenerate rather than hand-edit" and until S7.3 nothing
    could -- `build_domain_index` existed for the drift test but no command wrote
    its output. The gap matters precisely when a domain is added, so the tool
    that closes it must not paper over the one thing it cannot know.
    """
    import importlib.util
    import subprocess
    import sys as _sys

    script = ROOT / "experiments/scripts/rebuild_domain_index.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("rebuild_domain_index_ut", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # The leading comment block is preserved verbatim: it carries D102's
    # explanation, which no field encodes.
    header = mod.leading_comment(ROOT / "profiles/calibration/index.yaml")
    assert header.startswith("#")
    assert "D102" in header

    # --check is non-destructive and reports rather than writes.
    before = (ROOT / "profiles/calibration/index.yaml").read_bytes()
    subprocess.run([_sys.executable, str(script), "--check"],
                   cwd=ROOT, capture_output=True, timeout=120)
    assert (ROOT / "profiles/calibration/index.yaml").read_bytes() == before


# --- (vii) S6: the real A40 domain against V3's real P1 ----------------------
#
# Everything above uses a synthetic two-point domain so the mechanism can be
# exercised in isolation. These two pin the mechanism against the COMMITTED
# `profiles/calibration/a40.accuracy.yaml` at V3's actual operating point,
# because that pairing is what S6 leaves standing: the deployment was measured
# (v3_verdict_accuracy.md A.3, three repeats, NUMA-bound) and no A40 domain is
# fitted at its tp, so the residual is stated rather than margined.

#: V3's P1: one A40 island, tp=4, measured served concurrency 162.26.
V3_P1_CONCURRENCY = 162.26


def test_the_committed_a40_domain_refuses_v3s_p1_rather_than_margining_it():
    """The real file, the real operating point, the real refusal.

    162.26 is INSIDE the committed domain's load axis [4.043, 170.56], so the
    operating-point test passes and the old code would have handed back a
    margin — it did, and charged 1.13 % where the measured error was -44.63 %.
    What disqualifies it is tp, and `required_measurement` names the experiment
    that would qualify it.
    """
    real = load_accuracy_domains(ROOT)["A40"]
    assert real.parallelism.tp == 1, "the committed A40 domain is fitted at tp=1"
    assert real.in_domain(V3_P1_CONCURRENCY), "so this is not the load-axis refusal"

    policy = AccuracyDomainMargin({HW: real}, shape=SHAPE)
    decision = _decide(policy, _candidate(tp=4), conc=V3_P1_CONCURRENCY)

    assert decision.is_unmeasured
    assert decision.refusal == "condition_mismatch"
    assert decision.mismatch_fields == ["tp"]
    assert decision.required_measurement["tp"] == 4
    # No margin leaks through the refusal: unmeasured is not "margined by zero
    # and carried on", it is a candidate the search may not rank on this basis.
    assert decision.tpot_percent == 0.0 and decision.ttft_percent == 0.0


def test_warn_margins_v3s_p1_but_by_five_times_too_little():
    """Why S6 closes by STATING the residual instead of margining it.

    `--condition-mismatch warn` is the control arm: it applies the tp=1 domain
    across the mismatch and says so. The number it produces is the point. With
    the link priced from S3's measurement the simulator's residual error on P1
    is -7.01 % (v3_addendum_s4.md §3); the tp=1 domain consulted across the
    mismatch charges about 1.38 %. Off by ~5x, in the unsafe direction.

    So the residual is NOT margined by anything today, and this pins that it
    cannot be by reaching for the domain that exists. Closing it needs an A40
    domain fitted at tp=4 — a fit, not a single point, since one residual at one
    operating point has no slope to widen along. Skeleton in HANDOVER §2.12.
    """
    real = load_accuracy_domains(ROOT)["A40"]
    policy = AccuracyDomainMargin(
        {HW: real}, shape=SHAPE, condition_mismatch="warn"
    )
    decision = _decide(policy, _candidate(tp=4), conc=V3_P1_CONCURRENCY)

    assert not decision.is_unmeasured, "warn is the arm that does apply it"
    assert decision.mismatch_fields == ["tp"], "and still reports what it crossed"
    assert "APPLIED ACROSS" in decision.basis, "never silently"

    charged = decision.tpot_percent
    assert 1.3 < charged < 1.5, f"the tp=1 domain charges ~1.38 %, got {charged}"
    measured_residual = 7.01          # v3_addendum_s4.md §3, sim @ 8.8 measured
    assert measured_residual / charged > 4.0, (
        "if this ratio ever falls near 1 the domain has become applicable and "
        "this test is the wrong guard — check whether a tp=4 fit landed"
    )
