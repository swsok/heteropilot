"""The envelope refuses to be read past its ends (STEP 4.1).

`docs/rps_aware_planning_design.md` §2 and §9. Every property here is a
counter-measure to a specific way D22 went wrong:

  - `concurrency_metric: served` is mandatory, because the requested/served
    conflation is what the retraction was about;
  - `validity` is mandatory and `refuse` is the default, because the fatal step
    was reading a curve past the point where it had been measured;
  - `Saturated` carries a `kind`, because "the device cannot do this" and "nobody
    measured whether the device can do this" are different answers and D22
    collapsed them;
  - round-trip is exact at the measured points, because an interpolation that
    does not reproduce its own inputs is not reading the measurement.
"""

from __future__ import annotations

import pytest
import yaml

from planner.perf_envelope import (
    SATURATION_EXPONENT,
    EnvelopeError,
    OperatingPoint,
    PerfEnvelope,
    Saturated,
    find_envelope,
    load_envelope,
    solve_operating_point,
)

RNGD = "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"


def _raw(**over) -> dict:
    d = {
        "envelope_id": "T/m/bf16/tp1",
        "unit": "card",
        "source": "measured",
        "concurrency_metric": "served",
        "measured_on_workload": {"dataset": "d", "input_tokens_p50": 700,
                                 "output_tokens_p50": 600, "output_tokens_mean": 600},
        "points": [
            {"conc": 1.0, "tput_tok_s": 100.0, "tpot_p50": 10.0},
            {"conc": 2.0, "tput_tok_s": 180.0, "tpot_p50": 12.0},
            {"conc": 4.0, "tput_tok_s": 300.0, "tpot_p50": 15.0},
        ],
        "validity": {"conc_min": 1.0, "conc_max": 4.0},
    }
    d.update(over)
    return d


# --- the schema refuses what it cannot use ---------------------------------

def test_requested_concurrency_is_rejected():
    with pytest.raises(Exception, match="served"):
        PerfEnvelope.model_validate(_raw(concurrency_metric="requested"))


def test_a_file_without_validity_is_refused(tmp_path):
    raw = _raw()
    del raw["validity"]
    f = tmp_path / "e.yaml"
    f.write_text(yaml.safe_dump(raw))
    with pytest.raises(EnvelopeError, match="validity"):
        load_envelope(f)


def test_a_file_without_concurrency_metric_is_refused(tmp_path):
    raw = _raw()
    del raw["concurrency_metric"]
    f = tmp_path / "e.yaml"
    f.write_text(yaml.safe_dump(raw))
    with pytest.raises(EnvelopeError, match="concurrency_metric"):
        load_envelope(f)


def test_unsorted_points_are_refused():
    raw = _raw(points=[
        {"conc": 4.0, "tput_tok_s": 300.0},
        {"conc": 1.0, "tput_tok_s": 100.0},
    ])
    with pytest.raises(Exception, match="sorted"):
        PerfEnvelope.model_validate(raw)


def test_a_non_invertible_throughput_curve_is_refused():
    """Monotone throughput is what makes the solve unique; a dip would make the
    answer a choice of branch, and this module does not guess."""
    raw = _raw(points=[
        {"conc": 1.0, "tput_tok_s": 100.0},
        {"conc": 2.0, "tput_tok_s": 90.0},
        {"conc": 4.0, "tput_tok_s": 300.0},
    ])
    with pytest.raises(Exception, match="non-decreasing"):
        PerfEnvelope.model_validate(raw)


# --- interpolation ---------------------------------------------------------

def test_round_trip_is_exact_at_every_measured_point():
    """§9's <1 % requirement; the log-linear form reproduces its knots exactly."""
    env = load_envelope(RNGD)
    mean = env.measured_on_workload.output_tokens_mean
    assert mean is not None
    for pt in env.points:
        got = solve_operating_point(env, pt.tput_tok_s / mean, out_tokens=mean)
        assert isinstance(got, OperatingPoint), f"conc {pt.conc} came back {got}"
        assert got.concurrency == pytest.approx(pt.conc, rel=1e-6)


def test_reading_outside_the_measured_range_raises():
    env = PerfEnvelope.model_validate(_raw())
    with pytest.raises(EnvelopeError, match="outside the measured range"):
        env.throughput_at(0.5)
    with pytest.raises(EnvelopeError, match="outside the measured range"):
        env.throughput_at(8.0)


def test_a_field_measured_at_only_one_end_is_not_interpolated_across():
    """The RNGD curve has power below conc 15.59 and nothing above. Interpolating
    between a number and a hole would invent the hole's value."""
    env = load_envelope(RNGD)
    assert env.metric_at(4.0, "power_w") is not None
    assert env.metric_at(30.0, "power_w") is None, (
        "power was never measured above 15.59; returning a number there would be "
        "the A2 violation the null columns exist to prevent"
    )


def test_throughput_is_monotone_in_concurrency():
    env = load_envelope(RNGD)
    lo, hi = env.validity.conc_min, env.validity.conc_max
    xs = [lo + (hi - lo) * i / 40 for i in range(41)]
    tputs = [env.throughput_at(x) for x in xs]
    assert tputs == sorted(tputs)


# --- the solver ------------------------------------------------------------

def test_solved_concurrency_increases_with_rps():
    env = load_envelope(RNGD)
    mean = env.measured_on_workload.output_tokens_mean
    concs = []
    for rps in (0.15, 0.3, 0.6, 0.9, 1.4, 1.9):
        r = solve_operating_point(env, rps, out_tokens=mean)
        assert isinstance(r, OperatingPoint)
        concs.append(r.concurrency)
    assert concs == sorted(concs)


def test_past_the_top_of_a_still_climbing_curve_is_unmeasured():
    """RNGD's final interval has exponent 0.241 -- the curve had NOT flattened, so
    the honest answer is "we do not know", not "it cannot"."""
    env = load_envelope(RNGD)
    assert env.top_exponent > SATURATION_EXPONENT
    r = solve_operating_point(env, 5.0, out_tokens=652.51)
    assert isinstance(r, Saturated)
    assert r.kind == "unmeasured"
    assert "UNMEASURED, not impossible" in r.reason


def test_past_the_top_of_a_flattened_curve_is_genuine_saturation():
    raw = _raw(points=[
        {"conc": 1.0, "tput_tok_s": 100.0},
        {"conc": 2.0, "tput_tok_s": 190.0},
        {"conc": 8.0, "tput_tok_s": 191.0},     # flat: exponent ~0.004
    ], validity={"conc_min": 1.0, "conc_max": 8.0})
    env = PerfEnvelope.model_validate(raw)
    assert env.top_exponent < SATURATION_EXPONENT
    r = solve_operating_point(env, 10.0, out_tokens=600)
    assert isinstance(r, Saturated)
    assert r.kind == "genuine"
    assert "add replicas" in r.reason


def test_below_the_bottom_of_the_curve_is_also_unmeasured():
    env = load_envelope(RNGD)
    r = solve_operating_point(env, 0.01, out_tokens=652.51)
    assert isinstance(r, Saturated)
    assert r.kind == "unmeasured"
    assert "below the measured range" in r.reason


def test_saturated_is_returned_not_raised():
    """The planner's stance: infeasible is a diagnosis, not an error."""
    env = load_envelope(RNGD)
    r = solve_operating_point(env, 99.0, out_tokens=652.51)
    assert isinstance(r, Saturated) and r.ok is False


def test_the_solver_rejects_nonsense_inputs():
    env = PerfEnvelope.model_validate(_raw())
    with pytest.raises(EnvelopeError):
        solve_operating_point(env, 0.0, out_tokens=600)
    with pytest.raises(EnvelopeError):
        solve_operating_point(env, 1.0, out_tokens=0.0)


# --- workload conditionality ------------------------------------------------

def test_a_different_workload_is_reported_not_corrected():
    env = load_envelope(RNGD)
    assert env.workload_mismatch_pct(652.51) == pytest.approx(0.0, abs=1e-6)
    assert env.workload_mismatch_pct(900.0) == pytest.approx(37.9, abs=0.1)


# --- discovery --------------------------------------------------------------

def test_find_envelope_returns_none_for_hardware_without_one():
    assert find_envelope("NO-SUCH-HW", "meta-llama/Llama-3.1-8B", "bf16", 1) is None


def test_find_envelope_locates_the_committed_rngd_curve():
    env = find_envelope("RNGD-CARD", "meta-llama/Llama-3.1-8B", "bf16", 1)
    assert env is not None and len(env.points) == 9
