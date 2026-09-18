"""The open-loop pairing must refuse a pair that did not run the same workload.

`WORK_ORDER_domain_scoping.md` S7.3, `docs/deviations.md` D115.

Two guards, and the second exists because the first did not fire. The pairing has
always checked that the two sides landed at the same SERVED CONCURRENCY; what it
could not see was that they were generating different numbers of tokens. The
replay client caps completions at `min(output_toks, --max-tokens-cap)` and that
cap defaulted to 512 against a trace whose p50 is 632 and whose max is 1021, so
the card delivered 78.6 % of the simulator's tokens. The concurrency guard read
that as an 18-37 % concurrency error and passed one of the two points anyway,
because a shorter workload lands at a lower concurrency rather than a wrong one.

`profiles/calibration/openloop/a40.accuracy.openloop.yaml` already stated the
invariant -- "the simulator generates exactly the trace's output_toks ... every
stage delivered 195 753 output tokens, the trace's exact total" -- so the fix is
to check what that file asserts rather than to invent a new rule.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/openloop_sim_error.py"


def _load():
    spec = importlib.util.spec_from_file_location("openloop_sim_error_ut", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ose = _load()


def _real(conc: float, *, tokens: int, ok: int = 300, tpot99: float = 40.0,
          tpot50: float = 34.0, ttft99: float = 500.0, saturated: bool = False):
    return {
        "offered_rps": 2.0, "served_concurrency": conc, "requests_ok": ok,
        "requests_total": ok, "output_tokens": tokens, "saturated": saturated,
        "tpot_ms": {"p50": tpot50, "p99": tpot99}, "ttft_ms": {"p99": ttft99},
        "numa_bind": "node0", "_source": "x.json",
        "ttft_drift_slope_s_per_s": 0.0,
    }


def _sim(conc: float, tpot99: float = 38.0, ttft99: float = 420.0):
    return {"served_concurrency": conc, "p99_tpot_ms": tpot99,
            "p99_ttft_ms": ttft99}


def _pair(reals, sim, **kw):
    kw.setdefault("max_gap", 0.20)
    kw.setdefault("min_requests_p99", 300)
    return ose.pair(2.0, reals, sim, **kw)


def test_the_expected_token_count_comes_from_the_trace(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text("".join(
        f'{{"output_toks": {n}}}\n' for n in (632, 1021, 400)))
    assert ose.expected_output_tokens(trace, 0) == 2053
    assert ose.expected_output_tokens(trace, 2) == 1653


def test_a_truncated_run_is_refused_even_though_its_concurrency_looks_fine():
    """The exact shape of the bug: a short workload lands at a plausible
    concurrency, so the concurrency guard passes it."""
    expected = 195_753
    # 78.6 % of the tokens, and a concurrency only 8 % off -- inside the guard.
    rec = _pair([_real(35.0, tokens=153_900)], _sim(37.8),
                expected_tokens=expected)
    assert abs(rec["conc_gap_pct"]) < 20.0, "the concurrency guard would pass it"
    assert rec["usable"] is False
    assert any("output tokens" in n for n in rec["notes"])


def test_a_full_run_passes_the_token_check():
    expected = 195_753
    # Delivered = trace total + one SSE role chunk per request.
    rec = _pair([_real(38.0, tokens=expected + 300)], _sim(37.8),
                expected_tokens=expected)
    assert rec["usable"] is True
    assert rec["delivered_token_ratio"]["median"] == pytest.approx(1.0, abs=1e-6)


def test_the_role_chunk_is_subtracted_rather_than_widening_the_tolerance():
    """300 extra chunks on 195 753 tokens is 0.15 % -- inside any sane tolerance,
    which is exactly why it must not be absorbed by one: a real 0.15 % shortfall
    would then be invisible too."""
    expected = 195_753
    exact = _pair([_real(38.0, tokens=expected + 300)], _sim(37.8),
                  expected_tokens=expected, token_tol=0.0005)
    assert exact["usable"] is True


def test_the_concurrency_guard_still_fires_on_its_own():
    rec = _pair([_real(20.0, tokens=196_053)], _sim(38.0),
                expected_tokens=195_753)
    assert rec["usable"] is False
    assert any("served concurrency differs" in n for n in rec["notes"])


def test_without_a_dataset_the_token_check_is_simply_not_run():
    """It is opt-in, so a caller with no trace to hand still gets a pairing --
    but then nothing has checked the workloads matched, and the record says so by
    omitting the ratio."""
    rec = _pair([_real(38.0, tokens=0)], _sim(37.8))
    assert rec["usable"] is True
    assert "delivered_token_ratio" not in rec


def test_both_bases_are_formed_and_only_p99_has_a_simulated_counterpart():
    rec = _pair([_real(38.0, tokens=196_053, tpot99=40.0, tpot50=34.0)],
                _sim(37.8, tpot99=38.0), expected_tokens=195_753)
    assert rec["tpot_err_pct_p99"] == pytest.approx((38.0 - 40.0) / 40.0 * 100)
    # The S7.0 sim record carries p99 only; inventing a p50 is absolute rule 3.
    assert rec["tpot_err_pct_p50"] is None
    assert any("carries p99 only" in n for n in rec["notes"])


def test_repeats_are_medianed_and_their_spread_recorded():
    """S7.0 requires the run-to-run spread beside any verdict: a separation
    narrower than the spread is inside the noise."""
    reals = [_real(38.0, tokens=196_053, tpot99=t) for t in (39.0, 40.0, 44.0)]
    rec = _pair(reals, _sim(37.8), expected_tokens=195_753)
    assert rec["tpot_measured_p99"]["median"] == 40.0
    assert rec["tpot_measured_p99"]["spread"] == pytest.approx(5.0)
    assert rec["tpot_measured_p99"]["repeats"] == 3


def test_a_closed_loop_envelope_is_refused_by_name(tmp_path):
    """Pairing a closed-loop envelope here would be D19 with extra steps."""
    p = tmp_path / "envelope.json"
    p.write_text(json.dumps({"run": {"protocol": "closed_loop"}, "points": []}))
    with pytest.raises(SystemExit, match="open_loop"):
        ose.real_points([p])


def test_a_missing_token_count_is_not_silently_a_pass(tmp_path):
    """An artifact written before the invariant existed cannot answer for itself,
    and saying nothing would look exactly like saying yes."""
    rec = _pair([_real(38.0, tokens=196_053), {**_real(38.0, tokens=0),
                                               "output_tokens": None}],
                _sim(37.8), expected_tokens=195_753)
    assert any("COULD NOT BE CHECKED" in n for n in rec["notes"])
    assert "delivered_token_ratio" not in rec


def test_the_count_is_recovered_from_the_replay_report_beside_the_point(tmp_path):
    """Envelopes written before the point carried it keep the number next door."""
    (tmp_path / "replay_r2.json").write_text(json.dumps(
        {"summary": {"offered_rps": 2.0, "output_tokens": 196_053}}))
    (tmp_path / "envelope.json").write_text(json.dumps(
        {"run": {"protocol": "open_loop"},
         "points": [{"offered_rps": 2.0, "served_concurrency": 44.5}]}))
    got = ose.real_points([tmp_path / "envelope.json"])
    assert got[2.0][0]["output_tokens"] == 196_053


def test_several_sim_records_merge_by_rate(tmp_path):
    """S7.0 swept some rates and S7.3 added others; both are the same sweep."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"rows": [
        {"candidate_id": "x-s128-t2048", "rps": 1.5, "served_concurrency": 27.0,
         "p99_tpot_ms": 33.0, "p99_ttft_ms": 400.0}]}))
    b.write_text(json.dumps({"rows": [
        {"candidate_id": "x-s128-t2048", "rps": 1.75, "served_concurrency": 32.0,
         "p99_tpot_ms": 34.0, "p99_ttft_ms": 410.0}]}))
    got = ose.sim_points([a, b], "s128-t2048")
    assert sorted(got) == [1.5, 1.75]


def test_the_same_rate_simulated_twice_differently_is_an_error(tmp_path):
    """Silently preferring one would hide that they are different simulations."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for path, conc in ((a, 27.0), (b, 28.5)):
        path.write_text(json.dumps({"rows": [
            {"candidate_id": "x-s128-t2048", "rps": 1.5,
             "served_concurrency": conc, "p99_tpot_ms": 33.0,
             "p99_ttft_ms": 400.0}]}))
    with pytest.raises(SystemExit, match="more than one"):
        ose.sim_points([a, b], "s128-t2048")


# --- unpaired points are kept as evidence (S7.3, user direction) ------------


def _records_for_domain():
    """Three pairable points and two that are not, as the real sweep produced."""
    good = [
        _pair([_real(11.31, tokens=196_053, tpot99=27.548)], _sim(11.730, 28.106),
              expected_tokens=195_753),
        _pair([_real(28.71, tokens=196_053, tpot99=35.152)], _sim(27.002, 33.056),
              expected_tokens=195_753),
        _pair([_real(44.51, tokens=196_053, tpot99=42.895)], _sim(37.965, 35.332),
              expected_tokens=195_753),
    ]
    bad = [
        _pair([_real(73.997, tokens=196_053, tpot99=65.422)], _sim(49.527, 38.480),
              expected_tokens=195_753),
        _pair([_real(114.198, tokens=196_053, tpot99=89.422, saturated=True)],
              _sim(74.498, 46.691), expected_tokens=195_753),
    ]
    for i, r in enumerate(good + bad):
        r["offered_rps"] = [0.75, 1.5, 2.0, 2.5, 3.5][i]
    return good + bad


def test_an_unpaired_measurement_is_kept_as_evidence_not_dropped():
    """A rate the pairing cannot use is still a real measurement of the card, and
    it is the only evidence for where the domain stops."""
    doc = ose._domain_doc(_records_for_domain(), stat="p99", date="2026-09-18")
    assert "unpaired_points:" in doc
    assert "pairing: unpaired_served_l" in doc
    for field in ("served_l_real", "served_l_sim", "l_gap_pct",
                  "tpot_p99_measured_ms", "tpot_p99_sim_ms",
                  "tpot_p99_under_prediction_pct"):
        assert field in doc, field


def test_unpaired_measurements_never_reach_the_interpolation():
    """They live under `provenance`, which `_err_at` cannot see. A point at
    served concurrency 49.5 in `points` would be interpolated through."""
    import yaml

    doc = yaml.safe_load(
        ose._domain_doc(_records_for_domain(), stat="p99", date="2026-09-18"))
    domain = doc["hardware"]["RNGD-CARD"]["accuracy_domain"]
    concs = [p["conc"] for p in domain["points"]]
    assert concs == [11.730, 27.002, 37.965]
    unpaired = doc["provenance"]["unpaired_points"]
    assert [u["offered_rps"] for u in unpaired] == [2.5, 3.5]
    # And the saturated one says so, rather than the range being described as
    # steady-state wholesale.
    assert unpaired[0]["saturated"] is False
    assert unpaired[1]["saturated"] is True


def test_the_domain_loads_with_the_unpaired_block_present():
    """`AccuracyDomain` is extra=forbid, so the evidence has to sit outside it."""
    import yaml

    from planner.predictor.calibration import AccuracyDomain

    doc = yaml.safe_load(
        ose._domain_doc(_records_for_domain(), stat="p99", date="2026-09-18"))
    d = AccuracyDomain.model_validate(
        doc["hardware"]["RNGD-CARD"]["accuracy_domain"])
    assert d.compared_metric == "tpot_p99"
    assert d.in_domain(27.002) and not d.in_domain(49.527)
