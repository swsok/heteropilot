"""Switchover is observed; crossover is estimated (STEP 4.5).

`docs/rps_aware_planning_design.md` §6. The deliverable is not "the best plan" but
"the rate at which the best plan changes", and the two things the table reports
have different epistemic standing:

  * a **switchover** is a change of winner between two rates that were both RUN;
  * a **crossover** is where two tok/J curves would meet, interpolated between
    those rates. Nothing was run there, so it is labelled `estimated` wherever it
    appears -- and if either endpoint lacks tok/J, no rate is invented.
"""

from __future__ import annotations

from planner.sweep import SweepOutput, SwitchoverRow, find_crossovers, render


def _row(rps, backend, tokj=None, feasible=True, validity="measured"):
    return SwitchoverRow(
        rps=rps, feasible=feasible, plan_id=f"hp-{int(rps * 10):05d}",
        candidate_id=f"cand-{backend}-{rps}", backend_mix=backend,
        accelerators=4, tokens_per_joule=tokj, average_power_w=500.0,
        p99_tpot_ms=40.0, applied_tpot_margin_pct=3.0, validity=validity,
    )


def test_no_crossover_when_one_backend_wins_everywhere():
    rows = [_row(1, "cuda", 1.0), _row(3, "cuda", 2.0), _row(10, "cuda", 3.0)]
    assert find_crossovers(rows) == []


def test_a_backend_change_between_adjacent_rates_is_a_crossover():
    rows = [_row(1, "furiosa", 4.0), _row(10, "cuda", 2.0)]
    got = find_crossovers(rows)
    assert len(got) == 1
    c = got[0]
    assert (c.from_backend, c.to_backend) == ("furiosa", "cuda")
    assert c.between_rps == (1.0, 10.0)
    assert c.label == "estimated"


def test_the_crossing_rate_is_never_presented_as_measured():
    rows = [_row(1, "furiosa", 4.0), _row(10, "cuda", 2.0)]
    c = find_crossovers(rows)[0]
    assert c.estimated_rps is not None
    assert 1.0 <= c.estimated_rps <= 10.0
    assert c.label == "estimated"
    assert "estimated" in render(SweepOutput(
        service_model="m", cluster_id="c", rps_values=[1, 10],
        switchover=rows, crossovers=[c]))


def test_no_rate_is_invented_when_a_row_lacks_tokens_per_joule():
    rows = [_row(1, "furiosa", None), _row(10, "cuda", 2.0)]
    c = find_crossovers(rows)[0]
    assert c.estimated_rps is None, "a missing efficiency must not become a midpoint"


def test_infeasible_rows_do_not_produce_crossovers():
    rows = [_row(1, "furiosa", 4.0), _row(3, "cuda", None, feasible=False),
            _row(10, "furiosa", 3.0)]
    assert find_crossovers(rows) == []


def test_the_rendered_table_carries_the_validity_label():
    rows = [_row(1, "cuda", 1.0, validity="measured"),
            _row(10, "cuda", 3.0, validity="extrapolated")]
    text = render(SweepOutput(service_model="m", cluster_id="c",
                              rps_values=[1, 10], switchover=rows))
    assert "measured" in text and "extrapolated" in text, (
        "a switchover table read without its validity labels is the D22 mistake"
    )


def test_an_infeasible_rate_says_why():
    rows = [SwitchoverRow(rps=20.0, feasible=False,
                          rejected_summary={"outside_measured_envelope": 12})]
    text = render(SweepOutput(service_model="m", cluster_id="c",
                              rps_values=[20], switchover=rows))
    assert "INFEASIBLE" in text and "outside_measured_envelope=12" in text


def test_the_output_round_trips_through_yaml():
    import yaml
    rows = [_row(1, "furiosa", 4.0), _row(10, "cuda", 2.0)]
    s = SweepOutput(service_model="m", cluster_id="c", rps_values=[1, 10],
                    switchover=rows, crossovers=find_crossovers(rows))
    back = SweepOutput.model_validate(yaml.safe_load(yaml.safe_dump(
        s.model_dump(mode="json"))))
    assert back.switchover[0].candidate_id == rows[0].candidate_id
    assert back.crossovers[0].label == "estimated"


def test_rendering_is_deterministic():
    """§9 reproducibility: the same sweep renders byte-identically."""
    rows = [_row(1, "furiosa", 4.0), _row(10, "cuda", 2.0)]
    s = SweepOutput(service_model="m", cluster_id="c", rps_values=[1, 10],
                    switchover=rows, crossovers=find_crossovers(rows))
    assert render(s) == render(s)


def test_crossover_endpoints_are_the_rates_that_were_actually_run():
    rows = [_row(1, "furiosa", 4.0), _row(3.3, "furiosa", 3.0), _row(10, "cuda", 2.0)]
    c = find_crossovers(rows)[0]
    assert c.between_rps == (3.3, 10.0), (
        "the bracket must be the adjacent RUN rates, not the ends of the sweep"
    )
