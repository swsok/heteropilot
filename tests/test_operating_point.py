"""Served concurrency per hardware, including the P/D phase split (STEP 4.3).

The simulator's CSV records one row per request and books it against the instance
that COMPLETED it, so a disaggregated run attributes everything to the decode
engine and a naive grouping sees no prefill load at all. Little's law applies to
each phase separately, though, and `latency = TTFT + (latency - TTFT)` splits the
residence exactly -- so `sum(TTFT)/wall` is the prefill engine's occupancy.

That decomposition is the whole reason this module exists, and these tests pin it
with hand-built CSVs where the right answer is arithmetic rather than opinion.
"""

from __future__ import annotations

import pytest

from planner.util.operating_point import operating_points

NS = 1e9
HEADER = ("instance id,request id,model,input,output,arrival,end_time,latency,"
          "queuing_delay,TTFT,TPOT,ITL")


def _csv(tmp_path, rows):
    """rows: (instance, arrival_s, end_s, ttft_s)."""
    out = [HEADER]
    for i, (inst, arr, end, ttft) in enumerate(rows):
        lat = end - arr
        out.append(f"{inst},{i},m,100,10,{arr*NS},{end*NS},{lat*NS},0,{ttft*NS},1000,\"[]\"")
    p = tmp_path / "sim1.csv"
    p.write_text("\n".join(out) + "\n")
    return p


def _cfg(*instances):
    return {"nodes": [{"instances": list(instances)}]}


def _inst(hw, pd=None):
    return {"hardware": hw, "pd_type": pd, "tp_size": 1, "num_npus": 1}


def test_aggregated_single_instance_is_sum_latency_over_wall(tmp_path):
    # four requests, each resident 5 s, over a 10 s wall -> 2.0 concurrent
    rows = [(0, 0.0, 5.0, 0.1), (0, 0.0, 5.0, 0.1), (0, 5.0, 10.0, 0.1), (0, 5.0, 10.0, 0.1)]
    pts = operating_points(_csv(tmp_path, rows), _cfg(_inst("A40")))
    assert pts["A40"].concurrency == pytest.approx(2.0)
    assert pts["A40"].phase == "total"


def test_two_instances_of_one_hardware_report_the_busiest(tmp_path):
    """The margin must protect the loaded replica, not the average one."""
    rows = [(0, 0.0, 9.0, 0.1), (1, 0.0, 1.0, 0.1)]
    pts = operating_points(_csv(tmp_path, rows), _cfg(_inst("A40"), _inst("A40")))
    assert pts["A40"].concurrency == pytest.approx(1.0), "9/9, not (9+1)/9/2"


def test_pd_splits_residence_by_phase(tmp_path):
    """Every row is booked to instance 0, yet both engines get a load."""
    # 2 requests: 10 s each, of which 2 s is TTFT. wall 10 s.
    rows = [(0, 0.0, 10.0, 2.0), (0, 0.0, 10.0, 2.0)]
    cfg = _cfg(_inst("RNGD", "decode"), _inst("A40", "prefill"))
    pts = operating_points(_csv(tmp_path, rows), cfg)
    assert pts["A40"].phase == "prefill"
    assert pts["A40"].concurrency == pytest.approx(0.4)     # (2+2)/10
    assert pts["RNGD"].phase == "decode"
    assert pts["RNGD"].concurrency == pytest.approx(1.6)    # (8+8)/10
    assert (pts["A40"].concurrency + pts["RNGD"].concurrency
            == pytest.approx(2.0)), "the phases must sum to the total occupancy"


def test_homogeneous_pd_gives_one_hardware_both_phases(tmp_path):
    rows = [(0, 0.0, 10.0, 2.0), (0, 0.0, 10.0, 2.0)]
    cfg = _cfg(_inst("A40", "decode"), _inst("A40", "prefill"))
    pts = operating_points(_csv(tmp_path, rows), cfg)
    assert set(pts) == {"A40"}
    assert pts["A40"].phase == "total"
    assert pts["A40"].concurrency == pytest.approx(2.0)


def test_replicas_divide_the_phase_load(tmp_path):
    rows = [(0, 0.0, 10.0, 2.0), (0, 0.0, 10.0, 2.0)]
    cfg = _cfg(_inst("RNGD", "decode"), _inst("RNGD", "decode"), _inst("A40", "prefill"))
    pts = operating_points(_csv(tmp_path, rows), cfg)
    assert pts["RNGD"].concurrency == pytest.approx(0.8), "1.6 shared by two engines"


def test_a_missing_csv_yields_no_operating_point(tmp_path):
    """Which the caller turns into margin 0 and a note, never a guessed margin."""
    assert operating_points(tmp_path / "absent.csv", _cfg(_inst("A40"))) == {}


def test_an_empty_csv_yields_no_operating_point(tmp_path):
    p = tmp_path / "sim1.csv"
    p.write_text(HEADER + "\n")
    assert operating_points(p, _cfg(_inst("A40"))) == {}


def test_instance_index_follows_the_compiled_config_not_the_candidate(tmp_path):
    """`compile_to_sim_config` groups by node and emits nodes in cluster order, so
    index 0 can be the DECODE instance even though the candidate names prefill
    first. Reading the mapping from the config is what keeps that right."""
    rows = [(0, 0.0, 10.0, 2.0)]
    cfg = _cfg(_inst("RNGD", "decode"), _inst("A40", "prefill"))
    pts = operating_points(_csv(tmp_path, rows), cfg)
    assert pts["RNGD"].phase == "decode" and pts["A40"].phase == "prefill"
