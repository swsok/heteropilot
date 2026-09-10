"""The open-loop envelope point must land on the SAME axis as the one it joins.

`docs/HANDOVER.md` §2.2. The A40 accuracy domain's only point, served concurrency
170.56, was produced by `python -m bench run` -- an arrival-trace replay. A second
point is only useful if `in_domain` and the interpolation between them mean
something, and that requires both to be the same measurement of the same
quantity. So the decisive test here is not a unit test of an expression: it is
that this driver, run over the COMMITTED artifact that produced 170.56,
reproduces 170.56.

The rest guards the two ways an open-loop point can lie:

  - a saturated run reports a concurrency that is a fact about the queue, not
    about the offered load. The committed point IS saturated and must be flagged
    as such -- flagged, never dropped. The flag is the QUEUE DELAY SLOPE, and the
    tests below pin both sides of it, because the first metric tried here was
    wrong in the dangerous direction: completed-rps-against-offered divides by a
    wall containing the drain tail, so a healthy 20-request point at 0.5 rps read
    0.64 and looked saturated. Short runs are where low-load points live;
  - the bench window must exclude model loading. `python -m bench run` loads
    weights inside the same process, and a power mean that included a
    two-minute load phase would be an average of two different machines.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/measure_envelope_openloop.py"
COMMITTED = ROOT / "outputs/phase0_bench/A40/vllm"
#: A 20-request 0.5 rps run on the A40 node, kept as the unsaturated counterpart
#: to the committed saturated one. Small (20 rows) and the only fixture that
#: shows the drain-tail trap.
SMOKE = ROOT / "outputs/a40_envelope_openloop/lowload_reference"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ol = _load("openloop_under_test", SCRIPT)


def _committed():
    meta = json.loads((COMMITTED / "meta.json").read_text())
    reqs = [json.loads(line)
            for line in (COMMITTED / "requests.jsonl").read_text().splitlines()
            if line]
    return meta, reqs


@pytest.mark.skipif(not COMMITTED.exists(), reason="A40 bench artifact absent")
def test_it_reproduces_the_committed_17056_point():
    """The whole reason this driver exists rather than the closed-loop one.

    `profiles/calibration/a40.accuracy.yaml` records served concurrency 170.56
    over a 179.6 s wall, computed from this file. If this driver disagrees, its
    points are on a different axis and interpolating between them is meaningless
    -- which is exactly the kind of silent mismatch D22 was.
    """
    meta, reqs = _committed()
    # offered ~10.3 rps: the dataset is sps10 and the committed note records 10.3.
    point = ol.summarise_openloop_point(meta, reqs, rows=[], offered_rps=10.3)
    assert point["served_concurrency"] == pytest.approx(170.56, abs=0.01)
    assert point["wall_s"] == pytest.approx(179.56, abs=0.05)
    assert point["requests_ok"] == 300


@pytest.mark.skipif(not COMMITTED.exists(), reason="A40 bench artifact absent")
def test_the_committed_point_is_correctly_flagged_as_saturated():
    """A5(b) has no pool to check open-loop, and this is what replaces it.

    The committed run offered 10.3 rps and completed 1.671. Its concurrency of
    170 is a queue depth, not a load level -- and the accuracy domain's own note
    says so. A driver that did not flag it would let the next such run be read as
    "the A40 at concurrency 170" rather than "the A40 saturated".
    """
    meta, reqs = _committed()
    point = ol.summarise_openloop_point(meta, reqs, rows=[], offered_rps=10.3)
    assert point["saturated"] is True
    # Queue delay gained ~4 s per second of arrivals: the backlog grew for the
    # whole run. Two orders of magnitude above the threshold, not a close call.
    assert point["queue_delay_slope_s_per_s"] == pytest.approx(4.15, abs=0.05)
    assert point["completed_rps"] == pytest.approx(1.671, abs=0.01)
    assert any("queue grew" in n for n in point["notes"]), point["notes"]


@pytest.mark.skipif(not COMMITTED.exists(), reason="A40 bench artifact absent")
@pytest.mark.skipif(not SMOKE.exists(), reason="A40 low-load run absent")
def test_an_unsaturated_point_is_not_flagged():
    """The flag must discriminate, or it is decoration.

    This is the case the FIRST version of the metric got wrong. A 20-request run
    at 0.5 rps on this node completes every request with no waiting at all
    (queue-delay slope -0.0000), but its wall is 62 s against a 38 s arrival span
    because of the drain tail -- so completed-rps-against-offered read 0.643 and
    would have marked a clean low-load point as saturated. The whole purpose of a
    second accuracy-domain point is that it is at LOW load; a saturation test
    that fires on short low-load runs would have refused every point worth
    taking.
    """
    meta = json.loads((SMOKE / "meta.json").read_text())
    reqs = [json.loads(line)
            for line in (SMOKE / "requests.jsonl").read_text().splitlines() if line]
    point = ol.summarise_openloop_point(meta, reqs, rows=[], offered_rps=0.5)
    assert point["saturated"] is False
    assert point["queue_delay_slope_s_per_s"] == pytest.approx(0.0, abs=1e-3)
    assert point["notes"] == []
    # ... and the metric that would have lied is still recorded, so the trap is
    # visible in the artifact rather than only in this docstring.
    assert point["completed_rps"] / 0.5 < 0.7


def test_the_bench_window_comes_from_meta_not_from_wall_clock():
    """Model loading must be outside it. `bench run` loads weights in-process, so
    a window taken around the subprocess would average a two-minute load into a
    serving power figure -- on this node that is ~30 W of idle plus a copy phase
    against ~250 W of decode."""
    meta = {"started_at": "2026-09-10T00:01:00.000000Z",
            "finished_at": "2026-09-10T00:02:00.000000Z"}
    # One request, entirely inside the window, in one clock domain.
    reqs = [{"arrival_time": 100.0, "queued_ts": 100.0, "scheduled_ts": 100.0,
             "first_token_ts": 100.5, "last_token_ts": 110.0, "output_toks": 20}]
    # Samples spanning load (high power, before started_at) and serving.
    rows = [ol._me.SamplerRow(ts=ts, device_sn="GPU-x", power_w=power,
                              util_mean_pct=util, util_max_pct=util,
                              dram_used_ratio=0.5)
            for ts, power, util in
            # 00:00:00-00:00:59 model load at 120 W; 00:01:00+ serving at 250 W
            [(ol._epoch("2026-09-10T00:00:30.000000Z"), 120.0, 40.0),
             (ol._epoch("2026-09-10T00:01:30.000000Z"), 250.0, 99.0)]]
    point = ol.summarise_openloop_point(meta, reqs, rows, offered_rps=0.1,
                                        device_sn="GPU-x")
    assert point["bench_window"]["samples"] == 1
    assert point["bench_window"]["power_w"]["mean"] == pytest.approx(250.0)
    # A5(c): never a wattage without the utilisation from the same samples.
    assert point["bench_window"]["util_pct"]["mean"] == pytest.approx(99.0)


def test_respacing_agrees_with_the_simulator_side_driver(tmp_path):
    """Both sides of the comparison must space arrivals identically.

    `lowload_sim_error.py` writes the simulator's trace and this writes the
    hardware's. If the two constant-spacing implementations ever diverge, the two
    sides sit at different offered loads and the error figure is measuring that
    instead of the simulator.
    """
    sim = _load("lowload_for_openloop_test",
                ROOT / "experiments/scripts/lowload_sim_error.py")
    src = tmp_path / "src.jsonl"
    src.write_text("".join(
        json.dumps({"input_toks": 10, "output_toks": 20, "arrival_time_ns": 7, "i": i})
        + "\n" for i in range(6)
    ))
    a = ol.respace_trace(tmp_path / "a.jsonl", src, rps=0.3, n=4).read_text()
    b = sim.write_trace(tmp_path / "b.jsonl", src, rps=0.3, n=4).read_text()
    assert a == b


def test_a_run_with_no_usable_timestamps_reports_zero_rather_than_dividing_by_zero():
    meta = {"started_at": "2026-09-10T00:00:00.000000Z",
            "finished_at": "2026-09-10T00:01:00.000000Z"}
    reqs = [{"arrival_time": None, "queued_ts": None, "first_token_ts": None,
             "last_token_ts": None, "output_toks": 20}]
    point = ol.summarise_openloop_point(meta, reqs, rows=[], offered_rps=1.0)
    assert point["served_concurrency"] == 0.0
    assert point["requests_ok"] == 0
    assert point["ttft_ms"] is None
    assert any("no usable timestamps" in n for n in point["notes"]), point["notes"]
