"""The envelope analysis must enforce A5 rather than trust the operator (STEP 3.1).

`WORK_ORDER_rps_aware.md` rev 2. These tests need neither hardware nor a
simulator: `measure_envelope.summarise_point` is a pure function of a bench report
and a sampler CSV, which is why the orchestration was kept out of it.

What is asserted is the D22 failure mode. That envelope's c32 point was labelled by
the concurrency asked for; the pool held 24 requests, so it actually ran at
effective concurrency 21.2, and an exponent fitted across the point read a
pool-capped 1.74x interval as a doubling. Every rule below exists to make that
visible in the artifact instead of in a retraction:

  - served concurrency is Little's law, never the requested value;
  - a point that does not sustain its concurrency is MARKED, not dropped -- a
    suppressed point leaves a hole a fit will happily cross;
  - power is a sustained mean reported together with the utilisation from the
    same samples (A5(c)).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/measure_envelope.py"


def _load():
    """Import by path. Unlike the other drivers this one defines dataclasses, so
    the module must be in `sys.modules` before `exec_module` -- `@dataclass`
    resolves annotations through `sys.modules[cls.__module__]`."""
    spec = importlib.util.spec_from_file_location("measure_envelope_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


me = _load()


def _bench(concurrency: int, latencies_s: list[float], wall_s: float,
           ttft_ms: float = 100.0, tpot_ms: float = 10.0, failed: int = 0) -> dict:
    per_request = [
        {"error": None, "latency_ns": s * 1e9, "ttft_ns": ttft_ms * 1e6,
         "tpot_ns": tpot_ms * 1e6, "streamed_chunks": 20}
        for s in latencies_s
    ]
    per_request += [{"error": "boom", "latency_ns": 1e9, "ttft_ns": None,
                     "tpot_ns": None, "streamed_chunks": 0}] * failed
    return {"concurrency": concurrency, "wall_s": wall_s,
            "requests": len(per_request), "ok": len(latencies_s), "failed": failed,
            "per_request": per_request}


def _sampler_csv(tmp_path: Path, samples: list[tuple[float, float, float]],
                 sn: str = "RNGD-A", dropped_at: list[float] | None = None) -> Path:
    """samples: (ts, power_w, util_mean_pct)."""
    p = tmp_path / "power.csv"
    lines = ["# power_sampler.sh test fixture",
             "ts_info,ts_status,device_sn,pci_bdf,dev_name,power_w,util_mean_pct,"
             "util_min_pct,util_max_pct,pe_busy,pe_total,dram_used_ratio,"
             "temperature_c,raw_info,raw_status"]
    for ts, power, util in samples:
        lines.append(f'{ts},{ts},{sn},0000:03:00.0,npu0,{power},{util},'
                     f'{util},{util},8,8,0.5,40.0,"{{}}","{{}}"')
    for ts in (dropped_at or []):
        lines.append(f'{ts},{ts},,,,,,,,,,,,"query failed",""')
    p.write_text("\n".join(lines) + "\n")
    return p


# --- served concurrency ----------------------------------------------------

def test_served_concurrency_is_littles_law_not_the_requested_value():
    # 8 requests of 5 s each over a 10 s wall = 4.0 concurrent, not 8.
    bench = _bench(concurrency=8, latencies_s=[5.0] * 8, wall_s=10.0)
    assert me.served_concurrency(bench["per_request"], 10.0) == pytest.approx(4.0)


def test_failed_requests_do_not_inflate_served_concurrency():
    """Their partial latency would push the ratio up, hiding a pool-bound point."""
    bench = _bench(concurrency=4, latencies_s=[4.0] * 4, wall_s=8.0, failed=6)
    # only the 4 completed requests count: 16 s / 8 s = 2.0
    assert me.served_concurrency(bench["per_request"], 8.0) == pytest.approx(2.0)


def test_the_d22_c32_point_would_have_been_flagged(tmp_path):
    """The actual case: requested 32, sustained 21.2, pool of 24."""
    latencies = [21.2 * 100.0 / 24] * 24          # sum/wall = 21.2 at wall=100
    bench = _bench(concurrency=32, latencies_s=latencies, wall_s=100.0)
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, [(t, 300.0, 80.0)
                                                       for t in range(0, 100)]))
    out = me.summarise_point(bench, rows, me.Window(0, 100), pool_size=24)
    assert out["served_concurrency"] == pytest.approx(21.2, abs=0.05)
    assert out["served_ratio"] == pytest.approx(21.2 / 32, abs=0.002)
    assert out["pool_binding"] is True
    joined = " ".join(out["notes"])
    assert "4x requested concurrency" in joined, joined
    assert "did not sustain" in joined, joined


def test_a_healthy_point_is_not_flagged(tmp_path):
    bench = _bench(concurrency=8, latencies_s=[10.0] * 300, wall_s=375.0)
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, [(t, 300.0, 80.0)
                                                       for t in range(0, 400)]))
    out = me.summarise_point(bench, rows, me.Window(0, 375), pool_size=300)
    assert out["served_ratio"] == pytest.approx(1.0, abs=0.01)
    assert out["pool_binding"] is False
    assert out["notes"] == []


# --- power / utilisation windows -------------------------------------------

def test_power_is_averaged_over_the_bench_window_only(tmp_path):
    """Idle samples on either side must not drag the sustained mean down."""
    samples = ([(float(t), 40.0, 0.0) for t in range(0, 10)]      # idle before
               + [(float(t), 300.0, 90.0) for t in range(10, 20)]  # bench
               + [(float(t), 40.0, 0.0) for t in range(20, 30)])   # idle after
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, samples))
    bench_w = me.window_stats(rows, me.Window(10, 20))
    idle_w = me.window_stats(rows, me.Window(0, 10))
    assert bench_w["samples"] == 10
    assert bench_w["power_w"]["mean"] == pytest.approx(300.0)
    assert idle_w["power_w"]["mean"] == pytest.approx(40.0)


def test_power_is_never_reported_without_utilisation(tmp_path):
    """A5(c). The two come from different furiosa-smi subcommands, so the pairing
    is a property of the analysis, not of the vendor tool."""
    samples = [(float(t), 300.0, 90.0) for t in range(0, 10)]
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, samples))
    stats = me.window_stats(rows, me.Window(0, 10))
    assert stats["power_w"] is not None
    assert stats["util_pct"] is not None
    assert stats["util_pct"]["mean"] == pytest.approx(90.0)


def test_dropped_samples_are_excluded_but_counted(tmp_path):
    """A failed query is a hole in the series, not a reading of zero watts."""
    samples = [(float(t), 300.0, 90.0) for t in range(0, 10)]
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, samples, dropped_at=[3.5, 6.5]))
    stats = me.window_stats(rows, me.Window(0, 10))
    assert stats["samples"] == 10
    assert stats["dropped_samples"] == 2
    assert stats["power_w"]["mean"] == pytest.approx(300.0)


def test_a_window_with_no_samples_says_so_rather_than_returning_zero(tmp_path):
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, [(float(t), 300.0, 90.0)
                                                       for t in range(0, 5)]))
    stats = me.window_stats(rows, me.Window(100, 200))
    assert stats["samples"] == 0
    assert stats["power_w"] is None
    assert "no usable samples" in stats["note"]


def test_analysis_can_be_pinned_to_one_serial(tmp_path):
    """dev_name re-enumerates; device_sn does not. Selecting on the serial must
    exclude the other cards' power entirely."""
    p = tmp_path / "power.csv"
    header = ("ts_info,ts_status,device_sn,pci_bdf,dev_name,power_w,util_mean_pct,"
              "util_min_pct,util_max_pct,pe_busy,pe_total,dram_used_ratio,"
              "temperature_c,raw_info,raw_status")
    lines = [header]
    for t in range(0, 10):
        lines.append(f'{t},{t},RNGD-A,0000:03:00.0,npu0,300.0,90,90,90,8,8,0.5,40,"{{}}","{{}}"')
        lines.append(f'{t},{t},RNGD-B,0000:04:00.0,npu1,38.0,0,0,0,0,8,0,40,"{{}}","{{}}"')
    p.write_text("\n".join(lines) + "\n")
    rows = me.read_sampler_csv(p)
    win = me.Window(0, 10)
    assert me.window_stats(rows, win, "RNGD-A")["power_w"]["mean"] == pytest.approx(300.0)
    assert me.window_stats(rows, win, "RNGD-B")["power_w"]["mean"] == pytest.approx(38.0)
    assert me.window_stats(rows, win)["samples"] == 20


# --- percentiles come from the one util ------------------------------------

def test_latency_percentiles_use_the_planner_util(tmp_path):
    from planner.util.percentile import summary as planner_summary
    lat = [1.0] * 100
    bench = _bench(concurrency=4, latencies_s=lat, wall_s=25.0, ttft_ms=123.0)
    rows = me.read_sampler_csv(_sampler_csv(tmp_path, [(float(t), 300.0, 90.0)
                                                       for t in range(0, 30)]))
    out = me.summarise_point(bench, rows, me.Window(0, 25), pool_size=300)
    assert out["ttft_ms"] == planner_summary([123.0] * 100)


# --- the sampler's own output round-trips ----------------------------------

def test_reads_a_real_sampler_header_without_choking(tmp_path):
    """The provenance comment block is not data and must not become a row."""
    p = tmp_path / "power.csv"
    p.write_text(
        "# power_sampler.sh 2026-09-08T08:02:10Z\n"
        "# host_cores=96\n"
        "# - furiosa-smi\n"
        "#   2026.1.1\n"
        "# info_fields=arch,core_clock,dev_name\n"
        "ts_info,ts_status,device_sn,pci_bdf,dev_name,power_w,util_mean_pct,"
        "util_min_pct,util_max_pct,pe_busy,pe_total,dram_used_ratio,"
        "temperature_c,raw_info,raw_status\n"
        '"1788854531.005787112","1788854531.155880397","RNGD-A","0000:03:00.0",'
        '"npu0","38.00",0,0,0,0,8,0,"39.95","{}","{}"\n'
        "# stopped 2026-09-08T08:02:48Z\n"
    )
    rows = me.read_sampler_csv(p)
    assert len(rows) == 1
    assert rows[0].power_w == pytest.approx(38.0)
    assert rows[0].device_sn == "RNGD-A"
