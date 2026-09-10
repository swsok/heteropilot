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
import os
import shutil
import signal
import subprocess
import sys
import time
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


# --- the CUDA execution half (docs/HANDOVER.md §2.2) ------------------------
#
# The analysis above is what the A5 rules live in and it never learned a vendor.
# What follows guards the three places that DID know one, so that a CUDA point is
# held to the same rules as an RNGD one -- which is the only reason the two curves
# can be put in the same table.

def test_the_backend_table_covers_exactly_the_three_vendor_specific_places():
    """Server launch, sampler, bench interpreter. Nothing else branched."""
    assert set(me.BACKENDS) == {"furiosa", "cuda"}
    for backend, entry in me.BACKENDS.items():
        assert set(entry) == {"sampler", "bench_python"}, backend
        assert (ROOT / entry["sampler"]).exists(), entry["sampler"]


def test_the_furiosa_defaults_are_byte_for_byte_what_they_were():
    """Every committed RNGD invocation must still run unchanged. The system
    interpreter and `power_sampler.sh` are what STEP 3 measured with."""
    assert me.BACKENDS["furiosa"]["bench_python"] == "/usr/bin/python3"
    assert me.BACKENDS["furiosa"]["sampler"] == "experiments/scripts/power_sampler.sh"
    cmd, env = me.server_command("furiosa", "/path/to/artifact", 8000, card=2, tp=1)
    assert cmd[:2] == ["furiosa-llm", "serve"]
    assert "--devices" in cmd and cmd[cmd.index("--devices") + 1] == "npu:2:*"
    # FuriosaAI pins by flag, so it needs no environment at all -- an overlay here
    # would leak into the server and change what a re-run measures.
    assert env == {}


def test_cuda_pins_the_card_by_environment_because_there_is_no_flag():
    cmd, env = me.server_command("cuda", "meta-llama/Llama-3.1-8B", 8001,
                                 card=5, tp=1)
    assert cmd[:2] == ["vllm", "serve"]
    assert cmd[2] == "meta-llama/Llama-3.1-8B"
    assert env == {"CUDA_VISIBLE_DEVICES": "5"}
    # The server must NOT also be told the physical index: inside the process the
    # pinned card is device 0, and passing 5 through would address a card that is
    # not visible to it.
    assert "5" not in cmd


def test_cuda_tp_reaches_the_server_and_furiosa_ignores_it():
    """TP is a launch flag on vLLM and a property of the compiled artifact on
    FuriosaAI, so the same argument cannot mean the same thing on both."""
    cmd, _ = me.server_command("cuda", "m", 8000, card=0, tp=4)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    furiosa_cmd, _ = me.server_command("furiosa", "m", 8000, card=0, tp=4)
    assert "--tensor-parallel-size" not in furiosa_cmd


def test_an_unknown_backend_fails_loudly_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown backend"):
        me.server_command("rbln", "m", 8000, card=0, tp=1)


def _stub_nvidia_smi(tmp_path: Path, rows: str) -> Path:
    """A fake `nvidia-smi` so this test runs on the NPU and A5000 nodes too.

    The sampler is A40-only in production, but a test that only passes where the
    hardware is would be a gate that silently stops guarding on two of the three
    machines this repository moves between.
    """
    d = tmp_path / "stubbin"
    d.mkdir()
    stub = d / "nvidia-smi"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *--version*) echo \'NVIDIA-SMI version  : 560.35.05\' ;;\n'
        f"  *) printf '%s\\n' {rows!r} ;;\n"
        "esac\n"
    )
    stub.chmod(0o755)
    return d


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_nvidia_sampler_output_round_trips_through_the_analysis(tmp_path):
    """The whole reason a second sampler was cheap: one schema, one parser.

    Runs the real script against a stubbed `nvidia-smi` and feeds its output to
    the same `read_sampler_csv` the RNGD files go through. If the columns ever
    drift apart, every A40 power figure silently becomes null and this fails first.
    """
    row = ("3, GPU-deadbeef-0000-0000-0000-000000000000, NVIDIA A40, "
           "00000000:07:00.0, 148.55, 97, 23034, 46068, 61")
    stub_dir = _stub_nvidia_smi(tmp_path, row)
    out = tmp_path / "power.csv"

    env = {**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.Popen(
        [str(ROOT / "experiments/scripts/power_sampler_nvidia.sh"),
         "--out", str(out), "--devices", "3"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        # Two ticks is enough to show the loop writes more than its header.
        time.sleep(2.5)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)

    rows = me.read_sampler_csv(out)
    assert len(rows) >= 2, out.read_text()
    r = rows[0]
    assert r.power_w == pytest.approx(148.55)
    assert r.util_mean_pct == pytest.approx(97.0)
    assert r.util_max_pct == pytest.approx(97.0)
    # device_sn carries the UUID: nvidia-smi reports `serial` as [N/A] on these
    # boards and the index moves under CUDA_VISIBLE_DEVICES.
    assert r.device_sn == "GPU-deadbeef-0000-0000-0000-000000000000"
    assert r.dram_used_ratio == pytest.approx(23034 / 46068)
    assert r.dropped is False

    # A5(c) again, now end to end: the analysis must find both numbers, from the
    # same samples, or refuse to report either.
    stats = me.window_stats(rows, me.Window(rows[0].ts - 0.5, rows[-1].ts + 0.5),
                            device_sn=r.device_sn)
    assert stats["power_w"]["mean"] == pytest.approx(148.55)
    assert stats["util_pct"]["mean"] == pytest.approx(97.0)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_nvidia_sampler_records_a_failed_query_as_a_hole(tmp_path):
    """A dropped sample is a gap in the series, never a reading of zero watts --
    the same convention as the RNGD sampler, asserted separately because a
    sampler that silently skips looks identical to one that never ran."""
    stub_dir = _stub_nvidia_smi(tmp_path, "")   # emits nothing -> query failed
    out = tmp_path / "power.csv"
    env = {**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.Popen(
        [str(ROOT / "experiments/scripts/power_sampler_nvidia.sh"),
         "--out", str(out)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        time.sleep(2.5)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)

    rows = me.read_sampler_csv(out)
    assert rows and all(r.dropped for r in rows)
    stats = me.window_stats(rows, me.Window(rows[0].ts - 0.5, rows[-1].ts + 0.5))
    assert stats["samples"] == 0
    assert stats["dropped_samples"] == len(rows)
    assert stats["power_w"] is None
