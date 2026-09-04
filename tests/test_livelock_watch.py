"""`experiments/scripts/livelock_watch.sh` — the D23 spike's stopping rule.

WORK_ORDER_spikes.md STEP A.1. The watcher decides when a P/D simulation is
provably livelocked rather than merely slow, so a run that would have burned its
whole `--timeout` ends in seconds. Getting that decision wrong in either
direction is expensive: a false positive kills a healthy long run, a false
negative puts us back to four-hour sweeps.

No simulator here. The fixtures are the real progress lines from both sides:
the livelocked candidate's own log (D23) and a run that actually completed
(`outputs/.hp-pd-slo/`), so the healthy case is a control, not an invention.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WATCH = REPO / "experiments" / "scripts" / "livelock_watch.sh"

#: D23's signature, verbatim from
#: outputs/pd_slo_sweep_margin18/tight/retry3600_livelock_evidence.txt:
#: prefill pinned at one running request, memory flat (no KV block ever
#: allocated), queue only filling. Instance[1] never receives anything.
LIVELOCKED = """\
        ├─Running Instance[0]: 1 reqs, Waiting: 7 reqs, Total # 4 NPUs, Each NPU Memory Usage 3858.51 MB (9.304 % Used)
        ├─Running Instance[1]: 0 reqs, Waiting: 0 reqs, Total # 4 NPUs, Each NPU Memory Usage 3829.51 MB (9.234 % Used)
        ├─Running Instance[0]: 1 reqs, Waiting: 26 reqs, Total # 4 NPUs, Each NPU Memory Usage 3858.51 MB (9.304 % Used)
        ├─Running Instance[1]: 0 reqs, Waiting: 0 reqs, Total # 4 NPUs, Each NPU Memory Usage 3829.51 MB (9.234 % Used)
        ├─Running Instance[0]: 1 reqs, Waiting: 60 reqs, Total # 4 NPUs, Each NPU Memory Usage 3858.51 MB (9.304 % Used)
        ├─Running Instance[1]: 0 reqs, Waiting: 0 reqs, Total # 4 NPUs, Each NPU Memory Usage 3829.51 MB (9.234 % Used)
        ├─Running Instance[0]: 1 reqs, Waiting: 92 reqs, Total # 4 NPUs, Each NPU Memory Usage 3858.51 MB (9.304 % Used)
        ├─Running Instance[1]: 0 reqs, Waiting: 0 reqs, Total # 4 NPUs, Each NPU Memory Usage 3829.51 MB (9.234 % Used)
"""

#: A run that completed, from outputs/.hp-pd-slo/work/cuda-a40-node_a40a-tp1-dp1-s128-t8192/.
#: Running count and memory both climb — the watcher must never fire on this.
HEALTHY = """\
            ├─Running Instance[0]: 8 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 16064.51 MB (38.736 % Used)
            ├─Running Instance[0]: 18 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 17204.51 MB (41.485 % Used)
            ├─Running Instance[0]: 28 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 18314.51 MB (44.161 % Used)
            ├─Running Instance[0]: 38 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 19454.51 MB (46.910 % Used)
"""


def run_watch(stream: str, tmp_path: Path, *, ticks: int = 3, extra: str = "", timeout: int = 60):
    """Feed `stream` to the watcher as a command's output; return the CompletedProcess."""
    log = tmp_path / "stream.log"
    log.write_text(stream)
    cmd = [str(WATCH), "-n", str(ticks), "-q"]
    cmd += ["--", "bash", "-c", f"cat {log}{extra}"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_livelock_is_reported_as_exit_3(tmp_path):
    """The D23 signature stops the run and is distinguishable from a timeout."""
    proc = run_watch(LIVELOCKED, tmp_path, ticks=3, extra="; sleep 30")
    assert proc.returncode == 3, proc.stderr
    assert "LIVELOCK" in proc.stderr
    assert "Instance[0]" in proc.stderr, "the verdict must name which instance stalled"


def test_healthy_run_is_never_flagged(tmp_path):
    """The control: a run that completed must pass through untouched."""
    proc = run_watch(HEALTHY, tmp_path, ticks=3)
    assert proc.returncode == 0, proc.stderr
    assert "LIVELOCK" not in proc.stderr


def test_an_idle_instance_alone_does_not_trigger(tmp_path):
    """Instance[1] sits at running=0 for the whole hour; that is starvation, not a
    stall, and on its own it must not fire — otherwise every warm-up tick would."""
    idle = "".join(
        line + "\n" for line in LIVELOCKED.splitlines() if "Instance[1]" in line
    )
    proc = run_watch(idle * 5, tmp_path, ticks=3)
    assert proc.returncode == 0, proc.stderr


def test_a_growing_queue_alone_does_not_trigger(tmp_path):
    """Requests piling up while the instance is actually working is backlog, not
    livelock. Memory moving is what separates them."""
    stream = "".join(
        f"        ├─Running Instance[0]: 4 reqs, Waiting: {w} reqs, Total # 4 NPUs, "
        f"Each NPU Memory Usage {3858.51 + w} MB ({9.304 + w * 0.01:.3f} % Used)\n"
        for w in range(1, 12)
    )
    proc = run_watch(stream, tmp_path, ticks=3)
    assert proc.returncode == 0, proc.stderr


def test_the_streak_must_be_consecutive(tmp_path):
    """One healthy tick in the middle resets the count, so a marginal run that is
    merely slow keeps running."""
    lines = [line for line in LIVELOCKED.splitlines() if "Instance[0]" in line]
    healthy_tick = (
        "        ├─Running Instance[0]: 2 reqs, Waiting: 93 reqs, Total # 4 NPUs, "
        "Each NPU Memory Usage 3900.00 MB (9.400 % Used)"
    )
    interrupted = [*lines[:2], healthy_tick, *lines[2:]]
    proc = run_watch("\n".join(interrupted) + "\n", tmp_path, ticks=3)
    assert proc.returncode == 0, proc.stderr


def test_timeout_is_reported_as_124(tmp_path):
    """The wall-clock ceiling keeps `timeout`'s code, so the two failure modes stay
    distinguishable in a driver script."""
    proc = subprocess.run(
        [str(WATCH), "-t", "2", "-q", "--", "bash", "-c", "sleep 30"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 124, proc.stderr


def test_command_exit_code_passes_through(tmp_path):
    """A crashing simulator must not be reported as a livelock."""
    proc = subprocess.run(
        [str(WATCH), "-q", "--", "bash", "-c", "exit 7"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 7, proc.stderr


def test_a_run_that_never_reports_is_caught_by_grace(tmp_path):
    """The H6 stragglers' shape: the ASTRA-Sim child dies at startup, the frontend
    spins in `controller.read_wait` on EOF, and not one progress line is ever
    printed. The tick detector cannot see this -- it only judges runs that talk --
    so four of them burned a full 1800 s ceiling before -g existed."""
    proc = subprocess.run(
        [str(WATCH), "-g", "6", "-q", "--", "bash", "-c", "sleep 60"],
        capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 4, proc.stderr
    assert "NO PROGRESS" in proc.stderr
    assert "never started reporting" in proc.stderr


def test_a_run_that_stops_reporting_is_caught_by_silence(tmp_path):
    """Progress that simply stops is also not a tick-stall: there are no ticks to
    count. Distinct from grace because the run did start."""
    log = tmp_path / "s.log"
    log.write_text(HEALTHY)
    proc = subprocess.run(
        [str(WATCH), "-n", "999", "-s", "6", "-q", "--", "bash", "-c", f"cat {log}; sleep 60"],
        capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 4, proc.stderr
    assert "silent for" in proc.stderr


def test_grace_does_not_fire_on_a_slow_but_talking_run(tmp_path):
    """A healthy N=300 run spends ~740 log lines on banner and graph generation
    before its first tick. Grace must not clip that, so it is measured to the
    FIRST progress line only and then hands over to -s."""
    log = tmp_path / "s.log"
    log.write_text(HEALTHY)
    proc = subprocess.run(
        [str(WATCH), "-n", "999", "-g", "8", "-s", "0", "--",
         "bash", "-c", f"sleep 4; cat {log}; sleep 8"],
        capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, proc.stderr


def test_missing_separator_is_an_error(tmp_path):
    """`--` is mandatory; forgetting it would otherwise swallow the command."""
    proc = subprocess.run(
        [str(WATCH), "echo", "hi"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2
    assert "--" in proc.stderr


@pytest.mark.parametrize("stream,expected", [(LIVELOCKED, 3), (HEALTHY, 0)])
def test_output_is_forwarded_without_q(tmp_path, stream, expected):
    """Without -q the command's own output still reaches stdout, so the watcher can
    wrap a real run without hiding its log."""
    log = tmp_path / "s.log"
    log.write_text(stream)
    proc = subprocess.run(
        [str(WATCH), "-n", "3", "--", "bash", "-c", f"cat {log}; sleep 5"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == expected
    assert "Running Instance[0]" in proc.stdout
