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


# ---------------------------------------------------------------------------
# The drain must not outlive the run
# ---------------------------------------------------------------------------

def _orphaned_drains() -> set[int]:
    """PIDs of `cat /tmp/livelock_watch.*` processes, excluding our own ancestry.

    Owned by `os` rather than by `ps | grep`, and excluding self AND every
    ancestor by PID, because a pattern that names the thing you are looking for
    is also in the command line of the process doing the looking. That trap is
    `CLAUDE.md`'s, and it defeats a structural argv match just as readily as a
    `pkill -f`: three filters in a row reported this test's own shell as a
    survivor before the ancestry exclusion was added.
    """
    import os
    import re

    pattern = re.compile(r"^/tmp/livelock_watch\.[A-Za-z0-9]{6}$")

    ours, pid = set(), os.getpid()
    while pid and pid not in ours:
        ours.add(pid)
        try:
            with open(f"/proc/{pid}/status") as fh:
                pid = next(
                    int(line.split()[1]) for line in fh if line.startswith("PPid:")
                )
        except (OSError, StopIteration):
            break

    found = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in ours:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")[:-1]
        except OSError:
            continue
        if len(argv) == 2 and argv[0] == b"cat" and pattern.match(
            argv[1].decode("utf-8", "replace")
        ):
            found.add(int(entry))
    return found


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_a_stopped_run_leaves_no_orphaned_drain(tmp_path):
    """The verdict paths must not leak the drain — 250 of them once did.

    Between 2026-09-10 and 2026-09-18 every run ending in verdict 3 or 4 left
    one `cat` blocked forever in `wait_for_partner`: the drain reopened the FIFO
    *after* the child had been TERMed, so no writer was left for its `open()` to
    rendezvous with, it was never killed or waited for, and the EXIT trap then
    unlinked the FIFO out from under it. 250 had accumulated by the time anyone
    looked, all `ppid=1`.

    This is the regression test for the fd-3 fix, and it asserts the property
    that matters — *this run added none* — rather than an absolute count, so a
    machine with pre-existing orphans from an older checkout does not fail it.
    """
    before = _orphaned_drains()
    proc = run_watch(LIVELOCKED, tmp_path, ticks=3, extra="; sleep 30")
    assert proc.returncode == 3, "expected the livelock verdict, not this test's subject"

    # The drain is killed and reaped before the script exits, so there is no
    # settling window to wait out; if one is needed later, that is the bug.
    assert _orphaned_drains() - before == set(), (
        "the stopped run left an orphaned drain; see the fd-3 handling in "
        "livelock_watch.sh and docs/HANDOVER.md §3"
    )


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_a_run_caught_by_grace_leaves_no_orphaned_drain(tmp_path):
    """The other verdict path, which leaked identically."""
    before = _orphaned_drains()
    proc = subprocess.run(
        [str(WATCH), "-n", "999", "-g", "2", "-s", "0", "-q", "--",
         "bash", "-c", "sleep 60"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 4
    assert _orphaned_drains() - before == set()


# ---------------------------------------------------------------------------
# The reader must not reopen the FIFO either
# ---------------------------------------------------------------------------

def test_the_watcher_never_reopens_the_fifo_to_read_it():
    """fd 3 exists so that nothing opens the FIFO by name a second time.

    The fix above converted the *drain* to `cat <&3` but left the read loop
    ending `done <"$FIFO"` — a second `open(2)`, which blocks until a writer
    appears. A command that exits before the shell reaches that line has already
    taken the only writer with it, so the parent parked in `pipe_wait` forever,
    holding fd 3, with no child left to rendezvous with. That is the
    intermittent red this file showed on `main`: 5 hangs in 40 runs, always a
    60 s `TimeoutExpired`, and always on one of the cases whose command exits
    promptly — which is why a different test failed each run and each one passed
    alone.

    Asserted on the source because the behavioural test below can only sample a
    race, while the invariant is structural: after fd 3 is opened, `$FIFO` is a
    name to be unlinked, never one to be opened.
    """
    src = WATCH.read_text()
    _, after = src.split('exec 3<"$FIFO"', 1)
    offenders = [
        line for line in after.splitlines()
        if not line.lstrip().startswith("#") and '<"$FIFO"' in line
    ]
    assert offenders == [], (
        "the watcher reopens the FIFO after fd 3 is open; read from fd 3 "
        f"instead (`done <&3`, `cat <&3`): {offenders}"
    )
    assert "done <&3" in src, "the read loop must consume fd 3, not a fresh open"


def test_a_command_that_exits_immediately_never_hangs():
    """The behavioural half: the losing side of that race, sampled often enough.

    A child that is gone before the reader attaches is the whole failure mode,
    and `exit 7` is the cheapest way to be gone. The hang rate under the bug was
    12.5 % (5/40), so 60 runs fail it with probability 1 - 0.875**60 > 0.999,
    while the fixed script cannot hang here at all — there is no `open()` left to
    block in. A `timeout` therefore is the assertion; `subprocess.run` raises it.
    """
    for i in range(60):
        proc = subprocess.run(
            [str(WATCH), "-q", "--", "bash", "-c", "exit 7"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 7, f"run {i} returned {proc.returncode}: {proc.stderr}"
