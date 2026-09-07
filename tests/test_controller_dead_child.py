"""The frontend must notice when the ASTRA-Sim child dies (deviations.md D25-b).

`WORK_ORDER_d23_fix_revalidation.md` STEP 2. `Controller.read_wait` and
`check_end` loop on `p.stdout.readline()` until they see a specific line. A dead
child's `readline()` returns `""` forever, which matches neither exit condition,
so the loops used to spin at 100% CPU and grow their output list without bound
(measured: RSS 6.51 -> 6.68 GB in 30 s, child a zombie at exit 1) until the
caller's timeout fired. The failure then presented as "the simulator is slow", and
every longer timeout confirmed the wrong theory.

Two properties are tested, and the second matters as much as the first:

  - a dead child raises promptly, with the exit code and the child's stderr in the
    message;
  - a healthy stream still returns exactly what it returned before, because these
    two loops are on the path of every simulation this repository runs.

No simulator is launched; the child is a stub.
"""

from __future__ import annotations

import time

import pytest

from serving.core.controller import AstraSimChildDied, Controller


class _StubProcess:
    """Minimal stand-in for the `Popen` the controller is handed."""

    def __init__(self, lines, returncode=None, stderr_path=None):
        self._lines = list(lines)
        self._returncode = returncode
        self.reads = 0
        if stderr_path is not None:
            self.astra_stderr_path = str(stderr_path)
        self.stdout = self

    # -- stdout surface ----------------------------------------------------
    def readline(self):
        self.reads += 1
        return self._lines.pop(0) if self._lines else ""

    def flush(self):
        pass

    # -- process surface ---------------------------------------------------
    def poll(self):
        return self._returncode


def test_read_wait_raises_when_the_child_has_exited(tmp_path):
    err = tmp_path / "astra_stderr.log"
    err.write_text("Unable to open file: tmp__mem/remote_mem.json\n")
    p = _StubProcess([], returncode=1, stderr_path=err)

    started = time.monotonic()
    with pytest.raises(AstraSimChildDied) as excinfo:
        Controller(1).read_wait(p)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"took {elapsed:.2f}s; this must fail immediately, not eventually"
    message = str(excinfo.value)
    assert "exited with code 1" in message, message
    assert "read_wait" in message, message
    assert "remote_mem.json" in message, "the child's stderr must reach the message"


def test_check_end_raises_too():
    """Both loops have the same shape, so both need the guard."""
    p = _StubProcess([], returncode=134)
    with pytest.raises(AstraSimChildDied) as excinfo:
        Controller(1).check_end(p)
    assert "exited with code 134" in str(excinfo.value)
    assert "check_end" in str(excinfo.value)


def test_a_live_child_with_a_dead_stream_also_raises():
    """poll() is None but the stream is at EOF: bounded, not infinite."""
    p = _StubProcess([], returncode=None)
    with pytest.raises(AstraSimChildDied) as excinfo:
        Controller(1).read_wait(p)
    assert "consecutive empty reads" in str(excinfo.value)
    assert p.reads <= Controller._MAX_EMPTY_READS + 1, (
        f"read {p.reads} times; the guard should stop at {Controller._MAX_EMPTY_READS}"
    )


def test_message_says_so_when_stderr_was_not_captured():
    p = _StubProcess([], returncode=1)
    with pytest.raises(AstraSimChildDied) as excinfo:
        Controller(1).read_wait(p)
    assert "not captured to a file" in str(excinfo.value)


def test_message_survives_an_unreadable_stderr_path(tmp_path):
    """The hint must never be the reason the real error is lost."""
    p = _StubProcess([], returncode=1, stderr_path=tmp_path / "does_not_exist.log")
    with pytest.raises(AstraSimChildDied) as excinfo:
        Controller(1).read_wait(p)
    assert "exited with code 1" in str(excinfo.value)


def test_write_flush_translates_a_broken_pipe(tmp_path):
    """A killed child is often noticed on the *write*, not the read.

    Measured: killing AnalyticalAstra mid-run surfaced as a bare BrokenPipeError in
    write_flush, which ends the run promptly but names neither the exit code nor
    the stderr -- the half of D25-b that makes the failure readable.
    """
    err = tmp_path / "astra_stderr.log"
    err.write_text("terminate called without an active exception\n")

    class _ClosedStdin:
        def write(self, _):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    p = _StubProcess([], returncode=-9, stderr_path=err)
    p.stdin = _ClosedStdin()

    with pytest.raises(AstraSimChildDied) as excinfo:
        Controller(1).write_flush(p, "workload")
    message = str(excinfo.value)
    assert "exited with code -9" in message, message
    assert "terminate called" in message, "the child's stderr must reach the message"


def test_write_flush_is_unchanged_when_the_child_is_healthy():
    written = []

    class _OpenStdin:
        def write(self, s):
            written.append(s)

        def flush(self):
            pass

    p = _StubProcess([], returncode=None)
    p.stdin = _OpenStdin()
    Controller(1).write_flush(p, "workload")
    assert written == ["workload\n"]


# -- regression: the healthy path is unchanged ----------------------------

def test_read_wait_returns_the_same_list_as_before_on_the_waiting_line():
    lines = ["starting\n", "ring built\n", "Waiting\n"]
    p = _StubProcess(lines, returncode=None)
    out = Controller(1).read_wait(p)
    assert out == ["", *lines]


def test_read_wait_stops_on_the_non_exited_systems_line():
    lines = ["tick\n", "Checking Non-Exited Systems ...\n"]
    p = _StubProcess(lines, returncode=None)
    out = Controller(1).read_wait(p)
    assert out == ["", *lines]


def test_check_end_returns_the_same_list_as_before(capsys):
    lines = ["a\n", "b\n", "c\n", "All Request Has Been Exited\n", "trailing\n"]
    p = _StubProcess(lines, returncode=None)
    out = Controller(1).check_end(p)
    assert out[-2] == "All Request Has Been Exited\n"
    assert "".join(out) == "".join(["", "", *lines])


def test_a_blank_line_is_not_treated_as_eof():
    r"""A real empty line arrives as "\n"; only "" is EOF. Confusing the two would
    make the guard fire on healthy output."""
    lines = ["\n", "\n", "\n", "Waiting\n"]
    p = _StubProcess(lines, returncode=None)
    out = Controller(1).read_wait(p)
    assert out == ["", *lines]


def test_empty_read_counter_resets_on_real_output():
    """A stream that hiccups below the threshold and then recovers must not raise."""
    lines = ["", "", "progress\n", "", "Waiting\n"]
    p = _StubProcess(lines, returncode=None)
    out = Controller(1).read_wait(p)
    assert out[-1] == "Waiting\n"
