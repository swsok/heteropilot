"""`experiments/scripts/astra_isolated.sh` — the D23 workaround's contract.

WORK_ORDER_spikes.md STEP A. ASTRA-Sim writes `tmp__mem/<name>.json` at a fixed
cwd-relative path, so concurrent simulations delete each other's file and die
(measured: 13 of 64). This wrapper gives each process a private tmpfs over that
one directory.

Two properties matter and both are tested here:

  - it actually isolates, so concurrent runs stop colliding;
  - it **refuses** rather than degrading when isolation is unavailable. Running
    unisolated would reintroduce the race, whose failure mode is a four-hour
    timeout rather than an error, so a silent fallback would be worse than none.

No simulator: the wrapper is exercised with `sh`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAP = REPO / "experiments" / "scripts" / "astra_isolated.sh"
TARGET = REPO / "astra-sim" / "tmp__mem"


def _isolation_available() -> bool:
    return subprocess.run(
        [str(WRAP), "--check"], capture_output=True, text=True, timeout=60
    ).returncode == 0


needs_userns = pytest.mark.skipif(
    not _isolation_available(),
    reason="unprivileged user+mount namespaces are unavailable on this node",
)


def test_check_reports_a_definite_verdict():
    """--check must answer 0 or 2, never leave the caller guessing."""
    proc = subprocess.run([str(WRAP), "--check"], capture_output=True, text=True, timeout=60)
    assert proc.returncode in (0, 2)
    if proc.returncode == 0:
        assert "available" in proc.stdout
    else:
        assert "unshare" in proc.stderr or "namespaces" in proc.stderr


def test_missing_separator_is_refused():
    """Without `--` the command would be swallowed; refuse rather than guess."""
    proc = subprocess.run([str(WRAP), "echo", "hi"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert "usage" in proc.stderr


def test_no_command_is_refused():
    proc = subprocess.run([str(WRAP), "--"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2


@needs_userns
def test_writes_do_not_escape_the_namespace():
    """The property the whole thing exists for."""
    before = sorted(p.name for p in TARGET.iterdir()) if TARGET.exists() else []
    proc = subprocess.run(
        [str(WRAP), "--", "sh", "-c", f'touch "{TARGET}/probe_from_test"; ls "{TARGET}"'],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "probe_from_test" in proc.stdout, "the write should succeed inside"
    after = sorted(p.name for p in TARGET.iterdir()) if TARGET.exists() else []
    assert after == before, f"the write escaped: {set(after) - set(before)}"


@needs_userns
def test_two_runs_do_not_share_the_directory():
    """Concurrency is the point: what one writes must be invisible to the other,
    which is exactly what ASTRA-Sim's fixed path violates."""
    procs = [
        subprocess.Popen(
            [str(WRAP), "--", "sh", "-c",
             f'touch "{TARGET}/from_{i}"; sleep 1; ls "{TARGET}"'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for i in (1, 2)
    ]
    outs = [p.communicate(timeout=60) for p in procs]
    for i, (out, err) in zip((1, 2), outs, strict=True):
        assert f"from_{i}" in out, err
        other = 2 if i == 1 else 1
        assert f"from_{other}" not in out, "the two runs shared the directory"


@needs_userns
def test_exit_code_passes_through():
    """A failing simulation must not be reported as a wrapper failure."""
    proc = subprocess.run(
        [str(WRAP), "--", "sh", "-c", "exit 9"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 9


@needs_userns
def test_grandchildren_inherit_the_namespace():
    """The real caller is `python -m serving`, which spawns AnalyticalAstra. The
    isolation is worthless unless it reaches that grandchild."""
    proc = subprocess.run(
        [str(WRAP), "--", "sh", "-c",
         f'sh -c \'touch "{TARGET}/from_grandchild"\'; ls "{TARGET}"'],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "from_grandchild" in proc.stdout
    assert not (TARGET / "from_grandchild").exists(), "the grandchild's write escaped"
