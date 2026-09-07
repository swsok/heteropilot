"""The ASTRA-Sim child must get its own working directory (deviations.md D25-a).

`WORK_ORDER_d23_fix_revalidation.md` STEP 1. ASTRA-Sim's analytical backend writes,
reads and removes `tmp__mem/<name>.json` relative to its cwd, with no pid and no run
id, three times per start. The frontend chdirs into `astra-sim/`, so before this fix
every concurrent simulation shared one such directory: measured, 13 of 64 processes
launched together died, five with `Unable to open file: tmp__mem/remote_mem.json`
and eight with SIGABRT (`docs/d23_spike.md`).

The fix is one argument, so the test is one property: the child is started with
`cwd` set to this run's own absolute inputs root. No simulator runs -- the fake
`Popen` records its keyword arguments and raises, which stops `serving.__main__`
exactly at the call under test. It must not return a normal-looking child either:
a fake whose `stdout` is at EOF would send `controller.read_wait` into the infinite
loop that is D25-b's subject, and the test would hang instead of failing.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO = Path(__file__).resolve().parents[1]

needs_astra = pytest.mark.skipif(
    not (REPO / "astra-sim" / "build").exists(),
    reason="ASTRA-Sim is not built in this checkout",
)


class _StopAtPopen(Exception):
    """Raised by the fake child so the run ends at the call under test."""


class _RecordingPopen(subprocess.Popen):
    """Records every child, and stops the run at the ASTRA-Sim one.

    Subclassing rather than wrapping because `serving` also shells out to the
    Chakra converter (`graph_generator.py:43`, via `subprocess.run`), and that
    child has to actually run -- the trace it produces is what the call under test
    then passes to ASTRA-Sim.
    """

    calls: ClassVar[list[dict]] = []

    def __init__(self, args, **kwargs):
        type(self).calls.append({"args": args, **kwargs})
        if "AnalyticalAstra" in str(args[0]):
            # Popen.__del__ inspects this even when __init__ never completed.
            self._child_created = False
            raise _StopAtPopen
        super().__init__(args, **kwargs)


def _popen_kwargs_for(monkeypatch, argv: list[str]) -> dict:
    _RecordingPopen.calls = []
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    monkeypatch.setattr(sys, "argv", ["serving", *argv])
    monkeypatch.chdir(REPO)

    with pytest.raises(_StopAtPopen):
        runpy.run_module("serving", run_name="__main__", alter_sys=True)

    # `serving` also shells out to the Chakra converter with cwd=<chakra dir>
    # (graph_generator.py:43). That call is unaffected by D25-a -- its --input and
    # --output are absolute and it writes nothing into its cwd -- but it happens
    # first, so select on the binary rather than taking calls[0].
    astra = [c for c in _RecordingPopen.calls if "AnalyticalAstra" in str(c["args"][0])]
    assert astra, (
        "serving never started the ASTRA-Sim child; calls were "
        f"{[c['args'][0] for c in _RecordingPopen.calls]}"
    )
    return astra[0]


def _base_argv(tmp_path: Path, run_id: str) -> list[str]:
    return [
        "--cluster-config", "configs/cluster/single_node_single_instance.json",
        "--dataset", "workloads/example_trace.jsonl",
        "--num-reqs", "1",
        "--output", str(tmp_path / f"{run_id}.csv"),
        "--run-id", run_id,
        "--no-cleanup-inputs",
    ]


@needs_astra
def test_child_gets_its_own_absolute_cwd(monkeypatch, tmp_path):
    call = _popen_kwargs_for(monkeypatch, _base_argv(tmp_path, "test-d25a"))
    cwd = call.get("cwd")
    assert cwd is not None, "the child was started without cwd -- D25-a is not applied"
    assert os.path.isabs(cwd), f"cwd must be absolute, got {cwd!r}"
    assert cwd.endswith(os.path.join("inputs", "runs", "test-d25a")), (
        f"cwd must be this run's own inputs root, got {cwd!r}"
    )
    assert os.path.isdir(cwd), "the directory must exist before the child starts"


@needs_astra
def test_cwd_follows_an_overridden_inputs_root(monkeypatch, tmp_path):
    """`--inputs-root` moves the tree; the child's cwd must move with it, or the
    isolation silently reverts to the shared directory."""
    root = tmp_path / "elsewhere"
    argv = [*_base_argv(tmp_path, "test-d25a-override"), "--inputs-root", str(root)]
    call = _popen_kwargs_for(monkeypatch, argv)
    assert call.get("cwd") == str(root.resolve()), (
        f"cwd should follow --inputs-root, got {call.get('cwd')!r}"
    )


@needs_astra
def test_two_runs_do_not_share_the_directory(monkeypatch, tmp_path):
    """The property the fix exists for, stated directly: two run ids, two cwds."""
    seen = {
        _popen_kwargs_for(monkeypatch, _base_argv(tmp_path, rid)).get("cwd")
        for rid in ("test-d25a-one", "test-d25a-two")
    }
    assert len(seen) == 2, f"both runs shared a working directory: {seen}"


@needs_astra
def test_every_astra_argument_is_absolute(monkeypatch, tmp_path):
    """Why one argument is enough: if the child's cwd were used for anything else,
    changing it would break that thing instead of fixing the race."""
    call = _popen_kwargs_for(monkeypatch, _base_argv(tmp_path, "test-d25a-abs"))
    argv = call["args"]
    assert os.path.isabs(argv[0]), f"binary path is relative: {argv[0]!r}"

    # Only the configuration flags carry paths. --start-npu-ids / --end-npu-ids
    # carry rank lists, so this selects by flag rather than by "contains =".
    path_flags = (
        "--workload-configuration",
        "--system-configuration",
        "--network-configuration",
        "--memory-configuration",
        "--logical-topology-configuration",
    )
    seen = 0
    for arg in argv[1:]:
        flag = arg.split("=", 1)[0]
        if flag in path_flags:
            seen += 1
            value = arg.split("=", 1)[1]
            assert os.path.isabs(value), f"relative path passed to the child: {arg!r}"
    assert seen == 4, f"expected the four configuration paths, saw {seen}: {argv[1:]}"
