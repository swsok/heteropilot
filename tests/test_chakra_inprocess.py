"""The in-process converter must produce exactly what the subprocess did (D27).

`WORK_ORDER_rps_aware.md` STEP 1. `graph_generator.py` used to shell out to
`chakra.src.converter.converter` once per instance per iteration. Profiling put
that at 28.9 % of wall time at 10 rps, of which only 7.4 ms per call is the
conversion and 38.6 ms is fork, exec, interpreter start and imports
(`docs/sim_cost_profile.md`). D27 calls the converter directly.

The speedup is worth nothing if the bytes change, so that is what is asserted --
against the subprocess path itself, not against a stored digest, so the test stays
true if the vendored chakra is ever updated.

Two further properties are checked because they are what make the direct call
safe rather than merely fast:

  - repeated calls agree, i.e. the converter carries no module-level state (the
    work order flagged this as the risk that would force a revert);
  - the frontend's root logger is untouched, and no `debug.log` is dropped into
    the frontend's working directory. `converter.main()` calls `setup_logging()`,
    which does both -- `logging.basicConfig` reconfigures logging for the whole
    process, and its default `--log-filename debug.log` is cwd-relative. The
    subprocess absorbed both because it ran with `cwd=` the chakra directory; an
    in-process `main()` would not. That is why D27 bypasses `main()`.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHAKRA = REPO / "astra-sim" / "extern" / "graph_frontend" / "chakra"
FIXTURE = REPO / "tests" / "data" / "chakra_llm_trace.txt"

needs_chakra = pytest.mark.skipif(
    not CHAKRA.exists(), reason="the chakra submodule is not checked out"
)


def _digest(prefix: Path) -> str:
    """Hash the .et set by rank suffix, never by filename.

    The output prefix differs between the two paths by construction, so hashing
    names would make every comparison fail for the wrong reason.
    """
    files = {p.name[len(prefix.name):]: p for p in prefix.parent.glob(prefix.name + "*")}
    assert files, f"the converter produced nothing at {prefix}"
    h = hashlib.sha256()
    for suffix in sorted(files):
        h.update(suffix.encode())
        h.update(files[suffix].read_bytes())
    return h.hexdigest()


def _subprocess_convert(out: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "chakra.src.converter.converter", "LLM",
         "--input", str(FIXTURE), "--output", str(out),
         "--num-npus", "4", "--npu-offset", "0"],
        cwd=CHAKRA, capture_output=True, text=True, check=True, timeout=300,
    )


def _inprocess_convert(out: Path) -> None:
    from serving.core.graph_generator import _llm_converter
    _llm_converter()(str(FIXTURE), str(out), 4, 0, False).convert()


@needs_chakra
def test_in_process_output_matches_the_subprocess(tmp_path):
    sp, ip = tmp_path / "sp", tmp_path / "ip"
    _subprocess_convert(sp)
    _inprocess_convert(ip)
    assert _digest(ip) == _digest(sp), (
        "the in-process converter produced different .et bytes; D27 is not a "
        "no-op and must not be merged"
    )


@needs_chakra
def test_repeated_calls_agree(tmp_path):
    """No module-level state: the second call must not inherit the first's."""
    digests = set()
    for i in range(3):
        out = tmp_path / f"r{i}"
        _inprocess_convert(out)
        digests.add(_digest(out))
    assert len(digests) == 1, f"repeated conversions diverged: {digests}"


@needs_chakra
def test_the_frontends_logging_is_not_reconfigured(tmp_path):
    """Why D27 calls LLMConverter and not converter.main()."""
    before = (logging.root.level, list(logging.root.handlers))
    _inprocess_convert(tmp_path / "log")
    after = (logging.root.level, list(logging.root.handlers))
    assert before == after, (
        "converting reconfigured the root logger; that is what setup_logging() in "
        "converter.main() does, and D27 exists partly to avoid it"
    )


@needs_chakra
def test_no_log_file_is_dropped_in_the_working_directory(tmp_path, monkeypatch):
    """The other half of why D27 calls LLMConverter and not converter.main().

    `setup_logging()` defaults to a cwd-relative `debug.log`. In a subprocess run
    with `cwd=chakra/` that lands out of the way; called in-process it would land
    in the repo root, once per instance per iteration.
    """
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    _inprocess_convert(tmp_path / "nolog")
    stray = sorted(p.name for p in workdir.iterdir())
    assert stray == [], f"converting wrote into the frontend's cwd: {stray}"


def test_graph_generator_does_not_shell_out():
    """The defect D27 removes, asserted on the source so it cannot creep back."""
    src = (REPO / "serving" / "core" / "graph_generator.py").read_text()
    # comments mention both on purpose, so assert on code only
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "subprocess.run" not in code, "graph_generator still spawns a converter"
    assert "'python'" not in code, "a PATH-resolved interpreter is back (D26)"
