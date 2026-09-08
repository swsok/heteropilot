"""The Chakra converter must run under the frontend's own interpreter (D26, D27).

`WORK_ORDER_d23_fix_revalidation.md`, scope extended 2026-09-07. Before D26
`serving/core/graph_generator.py` invoked the converter as bare `python`, so which
interpreter built the ASTRA-Sim workload graph depended on the caller's PATH.

That is not cosmetic. On this node `python` resolves to `~/.local/bin/python`,
whose `chakra` sits beside protobuf 6.33.1 -- below the `>=7.35.1` CLAUDE.md
requires for Chakra's generated code -- while the venv has 7.36.0. The same trace
converted by the two produces **different .et bytes** (measured), and a P/D
simulation built from the wrong ones never finishes its first prefill batch:
prefill pinned at one request, decode never fed, memory flat, the simulated clock
racing. That is D23's signature, and it reproduced 6/6 under the bare-`python`
PATH against 18/18 completions with the venv first.

**D27 changed how this file guards that.** There is no longer a converter command
to inspect -- the converter is called in-process, so the interpreter question is
answered by construction and the two AST tests that lived here (the command names
`sys.executable`; `sys` is imported) were deleted with the construct they described.
`tests/test_chakra_inprocess.py::test_graph_generator_does_not_shell_out` is what
now keeps a spawned interpreter from coming back.

What survives is the environment check, and D27 makes it matter more rather than
less: the frontend converts with whatever `chakra` *it* can import, so a
mis-provisioned venv is now the only way to get the wrong bytes.
"""

from __future__ import annotations

import subprocess
import sys


def test_this_interpreter_can_actually_convert():
    """The frontend converts in-process, so its own chakra is the one that counts."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import chakra, google.protobuf as pb;"
         "v=tuple(int(x) for x in pb.__version__.split('.')[:2]);"
         "assert v >= (7, 35), pb.__version__;"
         "print('ok', pb.__version__)"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        "the interpreter running the tests cannot satisfy the converter's "
        f"requirements: {proc.stderr.strip()}\n"
        "CLAUDE.md: uv pip install ./astra-sim/extern/graph_frontend/chakra "
        'and uv pip install "protobuf>=7.35.1"'
    )


def test_there_is_no_user_site_fallback():
    """Why an ImportError here is the strictly better failure than D26's silence.

    D26 was dangerous because a wrong chakra was reachable and succeeded quietly.
    In-process, a venv without chakra raises -- but only if the wrong one is not on
    `sys.path` to be found instead. That is a property of the venv, so assert it.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "import site, sys;"
         "print(site.ENABLE_USER_SITE, any('.local' in p for p in sys.path))"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    enabled, local_on_path = proc.stdout.split()
    assert enabled == "False" and local_on_path == "False", (
        "the venv can see ~/.local/lib, so a chakra installed there could shadow or "
        f"substitute for the venv's (D26): ENABLE_USER_SITE={enabled}, "
        f"'.local' on sys.path={local_on_path}"
    )
