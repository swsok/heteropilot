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

    **The check was `any('.local' in p for p in sys.path)` and that was a proxy for
    the wrong thing.** It fails on the A40 node, where `uv` installs the
    INTERPRETER under `~/.local/share/uv/python/cpython-3.10-.../lib/python3.10`:
    those entries are the standard library, they are on every venv's path by
    construction, and no chakra can be dropped into them by a stray `pip install
    --user`. `~/.local/lib/python3.10/site-packages` -- the directory D26 was
    actually about -- is absent and `ENABLE_USER_SITE` is False, so the property
    held on the node where the assertion failed. It passed on the NPU node only
    because its interpreter happens to live outside `~/.local`.

    So the substring is replaced by the two things that are actually load-bearing:
    the user-site directory is not importable, and the chakra this interpreter
    resolves lives inside this interpreter's own prefix. The second is the
    conclusion rather than a proxy for it -- whatever path a future layout puts the
    wrong chakra on, an import from outside the venv fails here.
    """
    proc = subprocess.run(
        # The module the frontend actually imports (serving/core/graph_generator.py
        # line 21), not the `chakra` namespace above it -- that one is a namespace
        # package whose `__file__` is None, so it cannot answer "from where".
        [sys.executable, "-c",
         "import site, sys;"
         "from chakra.src.converter.llm_converter import LLMConverter;"
         "u = site.getusersitepackages();"
         "print(site.ENABLE_USER_SITE);"
         "print(u in sys.path);"
         "print(sys.prefix);"
         "print(sys.modules[LLMConverter.__module__].__file__)"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    enabled, user_site_on_path, prefix, chakra_file = proc.stdout.split("\n")[:4]

    assert enabled == "False" and user_site_on_path == "False", (
        "the venv can see the user site-packages directory, so a chakra installed "
        f"there could shadow or substitute for the venv's (D26): "
        f"ENABLE_USER_SITE={enabled}, user site on sys.path={user_site_on_path}"
    )
    # The decisive one: not "could the wrong chakra be reachable" but "is the one
    # we resolved the venv's". D27 converts with whatever this interpreter imports.
    assert chakra_file.startswith(prefix), (
        f"this interpreter's chakra resolves OUTSIDE its own prefix (D26): "
        f"{chakra_file} is not under {prefix}. The frontend converts in-process "
        "with exactly this module, so the .et bytes would come from it."
    )
