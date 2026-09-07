"""The Chakra converter must run under the frontend's own interpreter (D26).

`WORK_ORDER_d23_fix_revalidation.md`, scope extended 2026-09-07. Before this fix
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

So the property is: the converter command names `sys.executable`, never a
PATH-resolved name.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "serving" / "core" / "graph_generator.py"


def test_converter_command_uses_sys_executable():
    """Read the source rather than run a simulation: the defect is one token."""
    tree = ast.parse(SOURCE.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        # the converter command list is the one naming the chakra module
        literals = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if "chakra.src.converter.converter" not in literals:
            continue
        first = node.elts[0]
        found.append(first)
        assert not isinstance(first, ast.Constant), (
            f"the converter is invoked as the literal {first.value!r}; a PATH-resolved "
            "name picks up whichever interpreter comes first, which is D26/D23"
        )
        assert ast.unparse(first) == "sys.executable", (
            f"expected sys.executable, got {ast.unparse(first)}"
        )
    assert found, "no chakra converter command found in graph_generator.py"


def test_sys_is_imported():
    tree = ast.parse(SOURCE.read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "sys" in names, "graph_generator.py uses sys.executable but does not import sys"


def test_this_interpreter_can_actually_convert():
    """`sys.executable` is only the right answer if this interpreter has a working
    chakra. If the venv were mis-provisioned the fix would move the failure rather
    than remove it, so assert the import the converter needs."""
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
