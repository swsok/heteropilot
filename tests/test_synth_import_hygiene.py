"""Guarantees profiler.synth imports without the GPU stack (STEP 0).

These tests are the standing regression guard for the tiered-profile work:
Tier 0 bundle generation must run on CPU-only machines whose venv has no
torch/vllm. Never delete them.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

SYNTH_DIR = Path(__file__).resolve().parent.parent / "profiler" / "synth"

#: Modules profiler.synth must never import, directly or at module level.
FORBIDDEN = {"torch", "vllm", "profiler.core.writer"}


class _BlockingFinder:
    """Meta-path finder that makes torch/vllm imports fail loudly."""

    def find_module(self, fullname: str, path=None):  # pragma: no cover - protocol
        return self.find_spec(fullname, path)

    def find_spec(self, fullname: str, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"torch", "vllm"}:
            raise ImportError(f"blocked by test: {fullname}")
        return None


def test_synth_imports_without_torch_or_vllm(monkeypatch):
    """profiler.synth (all submodules) imports with torch/vllm blocked."""
    # Ensure a cached torch/vllm from another test cannot mask a real import.
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in {"torch", "vllm"} or name.startswith("profiler.synth"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockingFinder(), *sys.meta_path])

    package = importlib.import_module("profiler.synth")
    for info in pkgutil.walk_packages(package.__path__, prefix="profiler.synth."):
        importlib.import_module(info.name)


def test_no_writer_import_in_synth():
    """No profiler.synth module imports torch, vllm, or profiler.core.writer."""
    py_files = sorted(SYNTH_DIR.rglob("*.py"))
    assert py_files, "profiler/synth must exist and contain at least __init__.py"

    offenders: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                names += [f"{node.module}.{alias.name}" for alias in node.names]
            for name in names:
                root = name.split(".")[0]
                if root in {"torch", "vllm"} or any(
                    name == bad or name.startswith(bad + ".") for bad in FORBIDDEN
                ):
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"forbidden imports in profiler.synth: {offenders}"


def test_forbidden_finder_blocks_torch():
    """The blocking finder actually raises for torch (guards the guard)."""
    finder = _BlockingFinder()
    with pytest.raises(ImportError):
        finder.find_spec("torch.nn")
    assert finder.find_spec("json") is None
