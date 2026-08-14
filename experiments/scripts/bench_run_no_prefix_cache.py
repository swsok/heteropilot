#!/usr/bin/env python3
"""Run `bench run` with vLLM prefix caching disabled.

Why this exists: on a 24 GB card the simulator cannot complete the sharegpt-300
workload with prefix caching enabled - it raises out of its own memory model in
the prefix-cache accounting path (docs/deviations.md D12). The simulator *can*
complete it with `--no-enable-prefix-caching`, but comparing that against a real
vLLM run with prefix caching ON would be comparing two different systems.

`bench/core/runner.py` builds `AsyncEngineArgs` without passing
`enable_prefix_caching`, so it inherits vLLM's default of True, and the bench CLI
exposes no flag for it. Rather than edit upstream `bench/` (forbidden before
Phase 4), this wrapper patches the symbol the runner imports at call time and
then delegates to the real runner. Everything else - dataset handling, stat
logging, artifact writing - is upstream's code, unmodified.

Usage: same arguments as `python -m bench run`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts experiments/scripts/ on sys.path, not the repo
# root, so `bench` would not import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import vllm


def main(argv: list[str]) -> int:
    original = vllm.AsyncEngineArgs

    def patched(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("enable_prefix_caching", False)
        return original(*args, **kwargs)

    # runner.py does `from vllm import AsyncEngineArgs` inside the function body,
    # so replacing the attribute on the module is enough and is undone on exit.
    vllm.AsyncEngineArgs = patched  # type: ignore[assignment]
    saved_argv = sys.argv
    try:
        from bench.__main__ import main as bench_main

        # bench's main() reads sys.argv directly; it takes no argv parameter.
        sys.argv = ["bench", "run", *argv]
        return int(bench_main() or 0)
    finally:
        sys.argv = saved_argv
        vllm.AsyncEngineArgs = original  # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
