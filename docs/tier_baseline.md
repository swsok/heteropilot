# Tiered-profile work baseline (STEP 0)

Snapshot of the quality gates on `main` immediately before the tiered-profile
work (`WORK_ORDER_tiered_profiles.md`) began. Recorded per STEP 0 so later
steps can prove they did not regress anything that was green here.

- Date: 2026-09-02
- Commit: `c7dbf8ea89d9a2df2f7ba5ffd45268cb44f61325`

## Gate results

```
$ pytest -q 2>&1 | tail -5
366 passed, 1 warning in 43.04s

$ ruff check . 2>&1 | tail -3
All checks passed!

$ mypy 2>&1 | tail -3
Success: no issues found in 51 source files
```

## Pre-work investigation — divergences from WORK_ORDER_tiered_profiles.md §3

The work order instructs verifying its §3 investigation table before starting.
Two entries diverge from the actual code (verified 2026-09-02 on the commit
above, in the project `.venv`, which has no `torch`/`vllm`):

1. **`profiler/core/categories.py` is NOT importable without torch.**
   The table calls it "pure Python (no torch/vllm import)". Its own module
   body is pure, but line 28 does `from profiler.core.engine import
   RuntimeLimits`, and `engine.py` imports `torch` and `vllm` at module level.
   `import profiler.core.categories` therefore raises `ModuleNotFoundError:
   No module named 'torch'` in this venv. `RuntimeLimits` is used only in
   type annotations (the file has `from __future__ import annotations`), so a
   `TYPE_CHECKING` guard on that one import makes the module importable
   without the GPU stack while changing no behavior. That guard is applied in
   the STEP that first needs to import categories from `profiler.synth`
   (STEP 7), not here.

2. **`profiler/core/writer.py` does not import torch/vllm at module level.**
   The table says it does. Its module-level imports are pure; the
   `_vllm_version` / `_cuda_version` / `_gpu_name` helpers import torch/vllm
   *inside* the functions with try/except. `writer.py` is still not importable
   without torch, because it imports `categories.py` (divergence 1 above).
   The work order's design decision stands unchanged either way: `synth` does
   not reuse `writer.py` and writes CSVs itself, validated by
   `profiler/core/contract.py`.
