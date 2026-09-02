"""Synthetic (Tier 0/1) profile-bundle generation.

This package produces `profiler/perf/`-layout bundles from datasheet values
(Tier 0, ``analytical``) or datasheet + a small set of measured anchors
(Tier 1, ``calibrated``), per WORK_ORDER_tiered_profiles.md.

Import hygiene rule: no module in this package may import ``torch``,
``vllm``, or ``profiler.core.writer`` (which requires the GPU stack).
Tier 0 generation must run on a CPU-only machine with the base planner
venv. ``tests/test_synth_import_hygiene.py`` enforces this.
"""
