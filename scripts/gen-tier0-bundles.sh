#!/usr/bin/env bash
# Regenerate the synthetic Tier 0 perf bundles (WORK_ORDER_tiered_profiles.md
# STEP 10). Synthetic bundles are NOT committed: they are deterministic
# functions of the accelerator datasheets + this generator, and keeping them
# out of git means measured and synthetic data can never mix in the tree
# (absolute rules A1/A3). Run this after cloning, before planning against any
# cluster that references a *-t0 hardware label.
#
# Determinism: GENERATED_AT is pinned so meta.yaml (and thus the whole
# bundle) is byte-identical across runs; override to stamp a real time.
set -euo pipefail
cd "$(dirname "$0")/.."

GENERATED_AT="${GENERATED_AT:-1970-01-01T00:00:00+00:00}"

# Ascend Tier 0 (deviations D4): the only model the profile declares support
# for is Qwen3; the example hetero cluster's islands have 2 NPUs -> TP 1,2.
python -m profiler.synth emit \
  --accelerator profiles/accelerators/ascend_target.yaml \
  --model Qwen/Qwen3-32B --variant bf16 --tp 1,2 \
  --hardware-label ASCEND_TARGET-t0 \
  --generated-at "$GENERATED_AT" \
  --out profiler/perf --force

# A40 Tier 0 twin of the measured bundle (experiment E1's tier0 leg):
# mirror keys from the measured A40 bundle, efficiencies from the STEP 8 fit.
python -m profiler.synth emit \
  --accelerator profiles/accelerators/a40.yaml \
  --efficiency profiles/accelerators/a40.efficiency.yaml \
  --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1,2,4 \
  --hardware-label A40-t0 \
  --mirror-keys profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 \
  --generated-at "$GENERATED_AT" \
  --out profiler/perf --force

echo "Tier 0 bundles regenerated under profiler/perf/ (gitignored)."
