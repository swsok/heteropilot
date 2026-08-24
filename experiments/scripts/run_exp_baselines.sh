#!/bin/bash
# Exp: baselines + ablation (work order §12).
#
# Simulate-once, replay-select: one oracle-mode pass simulates every simulatable
# candidate, then all optimizer/resource/architecture baselines and replay-able
# ablations are scored over the shared cache (regret vs the oracle optimum). All
# numbers are LLMServingSim predictions; placeholder-profile (NPU stub) islands
# are excluded and counted. See experiments/results/exp_baselines_summary.md.
#
# The simulator spawns Chakra as `python -m chakra`, so the sim venv must be on
# PATH and the repo root importable as `planner`. Real GPUs are NOT required.
#
# Usage: ./experiments/scripts/run_exp_baselines.sh
# Override via env, e.g. NUM_REQS=300 CLUSTER=... ./experiments/scripts/run_exp_baselines.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SIM_VENV="${SIM_VENV:-.venv}"
NUM_REQS="${NUM_REQS:-120}"
SEED="${SEED:-42}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
SERVICE="${SERVICE:-experiments/configs/services/llama31-8b-goodputj.yaml}"
CLUSTER="${CLUSTER:-experiments/configs/clusters/exp2-local-lab.yaml}"
OUT_DIR="${OUT_DIR:-experiments/results}"
WORK_DIR="${WORK_DIR:-outputs/exp_baselines}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$SIM_VENV/bin/python" ]] || die "simulator venv not found at $SIM_VENV (see CLAUDE.md)."

export PATH="$REPO_ROOT/$SIM_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[exp-baselines] $CLUSTER, $NUM_REQS requests -> $OUT_DIR"
"$SIM_VENV/bin/python" experiments/scripts/exp_baselines.py \
    --service "$SERVICE" \
    --cluster "$CLUSTER" \
    --num-requests "$NUM_REQS" \
    --seed "$SEED" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --work-dir "$WORK_DIR" \
    --output-dir "$OUT_DIR"

echo "[exp-baselines] done."
echo "  table: $OUT_DIR/baselines_table.md"
echo "  json:  $OUT_DIR/baselines.json"
