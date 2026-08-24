#!/bin/bash
# Exp: router baselines RR / RAND / LOAD (work order §12).
#
# Routing policy is a simulator input, so this axis RE-SIMULATES the same
# multi-replica deployment once per policy (unlike the replay-based
# exp_baselines.py). All numbers are LLMServingSim predictions.
#
# The simulator spawns Chakra as `python -m chakra`, so the sim venv must be on
# PATH and the repo root importable as `planner`. No live GPU required.
#
# Usage: ./experiments/scripts/run_exp_router.sh
# Override via env, e.g. NUM_REQS=300 CLUSTER=... ./experiments/scripts/run_exp_router.sh

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
WORK_DIR="${WORK_DIR:-outputs/exp_router}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$SIM_VENV/bin/python" ]] || die "simulator venv not found at $SIM_VENV (see CLAUDE.md)."

export PATH="$REPO_ROOT/$SIM_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[exp-router] $CLUSTER, $NUM_REQS requests, RR/RAND/LOAD -> $OUT_DIR"
"$SIM_VENV/bin/python" experiments/scripts/exp_router.py \
    --service "$SERVICE" --cluster "$CLUSTER" \
    --num-requests "$NUM_REQS" --seed "$SEED" \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --work-dir "$WORK_DIR" --output-dir "$OUT_DIR"

echo "[exp-router] done."
echo "  table: $OUT_DIR/router_baselines_table.md"
echo "  json:  $OUT_DIR/router_baselines.json"
