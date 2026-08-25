#!/bin/bash
# Exp: surrogate top-K accuracy (work order §5.4 stage 6).
#
# Simulate every candidate once, then MEASURE (never assert) the surrogate top-K
# recall/regret/speedup vs K against the exhaustive oracle. All numbers are
# LLMServingSim predictions; placeholder-profile islands are excluded and counted.
#
# The simulator spawns Chakra as `python -m chakra`, so the sim venv must be on
# PATH and the repo root importable as `planner`. No live GPU required.
#
# Usage: ./experiments/scripts/run_exp_surrogate.sh
# Override via env, e.g. NUM_REQS=300 CLUSTER=... ./experiments/scripts/run_exp_surrogate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SIM_VENV="${SIM_VENV:-.venv}"
NUM_REQS="${NUM_REQS:-120}"
SEED="${SEED:-42}"
SERVICE="${SERVICE:-examples/service_specs/llama31-8b-light.yaml}"
CLUSTER="${CLUSTER:-experiments/configs/clusters/exp2-local-lab.yaml}"
OUT_DIR="${OUT_DIR:-experiments/results}"
WORK_DIR="${WORK_DIR:-outputs/exp_surrogate}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$SIM_VENV/bin/python" ]] || die "simulator venv not found at $SIM_VENV (see CLAUDE.md)."

export PATH="$REPO_ROOT/$SIM_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[exp-surrogate] $CLUSTER, $NUM_REQS requests -> $OUT_DIR"
"$SIM_VENV/bin/python" experiments/scripts/exp_surrogate.py \
    --service "$SERVICE" --cluster "$CLUSTER" \
    --num-requests "$NUM_REQS" --seed "$SEED" \
    --k-values 1 2 3 5 10 20 40 \
    --work-dir "$WORK_DIR" --output-dir "$OUT_DIR"

echo "[exp-surrogate] done."
echo "  table: $OUT_DIR/surrogate_table.md"
echo "  json:  $OUT_DIR/surrogate.json"
