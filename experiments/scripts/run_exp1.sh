#!/bin/bash
# Exp 1 — same-GPU TP=1/2/4 sweep (work order §12 Exp 1).
#
# Validates the planner pipeline across tensor-parallel degrees on ONE real
# accelerator class (A40): TTFT/TPOT/power per TP, plus planner-level metrics
# (generated/rejected candidates, prune ratio). All numbers are LLMServingSim
# predictions on the MEASURED A40 profile bundle (profiler/perf/A40/, TP 1/2/4).
#
# The size-4 A40 island's intra bandwidth is the PCIe bottleneck (64 GB/s); the
# TP=2 row is conservative vs a dedicated NVLink pair (see the cluster file and
# experiments/results/exp1_summary.md).
#
# The simulator spawns Chakra as a bare `python -m chakra` subprocess, so the sim
# venv MUST be on PATH (calling the venv python by full path is not enough), and
# the repo root must be importable as `planner`. Real GPU hardware is NOT required
# to run the sweep: LLMServingSim is analytical and uses the profiled traces.
# (A GPU + .venv-vllm IS required to (re)profile TP=4; see docs/hardware_roadmap.)
#
# Usage:
#     ./experiments/scripts/run_exp1.sh
# Override via env, e.g. NUM_REQS=100 TP=1,2 ./experiments/scripts/run_exp1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SIM_VENV="${SIM_VENV:-.venv}"
NUM_REQS="${NUM_REQS:-300}"
SEED="${SEED:-42}"
TP="${TP:-1,2,4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

SERVICE="${SERVICE:-experiments/configs/services/llama31-8b-goodputj.yaml}"
CLUSTER="${CLUSTER:-experiments/configs/clusters/exp1-a40-tp-sweep.yaml}"
OUT_DIR="${OUT_DIR:-experiments/results}"
WORK_DIR="${WORK_DIR:-outputs/exp1}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[exp1] $*"; }

[[ -x "$SIM_VENV/bin/python" ]] || die "simulator venv not found at $SIM_VENV (see CLAUDE.md)."

# TP=4 needs the profiled bundle; fail early with a clear message if it is absent.
if [[ ",$TP," == *",4,"* ]]; then
  [[ -f "profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16/tp4/skew_fit.csv" ]] \
    || die "A40 TP=4 profile bundle missing. Profile it first:
    CUDA_VISIBLE_DEVICES=<gpu> .venv-vllm/bin/python -m profiler profile \\
      meta-llama/Llama-3.1-8B --hardware A40 --tp 1,4 --variant bf16
  or run with TP=1,2 to skip TP=4."
fi

export PATH="$REPO_ROOT/$SIM_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="$SIM_VENV/bin/python"

note "TP sweep {$TP} on $CLUSTER, $NUM_REQS requests -> $OUT_DIR"
"$PY" experiments/scripts/exp1_tp_sweep.py \
    --service "$SERVICE" \
    --cluster "$CLUSTER" \
    --tp "$TP" \
    --num-requests "$NUM_REQS" \
    --seed "$SEED" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --work-dir "$WORK_DIR" \
    --output-dir "$OUT_DIR"

note "done."
note "  table: $OUT_DIR/exp1_tp_sweep_table.md"
note "  json:  $OUT_DIR/exp1_tp_sweep.json"
