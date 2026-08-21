#!/bin/bash
# Phase 5 increment 4 experiments — P/D network sweep + 4-combo comparison.
# docs/phase5_plan.md increment 4; work order §12 Exp 3 (sweep) + Exp 5 (combos).
#
# Two experiments, both PLANNER-SIDE for the bandwidth effect:
#   1. Network sweep (the headline). The simulator charges the prefill->decode KV
#      transfer as free (docs/phase5_plan.md increment 2), so the whole sweep effect
#      is the planner's analytical transfer term. The driver simulates every
#      candidate ONCE, caches the raw metrics, then re-evaluates each bandwidth by
#      re-running only the planner-side transfer cost + feasibility + ranking. It
#      does NOT re-simulate per bandwidth.
#   2. 4-combo P/D comparison. GPU-P+GPU-D runs on the measured A5000 profile; the
#      three NPU-touching combos are SIM-PROXY (A5000 compute model standing in for
#      an NPU — NOT an NPU measurement; see ascend-sim-proxy.yaml).
#
# The simulator spawns Chakra as a bare `python -m chakra` subprocess, so the sim
# venv MUST be on PATH (calling the venv python by full path is not enough). The
# driver stages every run under an isolated --run-id inputs-root, so no locking is
# needed. Real GPU/NPU hardware is NOT required: LLMServingSim is analytical and
# uses the profiled traces, not model weights.
#
# Usage:
#     ./experiments/scripts/run_exp_pd.sh
# Override any variable via the environment, e.g. NUM_REQS=200 ./...run_exp_pd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SIM_VENV="${SIM_VENV:-.venv}"
# 120 requests reproduces the committed experiments/results/ numbers; the crossing
# is insensitive to request count (it is set by the p99 prompt KV / fabric bandwidth).
NUM_REQS="${NUM_REQS:-120}"
SEED="${SEED:-42}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
BANDWIDTHS="${BANDWIDTHS:-400 200 100 25 10 1}"

SWEEP_SERVICE="experiments/configs/services/pd-sweep-llama31-8b.yaml"
SWEEP_CLUSTER="experiments/configs/clusters/pd-network-sweep.yaml"
COMBO_CLUSTER="experiments/configs/clusters/pd-4combo-sim.yaml"

SWEEP_OUT="${SWEEP_OUT:-outputs/.hp-pd-sweep}"
COMBO_OUT="${COMBO_OUT:-outputs/.hp-pd-combo}"
FIGURE="${FIGURE:-experiments/figures/pd_network_sweep.png}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[exp-pd] $*"; }

[[ -x "$SIM_VENV/bin/python" ]] || die "simulator venv not found at $SIM_VENV (see CLAUDE.md)."

# The nested `python -m chakra` needs the venv on PATH, and the driver needs the
# repo root importable as `planner`.
export PATH="$REPO_ROOT/$SIM_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="$SIM_VENV/bin/python"

note "1/2 network sweep -> $SWEEP_OUT (simulate once, then sweep {$BANDWIDTHS} GB/s)"
# shellcheck disable=SC2086
"$PY" experiments/scripts/pd_network_sweep.py \
    --service "$SWEEP_SERVICE" \
    --cluster "$SWEEP_CLUSTER" \
    --fabric-link fabric-node0-node1 \
    --bandwidths $BANDWIDTHS \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --num-requests "$NUM_REQS" \
    --seed "$SEED" \
    --output-dir "$SWEEP_OUT" \
    --figure "$FIGURE"

note "2/2 4-combo comparison -> $COMBO_OUT (GPU-P+GPU-D real; NPU combos SIM-PROXY)"
"$PY" experiments/scripts/pd_combo_compare.py \
    --service "$SWEEP_SERVICE" \
    --cluster "$COMBO_CLUSTER" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --num-requests "$NUM_REQS" \
    --seed "$SEED" \
    --output-dir "$COMBO_OUT"

note "done."
note "  sweep table:   $SWEEP_OUT/pd_network_sweep_table.md"
note "  sweep json:    $SWEEP_OUT/pd_network_sweep.json"
note "  combo table:   $COMBO_OUT/pd_4combo_table.md"
note "  figure:        $FIGURE (skipped if matplotlib absent)"
