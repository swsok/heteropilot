#!/bin/bash
# A40 sim-vs-real validation — one-command runner (RUN LATER, on the A40 server).
#
# Roadmap step 4 / Phase 4 entry (docs/hardware_roadmap.md, HANDOVER.md §7 step 4,
# docs/a40_sim_vs_real_plan.md). Repeats the A5000 protocol on the real A40x8
# migration server: measure real vLLM, simulate the matching config at BOTH a
# nominal and a KV-matched mem_size, and compare — so the D10 memory-accounting
# error is separated from profile error.
#
# THIS SCRIPT RUNS NOTHING AT AUTHORING TIME. It only acts when executed, and it
# refuses to start until every prerequisite exists (A40 perf bundle, filled-in
# KV-matched mem_size, both venvs, the dataset). Each missing piece prints the
# exact command that produces it, then exits non-zero.
#
# Usage (on the A40 server, once profiling + power measurement are done):
#     CUDA_VISIBLE_DEVICES=0 ./experiments/scripts/run_a40_sim_vs_real.sh
#
# Override any variable via the environment, e.g.:
#     CUDA_VISIBLE_DEVICES=2 MODEL=meta-llama/Llama-3.1-8B \
#     DATASET=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
#     ./experiments/scripts/run_a40_sim_vs_real.sh
#
# Set SKIP_KVMATCHED=1 to run only the nominal side before the KV budget has been
# measured (produces a partial result; the headline nominal-vs-KV-matched
# comparison needs both).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# PARAMETERS — defaults mirror the A5000 / bench-example protocol exactly.
# (bench/examples/configs, outputs/phase0_bench/A5000*/vllm/meta.json,
#  docs/phase0_bench_plan.md §1/§2b). Change only with reason.
# =============================================================================
HARDWARE="${HARDWARE:-A40}"
# MODEL drives the SIMULATOR side: it must match a profiler/perf/<HW>/<MODEL>/
# bundle and a configs/model/<MODEL>.json, so it stays the canonical (gated) id.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
# BENCH_MODEL drives the REAL vLLM side, which loads weights. meta-llama is gated
# and its weights are not cached here, so default to the ungated NousResearch
# mirror (identical weights + tokenizer -> identical compute, a valid
# comparison). Override to MODEL once the gated weights are available. The bench
# runs with HF_HUB_OFFLINE by default since the mirror is already cached.
BENCH_MODEL="${BENCH_MODEL:-NousResearch/Meta-Llama-3.1-8B}"
DATASET="${DATASET:-workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl}"

TP="${TP:-1}"
DP="${DP:-1}"
NUM_REQS="${NUM_REQS:-300}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SEED="${SEED:-42}"
TICK_SECONDS="${TICK_SECONDS:-1.0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NETWORK_BACKEND="${NETWORK_BACKEND:-analytical}"

# Venvs (HANDOVER.md §6: two separate venvs — do not mix).
SIM_VENV="${SIM_VENV:-.venv}"                     # simulator (python -m serving / bench validate)
VLLM_VENV="${VLLM_VENV:-.venv-vllm}"              # real vLLM (python -m bench run)

# Cluster configs authored alongside this script.
NOMINAL_CONFIG="experiments/configs/clusters/a40-llama31-8b-tp1.json"
KVMATCHED_CONFIG="experiments/configs/clusters/a40-llama31-8b-tp1-kvmatched.json"
KV_SENTINEL="TODO_MEASURE_A40_KV_BUDGET_GB"

# Output layout mirrors outputs/phase0_bench/A5000-np-*.
OUT_ROOT="${OUT_ROOT:-outputs/phase0_bench}"
VLLM_DIR="$OUT_ROOT/$HARDWARE/vllm"
NOMINAL_DIR="$OUT_ROOT/$HARDWARE-nominal"
KVMATCHED_DIR="$OUT_ROOT/$HARDWARE-kvmatched"

SKIP_KVMATCHED="${SKIP_KVMATCHED:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[a40-sim-vs-real] $*"; }

# =============================================================================
# PREFLIGHT — refuse to run until prerequisites exist. Nothing below touches a
# GPU or the simulator; every check is a filesystem / grep test.
# =============================================================================
note "preflight checks ..."

[[ -x "$VLLM_VENV/bin/python" ]] || die \
  "vLLM venv not found at $VLLM_VENV. Create it (HANDOVER.md §6):
     uv venv --python 3.12 $VLLM_VENV
     VLLM_USE_PRECOMPILED=1 uv pip install --python $VLLM_VENV/bin/python vllm==0.19.0 --no-build-isolation
     uv pip install --python $VLLM_VENV/bin/python datasets matplotlib pandas"

[[ -x "$SIM_VENV/bin/python" ]] || die \
  "simulator venv not found at $SIM_VENV. Create it and build ASTRA-Sim (HANDOVER.md §6)."

[[ -f "$DATASET" ]] || die "dataset not found: $DATASET"

# A40 perf bundle must exist, else `python -m serving` fails upstream validation
# on `"hardware": "A40"` (profiles/accelerators/a40.yaml sim_hardware is null
# until profiled — HANDOVER.md §7 step 2).
if [[ ! -d "profiler/perf/$HARDWARE" ]]; then
  die "A40 perf bundle missing: profiler/perf/$HARDWARE/ does not exist.
     Profile it first (HANDOVER.md §7 step 2), then set sim_hardware: A40 in
     profiles/accelerators/a40.yaml:
       CUDA_VISIBLE_DEVICES=0 $VLLM_VENV/bin/python -m profiler profile \\
         $MODEL --hardware $HARDWARE --tp 1,2,4,8 \\
         --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS --max-num-seqs 256 \\
         --measurement-iterations 3"
fi

if grep -q 'sim_hardware:[[:space:]]*null' profiles/accelerators/a40.yaml 2>/dev/null; then
  die "profiles/accelerators/a40.yaml still has 'sim_hardware: null'. Write
     'sim_hardware: A40' after the perf bundle above is produced."
fi

# KV-matched config must have its placeholder replaced with the measured budget.
KV_NEEDED=1
if [[ "$SKIP_KVMATCHED" == "1" ]]; then
  KV_NEEDED=0
  note "SKIP_KVMATCHED=1 — running the NOMINAL side only (partial result)."
elif grep -q "$KV_SENTINEL" "$KVMATCHED_CONFIG"; then
  die "$KVMATCHED_CONFIG still contains the placeholder '$KV_SENTINEL'.
     Fill in the measured KV-matched mem_size first. Derivation
     (deviations.md D10, a40-llama31-8b-tp1.provenance.yaml):
       kv_matched_mem_size = 14.96 (Llama-3.1-8B bf16 weights, GiB)
                           + <Available KV cache memory GiB printed by vLLM
                              at A40 engine startup under THESE bench settings>
     Read that line from the real bench run's log (step 1 below), replace the
     sentinel with the numeric GB value, then flip source: to measured in the
     provenance file. Or re-run with SKIP_KVMATCHED=1 to do the nominal side now."
fi

note "preflight OK. HARDWARE=$HARDWARE MODEL=$MODEL DATASET=$DATASET"
note "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset — vLLM will use all visible GPUs>}"

# =============================================================================
# STEP 1 — real vLLM bench, prefix caching OFF.
# Prefix caching is disabled to match the simulator, which cannot complete this
# workload on a saturated card with it on (deviations.md D12); the A5000 protocol
# used the same wrapper (phase0_bench_plan.md §2b). Comparing prefix-off sim
# against prefix-on vLLM would compare two different systems.
# =============================================================================
mkdir -p "$VLLM_DIR"
note "STEP 1/4: real vLLM bench (prefix caching OFF, model=$BENCH_MODEL) -> $VLLM_DIR"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
"$VLLM_VENV/bin/python" experiments/scripts/bench_run_no_prefix_cache.py \
    --model "$BENCH_MODEL" \
    --dataset "$DATASET" \
    --output-dir "$VLLM_DIR" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --dtype "$DTYPE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --seed "$SEED" \
    --tick-seconds "$TICK_SECONDS" \
    --num-reqs "$NUM_REQS"

# -----------------------------------------------------------------------------
# Helper: run the simulator side for one config, then validate against the bench.
# -----------------------------------------------------------------------------
run_sim_and_validate() {
    local label="$1"      # human label, e.g. nominal / kvmatched
    local config="$2"     # cluster config JSON
    local out_dir="$3"    # output dir for sim.csv / sim.log / validation

    mkdir -p "$out_dir"
    note "  sim ($label): config=$config -> $out_dir/sim.csv"
    # serving spawns Chakra as a bare `python -m chakra...` subprocess
    # (serving/core/graph_generator.py), so the sim venv must be on PATH or that
    # nested call falls back to a system python without chakra. Calling the venv
    # python by full path alone is not enough.
    PATH="$REPO_ROOT/$SIM_VENV/bin:$PATH" \
    "$SIM_VENV/bin/python" -m serving \
        --cluster-config "$config" \
        --dataset "$DATASET" \
        --output "$out_dir/sim.csv" \
        --num-reqs "$NUM_REQS" \
        --dtype "$DTYPE" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --block-size "$BLOCK_SIZE" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --no-enable-prefix-caching \
        --log-level WARNING \
        --network-backend "$NETWORK_BACKEND" \
        > "$out_dir/sim.log" 2>&1
    # --no-enable-prefix-caching: serving defaults prefix caching ON
    # (serving/__main__.py:242). The vLLM side (STEP 1) is prefix-off, and D12
    # forbids prefix caching on a saturated sim run, so the sim side must match
    # or we would compare two different systems (and risk a D12 crash on the
    # memory-bound kvmatched config).

    note "  validate ($label): $VLLM_DIR vs $out_dir/sim.csv"
    # bench validate writes to <bench-dir>/<output-subdir>/. bench-dir is the
    # shared vLLM run, so the subdir must be per-label or nominal and kvmatched
    # overwrite each other.
    "$SIM_VENV/bin/python" -m bench validate \
        --bench-dir "$VLLM_DIR" \
        --sim-csv "$out_dir/sim.csv" \
        --sim-log "$out_dir/sim.log" \
        --output-subdir "validation-$label" \
        --title "vLLM vs LLMServingSim - $HARDWARE $label" \
        --log-level INFO
}

# =============================================================================
# STEP 2 — simulator @ nominal mem_size + validate.
# =============================================================================
note "STEP 2/4: simulator @ NOMINAL mem_size + validate"
run_sim_and_validate "nominal" "$NOMINAL_CONFIG" "$NOMINAL_DIR"

# =============================================================================
# STEP 3 — simulator @ KV-matched mem_size + validate.
# =============================================================================
if [[ "$KV_NEEDED" == "1" ]]; then
    note "STEP 3/4: simulator @ KV-matched mem_size + validate"
    run_sim_and_validate "kvmatched" "$KVMATCHED_CONFIG" "$KVMATCHED_DIR"
else
    note "STEP 3/4: skipped (SKIP_KVMATCHED=1)"
fi

# =============================================================================
# STEP 4 — nominal vs KV-matched side-by-side (mean |error| over the 15 metrics).
# =============================================================================
if [[ "$KV_NEEDED" == "1" ]]; then
    note "STEP 4/4: nominal vs KV-matched comparison"
    "$SIM_VENV/bin/python" experiments/scripts/compare_validations.py \
        "nominal=$VLLM_DIR/validation-nominal/summary.txt" \
        "kvmatched=$VLLM_DIR/validation-kvmatched/summary.txt"
else
    note "STEP 4/4: skipped (need the KV-matched run for the comparison)"
fi

note "done. Summaries:"
note "  nominal:   $VLLM_DIR/validation-nominal/summary.txt"
# Guard the conditional note so a false test does not become the script's exit
# status under `set -e` (the trailing `&& note` would otherwise return 1).
if [[ "$KV_NEEDED" == "1" ]]; then
    note "  kvmatched: $VLLM_DIR/validation-kvmatched/summary.txt"
fi
