#!/usr/bin/env bash
# WORK_ORDER_d23_fix_revalidation.md §2 — the two regression anchors.
#
# Every sanctioned serving/ edit in this work order must leave both byte-identical.
# Run once before the edits (STEP 0.3) and once after each STEP.
#
#   d23fix_anchors.sh <label>        # e.g. before, after_step1, after_step2
#
# R1  upstream anchor: the three bench/examples/ runs that docs/phase0_formats.md
#     §2.1 guarantees reproduce exactly at the pin. run.sh hardcodes its output
#     into bench/examples/<model>/outputs/, which CLAUDE.md forbids overwriting,
#     so the command is reconstructed here with --output redirected. Everything
#     else matches run.sh:104-120 and each vllm/meta.json.
#
# R2  P/D anchor: the completed candidate docs/d23_spike.md timed at 343 s solo.
#     Flags are the predictor's (planner/predictor/llmservingsim.py:393-415), read
#     back from the recorded sim1.log banner.
#
# Deliberately sequential: these are the reference, so they must not race, and
# every run goes through livelock_watch.sh with a retry. That is not caution for
# its own sake: on 2026-09-04 one R2 run hung with D23's exact signature and, with
# no watcher, sat for two hours before it was noticed. It has not recurred in the
# 14 solo runs since -- all byte-identical to the committed sim1.csv -- so the
# event is rare, not a property of the input. A reference run must not be able to
# stall the anchor silently.
set -uo pipefail
cd "$(dirname "$(realpath "$0")")/../.." || exit 1
LABEL="${1:?usage: d23fix_anchors.sh <label>}"
OUT="outputs/d23fix/anchor/${LABEL}"
mkdir -p "$OUT"
FAILED=0
PY=.venv/bin/python
export PYTHONPATH="$PWD"
# Deliberately NOT putting .venv/bin on PATH. Before D26 that omission decided
# whether a P/D run completed, because the Chakra converter was invoked as bare
# `python`; these anchors hung 4/4 because of it. Since D26 the frontend uses
# sys.executable, so a bare PATH must now be harmless -- and this script is where
# that gets checked on every run.

log() { printf '%s\n' "$*" | tee -a "$OUT/anchors.log"; }

# run_guarded <name> <expected_rows> -- <command...>
# livelock_watch exits 3 on a tick stall and 4 when a run never reports or goes
# quiet; either means retry rather than wait. Two attempts, then give up loudly.
run_guarded() {
  local name="$1" want="$2"; shift 2; [ "$1" = "--" ] && shift
  local attempt rc s rows
  for attempt in 1 2; do
    s=$(date +%s)
    experiments/scripts/livelock_watch.sh -n 45 -g 600 -s 600 -t 1500 -- "$@" \
      > "$OUT/$name.log" 2>&1
    rc=$?
    rows=$(wc -l < "$OUT/$name.csv" 2>/dev/null || echo 0)
    if [ "$rc" = "0" ] && [ "$rows" = "$want" ]; then
      log "$name attempt=$attempt exit=0 elapsed=$(( $(date +%s) - s ))s rows=$rows"
      return 0
    fi
    log "$name attempt=$attempt exit=$rc elapsed=$(( $(date +%s) - s ))s rows=$rows -- retrying"
    cp "$OUT/$name.log" "$OUT/$name.attempt$attempt.log" 2>/dev/null
  done
  log "$name FAILED after 2 attempts -- the anchor is unusable, do not merge on it"
  return 1
}
log "=== anchors '$LABEL' — $(date -u +%Y-%m-%dT%H:%M:%SZ) — $(git rev-parse --short HEAD) ==="

# ---- R1 -------------------------------------------------------------------
for m in Llama-3.1-8B Qwen3-32B Qwen3-30B-A3B-Instruct-2507; do
  meta="bench/examples/$m/vllm/meta.json"
  read -r ds n dt kv seqs toks < <("$PY" - "$meta" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); ek = d["engine_kwargs"]
print(d["dataset_path"], d["num_requests"], ek["dtype"], ek["kv_cache_dtype"],
      ek["max_num_seqs"], ek["max_num_batched_tokens"])
PY
)
  run_guarded "r1_$m" "$(( n + 1 ))" -- \
    "$PY" -m serving \
    --cluster-config "bench/examples/configs/$m.json" \
    --dataset "$ds" --output "$OUT/r1_$m.csv" \
    --num-reqs "$n" --dtype "$dt" --kv-cache-dtype "$kv" \
    --block-size 16 --max-num-seqs "$seqs" --max-num-batched-tokens "$toks" \
    --log-level WARNING --network-backend analytical \
    --run-id "d23fix-r1-$LABEL-$m" || FAILED=1
done

# ---- R2 -------------------------------------------------------------------
R2DIR="outputs/.hp-pd-slo/work/pd_cuda-a40-node_a40a-tp4-dp1_P___cuda-a40-node_a40b-tp4-dp1_D_-s256-t8192"
run_guarded "r2_pd_a40tp4" 301 -- \
  "$PY" -m serving \
  --cluster-config "$R2DIR/cluster.json" \
  --dataset outputs/.hp-pd-slo/work/sweep_trace.jsonl \
  --output "$OUT/r2_pd_a40tp4.csv" \
  --num-reqs 300 --dtype bfloat16 --kv-cache-dtype auto \
  --block-size 16 --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --request-routing-policy LOAD --network-backend analytical \
  --log-level WARNING --log-interval 1.0 --no-enable-prefix-caching \
  --run-id "d23fix-r2-$LABEL" || FAILED=1

# ---- hashes ---------------------------------------------------------------
( cd "$OUT" && sha256sum r1_*.csv r2_*.csv > SHA256SUMS 2>/dev/null )
log "--- SHA256SUMS ---"; cat "$OUT/SHA256SUMS" | tee -a "$OUT/anchors.log"
log "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) — FAILED=$FAILED ==="
exit "$FAILED"
