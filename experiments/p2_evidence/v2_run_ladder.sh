#!/usr/bin/env bash
# V2's open-loop rate ladder against the deployed A40 server.
#
# WORK_ORDER_p2_regular_spec_evidence.md STEP V2. The rates are NOT the work
# order's 2/5/10/15: the capacity probe measured this deployment at 682.8 output
# tok/s, and the workload's mean output is 652.5 tokens, so saturation is at
# ~1.05 rps and every rate the work order named is past it. The ladder keeps the
# work order's INTENT -- below saturation, climbing -- at the measured capacity.
#
# The 10 rps stage is deliberately far past saturation: it is the L > 170 point
# added on 2026-09-16, which is where the A40 domain's committed 170.56 anchor
# sits and where V3's ten held candidates (L 172.4-197.9) live.
set -u
BASE=http://127.0.0.1:8100/v1
MODEL=NousResearch/Meta-Llama-3.1-8B
DATA=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
OUT=outputs/p2_evidence/v2

run() {  # rate, repeat index
  local rps="$1" rep="$2"
  local tag="rps${rps}_r${rep}"
  echo "=== $(date -Is) stage $tag ==="
  timeout 2400 .venv-vllm/bin/python experiments/scripts/replay_to_endpoint.py \
    --base-url "$BASE" --model "$MODEL" --dataset "$DATA" \
    --open-loop --ignore-eos --max-tokens-cap 1024 \
    --target-rps "$rps" --out "$OUT/${tag}.json" 2>&1 | grep -vE '^\s+File "|^Traceback|^\s+[a-z_]+\(|GeneratorExit|RuntimeError|^\s+\^|^During handling|^\s*$'
  echo "--- settle ---"
  sleep 30
}

for rps in 0.5 0.75 0.9 1.0 10.0; do run "$rps" 0; done
# Repeats for the run-to-run p99 spread V3's P3 result depends on (its margin
# headroom is 0.623 ms). 0.9 rps is the stage nearest P3's operating point
# (L = 28.1); 10.0 is the L > 170 stage.
for rep in 1 2; do run 0.9 "$rep"; done
run 10.0 1
echo "=== $(date -Is) ladder complete ==="
