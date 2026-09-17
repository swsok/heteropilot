#!/usr/bin/env bash
# V3: deploy P1, drive it at its predicted operating point, three times.
#
# WORK_ORDER_p2_regular_spec_evidence.md STEP V3. Load is the trace at its OWN
# arrival rate -- the service spec's `arrival_rate_rps: 10`, which is what the
# simulator was given and therefore what produced the predicted L of 127.28.
# Open loop, per D19: P1's TTFT is checked against a 25 s SLO and a closed-loop
# burst would inflate it.
#
# Three repeats for the spread, as V2 did. `--ignore-eos` is not optional: the
# simulator generates the trace's output_toks exactly and without it the engine
# stops at EOS, so the two sides run different workloads.
set -u
PLAN=${1:-outputs/plans/p2ev-v3-p1.yaml}
ID=${2:-p2ev-v3-p1}
PORT=${3:-8200}
OUT=outputs/p2_evidence/v3
MODEL=NousResearch/Meta-Llama-3.1-8B
DATA=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
mkdir -p "$OUT"

echo "=== $(date -Is) deploying $ID ==="
PATH="$PWD/.venv-vllm/bin:$PATH" .venv/bin/python -m planner deploy \
  --plan "$PLAN" --cluster "${4:-experiments/configs/clusters/pd-rngd-gpu-card.yaml}" \
  --port "$PORT" --no-dry-run 2>&1 | tail -6

for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo "healthy after ~$((i*10))s"; break; }
  sleep 10
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health")" != "200" ]; then
  echo "!! server never became healthy; aborting"; .venv/bin/python -m planner stop --deployment "$ID"; exit 1
fi

for rep in 0 1 2; do
  echo "=== $(date -Is) ${ID} repeat $rep ==="
  timeout 3600 .venv-vllm/bin/python experiments/scripts/replay_to_endpoint.py \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --dataset "$DATA" \
    --open-loop --ignore-eos --max-tokens-cap 1024 \
    --out "$OUT/${ID}_r${rep}.json" 2>&1 \
    | grep -vE '^\s+File "|^Traceback|^\s+[a-z_]+\(|GeneratorExit|RuntimeError|^\s+\^|^During handling|^\s*$'
  sleep 30
done

echo "=== $(date -Is) stopping $ID ==="
.venv/bin/python -m planner stop --deployment "$ID" 2>&1 | tail -2
echo "=== $(date -Is) done ==="
