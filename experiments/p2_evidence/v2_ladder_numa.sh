#!/usr/bin/env bash
# V2's rate ladder again, with the server bound to the NUMA node its GPU is on.
#
# Appendix A.3 found that leaving the binding to chance is worth 1.93x of
# throughput on this host, and that every A40 measurement before it -- V2's eight
# ladder stages included -- ran unbound. GPUs 0-3 happened to land well, which is
# an argument, not a check. This is the check: the same five rates, one repeat
# each, with `numactl --cpunodebind=0 --membind=0`, against the committed
# unbound numbers in outputs/p2_evidence/v2/.
#
# GPU 0 is on NUMA node 0 (CPU 0-15, 32-47), so node 0 is the correct binding.
# A fresh plan id, because reusing one overwrites the committed deployment record.
set -u
BASE=http://127.0.0.1:8500/v1
MODEL=NousResearch/Meta-Llama-3.1-8B
DATA=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
OUT=outputs/p2_evidence/v2_numa
ID=p2ev-v2-numa
mkdir -p "$OUT"

echo "=== $(date -Is) deploying $ID (NUMA 0 bound) ==="
PATH="$PWD/.venv-vllm/bin:$PATH" .venv/bin/python -m planner deploy \
  --plan outputs/plans/p2ev-v2-numa.yaml --cluster experiments/configs/clusters/a40x8.yaml \
  --port 8500 --no-dry-run 2>&1 | tail -4
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8500/health)" = "200" ] \
    && { echo "healthy after ~$((i*10))s"; break; }
  sleep 10
done

for rps in 0.5 0.75 0.9 1.0 10.0; do
  echo "=== $(date -Is) stage rps${rps} (NUMA 0) ==="
  timeout 2400 .venv-vllm/bin/python experiments/scripts/replay_to_endpoint.py \
    --base-url "$BASE" --model "$MODEL" --dataset "$DATA" \
    --open-loop --ignore-eos --max-tokens-cap 1024 --target-rps "$rps" \
    --out "$OUT/rps${rps}_numa.json" 2>&1 \
    | grep -vE '^\s+File "|^Traceback|^\s+[a-z_]+\(|GeneratorExit|RuntimeError|^\s+\^|^During handling|^\s*$'
  sleep 30
done

echo "=== $(date -Is) stopping ==="
.venv/bin/python -m planner stop --deployment "$ID" 2>&1 | tail -1
echo "=== $(date -Is) V2 NUMA ladder complete ==="
