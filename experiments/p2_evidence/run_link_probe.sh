#!/usr/bin/env bash
# Every link case V4 left open, in one pass. No deployment, no model — just the
# interconnect, so it is cheap and repeatable.
set -u
OUT=outputs/p2_evidence/link
mkdir -p "$OUT"
PY=.venv-vllm/bin/torchrun

run() {  # label, nproc, ranks, extra env
  local label="$1" nproc="$2" ranks="$3"; shift 3
  echo "=== $(date -Is) $label (ranks $ranks) $* ==="
  env "$@" MASTER_ADDR=127.0.0.1 MASTER_PORT=29571 \
    $PY --nproc-per-node="$nproc" --master-port=29571 \
    experiments/p2_evidence/link_probe.py \
    --ranks "$ranks" --label "$label" --out "$OUT/${label}.json" 2>&1 \
    | grep -vE '^\[|WARNING|warnings.warn|^\s*$'
}

# The TP=4 group the P1 candidate actually runs on.
run tp4                4 0,1,2,3
run tp4_nop2p          4 0,1,2,3 NCCL_P2P_DISABLE=1
# Inside one NVLink pair, and across the PCIe bridge between pairs. This is the
# pair of numbers the cluster spec guesses: 112.5 nvlink vs 64.0 vendor_spec pcie.
run pair_nvlink        2 0,1
run pair_pcie_02       2 0,1     CUDA_VISIBLE_DEVICES=0,2
run pair_pcie_02_nop2p 2 0,1     CUDA_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1
echo "=== $(date -Is) link probe complete ==="
