#!/usr/bin/env bash
# The same five link cases, on GPUs 4-7 instead of 0-3.
#
# The two groups are topologically symmetric -- 4<->5 and 6<->7 are NV4, the
# pairs are bridged by NODE, and 0-3 to 4-7 is SYS -- so the numbers should
# transfer. They differ in NUMA affinity (0-3 on node 0, 4-7 on node 1) and only
# GPU6 has a PHB path to the NIC, so running both is also a check on that.
set -u
OUT=outputs/p2_evidence/link47
mkdir -p "$OUT"
PY=.venv-vllm/bin/torchrun

run() {  # label, nproc, ranks, visible devices, extra env
  local label="$1" nproc="$2" ranks="$3" vis="$4"; shift 4
  echo "=== $(date -Is) $label (ranks $ranks, physical $vis) $* ==="
  env CUDA_VISIBLE_DEVICES="$vis" "$@" MASTER_ADDR=127.0.0.1 MASTER_PORT=29581 \
    $PY --nproc-per-node="$nproc" --master-port=29581 \
    experiments/p2_evidence/link_probe.py \
    --ranks "$ranks" --label "$label" --out "$OUT/${label}.json" 2>&1 \
    | grep -vE '^\[|WARNING|warnings.warn|^\s*$'
}

run tp4                4 0,1,2,3 4,5,6,7
run tp4_nop2p          4 0,1,2,3 4,5,6,7 NCCL_P2P_DISABLE=1
run pair_nvlink        2 0,1     4,5
run pair_pcie_46       2 0,1     4,6
run pair_pcie_46_nop2p 2 0,1     4,6     NCCL_P2P_DISABLE=1
echo "=== $(date -Is) link probe (4-7) complete ==="
