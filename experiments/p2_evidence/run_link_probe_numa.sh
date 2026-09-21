#!/usr/bin/env bash
# The same link cases as run_link_probe.sh / run_link_probe47.sh, NUMA-BOUND.
#
# WORK_ORDER_domain_scoping.md S6(ii) asks for the link figures "고정/미고정 각각"
# -- bound and unbound each. Everything filed in `Link.measurements[]` today is
# `binding: unpinned`, because neither existing driver wraps anything in
# `numactl` (pd-rngd-gpu-card-measured-pcie.yaml says so in as many words). This
# is the other half of that key.
#
# THE PROBE IS NOT TOUCHED. `binding` is one axis of the measurement key, so the
# bound and unbound runs must differ in binding and in nothing else; editing
# link_probe.py would make the pair incomparable and would also re-open A3 on
# figures already filed. The only difference here is the `numactl` prefix and
# the output directory.
#
# Binding is verified rather than trusted: `numactl --show` and `taskset -cp` of
# the launcher are recorded beside each group's results, because a flag that is
# accepted is not a flag that took effect (docs/HANDOVER.md §2.10.1 item 2
# records that trap on the RNGD side; it is the same trap here).
#
# Group -> NUMA node comes from docs/nodes/a40.md: GPUs 0-3 are NUMA 0
# (CPUs 0-15,32-47), GPUs 4-7 are NUMA 1 (CPUs 16-31,48-63).
set -u
OUT=outputs/p2_evidence/link_numa
mkdir -p "$OUT"
PY=.venv-vllm/bin/torchrun

# One port per group so the two groups can never collide on a rendezvous.
run() {  # node, label, nproc, ranks, visible devices, extra env
  local node="$1" label="$2" nproc="$3" ranks="$4" vis="$5"; shift 5
  local dir="$OUT/numa${node}"
  local port=$((29591 + node * 10))
  mkdir -p "$dir"
  echo "=== $(date -Is) numa${node} $label (ranks $ranks, physical $vis) $* ==="
  numactl --cpunodebind="$node" --membind="$node" \
    env CUDA_VISIBLE_DEVICES="$vis" "$@" MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
    $PY --nproc-per-node="$nproc" --master-port=$port \
    experiments/p2_evidence/link_probe.py \
    --ranks "$ranks" --label "$label" --out "$dir/${label}.json" 2>&1 \
    | grep -vE '^\[|WARNING|warnings.warn|^\s*$'
}

# Record that the binding took effect, for each node, before measuring under it.
prove_binding() {  # node
  local node="$1" dir="$OUT/numa${node}"
  mkdir -p "$dir"
  {
    echo "=== numactl --show under --cpunodebind=$node --membind=$node ==="
    numactl --cpunodebind="$node" --membind="$node" numactl --show
    echo
    echo "=== taskset -cp of a process started that way ==="
    numactl --cpunodebind="$node" --membind="$node" bash -c 'taskset -cp $$'
  } | tee "$dir/binding_proof.txt"
  echo
}

for node in 0 1; do
  prove_binding "$node"
done

# GPUs 0-3 on NUMA 0. Physical devices equal logical ones here, so the case
# names match run_link_probe.sh exactly.
run 0 tp4                4 0,1,2,3 0,1,2,3
run 0 tp4_nop2p          4 0,1,2,3 0,1,2,3 NCCL_P2P_DISABLE=1
run 0 pair_nvlink        2 0,1     0,1
run 0 pair_pcie_02       2 0,1     0,2
run 0 pair_pcie_02_nop2p 2 0,1     0,2     NCCL_P2P_DISABLE=1

# GPUs 4-7 on NUMA 1 -- the group V3's P1 was deployed on, and the one where
# binding was worth 1.93x of deployment throughput (v3_verdict_accuracy.md A.3).
run 1 tp4                4 0,1,2,3 4,5,6,7
run 1 tp4_nop2p          4 0,1,2,3 4,5,6,7 NCCL_P2P_DISABLE=1
run 1 pair_nvlink        2 0,1     4,5
run 1 pair_pcie_46       2 0,1     4,6
run 1 pair_pcie_46_nop2p 2 0,1     4,6     NCCL_P2P_DISABLE=1

echo "=== $(date -Is) NUMA-bound link probe complete ==="
