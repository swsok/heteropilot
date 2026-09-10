#!/usr/bin/env bash
# R3 -- the slab3d equivalence anchor (WORK_ORDER_rps_aware.md rev 2 STEP 2.1).
#
# Two colocated tp4 instances are TWO HALF SLABS under slab3d, which yields
# dims [4, 2] and tp_dim [T, F] per instance -- exactly what `auto` produces for
# two independent tp4 instances. So the two configs must simulate to the same
# bytes. That is a stronger statement than "auto is unchanged": it says the new
# path computes the same topology where the two overlap, rather than merely
# staying out of the way.
#
#   slab3d_anchors.sh <label>
set -uo pipefail
cd "$(dirname "$(realpath "$0")")/../.." || exit 1
LABEL="${1:?usage: slab3d_anchors.sh <label>}"
OUT="outputs/d14/anchor/${LABEL}"
mkdir -p "$OUT"
PY=.venv/bin/python
export PYTHONPATH="$PWD"
FAILED=0
log() { printf '%s\n' "$*" | tee -a "$OUT/anchors.log"; }

run() {
  local name="$1" cfg="$2" s rc rows
  s=$(date +%s)
  experiments/scripts/livelock_watch.sh -n 45 -g 600 -s 600 -t 1500 -- \
    "$PY" -m serving --cluster-config "$cfg" \
      --dataset outputs/d23/trace20.jsonl --output "$OUT/$name.csv" \
      --num-reqs 20 --dtype bfloat16 --kv-cache-dtype auto --block-size 16 \
      --max-num-seqs 256 --max-num-batched-tokens 8192 \
      --request-routing-policy LOAD --network-backend analytical \
      --log-level WARNING --log-interval 1.0 --no-enable-prefix-caching \
      --run-id "r3-$LABEL-$name" > "$OUT/$name.log" 2>&1
  rc=$?
  rows=$(wc -l < "$OUT/$name.csv" 2>/dev/null || echo 0)
  log "$name exit=$rc elapsed=$(( $(date +%s) - s ))s rows=$rows"
  [ "$rc" = "0" ] || FAILED=1
}

run auto   experiments/configs/clusters/colocated-tp4x2-auto.json
run slab3d experiments/configs/clusters/colocated-tp4x2-slab3d.json

if [ -s "$OUT/auto.csv" ] && cmp -s "$OUT/auto.csv" "$OUT/slab3d.csv"; then
  log "R3 PASS: auto and slab3d are byte-identical"
else
  log "R3 FAIL: auto and slab3d differ (or a run produced nothing)"
  FAILED=1
fi
( cd "$OUT" && sha256sum ./*.csv > SHA256SUMS 2>/dev/null )
exit "$FAILED"
