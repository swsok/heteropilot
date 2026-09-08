#!/usr/bin/env bash
# Sample RNGD power AND per-PE utilisation at a drift-free 1 Hz.
#
# `WORK_ORDER_rps_aware.md` rev 2 STEP 3.1. A5(c): power recorded without
# utilisation is not a measurement, because the same wattage means different
# things on a busy card and an idle one. The two numbers live in different
# subcommands -- `info` has power, `status` has `pe_utilizations` and DRAM
# `used_ratio`, neither has both -- so every tick calls both and records a
# timestamp for each, and the gap between them is data the consumer can check
# rather than an error the sampler hides.
#
# Three node facts this encodes (docs/rps_step0_retro.md):
#
#   - Power is quantised to 1 W. At an idle near 38 W one quantum is 2.6 %, so a
#     5 % repeat threshold is about two quanta. Bench windows must be long
#     enough (>= 60 s) for the mean to mean anything.
#   - Utilisation is PER PE, not per card. A card running TP=8 has eight numbers
#     per sample; a partly-idle card and a uniformly-loaded one at the same mean
#     are different operating points, so mean, min and max are all recorded.
#   - `dev_name` (npu0/1/2) is NOT stable -- it re-enumerated between 2026-09-04
#     and 2026-09-07 while the card count stayed at 3. `device_sn` and `pci_bdf`
#     survive. All three are recorded; select on `device_sn`.
#
# Serial numbers identify physical hardware, so `redact_power_csv.py` rewrites
# them to stable pseudonyms before an artifact is committed.
#
# Usage:
#   experiments/scripts/power_sampler.sh --out power_c8.csv [--devices npu0,npu1]
#   ... then SIGTERM/SIGINT it; it flushes and exits 0.

set -uo pipefail

OUT=""
DEVICES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "error: --out is required" >&2; exit 2; }
command -v furiosa-smi >/dev/null || { echo "error: furiosa-smi not on PATH" >&2; exit 2; }
command -v jq >/dev/null || { echo "error: jq not on PATH" >&2; exit 2; }

mkdir -p "$(dirname "$OUT")"

# ---- header: what produced this file, and what the fields were at the time ----
{
  echo "# power_sampler.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# host_cores=$(nproc)"
  furiosa-smi version 2>&1 | sed 's/^/# /'
  echo "# info_fields=$(furiosa-smi info --format json 2>/dev/null | jq -r '.[0] | keys | join(",")')"
  echo "# status_fields=$(furiosa-smi status --format json 2>/dev/null | jq -r '.[0] | keys | join(",")')"
  echo "# devices_filter=${DEVICES:-<all>}"
  echo "# NOTE power is quantised to 1 W; util is per-PE (mean/min/max over the card's PEs)"
} > "$OUT"
echo "ts_info,ts_status,device_sn,pci_bdf,dev_name,power_w,util_mean_pct,util_min_pct,util_max_pct,pe_busy,pe_total,dram_used_ratio,temperature_c,raw_info,raw_status" >> "$OUT"

RUNNING=1
trap 'RUNNING=0' TERM INT

# jq filter shared by the two joins below. Kept out of the loop so the loop body
# stays short enough to finish well inside one second.
JQ_JOIN='
  ($status | map({key: .device, value: .}) | from_entries) as $st
  | $info[]
  | . as $i
  | ($st[$i.dev_name] // {}) as $s
  | ($s.pe_utilizations // []) as $pe
  | ($pe | map(.pe_utilization)) as $u
  | [ $ts_info, $ts_status, $i.device_sn, $i.pci_bdf, $i.dev_name,
      ($i.power // "" | tostring | sub(" *W$"; "")),
      (if ($u | length) > 0 then ($u | add / length) else null end),
      (if ($u | length) > 0 then ($u | min) else null end),
      (if ($u | length) > 0 then ($u | max) else null end),
      ($pe | map(select(.pe_occupancy)) | length),
      ($pe | length),
      ($s.memory.DRAM.used_ratio // null),
      ($i.temperature // "" | tostring | sub("°C$"; "")),
      ($i | tojson), ($s | tojson) ]
  | @csv'

while [ "$RUNNING" = "1" ]; do
  # Wait to the next whole second rather than sleeping a fixed interval: a
  # `sleep 1` accumulates the cost of each tick's own work as drift, and these
  # samples are correlated against bench timestamps.
  now=$(date +%s.%N)
  frac=${now#*.}
  # 1e9 - fractional nanoseconds, guarded so a tick that overran does not sleep ~1 s
  ns_to_go=$(( 1000000000 - 10#${frac:0:9} ))
  [ "$ns_to_go" -gt 0 ] && sleep "0.$(printf '%09d' "$ns_to_go")"
  [ "$RUNNING" = "1" ] || break

  ts_info=$(date +%s.%N)
  info=$(furiosa-smi info --format json 2>/dev/null)
  ts_status=$(date +%s.%N)
  status=$(furiosa-smi status --format json 2>/dev/null)

  if [ -z "$info" ] || [ -z "$status" ]; then
    # A signal arriving mid-tick kills the in-flight furiosa-smi, which is a
    # shutdown, not a dropped sample -- recording it would put a spurious hole at
    # the end of every run.
    [ "$RUNNING" = "1" ] || break
    # A real dropped sample IS recorded, never silently skipped: a gap in the
    # series would otherwise look like a gap in time.
    echo "$ts_info,$ts_status,,,,,,,,,,,,\"query failed\",\"\"" >> "$OUT"
    continue
  fi

  jq -r --argjson info "$info" --argjson status "$status" \
        --arg ts_info "$ts_info" --arg ts_status "$ts_status" \
        -n "$JQ_JOIN" 2>/dev/null \
    | { if [ -n "$DEVICES" ]; then grep -E "\"($(echo "$DEVICES" | tr ',' '|'))\"" || true; else cat; fi; } \
    >> "$OUT"
done

echo "# stopped $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
exit 0
