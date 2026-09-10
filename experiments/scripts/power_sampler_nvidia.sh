#!/usr/bin/env bash
# Sample NVIDIA GPU power AND utilisation at a drift-free 1 Hz.
#
# The CUDA twin of `power_sampler.sh`. `docs/HANDOVER.md` §2.2: the analysis half
# of `measure_envelope.py` is vendor-agnostic and `read_sampler_csv` cares about a
# CSV schema rather than a vendor, so the only thing an A40 envelope point needed
# was a sampler emitting that schema. This is it: SAME header, SAME column order,
# SAME dropped-sample convention, so `read_sampler_csv` parses both without a
# branch and A5(c) is enforced identically on either node.
#
# A5(c): power recorded without utilisation is not a measurement, because the same
# wattage means different things on a busy card and an idle one. On RNGD that cost
# two subcommands per tick (`info` has power, `status` has utilisation); NVML
# returns both from one query, so `ts_info` and `ts_status` are the same timestamp
# here by construction rather than by luck. The columns are kept anyway so the two
# files are one schema, and a consumer that checks the gap finds 0.
#
# Three differences from the RNGD sampler, each a fact about the hardware and none
# of them a fudge:
#
#   - A GPU is ONE execution unit, not eight PEs. `utilization.gpu` is a single
#     number, so `util_mean_pct`, `util_min_pct` and `util_max_pct` are all it and
#     `pe_total` is 1. `pe_busy` is left EMPTY: RNGD's `pe_occupancy` means a PE
#     has a program resident, and NVML exposes no analogue. Inventing one from
#     `utilization > 0` would be a different quantity wearing the same column name.
#   - `utilization.gpu` is the fraction of the last sampling period in which ANY
#     kernel was resident -- occupancy in time, not the fraction of SMs busy. A
#     decode step at batch 1 reads 100 % while using a sliver of the machine. It
#     bounds the "was the card idle?" question A5(c) asks and nothing finer; the
#     RNGD per-PE figure is not the same measurement and the two do not compare.
#   - `device_sn` carries the GPU **UUID**, not a serial. `nvidia-smi` reports
#     `serial` as [N/A] on these boards, and the index re-enumerates under
#     `CUDA_VISIBLE_DEVICES` exactly as RNGD's `dev_name` did between 2026-09-04
#     and 2026-09-07. The UUID survives both. Select on it.
#
# Power quantisation differs too, and in the helpful direction: NVML reports
# milliwatts (~0.01 W here) against RNGD's 1 W, so the >= 60 s bench window that
# the RNGD note demanded for the mean to mean anything is not the binding
# constraint on this node. It is kept regardless -- the windows are what make the
# two envelopes comparable.
#
# Usage:
#   experiments/scripts/power_sampler_nvidia.sh --out power_c8.csv [--devices 0]
#   ... then SIGTERM/SIGINT it; it flushes and exits 0.
#
# --devices takes what `nvidia-smi -i` takes (indices or UUIDs) and is passed
# straight through, so the sampler observes exactly the cards named. Pin it to the
# same card the server got via CUDA_VISIBLE_DEVICES: an unfiltered sampler on an
# 8-GPU node records seven idle cards beside the one under test, and their 30 W
# each would drag any mean computed over the file.

set -uo pipefail

OUT=""
DEVICES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "error: --out is required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi not on PATH" >&2; exit 2; }

# No `jq` on this node, and none needed: nvidia-smi emits CSV natively. The RNGD
# sampler needs it only because furiosa-smi speaks JSON.

QUERY="index,uuid,name,pci.bus_id,power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu"
SMI_ARGS=(--query-gpu="$QUERY" --format=csv,noheader,nounits)
[ -n "$DEVICES" ] && SMI_ARGS+=(-i "$DEVICES")

mkdir -p "$(dirname "$OUT")"

# ---- header: what produced this file, and what the fields were at the time ----
{
  echo "# power_sampler_nvidia.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# host_cores=$(nproc)"
  nvidia-smi --version 2>&1 | sed 's/^/# /'
  echo "# query_fields=$QUERY"
  echo "# devices_filter=${DEVICES:-<all>}"
  echo "# NOTE power is NVML milliwatt-resolution; util is utilization.gpu, a single"
  echo "# NOTE per-GPU number (time-occupancy, NOT SM occupancy) -- mean/min/max are all it,"
  echo "# NOTE pe_total=1 and pe_busy is empty because NVML has no pe_occupancy analogue."
  echo "# NOTE device_sn carries the GPU UUID; nvidia-smi reports serial as [N/A] here."
} > "$OUT"
echo "ts_info,ts_status,device_sn,pci_bdf,dev_name,power_w,util_mean_pct,util_min_pct,util_max_pct,pe_busy,pe_total,dram_used_ratio,temperature_c,raw_info,raw_status" >> "$OUT"

RUNNING=1
trap 'RUNNING=0' TERM INT

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

  ts=$(date +%s.%N)
  # One query returns power and utilisation together, so they are the same sample
  # rather than two reads a consumer has to reconcile. That IS A5(c), not a
  # convenience.
  raw=$(nvidia-smi "${SMI_ARGS[@]}" 2>/dev/null)

  if [ -z "$raw" ]; then
    # A signal arriving mid-tick kills the in-flight nvidia-smi, which is a
    # shutdown, not a dropped sample -- recording it would put a spurious hole at
    # the end of every run.
    [ "$RUNNING" = "1" ] || break
    # A real dropped sample IS recorded, never silently skipped: a gap in the
    # series would otherwise look like a gap in time.
    echo "$ts,$ts,,,,,,,,,,,,\"query failed\",\"\"" >> "$OUT"
    continue
  fi

  echo "$raw" | awk -F', *' -v ts="$ts" 'NF >= 9 {
    used = $7 + 0; total = $8 + 0;
    dram = (total > 0) ? used / total : "";
    # raw_info keeps the unparsed row so a reader can check this parse; raw_status
    # is empty because there is no second subcommand whose skew needs recording.
    raw = $0; gsub(/"/, "\"\"", raw);
    # pe_busy is deliberately empty -- see the header note.
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,,1,%s,%s,\"%s\",\"\"\n",
           ts, ts, $2, $4, $1, $5, $6, $6, $6, dram, $9, raw;
  }' >> "$OUT"
done

echo "# stopped $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
exit 0
