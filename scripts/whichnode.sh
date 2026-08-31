#!/bin/bash
# Which node is this? Detect it; never assume it.
#
# This repository moves between an A40 node, an A5000 node and an NPU node. Any
# committed statement about what hardware is present is true on at most one of
# them, and CLAUDE.md used to carry such a statement -- which is how a session
# once opened on a box with eight A40s while reading "no NVIDIA GPU at all".
#
# HOSTNAME DOES NOT DISCRIMINATE. Every node reports `s8` and the same kernel
# (5.4.0-216-generic); the only incidental difference in committed provenance is
# the CPU count. So this script probes the accelerators themselves.
#
# It answers the question a session actually has -- not "what chips are here" but
# "what can I run here" -- and prints nothing that it has not just observed.
#
# Usage:  bash scripts/whichnode.sh          (no venv, no root, no arguments)

set -uo pipefail   # deliberately NOT -e: every probe is allowed to fail

have() { command -v "$1" >/dev/null 2>&1; }

# --- probe: NVIDIA -----------------------------------------------------------
GPU_MODEL=""; GPU_COUNT=0
if have nvidia-smi; then
    if GPU_LIST=$(timeout 30 nvidia-smi -L 2>/dev/null) && [ -n "$GPU_LIST" ]; then
        GPU_COUNT=$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU ')
        GPU_MODEL=$(printf '%s\n' "$GPU_LIST" | head -1 | sed 's/.*: \(.*\) (UUID.*/\1/')
    fi
fi

# --- probe: FuriosaAI RNGD ---------------------------------------------------
# The sysfs management nodes are the reliable signal: furiosa-smi under-reports
# when another tenant's pods hold the cards, and /dev entries have come and gone.
RNGD_CARDS=0
if [ -d /sys/class/rngd_mgmt ]; then
    RNGD_CARDS=$(find /sys/class/rngd_mgmt -maxdepth 1 -name 'rngd!npu*mgmt' 2>/dev/null | wc -l)
fi
[ "$RNGD_CARDS" -eq 0 ] && [ -d /dev/rngd ] && RNGD_CARDS=$(ls /dev/rngd 2>/dev/null | grep -c 'npu[0-9]*$')

# --- probe: Rebellions ATOM --------------------------------------------------
ATOM_CARDS=$(ls /dev/rbln* 2>/dev/null | wc -l)

# --- classify ----------------------------------------------------------------
NODE="unknown"; NODE_DOC=""
if [ "$RNGD_CARDS" -gt 0 ] || [ "$ATOM_CARDS" -gt 0 ]; then
    NODE="npu"; NODE_DOC="docs/nodes/npu.md"
elif [ "$GPU_COUNT" -gt 0 ]; then
    case "$GPU_MODEL" in
        *A40*)   NODE="a40";   NODE_DOC="docs/nodes/a40.md" ;;
        *A5000*) NODE="a5000"; NODE_DOC="docs/nodes/a5000.md" ;;
        *)       NODE="cuda-other" ;;
    esac
fi

echo "=== node detection ==============================================="
echo "  detected node : $NODE${NODE_DOC:+   (read $NODE_DOC)}"
echo "  hostname      : $(hostname)   <- same on every node, do not key off it"
echo "  cores / RAM   : $(nproc) / $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GiB"}')"
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "  NVIDIA        : ${GPU_COUNT} x ${GPU_MODEL}"
else
    echo "  NVIDIA        : none reachable"
fi
echo "  RNGD cards    : $RNGD_CARDS"
echo "  ATOM devices  : $ATOM_CARDS"

echo
echo "=== what you can run here ========================================"
echo "  planner + analytical sim : YES on every node (needs .venv, no device)"

if [ "$GPU_COUNT" -gt 0 ]; then
    echo "  real vLLM bench (bench/) : YES  -- and it is a measurement OF THIS NODE."
    echo "                             Label artifacts with the node; never relabel"
    echo "                             another node's numbers as this one's."
    echo "  CUDA layerwise profiler  : YES  (needs .venv-vllm)"
else
    echo "  real vLLM bench (bench/) : NO   -- no NVIDIA driver reachable."
    echo "                             Committed A40 / A5000 artifacts stay valid as"
    echo "                             measurements of THOSE machines. Do not re-run,"
    echo "                             extend or relabel them here."
    echo "  CUDA layerwise profiler  : NO"
fi

if [ "$RNGD_CARDS" -gt 0 ]; then
    echo "  RNGD profiling           : YES via furiosa.torch (system python3, NOT .venv)"
    echo "                             Check PE availability before planning a run:"
    echo "                             cat /sys/class/rngd_mgmt/rngd!npu<N>pe<M>/alloc_status"
    echo "                             (non-empty == claimed by another tenant's pod)"
else
    echo "  RNGD profiling           : NO   -- no RNGD cards present"
fi
[ "$ATOM_CARDS" -gt 0 ] \
    && echo "  ATOM profiling           : device present; vendor install has been broken (see docs/nodes/npu.md)" \
    || echo "  ATOM profiling           : NO   -- no ATOM devices present"

echo
if [ "$NODE" = "unknown" ]; then
    echo "  !! No accelerator detected at all. Either this is a fourth machine or a"
    echo "     driver is down. Do not assume a node profile -- find out first."
elif [ -n "$NODE_DOC" ] && [ ! -f "$NODE_DOC" ]; then
    echo "  !! $NODE_DOC is missing; the node profile is undocumented."
fi
echo "  Absolute rule 3: never claim results from hardware that is not listed above."
