#!/usr/bin/env bash
# Run a simulator command and kill it as soon as it is provably livelocked.
#
# WORK_ORDER_spikes.md STEP A. D23 is not a timeout: the simulator's clock keeps
# advancing (52,903 progress ticks in one hour) while no work retires. Waiting for
# --timeout to expire therefore costs the full ceiling for no information, which is
# how the tight-TTFT re-run spent 4h37m. This watcher reads the progress lines the
# simulator already prints at --log-interval and stops the run in seconds once the
# signature is unambiguous.
#
# The signature, per instance, from outputs/pd_slo_sweep_margin18/tight/
# retry3600_livelock_evidence.txt:
#
#   running count constant and > 0   -- a request is admitted and never retires
#   memory %       constant          -- no KV block is ever allocated
#   waiting count  never decreasing  -- the queue only fills
#
# A healthy run breaks all three within a tick or two: the completed control in
# outputs/.hp-pd-slo/work/*/sim1.log moves 8 -> 18 running and 16064 -> 17204 MB.
#
# Usage:
#   livelock_watch.sh [-n TICKS] [-s SECONDS] [-g SECONDS] [-t SECONDS] [-q] -- <command...>
#
#   -n  consecutive stalled ticks before declaring livelock (default 300)
#   -s  seconds of silence AFTER the first progress line before declaring a stall
#       (default 120). The stall detector above only fires on runs that are
#       talking; this catches one that stops talking altogether.
#   -g  grace in seconds for the FIRST progress line (default 900). A healthy
#       N=300 run spends ~740 log lines on banner and graph generation before its
#       first tick, so this must stay generous. A run whose ASTRA-Sim child dies
#       at startup never emits one at all -- that is what -g is for.
#   -t  overall wall-clock ceiling in seconds (default 0 = none)
#   -q  do not forward the command's output to stdout (the log file still gets it
#       if the caller redirects; useful in tests)
#
# Exit codes:
#   3    LIVELOCK -- the stall signature held for -n ticks
#   4    NO PROGRESS -- silent past -s (or never spoke, past -g). Distinct from 3
#        because the cause is different: 3 is a run whose scheduler spins, 4 is
#        typically a dead ASTRA-Sim child that the frontend never notices
#        (serving/core/controller.py read_wait loops forever on EOF).
#   124  the -t ceiling expired (same code `timeout` uses)
#   *    whatever the command exited with (0 = the simulation completed)

set -uo pipefail

TICKS=300
SILENCE=120
GRACE=900
TIMEOUT=0
QUIET=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n) TICKS="$2"; shift 2 ;;
        -s) SILENCE="$2"; shift 2 ;;
        -g) GRACE="$2"; shift 2 ;;
        -t) TIMEOUT="$2"; shift 2 ;;
        -q) QUIET=1; shift ;;
        --) shift; break ;;
        *)  echo "livelock_watch: unexpected argument '$1' (did you forget --?)" >&2
            exit 2 ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "usage: livelock_watch.sh [-n TICKS] [-s SECONDS] [-g SECONDS] [-t SECONDS] [-q] -- <command...>" >&2
    exit 2
fi

FIFO=$(mktemp -u -t livelock_watch.XXXXXX)
mkfifo "$FIFO" || exit 2
cleanup() {
    # A drain, if one was started, must not outlive us. Killed by PID, never by
    # pattern -- `pkill -f livelock_watch` would match this very script.
    [ -n "${DRAIN:-}" ] && kill "$DRAIN" 2>/dev/null
    exec 3<&- 2>/dev/null
    rm -f "$FIFO"
}
trap cleanup EXIT

# Own process group, so a kill reaches the simulator's children (python -m serving
# spawns ASTRA-Sim) and not just the wrapper.
if [ "$TIMEOUT" -gt 0 ] 2>/dev/null; then
    setsid timeout "$TIMEOUT" "$@" >"$FIFO" 2>&1 &
else
    setsid "$@" >"$FIFO" 2>&1 &
fi
CHILD=$!

# Open the read end ONCE, on fd 3, and read the loop and the drain from it.
#
# This has to happen after the child is launched: opening a FIFO read-only
# blocks in open(2) until a writer appears, and the child's `>"$FIFO"` is that
# writer. The two rendezvous here.
#
# Why a named fd at all: the drain below used to reopen the FIFO with
# `cat "$FIFO"`, and by then the child had already been TERMed, so there was no
# writer left and that open() parked in `wait_for_partner` forever. It was
# never killed or waited for, and the EXIT trap then unlinked the name out from
# under it. **250 orphaned `cat` processes accumulated that way between
# 2026-09-10 and 2026-09-18**, one per run ending in verdict 3 or 4. Reading
# from an already-open descriptor cannot block in open() and cannot leak.
#
# Read-only and not `3<>`: holding a write end too would mean the loop never
# sees EOF when the child exits, and a clean run would be misreported as a
# no-progress livelock.
#
# The loop below reads from this descriptor (`done <&3`) and MUST NOT reopen the
# FIFO by name. It used to end `done <"$FIFO"`, which is a second open(2), and
# that open blocks until a writer appears -- a writer that a child exiting
# faster than the shell reaches this line has already taken with it. The parent
# then parked in `pipe_wait` forever, holding fd 3, with no child left. Measured
# at 5 hangs in 40 runs of `livelock_watch.sh -q -- bash -c 'exit 7'`, which is
# the intermittent red `tests/test_livelock_watch.py` showed on `main`: only the
# cases whose command exits promptly can lose that race, which is why a
# different test failed each run and each one passed alone.
exec 3<"$FIFO"

VERDICT=0          # 0 none, 3 livelock, 4 no-progress
STARTED=$(date +%s)
LAST_TICK=0        # epoch of the last progress line; 0 = none seen yet

# `read -t` so an entirely silent child is still noticed. Without it the loop
# blocks forever on the FIFO, which is exactly how four H6 runs burned their
# full 1800 s ceiling with a dead ASTRA-Sim child and no output at all.
while true; do
    if IFS= read -r -t 5 line; then
        :
    else
        rc=$?
        # >128 is the read timeout; anything else is EOF, i.e. the child is done.
        [ "$rc" -le 128 ] && break
        line=""
    fi

    if [ -n "$line" ]; then
        [ "$QUIET" -eq 0 ] && printf '%s\n' "$line"
    fi

    now=$(date +%s)
    if [ "$LAST_TICK" -eq 0 ]; then
        if [ "$GRACE" -gt 0 ] && [ $((now - STARTED)) -ge "$GRACE" ]; then
            echo "livelock_watch: NO PROGRESS -- not one progress line in ${GRACE}s." >&2
            echo "livelock_watch: the run never started reporting. Check whether the" >&2
            echo "livelock_watch: ASTRA-Sim child died (a zombie child plus 100% CPU in" >&2
            echo "livelock_watch: the parent is controller.read_wait spinning on EOF)." >&2
            VERDICT=4; kill -TERM -"$CHILD" 2>/dev/null; break
        fi
    elif [ "$SILENCE" -gt 0 ] && [ $((now - LAST_TICK)) -ge "$SILENCE" ]; then
        echo "livelock_watch: NO PROGRESS -- silent for ${SILENCE}s after reporting." >&2
        VERDICT=4; kill -TERM -"$CHILD" 2>/dev/null; break
    fi

    case "$line" in
        *"Running Instance["*) ;;
        *) continue ;;
    esac
    LAST_TICK=$now
    # Running Instance[0]: 1 reqs, Waiting: 7 reqs, ... 3858.51 MB (9.304 % Used)
    parsed=$(printf '%s\n' "$line" | sed -n \
        's/.*Running Instance\[\([0-9]\+\)\]: \([0-9]\+\) reqs, Waiting: \([0-9]\+\) reqs.*(\([0-9.]\+\) % Used).*/\1 \2 \3 \4/p')
    [ -z "$parsed" ] && continue
    set -- $parsed
    idx=$1; running=$2; waiting=$3; mem=$4
    eval "prev_r=\${R_$idx-}"; eval "prev_w=\${W_$idx-}"; eval "prev_m=\${M_$idx-}"
    eval "streak=\${S_$idx:-0}"
    if [ -n "$prev_r" ] && [ "$running" = "$prev_r" ] && [ "$running" != "0" ] \
       && [ "$mem" = "$prev_m" ] && [ "$waiting" -ge "$prev_w" ]; then
        streak=$((streak + 1))
    else
        streak=0
    fi
    eval "R_$idx=\$running"; eval "W_$idx=\$waiting"; eval "M_$idx=\$mem"; eval "S_$idx=\$streak"
    if [ "$streak" -ge "$TICKS" ]; then
        echo "livelock_watch: LIVELOCK -- Instance[$idx] held running=$running, mem=$mem %," >&2
        echo "livelock_watch: waiting non-decreasing at $waiting, for $streak consecutive ticks." >&2
        echo "livelock_watch: killing the process group (this is D23, not a timeout)." >&2
        VERDICT=3
        kill -TERM -"$CHILD" 2>/dev/null
        break
    fi
done <&3

# Drain anything still buffered so the child is not killed by SIGPIPE mid-write.
# `<&3` inherits the descriptor opened above instead of reopening the FIFO, so
# it cannot park in open(); and it is reaped here rather than left to init.
if [ "$VERDICT" -ne 0 ]; then
    cat <&3 >/dev/null 2>&1 &
    DRAIN=$!
    sleep 1
    kill -KILL -"$CHILD" 2>/dev/null
    kill "$DRAIN" 2>/dev/null
    wait "$DRAIN" 2>/dev/null
    DRAIN=""
fi

wait "$CHILD"
RC=$?
[ "$VERDICT" -ne 0 ] && exit "$VERDICT"
exit "$RC"
