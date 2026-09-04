#!/usr/bin/env bash
# Run a simulator command with ASTRA-Sim's shared temp directory made private.
#
# WORK_ORDER_spikes.md STEP A / deviations D23. ASTRA-Sim's analytical backend
# writes, reads and removes `tmp__mem/<name>.json` at a fixed path relative to its
# working directory, with no pid and no run id
# (astra-sim/.../congestion_unaware/main.cc:27, three times per start). Every
# concurrent simulation shares one working directory, so one process deletes the
# file another is opening. Measured on this node: of 64 processes launched
# together on identical inputs, 13 died -- five with
#
#     Unable to open file: tmp__mem/remote_mem.json
#
# and eight with SIGABRT. With this wrapper, 64 of 64 survive.
#
# `--run-id` / `--inputs-root` do not help: they isolate the *input tree*, and
# tmp__mem/ is outside it.
#
# The fix here is an unprivileged mount namespace with a private tmpfs over that
# one directory. Nothing in serving/ or astra-sim/ is modified, which is why this
# is usable today -- absolute rule 1 forbids touching astra-sim/ before Phase 5,
# and the real fix belongs upstream (docs/upstream_issues/).
#
# Usage:
#   astra_isolated.sh --check              # is isolation available here? 0 / 2
#   astra_isolated.sh -- <command...>      # run it isolated
#
# Exit codes:
#   2   isolation is unavailable or failed to set up -- the command is NOT run.
#       Refusing is deliberate: running unisolated would silently reintroduce the
#       race, and its failure mode is a four-hour timeout, not an error.
#   *   whatever the command exited with.
#
# Note the wrapper is not a substitute for fixing this. Other shared-resource
# paths may exist; tmp__mem is the one that has been measured.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TARGET="${ASTRA_TMP_DIR:-${REPO_ROOT}/astra-sim/tmp__mem}"

usage() {
    echo "usage: astra_isolated.sh --check | astra_isolated.sh -- <command...>" >&2
}

isolation_available() {
    command -v unshare >/dev/null 2>&1 || {
        echo "astra_isolated: unshare(1) not found (util-linux)." >&2
        return 1
    }
    unshare -Urm true 2>/dev/null || {
        echo "astra_isolated: unprivileged user+mount namespaces are refused here." >&2
        echo "astra_isolated: check /proc/sys/kernel/unprivileged_userns_clone (want 1) and" >&2
        echo "astra_isolated: /proc/sys/kernel/apparmor_restrict_unprivileged_userns (want 0)." >&2
        return 1
    }
    return 0
}

case "${1-}" in
    --check)
        if isolation_available; then
            echo "astra_isolated: available (mount point ${TARGET})"
            exit 0
        fi
        exit 2
        ;;
    --) shift ;;
    *)  usage; exit 2 ;;
esac

[ $# -gt 0 ] || { usage; exit 2; }

isolation_available || exit 2

# The mount point has to exist on the real filesystem before it can be covered.
# It stays empty: everything written to it lands in the per-process tmpfs. Since
# astra-sim/ is a submodule, an empty untracked directory there is the entire
# footprint this wrapper leaves behind.
mkdir -p "$TARGET" || {
    echo "astra_isolated: cannot create the mount point $TARGET" >&2
    exit 2
}

# --propagation private so the tmpfs is not shared back with the host namespace.
# `exec` keeps the command as the namespace's own process, so its exit code is
# this script's and its children -- including AnalyticalAstra -- inherit the mount.
unshare -Urm --propagation private /bin/sh -c '
    mount -t tmpfs none "$1" || {
        echo "astra_isolated: failed to mount tmpfs on $1" >&2
        exit 2
    }
    shift
    exec "$@"
' _ "$TARGET" "$@"
