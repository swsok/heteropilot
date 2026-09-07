"""Do the sweep's candidates differ in the JSON that ASTRA-Sim's temp-file race swaps?

`WORK_ORDER_d23_fix_revalidation.md` STEP 3.1. Before D25, every concurrent
simulation wrote `tmp__mem/<name>.json` to one shared path, so a candidate could
read a *different* candidate's memory configuration. The spike could only ever see
the crashing half of that, because it launched 64 copies of one candidate --
swapping identical files changes nothing. The real sweep ran 64 different ones.

Whether the silent half could have changed a result depends on whether the file
contents actually differ between candidates. This answers that from the committed
work directories, with no simulation: it calls the real
`serving.core.config_builder.build_cluster_config` on each candidate's
`cluster.json` and hashes the `local_mem` / `remote_mem` / `cxl_mem` objects of the
`memory_expansion.json` it generates.

Usage:
    d23_memjson_census.py outputs/.hp-slo-margin18-tight-pd-rngd-gpu [...]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEM_KEYS = ("local_mem", "remote_mem", "cxl_mem")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()[:12]


def census(work_dirs: list[Path]) -> int:
    # build_cluster_config prepends "../" to a relative cluster path and expects to
    # be run from astra-sim/; it imports and runs in .venv without torch (verified
    # by WORK_ORDER_spikes.md STEP B, which used it directly).
    sys.path.insert(0, str(REPO))
    os.chdir(REPO / "astra-sim")
    from serving.core.config_builder import build_cluster_config

    per_key: dict[str, Counter] = {k: Counter() for k in MEM_KEYS}
    contents: dict[str, dict[str, object]] = {k: {} for k in MEM_KEYS}
    single_form = Counter()
    failures: list[tuple[str, str]] = []
    field_values: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    total = 0

    configs: list[Path] = []
    for work in work_dirs:
        base = REPO / work
        # accept either the sweep's --output-dir or its work/ subdirectory
        found = sorted(base.glob("*/cluster.json")) + sorted(base.glob("work/*/cluster.json"))
        if not found:
            print(f"warning: no */cluster.json under {work}", file=sys.stderr)
        configs.extend(found)

    if not configs:
        # A census over nothing must not print a reassuring verdict. The first
        # version of this script did exactly that -- it reported "every candidate
        # is identical" from zero candidates, because the glob was one directory
        # level off.
        print(
            "ERROR: no candidate cluster.json found. Point this at a sweep's "
            "--output-dir or its work/ directory; refusing to draw a conclusion "
            "from an empty set.",
            file=sys.stderr,
        )
        return 2

    for cfg in configs:
        total += 1
        rel = os.path.relpath(cfg, REPO)
        root = tempfile.mkdtemp(prefix="memcensus-")
        try:
            build_cluster_config("./", rel, inputs_root=root)
            with open(os.path.join(root, "memory", "memory_expansion.json")) as f:
                mem = json.load(f)
        except Exception as exc:
            failures.append((cfg.parent.name, f"{type(exc).__name__}: {exc}"))
            continue
        if all(k in mem for k in ("memory-type", "mem-latency", "mem-bw")):
            # main.cc's is_single branch reads --memory-configuration directly and
            # never touches tmp__mem, so these candidates cannot be affected.
            single_form[_digest(mem)] += 1
            continue

        for key in MEM_KEYS:
            if key in mem and isinstance(mem[key], dict):
                d = _digest(mem[key])
                per_key[key][d] += 1
                contents[key][d] = mem[key]
                for field, value in mem[key].items():
                    field_values[key][field][json.dumps(value)] += 1

    print(f"candidates inspected : {total}")
    print(f"  single-form memory config (no tmp__mem at all): {sum(single_form.values())}")
    print(f"  failed to compile                              : {len(failures)}")
    for name, why in failures[:5]:
        print(f"      {name[:60]}: {why}")

    swappable = False
    for key in MEM_KEYS:
        if not per_key[key]:
            continue
        print(f"\n=== {key}: {sum(per_key[key].values())} candidates, "
              f"{len(per_key[key])} distinct content(s) ===")
        for d, n in per_key[key].most_common():
            print(f"  {d}  x{n:<5} {_canonical(contents[key][d])}")
        if len(per_key[key]) > 1:
            swappable = True
            print("  fields that vary:")
            for field, vals in field_values[key].items():
                if len(vals) > 1:
                    print(f"    {field}: " + ", ".join(f"{v} x{n}" for v, n in vals.most_common()))

    print("\n--- verdict ---")
    if not swappable:
        print("Every candidate's memory JSON is IDENTICAL. Reading another candidate's")
        print("file could not have changed any number: the silent half of the D25 race")
        print("was a no-op here, and only its crashing half was ever observable.")
    else:
        print("Candidate memory JSONs DIFFER, so the D25 race had a silent half that")
        print("could in principle change a result. Which fields differ is listed above;")
        print("judge each against what AnalyticalMemory does with it.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(census([Path(a) for a in sys.argv[1:]]))
