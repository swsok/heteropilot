"""Compare a re-validation sweep against a committed one, point by point.

`WORK_ORDER_d23_fix_revalidation.md` STEP 3.2/3.3. The simulator is deterministic,
so a sweep point that was decided in both runs must agree to the last decimal. A
field that differs is evidence of silent contamination and has to be chased, not
averaged.

`pd_slo_sweep.py` writes one record per TTFT point — `feasible`, `generated`,
`evaluated`, and the winning plan under `recommended` — so that is the granularity
here. Per-candidate detail lives in the envelope cache, which
`d23_memjson_census.py` and the tables in `docs/d23_revalidation.md` cover.

Usage:
    d23_reval_compare.py <committed.json> <revalidated.json>

Exits 1 if any compared field differs, 0 if none does, 2 if the inputs are not
sweep summaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The fields the work order asks to be compared, plus the identity of the winner.
PLAN_FIELDS = (
    "plan_id",
    "arch",
    "backend_mix",
    "accelerators",
    "tokens_per_joule",
    "slo_goodput_rps",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "average_power_w",
    "slo_attainment",
)
POINT_FIELDS = ("feasible", "generated", "evaluated")


def _points(path: Path) -> dict[float, dict]:
    doc = json.loads(path.read_text())
    sweep = doc.get("sweep")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError(f"{path} has no non-empty 'sweep' list")
    out = {}
    for entry in sweep:
        if "ttft_slo_max_ms" not in entry:
            raise ValueError(f"{path}: a sweep entry has no ttft_slo_max_ms")
        out[float(entry["ttft_slo_max_ms"])] = entry
    return out


def main(before_path: str, after_path: str) -> int:
    try:
        before = _points(Path(before_path))
        after = _points(Path(after_path))
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Both arguments must be pd_slo_sweep.py summary JSONs.", file=sys.stderr)
        return 2

    shared = sorted(set(before) & set(after))
    if not shared:
        print(
            f"ERROR: no TTFT point in common. committed={sorted(before)} "
            f"rerun={sorted(after)}; refusing to compare.",
            file=sys.stderr,
        )
        return 2

    for only, label in ((set(before) - set(after), "committed only"),
                        (set(after) - set(before), "re-run only")):
        if only:
            print(f"note: TTFT points {sorted(only)} are {label} and are skipped")

    differing = 0
    for ttft in shared:
        eo, en = before[ttft], after[ttft]
        print(f"\n--- ttft <= {ttft:.0f} ms")
        for f in POINT_FIELDS:
            a, b = eo.get(f), en.get(f)
            mark = "same" if a == b else "** DIFFERS **"
            differing += a != b
            print(f"    {f:<18} {a!s:<26} {b!s:<26} {mark}")

        ro, rn = eo.get("recommended") or {}, en.get("recommended") or {}
        if not ro and not rn:
            print("    (both infeasible; no winner to compare)")
            print(f"    committed reason: {str(eo.get('reason'))[:90]}")
            print(f"    rerun     reason: {str(en.get('reason'))[:90]}")
            continue
        if bool(ro) != bool(rn):
            # This is the interesting case, not an error: a point that was
            # INFEASIBLE and is now decided, or the reverse.
            side = "re-run only" if rn else "committed only"
            print(f"    ** the winner exists {side} -- the verdict changed **")
            for f in PLAN_FIELDS:
                v = (rn or ro).get(f)
                if v is not None:
                    print(f"    {f:<18} {v}")
            differing += 1
            continue
        for f in PLAN_FIELDS:
            a, b = ro.get(f), rn.get(f)
            if a is None and b is None:
                continue
            mark = "same" if a == b else "** DIFFERS **"
            differing += a != b
            print(f"    {f:<18} {a!s:<26} {b!s:<26} {mark}")

    print(f"\n--- {differing} differing field(s) across {len(shared)} shared point(s) ---")
    if differing:
        print("The simulator is deterministic. A difference in a point that was decided")
        print("both times is evidence of silent contamination: diff the two CSVs for")
        print("that candidate and escalate per the work order's risk table. A point")
        print("that changed from infeasible to decided is a different thing -- it means")
        print("candidates that had no number now have one.")
    return 1 if differing else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
