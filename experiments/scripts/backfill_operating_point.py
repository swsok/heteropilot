"""Add operating points to cache entries written before they were recorded.

`WORK_ORDER_rps_aware.md` rev 2 STEP 5. STEP 4.3 made `SimResult.operating_point`
part of what an envelope-cache entry stores, because deriving it from the run's
CSV at read time meant a cache hit silently dropped the accuracy-domain margin to
zero. Entries written before that carry no operating point and announce the gap,
which is correct but leaves E5 and E6 unable to reuse hours of simulation.

This computes the missing field from the run's OWN artifacts -- the `sim*.csv` and
`cluster.json` in the work directory the entry came from -- which is precisely
what the predictor now does inline. Nothing is invented: an entry whose work
directory is absent is left alone and counted.

    backfill_operating_point.py --cache outputs/.hp-x/cache --work outputs/.hp-x/work
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.util.operating_point import operating_points  # noqa: E402


def safe_id(candidate_id: str) -> str:
    """The predictor's work-directory name for a candidate id."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in candidate_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="write the entries; without it, report what would change")
    args = ap.parse_args()

    filled = skipped_present = skipped_no_run = 0
    for entry in sorted(args.cache.glob("*.json")):
        payload = json.loads(entry.read_text())
        if payload.get("operating_point"):
            skipped_present += 1
            continue
        cid = payload.get("candidate_id")
        if not cid:
            skipped_no_run += 1
            continue
        run_dir = args.work / safe_id(cid)
        csv = next(iter(sorted(run_dir.glob("sim*.csv"))), None)
        cfg = run_dir / "cluster.json"
        if csv is None or not cfg.exists():
            skipped_no_run += 1
            continue
        points = operating_points(csv, json.loads(cfg.read_text()))
        if not points:
            skipped_no_run += 1
            continue
        payload["operating_point"] = {
            hw: {"concurrency": p.concurrency, "phase": p.phase,
                 "requests": p.requests, "wall_s": p.wall_s}
            for hw, p in points.items()
        }
        payload.setdefault("provenance", {})["operating_point_backfilled"] = True
        if args.apply:
            entry.write_text(json.dumps(payload, indent=2, sort_keys=True))
        filled += 1

    verb = "filled" if args.apply else "would fill"
    print(f"{verb} {filled}; {skipped_present} already had one; "
          f"{skipped_no_run} had no usable run directory", file=sys.stderr)
    if not args.apply and filled:
        print("re-run with --apply to write", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
