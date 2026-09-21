#!/usr/bin/env python
"""Regenerate `profiles/calibration/index.yaml` from the domain files.

The index's own header says "Regenerate rather than hand-edit" and **nothing
could**: `build_domain_index` existed for the drift test
(`tests/test_calibration_condition.py::test_the_index_matches_the_domain_files`)
but no command wrote its output, so the file was maintained by hand against a
checker. That works until it does not, and adding a domain is exactly when it
stops working. `WORK_ORDER_domain_scoping.md` S7.3 needed two new rows, so this
is the missing half.

**Notes are authored, not derived, and that is deliberate.** The drift test
compares every field EXCEPT `note` and then asserts each note is non-empty: the
structural fields are a copy of the domain files and must agree with them, while
the note is the one place a human says what the file is for -- "card-as-device
abstraction, rebuilt at 300 requests (D32)" is not derivable from any field. So
this preserves existing notes by path and **refuses** to write a row it has no
note for, rather than inventing one or leaving it blank for the test to catch
later.

The leading comment block is preserved verbatim for the same reason: it carries
D102's explanation of why two A40 domains coexist and which one the default glob
path reaches, which no generator can reconstruct.

    experiments/scripts/rebuild_domain_index.py \\
        --note profiles/calibration/openloop/rngd_card.accuracy.openloop.yaml="..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.predictor.calibration import (  # noqa: E402
    DOMAIN_INDEX_PATH,
    build_domain_index,
    load_domain_index,
)


def leading_comment(path: Path) -> str:
    """The `#` block at the top, which explains what no field can."""
    out = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        break
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--note", action="append", default=[],
                    metavar="PATH=TEXT",
                    help="the note for a domain this index does not yet carry")
    ap.add_argument("--check", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    index_path = args.root / DOMAIN_INDEX_PATH
    header = leading_comment(index_path)
    existing = {e.path: e.note for e in load_domain_index(args.root).domains}
    for item in args.note:
        path, _, text = item.partition("=")
        if not text:
            print(f"error: --note {item!r} has no text after '='", file=sys.stderr)
            return 1
        existing[path] = text

    rebuilt = build_domain_index(args.root)
    missing = [e.path for e in rebuilt.domains if not existing.get(e.path)]
    if missing:
        print("error: no note for:\n  " + "\n  ".join(missing)
              + "\n\nA note says what the file is FOR, which no field carries. "
                "Pass --note PATH=TEXT for each.", file=sys.stderr)
        return 1
    for entry in rebuilt.domains:
        entry.note = existing[entry.path]

    body = yaml.safe_dump(
        {"domains": [e.model_dump() for e in rebuilt.domains]},
        sort_keys=False, width=100, allow_unicode=True)
    text = f"{header}\n{body}"

    if args.check:
        same = index_path.read_text() == text
        print("index.yaml is up to date" if same
              else "index.yaml would CHANGE; re-run without --check")
        return 0 if same else 1
    index_path.write_text(text)
    print(f"wrote {index_path} ({len(rebuilt.domains)} domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
