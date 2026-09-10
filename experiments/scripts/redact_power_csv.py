"""Rewrite RNGD serial numbers and UUIDs to stable pseudonyms before committing.

`WORK_ORDER_rps_aware.md` rev 2 STEP 3.1. `power_sampler.sh` records `device_sn`
and `pci_bdf` because `dev_name` re-enumerates and a device-pinned measurement
must select on something stable. Serials identify physical hardware, and this
repository has not committed one yet -- the STEP 0 `furiosa-smi` captures were
redacted for the same reason.

Redaction here preserves the analytical content: each distinct serial becomes
`RNGD-A`, `RNGD-B`, ... in first-appearance order, so rows can still be grouped
by device and two files from the same session stay comparable. The mapping is
printed, never written into the artifact.

The raw JSON columns carry the same identifiers, so they are rewritten too --
redacting only the named columns would leave the serial in the file.

    experiments/scripts/redact_power_csv.py power_c8.csv [more.csv ...] --in-place
"""

from __future__ import annotations

import argparse
import re
import string
import sys
from pathlib import Path

SERIAL_RE = re.compile(r"RNG[0-9A-Z]{10,}")
UUID_RE = re.compile(r"\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\b")


def _names():
    """RNGD-A .. RNGD-Z, then RNGD-AA onwards. More than 26 cards is not expected,
    but running out of names silently would corrupt the grouping."""
    for c in string.ascii_uppercase:
        yield f"RNGD-{c}"
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            yield f"RNGD-{a}{b}"


def redact(text: str, mapping: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    """Replace serials with pseudonyms and UUIDs with a fixed marker.

    `mapping` is threaded through so several files from one session agree on
    which card is `RNGD-A`.
    """
    mapping = dict(mapping or {})
    names = _names()
    for used in mapping.values():
        # keep the generator ahead of names already assigned
        for candidate in names:
            if candidate == used:
                break
    for serial in SERIAL_RE.findall(text):
        if serial not in mapping:
            mapping[serial] = next(names)
    for serial, alias in mapping.items():
        text = text.replace(serial, alias)
    # UUIDs carry no grouping information the serial does not already carry.
    text = UUID_RE.sub("<redacted-uuid>", text)
    return text, mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite the files; without it, report what would change")
    args = ap.parse_args()

    mapping: dict[str, str] = {}
    changed = 0
    for path in args.paths:
        original = path.read_text()
        redacted, mapping = redact(original, mapping)
        if redacted == original:
            print(f"  {path}: already clean")
            continue
        changed += 1
        if args.in_place:
            path.write_text(redacted)
            print(f"  {path}: redacted")
        else:
            print(f"  {path}: WOULD redact")

    if mapping:
        print("\nmapping (not written to any file):", file=sys.stderr)
        for serial, alias in mapping.items():
            print(f"  {serial} -> {alias}", file=sys.stderr)
    if changed and not args.in_place:
        print("\nre-run with --in-place to apply", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
