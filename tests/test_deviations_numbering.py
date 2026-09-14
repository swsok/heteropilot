"""`docs/deviations.md`'s numbering is a shared counter, and git cannot guard it.

On 2026-09-14 two work orders running in parallel both wrote a `D34` --
`WORK_ORDER_uncertainty_planner.md` STEP B5 (PR #81) and the B4 experiments
(PR #82). The merge did **not** conflict: the two entries land about 190 lines
apart in a 2,500-line file, so there is no textual overlap and git happily
produced a document containing two `## D34` headings. A person noticed. That is
not a defence that repeats.

These two checks are the defence that does. They are deliberately narrow -- a
duplicate heading, and a reference to an entry that does not exist -- because
those are the two failures a reader cannot be relied on to see, and both are
decidable from the file itself.

CLAUDE.md carries the rule they enforce: each concurrent work order takes numbers
from its own block rather than reaching for the next free one.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEVIATIONS = ROOT / "docs/deviations.md"

#: A heading is `## D<n>` with an optional letter suffix -- `D29b` exists because
#: an entry had to be squeezed between two that were already referenced.
_HEADING = re.compile(r"^## (D\d+[a-z]?)\b", re.M)
#: A reference is the bare id in prose or in a comment. Bounded on both sides so
#: `D3` does not match inside `D30`, and a stray `3D` does not match at all.
_REFERENCE = re.compile(r"(?<![0-9A-Za-z_])D(\d{1,2}[a-z]?)(?![0-9A-Za-z_])")
#: `D40-D49` names a RESERVED RANGE, not two citations -- CLAUDE.md's block table
#: is written that way and would otherwise report every block as dangling. Both
#: endpoints are dropped before scanning. Skipping a range that really was a
#: citation only loses coverage; it cannot invent a failure.
_RANGE = re.compile(r"D\d{1,2}\s*[-\u2013\u2014]\s*D\d{1,2}")

#: Where a D-number may be cited. `outputs/` is excluded: it holds captured
#: stdout and result JSON, where a retracted id can legitimately survive as a
#: record of what a past run said.
_SEARCH_ROOTS = ("docs", "planner", "tests", "experiments", "profiles", "CLAUDE.md")
#: Work orders cite D-numbers heavily and are the documents most likely to be
#: edited long after an entry was written -- exactly where a renumber goes stale.
_SEARCH_GLOBS = ("WORK_ORDER_*.md",)
_SUFFIXES = (".md", ".py", ".yaml", ".yml", ".sh")


def _entries() -> list[str]:
    return _HEADING.findall(DEVIATIONS.read_text())


def _files():
    for name in _SEARCH_ROOTS:
        path = ROOT / name
        if path.is_file():
            yield path
        elif path.is_dir():
            for found in path.rglob("*"):
                if found.suffix in _SUFFIXES and found.is_file():
                    yield found
    for pattern in _SEARCH_GLOBS:
        yield from sorted(ROOT.glob(pattern))


def test_no_deviation_number_is_used_twice() -> None:
    """The 2026-09-14 collision, caught mechanically.

    Two entries under one number is not a merge conflict and never will be, so
    nothing else in the toolchain reports it.
    """
    counts = Counter(_entries())
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    assert not duplicates, (
        f"{duplicates} appear more than once in docs/deviations.md. Two work "
        f"orders have taken the same number; see CLAUDE.md, 'Claim a D-number "
        f"from your work order's block'."
    )


def test_every_cited_deviation_exists() -> None:
    """A renumber that misses a citation leaves a pointer to nothing.

    This is the other half of the same problem: fixing a collision by moving a
    number is only safe if every reference moves with it, and there are roughly
    1,100 of them.
    """
    known = set(_entries())
    dangling: dict[str, set[str]] = defaultdict(set)
    for path in _files():
        if path == Path(__file__).resolve():
            continue  # this file names the blocks as literals, not as citations
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for number in _REFERENCE.findall(_RANGE.sub(" ", text)):
            cited = f"D{number}"
            if cited not in known:
                dangling[cited].add(str(path.relative_to(ROOT)))
    assert not dangling, "\n".join(
        f"{cited} is cited in {len(where)} file(s) but has no entry in "
        f"docs/deviations.md, e.g. {sorted(where)[0]}"
        for cited, where in sorted(dangling.items())
    )


#: Below this, numbering is history and must be dense. At and above it, a gap is
#: a reservation -- CLAUDE.md's D37-D39 escape hatch and the D40+ blocks.
HISTORY_CEILING = 37
#: The blocks CLAUDE.md assigns, plus the block-free D37-D39.
BLOCK_RANGE = (37, 89)


def test_the_numbering_is_dense_so_a_gap_means_a_reservation() -> None:
    """Dense through the historical range; above it, a gap is a reservation.

    A gap BELOW the blocks would be ambiguous -- retired entry, or someone's
    reservation? Blocks start at D40 precisely so that every gap above D36 is a
    reservation and every number below it is history. D37-D39 are the exception
    CLAUDE.md names, and are free.

    The bound used to be `numbers[-1]`, the largest entry, which contradicted
    that docstring: the first entry to obey the block rule -- D40, the earliest
    number `WORK_ORDER_uncertainty_planner.md` is allowed to take -- reported
    D36-D39 as holes and failed. Density is therefore checked through the
    historical range only, and the range above it is checked for something
    stronger instead: that every number there falls in a block CLAUDE.md
    actually documents, so `D95` cannot be invented either.
    """
    numbers = sorted({int(re.match(r"D(\d+)", e).group(1)) for e in _entries()})
    assert numbers[0] == 1

    history = [n for n in numbers if n < HISTORY_CEILING]
    missing = [n for n in range(1, max(history) + 1) if n not in history]
    assert not missing, (
        f"D{missing} has no entry. A hole below the reserved blocks is ambiguous: "
        f"say in docs/deviations.md that the number is retired, or reuse it."
    )

    lo, hi = BLOCK_RANGE
    stray = [n for n in numbers if n >= HISTORY_CEILING and not lo <= n <= hi]
    assert not stray, (
        f"D{stray} is above the historical range but outside every block "
        f"CLAUDE.md assigns (D{lo}-D{hi}). Claim a block in CLAUDE.md in the same "
        f"commit, or take a number from the one your work order already owns."
    )


@pytest.mark.parametrize("block", ["D40", "D50", "D60", "D70", "D80"])
def test_claude_md_still_documents_the_blocks(block: str) -> None:
    """The table is the whole mechanism -- a test that enforces a rule nobody can
    find is worse than no test."""
    assert block in (ROOT / "CLAUDE.md").read_text(), (
        f"the {block} block vanished from CLAUDE.md; the numbering rule and the "
        f"check that enforces it have to stay together"
    )
