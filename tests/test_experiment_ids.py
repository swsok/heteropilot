"""An experiment id is a name within a work order, and nothing says which one.

`docs/deviations.md`'s numbering had this failure and a test now guards it
(`test_deviations_numbering.py`). Experiment ids have the same shape and are one
step behind: **the collision is already written down, it simply has not fired.**
`WORK_ORDER_cxl_kv_pool.md` defines `E1`-`E6`; `WORK_ORDER_tiered_profiles.md`
defines `E1`-`E4` and `WORK_ORDER_rps_aware.md` defines `E5`-`E7`. Meanwhile
`docs/CLAIMS.md`, `docs/HANDOVER.md` and `docs/PROJECT_REPORT.md` cite a bare
`E5`/`E6` meaning `rps_aware`'s. That reads unambiguously today only because the
CXL work order has not run.

This test does **not** try to fix that. Renaming would touch ~460 references and
is exactly the large risky edit CLAUDE.md's block rule exists to avoid. It freezes
the six known collisions instead and fails on a seventh, so the backlog can be
worked through without new ones accumulating behind it.

CLAUDE.md carries the rule: a new experiment takes a tagged id, `E-<tag><n>`,
whose tag belongs to one work order.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: An experiment is DECLARED by a heading (`### E5.`, `## E6 ...`) or by a bold
#: run at the start of a line (`**E-B1** ...`). A mention in running prose is a
#: reference, not a declaration, and must not make the mentioning file an owner.
DECLARATION = re.compile(r"(?m)^(?:#{2,4} +|\*\*)(E-?[A-Z]?\d+)\b")

#: Collisions that predate the rule, each `id -> the work orders that define it`.
#: Every one involves `cxl_kv_pool`, which has not run. Shrink this list as they
#: are resolved; never add to it. A new entry here means the rule was bypassed.
KNOWN_LEGACY: dict[str, set[str]] = {
    "E1": {"WORK_ORDER_cxl_kv_pool.md", "WORK_ORDER_tiered_profiles.md"},
    "E2": {"WORK_ORDER_cxl_kv_pool.md", "WORK_ORDER_tiered_profiles.md"},
    "E3": {"WORK_ORDER_cxl_kv_pool.md", "WORK_ORDER_tiered_profiles.md"},
    "E4": {"WORK_ORDER_cxl_kv_pool.md", "WORK_ORDER_tiered_profiles.md"},
    "E5": {"WORK_ORDER_cxl_kv_pool.md", "WORK_ORDER_rps_aware.md"},
    "E6": {"WORK_ORDER_cxl_kv_pool.md", "WORK_ORDER_rps_aware.md"},
}


def _owners() -> dict[str, set[str]]:
    """`experiment id -> the work orders that DECLARE it`."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in sorted(ROOT.glob("WORK_ORDER_*.md")):
        for ident in set(DECLARATION.findall(path.read_text())):
            out[ident].add(path.name)
    return out


def test_no_new_work_order_collision() -> None:
    """Two work orders declaring one id, beyond the frozen backlog."""
    owners = _owners()
    fresh = {
        ident: who
        for ident, who in owners.items()
        if len(who) > 1 and who != KNOWN_LEGACY.get(ident)
    }
    assert not fresh, "\n".join(
        f"{ident} is declared by {', '.join(sorted(who))} — give the new one a "
        f"tagged id (E-<tag><n>) from CLAUDE.md's table instead of a bare number"
        for ident, who in sorted(fresh.items())
    )


def test_the_legacy_list_does_not_grow_stale() -> None:
    """A frozen list that stops matching reality is worse than none at all.

    If a collision here has been resolved, the entry must go — otherwise the
    list quietly licenses re-introducing it.
    """
    owners = _owners()
    stale = {
        ident: sorted(who)
        for ident, who in KNOWN_LEGACY.items()
        if owners.get(ident, set()) != who
    }
    assert not stale, (
        f"KNOWN_LEGACY no longer matches the work orders for {sorted(stale)}. If "
        f"the collision was resolved, delete the entry; if the owners changed, "
        f"the rule was bypassed."
    )


def test_a_tagged_id_is_owned_by_exactly_one_work_order() -> None:
    """The whole point of a tag: `E-A1` may mean one thing, repo-wide."""
    shared = {
        ident: sorted(who)
        for ident, who in _owners().items()
        if "-" in ident and len(who) > 1
    }
    assert not shared, (
        f"tagged ids declared by more than one work order: {shared}. A tag belongs "
        f"to one work order — claim a different one in CLAUDE.md's table."
    )


def test_claude_md_still_documents_the_tag_rule() -> None:
    """A rule nobody can find is worse than no rule — the sibling test says so
    about D-blocks, and it is just as true here."""
    text = (ROOT / "CLAUDE.md").read_text()
    assert "E-<tag><n>" in text, (
        "the experiment-id tag rule vanished from CLAUDE.md; the rule and the "
        "check that enforces it have to stay together"
    )
    assert "E-A*" in text, "the uncertainty planner's claimed tags left the table"
