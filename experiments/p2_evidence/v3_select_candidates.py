#!/usr/bin/env python
"""V3: pick the candidates whose deployment would actually discriminate the four rules.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V3, the CPU half. E-A1 recorded how
many candidates each margin rule passes (70 / 10 / 50, with 30 held) but never
which of them the hardware agrees with, so the disclosure's §6 numbers -- the
pass rate of candidates that really violate, the exclusion rate of candidates
that really satisfy -- have no ground truth. This script chooses the candidates
whose measurement would supply it.

**It selects, it does not deploy.** The work order requires the table to be
confirmed before anything is deployed.

Four groups, from the work order:

  P1  feasible under all four rules             -- the agreed baseline
  P2  feasible unmargined, rejected per-point   -- the optimistic candidate the
                                                   invention claims to catch
  P3  rejected by the global 18 %, feasible
      per-point                                 -- the candidate it rescues
  P4  held as `outside_calibration_domain`      -- was the refusal warranted?

Run:  .venv/bin/python experiments/p2_evidence/v3_select_candidates.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

EA1 = REPO / "outputs/uncertainty/ea1/ea1_margin_modes.json"
CACHE = REPO / "outputs/uncertainty/ea1/cache"

#: The rules, in the order the work order names them.
RULES = ("a_margin0", "b_global18", "c_accuracy_domain", "d_refuse")

#: This node has eight A40s and the fixture's two CUDA islands are 4 + 4, so any
#: A40-only candidate fits. Kept explicit because a candidate that does not fit
#: is not a selection problem, it is an unrunnable row.
A40_BUDGET = 8

TPOT_SLO_MS = 50.0
TTFT_SLO_MS = 25000.0


def load() -> tuple[dict, dict]:
    ea1 = json.loads(EA1.read_text())
    cache = {}
    for path in glob.glob(str(CACHE / "*.json")):
        entry = json.loads(Path(path).read_text())
        if "candidate_id" in entry:
            cache[entry["candidate_id"]] = entry
    return ea1, cache


def is_a40_only(cid: str) -> bool:
    return "cuda-a40" in cid and "furiosa" not in cid


def is_single_island(cid: str) -> bool:
    """Not a `mix(...)` candidate.

    D40 is the reason this matters. `EnvelopeCache` keys on a placement string
    that does not record WHICH island got which share, so two candidates that
    differ only by mirroring collapse to one cached entry and one of them is
    served the other's numbers -- measured at 8.5 % on TTFT for one A40 pair.
    D40 also establishes that dedup IS sound for single-island aggregated
    candidates. V3 compares a prediction against a measurement, so a candidate
    whose prediction might belong to its mirror is worthless here.
    """
    return not cid.startswith("mix(")


def accelerators(cid: str) -> int:
    """TP x DP summed over the candidate's islands."""
    total = 0
    for tp, dp in re.findall(r"-tp(\d+)-dp(\d+)", cid):
        total += int(tp) * int(dp)
    return total


def parse_reason(reason: str) -> dict:
    """The metrics a rejected verdict carries in its reason string."""
    out = {}
    for key, pattern in (("p99_ttft_ms", r"p99_ttft_ms=([\d.]+)"),
                         ("p99_tpot_ms", r"p99_tpot_ms=([\d.]+)")):
        m = re.search(pattern, reason)
        if m:
            out[key] = float(m.group(1))
    return out


def row_for(cid: str, ea1: dict, cache: dict) -> dict:
    v = {r: ea1["conditions"][r]["verdicts"][cid] for r in RULES}
    entry = cache.get(cid, {})
    metrics = entry.get("metrics", {})
    # A rejected verdict states the metrics it judged; a feasible one does not,
    # so the cache is the only source for those.
    from_reason: dict = {}
    for r in RULES:
        from_reason.update(parse_reason(v[r].get("reason", "")))
    return {
        "candidate_id": cid,
        "accelerators": accelerators(cid),
        "single_island": is_single_island(cid),
        "cached": bool(metrics),
        "p99_tpot_ms": metrics.get("p99_tpot_ms") or from_reason.get("p99_tpot_ms"),
        "p99_ttft_ms": metrics.get("p99_ttft_ms") or from_reason.get("p99_ttft_ms"),
        "served_concurrency": metrics.get("served_concurrency"),
        "tokens_per_joule": metrics.get("tokens_per_joule"),
        "verdicts": {r: v[r]["verdict"] for r in RULES},
        "stages": {r: v[r]["stage"] for r in RULES},
        "tpot_margin_pct": {r: v[r].get("tpot_percent") for r in RULES},
        "operating_point": {r: v[r].get("concurrency") for r in RULES},
        "basis": v["c_accuracy_domain"].get("basis", ""),
        "ttft_margin_pct": {r: v[r].get("ttft_percent") for r in RULES},
        "reasons": {r: v[r].get("reason", "") for r in RULES},
    }


def driver_metric(reason: str) -> str:
    """Which SLO a rejection was decided on. A candidate can breach either."""
    has_tpot = "p99_tpot_ms" in reason
    has_ttft = "p99_ttft_ms" in reason
    return ("TPOT+TTFT" if has_tpot and has_ttft
            else "TPOT" if has_tpot else "TTFT" if has_ttft else "")


def overshoot_ms(reason: str) -> float | None:
    """How far past its target the deciding metric landed.

    This is the number that decides whether a deployment can TELL. A verdict that
    turns on 0.3 ms of p99 TPOT is not a verdict a measurement can confirm or
    refute; one that turns on 2.7 s of p99 TTFT is.
    """
    worst = None
    for pattern in (r"p99_tpot_ms=([\d.]+) vs target ([\d.]+)",
                    r"p99_ttft_ms=([\d.]+) vs target ([\d.]+)"):
        m = re.search(pattern, reason)
        if m:
            gap = float(m.group(1)) - float(m.group(2))
            worst = gap if worst is None else max(worst, gap)
    return worst


def classify(row: dict) -> str | None:
    """Which of the work order's four groups this candidate belongs to, if any."""
    a, b, c, d = (row["verdicts"][r] for r in RULES)
    held = row["stages"]["d_refuse"] == "outside_calibration_domain"
    if held:
        return "P4"
    if a == b == c == d == "feasible":
        return "P1"
    if a == "feasible" and c == "rejected":
        return "P2"
    if b == "rejected" and c == "feasible":
        return "P3"
    return None


def usable(row: dict) -> bool:
    """A row that could actually be deployed and compared on this node.

    `mix(...)` candidates are NOT excluded here -- P3 has none but mixes, so
    excluding them would silently empty a group the work order asks for. They are
    flagged instead (`single_island`), because D40 means their cached prediction
    may belong to their mirror and a V3 comparison needs the prediction to be
    theirs. A flagged candidate has to be re-simulated before it is deployed.
    """
    return row["accelerators"] <= A40_BUDGET and row["p99_tpot_ms"] is not None


def rank(group: str, row: dict) -> tuple:
    """Within a group, the clearest case first.

    P2 and P3 are the cases the disclosure rests on, so the clearest is the one
    whose two rules disagree by the widest TPOT margin -- a candidate that only
    just changes verdict would need a measurement more precise than the run is.
    P1 and P4 want the candidate closest to the SLO, where a measurement is most
    informative.
    """
    if group in ("P2", "P3"):
        # Widest measurable separation first, then a prediction that is this
        # candidate's own (D40), then the smallest deployment.
        rule = "c_accuracy_domain" if group == "P2" else "b_global18"
        gap = overshoot_ms(row["reasons"][rule]) or 0.0
        return (-gap, not row["single_island"], row["accelerators"])
    # P1 and P4 both have single-island members, and a baseline whose prediction
    # might belong to its mirror is a poor baseline -- so D40-cleanliness comes
    # first here. P2 and P3 cannot afford that: their single-island options are
    # unmeasurable or absent, and an unmeasurable candidate is worse than a
    # re-simulation.
    return (not row["single_island"],
            abs((row["p99_tpot_ms"] or 0.0) - TPOT_SLO_MS), row["accelerators"])


def main() -> None:
    ea1, cache = load()
    ids = [c for c in ea1["conditions"]["a_margin0"]["verdicts"] if is_a40_only(c)]
    rows = [row_for(c, ea1, cache) for c in ids]

    groups: dict[str, list[dict]] = {g: [] for g in ("P1", "P2", "P3", "P4")}
    for row in rows:
        g = classify(row)
        if g and usable(row):
            groups[g].append(row)
    for g in groups:
        groups[g].sort(key=lambda r: rank(g, r))

    print(f"A40-only candidates: {len(ids)}  "
          f"(single-island {sum(r['single_island'] for r in rows)}, "
          f"cached {sum(r['cached'] for r in rows)})\n")

    out = {"fixture": ea1["fixture"]["cluster"], "service": ea1["fixture"]["service"],
           "slo": {"tpot_ms": TPOT_SLO_MS, "ttft_ms": TTFT_SLO_MS},
           "rules": {r: ea1["conditions"][r]["label"] for r in RULES},
           "groups": {}}

    for g, members in groups.items():
        print(f"=== {g}: {len(members)} usable candidate(s) ===")
        for i, row in enumerate(members[:3]):
            tag = "PRIMARY " if i == 0 else f"alt {i}   "
            print(f" {tag}{row['candidate_id']}")
            def _f(x, nd=2):
                return "n/a" if x is None else f"{x:.{nd}f}"
            print(f"          accel={row['accelerators']} "
                  f"p99_tpot={_f(row['p99_tpot_ms'])} "
                  f"p99_ttft={_f(row['p99_ttft_ms'], 1)} "
                  f"L={_f(row['served_concurrency'], 1)} "
                  f"cached={row['cached']}")
            print("          verdicts: " + "  ".join(
                f"{r.split('_')[0]}={row['verdicts'][r]}"
                + ("(held)" if row["stages"][r] == "outside_calibration_domain" else "")
                for r in RULES))
            print("          tpot margin %: " + "  ".join(
                f"{r.split('_')[0]}={row['tpot_margin_pct'][r]:.2f}" for r in RULES))
            rule = ("c_accuracy_domain" if g == "P2"
                    else "b_global18" if g == "P3" else "d_refuse")
            gap = overshoot_ms(row["reasons"][rule])
            print(f"          decided on {driver_metric(row['reasons'][rule]) or '-'}"
                  f" by {_f(gap)} ms"
                  + ("" if row["single_island"] else "   [D40: mix, re-simulate first]"))
        out["groups"][g] = {"count": len(members),
                            "primary": members[0] if members else None,
                            "alternates": members[1:3]}
        print()

    dest = REPO / "experiments/p2_evidence/results/v3_candidates.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
