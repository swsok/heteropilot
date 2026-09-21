#!/usr/bin/env python
"""How wrong is the simulator OPEN LOOP, at the operating points a plan reaches?

`WORK_ORDER_domain_scoping.md` STEP S7.3. The closed-loop analogue is
`lowload_sim_error.py`; this is the open-loop one, and the difference is not
cosmetic.

**Why this exists rather than more points on the committed domain.** S4 stated
`arrival_process: closed_loop` on `rngd_card_edf.yaml` (D113), and
`python -m planner plan` always replays an arrival trace, so under the default
`condition_mismatch: refuse` **no RNGD candidate may consult that domain at all**
— 72 of 72 rows in S7.0's sweep were held on that field. What answers them is not
a wider load range but a refit under the arrival process the planner actually
performs. Two protocols must never share one interpolation axis (D22 is what that
error looks like; D102 keeps the A40's two harnesses in two files), so this writes
a **new** domain file and touches no committed `points` value (rule A3).

**What D19 no longer blocks.** The closed-loop pairing could compare only TPOT:
a burst against a spread arrival process differs and the difference lands
entirely in TTFT. Here both sides replay the same arrival process at the same
offered rate, so **TTFT is comparable too** — which is the reason
`profiles/calibration/openloop/a40.accuracy.openloop.yaml` prefers this route for
a calibration point. One caveat carries over from that file and is measured, not
guessed: over HTTP the client's TTFT includes request serialisation and the SSE
first-chunk path, **+20.26 ms** on the A40 at its load, and the values here are
raw with nothing subtracted.

**Pairing, and the guard that decides whether a point exists.** The same offered
rate goes to both sides (`--match offered`, the committed convention). Served
concurrency is then an OUTCOME on both, and if the two land far apart the
simulator's latency is a different operating point's latency — the failure
`rngd_card_edf.yaml` records above c29.3, where the sim settled at 37.67 and
44.47 against measured 59.2 and 107.2. A point whose gap exceeds
`--max-conc-gap` is **flagged and excluded**, never quietly used.

**The x axis is the SIMULATOR's served concurrency**, because that is what the
planner knows when it consults the table (`planner/util/operating_point.py`).

**Two files out, and `compared_metric` says which is which** (user decision,
2026-09-18). The loaded file is fitted **p99 against p99**, because that is what
the margin is applied to (D101) and what S7.4's verdicts are; the `.p50.yaml`
sibling is fitted p50 against p50 and exists only so this measurement can be
compared with `rngd_card_edf.yaml`, which is p50
(`lowload_sim_error.py` records `compared_metric: tpot_p50`). Migrating the two
committed domains onto p99 is a separate step and is NOT done here.

**p99 is far more sample-sensitive than p50.** A 300-request run puts about three
observations above the p99, so `--min-requests-p99` refuses to fit one on less,
and such a point is emitted with `tpot_err_pct` **empty** and only
`tpot_err_pct_p50` filled rather than being dropped. The empirical check on
stability is the run-to-run spread across repeats, which is recorded per point
and which S7.0 requires beside any verdict. A5(b)'s `pool >= 4x` does not
transfer — there is no pool in an open loop — and its purpose is served by the
saturation slope the harness already records.

Run::

    experiments/scripts/openloop_sim_error.py \\
        --real outputs/s73_map/envelope.json outputs/s73_reps/envelope.json \\
        --sim experiments/p2_evidence/results/v3r_candidates.json \\
        --candidate s128-t2048 \\
        --out-dir outputs/s73
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from planner.util import provenance as prov  # noqa: E402

#: `e = (sim - measured) / measured`, the sign convention every domain file
#: declares. Negative TPOT = the simulator is optimistic, the direction that
#: earns a margin and the direction that produced the D22 retraction.
STATS = ("p50", "p99")


def real_points(paths: list[Path]) -> dict[float, list[dict]]:
    """Every measured open-loop point, grouped by offered rate.

    Repeats may arrive in one `envelope.json` (`--repeats N`) or across several
    runs, so this reads a list and groups rather than trusting one file to hold
    them all.
    """
    out: dict[float, list[dict]] = {}
    for path in paths:
        doc = json.loads(path.read_text())
        run = doc.get("run", {})
        if run.get("protocol") != "open_loop":
            raise SystemExit(
                f"{path}: protocol is {run.get('protocol')!r}, not 'open_loop'. "
                "A closed-loop envelope cannot be paired here -- that is D19, and "
                "lowload_sim_error.py is the script for it."
            )
        for point in doc.get("points", []):
            point["_source"] = str(path)
            if point.get("output_tokens") is None:
                point["output_tokens"] = _tokens_from_replay(
                    path.parent, float(point["offered_rps"]))
            out.setdefault(float(point["offered_rps"]), []).append(point)
    return out


def _tokens_from_replay(out_dir: Path, offered_rps: float) -> int | None:
    """Recover the delivered token count from the replay report beside the point.

    `measure_envelope.py` only began carrying `output_tokens` on the point when
    the token invariant was added, so an envelope written before that has the
    number in its sibling `replay_*.json` and not in the point. Matched on the
    offered rate rather than on a reconstructed filename, so this does not have to
    know how the harness spells its tags.
    """
    for candidate in sorted(out_dir.glob("replay_*.json")):
        try:
            doc = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        summary = doc.get("summary") or {}
        if abs(float(summary.get("offered_rps", -1)) - offered_rps) < 1e-9:
            return summary.get("output_tokens")
    return None


def expected_output_tokens(dataset: Path, num_reqs: int) -> int:
    """What the trace asks for over the rows a point replayed.

    `profiles/calibration/openloop/a40.accuracy.openloop.yaml` states the
    invariant this checks: *"The simulator generates exactly the trace's
    output_toks"*, so with `--ignore-eos` a real run must deliver them too, and
    "every stage delivered 195 753 output tokens, the trace's exact total".

    It is checked rather than trusted because it silently failed. The replay
    client caps completions at `min(output_toks, --max-tokens-cap)` and that cap
    defaulted to 512 against a trace whose p50 is 632 and whose max is 1021, so
    the first S7.3 runs delivered 78.6 % of the tokens and the shortfall read as
    the simulator over-predicting served concurrency by 18-37 % (D115).
    """
    total = 0
    for i, line in enumerate(dataset.read_text().splitlines()):
        if num_reqs and i >= num_reqs:
            break
        if line.strip():
            total += int(json.loads(line)["output_toks"])
    return total


def sim_points(paths: list[Path], candidate: str,
               cache_dir: Path | None = None) -> dict[float, dict]:
    """The simulated side, keyed by offered rate, for one candidate's knobs.

    The S7.0 record carries p99 only. The envelope cache that produced it carries
    every percentile, so `cache_dir` enriches each row with the SAME simulation's
    p50 -- matched on `served_concurrency`, which is the simulation's own output
    and is therefore an exact key, not a reconstructed one. Without it the p50
    file has nothing to fit and says so rather than deriving a p50 from a p99,
    which would be inventing a number (absolute rule 3).
    """
    out: dict[float, dict] = {}
    for path in paths:
        doc = json.loads(path.read_text())
        for row in doc.get("rows", []):
            if not row["candidate_id"].endswith(candidate):
                continue
            rate = float(row["rps"])
            if rate in out and out[rate]["served_concurrency"] != row["served_concurrency"]:
                raise SystemExit(
                    f"rps {rate:g} appears in more than one --sim record with "
                    f"different served concurrency "
                    f"({out[rate]['served_concurrency']} vs "
                    f"{row['served_concurrency']}). They are different "
                    f"simulations and silently preferring one would hide it."
                )
            out[rate] = row
    if not out:
        raise SystemExit(
            f"no rows for a candidate ending {candidate!r} in "
            f"{', '.join(str(p) for p in paths)}")
    if cache_dir:
        by_conc = {}
        for entry_path in sorted(cache_dir.glob("*.json")):
            try:
                entry = json.loads(entry_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not str(entry.get("candidate_id", "")).endswith(candidate):
                continue
            metrics = entry.get("metrics") or {}
            if metrics.get("served_concurrency") is not None:
                by_conc[metrics["served_concurrency"]] = metrics
        for row in out.values():
            metrics = by_conc.get(row["served_concurrency"])
            if metrics:
                row["p50_tpot_ms"] = metrics.get("p50_tpot_ms")
                row["p50_ttft_ms"] = metrics.get("p50_ttft_ms")
    return out


def _agg(values: list[float]) -> dict:
    """Median across repeats, with the spread that says whether it is stable.

    The median rather than the mean, because a single slow repeat should move the
    point as little as it moves a percentile. The spread is what S7.0 requires
    beside a verdict: a separation narrower than the spread is inside the noise.
    """
    values = sorted(values)
    return {
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "spread": values[-1] - values[0],
        "repeats": len(values),
    }


def pair(rps: float, reals: list[dict], sim: dict, *, max_gap: float,
         min_requests_p99: int, expected_tokens: int = 0,
         token_tol: float = 0.01) -> dict:
    """One candidate domain point, or a record of why there is not one."""
    notes: list[str] = []
    conc_real = _agg([r["served_concurrency"] for r in reals])
    conc_sim = sim["served_concurrency"]
    gap = (conc_sim - conc_real["median"]) / conc_real["median"]

    rec: dict = {
        "offered_rps": rps,
        "conc_measured": conc_real,
        "conc_sim": conc_sim,
        # The x axis the DOMAIN is keyed on, which is not always `conc_sim`.
        # `rngd_card_edf.yaml` states the convention: "the operating point AS
        # COMPUTED FROM THE SIMULATION (planner/util/operating_point.py),
        # because that is what the planner knows when it consults this table".
        # The margin policy looks up the per-island operating point, and that
        # differs from the run-level served concurrency by up to 1 ULP --
        # enough, at a boundary point, for the very candidate a domain's edge
        # was measured at to fall outside it. Keyed on the wrong one, the
        # domain's top and bottom points are unreachable.
        "conc_axis": sim.get("operating_point") or conc_sim,
        "conc_gap_pct": gap * 100.0,
        "requests_ok": _agg([float(r["requests_ok"]) for r in reals]),
        "saturated_any": any(r["saturated"] for r in reals),
        "numa_bind": sorted({r.get("numa_bind") for r in reals}),
        "sources": sorted({r["_source"] for r in reals}),
        "usable": True,
    }

    if expected_tokens and any(r.get("output_tokens") is None for r in reals):
        notes.append(
            "delivered output tokens are not recorded on every repeat, so the "
            "workload-match invariant COULD NOT BE CHECKED here. That is not a "
            "pass: it is the check that catches a truncated completion cap "
            "(D115), and an artifact predating it cannot answer for itself"
        )
    elif expected_tokens:
        # One extra chunk per request is the SSE role chunk, not a token, so it is
        # subtracted before the ratio rather than widening the tolerance.
        delivered = [float(r["output_tokens"]) for r in reals]
        ok_counts = [float(r["requests_ok"]) for r in reals]
        ratios = [(d - n) / expected_tokens
                  for d, n in zip(delivered, ok_counts, strict=True)
                  if expected_tokens]
        rec["delivered_token_ratio"] = _agg(ratios) if ratios else None
        worst = min(ratios, default=1.0)
        if ratios and not (1.0 - token_tol <= min(ratios) <= 1.0 + token_tol):
            rec["usable"] = False
            notes.append(
                f"delivered {worst * 100:.1f}% of the trace's output tokens "
                f"(tolerance +/-{token_tol * 100:.0f}%): the card and the "
                f"simulator did not run the same workload, so this is not a "
                f"calibration pair. A completion cap below the trace's longest "
                f"row is the way this happens -- D115"
            )

    if abs(gap) > max_gap:
        rec["usable"] = False
        notes.append(
            f"served concurrency differs by {gap * 100:+.1f}% "
            f"(sim {conc_sim:.3f} vs measured {conc_real['median']:.3f}, guard "
            f"+/-{max_gap * 100:.0f}%): the simulator's latency here is a "
            f"DIFFERENT operating point's latency, so this pair is not a point"
        )
    if rec["saturated_any"]:
        notes.append("at least one repeat grew its queue (saturated): its served "
                     "concurrency is a backlog depth, not the offered load")

    for stat in STATS:
        measured = _agg([r["tpot_ms"][stat] for r in reals])
        simulated = (sim["p99_tpot_ms"] if stat == "p99"
                     else sim.get("p50_tpot_ms"))
        rec[f"tpot_measured_{stat}"] = measured
        if simulated is None:
            # The sim record from S7.0 carries p99 only. A p50 comparison needs a
            # p50 from the same simulation, and inventing one from the p99 is not
            # an option (absolute rule 3).
            rec[f"tpot_sim_{stat}"] = None
            rec[f"tpot_err_pct_{stat}"] = None
            notes.append(f"no simulated tpot {stat}: the sim record carries p99 "
                         f"only and no --sim-cache was given, so the {stat} "
                         f"error cannot be formed")
            continue
        rec[f"tpot_sim_{stat}"] = simulated
        rec[f"tpot_err_pct_{stat}"] = (
            (simulated - measured["median"]) / measured["median"] * 100.0)

    ttft_measured = _agg([r["ttft_ms"]["p99"] for r in reals])
    rec["ttft_measured_p99"] = ttft_measured
    rec["ttft_sim_p99"] = sim["p99_ttft_ms"]
    rec["ttft_err_pct_p99"] = (
        (sim["p99_ttft_ms"] - ttft_measured["median"]) / ttft_measured["median"]
        * 100.0)
    notes.append("ttft_err_pct is RAW client-side: it includes request "
                 "serialisation and the SSE first-chunk path, measured at "
                 "+20.26 ms on the A40 (a40.accuracy.openloop.yaml). Nothing is "
                 "subtracted here")

    if rec["requests_ok"]["min"] < min_requests_p99:
        rec["p99_undersampled"] = True
        notes.append(
            f"a repeat completed {rec['requests_ok']['min']:.0f} requests, below "
            f"--min-requests-p99 {min_requests_p99}: a p99 over that few "
            f"observations is not a fit, so the p99 domain leaves tpot_err_pct "
            f"empty for this point and only the p50 file carries it"
        )
    else:
        rec["p99_undersampled"] = False

    rec["notes"] = notes
    return rec


DOMAIN_HEADER = """\
# RNGD-CARD accuracy domain -- OPEN-LOOP points, measured {date} on the NPU node.
#
# WORK_ORDER_domain_scoping.md STEP S7.3; docs/deviations.md D115.
#
# A SEPARATE FILE, per rule A3 and for the reason a40.accuracy.yaml's header
# gives: rngd_card_edf.yaml is CLOSED-LOOP (D113 states it on the file itself),
# `python -m planner plan` always replays an arrival trace, and two protocols on
# one interpolation axis is the class of error D22 was. Nothing there is touched.
#
# WHY IT EXISTS. Under the default `condition_mismatch: refuse`, every RNGD
# candidate is held against the closed-loop domain before a margin is ever
# computed -- 72 of 72 rows in S7.0's sweep, all on `arrival_process`. A wider
# load range does not answer them; a measurement under the arrival process the
# planner performs does.
#
# COMPARED METRIC: {compared}. {compared_why}
#
# Sign: (sim - measured) / measured. Negative = the simulator is optimistic,
# the direction that earns a margin.
#
# TTFT errors are RAW CLIENT-SIDE. Over HTTP the client's TTFT includes request
# serialisation and the SSE first-chunk path -- measured at +20.26 ms on the A40
# (openloop/a40.accuracy.openloop.yaml) -- and nothing is subtracted here.
#
# THE WORKLOAD-MATCH INVARIANT IS CHECKED, not assumed. With --ignore-eos the
# card must deliver the trace's own output_toks, because the simulator generates
# exactly those. The first S7.3 runs did not: replay_to_endpoint caps completions
# at min(output_toks, --max-tokens-cap) and that cap defaulted to 512 against a
# trace whose p50 is 632, so the card did 78.6 % of the work and the shortfall
# read as an 18-37 % concurrency error. Every point below delivered the trace's
# tokens to within 0.01 %; the cap is now resolved from the trace (D115).
"""


def _domain_doc(records: list[dict], *, stat: str, date: str) -> str:
    """One domain file, for one percentile basis."""
    import io

    usable = [r for r in records if r["usable"]
              and r.get(f"tpot_err_pct_{stat}") is not None]
    if not usable:
        raise SystemExit(f"no usable {stat} points: nothing to write")
    compared = f"tpot_{stat}"
    why = ("This is the basis the margin is APPLIED on (D101) and the basis "
           "S7.4's verdicts are read at, so the file is fitted where it is "
           "used. Its `.p50.yaml` sibling carries the same measurements on the "
           "p50 basis, for comparison with rngd_card_edf.yaml, which is p50."
           if stat == "p99" else
           "COMPARISON MATERIAL ONLY. The loaded domain is the p99 file beside "
           "this one; this exists so these measurements can be compared with "
           "rngd_card_edf.yaml, which `lowload_sim_error.py` fitted on p50. Do "
           "not point the planner at this file: the margin is applied to a p99.")
    buf = io.StringIO()
    buf.write(DOMAIN_HEADER.format(date=date, compared=compared,
                                   compared_why=why))
    buf.write("hardware:\n  RNGD-CARD:\n    hardware: RNGD-CARD\n")
    buf.write("    accuracy_domain:\n")
    # The lowest measured point, which is what `planner fit-accuracy-domain`
    # records when the flag is not given ("recording the lowest measured point
    # ... as where the profile was validated"). It is metadata -- neither the
    # optimizer nor the uncertainty stack reads it at judgement time -- so
    # matching that command's convention beats inventing a second one.
    buf.write(f"      fitted_at_concurrency: {usable[0]['conc_axis']!r}\n")
    span = usable[-1]["conc_axis"] / usable[0]["conc_axis"]
    buf.write(f"      # `refuse` is the D33 default for a new domain, and it is\n"
              f"      # kept: these points span a {span:.2g}x concurrency range\n"
              f"      # and say nothing about what happens outside it.\n")
    buf.write("      outside_domain: refuse\n")
    buf.write("      source: measured\n")
    buf.write("      arrival_process: open_loop\n")
    buf.write(f"      compared_metric: {compared}\n")
    buf.write("      # The card as one accelerator: one card is tp=1 on one\n"
              "      # island, the same convention rngd_card_edf.yaml states.\n")
    buf.write("      parallelism: {tp: 1, pp: 1, dp: 1}\n")
    buf.write("      placement:\n        islands: 1\n")
    buf.write("        # MEASURED BOUND, unlike every earlier RNGD artifact:\n"
              "        # the harness resolves the card's NUMA node through the\n"
              "        # BDF and binds server and client to it (default `auto`).\n")
    buf.write("        device_binding: numa_pinned\n")
    buf.write("      points:\n")
    for r in usable:
        err = r[f"tpot_err_pct_{stat}"]
        other = r["tpot_err_pct_p50"] if stat == "p99" else r["tpot_err_pct_p99"]
        meas = r[f"tpot_measured_{stat}"]
        # `conc` is the x axis and is written at FULL PRECISION, not rounded.
        # At .3f the top point 37.96546697025815 became 37.965, and the very
        # candidate that point was measured at then fell OUTSIDE its own domain
        # by 0.00047 and was refused. A coordinate is not a display value.
        line = (f"        - {{conc: {r['conc_axis']!r}, "
                f"tpot_err_pct: {err:.2f}, ")
        if stat == "p99" and other is not None:
            line += f"tpot_err_pct_p50: {other:.2f}, "
        line += f"ttft_err_pct: {r['ttft_err_pct_p99']:.2f},\n"
        buf.write(line)
        spread = (f"p99 TPOT spread over the {meas['repeats']} runs "
                  f"{meas['spread']:.4f} ms" if meas["repeats"] > 1
                  else "single run, so no run-to-run spread is known here")
        note = (f"{r['offered_rps']:g} rps, {int(r['requests_ok']['median'])} req, "
                f"{meas['repeats']} run{'s' if meas['repeats'] > 1 else ''}. "
                f"measured L {r['conc_measured']['median']:.2f} "
                f"(gap {r['conc_gap_pct']:+.1f}%). {spread}")
        buf.write(f'           note: "{note}"}}\n')
    buf.write("      note: >\n")
    buf.write(f"        {len(usable)} open-loop points from a deployed furiosa-llm\n"
              f"        server, served concurrency {usable[0]['conc_axis']:.4g} to\n"
              f"        {usable[-1]['conc_axis']:.4g} as the SIMULATOR computes it\n"
              f"        (the committed files' x-axis convention). Fitted on\n"
              f"        {compared}. The upper bound is NOT where measuring\n"
              f"        stopped: see provenance.unpaired_points, which records\n"
              f"        every rate measured above it and why none of them is a\n"
              f"        point.\n")

    unpaired = [r for r in records if not r["usable"]]
    buf.write("provenance:\n")
    buf.write("  harness: experiments/scripts/measure_envelope.py --mode open\n")
    buf.write("  driver: experiments/scripts/replay_to_endpoint.py --open-loop "
              "--ignore-eos\n")
    buf.write("  pairing: experiments/scripts/openloop_sim_error.py, --match offered\n")
    if unpaired:
        buf.write("  # MEASURED AND KEPT, not discarded. Each of these is a real\n"
                  "  # measurement of the card at a real offered rate; what it is\n"
                  "  # NOT is a calibration point, because the simulator sat at a\n"
                  "  # different served concurrency, so its p99 is another\n"
                  "  # operating point's p99. They are the evidence for where the\n"
                  "  # domain stops, which is otherwise an assertion. Keyed by\n"
                  "  # OFFERED rps -- the one axis both sides share up here.\n")
        buf.write("  unpaired_points:\n")
        for r in unpaired:
            meas = r["tpot_measured_p99"]
            under = ((r["tpot_sim_p99"] - meas["median"]) / meas["median"] * 100.0)
            buf.write(f"    - offered_rps: {r['offered_rps']:g}\n")
            buf.write("      pairing: unpaired_served_l\n")
            buf.write(f"      served_l_real: {r['conc_measured']['median']:.3f}\n")
            buf.write(f"      served_l_sim: {r['conc_sim']:.3f}\n")
            buf.write(f"      l_gap_pct: {r['conc_gap_pct']:.2f}\n")
            buf.write(f"      tpot_p99_measured_ms: {meas['median']:.3f}\n")
            buf.write(f"      tpot_p99_sim_ms: {r['tpot_sim_p99']:.3f}\n")
            buf.write(f"      tpot_p99_under_prediction_pct: {under:.2f}\n")
            buf.write(f"      saturated: {str(r['saturated_any']).lower()}\n")
            buf.write(f"      runs: {meas['repeats']}\n")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, nargs="+", required=True,
                    help="envelope.json files from measure_envelope.py --mode open")
    ap.add_argument("--sim", type=Path, nargs="+", required=True,
                    help="the simulated side, e.g. v3r_candidates.json. Several "
                         "are merged by rate; a rate present twice with "
                         "different results is an error, not a preference.")
    ap.add_argument("--sim-cache", type=Path, default=None,
                    help="the envelope cache behind --sim. It carries every "
                         "percentile, so this is what lets the p50 file be "
                         "fitted from the SAME simulations as the p99 one.")
    ap.add_argument("--candidate", default="s128-t2048",
                    help="which knob setting to pair against. At these rates "
                         "s128 and s256 are identical in the simulator and the "
                         "real server sets no max_num_seqs, so the choice is "
                         "immaterial below ~4 rps and must be stated above it.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--write-domain", type=Path, default=None,
                    help="emit the p99 domain here and its p50 sibling beside it")
    ap.add_argument("--max-conc-gap", type=float, default=0.20,
                    help="pairing guard, as a fraction (default 0.20)")
    ap.add_argument("--min-requests-p99", type=int, default=300,
                    help="fewest completed requests a p99 fit is allowed")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="the trace both sides replayed. Given, the delivered "
                         "output tokens are checked against it -- the invariant "
                         "a40.accuracy.openloop.yaml states and that a completion "
                         "cap silently broke (D115).")
    ap.add_argument("--num-reqs", type=int, default=300,
                    help="rows of --dataset a point replayed")
    ap.add_argument("--token-tolerance", type=float, default=0.01,
                    help="how far the delivered token count may sit from the "
                         "trace's (default 1 %%)")
    args = ap.parse_args()

    reals = real_points(args.real)
    sims = sim_points(list(args.sim), args.candidate, args.sim_cache)
    expected = (expected_output_tokens(args.dataset, args.num_reqs)
                if args.dataset else 0)
    if expected:
        print(f"trace asks for {expected} output tokens over {args.num_reqs} rows",
              file=sys.stderr)

    records = []
    for rps in sorted(reals):
        if rps not in sims:
            print(f"rps {rps:g}: measured but not simulated; skipped",
                  file=sys.stderr)
            continue
        records.append(pair(rps, reals[rps], sims[rps],
                            max_gap=args.max_conc_gap,
                            min_requests_p99=args.min_requests_p99,
                            expected_tokens=expected,
                            token_tol=args.token_tolerance))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "openloop_sim_error.json"
    out_json.write_text(json.dumps(
        {"sim": [str(s) for s in args.sim], "candidate": args.candidate,
         "real": [str(p) for p in args.real],
         "max_conc_gap": args.max_conc_gap,
         "min_requests_p99": args.min_requests_p99,
         # WHICH machine, not merely what kind (D80). Without it a pairing from a
         # second RNGD node is indistinguishable from this one, and these points
         # cannot be appended to a domain built on different silicon.
         "provenance": prov.collect(),
         "points": records}, indent=2) + "\n")

    print(f"{'rps':>5} {'L meas':>8} {'L sim':>8} {'gap%':>7} "
          f"{'tpot p99 meas':>13} {'sim':>8} {'err%':>8} {'usable':>7}")
    for r in records:
        e = r["tpot_err_pct_p99"]
        print(f"{r['offered_rps']:5g} {r['conc_measured']['median']:8.3f} "
              f"{r['conc_sim']:8.3f} {r['conc_gap_pct']:+7.2f} "
              f"{r['tpot_measured_p99']['median']:13.3f} "
              f"{r['tpot_sim_p99']:8.3f} "
              f"{(f'{e:+.3f}' if e is not None else 'n/a'):>8} "
              f"{r['usable']!s:>7}")
    print(f"\nwrote {out_json}")

    if args.write_domain:
        import datetime

        date = datetime.date.today().isoformat()
        args.write_domain.parent.mkdir(parents=True, exist_ok=True)
        args.write_domain.write_text(_domain_doc(records, stat="p99", date=date))
        print(f"wrote {args.write_domain}")
        p50_path = args.write_domain.with_suffix("")
        p50_path = p50_path.with_name(p50_path.name + ".p50.yaml")
        try:
            p50_path.write_text(_domain_doc(records, stat="p50", date=date))
            print(f"wrote {p50_path}")
        except SystemExit as exc:
            print(f"p50 sibling NOT written: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
