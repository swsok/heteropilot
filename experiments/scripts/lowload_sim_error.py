"""How wrong is the simulator at LOW load? (WORK_ORDER_rps_aware.md STEP 4.2)

The RNGD accuracy domain is anchored at served concurrency 16.6 and 76. E6 is
about the other end -- 1 to 16 -- so every candidate it ranks would otherwise get
an EXTRAPOLATED margin, from a `widen_error_bars` policy whose whole justification
is that it is a stopgap. STEP 3 measured the hardware there; this measures the
simulator at the same operating points so the domain has data instead.

Method, and the trap it steps around. The bench is closed-loop (a fixed number in
flight); the simulator replays an open-loop trace. Comparing them directly is D19 --
a burst against a spread arrival process, with the whole difference landing in
TTFT. So:

  * the arrival rate is set to the one the envelope records for each measured
    point, so the two runs sit at the same OFFERED load;
  * the sim's own SERVED concurrency is computed from its CSV by Little's law and
    reported. If it lands away from the measured point, the comparison is not
    apples to apples and the point is flagged rather than used;
  * only TPOT is compared. TTFT is closed-loop on the bench side and the envelope
    says not to compare it.

**Not RNGD-only since 2026-09-10** (`docs/HANDOVER.md` §2.2). The envelope, the
cluster config and the dataset were module constants, which pinned the script to
one device for no reason other than that it was written for one. They are
arguments now, defaulting to the RNGD-CARD trio so every committed invocation
still runs verbatim. The A40 materials are:

    experiments/scripts/lowload_sim_error.py \
        --envelope profiles/envelopes/A40/meta-llama/Llama-3.1-8B/bf16/tp1.yaml \
        --cluster experiments/configs/clusters/a40-llama31-8b-tp1.json \
        --out outputs/a40_lowload_sim_error

The `--dataset` must be the one the envelope was measured on -- the offered rate
is derived from the envelope's own `output_tokens_mean`, so a different token
distribution silently changes what "the same load" means. The script refuses a
mismatch rather than reporting one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.perf_envelope import EnvelopeError, load_envelope  # noqa: E402
from planner.util.percentile import percentile  # noqa: E402

#: Defaults, not constants. Keeping them here means the RNGD invocation in
#: `docs/rps_step*` runs with no arguments exactly as it did.
DEFAULT_ENV = ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"
DEFAULT_DATASET = ROOT / "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl"
DEFAULT_CLUSTER = ROOT / "experiments/configs/clusters/rngd-card-llama31-8b-tp1.json"


def _arg_path(p: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    The simulator is launched with `cwd=ROOT` and was handed
    `Path.relative_to(ROOT)`, which RAISES for any path outside the tree -- so
    `--out /tmp/...` crashed in the argument list rather than in the run. Both
    forms resolve to the same file from ROOT; the relative one is kept because it
    is what the committed command lines show.
    """
    try:
        # resolve() FIRST: a relative argument is not `relative_to` anything, so
        # comparing it raw sent every repo-local path down the absolute branch --
        # and `python -m serving` prepends `../` to what it is given, so an
        # absolute path became `..//home/...` and the run died at startup.
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def write_trace(out: Path, dataset: Path, rps: float, n: int) -> Path:
    """The dataset's first `n` requests, re-spaced at a constant `rps`.

    Constant spacing rather than Poisson: the envelope point is a steady-state
    average, and a deterministic arrival process reaches the same mean occupancy
    without adding a second source of variance to explain.
    """
    rows = []
    with dataset.open() as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rec = json.loads(line)
            rec["arrival_time_ns"] = int(i / rps * 1e9)
            rows.append(rec)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def score(env, raw: dict, match: str, stat: str = "p50") -> dict:
    """Pair one simulated run with the measured curve and score its TPOT error.

    There are two defensible pairings and which one is right depends on how
    wrong the model is.

    `offered` compares against the envelope point whose arrival rate was
    offered, so both sides sit at the same OFFERED load, and the +/-20 % guard
    then checks that they also landed at the same SERVED one. If they did not,
    the simulator's TPOT belongs to a different operating point and the pair is
    refused rather than reported. The RNGD-CARD and A40 domains were built this
    way and the guard passed there (gaps -0.8 % to -19 %).

    `served` compares against the measured curve INTERPOLATED at the simulator's
    own served concurrency. It is needed when the model is wrong enough that
    matching the offered rate cannot also match the operating point: the per-PE
    RNGD fixture sits at served 1.86 where the hardware sits at 1.00, so
    `offered` refuses the point and measures nothing. This pairing also asks the
    question the accuracy domain is indexed by -- `planner/util/operating_point.py`
    keys it on the concurrency the SIMULATION reports, not on the one the
    hardware would have reached -- and it charges the model for its TPOT error
    alone instead of summing that with its throughput error.

    Either way the x axis of the resulting domain point is `conc_sim`.
    """
    conc_gap = ((raw["conc_sim"] - raw["conc_measured"])
                / raw["conc_measured"] * 100)
    out = {
        "conc_measured": raw["conc_measured"], "conc_sim": raw["conc_sim"],
        "conc_gap_pct": conc_gap, "offered_rps": raw["offered_rps"],
        "tpot_measured_ms": raw["tpot_measured_ms"],
        "tpot_sim_ms": raw["tpot_sim_ms"],
    }
    refused = None
    if match == "offered":
        ref: float | None = raw["tpot_measured_ms"]
        comparable = abs(conc_gap) <= 20.0
    else:
        try:
            ref = env.metric_at(raw["conc_sim"], f"tpot_{stat}")
        except EnvelopeError as exc:
            # `extrapolation: refuse` reaching up through the envelope. A sim
            # operating point outside the measured range has no reference TPOT
            # and inventing one is what the policy exists to prevent.
            ref, refused = None, str(exc)
        if ref is None and refused is None:
            refused = (f"tpot_{stat} is unmeasured at an end of the bracketing "
                       f"interval")
        comparable = ref is not None
    out["tpot_err_pct"] = (None if not ref
                           else (raw["tpot_sim_ms"] - ref) / ref * 100)
    out["requests"] = raw["requests"]
    out["rc"] = raw["rc"]
    out["comparable"] = comparable
    if match != "offered":
        out["tpot_ref_ms"] = ref
        out["match"] = match
        if refused:
            out["refused"] = refused
    return out


def sim_metrics(csv: Path) -> dict:
    """Served concurrency (Little's law) and TPOT percentiles from the sim CSV.

    Both percentiles are computed whichever one `--compare-stat` asks for, so the
    artifact records what was available rather than only what was used.
    """
    lines = csv.read_text().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    i_lat, i_tpot = header.index("latency"), header.index("TPOT")
    i_arr, i_end = header.index("arrival"), header.index("end_time")
    lat, tpot, arr, end = [], [], [], []
    for line in lines[1:]:
        c = line.split(",")
        try:
            lat.append(float(c[i_lat]))
            tpot.append(float(c[i_tpot]) / 1e6)
            arr.append(float(c[i_arr]))
            end.append(float(c[i_end]))
        except (ValueError, IndexError):
            continue
    wall = max(end) - min(arr)
    return {
        "n": len(lat),
        "served_conc": sum(lat) / wall if wall else 0.0,
        "tpot_p50_ms": percentile(tpot, 50) if tpot else None,
        "tpot_p99_ms": percentile(tpot, 99) if tpot else None,
        "wall_ns": wall,
    }


def _write(args, results: list[dict]) -> None:
    (args.out / "lowload_sim_error.json").write_text(
        json.dumps(results, indent=2) + "\n")
    # Beside it, not inside it: `e6b_measured_curve.py` reads the file above as a
    # bare list, and now that the inputs vary the artifact has to say which device
    # it describes -- an A40 file and an RNGD one are otherwise indistinguishable.
    run = {
        "envelope": _arg_path(args.envelope),
        "cluster": _arg_path(args.cluster),
        "dataset": _arg_path(args.dataset),
        "num_reqs": args.num_reqs,
        "max_conc": args.max_conc,
        **({"min_conc": args.min_conc} if args.min_conc else {}),
        "run_prefix": args.run_prefix,
        "compared_metric": ("tpot_p50" if args.compare_stat == "p50"
                            else f"tpot_{args.compare_stat}"),
        "note": "TPOT only. The bench side is closed-loop and the simulator "
                "replays an arrival process, so TTFT is not comparable (D19).",
    }
    # Only when it is not the default, so a re-run of a committed invocation
    # writes the same run.json it wrote before.
    if args.match != "offered":
        run["match"] = args.match
        run["match_note"] = (
            "tpot_err_pct is against the measured curve interpolated at the "
            "SIMULATOR's served concurrency, not against the envelope point whose "
            "rate was offered. See score() in this script."
        )
    if args.from_raw:
        run["rescored_from"] = _arg_path(args.from_raw)
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"\nwrote {args.out/'lowload_sim_error.json'}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envelope", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--cluster", type=Path, default=DEFAULT_CLUSTER)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/lowload_sim_error")
    ap.add_argument("--num-reqs", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--max-conc", type=float, default=20.0,
                    help="only the envelope points at or below this concurrency")
    ap.add_argument("--min-conc", type=float, default=0.0,
                    help="and at or above this one. Extending a sweep upwards "
                         "costs a re-run of everything below it otherwise, which "
                         "on the per-PE fixture was 1 h 52 m of settled results.")
    ap.add_argument("--run-prefix", default="lowload",
                    help="--run-id prefix, so two devices' runs cannot collide in "
                         "one ASTRA-Sim input root")
    ap.add_argument("--compare-stat", choices=("p50", "p99", "both"),
                    default="p50",
                    help="which TPOT percentile to compare. Default `p50`, which "
                         "is what every committed domain point was fitted on, so "
                         "a committed invocation writes the same artifact it "
                         "wrote before. `p99` is what the margin is actually "
                         "applied to (D101) and what S7.3 onward fits on; `both` "
                         "emits a record per stat. Migrating the committed "
                         "domains onto p99 is a separate step and this flag is "
                         "what it will use.")
    ap.add_argument("--match", choices=("offered", "served"), default="offered",
                    help="how a simulated run is paired with the measured curve; "
                         "see score(). The default is what the committed RNGD and "
                         "A40 invocations were built with, so they still run "
                         "verbatim.")
    ap.add_argument("--sim-arg", action="append", default=[], metavar="ARG",
                    help="an extra argument appended to the `python -m serving` "
                         "command line, repeatable. Write it as `--sim-arg=VALUE`: "
                         "argparse refuses a separate value that itself starts "
                         "with a dash, which every flag being forwarded does. So "
                         "`--sim-arg=--log-level --sim-arg=INFO` raises the "
                         "verbosity and `--sim-arg=--no-enable-chunked-prefill` "
                         "changes the step policy. A later occurrence of a flag "
                         "wins over the block below. "
                         "Added for the NPU execution-model spike, whose STEP A.1 "
                         "needs the INFO batch log and whose A.2 re-runs the same "
                         "points under runtime-shaped scheduler knobs; without it "
                         "both meant a forked copy of this script. Omitting it "
                         "leaves every committed invocation byte-identical.")
    ap.add_argument("--from-raw", type=Path, default=None,
                    help="re-score a previous run's lowload_sim_error.json instead "
                         "of simulating. The pairing is arithmetic on facts the "
                         "artifact already records, so changing --match costs "
                         "nothing and re-running the simulator to get it would be "
                         "hours of compute for the same answer.")
    args = ap.parse_args()
    # Resolved once so the loop below reads a list, not a mode string.
    args._stats = ("p50", "p99") if args.compare_stat == "both" else (args.compare_stat,)
    args.out.mkdir(parents=True, exist_ok=True)

    env = load_envelope(args.envelope)
    mean_out = env.measured_on_workload.output_tokens_mean
    assert mean_out is not None

    # The offered rate is derived from the ENVELOPE's mean output length, so a
    # dataset other than the measured one would put the two sides at different
    # loads while every printed number still looked plausible. That is the D19
    # failure mode wearing a different hat, so it is refused rather than noted.
    measured_on = env.measured_on_workload.dataset
    if measured_on and Path(measured_on).name != args.dataset.name:
        raise SystemExit(
            f"--dataset {args.dataset.name} is not the workload this envelope was "
            f"measured on ({measured_on}). The offered rate is computed from that "
            f"workload's mean output length ({mean_out}), so the two sides would "
            f"sit at different loads. Pass the measured dataset."
        )

    if args.from_raw:
        prior = json.loads(args.from_raw.read_text())
        # A record without `conc_sim` is a failed run, kept verbatim: it has no
        # facts to re-score and dropping it would silently shorten the sweep.
        results = [score(env, r, args.match) if "conc_sim" in r else r
                   for r in prior]
        for r in results:
            if "conc_sim" not in r:
                continue
            err = r["tpot_err_pct"]
            print(f"  sim conc {r['conc_sim']:.2f}: tpot {r['tpot_sim_ms']:.2f} vs "
                  f"{r['tpot_ref_ms'] if r.get('tpot_ref_ms') else r['tpot_measured_ms']}"
                  f"  err {'refused' if err is None else f'{err:+.2f}%'}",
                  file=sys.stderr)
        _write(args, results)
        return 0

    results = []
    for pt in env.points:
        if not (args.min_conc <= pt.conc <= args.max_conc) or pt.tpot_p50 is None:
            continue
        rps = pt.tput_tok_s / mean_out
        tag = f"c{pt.conc}".replace(".", "p")
        trace = write_trace(args.out / f"trace_{tag}.jsonl", args.dataset,
                            rps, args.num_reqs)
        csv = args.out / f"sim_{tag}.csv"
        log = args.out / f"sim_{tag}.log"
        cmd = [
            "experiments/scripts/livelock_watch.sh", "-n", "45", "-g", "1800",
            "-s", "1800", "-t", str(int(args.timeout)), "--",
            ".venv/bin/python", "-m", "serving",
            "--cluster-config", _arg_path(args.cluster),
            "--dataset", _arg_path(trace),
            "--output", _arg_path(csv), "--num-reqs", str(args.num_reqs),
            "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--block-size", "16",
            "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
            "--request-routing-policy", "LOAD", "--network-backend", "analytical",
            "--log-level", "WARNING", "--log-interval", "1.0",
            "--no-enable-prefix-caching", "--run-id", f"{args.run_prefix}-{tag}",
        ]
        # Appended, not merged: argparse on the simulator's side lets the last
        # occurrence win, so a caller overrides any flag above by naming it again.
        cmd += args.sim_arg
        print(f"  conc {pt.conc}: offering {rps:.4f} rps", file=sys.stderr)
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=fh,
                                timeout=args.timeout + 300).returncode
        if rc != 0 or not csv.exists():
            results.append({"conc": pt.conc, "rc": rc, "error": "run failed"})
            print(f"    exit {rc}", file=sys.stderr)
            continue
        m = sim_metrics(csv)
        measured_of = {"p50": pt.tpot_p50, "p99": pt.tpot_p99}
        for stat in args._stats:
            rec = score(env, {
                "conc_measured": pt.conc, "conc_sim": m["served_conc"],
                "offered_rps": rps, "tpot_measured_ms": measured_of[stat],
                "tpot_sim_ms": m[f"tpot_{stat}_ms"], "requests": m["n"], "rc": rc,
            }, args.match, stat)
            # Only the extra stats are tagged, so a default `p50` run writes the
            # same record shape it always wrote.
            if args.compare_stat != "p50":
                rec["compared_metric"] = f"tpot_{stat}"
            results.append(rec)
            ref = rec.get("tpot_ref_ms") or measured_of[stat]
            err = rec["tpot_err_pct"]
            simv = m[f"tpot_{stat}_ms"]
            print(f"    sim served {m['served_conc']:.2f} "
                  f"(gap {rec['conc_gap_pct']:+.1f}%)  "
                  f"tpot {stat} {simv:.2f} vs "
                  f"{'n/a' if ref is None else f'{ref:.2f}'}  "
                  f"err {'refused' if err is None else f'{err:+.2f}%'}",
                  file=sys.stderr)

    _write(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
