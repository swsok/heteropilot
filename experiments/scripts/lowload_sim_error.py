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
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from planner.perf_envelope import load_envelope  # noqa: E402
from planner.util.percentile import percentile  # noqa: E402

ENV = ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"
DATASET = ROOT / "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl"
CLUSTER = ROOT / "experiments/configs/clusters/rngd-card-llama31-8b-tp1.json"


def write_trace(out: Path, rps: float, n: int) -> Path:
    """The dataset's first `n` requests, re-spaced at a constant `rps`.

    Constant spacing rather than Poisson: the envelope point is a steady-state
    average, and a deterministic arrival process reaches the same mean occupancy
    without adding a second source of variance to explain.
    """
    rows = []
    with DATASET.open() as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rec = json.loads(line)
            rec["arrival_time_ns"] = int(i / rps * 1e9)
            rows.append(rec)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def sim_metrics(csv: Path) -> dict:
    """Served concurrency (Little's law) and TPOT p50 from the simulator CSV."""
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
        "wall_ns": wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/lowload_sim_error")
    ap.add_argument("--num-reqs", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--max-conc", type=float, default=20.0,
                    help="only the envelope points at or below this concurrency")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = load_envelope(ENV)
    mean_out = env.measured_on_workload.output_tokens_mean
    assert mean_out is not None

    results = []
    for pt in env.points:
        if pt.conc > args.max_conc or pt.tpot_p50 is None:
            continue
        rps = pt.tput_tok_s / mean_out
        tag = f"c{pt.conc}".replace(".", "p")
        trace = write_trace(args.out / f"trace_{tag}.jsonl", rps, args.num_reqs)
        csv = args.out / f"sim_{tag}.csv"
        log = args.out / f"sim_{tag}.log"
        cmd = [
            "experiments/scripts/livelock_watch.sh", "-n", "45", "-g", "1800",
            "-s", "1800", "-t", str(int(args.timeout)), "--",
            ".venv/bin/python", "-m", "serving",
            "--cluster-config", str(CLUSTER.relative_to(ROOT)),
            "--dataset", str(trace.relative_to(ROOT)),
            "--output", str(csv.relative_to(ROOT)), "--num-reqs", str(args.num_reqs),
            "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--block-size", "16",
            "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
            "--request-routing-policy", "LOAD", "--network-backend", "analytical",
            "--log-level", "WARNING", "--log-interval", "1.0",
            "--no-enable-prefix-caching", "--run-id", f"lowload-{tag}",
        ]
        print(f"  conc {pt.conc}: offering {rps:.4f} rps", file=sys.stderr)
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=fh,
                                timeout=args.timeout + 300).returncode
        if rc != 0 or not csv.exists():
            results.append({"conc": pt.conc, "rc": rc, "error": "run failed"})
            print(f"    exit {rc}", file=sys.stderr)
            continue
        m = sim_metrics(csv)
        err = (m["tpot_p50_ms"] - pt.tpot_p50) / pt.tpot_p50 * 100
        conc_gap = (m["served_conc"] - pt.conc) / pt.conc * 100
        results.append({
            "conc_measured": pt.conc, "conc_sim": m["served_conc"],
            "conc_gap_pct": conc_gap, "offered_rps": rps,
            "tpot_measured_ms": pt.tpot_p50, "tpot_sim_ms": m["tpot_p50_ms"],
            "tpot_err_pct": err, "requests": m["n"], "rc": rc,
            "comparable": abs(conc_gap) <= 20.0,
        })
        print(f"    sim served {m['served_conc']:.2f} (gap {conc_gap:+.1f}%)  "
              f"tpot {m['tpot_p50_ms']:.2f} vs {pt.tpot_p50:.2f}  err {err:+.2f}%",
              file=sys.stderr)

    (args.out / "lowload_sim_error.json").write_text(
        json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out/'lowload_sim_error.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
