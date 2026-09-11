"""Measure a performance envelope point-by-point: served load, latency, power.

`WORK_ORDER_rps_aware.md` rev 2 STEP 3.1, implementing
`docs/rps_aware_planning_design.md` §7. Vendor-agnostic core: the analysis knows
about bench reports and sampler CSVs, not about FuriosaAI.

This exists because of D22. That envelope's c32 point was labelled by the
concurrency *asked for*; a 24-request pool meant it actually ran at effective
concurrency 21.2, and an exponent fitted across it read a pool-capped 1.74x
interval as a doubling. So the A5 rules are enforced here rather than left to the
operator:

  (a) concurrency is always reported as SERVED -- Little's law, sum(latency)/wall
      -- with the requested value beside it and the ratio between them;
  (b) the pool must be >= 4x the concurrency and served/requested >= 0.9, and a
      point that misses is marked `pool_binding: true` rather than dropped;
  (c) power is a SUSTAINED mean over the bench window, recorded together with the
      utilisation from the same samples -- power without utilisation is not a
      measurement (A5(c));
  (d) idle is 45 s of settling discarded, then a 60 s mean;
  (e) every point runs the same workload.

Percentiles come from `planner/util/percentile.py` so this agrees with the
planner and the simulator rather than introducing a third interpolation.

    experiments/scripts/measure_envelope.py \
        --artifact <path> --dataset <sharegpt.jsonl> --card 0 \
        --concurrency 1,2,4,8,16 --out outputs/rngd_envelope_lowload

**Two backends since 2026-09-10** (`docs/HANDOVER.md` §2.2). The analysis half was
already vendor-agnostic; the execution half was hardcoded to FuriosaAI in exactly
three places -- how the server starts, which sampler runs, and which interpreter
drives the bench client -- and `--backend cuda` supplies the other value for each:

    experiments/scripts/measure_envelope.py --backend cuda \
        --artifact meta-llama/Llama-3.1-8B --dataset <sharegpt.jsonl> --card 0 \
        --concurrency 4,8,16 --out outputs/a40_envelope_lowload

Nothing else moves. The A5 rules live in `summarise_point`, which neither backend
can reach, so a CUDA point is held to the same pool floor, the same served/
requested ratio and the same power-with-utilisation requirement as an RNGD one --
which is the point, because the two curves are meant to be compared.

The bench client is shared as-is. `bench_furiosa_endpoint.py` is named for the
node it was written on but is a plain `AsyncOpenAI` client, so it drives a vLLM
OpenAI server unchanged. It is CLOSED-LOOP: note D19 before comparing anything it
produces against `python -m serving`, which replays an arrival process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from planner.util.percentile import percentile, summary  # noqa: E402

NS_PER_S = 1e9
SETTLE_S = 45.0
IDLE_WINDOW_S = 60.0
#: A5(b). Below this the point is reported but marked as pool-bound.
SERVED_RATIO_FLOOR = 0.9
#: A5(b). The pool must be at least this multiple of the requested concurrency.
POOL_MULTIPLE = 4
MIN_POOL = 300


# ---------------------------------------------------------------------------
# pure analysis -- no hardware, no subprocesses; this is what the tests exercise
# ---------------------------------------------------------------------------

@dataclass
class SamplerRow:
    ts: float
    device_sn: str
    power_w: float | None
    util_mean_pct: float | None
    util_max_pct: float | None
    dram_used_ratio: float | None
    dropped: bool = False


@dataclass
class Window:
    """A half-open [start, end) interval of wall-clock seconds."""
    start: float
    end: float

    def contains(self, ts: float) -> bool:
        return self.start <= ts < self.end


@dataclass
class PointResult:
    requested_concurrency: int
    served_concurrency: float
    served_ratio: float
    pool_binding: bool
    notes: list[str] = field(default_factory=list)


def _num(rec: dict, key: str) -> float | None:
    v = (rec.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def read_sampler_csv(path: Path) -> list[SamplerRow]:
    """Parse `power_sampler.sh` output. Comment lines carry provenance, not data."""
    rows: list[SamplerRow] = []
    with path.open() as fh:
        body = (line for line in fh if not line.startswith("#"))
        for rec in csv.DictReader(body):
            dropped = (rec.get("raw_info") or "") == "query failed"

            def num(key: str, _rec: dict = rec) -> float | None:
                return _num(_rec, key)

            ts = num("ts_info")
            if ts is None:
                continue
            rows.append(SamplerRow(
                ts=ts,
                device_sn=(rec.get("device_sn") or "").strip(),
                power_w=num("power_w"),
                util_mean_pct=num("util_mean_pct"),
                util_max_pct=num("util_max_pct"),
                dram_used_ratio=num("dram_used_ratio"),
                dropped=dropped,
            ))
    return rows


def served_concurrency(per_request: list[dict], wall_s: float) -> float:
    """Little's law. The number D22 should have reported and did not.

    Sums the latency of requests that COMPLETED: a failed request occupied the
    server for an unknown time, and counting its partial latency would inflate
    the answer in exactly the direction that flatters a pool-bound point.
    """
    if wall_s <= 0:
        return 0.0
    total_ns = sum(
        r["latency_ns"] for r in per_request
        if r.get("error") is None and r.get("latency_ns")
    )
    return (total_ns / NS_PER_S) / wall_s


def window_stats(rows: list[SamplerRow], window: Window,
                 device_sn: str | None = None) -> dict:
    """Sustained power AND utilisation over one window, from the same samples.

    A5(c): they are reported together or not at all, so a caller cannot quote a
    wattage without the load that produced it.
    """
    in_window = [r for r in rows if window.contains(r.ts)]
    # A dropped row carries no device_sn, so it is counted before the device
    # filter -- otherwise pinning to a serial would silently report zero holes.
    dropped = sum(1 for r in in_window if r.dropped)
    sel = [
        r for r in in_window
        if not r.dropped and (device_sn is None or r.device_sn == device_sn)
    ]
    usable = [r for r in sel if r.power_w is not None]
    if not usable:
        return {"samples": 0, "dropped_samples": dropped,
                "power_w": None, "util_pct": None,
                "note": "no usable samples in this window"}
    powers = [r.power_w for r in usable]
    utils = [r.util_mean_pct for r in usable if r.util_mean_pct is not None]
    util_max = [r.util_max_pct for r in usable if r.util_max_pct is not None]
    dram = [r.dram_used_ratio for r in usable if r.dram_used_ratio is not None]
    return {
        "samples": len(usable),
        "dropped_samples": dropped,
        "duration_s": round(window.end - window.start, 3),
        "power_w": {
            "mean": sum(powers) / len(powers),
            "p5": percentile(powers, 5),
            "p95": percentile(powers, 95),
            "min": min(powers),
            "max": max(powers),
        },
        # A5(c): never a power figure without the utilisation from the same samples.
        "util_pct": None if not utils else {
            "mean": sum(utils) / len(utils),
            "max_of_per_pe_max": max(util_max) if util_max else None,
        },
        "dram_used_ratio_mean": (sum(dram) / len(dram)) if dram else None,
    }


def summarise_point(bench: dict, rows: list[SamplerRow], bench_window: Window,
                    idle_window: Window | None = None,
                    device_sn: str | None = None,
                    pool_size: int | None = None) -> dict:
    """Everything one envelope point claims, assembled from measurements only."""
    per_request = bench.get("per_request", [])
    ok = [r for r in per_request if r.get("error") is None and r.get("ttft_ns")]
    wall_s = float(bench.get("wall_s") or 0.0)
    requested = int(bench.get("concurrency") or 0)
    served = served_concurrency(per_request, wall_s)
    ratio = (served / requested) if requested else 0.0

    notes: list[str] = []
    pool = pool_size if pool_size is not None else int(bench.get("requests") or 0)
    if requested and pool < POOL_MULTIPLE * requested:
        notes.append(
            f"pool {pool} < {POOL_MULTIPLE}x requested concurrency {requested} "
            f"(A5(b)); this is the D22 c32 failure mode"
        )
    if ratio < SERVED_RATIO_FLOOR:
        notes.append(
            f"served/requested {ratio:.3f} < {SERVED_RATIO_FLOOR}: the point did "
            f"not sustain the concurrency it asked for"
        )
    if bench.get("failed"):
        notes.append(f"{bench['failed']} request(s) failed and are excluded from latency")

    ttft_ms = [r["ttft_ns"] / 1e6 for r in ok]
    tpot_ms = [r["tpot_ns"] / 1e6 for r in ok if r.get("tpot_ns")]
    out_toks = sum(r.get("streamed_chunks", 0) for r in ok)

    return {
        "requested_concurrency": requested,
        "served_concurrency": served,
        "served_ratio": ratio,
        # A5(b): reported, never dropped -- a suppressed point is a curve with a
        # hole in it, and D22's exponent was fitted straight across one.
        "pool_binding": ratio < SERVED_RATIO_FLOOR,
        "pool_size": pool,
        "requests_ok": len(ok),
        "requests_failed": int(bench.get("failed") or 0),
        "wall_s": wall_s,
        "throughput_tok_s": (out_toks / wall_s) if wall_s else None,
        "ttft_ms": summary(ttft_ms) if ttft_ms else None,
        "tpot_ms": summary(tpot_ms) if tpot_ms else None,
        "bench_window": window_stats(rows, bench_window, device_sn),
        "idle_window": window_stats(rows, idle_window, device_sn) if idle_window else None,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# orchestration -- needs the hardware
# ---------------------------------------------------------------------------

#: The three places the execution half knew a vendor. Each entry is what
#: `run_point` needs and nothing more, so adding a third backend is a table row
#: rather than a branch in the orchestration.
BACKENDS = {
    "furiosa": {
        # On PATH: the FuriosaAI stack is a system install.
        "server_bin": "furiosa-llm",
        "sampler": "experiments/scripts/power_sampler.sh",
        # The vendor stack lives in the system interpreter; .venv has no `openai`
        # (verified 2026-09-08).
        "bench_python": "/usr/bin/python3",
    },
    "cuda": {
        # NOT on PATH, and that is the "one venv per vendor" rule showing up in
        # the launch line: vLLM is installed only in .venv-vllm, so a bare
        # `vllm` resolves to nothing and the server never starts. Same venv as
        # `bench_python` below, deliberately -- server and client must agree on
        # the vLLM version they are speaking about.
        "server_bin": ".venv-vllm/bin/vllm",
        "sampler": "experiments/scripts/power_sampler_nvidia.sh",
        # .venv-vllm is the CUDA venv and the only one here with vLLM + openai.
        # HANDOVER "One venv per vendor": never install vLLM into .venv.
        "bench_python": ".venv-vllm/bin/python",
    },
}


def _server_bin(backend: str) -> str:
    """Absolute path for a repo-relative binary, verbatim for one on PATH."""
    binary = BACKENDS[backend]["server_bin"]
    return str(REPO_ROOT / binary) if binary.startswith(".") else binary


def server_command(backend: str, artifact: str, port: int,
                   card: int, tp: int,
                   engine: dict[str, str] | None = None,
                   ) -> tuple[list[str], dict[str, str]]:
    """The launch line and the environment it needs, per backend.

    Returns the env OVERLAY, not a whole environment: the caller merges it, so a
    backend that pins by environment variable (CUDA) and one that pins by flag
    (FuriosaAI) are expressed the same way.

    Device pinning is the part worth stating. `--devices npu:N:*` names the card
    on the FuriosaAI side; CUDA has no such flag, so the card is selected by
    making it the only one the process can see. That also fixes what the server
    calls it -- inside the process the pinned card is always device 0 -- while the
    sampler and `nvidia-smi` keep using the PHYSICAL index. `--card` is the
    physical index in both roles and the two must not be allowed to drift apart.
    """
    if backend == "furiosa":
        return ([
            _server_bin("furiosa"), "serve", artifact,
            "--host", "127.0.0.1", "--port", str(port),
            "--devices", f"npu:{card}:*",
        ], {})
    if backend == "cuda":
        cmd = [
            _server_bin("cuda"), "serve", artifact,
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", str(tp),
        ]
        # Engine knobs are passed through rather than left at vLLM's defaults,
        # and that is load-bearing rather than tidy. The open-loop A40 points are
        # measured by `python -m bench run` at max_num_seqs 128 /
        # max_num_batched_tokens 2048 (the config that produced the 170.56 point).
        # A closed-loop curve taken at vLLM's defaults would differ from it in the
        # SCHEDULER as well as in the load generator, and the whole purpose of
        # measuring both is to attribute the difference to the protocol.
        for flag, value in sorted((engine or {}).items()):
            cmd += [flag, str(value)]
        return (cmd, {"CUDA_VISIBLE_DEVICES": str(card)})
    raise ValueError(f"unknown backend: {backend}")


def gpu_uuid(card: int) -> str | None:
    """The pinned GPU's UUID, for the A5(c) analysis filter.

    Resolved rather than asked for, because the alternative is an operator pasting
    a serial and the analysis silently averaging eight cards when they get it
    wrong -- seven of which are idle at ~30 W on this node and would drag the mean
    of the one under test toward idle. `nvidia-smi` reports `serial` as [N/A] on
    these boards, so the UUID is the stable identifier; `power_sampler_nvidia.sh`
    writes it into `device_sn` for exactly this join.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
             "-i", str(card)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0].strip() if out else None


def wait_for_server(port: int, timeout: float) -> str | None:
    """Reuses the pattern in rebuild_rngd_bundle_from_edf.py."""
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
                data = json.loads(resp.read())
                return data["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
            time.sleep(2.0)
    return None


def run_point(args, concurrency: int, out_dir: Path, repeat: int = 0) -> dict | None:
    # A5(b) sizes the pool by default. `--pool` exists for plumbing checks and for
    # the case rev 2 anticipates -- trimming a point that would otherwise run for
    # hours -- and it does NOT relax the rule: `summarise_point` still flags a pool
    # below 4x the concurrency, and the flag lands in the artifact.
    pool = args.pool if args.pool else max(MIN_POOL, POOL_MULTIPLE * concurrency)
    tag = f"c{concurrency}" + (f"_r{repeat}" if repeat else "")
    sampler_csv = out_dir / f"power_{tag}.csv"
    bench_json = out_dir / f"bench_{tag}.json"
    log_path = out_dir / f"serve_{tag}.log"

    cmd, env_overlay = server_command(
        args.backend, str(args.artifact), args.port, args.card, args.tp,
        engine=args.engine)
    env = {**os.environ, **env_overlay}
    with log_path.open("w") as log:
        server = subprocess.Popen(
            cmd, stdout=log, stderr=log, start_new_session=True, env=env,
        )
    sampler = subprocess.Popen(
        [str(REPO_ROOT / BACKENDS[args.backend]["sampler"]),
         "--out", str(sampler_csv)] +
        (["--devices", args.sample_devices] if args.sample_devices else []),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        model = wait_for_server(args.port, args.startup_timeout)
        if model is None:
            print(f"c{concurrency}: server never came up; see {log_path}", file=sys.stderr)
            return None

        # (d) settle, then the idle window the point is compared against.
        print(f"c{concurrency}: settling {SETTLE_S:.0f} s", file=sys.stderr)
        time.sleep(SETTLE_S)
        idle_start = time.time()
        time.sleep(IDLE_WINDOW_S)
        idle_window = Window(idle_start, time.time())

        print(f"c{concurrency}: benching pool={pool}", file=sys.stderr)
        bench_t0 = time.time()
        subprocess.run(
            # sys.executable may be a venv without the vendor stack, so the
            # interpreter is per backend: the FuriosaAI stack lives in the system
            # python (see rebuild_rngd_bundle_from_edf), the CUDA one in
            # .venv-vllm. Neither is `.venv`, which has no `openai` at all.
            [args.bench_python, "-u",
             str(REPO_ROOT / "experiments/scripts/bench_furiosa_endpoint.py"),
             "--base-url", f"http://127.0.0.1:{args.port}/v1",
             "--model", model, "--dataset", str(args.dataset),
             "--num-reqs", str(pool), "--concurrency", str(concurrency),
             "--out", str(bench_json)],
            cwd=REPO_ROOT, check=False, timeout=args.bench_timeout,
        )
        bench_window = Window(bench_t0, time.time())

        print(f"c{concurrency}: trailing idle {IDLE_WINDOW_S:.0f} s", file=sys.stderr)
        time.sleep(IDLE_WINDOW_S)
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=15)
        except subprocess.TimeoutExpired:
            sampler.kill()
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        try:
            server.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)

    if not bench_json.exists():
        print(f"c{concurrency}: the bench wrote no report", file=sys.stderr)
        return None
    bench = json.loads(bench_json.read_text())
    rows = read_sampler_csv(sampler_csv)
    return summarise_point(bench, rows, bench_window, idle_window,
                           device_sn=args.device_sn, pool_size=pool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="furiosa",
                    help="which execution half to use. Default stays `furiosa` so "
                         "every committed RNGD invocation runs unchanged.")
    # Not a Path on the CUDA side: `vllm serve` takes an HF model id as readily as
    # a directory, and coercing "meta-llama/Llama-3.1-8B" through Path would make
    # it look local. It is passed through verbatim and recorded in the artifact.
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--card", type=int, default=0,
                    help="PHYSICAL device index. On CUDA it becomes "
                         "CUDA_VISIBLE_DEVICES for the server (which then sees it "
                         "as device 0) and `nvidia-smi -i` for the sampler.")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel degree; CUDA only. The FuriosaAI side "
                         "takes it from the compiled artifact, not from a flag.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device-sn", default=None,
                    help="restrict the power/util analysis to this serial; "
                         "dev_name is not stable across re-enumeration")
    ap.add_argument("--sample-devices", default=None,
                    help="comma-separated dev_names for the sampler to record")
    ap.add_argument("--bench-python", default=None,
                    help="interpreter for the bench client. Default is per "
                         "backend: /usr/bin/python3 for furiosa (the vendor stack "
                         "lives there), .venv-vllm/bin/python for cuda. `.venv` "
                         "has no `openai` under either (verified 2026-09-08).")
    ap.add_argument("--pool", type=int, default=None,
                    help="override the A5(b) pool size; the point is still flagged "
                         "if it falls below 4x the concurrency")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent processes per point (STEP 3.2 asks for 2)")
    # Defaults are the engine config the A40's open-loop points were measured
    # under (outputs/phase0_bench/A40/vllm/meta.json), so the closed-loop and
    # open-loop halves differ ONLY in how load is offered. CUDA only; the
    # FuriosaAI side takes these from the compiled artifact.
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--startup-timeout", type=float, default=1800.0)
    ap.add_argument("--bench-timeout", type=float, default=7200.0)
    args = ap.parse_args()

    if args.bench_python is None:
        args.bench_python = BACKENDS[args.backend]["bench_python"]
    args.engine = {}
    if args.backend == "cuda":
        args.engine = {
            "--max-num-seqs": args.max_num_seqs,
            "--max-num-batched-tokens": args.max_num_batched_tokens,
            "--dtype": args.dtype,
            "--kv-cache-dtype": args.kv_cache_dtype,
            "--seed": args.seed,
        }
        # Pin BOTH halves to the same physical card by default. Leaving the
        # sampler unfiltered on an 8-GPU node records seven idle cards beside the
        # one under test; leaving `--device-sn` unset then averages all eight, and
        # A5(c) would be satisfied by a number that is mostly idle.
        if args.sample_devices is None:
            args.sample_devices = str(args.card)
        if args.device_sn is None:
            args.device_sn = gpu_uuid(args.card)
            if args.device_sn is None:
                print("could not resolve the GPU UUID; the power analysis will "
                      "average every device the sampler recorded", file=sys.stderr)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    points = [int(v) for v in args.concurrency.split(",") if v]

    results = []
    for c in points:
        for rep in range(args.repeats):
            r = run_point(args, c, out_dir, repeat=rep)
            if r is None:
                print(f"c{c} r{rep}: no result", file=sys.stderr)
                continue
            r["repeat"] = rep
            results.append(r)
            tag = f"c{c}" + (f"_r{rep}" if rep else "")
            (out_dir / f"point_{tag}.json").write_text(json.dumps(r, indent=2) + "\n")
            flag = "  POOL-BINDING" if r["pool_binding"] else ""
            pw = r["bench_window"].get("power_w")
            pstr = f" {pw['mean']:.1f} W" if pw else " (no power)"
            print(f"c{c} r{rep}: served {r['served_concurrency']:.2f} "
                  f"(ratio {r['served_ratio']:.3f}){pstr}{flag}", file=sys.stderr)

    # The run block is new with the CUDA backend and is not decoration: an
    # envelope file that does not say which backend, which artifact and which
    # physical card produced it cannot be checked against absolute rule 3 later,
    # and this repository moves between three nodes. Nothing reads `envelope.json`
    # programmatically, so adding it breaks no consumer.
    (out_dir / "envelope.json").write_text(
        json.dumps({
            "run": {
                "backend": args.backend,
                "artifact": args.artifact,
                "dataset": str(args.dataset),
                "card": args.card,
                "tp": args.tp,
                "device_sn": args.device_sn,
                "sample_devices": args.sample_devices,
                "bench_python": args.bench_python,
                "engine": args.engine,
                "closed_loop": True,
                "repeats": args.repeats,
            },
            "points": results,
        }, indent=2) + "\n")
    print(f"wrote {out_dir / 'envelope.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
