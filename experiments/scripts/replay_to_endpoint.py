#!/usr/bin/env python3
"""Replay a workload JSONL against a running OpenAI-compatible vLLM endpoint.

Drives load at a deployed HeteroPilot server (planner `deploy`) so `planner
status` sees real /metrics under traffic.

**Two load modes, and the difference is the whole point.**

*Closed loop* (the default, unchanged): a fixed number of requests in flight,
arrival timing ignored. A saturation load, like the bench. The server's own
Prometheus histograms are the source of truth for TTFT/TPOT; this client only
generates load and reports wall time and completion counts.

*Open loop* (``--open-loop``): every row is fired at its own
``arrival_time_ns``, with **no** concurrency bound. This is the mode
`WORK_ORDER_p2_regular_spec_evidence.md` STEP V2 needs, because the two modes
answer different questions. Under a closed loop the served concurrency is
whatever you set, so a predictor cannot disagree with it; under an open loop the
arrival process is fixed and the concurrency is an OUTCOME -- by Little's law an
optimistic predictor lands below the measurement and a pessimistic one above it.
That divergence is what claim 2 rests on and no committed measurement contains
it: every RNGD and A40 point to date is closed-loop or arrival-rate-matched, so
`L_pred ~ L_meas` holds by construction.

Open loop also records per-request client-side timestamps -- arrival, first
token, completion -- because a served concurrency cannot be recovered from
aggregate counts. `--out` writes them, along with

  L_meas = sum(residency) / window,

the disclosure's definition (`WORK_ORDER_p2_regular_spec_evidence.md` §0.3).
Residency runs from ARRIVAL, so a request waiting in the server's queue counts;
that is deliberate and it is what makes the figure comparable with the
simulator's `served_concurrency_from_sim`.

Streaming is forced on in open loop: without it there is no first-token time and
therefore no TTFT and no TPOT.

Usage:
  # closed loop, as before
  .venv-vllm/bin/python experiments/scripts/replay_to_endpoint.py \
      --base-url http://127.0.0.1:8000/v1 --model <id> \
      --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
      --concurrency 128 [--num-reqs N] [--max-tokens-cap 512]

  # open loop at a chosen offered rate
  .venv-vllm/bin/python experiments/scripts/replay_to_endpoint.py \
      --base-url http://127.0.0.1:8000/v1 --model <id> \
      --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
      --open-loop --target-rps 5 --out outputs/p2_evidence/v2/a40_rps5.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI

# `openai` lives in `.venv-vllm`, not in the `.venv` the test suite runs under, so
# it is imported where it is used rather than at module scope. Everything except
# the two request coroutines -- the arrival schedule, the rescale, the Little's
# law summary -- is then testable without it, and the coroutines take the client
# as an argument so a stub can stand in for one.
NS_PER_S = 1e9
#: A warm-up only has to open a connection; it must not become a wait.
WARM_UP_TIMEOUT_S = 10.0


def _load(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _percentile(values: list[float], pct: float) -> float:
    """Linear interpolation, matching `planner/util/percentile.py`.

    Duplicated rather than imported: this script runs under `.venv-vllm`, which
    is a different interpreter from the one `planner/` is installed in.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# ---------------------------------------------------------------------------
# Arrival schedule
# ---------------------------------------------------------------------------

def arrival_offsets_s(rows: list[dict]) -> list[float]:
    """Each row's arrival, in seconds from the start of the arrival process.

    The generator (`planner/util/workload.py`) lays a Poisson process down from
    t = 0, so the offsets are used as they stand -- the first arrival is a
    positive number and shifting it to zero would change the process.
    """
    return [float(r.get("arrival_time_ns", 0)) / NS_PER_S for r in rows]


def trace_rps(offsets: list[float]) -> float:
    """The trace's own offered rate: n events observed up to the last one.

    For a Poisson process of rate L the n-th event arrives at E[T_n] = n / L, so
    n / T_n estimates the rate. This is what `--target-rps` rescales against; pass
    `--trace-rps` to state it instead of estimating it.
    """
    span = max(offsets) if offsets else 0.0
    return (len(offsets) / span) if span > 0 else 0.0


def rescale(offsets: list[float], source_rps: float, target_rps: float) -> list[float]:
    """Stretch or compress the schedule to a new offered rate.

    A constant factor preserves the shape of the arrival process -- a rescaled
    Poisson process is still Poisson -- and divides the rate exactly.
    """
    if target_rps <= 0:
        raise ValueError("--target-rps must be positive")
    if source_rps <= 0:
        raise ValueError("cannot rescale a trace whose arrival span is zero")
    factor = source_rps / target_rps
    return [o * factor for o in offsets]


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------

async def _one(client: AsyncOpenAI, model: str, row: dict, cap: int,
               sem: asyncio.Semaphore) -> dict:
    """Closed-loop request: bounded by `sem`, no arrival timing, no streaming."""
    prompt = row.get("input_tok_ids") or row.get("input_toks")
    max_tokens = min(int(row.get("output_toks", 128)), cap) or 1
    async with sem:
        t0 = time.monotonic()
        try:
            r = await client.completions.create(
                model=model, prompt=prompt, max_tokens=max_tokens,
                temperature=0.0, stream=False,
            )
            usage = r.usage
            return {"ok": True, "latency_s": time.monotonic() - t0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0}
        except Exception as exc:
            return {"ok": False, "latency_s": time.monotonic() - t0, "error": str(exc)[:200]}


async def _one_open_loop(client: AsyncOpenAI, model: str, row: dict, cap: int,
                         index: int, due_s: float, t_base: float) -> dict:
    """Open-loop request: fired at `due_s` after `t_base`, unbounded, streamed.

    `launch_error_s` is reported per request rather than asserted here. A driver
    that silently ran late would look exactly like a server that was slow, so the
    error belongs in the record.
    """
    prompt = row.get("input_tok_ids") or row.get("input_toks")
    max_tokens = min(int(row.get("output_toks", 128)), cap) or 1

    delay = due_s - (time.monotonic() - t_base)
    if delay > 0:
        await asyncio.sleep(delay)
    launched = time.monotonic()

    rec = {"index": index, "ok": False,
           "due_s": due_s,
           "arrival_s": launched - t_base,
           "launch_error_s": (launched - t_base) - due_s,
           "first_token_s": None, "completion_s": None,
           "streamed_chunks": 0, "requested_output_toks": max_tokens}
    try:
        stream = await client.completions.create(
            model=model, prompt=prompt, max_tokens=max_tokens,
            temperature=0.0, stream=True,
        )
        first = None
        chunks = 0
        try:
            async for chunk in stream:
                text = chunk.choices[0].text if chunk.choices else ""
                if not text:
                    continue
                chunks += 1
                if first is None:
                    first = time.monotonic()
        finally:
            # Leaving the SSE body open makes httpcore raise GeneratorExit out
            # of its own teardown, which looks like a driver failure in the log.
            closer = getattr(stream, "close", None)
            if closer is not None:
                await closer()
        done = time.monotonic()
        rec.update(ok=chunks > 0, streamed_chunks=chunks,
                   first_token_s=(first - t_base) if first else None,
                   completion_s=done - t_base)
    except Exception as exc:
        rec["error"] = str(exc)[:200]
        rec["completion_s"] = time.monotonic() - t_base
    return rec


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def summarise_open_loop(records: list[dict], offered_rps: float) -> dict:
    """Served concurrency by Little's law, plus TTFT/TPOT from client timestamps."""
    ok = [r for r in records if r.get("ok")]
    summary: dict = {
        "requests": len(records), "ok": len(ok), "failed": len(records) - len(ok),
        "offered_rps": offered_rps,
    }
    errs = [abs(r["launch_error_s"]) for r in records if r.get("launch_error_s") is not None]
    if errs:
        summary["launch_error_ms"] = {
            "mean": sum(errs) / len(errs) * 1e3,
            "p50": _percentile(errs, 50) * 1e3,
            "p99": _percentile(errs, 99) * 1e3,
            "max": max(errs) * 1e3,
        }
    if not ok:
        return summary

    residency = [r["completion_s"] - r["arrival_s"] for r in ok]
    window = max(r["completion_s"] for r in ok) - min(r["arrival_s"] for r in ok)
    summary["window_s"] = window
    # The disclosure's definition: total residency over the observation window.
    # Residency starts at ARRIVAL, so queued requests count.
    summary["served_concurrency"] = (sum(residency) / window) if window > 0 else None

    ttft = [(r["first_token_s"] - r["arrival_s"]) * 1e3 for r in ok if r["first_token_s"]]
    tpot = [((r["completion_s"] - r["first_token_s"]) / (r["streamed_chunks"] - 1)) * 1e3
            for r in ok if r["first_token_s"] and r["streamed_chunks"] > 1]
    for name, values in (("ttft_ms", ttft), ("tpot_ms", tpot)):
        if values:
            summary[name] = {"mean": sum(values) / len(values),
                             "p50": _percentile(values, 50),
                             "p95": _percentile(values, 95),
                             "p99": _percentile(values, 99)}
    chunks = sum(r["streamed_chunks"] for r in ok)
    summary["output_tokens"] = chunks
    summary["output_tok_s"] = chunks / window if window > 0 else None
    return summary


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

async def warm_up(client: AsyncOpenAI) -> float:
    """Open the connection before the schedule starts, and say what it cost.

    The client builds its httpx transport and its connection pool on first use,
    and that work is synchronous: measured against a local stub it blocked the
    event loop for **87 ms**, which arrived as a single late request -- the one
    due while the first was connecting -- and nothing else. An open-loop driver
    whose first inter-arrival is wrong is offering a different arrival process
    from the one it reports, so this is paid before `t_base` rather than during
    the run.

    `models.list()` is used rather than a throwaway completion: it opens the same
    pooled connection without putting a generation on the server under test.
    Failure is not fatal -- an endpoint that does not implement it still gets a
    connection attempt, which is most of the cost.
    """
    t0 = time.monotonic()
    # No retries: an endpoint without `/v1/models` answers 5xx, and the SDK's
    # default backoff then spends seconds on a call whose only job is to open a
    # socket. Measured against a stub that does not implement it, retries cost
    # 1.68 s against ~30 ms for the single attempt.
    probe = client
    with contextlib.suppress(Exception):
        probe = client.with_options(max_retries=0, timeout=WARM_UP_TIMEOUT_S)
    with contextlib.suppress(Exception):
        await probe.models.list()
    return time.monotonic() - t0


async def _run_open_loop(args: argparse.Namespace, rows: list[dict],
                         client: AsyncOpenAI) -> int:
    offsets = arrival_offsets_s(rows)
    source = args.trace_rps or trace_rps(offsets)
    if args.target_rps:
        offsets = rescale(offsets, source, args.target_rps)
        offered = args.target_rps
    else:
        offered = source
    print(f"open loop: {len(rows)} requests -> {args.base_url} "
          f"(trace {source:.3f} rps, offering {offered:.3f} rps, "
          f"span {max(offsets):.1f}s, no concurrency bound)", flush=True)

    warm_s = await warm_up(client) if args.warmup else None
    if warm_s is not None:
        print(f"warm-up: {warm_s * 1e3:.1f} ms (paid before the schedule starts)",
              flush=True)

    t_base = time.monotonic()
    records = await asyncio.gather(*[
        _one_open_loop(client, args.model, row, args.max_tokens_cap, i, due, t_base)
        for i, (row, due) in enumerate(zip(rows, offsets, strict=True))
    ])
    summary = summarise_open_loop(list(records), offered)

    le = summary.get("launch_error_ms", {})
    print(f"done: {summary['ok']}/{summary['requests']} ok, "
          f"{summary['failed']} failed in {summary.get('window_s', 0):.1f}s")
    if le:
        print(f"launch error (ms): mean={le['mean']:.2f} p99={le['p99']:.2f} "
              f"max={le['max']:.2f}")
    if summary.get("served_concurrency") is not None:
        print(f"served concurrency (Little's law) = {summary['served_concurrency']:.3f}")
    for name in ("ttft_ms", "tpot_ms"):
        if name in summary:
            s = summary[name]
            print(f"{name}: mean={s['mean']:.2f} p50={s['p50']:.2f} "
                  f"p95={s['p95']:.2f} p99={s['p99']:.2f}")
    if summary.get("output_tok_s"):
        print(f"output tokens={summary['output_tokens']}, "
              f"{summary['output_tok_s']:.1f} tok/s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"base_url": args.base_url, "model": args.model,
             "dataset": str(args.dataset), "mode": "open_loop",
             "trace_rps": source, "offered_rps": offered,
             "max_tokens_cap": args.max_tokens_cap, "warm_up_s": warm_s,
             "summary": summary, "per_request": list(records)}, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if summary["ok"] else 1


async def _run_closed_loop(args: argparse.Namespace, rows: list[dict],
                           client: AsyncOpenAI) -> int:
    sem = asyncio.Semaphore(args.concurrency)
    print(f"replaying {len(rows)} requests -> {args.base_url} "
          f"(model={args.model}, concurrency={args.concurrency})", flush=True)
    t0 = time.monotonic()
    results = await asyncio.gather(*[_one(client, args.model, r, args.max_tokens_cap, sem)
                                     for r in rows])
    wall = time.monotonic() - t0
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    out_tok = sum(r["completion_tokens"] for r in ok)
    print(f"done: {len(ok)}/{len(results)} ok, {len(fail)} failed in {wall:.1f}s")
    if ok:
        lat = sorted(r["latency_s"] for r in ok)
        print(f"client latency (s): p50={lat[len(lat)//2]:.2f} "
              f"p99={lat[min(len(lat)-1, int(len(lat)*0.99))]:.2f} max={lat[-1]:.2f}")
        print(f"output tokens={out_tok}, client throughput={out_tok/wall:.1f} tok/s")
    if fail:
        print("first error:", fail[0].get("error"))
    return 0 if ok else 1


async def _run(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    rows = _load(Path(args.dataset), args.num_reqs)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    try:
        if args.open_loop:
            return await _run_open_loop(args, rows, client)
        return await _run_closed_loop(args, rows, client)
    finally:
        # Without this the connection pool is still holding streamed response
        # bodies when the loop tears down, and httpcore finalises them from
        # asyncgen shutdown -- which prints a RuntimeError traceback on a run
        # that actually succeeded. Exit status was 0 and the numbers were right;
        # the noise is what makes a good run look like a failed one.
        with contextlib.suppress(Exception):
            await client.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=128,
                   help="closed loop only; open loop is unbounded by definition")
    p.add_argument("--num-reqs", type=int, default=0, help="0 = whole dataset")
    p.add_argument("--max-tokens-cap", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--open-loop", action="store_true",
                   help="honour arrival_time_ns, no concurrency bound, stream")
    p.add_argument("--target-rps", type=float, default=None,
                   help="open loop: rescale the schedule to this offered rate")
    p.add_argument("--trace-rps", type=float, default=None,
                   help="open loop: the trace's own rate, instead of estimating it")
    p.add_argument("--out", type=Path, default=None,
                   help="open loop: write per-request records and the summary here")
    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   help="open loop: skip the pre-schedule connection warm-up "
                        "(costs ~87 ms of launch accuracy on the first arrival)")
    p.set_defaults(warmup=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.open_loop and (args.target_rps or args.out):
        raise SystemExit("--target-rps and --out apply to --open-loop only")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
