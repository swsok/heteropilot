"""Benchmark a running ``furiosa-llm serve`` endpoint, per request.

Companion to ``experiments/scripts/profile_rngd.py``: that one measures layers
and feeds the simulator, this one measures the real serving stack end to end so
the two can be compared (Phase 4 sim-vs-real, `docs/HANDOVER_NPU.md` §3 step 6).

Why not ``replay_to_endpoint.py``: that script deliberately only generates load
and leaves TTFT/TPOT to the server's Prometheus histograms. ``furiosa-llm``'s
metrics surface is not the vLLM one, so this client streams and takes the
timings itself -- first-token arrival for TTFT, the gaps after it for TPOT.

IGNORES ``arrival_time_ns``. Every row is dispatched under one
``asyncio.Semaphore(--concurrency)``, so with concurrency >= the request count the
whole trace starts at t=0 -- a burst, whatever timestamps the file carries. That
matters when comparing against ``python -m serving``, which DOES replay the column:
feeding both the same file compares a burst against a spread arrival process, and
the queueing difference lands entirely in TTFT. This is what produced the
"-71 % TTFT" figure retracted in deviations D19. Either zero the arrivals on the
simulator side (``outputs/envcheck/rngd20_burst.jsonl``) or teach this script to
sleep until each row's arrival before comparing.

Fidelity note: the simulator replays each request's exact input and output token
counts, so a fair comparison needs the real run to generate the same number of
tokens. ``ignore_eos`` is requested to force that; the script records the token
counts it actually got, and ``compare_rngd_sim_vs_real.py`` reports the drift
rather than hiding it.

Usage::

    PYTHONPATH=$PWD python3 experiments/scripts/bench_furiosa_endpoint.py \
        --base-url http://127.0.0.1:8000/v1 --model furiosa-ai/Llama-3.1-8B-Instruct \
        --dataset outputs/envcheck/rngd20.jsonl \
        --out outputs/rngd_bench/real_tp8.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

NS_PER_S = 1_000_000_000


def load_trace(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


async def one_request(client: AsyncOpenAI, model: str, row: dict, index: int,
                      ignore_eos: bool, timeout: float) -> dict:
    """Stream one completion and time the first token separately from the rest."""
    prompt = row.get("input_tok_ids") or row["input_toks"]
    max_tokens = int(row["output_toks"])
    extra: dict = {}
    if ignore_eos:
        # vLLM-style knob; furiosa-llm may ignore it. The caller checks the
        # returned token count rather than trusting this to have worked.
        extra["ignore_eos"] = True

    started = time.perf_counter_ns()
    first_token_ns: int | None = None
    token_times: list[int] = []
    text_parts: list[str] = []
    error: str | None = None
    try:
        stream = await client.completions.create(
            model=model, prompt=prompt, max_tokens=max_tokens,
            temperature=0.0, stream=True, timeout=timeout,
            extra_body=extra or None,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].text or ""
            if piece == "":
                continue
            now = time.perf_counter_ns()
            if first_token_ns is None:
                first_token_ns = now
            token_times.append(now)
            text_parts.append(piece)
    except Exception as exc:  # a failed request is data, not a crash
        error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"

    ended = time.perf_counter_ns()
    n_chunks = len(token_times)
    ttft_ns = (first_token_ns - started) if first_token_ns else None
    # TPOT excludes the first token, matching the simulator's definition.
    tpot_ns = None
    if ttft_ns is not None and n_chunks > 1:
        tpot_ns = (token_times[-1] - first_token_ns) / (n_chunks - 1)
    return {
        "index": index,
        "requested_input_toks": int(row["input_toks"]),
        "requested_output_toks": max_tokens,
        "streamed_chunks": n_chunks,
        "ttft_ns": ttft_ns,
        "tpot_ns": tpot_ns,
        "latency_ns": ended - started,
        "chars": sum(len(p) for p in text_parts),
        "error": error,
    }


async def run(args) -> dict:
    rows = load_trace(args.dataset, args.num_reqs)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "none")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(row, index):
        async with semaphore:
            return await one_request(client, args.model, row, index,
                                     args.ignore_eos, args.timeout)

    started = time.perf_counter_ns()
    results = await asyncio.gather(*(guarded(r, i) for i, r in enumerate(rows)))
    wall_ns = time.perf_counter_ns() - started

    ok = [r for r in results if r["error"] is None and r["ttft_ns"] is not None]
    failed = [r for r in results if r["error"] is not None]
    return {
        "base_url": args.base_url,
        "model": args.model,
        "dataset": str(args.dataset),
        "concurrency": args.concurrency,
        "ignore_eos_requested": args.ignore_eos,
        "requests": len(rows),
        "ok": len(ok),
        "failed": len(failed),
        "wall_s": round(wall_ns / NS_PER_S, 3),
        "generated_chunks_total": sum(r["streamed_chunks"] for r in ok),
        "requested_output_toks_total": sum(r["requested_output_toks"] for r in ok),
        "per_request": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--num-reqs", type=int, default=0, help="0 = whole dataset")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"requests {report['ok']} ok / {report['failed']} failed "
          f"in {report['wall_s']:.1f}s")
    print(f"generated chunks {report['generated_chunks_total']} vs requested "
          f"{report['requested_output_toks_total']} output tokens")
    if report["failed"]:
        print("first failures:")
        for row in report["per_request"]:
            if row["error"]:
                print(f"  req {row['index']}: {row['error']}")
                break
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
