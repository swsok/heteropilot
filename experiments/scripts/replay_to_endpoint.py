#!/usr/bin/env python3
"""Replay a workload JSONL against a running OpenAI-compatible vLLM endpoint.

Drives load at a deployed HeteroPilot server (planner `deploy`) so `planner
status` sees real /metrics under traffic. The server's own Prometheus histograms
are the source of truth for TTFT/TPOT (this client only generates load and
reports client-side wall time / completion counts).

Each JSONL row carries `input_tok_ids` (list[int]) and `output_toks` (int); we
send the token ids as the prompt and cap generation at `output_toks`. Requests
are fired with bounded concurrency (a saturation load, like the bench), arrival
timing intentionally ignored.

Usage:
  .venv-vllm/bin/python experiments/scripts/replay_to_endpoint.py \
      --base-url http://127.0.0.1:8000/v1 --model <id> \
      --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
      --concurrency 128 [--num-reqs N] [--max-tokens-cap 512]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI


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


async def _one(client: AsyncOpenAI, model: str, row: dict, cap: int,
               sem: asyncio.Semaphore) -> dict:
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


async def _run(args: argparse.Namespace) -> int:
    rows = _load(Path(args.dataset), args.num_reqs)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--num-reqs", type=int, default=0, help="0 = whole dataset")
    p.add_argument("--max-tokens-cap", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600.0)
    return asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
