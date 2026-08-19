# A40 live deployment loop (Phase 4) — 2026-08-19

First end-to-end run of the Phase 4 path on real hardware: the planner's
`deploy` launched a real vLLM server on an A40, a load driver drove the sharegpt
workload at it, `status` scraped live metrics from the deployed server, and a
calibration model was fit from the matched sim-vs-real data. This validated the
deploy/monitor/calibration code on real hardware and caught one metric bug that
unit tests could not (see below).

## What ran
- **Deploy**: `deploy --plan outputs/plans/a40-live-llama-tp1.yaml --cluster
  experiments/configs/clusters/a40x8.yaml --no-dry-run` launched
  `vllm serve NousResearch/Meta-Llama-3.1-8B` (ungated mirror of Llama-3.1-8B;
  this host's HF token lacks the gated meta-llama license) on island
  `cuda-a40-node0-0` -> `CUDA_VISIBLE_DEVICES=0,1`, TP=1, prefix off. Server came
  up healthy (`/health` 200, `/v1/models` served). Real A40 KV budget observed at
  boot: **Available KV cache 24.64 GiB, GPU KV cache 201,856 tokens**
  (max_model_len 131072, concurrency 1.54x).
- **Load**: `experiments/scripts/replay_to_endpoint.py` replayed the 300 sharegpt
  requests against `/v1/completions` at concurrency 128 — 300/300 ok in 107.5 s,
  104,250 output tokens, ~970 tok/s client throughput.
- **Status (under load)**: throughput 214.7 tok/s (windowed), tokens/J 0.815,
  avg 330.6 W / peak 333.0 W over island GPUs 0+1 (TP=1 leaves GPU1 idle), TTFT
  p50/p95/p99 = 8.55 / 18.79 / 19.76 s.
- **Status (after load)**: cumulative 300 req / 104,250 tok; TPOT p50/p95/p99 =
  64.4 / 373.9 / 468.2 ms; throughput 0 tok/s and tokens/J 0 correctly (no active
  traffic in the window — the windowed delta, not a lifetime figure).
- **Stop**: `backend.stop()` (and the new `stop` CLI) terminated the server and
  freed both GPUs; the reused-pid + cmdline guard was exercised live.

## Calibration (matched data)
Fit from the committed A40 sim-vs-real summary (nominal, 1.32% mean |err|), saved
to `profiles/calibration/a40.yaml`:
- TTFT: `real = 1.0161 * sim + 150.74 ms` (source measured)
- TPOT: `real = 1.0081 * sim + 0.30 ms`
- bucket `sharegpt-llama31-8b-300`: TTFT p95 abs err 1.97%, TPOT 1.35%, n=5

A40 slightly under-predicts (alpha > 1), the same sign as the A5000 — so the
per-hardware `robust_metric = predicted * (1 + p95_err)` adds a ~2% TTFT margin.
Applying margins to planning stays opt-in (default off).

## Bug the live run caught (fixed)
`planner/monitor/metrics.py` looked for `vllm:time_per_output_token_seconds`, but
vLLM 0.19 emits `vllm:inter_token_latency_seconds`; TPOT came back all-nan. Fixed
by trying a candidate list of metric names (first with data wins) and adding a
regression test. Unit tests used a fixture with the old name, so only a real
scrape surfaced this.

## Known gaps / follow-ups
- **`VllmKnobs` has no `max_model_len`**, so `deploy` uses the model default
  (131072) while the sim configs pin 8192. For a matched live-vs-sim calibration,
  add `max_model_len` to the deploy knobs (and thus the serve command).
- The deploy backend resolves `vllm` from `PATH`; in this repo's dual-venv setup
  that means running `deploy --no-dry-run` with `.venv-vllm/bin` on `PATH`
  (documented). A configurable serve-executable would be cleaner.
- The live server's own metrics were not used for the fit (its max_model_len
  differs from the sim); the fit uses the matched bench data instead.
