# RNGD sim-vs-real: layerwise profile prediction vs furiosa-llm

*Measured 2026-08-25 on this NPU server. First sim-vs-real check on NPU hardware
in this project; the A40 equivalent is `profiles/calibration/a40.yaml`.*

## What was compared

| | prediction | reality |
| --- | --- | --- |
| stack | LLMServingSim on `profiler/perf/RNGD/.../tp8/` | `furiosa-llm serve`, prebuilt vendor artifact |
| how the numbers were obtained | layers measured one graph at a time through `furiosa.torch`, then interpolated | streamed OpenAI-compatible requests, TTFT/TPOT timed client-side |
| placement | 8 accelerators, TP=8 | `npu1pe0-3` + `npu1pe4-7`, one card, TP=8 |
| workload | 20 sharegpt requests, `outputs/envcheck/rngd20.jsonl` | same file, same 20 requests |
| prefix cache | enabled, 0.00 % hits | enabled, 0 hits |

Reproduce:

```bash
# real
furiosa-llm serve <artifact> --devices "npu:1:*" --port 8000
PYTHONPATH=$PWD python3 experiments/scripts/bench_furiosa_endpoint.py \
    --base-url http://127.0.0.1:8000/v1 --model <id> \
    --dataset outputs/envcheck/rngd20.jsonl --out outputs/rngd_bench/real_tp8.json
# prediction
python -m serving --cluster-config experiments/configs/clusters/rngd-llama31-8b-tp8.json \
    --dtype bfloat16 --block-size 16 --dataset outputs/envcheck/rngd20.jsonl \
    --output outputs/envcheck/rngd_verify_tp8.csv
# compare + fit
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/compare_rngd_sim_vs_real.py \
    --sim-csv outputs/envcheck/rngd_verify_tp8.csv \
    --real-json outputs/rngd_bench/real_tp8.json \
    --out-dir outputs/rngd_bench --prefix rngd-tp8 \
    --calibration-out profiles/calibration/rngd.yaml
```

**Token counts match exactly** — 13,787 generated against 13,787 requested, 20/20
requests, 0 failures. So this is a token-for-token comparison, not two runs that
happened to do different amounts of work.

## Result

| metric | real (ms) | sim (ms) | sim error |
| --- | ---: | ---: | ---: |
| TTFT mean | 1404.1 | 946.9 | **−32.6 %** |
| TTFT median | 1157.9 | 747.9 | −35.4 % |
| TTFT p99 | 2895.4 | 2119.0 | −26.8 % |
| TPOT mean | 28.4 | 35.7 | **+25.5 %** |
| TPOT median | 28.7 | 35.8 | +24.9 % |
| TPOT p99 | 30.1 | 38.2 | +27.1 % |
| Latency mean | 20941.3 | 25452.9 | +21.5 % |

Fitted `real = alpha * sim + beta` (`profiles/calibration/rngd.yaml`):

| metric | alpha | beta (ms) | mean rel. error |
| --- | ---: | ---: | ---: |
| TTFT | 1.3356 | 183.00 | 0.473 |
| TPOT | 0.6855 | 4.07 | −0.204 |

**The headline is not the size of the error but its sign: the two are
opposite.** The simulator is optimistic on prefill by about a third and
pessimistic on decode by about a quarter. They partly cancel in end-to-end
latency (+21.5 %, decode-dominated because output tokens far outnumber prompt
tokens here), which is exactly why quoting only total latency would have hidden
both.

For scale, the A40 bundle — profiled through vLLM's own layerwise profiler, which
measures the kernels the server actually runs — fits `alpha` 1.017 / 1.011 with
~2 % error. RNGD is an order of magnitude further off, and the causes below are
structural, not noise.

## Why prefill is under-predicted (real slower than sim)

**Bucket quantisation, worth ~11 %.** The artifact is compiled for fixed
`tokenwise_buckets` = 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024 and
`attention_buckets` stepping in 128s. A prompt does not pay for its own length,
it pays for the buckets that cover it: a 399-token prompt runs the 512 bucket, a
539-token prompt runs 1024. Over these 20 requests the charged prefill work is
27,408 tokens against 24,725 actual — **+10.9 % aggregate, +20.7 % mean per
request**. The simulator interpolates on exact token counts and so cannot see
this at all.

**Scheduler behaviour, the remaining ~20 %.** The server ran the eager scheduler
with `npu_queue_limit: 1`, i.e. one request queued to the NPU at a time, against
client concurrency 64. The simulator models continuous batching with a token
budget. Queueing delay under that mismatch lands in TTFT, not TPOT — consistent
with the errors having opposite signs.

## Why decode is over-predicted (real faster than sim)

**The bundle measures unfused layers; the server runs a fused graph.**
`furiosa-llm` loaded "46 AOT wired pipelines" — the decoder is compiled as whole
pipelines, so activations stay resident across layers. The profiling harness
compiles *one canonical layer per graph* by necessity (device spans are
hardware-unit-level and cannot be attributed to layers inside a larger graph), so
every measured layer pays entry and exit traffic that the real graph does not.
That inflates per-layer time, and decode is a pure sum of per-layer times — so
the prediction is systematically slow. This is the price of the measurement
method, and it is the first thing to attack if RNGD prediction accuracy matters.

**tp8 is the weakest bundle.** 40 shapes fell back to the CPU at tp8 against 1 at
tp1/tp2/tp4, because tp8 shards the dims small enough (intermediate 1792, 4
heads, 1 KV head) that the compiler stops using the tensor unit. Every layer
still holds 6–10 points, so interpolation works, but it is coarser here than at
any other TP degree — and tp8 is precisely the degree the vendor artifact forced
this comparison to use.

## What this does and does not license

- The calibration model in `profiles/calibration/rngd.yaml` is usable for
  correcting RNGD predictions **for this workload and TP degree**, the same way
  `a40.yaml` is scoped to its bucket. One bucket, 20 requests, five statistics
  per metric: thinner than the A40 fit.
- It does **not** license a GPU-vs-NPU efficiency claim yet. Two independent
  gaps stack: this sim-vs-real error, and the fact that the A40 and RNGD bundles
  were produced by different measurement methods (vLLM's own profiler vs this
  harness). Exp 4 should either calibrate both arms or state the asymmetry.
- Power is unaffected by any of this: the per-PE figures come from a direct
  0..8 loaded-PE board sweep (`outputs/rngd_profile/pe_power_sweep.json`), not
  from the latency model.

## Next, in order of leverage

1. **Model the buckets.** A prefill-cost model that rounds token counts up to the
   artifact's bucket ladder should recover most of the 11 %, and it is cheap:
   the ladder is readable from `artifact.json`.
2. **Measure fused blocks, not single layers.** Compiling a whole decoder layer
   (or several) as one graph and dividing gets closer to what the server runs,
   at the cost of the per-layer breakdown the contract's `dense.csv` wants. A
   hybrid — per-layer shapes for the grid, a fused-block correction factor — is
   likely the practical answer.
3. **Re-check at tp4.** Building an artifact with `furiosa-llm build -tp 4` would
   compare against the bundle that has near-complete coverage, separating
   "tp8 bundle is thin" from "the method is biased".
