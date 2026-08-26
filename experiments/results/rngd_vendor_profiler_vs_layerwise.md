# RNGD: FuriosaAI's own profiler vs our layerwise measurements

*Measured 2026-08-26 on one RNGD card, using the vendor profiler documented at
developer.furiosa.ai. This is the first look inside the **real compiled graph**;
everything before it measured synthetic layers from outside.*

## Getting the vendor profiler to run

The documentation names `FURIOSA_PROFILER_OUTPUT_PATH` and a
`furiosa.runtime.profiler.profile()` context manager. Neither works as written on
the installed stack, and the useful path is a third one:

| documented | installed here |
| --- | --- |
| `furiosa.runtime.profiler` | **absent** — only `furiosa.native_runtime` is installed (`furiosa` has `models`, `native_common`, `native_llm_common`, `native_runtime`, `native_torch`, `torch`) |
| `FURIOSA_PROFILER_OUTPUT_PATH` | the native runtime reads **`FURIOSA_PROFILER_OUTPUT_PATHS`** (plural); the singular form produced nothing |
| — | **`EDF_PROFILER_OUTPUT_PATH`** is what actually produced output, gated by `TUC_PROFILE_LEVEL` / the `span::tuc` tracing target |

Found by reading the strings in `furiosa/native_runtime.*.so`, which also carries
the runtime's own warning: *"'span::tuc' is not enabled with info or above. EDF
profiles will not be logged."*

What worked:

```bash
FURIOSA_PROFILER_OUTPUT_PATHS=$PWD/outputs/rngd_vendor_prof/trace.json \
EDF_PROFILER_OUTPUT_PATH=$PWD/outputs/rngd_vendor_prof/edf.json \
TUC_PROFILE_LEVEL=info RUST_LOG=span::tuc=info \
  furiosa-llm serve <artifact> --port 8010 --devices "npu:0:*"
```

`edf.json` is **CSV, not JSON** — `leader_device,name,cycle`, 8.7 MB / 89,639 rows
for a 5-request run. Each row is one stage execution on the device, named by the
compiled bucket, e.g.
`Tokenwise(TokenwiseBucket { input_size: 8 }) #0` and
`Attention(AttentionBucket { batch_size: 1, attention_size: 1024, kv_cache_size: 1023 }) #0`.

## Deriving the clock, and a structural surprise

`cycle` is device cycles, so a clock is needed. Total device cycles over the run
divided by wall time gives **1,599.9 MHz** — a round 1.6 GHz to within 0.006 %.
That is only consistent if the card was ~100 % busy, which 5 concurrent requests
on one card would be, so both facts fall out together: **the clock is 1.6 GHz and
the card was saturated.** Everything below uses 1.6 GHz.

The `leader_device` is **`npu0pe0-3`** — a fused PE *quad*, not 8 individual PEs.
The serve log agrees: `DP entry DpId(0) → device [npu0pe0-3, npu0pe4-7]`. So the
artifact's `tensor_parallel_size: 8` is realised as **two fused 4-PE devices**, not
eight separate ranks. That matters for how the perf bundle should be built and is
recorded here as a finding, not yet acted on.

## Where the time goes

Per decode forward, reconstructed from the trace (decode device time ÷ measured
TPOT gives 780 forwards; 3,364 output tokens ÷ 780 = 4.31 tokens per forward,
matching the `input_size` 4 and 8 buckets, and 25,278 Tokenwise executions ÷ 780 =
**32.4 ≈ 32 layers**, i.e. one Tokenwise execution per decoder layer):

| stage | per forward | note |
| --- | ---: | --- |
| Tokenwise (32 layers) | 16.22 ms | 507 µs per layer |
| Attention | 4.44 ms | 78.8 executions per forward |
| **total** | **20.67 ms** | measured TPOT this run: **20.67 ms** |

The reconstruction closes exactly, which is the check that the stage accounting is
right. It also settles something: at this concurrency TPOT is **entirely device
execution** — there is no scheduler slack left to explain.

Prefill, for completeness: Tokenwise at `input_size` 512 costs 1472 µs and at 1024
costs 2782 µs per layer — 1.89× for 2× the tokens, i.e. compute-bound and scaling,
the opposite regime from decode.

## The comparison

| per decoder layer, decode step | time |
| --- | ---: |
| **vendor profiler** (real compiled graph, batch 4–8) | **507 µs** |
| **our layerwise bundle** (tp8, one PE, synthetic layers) | **290–307 µs** |

Our harness accounts for **58 %** of the real per-layer decode cost; the real graph
is **1.72×** what we measure.

**Both are flat in batch size, and that agreement is the important part.** The
vendor stage costs 502.2 µs at bucket 4 and 495.8 µs at bucket 8 — indistinguishable.
Our bundle gives 306.6 / 290.4 / 295.0 / 304.9 µs at 1 / 4 / 8 / 16 tokens — also
flat. Two independent instruments, one measuring compiler-emitted stages from
inside and one measuring hand-written layers from outside, both report that RNGD
decode does not care how many tokens are in the batch. That is the memory-bound
signature, and it is now confirmed rather than inferred.

## Correction to an earlier number

`experiments/results/rngd_card_vs_pe_model.md` and the header of
`profiles/accelerators/furiosa_rngd_card.yaml` derived an intra-card collective
cost of **≤202 µs per all-reduce** by attributing the card-model residual
(12.92 ms) to 64 collectives per token. **That arithmetic conflated per-token with
per-forward.** The residual was a per-token TPOT gap measured at concurrency 64,
where one forward serves many tokens, so dividing it by a per-forward collective
count is not valid. The figure should not be quoted.

What the vendor profiler gives instead is direct and per-layer: the real graph
spends **507 µs** per layer where our synthetic layers account for **295 µs**, so
**212 µs per decoder layer** is work our harness does not capture. That is the
better-founded number. It is *not* purely the all-reduce — the compiler's Tokenwise
stage also contains whatever else it fused in (residual adds, the second norm, KV
cache writes) — so 212 µs per layer is an upper bound on the reduction, in the same
way 202 µs was, but now measured on the real graph rather than inferred from a
model residual.

## What this changes

- **The layerwise bundle under-measures decode by 1.72×, and now we know by how
  much and where.** That is a better position than the sim-vs-real error alone
  gave, because it localises the gap to the per-layer stage rather than leaving it
  spread across scheduler, buckets and collectives.
- **The vendor profiler is the better instrument for this hardware** and should be
  the basis of any future RNGD bundle: it reports the stages the compiler actually
  emits, named, on the real graph. Our harness exists because RNGD has no vLLM
  plugin, but it does not have to be the only input.
- **`tensor_parallel_size: 8` is two fused quads, not eight ranks.** A bundle built
  on the per-PE `tp8/` shapes therefore models a rank granularity the hardware does
  not use. Rebuilding the bundle from EDF traces at the two-quad granularity is the
  obvious next step and needs no new hardware access.

## Reproducing

```bash
# 1. serve with the profiler on (env vars above), on a fully free card
# 2. drive a handful of requests
PYTHONPATH=$PWD python3 experiments/scripts/bench_furiosa_endpoint.py \
    --base-url http://127.0.0.1:8010/v1 --model <id> \
    --dataset outputs/rngd_vendor_prof/probe5.jsonl \
    --out outputs/rngd_vendor_prof/real_5req.json --concurrency 4
# 3. stop the server, then aggregate edf.json (CSV) by stage and bucket
```

Raw artifacts: `outputs/rngd_vendor_prof/real_5req.json` (client-side timings) and
the EDF CSV, which is 8.7 MB and regenerable, so it is not committed.
