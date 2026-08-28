# Rebuilding the RNGD perf bundle from FuriosaAI's EDF traces

*Built 2026-08-26 from six `furiosa-llm serve` runs on RNGD npu0, 1.74 M stage
executions. This replaces synthetic layer measurements with the stages the
vendor compiler actually emits, and it is the first RNGD bundle whose decode
prediction is accurate.*

## The result first

Same 20 sharegpt requests, same real furiosa-llm TP=8 benchmark
(`rngd_sim_vs_real_summary.md`), three bundles:

| bundle | TTFT mean | err | TPOT mean | err | latency mean | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **real** furiosa-llm TP=8 | 1404.1 ms | — | **28.4 ms** | — | 20941 ms | — |
| per-PE harness, 8 acc @ tp8 | 946.9 | −32.6 % | 35.7 | **+25.7 %** | 25453 | +21.5 % |
| card harness, 1 acc @ tp1 | 310.0 | −77.9 % | 15.5 | **−45.5 %** | — | — |
| **card EDF, 1 acc @ tp1** | 403.0 | −71.3 % | **27.6** | **−3.1 %** | 19287 | −7.9 % |

Decode, every percentile: −3.4 / −1.5 / −2.4 / −1.4 % at p50 / p90 / p95 / p99.
That is the accuracy class of the A40 bundle profiled through vLLM's own
profiler (~2 %), reached on hardware with no vLLM plugin at all. It also holds at
the other end of the load curve — **+2.4 % on unloaded TPOT** (15.4 ms against a
measured 15.1 ms), which is what makes it a working cost model rather than a fit
to one operating point.

**Why the card abstraction works now when it failed before.** A `tp1` instance
has no TP group, so the simulator adds no collective. With per-PE harness numbers
that was fatal — the intra-card all-reduce was simply missing, and decode came
out 45.5 % fast. An EDF stage time is what the *card* took to execute that stage,
**reduction included**, so the measurement has already paid for the
communication; charging it again would double-count. The abstraction was never
the problem. The input was.

The granularity now matches the hardware too: the artifact's
`tensor_parallel_size: 8` is realised as **two fused 4-PE quads** (`leader_device`
is `npu0pe0-3`; the serve log confirms `DpId(0) → [npu0pe0-3, npu0pe4-7]`), so a
per-PE `tp8` bundle modelled a rank granularity that does not exist.

## Three things the traces contain that the first draft missed

**1. There are three stage kinds, not two, and the third is 98.8 % of device
cycles at concurrency 1.** Besides `Tokenwise` and `Attention` there is
`Composed(a, b)`: nine variants partitioning `[0, 64]` in steps of 8 plus a
terminal `Composed(64, 64)`, each executing **exactly once per forward** (n
identical across all nine — 16,473 against 16,495 generated tokens). So the
runtime has two compiled plans:

| batch | plan |
| --- | --- |
| 1 | one fully-fused `Composed` graph — no `Tokenwise`, no `Attention` |
| ≥ 2 | per-layer `Tokenwise` + `Attention`, exactly 32 `Tokenwise` per forward |

| concurrency | Composed | Tokenwise | Attention |
| --- | ---: | ---: | ---: |
| 1 | **98.8 %** | 1.0 % | 0.1 % |
| 2 | 30.8 % | 59.0 % | 10.2 % |
| 4 | 1.7 % | 78.3 % | 19.9 % |
| 32 | 0.5 % | 64.7 % | 34.8 % |

This is why there is **no `input_size: 1` Tokenwise bucket anywhere in 1.74 M
executions** — at batch 1 the bucketed path is not used at all. `tokens=1` is the
row the simulator needs most for decode, so it is built from `Composed`.

**2. Stage times overlap only on the fused path.** Summed device cycles against
wall time: 99.5 % at c4, 99.3 % at c8, 100.5 % at c16, 98.7 % at c32 — the
bucketed path sums to wall time and can be used as measured. c1 is **114.7 %**,
so the fused graph pipelines internally and a sum of its nine medians
over-counts a forward by ~15 %. The builder divides that back down to the
measured wall-clock forward.

**3. Attention has two regimes, separated by `attention_size − kv_cache_size`.**

| `asz − kv` | meaning | mapping |
| --- | --- | --- |
| `== 1` | decode step | `n_decode = batch_size`, `kv_decode = kv_cache_size` |
| `> 1` | prefill chunk of `asz − kv` new tokens on `kv` already cached | `prefill_chunk = asz − kv`, `kv_prefill = kv` |

The first draft mapped every `kv > 0` bucket to decode, which put 32
chunked-prefill continuations on the decode axis and made it non-monotonic in KV
— 154 µs at `kv=128` against 49 µs at `kv=1023`. Decode attention is in fact
**flat in KV length** (49.1 / 38.6 / 46.0 / 48.6 µs at kv 1023 / 2047 / 4095 /
8191 at batch 1): at these lengths it is overhead-bound, not bandwidth-bound.

## What is measured and what is inherited

This is the part to read before quoting any single row.

| bundle content | source |
| --- | --- |
| absolute per-decoder-layer latency, per token bucket | **measured**, real graph |
| decode attention per layer | **measured**, calibrated per traffic mix (below) |
| prefill attention per bucket | **measured** |
| the head (final norm + lm_head + sampling) at 1 sequence | **measured**, terminal `Composed` segment |
| split of a layer's time across `qkv_proj` / `o_proj` / … | **inherited** from the harness — not a vendor measurement |
| how the head scales with sequence count | **inherited** from the harness |

The compiler fuses a whole decoder layer into one stage and does not expose the
pieces. The simulator only ever *sums* the per-layer lookups for an iteration, so
the sum is what has to be right: magnitude comes from the vendor, distribution
from the harness, rescaled to sum to the vendor figure. The ratio is recorded per
bucket in `outputs/rngd_edf_bundle/edf_vs_harness_dense.csv`:

| tokens | vendor µs/layer | harness sum µs | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 388.9 (dense, from `Composed`) | 306.6 | ×1.27 |
| 2 | 482.4 | 299.8 | ×1.61 |
| 8 | 471.0 | 295.0 | ×1.60 |
| 64 | 510.4 | 332.9 | ×1.53 |
| 512 | 1431.5 | 951.7 | ×1.50 |
| 1024 | 2685.7 | 2305.9 | ×1.16 |

The harness under-measures decode by 1.5–1.65×, consistent with the 1.72× found
in `rngd_vendor_profiler_vs_layerwise.md`, and the gap closes at prefill sizes
where compute rather than per-stage overhead dominates.

**The `tokens=1` row takes three corrections**, in order: the nine `Composed`
segments are scaled so their sum equals the measured wall-clock forward (union,
×0.870); the terminal segment is moved to `per_sequence.csv` because it is the
head, not a layer; and the measured batch-1 attention (46.9 µs/layer) is
subtracted so the simulator's separate attention charge is not counted twice.
Result: 435.8 µs/layer total → **388.9 µs/layer of dense**.

**Decode attention is calibrated to this traffic mix, and this is a real
limitation.** The runtime groups a decode batch by KV bucket, so per-layer
attention depends on the batch's KV *diversity*, not just its size:

| concurrency | sequences/forward | attention execs per layer | µs/layer | mean KV |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.95 | 1.95 | 88.3 | 2396 |
| 4 | 3.91 | 2.40 | 131.3 | 2207 |
| 8 | 8.91 | 2.87 | 198.0 | 2194 |
| 16 | 15.16 | 3.03 | 254.2 | 2201 |
| 32 | 29.09 | 3.08 | 329.7 | 2220 |

The §3.7 contract asks for one number given `n_decode` and a mean KV, so it
cannot express "3 executions because the batch spans 3 KV buckets". Charging a
single bucket median would under-count per-layer attention by ~3× at large batch.
Each concurrency therefore contributes one row whose time is total
decode-attention device time ÷ (forwards × 32), which makes the total close but
ties the decode-attention axis to sharegpt-like traffic at mean KV ≈ 2200 (stable
to ±1 % across all five batched traces, which is why one calibration serves them
all).

## The check the synthetic bundle could not pass

Predicted wall time from this bundle against measured wall time, over a 32×
concurrency range:

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| predicted wall (s) | 255.3 | 157.0 | 93.7 | 52.5 | 36.6 | 25.6 |
| measured wall (s) | 255.2 | 156.9 | 93.3 | 55.1 | 38.6 | 25.5 |
| error | +0.0 % | +0.1 % | +0.5 % | −4.7 % | −5.2 % | +0.2 % |

Prediction = decode forwards × (32 × (dense + attention) + head + per-iteration)
+ measured prefill device time. c1 and c2 sum both plans (c2 runs 2,744 fused
forwards alongside 5,738 bucketed ones). This is self-consistency, not
independent validation — it checks that the mapping onto the contract preserves
the measured totals, which is exactly what the synthetic card bundle got wrong.

## The remaining error is TTFT, and it is not the bundle

TTFT is still −61 % to −80 % across percentiles. The traces say why it cannot be
a magnitude problem: **real TTFT for identical prompts rises 12× with load while
the prefill work is unchanged.**

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real TTFT mean (ms) | 158.4 | 205.0 | 260.3 | 406.7 | 895.5 | 1894.4 |
| real TPOT mean (ms) | 15.26 | 18.59 | 21.41 | 24.14 | 27.24 | 30.14 |

Same 24 requests, same 1,220-token mean prompt, every time. Isolated prefill
costs 158 ms; the 1404 ms in the validation benchmark is therefore ~90 % queuing.
The simulator's 403 ms means it queues far less — an admission and batching
difference between upstream's scheduler and furiosa-llm's, not a profile error.
Fitting `real = α·sim + β` gives α = 2.09 for TTFT against α = 0.81 for TPOT,
i.e. the TTFT residual is nearly a pure scale factor, which is what a queuing
difference looks like.

### Separating the bundle from the scheduler

The clean test: simulate the **same six requests** with arrivals 15 s apart so
nothing queues, and compare against the real concurrency-1 run of those same six.

| unloaded, 6 requests, 1016-token mean prompt | real (c1) | sim (sparse) | error |
| --- | ---: | ---: | ---: |
| TPOT mean | 15.07 ms | 15.43 ms | **+2.4 %** |
| TTFT mean | 162.16 ms | 102.87 ms | **−36.6 %** |

Put beside the loaded benchmark, this decomposes the error completely:

| | TPOT | TTFT |
| --- | ---: | ---: |
| unloaded (c1) | +2.4 % | −36.6 % |
| loaded (≈c20) | −3.1 % | −71.3 % |

**Decode is right at both ends of the load curve** — +2.4 % unloaded and −3.1 %
loaded, from 15 ms to 28 ms per token. Two operating points a factor of two apart
in absolute cost, both within 3 %, is independent confirmation that the bundle's
per-layer magnitudes are correct and not fitted to one point.

> **RETRACTED 2026-08-28 (deviations D19).** The queuing half of what follows was
> misattributed. The simulator replays the trace's `arrival_time_ns`, spread over
> 1.78 s; `bench_furiosa_endpoint.py` fires all 20 requests at once under
> `Semaphore(concurrency=64)` and never reads that column. With arrivals matched
> the card-EDF error is **−5.1 %**, not −71.3 %, and there is no evidence of a
> scheduler difference. The −36.6 % unloaded figure below stands — it comes from a
> sparse-arrival run where nothing queues. See
> `experiments/results/rngd_ttft_gap_resolved.md`.

**Prefill splits into two separate errors.** −36.6 % is present with no queuing at
all, so it is a genuine prefill-cost gap: bucket quantisation accounts for ~11 %
of it (`rngd_sim_vs_real_summary.md`), and the rest is server-side work the device
trace does not see — tokenisation, request admission, the HTTP hop. The remaining
stretch from −36.6 % to −71.3 % is **queuing**: reality amplifies TTFT 8.7×
(162 → 1404 ms) between unloaded and loaded, where the simulator amplifies it
3.9× (103 → 403 ms). Upstream's scheduler queues roughly 2.2× less than
furiosa-llm's at the same load.

So the bundle is not the thing to fix for TTFT. The scheduler is, and that is a
Phase 4/5 calibration question rather than a profiling one.

## Open

- **The within-layer split stays inherited** until the compiler exposes
  sub-stages. It does not affect iteration totals but no single dense row should
  be quoted as a vendor measurement of that layer.
- **Mixed prefill+decode steps are absent from the 4D grid.** Every attention
  bucket in the traces is pure prefill or pure decode, so the grid has data only
  on the two axis planes and the simulator's nearest-slice fallback approximates
  the interior. Whether furiosa-llm actually never mixes them cannot be settled
  from these traces — the EDF CSV carries durations, not timestamps, so
  co-occurrence within one forward is not observable.
- **The TTFT gap is resolved, and not the way this file first said** (D19). It was
  the validation harness: matched arrivals give −5.1 %. No scheduler knob is
  implicated and the scheduler comparison this item called for is not needed. What
  remains is a 10–17 % tail under-prediction at p90–p99, plausibly bucket
  quantisation.
- **`max_tp_size` stays 1.** TP *across* cards has never been built or served
  here.
- **This does not license an RNGD P/D claim.** FuriosaAI's llm-d documentation
  states Furiosa-LLM does not support prefill/decode disaggregation, so every
  RNGD P/D result stays simulator-only.

## Reproducing

```bash
# 1. collect: starts and stops `furiosa-llm serve` once per concurrency,
#    with EDF_PROFILER_OUTPUT_PATH / TUC_PROFILE_LEVEL=info / RUST_LOG=span::tuc=info
PYTHONPATH=$PWD /usr/bin/python3 experiments/scripts/rebuild_rngd_bundle_from_edf.py collect \
    --artifact ~/.cache/huggingface/hub/models--furiosa-ai--Llama-3.1-8B-Instruct/snapshots/<id> \
    --card 0 --port 8020 --concurrency 1,2,4,8,16,32 --num-reqs 24 \
    --out outputs/rngd_edf_bundle
# 2. build the §3.7 bundle
PYTHONPATH=$PWD /usr/bin/python3 experiments/scripts/rebuild_rngd_bundle_from_edf.py build \
    --out outputs/rngd_edf_bundle
# 3. import as its own hardware id
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/import_rngd_profile.py \
    --src outputs/rngd_edf_bundle/bundle --hardware RNGD-CARD --edf --card-level \
    --tp 1 --overwrite
# 4. validate against the real run
.venv/bin/python -m serving \
    --cluster-config experiments/configs/clusters/rngd-card-llama31-8b-tp1.json \
    --dtype bfloat16 --block-size 16 --dataset outputs/envcheck/rngd20.jsonl \
    --output outputs/envcheck/rngd_verify_card_edf.csv --run-id edfcard
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/compare_rngd_sim_vs_real.py \
    --sim-csv outputs/envcheck/rngd_verify_card_edf.csv \
    --real-json outputs/rngd_bench/real_tp8.json \
    --out-dir outputs/rngd_bench --prefix rngd-card-edf
```

The six EDF traces total 162 MB and are regenerable, so they are not committed;
the derived audit files (`edf_vs_harness_dense.csv`,
`edf_decode_attention.csv`, `edf_composed.json`) are.
