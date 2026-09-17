# RNGD — which compiled path the runtime actually takes, by concurrency

**The question.** `d6ae6a43` compiles 47 pipelines: pipeline 0 is `kernelwise` (a composable IR that
assembles a step from per-layer kernels) and pipelines 1–46 are `composed` — one fully fused graph
per decode bucket (`experiments/results/rngd_artifact_buckets.md`). Both can execute the same step.
Which one runs decides what an execution model has to reproduce, so it has to be measured, not
assumed.

**The measurement.** `furiosa-llm` logs `Wire pipeline hit rate` on every metrics line
(`furiosa_llm/server/metrics.py:98-99`), the fraction of steps served by a `composed` pipeline.
Tabulated from the **committed** serve logs of the two envelope campaigns with
`experiments/scripts/serve_log_hit_rates.py`; no new hardware run was needed.

```bash
python experiments/scripts/serve_log_hit_rates.py --markdown \
    outputs/rngd_envelope_lowload/serve_*.log outputs/rngd_envelope/edf/serve_*.log
```

## Result

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wire (composed) hit %, last | 99.8 | 47.5 | 12.0 | 1.8 | 0.5 | 0.0 | 0.0 | 0.0 |
| repeat run (`_r1`) | 99.8 | 47.5 | 12.1 | 1.8 | 0.4 | — | — | — |
| max over the run | 99.8 | 53.3 | 16.0 | 6.4 | 0.5 | 0.0 | 0.0 | 0.0 |

Repeats agree within 0.2 pp. c1–c16 are `outputs/rngd_envelope_lowload/` (two runs each), c16–c128
are `outputs/rngd_envelope/edf/` (one run each); the two c16 rows come from different campaigns and
agree (0.5 vs 0.0 last, both ≈ 0).

**Conclusion — established.** At every load that matters to planning (c8 and above) the runtime is on
the **kernelwise** path; `composed` is effectively a c1-only path. This is consistent with D17's
per-layer EDF traces at c16–c32 being per-layer costs at all: a `composed` step has no separate
`Attention` or `Tokenwise` stage to trace.

## Two statistics that must not be averaged together

The same tabulation prints a prefix-cache column for both stacks, and they are not the same
quantity:

- **furiosa-llm** prints `DpMetrics.prefix_cache_hit_rate` and `wire_hit_rate` straight from the
  native runtime. The throughput fields on the same line are interval diffs; these two are **not** —
  they are gauges the Rust side maintains, and the Python layer documents no window for them.
- **vLLM** prints `CachingMetrics.hit_rate` (`vllm/v1/metrics/loggers.py:259` → `stats.py:107`), an
  explicit sliding window over the most recent `max_recent_requests` requests (default 1000).

So the script reports first / last / min / max and never a mean, and the table above quotes `last`.
Within one run the per-line values are not independent samples of a rate either.

**Cross-check.** Prefix cache hit rate settles at 7.6–9.0 % across all runs, against
`prefix_share_ratio: 0.07` in `examples/service_specs/llama31-8b.yaml` — the spec's workload
assumption and the vendor runtime's own counter agree.

**Noted for STEP B.1 P5.** RNGD KV cache usage reaches 99.6–100.0 % in every run, including c1
where one sequence is in flight. Whether that reflects bucket-granular reservation (the `bs=1`
bucket reserving its full 131072-token budget) or only how the gauge is defined is **open**; it is
the open investigation item under P5, and nothing here settles it.

## Why the hit rate collapses — open, two hypotheses

- **H-a.** The running batch size fluctuates, so the batch rarely matches a compiled `bs` exactly.
- **H-b.** A `composed` pipeline is compiled for **one** `(bs, attention_size)` bucket, so it can
  serve a step only when every sequence in the batch shares that KV bucket. Mixed KV falls back to
  kernelwise, where attention can be issued per group.

H-b fits the numbers better: 47.5 % at c2 is about the chance that two sequences share a bucket, and
12.0 % at c4 about the chance that four do. H-a and H-b are not mutually exclusive.

**STEP C.1 decides it** (~20 min on the NPU node): one c4 run on a synthetic workload where every
prompt is the same length. ~100 % ⇒ H-b; still ~12 % ⇒ H-a. Then c3 on the same workload: ~0 % ⇒ an
exact batch-size match is required, ~100 % ⇒ the batch is padded up. This file gets the answer
appended; until then the rule is **undecided**.

Priority is low for the spike's own conclusion: the c1 path is left unchanged in the prototype
(STEP B.1 P4), so neither hypothesis moves the decomposition.

---

## Resolved 2026-09-17 — H-b, and the batch is padded (E-N6, STEP C.1)

**Measured on `npu2` (PCI `45:00.0`), not on `npu0`.** `npu0` — the card every
committed RNGD measurement was taken on — was held throughout by another tenant's
`rngd_pd.serving.cluster --chip 0` pod (`docs/nodes/npu.md`). The conclusion below
is about the *artifact's* pipeline-selection rule rather than about the card, so
it carries; the hit-rate numbers themselves are `npu2`'s.

Same artifact `d6ae6a43`, same `furiosa-llm`, one card at the vendor default
`tp=8`. The only thing changed from the sharegpt campaign is the workload:
`workloads/fixedlen-512in-128out-64.jsonl`, every prompt exactly 512 tokens and
every completion 128, so every sequence's KV stays inside `(0, 1024]` — one decode
attention bucket — for the whole run.

| workload | concurrency | wire (composed) hit rate |
| --- | ---: | ---: |
| sharegpt, variable length | 4 | **12.0 %** |
| fixed 512 / 128 | 4 | **97.5 %** |
| fixed 512 / 128 | 3 | **96.0 – 98.1 %** |

**H-b, not H-a.** Holding batch size constant and removing only the KV diversity
takes the hit rate from 12.0 % to 97.5 % at the same concurrency on the same card.
Under H-a — the running batch size rarely matching a compiled `bs` — changing the
prompt-length distribution would have changed nothing. A `composed` pipeline is
compiled for one `(batch_size, attention_size)` pair and can serve a step only
when every sequence in the batch shares that attention bucket; a mixed-KV batch
falls back to `kernelwise`, where attention can be issued per group.

**The batch size is padded up, not matched exactly.** Three is not a compiled
decode batch size — the 46 composed decode plans use `bs ∈ {1, 2, 4, 8, 16, 32,
64, 128, 256}` — yet c3 holds 96–98 %. An exact-match rule would have read ~0 %.
So the selection rule is: round the batch up to the next compiled `bs`, then
require every sequence to share that `bs`'s attention bucket.

**What this does and does not change.** It settles the c1-only behaviour of the
`composed` path recorded above: `composed` is not "batch-1 only", it is
"single-KV-bucket only", and sharegpt simply never presents such a batch above
c1. It does **not** move the spike's decomposition — the prototype leaves the c1
path alone (STEP B.1 P4) and every load the planner cares about is still on the
kernelwise path, because real traffic has mixed KV.

It does sharpen one thing for STEP C.3. The runtime demonstrably *can* run a
4-sequence batch as one fused execution when the KV agrees. So whatever caps D17's
measured attention executions per layer near three is not a per-sequence cost —
the hardware is willing to do a whole batch in one go. C.3 should look for a cap
on the number of distinct *groups*, not on the batch.
