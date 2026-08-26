# RNGD: card-as-device vs PE-as-device, and what the gap measures

*Measured 2026-08-26. Settles a modelling question with the real furiosa-llm
benchmark instead of an argument, and the answer is not the one the reasoning
predicted.*

## The question

An accelerator in HeteroPilot's model is whatever the simulator counts when it
picks a `tp<N>` bundle. For RNGD there are two defensible choices:

- **PE-as-device.** One accelerator = one PE. Mirrors `furiosa-llm build -tp`,
  which counts PEs and defaults to 8. A card is a TP-8 island of 8 accelerators,
  and ASTRA-Sim prices the intra-card collectives.
- **Card-as-device.** One accelerator = one whole card running TP=8 internally.
  The card is a `tp1` instance; the bundle's `tp1/` is the measured per-PE `tp8/`,
  because the 8 PEs each do 1/8 of every layer in parallel so the card's per-layer
  latency equals one PE's sharded latency.

The case for card-as-device was strong, and none of it turned out to be wrong:

| same 8 PEs, Llama-3.1-8B bf16 | replicas | total KV tokens |
| --- | ---: | ---: |
| **TP=8** | 1 | **246,079** |
| TP=4 × DP=2 | 2 | 123,550 |
| TP=2 × DP=4 | — | does not fit (−1.86 GiB/PE) |

TP=8 shards the weights **once** across the 8 PEs (1.87 GiB each) rather than
replicating them per group (3.74 GiB at TP=4), so it holds twice the KV on
identical silicon. So TP<8 does not merely idle compute — it throws away half the
memory that makes decode possible. TP=8 is also the only configuration that has
ever been served here: `furiosa-llm build -tp` defaults to 8 and the vendor's
prebuilt artifact is `tensor_parallel_size: 8`.

Card-as-device also fixes three real annoyances: a 47.5 GB device has KV headroom
comparable to an A40, which removes the decode-KV exhaustion that crashed the P/D
simulations; 4 cards give TP candidates {1,2,4} that overlap an A40 island's
{1,2,4}, so cross-vendor P/D becomes expressible without the PCIe-bridging
fixture hack of deviations D16; and the power block takes the measured card totals
verbatim, with no per-PE split and no `base_node_power` hand-carrying.

## The answer: card-as-device is much less accurate

Same 20 sharegpt requests, same TP=8 hardware, against the real furiosa-llm run
(`experiments/results/rngd_sim_vs_real_summary.md`):

| model | TTFT mean | error | TPOT mean | error |
| --- | ---: | ---: | ---: | ---: |
| **real** (furiosa-llm, TP=8) | 1404.1 ms | — | **28.4 ms** | — |
| per-PE (8 accelerators, tp8) | 946.9 ms | −32.6 % | 35.7 ms | **+25.7 %** |
| card (1 accelerator, tp1) | 310.0 ms | −77.9 % | 15.5 ms | **−45.5 %** |

The truth sits between the two, and the per-PE model is both closer and
conservative. Card-as-device under-predicts decode by a factor of two.

**Why.** Dropping the TP group hands the model a free 8× parallel speedup with no
communication. The per-layer scaling in the measured bundle is nearly perfect, so
there is a lot of speedup to hand over — summed across a decode step, tp1 → tp8 is
6.94×:

| layer (1 token) | tp1 | tp8 | ratio |
| --- | ---: | ---: | ---: |
| qkv_proj | 259.9 µs | 35.7 µs | 7.29 |
| o_proj | 174.3 | 21.7 | 8.04 |
| gate_up_proj | 1197.8 | 153.9 | 7.78 |
| down_proj | 507.4 | 79.1 | 6.42 |
| **sum (all dense layers)** | **2160.4** | **311.4** | **6.94** |

A `tp1` instance has no TP group, so the simulator charges nothing for the 64
all-reduces per token that make that scaling possible.

## What the residual measures: RNGD decode is collective-latency bound

The gap is not noise, and it is the useful part of a failed experiment:

```
28.4 ms (real) − 15.5 ms (card model) = 12.92 ms per token step unaccounted
32 layers × 2 all-reduces            = 64 collectives per token
                                     → ~202 µs per collective
all-reduce payload = hidden 4096 × bf16 = 8 KiB
```

**8 KiB in 202 µs is latency, not bandwidth.** So RNGD decode at TP=8 is dominated
by intra-card collective latency — about 45 % of the real TPOT — not by compute.
That reframes the part: its measured strength is bandwidth per watt (5.84 GB/s/W
at card level, 3.1× an A40), but at TP=8 that strength is spent behind 64 small
synchronisations per token.

Treat 202 µs as an **upper bound**. The residual also contains the
bucket-quantisation (+10.9 % of prefill work) and eager-scheduler effects already
quantified in the sim-vs-real write-up, so the collective share is at most this.
It is still the first empirical handle on the `ONPACKAGE` fabric that deviations
D3/D16 leave as `placeholder`, and it is worth far more than the abstraction it
came from.

## What was kept

- **PE-as-device stays the default** (`profiles/accelerators/furiosa_rngd.yaml`).
  Keeping the 8 PEs as accelerators is what lets ASTRA-Sim price the dominant term.
- **`furiosa_rngd_card.yaml` is retained but marked not-the-default-path**, with
  this validation in its header. It is the record of why the abstraction was
  rejected and the derivation of the collective cost.
- **"TP=8 is a must" is right, but belongs in the objective, not the abstraction.**
  With collectives correctly charged, TP=8 wins on its merits — 2× the KV and
  ~7× the dense-layer throughput of tp1 — so the optimizer selects it without
  being told to. Hard-coding TP=8 by collapsing the card into one device buys a
  preference the search would have reached anyway, and pays for it with a 45 %
  decode error. Forbidding TP<8 outright would also be wrong in general: for a
  model small enough that weights fit at TP=4, tp4×dp2 is a legitimate
  configuration, and only the optimizer knows which model it is looking at.

## Open, and where it goes next

- **Measure the on-package all-reduce directly** instead of inferring it. That
  turns the ~202 µs upper bound into a number, and gives `ONPACKAGE` a real
  bandwidth/latency pair instead of a placeholder. A multi-PE collective
  microbenchmark through `furiosa.torch` is the obvious route; nothing about it
  needs a GPU, so it can be done on this machine.
- **The per-PE model's remaining +25.7 % TPOT error** is then attributable: if the
  measured collective cost explains it, the bundle is sound and the residual is
  scheduler and buckets. If it over-explains, ASTRA-Sim's collective model is
  mis-parameterised for this fabric.
- **None of this licenses an RNGD P/D deployment claim.** FuriosaAI's own llm-d
  documentation states Furiosa-LLM does not support prefill/decode disaggregation,
  so every RNGD P/D result remains simulator-only regardless of how the accelerator
  is modelled.
