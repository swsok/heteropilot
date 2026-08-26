# Does heterogeneous RNGD+GPU P/D ever pay? Sweeping the TTFT SLO

*Simulated 2026-08-26. 300 requests, seed 42, `examples/service_specs/llama31-8b.yaml`
on `experiments/configs/clusters/pd-rngd-gpu.yaml` (2 RNGD cards + 2 A40 islands
of 4). 496 candidates generated, 492 evaluated, at each of 8 TTFT p99 SLO points.
The SLO is swept because the answer turns out to depend entirely on it.*

## The short answer

**Three regimes, and heterogeneous P/D wins in none of them.**

| TTFT p99 SLO | recommended | arch | acc | tok/J | goodput (rps) | p99 TTFT | p99 TPOT | avg W |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 64 s | `agg[furiosa:tp8]` | aggregated | 8 | **4.956** | 3.75 | 29547 | 48.4 | 481 |
| ≤ 32 s | `agg[furiosa:tp8]` | aggregated | 8 | **4.956** | 3.75 | 29547 | 48.4 | 481 |
| ≤ 16 s | `agg[cuda:tp4]` | aggregated | 4 | 2.963 | **5.92** | 2972 | 49.4 | 1283 |
| ≤ 8 s | `agg[cuda:tp4]` | aggregated | 4 | 2.963 | **5.92** | 2972 | 49.4 | 1283 |
| ≤ 4 s | `agg[cuda:tp4]` | aggregated | 4 | 2.963 | **5.92** | 2972 | 49.4 | 1283 |
| ≤ 2 s | `P[cuda:tp4] D[cuda:tp4]` | **pd_split** | 8 | 2.206 | 5.78 | **372** | 37.3 | 1675 |
| ≤ 1 s | `P[cuda:tp4] D[cuda:tp4]` | pd_split | 8 | 2.206 | 5.78 | **372** | 37.3 | 1675 |
| ≤ 0.5 s | `P[cuda:tp4] D[cuda:tp4]` | pd_split | 8 | 2.206 | 5.78 | **372** | 37.3 | 1675 |

1. **Loose TTFT (≥ 32 s): RNGD wins on energy, 1.67×.** 4.956 tok/J against 2.963,
   at 481 W against 1283 W — but 63 % of the goodput and a 29.5 s p99 TTFT. This is
   the regime where RNGD's measured bandwidth-per-watt advantage actually shows up
   in an end-to-end metric.
2. **Middle (4–16 s): plain A40, aggregated.** RNGD cannot meet the TTFT at all and
   P/D is not needed yet.
3. **Tight (≤ 2 s): P/D disaggregation becomes the answer — and it is homogeneous.**
   Splitting A40 prefill from A40 decode cuts p99 TTFT 2972 → 372 ms (**8×**) for
   25 % of the energy efficiency (2.963 → 2.206 tok/J) and 2 % of the goodput. That
   is a real and clean demonstration of what P/D buys, on the one arm of this
   fixture whose profile is trustworthy to ~2 %.

**So P/D pays, and heterogeneity does not.** But the second half of that sentence
needs three caveats, and they are large enough that it should not be quoted
without them.

## Why "heterogeneous P/D does not pay" is NOT a supported conclusion

**Caveat 1 — only one of the two cross-vendor directions produced a result, and it
is the wrong one.** The enumerator emitted both, at the only shared TP degree
D14 allows (tp4 on both sides):

| cross-vendor combo | outcome |
| --- | --- |
| RNGD tp4 prefill → A40 tp4 decode | simulated, 6 configs |
| **A40 tp4 prefill → RNGD tp4 decode** | **crashed, all 6 configs** |

All six configurations of the GPU-prefill/NPU-decode direction died in
`serving/core/memory_model.py`:

```
RuntimeError: [MemoryModel] [node_id=0,inst=0] NPU:
tried to load 39.00MB but only 3.49MB is available.
```

RNGD decode at tp4-dp1 replicates the 14 GB of weights across 4 PEs of 6.25 GB
and has essentially no KV headroom left. So **the promising direction was never
evaluated — it was unsimulatable**, and its absence from the ranking is not
evidence against it.

**Caveat 2 — the direction that *was* simulated is the backwards one, and it is
dominated.** Best cross-vendor result (RNGD prefill → A40 decode):

| | tok/J | goodput | p99 TTFT | p99 TPOT |
| --- | ---: | ---: | ---: | ---: |
| cross-vendor P/D (best) | 2.234 | 4.57 | **18466 ms** | **28.2 ms** |
| `agg[cuda:tp4]` | 2.963 | 5.92 | 2972 | 49.4 |
| `agg[furiosa:tp8]` | 4.956 | 3.75 | 29547 | 48.4 |

It is strictly worse than aggregated A40 on energy *and* TTFT *and* goodput — a
dominated point, not a trade-off. And the reason is legible: it has **the best
p99 TPOT of any family** (28.2 ms, against 49.4 for aggregated A40), so pairing
the two vendors genuinely does help *decode*; it is ruined by putting prefill on
RNGD at tp4, which is a poor prefill engine. The pairing is inverted.

**Caveat 3 — the shape the literature recommends was never enumerated.** D14's
uniform-TP constraint requires `tp_p == tp_d`, so `A40 tp4 prefill + RNGD tp8
decode` — big TP on the memory-bound decode phase, which is exactly what NVIDIA
Dynamo recommends and AWS Neuron documents — cannot be expressed at all. RNGD at
tp8 holds 246,079 KV tokens against 61,775 at tp4-dp1, so the constraint forces
away a 4× KV advantage precisely on the side that needs it.

Taken together: **what this sweep establishes is that the heterogeneous P/D
configurations D14 permits and the per-PE profile can simulate do not pay. It
does not establish anything about the configuration that should work.**

## What to run next, and why it is now possible

The blocker in caveats 1 and 3 is the same thing: an accelerator being one PE of
6.25 GB. The EDF-rebuilt **card-as-device** profile
(`experiments/results/rngd_edf_bundle_notes.md`) removes it:

- one accelerator = one 47.5 GB card at tp1, so RNGD decode has A40-comparable KV
  headroom and the crash in caveat 1 disappears;
- TP candidates become {1}, which overlaps the A40 island's {1,2,4} at tp1, so
  cross-vendor P/D is expressible without the PCIe-bridging fixture hack of D16;
- and its decode prediction is accurate to −3.1 % against the real furiosa-llm
  run, where the per-PE profile used here is +25.7 % — i.e. **this sweep
  over-charges RNGD decode by about a quarter**, which is the direction that would
  hurt an RNGD-decode combo most.

That last point matters for how much weight to put on the table above: the arm
of this fixture that a heterogeneous P/D win would depend on is the one measured
with the least accurate profile, and the error is pessimistic.

A re-run on `experiments/configs/clusters/pd-rngd-gpu-card.yaml` with the same 8
SLO points is the direct test.

## Provenance and what the numbers do not carry

- **A40 numbers are measured on a different machine by a different method** (vLLM's
  own layerwise profiler, ~2 % sim-vs-real). There is no NVIDIA GPU on this host.
- **RNGD numbers here use the per-PE harness bundle**, which is +25.7 % on TPOT and
  −32.6 % on TTFT against the real run. A GPU-vs-NPU efficiency ratio read off
  this fixture carries both gaps.
- **The prefill→decode fabric is a placeholder**: 35 GB/s is the *measured NPU leg*
  (host→PE, 8 parallel streams) used as an upper bound on a GPU→host→NPU path
  whose GPU leg cannot be measured on this host. `docs/HANDOVER_A40.md` carries
  that open item.
- **furiosa-llm does not support P/D disaggregation at all today** (FuriosaAI's own
  llm-d documentation), so every RNGD P/D row here is simulator-only regardless of
  how well it scores.
- The `mix` family's best-tok/J entry shows `goodput 0.00` at a 748 ms p99 TTFT:
  those rows fail the TPOT half of the SLO (75 ms), which is why they never win.

## Reproducing

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/pd_slo_sweep.py \
    --service examples/service_specs/llama31-8b.yaml \
    --cluster experiments/configs/clusters/pd-rngd-gpu.yaml \
    --ttft-ms 500,1000,2000,4000,8000,16000,32000,64000 \
    --num-requests 300 --seed 42 --workers 32 --output-dir outputs/.hp-pd-slo
```

Points run loosest→tightest so every tighter point is an envelope-cache hit; the
whole sweep is one candidate evaluation pass (492 simulations) plus seven
re-rankings. Raw results: `outputs/.hp-pd-slo/pd_slo_sweep.json`; per-candidate
metrics stay in `outputs/.hp-pd-slo/cache/` (223 entries, untracked and
regenerable).
