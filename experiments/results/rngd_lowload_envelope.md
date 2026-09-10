# The RNGD card at low load — and why "tokens/J is unimodal" does not hold here

*`WORK_ORDER_rps_aware.md` rev 2 STEP 3.2–3.3. Measured 2026-09-08 on the NPU node
(`scripts/whichnode.sh`: npu, 3 RNGD cards), card `npu0` / `pci_bdf 0000:03:00.0`,
Llama-3.1-8B bf16, TP=8 inside one card. Same artifact and same dataset as D22.
Raw: `outputs/rngd_envelope_lowload/`. Curve:
`profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml`.*

Three results, one sentence each:

1. **Energy efficiency spans 9.4× between concurrency 1 and 15.6, and essentially
   all of it is throughput** — the card's power moves 1.085× across a 9.5×
   throughput range.
2. **The design's "tokens/J is unimodal in concurrency" is not observed** anywhere
   in the measured range [1, 107.2]; the curve is monotonically increasing, and
   that holds under the most pessimistic power assumption the hardware allows.
3. **A loaded-but-idle card draws 40.0 W**, the same as an empty one, which
   confirms the profile rather than contradicting it.

## The measurement

Two independent processes per point, pool 300 everywhere (≥ 19× the concurrency
even at c16). **Zero failed requests in 3000.** Power is the sustained mean over
the bench window with the per-PE utilisation from the same samples (A5(c)).

| requested | **served** | ratio | tput tok/s | TPOT p50 | TPOT p99 | util % | **power W** | **tok/J** | repeat spread |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **1.00** | 1.000 | 63.1 | 15.71 | 16.00 | 92.1 | 151.1 | **0.418** | 0.28 % |
| 2 | **1.99** | 0.997 | 110.1 | 18.01 | 19.92 | 88.0 | 140.6 | **0.783** | 0.26 % |
| 4 | **3.98** | 0.994 | 200.9 | 19.48 | 21.72 | 86.5 | 139.9 | **1.436** | 0.18 % |
| 8 | **7.88** | 0.985 | 356.8 | 21.86 | 24.34 | 85.3 | 143.2 | **2.492** | 0.57 % |
| 16 | **15.59** | 0.974 | 598.2 | 26.05 | 28.47 | 84.7 | 151.8 | **3.941** | 0.72 % |

Every point cleared A5(b): `served/requested ≥ 0.974` and the pool ≥ 4× the
concurrency, so **no point is pool-bound** and none carries a note. Repeat spread
never approached the 5 % threshold, so no third repeat was needed — and at 1 W
quantisation on a ~150 W reading, 0.72 % is roughly one quantum.

`experiments/figures/rngd_lowload_envelope.png` plots all four panels.

## The seam with D22

The two halves are eight days, two harnesses and two pool sizes apart, and they
meet at c16:

| | served conc | tput tok/s | TPOT |
| --- | ---: | ---: | ---: |
| D22 (2026-08-31, pool 128) | 15.3 | 585.8 | 25.71 avg |
| this run (pool 300), ×2 | **15.59** | **598.2** | 26.05 p50 |

+1.9 % on served concurrency and +2.1 % on throughput. The curve is one curve.

## Power: a shallow U, and utilisation does not explain it

Power falls from 151.1 W at c1 to 139.9 W at c4 and rises to 151.8 W at c15.59 —
**a total span of 8.2 %** where throughput spans 9.5×.

The U is physical. At concurrency 1 every decode step streams the whole model to
produce a single token, so the card is memory-bandwidth-bound and burns 151 W for
63 tok/s. Batching amortises that traffic; the minimum near c4 is where the
memory-bound term stops falling faster than the compute-bound term rises.

**Utilisation is the wrong explanatory variable here.** It falls monotonically
(92.1 → 84.7 %) while power falls and then rises: no monotone function of util
fits, and the correlation is r = +0.24 over a 7.4 pp span. The design's §3 schema
(`piecewise_linear_in_util`) was built on an **ATOM** measurement spanning 36–95 %
utilisation, where it is the right model. For this card the `power_model` added to
`furiosa_rngd_card.yaml` is keyed on served concurrency, with the utilisation kept
beside each point because A5(c) is about provenance, not about the fit.
`deviations.md` **D31**.

## The hypothesis, tested

`docs/rps_aware_planning_design.md` §1 states it as a consequence, not a
measurement:

> **tokens/J is unimodal in concurrency.** At low load idle power dominates the
> denominator; at high load throughput roll-off starves the numerator.

**The low-end half is right in direction and wrong in mechanism, in the direction
that makes it worse.** Low load is indeed far less efficient — 0.418 against
3.941 tok/J, a 9.4× penalty. But it is not idle power dominating: the card is not
mostly idle at c1, it draws **151 W**, within 0.5 % of its draw at c15.59. You do
not pay a 40 W overhead on top of useful work; you pay essentially the full
operating power for a ninth of the output.

**The high-end half is not observed at all.** For a maximum to exist inside the
measured range, power would have to rise faster than throughput somewhere. It does
not, and the bound does not depend on the unmeasured power above c15.59:

| assumed power at conc 107.2 (tput 1473.3 tok/s) | tok/J |
| --- | ---: |
| unchanged from c15.59 — 151.8 W | 9.71 |
| the profile's `active_power`, all 8 PEs — 290.93 W | 5.06 |
| **the highest wattage ever observed on this card — 293 W** | **5.03** |
| *measured at c15.59* | *3.94* |

The card saturates near 291 W — the profile records that the 8th PE adds only
+17.4 W where PEs 1–4 added ~31 W each — so 293 W is an upper bound, not a guess.
**Even at that bound, efficiency at the top of the envelope exceeds efficiency at
c15.59 by 1.28×.** So tokens/J is **monotonically increasing across the entire
measured range**, and no interior optimum exists in [1, 107.2].

**What this does and does not change.** It does not weaken the design's
deliverable: a crossover between two devices requires their curves to cross, not
that either has a peak. It does remove the reasoning that each device has a sweet
spot to be found — for this device on this workload, more load is monotonically
better for energy per token, and the binding constraint on operating point is the
**TPOT SLO**, not efficiency. §4's operating-point solver should be read that way.
Whether a maximum exists above c107 is unknown and this measurement refuses to
guess (`validity.extrapolation: refuse`).

## Idle and standby — the profile is confirmed, not contradicted

Ten idle windows, each 45 s of settling discarded then a 60 s mean, taken with the
model resident (`dram_used_ratio` 0.895 in every one):

**40.0 W, utilisation 0.00 %**, range 38–41 W across all ten.

The profile's `idle_power: 39.35` was measured with **0 PEs loaded**. This shows a
card with the model resident draws the same, which closes a gap rather than moving
a number.

`standby_power: 265.0` is **not** contradicted. It is a 2 s post-load transient —
`standby_duration: 2000000000` ns — and `power_model.py:68` charges it as an excess
over idle for at most that long: `(standby_power − idle_power) × min(latency, 2 s)`.
Every window here begins 45 s after the server reports ready, so it measures a
different thing. An earlier reading of these numbers as a "7× discrepancy" was
wrong and is corrected here.

## tokens/J against RPS — the input E6b needs

Each closed-loop point corresponds to an open-loop arrival rate
`rps = served_conc / mean request latency`:

| served conc | mean latency s | **RPS** | **tok/J** |
| ---: | ---: | ---: | ---: |
| 1.00 | 10.34 | 0.097 | 0.418 |
| 1.99 | 11.80 | 0.169 | 0.783 |
| 3.98 | 12.91 | 0.308 | 1.436 |
| 7.88 | 14.41 | 0.547 | 2.492 |
| 15.59 | 16.99 | 0.917 | 3.941 |

These are **per card**. The RPS figures are low because this workload's requests
are long — 732 input and 632 output tokens at p50 — so one card at c15.59 serves
under 1 rps of it. A fleet's rps scales with the card count, which is what E6a
varies.

**This is the measured half of E6b.** The A40 side is simulation only, so any
crossover sentence must read "RNGD **measured** against A40 **simulated**", per the
work order.

## What is not established

- **Power above served concurrency 15.59.** Never sampled. The envelope's
  `validity.power_valid_range` says so and D22's four points keep `power_w: null`
  rather than being back-filled (A2).
- **Anything about a second card, or two cards together.** `additive_across_units:
  false` is carried from the existing profile, not re-measured here.
- **TTFT as an open-loop quantity.** The whole pool is fired at once (D19), so
  `ttft_p99` here is closed-loop and must not be compared with a Poisson p99.
  Throughput, TPOT and power are the transferable axes.
