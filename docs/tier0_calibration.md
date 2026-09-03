# Tier 0 calibration — accuracy measured in THIS environment (STEP 8)

The tiered-profile work order requires Tier 0's error to be quantified
against our own measured bundles, not quoted from literature. This document
records the STEP 8 experiment: the design-choice bake-off (bytes / attention
combination variants), the fitted per-family efficiencies, and the residual
error of the adopted configuration.

- Date: 2026-09-02 · repo commit at fit time: `250bb98c`
- Model: `meta-llama/Llama-3.1-8B` bf16, TP=1
- Measured references: `profiler/perf/A40/.../bf16/tp1` (this fork's own
  profiling run) and `profiler/perf/RTXPRO6000/.../bf16/tp1` (the
  upstream-shipped bundle)
- Method: for each variant, per-family efficiencies were fitted as
  `median(theoretical_lower_bound / measured)` on the SAME bundle
  (`python -m profiler.synth fit-efficiency`), a mirror-key Tier 0 bundle
  was emitted with those efficiencies, and `python -m profiler.synth diff`
  compared it key-by-key. Self-fit numbers are therefore *optimistic*
  (train = test); hold-out behavior is STEP 9's business.

## Variant bake-off (overall MAPE, %, lower is better)

| variant | bytes_moved | attention combine | A40 | RTXPRO6000 |
| --- | --- | --- | --- | --- |
| V1 | sum(weights, activations) | max (fused roofline) | 40.3 | 37.1 |
| V2 | max(weights, activations) | max | 40.3 | 37.1 |
| **V3** | **sum** | **sum (phase times added)** | **38.9** | **33.0** |
| V4 | max | sum | 38.9 | 33.0 |

**Adopted default: V3 (`bytes_mode=sum`, `attn_mode=sum`).**

- The attention combine mode is the only decision that matters at this grid:
  `sum` beats `max` by 1.4pp (A40) and 4.1pp (RTXPRO6000) on attention MAPE,
  because mixed chunked-prefill + decode steps behave like two serialized
  phases rather than one perfectly fused kernel.
- The bytes variants are indistinguishable to the median (weights dominate
  every GEMM at this grid's token counts, so `sum ~= max`); `sum` is kept as
  it is the physically true minimum traffic (weights AND activations must
  move) and marginally better on dense MAPE (24.3 vs 24.5 on A40).

## Fitted efficiencies (V3, self-fit, TP=1)

Committed as `profiles/accelerators/{a40,rtxpro6000}.efficiency.yaml` with
full provenance (`derived_from`). Merging into the profile `datasheet:` is a
human decision; the profiles' `flops_efficiency` / `mem_efficiency` remain
empty until then.

| family | A40 (n) | RTXPRO6000 (n) |
| --- | --- | --- |
| gemm | 0.747 (648) | 0.710 (648) |
| elementwise | 0.532 (648) | **1.007** (648) — bound violation, see below |
| gather | 0.796 (152) | 0.690 (152) |
| attention | 0.262 (8643) | 0.354 (19364) |

**RTXPRO6000 elementwise bound violation (+0.7%).** The fitted value
exceeds 1.0, i.e. the measured elementwise kernels marginally beat the
1792 GB/s DRAM roofline. GB202 carries a 128 MB L2 (vs GA102's 6 MB), so
elementwise working sets at most grid sizes are L2-resident and the DRAM
bandwidth is not a true lower bound there. Recorded unclamped in
`rtxpro6000.efficiency.yaml` under `bound_violations`; that family cannot be
merged into a `Datasheet` (which enforces (0, 1]) without an explicit
decision. The same effect capped at +5.3% on the A40 (STEP 5's 6% test
tolerance).

Cross-device transfer check: attention efficiency differs 1.35x between the
two GPUs (0.26 vs 0.35) — consistent with the literature's warning that
attention efficiencies do not transfer across devices, and the reason
STEP 9 concentrates the anchor budget on attention.

## Residual error of the adopted V3 (self-fit)

| group | A40 MAPE% / med.ratio / Spearman | RTXPRO6000 MAPE% / med.ratio / Spearman |
| --- | --- | --- |
| dense.csv | 24.3 / 1.002 / 0.990 | 32.0 / 1.000 / 0.988 |
| per_sequence.csv | 42.3 / 0.629 / 0.997 | 50.9 / 0.671 / 0.991 |
| attention.csv | 41.1 / 1.000 / 0.978 | 33.0 / 1.000 / 0.972 |
| family gemm | 9.5 | 11.3 |
| family elementwise | 41.7 | 50.8 |
| family gather | 22.9 | 50.0 |
| family attention | 41.1 | 33.0 |
| **overall** | **38.9** | **33.0** |

Reading:

- **GEMM transfers well** (9.5–11.3% MAPE): a single scalar efficiency
  nearly explains dense projections. This is where a roofline is a good
  model.
- **A single scalar per family is NOT enough for attention/elementwise**
  (33–51% MAPE): the median ratio is pinned at 1.0 by construction but the
  spread is wide, i.e. the efficiency depends on the operating point (small
  vs large keys). Rank correlations stay high (0.97+ attention, 0.92+
  elementwise on A40), so ordering survives — exactly the situation the
  work order's Tier 1 (piecewise per-family scaling, STEP 9) is designed
  to fix.
- The per-sequence median ratio (~0.63–0.67) shows the sampler/lm_head
  coefficient drafts under-cost per-sequence work by ~1.5x; the family
  median is dominated by the (many more) dense elementwise rows, so the
  per-sequence rows sit below 1.0 after derating.
- Overall MAPE ~33–39% is within the work order's expectation band for
  Tier 0 (risk rule: >50% would still not be "failure", but restricts use
  to coarse candidate screening). Whether this error changes *planning
  decisions* is measured by experiment E1 (STEP 11), not here.

## Reproduction

```bash
# fit (per hardware)
python -m profiler.synth fit-efficiency \
  --accelerator profiles/accelerators/a40.yaml \
  --measured profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 \
  --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1 \
  --out profiles/accelerators/a40.efficiency.yaml

# mirror-key emit with the fitted efficiencies merged into a working copy
# of the datasheet, then:
python -m profiler.synth diff \
  --measured profiler/perf/A40/meta-llama/Llama-3.1-8B/bf16 \
  --synth <emitted A40-t0 variant root> --tp 1 --format table
```
