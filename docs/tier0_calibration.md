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

## Tier 1 hold-out curve (STEP 9)

Anchor subsets were drawn uniformly at random (seed 42) from the measured
A40 tp1 rows; the ScalingTable (`python -m profiler.synth calibrate`) was
fitted on the anchors and evaluated on the REMAINING rows. Tier 0 here is
the V3 backend with the self-fitted efficiencies above; Tier 1 multiplies
it by the fitted per-family scaling (scalar + log-feature piecewise bins).

| anchor share | n anchors | Tier 0 MAPE % (hold-out) | Tier 1 MAPE % (hold-out) |
| --- | --- | --- | --- |
| 1% | 100 | 38.9 | 33.1 |
| 2% | 201 | 38.8 | 30.4 |
| 5% | 504 | 38.9 | 29.7 |
| 10% | 1009 | 38.8 | 27.8 |
| 20% | 2018 | 38.6 | 27.3 |

Findings:

- **Tier 1 improves on Tier 0 at every budget**, with most of the gain
  already at 1-2% (~100-200 measured points): 38.9% -> 33.1%/30.4%. Returns
  flatten past 10%.
- **A stability guard was required.** The first fit let a family scalar be
  computed from as few as one anchor; a single launch-floor `embedding` key
  fitted `gather` at 36.3x and drove hold-out MAPE to 77-84% at the 1-2%
  budgets. `fit_from_anchors` therefore treats a family with fewer than
  `min_family_anchors` (default 8) anchors as having NO data - identity
  scale (A2: too little data is no data, never a confident multiplier).
- Piecewise bins (8 log-spaced over the Tier 0-time feature, >= 4 anchors
  per bin, family scalar as fallback) carry most of the improvement beyond
  the scalar: at 5% share, scalar-only reaches 39.1% while piecewise
  reaches 29.7%.

## Validation experiments E1-E4 (STEP 11)

Harnesses live in `experiments/tier_validation/`; JSON+table reports (with
§3.8 provenance) under `outputs/tier_validation/e<N>/`. Results below were
produced 2026-09-02 on this repo's committed measured bundles.

### E2 — where should a measurement budget go? (`e2_budget_pareto`)

A40 / Llama-3.1-8B, budget 200 anchors, hold-out MAPE vs the measured bundle:

| condition | measured points | hold-out MAPE |
| --- | --- | --- |
| A: Tier 0, no anchors | 0 | 38.9% |
| B: 200 anchors, ALL attention | 200 | **29.5%** |
| C: 200 anchors, uniform spread | 200 | 42.9% |
| D: fully measured | 10 091 | 0% |

Attention-focused anchoring buys most of the Tier 1 gain (consistent with
the KernelSight-LM observation and the hold-out curve above). The uniform
condition actually LOST accuracy: spreading 200 anchors across five
families leaves each non-attention family with a handful of deterministic
picks whose launch-floor keys skew the fitted scalars - the same failure
mode the `min_family_anchors` guard bounds but cannot eliminate at
tiny per-family counts.

### E3 — shape overlap across bundles (`e3_shape_overlap`)

Raw keys are model-namespaced (a raw `(layer, tokens)` key only names the
same GEMM within one model); normalization reduces every row to its
physical work signature (family, FLOPs, bytes).

- **Same model, different hardware: 100% key overlap** (identical grids) -
  the grid transfers, the measurements do not.
- **Different models on one hardware: near-zero shape reuse** - normalized
  overlap is 0.8% (Qwen3-32B vs Llama-3.1-8B), 2.6% (Qwen3-30B-A3B vs
  Qwen3-32B), 16% max on any cross-model pair. A shape cache across THIS
  model catalog would save almost nothing (vs Dooly's 56.4%, which counts
  intra-workload reuse) - the S2 follow-up is not currently worth building.

### E4 — Ascend datasheet sensitivity (`e4_sensitivity`)

hetero-gpu-ascend x qwen3-32b, each of {peak_tflops, memory_bandwidth_gbps,
flops_efficiency, mem_efficiency} swept +-30% (13 steps, datasheet-driven
analytical predictor): **the recommended plan never flips** - the RTXPRO6000
plan dominates across the whole sweep, so the secondary-source uncertainty
in the Ascend datasheet does not change this cluster's decision. (A cluster
where the Ascend island is the marginal choice would need the sweep rerun.)

### E1 — plan agreement (`e1_plan_agreement`)

Four legs per condition (greedy proxy / Tier 0 + sim / Tier 2 + sim = truth
/ oracle) on the A40 TP-sweep cluster x {llama31-8b, llama31-8b-light};
metrics: top-1 agreement, top-3 containment, chosen-plan objective error
under the truth leg's scores, Kendall tau. See
`outputs/tier_validation/e1/` for the report produced by the full-simulation
run (`--num-requests 20 --workers 4`).

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
