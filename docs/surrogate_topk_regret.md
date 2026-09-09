# The surrogate top-K cannot be used at K=20 — measured on three fixtures

*`WORK_ORDER_rps_aware.md` STEP 4, pulled forward because STEP 1's E6a budget
depends on it. Measured 2026-09-08 on the NPU node, `main` = `2c373b9` + D27.
Replay only — no new simulation. Artifacts: `outputs/exp_surrogate/`.*

STEP 1 found `--top-k 20` turning a FEASIBLE plan INFEASIBLE on `pd-rngd-gpu`.
`WORK_ORDER_rps_aware.md` §E6a plans to cut a 218 h sweep with `--top-k 20`, citing
*"§4.7의 regret 0 근거"*. **That basis does not hold, and the obvious repair does not
hold either.** Both statements below are measurements, not arguments.

## What the shipped surrogate actually optimises

`AnalyticalRooflineRanker` orders by `greedy.estimate`'s proxy tok/J. That quantity
is **algebraically invariant to TP and DP**:

```
throughput = Σ active/step_s × dp_replicas      step_s = (W/tp + a·K/tp) / BW
power      = Σ active_power × tp × dp
```

Both numerator and denominator scale with `tp · dp`, so the ratio cancels exactly.
Measured on `pd-rngd-gpu`, A40 aggregated candidates:

| `max_num_seqs` | distinct tp/dp configs | proxy tok/J spread |
| ---: | ---: | ---: |
| 32 | 7 | **0.000463** (4.6545 .. 4.6550) |
| 128 | 7 | **0.001848** (18.6035 .. 18.6053) |
| 256 | 7 | 14.94 — but only because tp1 is KV-capacity-clipped; tp2 and tp4 differ by **0.0025** |

`dp1`, `dp2`, `dp3` and `dp4` are identical to six decimal places. **The ranker sees
the accelerator and `max_num_seqs`, and is blind to the parallelism configuration** —
which on this fixture is the axis that decides feasibility (`tp4-dp1` meets the 50 ms
TPOT SLO at 49.40 ms; `tp2-dp2` misses at 53.47 ms).

The one term in `greedy.estimate` that *does* vary with parallelism is
`roofline_tpot_ms` (5.78 ms for tp4-dp1 against 11.56 ms for tp2). It is used only to
set a binary `likely_infeasible` flag — **which fired for 0 of 324 candidates**,
because the floor underestimates simulated p99 TPOT by a median of **2.92×** (range
1.77–11.26×, n=180). A floor of 11.56 ms against a 50 ms SLO looks safe; the truth is
53 ms. So the flag is dead code here and the ordering turns on tok/J differences in
the fifth significant digit.

## Regret, measured on three corpora

`experiments/scripts/exp_surrogate.py --cache-dir` replays a past sweep's
`EnvelopeCache` — hours of simulation already paid for. A candidate absent from the
corpus is **kept in the ranking and yields no plan**, because the cache stores only
`result.ok`: an absent candidate is one the sweep simulated and got nothing from. It
consumed a top-K slot in the real run, and dropping it would delete exactly the
high-ranked candidates that deliver nothing.

That the replay **reproduces the observed failure** — `roofline` false-infeasible at
K=20 on `pd-rngd-gpu`, matching the real `--top-k 20` run — is what licenses the rest
of this table.

| fixture | corpus | N | ranker | K=10 | K=20 | K=30 | K=50 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `pd-rngd-gpu` (no P/D) | `perf/topk/cache_full` | 324 | **roofline** | **infeas** | **infeas** | 0.000 | 0.000 |
| | | | `floor` | 0.142 | **0.000** | 0.000 | 0.000 |
| `pd-rngd-gpu` (P/D) | `.hp-pd-slo/cache` | 492 | **roofline** | **infeas** | **infeas** | 0.000 | 0.000 |
| | | | `floor` | **infeas** | 0.142 | 0.142 | 0.000 |
| `pd-rngd-gpu-card` (P/D) | `.hp-reval-margin18-…-card/cache` | 468 | **roofline** | **0.000** | **0.000** | 0.000 | 0.000 |
| | | | `floor` | **infeas** | 0.219 | 0.068 | 0.068 |

*"infeas" = false-infeasible: the oracle has a feasible plan and the top-K has none.
Values are regret@K; recall and speedup are in `outputs/exp_surrogate/*/surrogate.json`.*

**Two findings, and the second is the one that matters.**

1. **The shipped ranker is false-infeasible at K=20 on two of the three corpora.**
   `--top-k 20` does not merely cost accuracy there; it reports no plan at all.
2. **Ordering by the roofline floor fixes those two and breaks the third.** On
   `pd-rngd-gpu-card` the shipped ranker is already perfect at K=10 and `floor` is
   false-infeasible there, never recalling the optimum until K=468.

So `floor` is not a fix. It is a second fixture-specific ranker, and adopting it on
the strength of the two corpora where it wins would repeat exactly the error that
produced *"N=78, regret 0.000 at every K down to K=1"* — a curve measured on one
aggregated-heavy fixture and then relied on generally. **No ranker change is made.**

## Consequence for E6a

`--top-k 20` is not available as a cost lever on these fixtures. E6a's budget stands
at the D27 figures — **~140 h full axis, ~95 h reduced** (`docs/sim_cost_profile.md`)
— and needs a different lever or a smaller experiment. What K would be safe is not
established either: K=30 is clean on all three corpora here, but three fixtures do not
license a general threshold, and the corpora were swept under different SLO margins.

## Two defects fixed in the harness

- **`--cache-dir` replay.** The driver was simulate-only, so measuring regret on an
  existing corpus meant re-running it. It is a pure function of
  `{candidate → SimResult}`; now it can replay one.
- **Regret was unmeasurable for every minimisation objective.**
  `pareto.objective_value` negates minimisation objectives so the caller can always
  maximise, and the regret formula divided by the *signed* value behind an
  `oracle_value > 0` guard. For `minimize_energy` — which `llama31-8b.yaml` uses — that
  guard was never true and regret came back `None` at every K. The denominator is now
  `abs(oracle_value)`. **Any past regret number read off this driver for a
  minimisation spec was `-`, not `0`**; the published `0.000` figures are from
  `maximize_slo_goodput_per_joule` specs and are unaffected.
