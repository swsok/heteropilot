# The surrogate top-K — a ranker that sees the parallelism axis

> **Superseded in part, 2026-09-11.** Everything below the line was measured on
> 2026-09-08 and stands as the diagnosis. What changed is the conclusion
> *"**No ranker change is made**"*: a ranker that fixes the diagnosed fault
> without breaking any corpus now exists, has been measured on **sixteen** corpora
> instead of three, and **ships as the default**. The 2026-09-08 sections are kept
> unchanged underneath, per the retract-in-public rule.

## What was wrong with the fix attempts, and what the diagnosis already implied

The 2026-09-08 measurement established two things: the shipped proxy tok/J is
algebraically invariant to TP and DP, and ordering by the roofline TPOT floor
instead repairs two corpora and breaks a third. It also recorded, without drawing
the conclusion, the fact that settles it:

> `tpj_then_floor` measured **byte-identical to `roofline`**.

An explicit tie-break on the parallelism-sensitive term changed nothing, because
**no two proxy values are ever exactly tied**. The cancellation is algebraic but
the arithmetic is floating point, so a spread of about one part in ten thousand
survives — `0.000463` on a value of `4.65` — and `sorted()` reads it as a
preference. The shipped ranker does not ignore the parallelism axis so much as
**let rounding dust pick for it**.

That turns the fix into a one-line statement of the diagnosis: make the tie
explicit. Group candidates whose proxy tok/J differs by less than a relative
tolerance, treat the group as tied, and order **inside** it by
`roofline_tpot_ms`. The coarse order — accelerator, `max_num_seqs`, which differ
by factors — is untouched, which is why this does not break the corpus `floor`
broke.

Bins are cut where the relative gap between consecutive sorted values exceeds the
tolerance, not rounded onto a fixed grid: a grid puts a boundary at an arbitrary
value and splits a genuinely tied group whenever one straddles it.

## The tolerance is not fitted

0.1 %, 1 % and 5 % — three orders of magnitude above the proxy's noise, two below
the factor-scale gaps it must preserve — give **identical regret curves on every
corpus** and select the **identical set of candidates at every K measured**. They
do not give the identical order: a wider bin merges groups a narrower one keeps
apart, differing at 56 and 74 positions of 324 deep in the list. Top-K reads only
the membership. `tests/test_binned_surrogate.py` pins that.

## Regret on sixteen corpora

*Replay only, no new simulation. `roofline` is the 2026-09-08 default,
`binned` is `BinnedRooflineRanker(0.01)`. Artifacts: `outputs/surrogate_v3/`.*

The eight starred corpora are single-sweep caches at the service spec's own
arrival rate; the eight below them are E6's two sweeps read **one rate at a
time** through the planner's own cache key (`--cache-rps`), which is what adds
the arrival-rate axis every single-point corpus lacks.

| fixture | corpus | rps | N | ranker | K=5 | K=10 | K=20 | K=30 | K=50 |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `pd-rngd-gpu` no P/D | `perf/topk/cache_full` | 1.0* | 324 | `roofline` (was) | **infeas** | **infeas** | **infeas** | 0.000 | 0.000 |
| | | | | `floor` | **infeas** | 0.142 | 0.000 | 0.000 | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `.hp-pd-slo` | 1.0* | 612 | `roofline` (was) | **infeas** | **infeas** | **infeas** | 0.701 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 0.142 | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `.hp-reval-margin18-…gpu` | 1.0* | 612 | `roofline` (was) | **infeas** | **infeas** | **infeas** | 0.711 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 0.142 | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `.hp-reval-tight-…gpu` | 1.0* | 612 | `roofline` (was) | **infeas** | **infeas** | **infeas** | 0.711 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 0.142 | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `.hp-slo-margin18-tight-…gpu` | 1.0* | 612 | `roofline` (was) | **infeas** | **infeas** | **infeas** | **infeas** | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 0.142 | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu-card` P/D | `.hp-reval-margin18-…card` | 1.0* | 528 | `roofline` (was) | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | 0.219 | 0.068 | 0.068 |
| | | | | **`binned`** | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu-card` P/D | `.hp-reval-tight-…card` | 1.0* | 528 | `roofline` (was) | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | 0.219 | 0.068 | 0.068 |
| | | | | **`binned`** | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu-card` P/D | `.hp-slo-margin18-tight-…card` | 1.0* | 528 | `roofline` (was) | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | 0.219 | 0.068 | 0.068 |
| | | | | **`binned`** | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `e6/pd-rngd-gpu` | 1 | 612 | `roofline` (was) | 0.620 | 0.286 | 0.286 | 0.286 | 0.286 |
| | | | | `floor` | **infeas** | 2.734 | 2.734 | 2.494 | 2.117 |
| | | | | **`binned`** | 0.620 | 0.286 | 0.286 | 0.286 | 0.286 |
| `pd-rngd-gpu` P/D | `e6/pd-rngd-gpu` | 3.3 | 612 | `roofline` (was) | **infeas** | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 1.388 | 1.317 |
| | | | | **`binned`** | **infeas** | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `e6/pd-rngd-gpu` | 10 | 612 | `roofline` (was) | **infeas** | **infeas** | **infeas** | 0.711 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 0.142 | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu` P/D | `e6/pd-rngd-gpu` | 20 | 612 | `roofline` (was) | **infeas** | **infeas** | **infeas** | **infeas** | **infeas** |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | 0.210 | 0.210 |
| | | | | **`binned`** | **infeas** | **infeas** | **infeas** | **infeas** | **infeas** |
| `pd-rngd-gpu-card` P/D | `e6/pd-rngd-gpu-card` | 1 | 528 | `roofline` (was) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | 1.343 | 1.343 | 1.343 | 1.343 | 0.000 |
| | | | | **`binned`** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu-card` P/D | `e6/pd-rngd-gpu-card` | 3.3 | 528 | `roofline` (was) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | 1.980 | 1.980 | 1.086 | 1.086 | 0.974 |
| | | | | **`binned`** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu-card` P/D | `e6/pd-rngd-gpu-card` | 10 | 528 | `roofline` (was) | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| | | | | `floor` | **infeas** | **infeas** | 0.219 | 0.068 | 0.068 |
| | | | | **`binned`** | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pd-rngd-gpu-card` P/D | `e6/pd-rngd-gpu-card` | 20 | 528 | `roofline` (was) | **infeas** | **infeas** | **infeas** | **infeas** | **infeas** |
| | | | | `floor` | **infeas** | **infeas** | **infeas** | **infeas** | 0.000 |
| | | | | **`binned`** | **infeas** | **infeas** | **infeas** | **infeas** | **infeas** |

### What the table says

**`binned` weakly dominates the shipped ranker.** Across all 96 (corpus, K)
cells: **11 strictly better, 0 worse, 85 identical**. That is the standard the
2026-09-08 page set and `floor` failed — a replacement must not be justified on
the fixtures where it happens to win.

**False-infeasibility at K=20 falls from 8 corpora to 2**, and clean `0.000`
rises from 7 to 13. `--top-k 20`, which §E6a wanted and STEP 1 found unusable, is
now clean on thirteen of sixteen corpora.

**`floor` is confirmed as the wrong fix, more strongly than before.** The
rate-keyed corpora catch it doing real damage rather than merely being
false-infeasible: regret **2.734** at 1 rps on `pd-rngd-gpu` and **1.980** at
3.3 rps on the card fixture, where both efficiency rankers are at 0.000 or
0.286.

### What it does not fix, and the shape of the remaining gap

**At 20 rps every efficiency-ordered ranker is false-infeasible to K=50, on both
fixtures.** `floor` finds a plan there — 0.210 at K=30 on `pd-rngd-gpu`, 0.000 at
K=50 on the card fixture — so the optimum is reachable and the efficiency
ordering simply does not reach it. At the highest load feasibility is decided by
the TPOT floor alone, and no amount of binning promotes a candidate out of a
low-tok/J bin. **`--top-k` is therefore still not a general cost lever**; what
has changed is that the exception is now characterised (the top of the load axis)
instead of unknown.

**K=5 and K=10 are unchanged**: false-infeasible on 9 and 8 of 16, exactly as
`roofline` was. The binning acts inside bins, and at those depths the bins
themselves are the constraint.

**Two cluster fixtures, still.** Sixteen corpora is not sixteen fixtures — it is
`pd-rngd-gpu` and `pd-rngd-gpu-card` under eight sweep configurations and four
arrival rates. A third fixture would need a corpus that does not exist.

### Rank fusion was tried and is worse — do not repeat it

The natural next thought, given that `binned` wins at low load and `floor` wins
at 20 rps, is to merge the two orderings round-robin: each parent's top ~K/2 lands
inside any top-K, so the merge is false-infeasible only where **both** parents
are, and it has no fitted weight. Measured as `merge_bin_floor` and
`merge_roofline_floor`, it is **worse than `binned` almost everywhere** —
false-infeasible at K=20 on 7 corpora against 2, and at K=30 on 7 against 2. The
depth it gives up costs more than the complementarity buys, and it does not even
rescue 20 rps. Both merges stay in `exp_surrogate.py --rankers` so the result is
reproducible; neither ships.

## Two harness changes this needed

- **`--cache-rps`.** A multi-point sweep's cache holds one entry per arrival rate
  under each candidate id — `outputs/e6/pd-rngd-gpu/cache` is 660 files for 289
  distinct candidates — and `_load_cache_corpus` reads by candidate id, so it
  would have kept whichever sorted last and **silently mixed four arrival rates
  into one ranking**. Such a corpus is now refused by default, and `--cache-rps`
  resolves it properly by asking the planner's own `EnvelopeCache` for the entry
  with the trace regenerated at that rate. The lookup's own check: at 10 rps it
  returns 243 entries, exactly the `cache_hits: 243` that sweep recorded.
- **The measured class is the shipped class.** `exp_surrogate.py` imports
  `BinnedRooflineRanker` from `planner/optimizer/surrogate.py` rather than
  defining its own copy, so the ranker that produced this table cannot drift from
  the one that ships.

---

*Everything below is the 2026-09-08 page, unchanged.*

## The surrogate top-K cannot be used at K=20 — measured on three fixtures (2026-09-08)

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
