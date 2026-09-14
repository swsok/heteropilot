# E-B1 — decision regret against measurement budget (uncertainty work order STEP B4)

**What this is.** The §6 effect claim's quantitative basis. Start from a fully
measured fixture, degrade `k` of its inputs to a lower grade, then let five
measurement strategies buy the truth back one input at a time and watch the
regret fall. The strategy that reaches zero regret for the fewest server-hours
wins. Nothing is re-simulated: truth comes from the E-A1 envelope cache and every
degraded or restored state is B1's closed-form perturbation of it.

```
R(B) = V(π_truth) − V_truth(π̂_B)
```

`V` is the spec's primary objective, `minimize_energy`, so a regret of 8 000 is
8 000 joules of avoidable consumption.

> **Run 2026-09-14 on the reconciled architecture** (PRs #78–#80, `docs/deviations.md`
> D33). The 2026-09-11 run of this experiment is **not** superseded in its
> conclusions but its numbers do not carry over: the margin policy, the `refuse`
> default and the partial-coverage rule all changed, and so did the truth itself —
> §"What the truth is now".

## The answer

*231 degraded sets (k ∈ {1,2,3}, 10 seeds), default penalty. Full tables for both
penalties in `outputs/uncertainty/eb1/eb1_summary.json`; figure:
`figures/eb1_regret_vs_budget.png`.*

| strategy | R(0 h) | R(0.041 h) | R(2.1 h) | mean h to regret 0 |
| --- | ---: | ---: | ---: | ---: |
| `oracle` | 4,429.2 | **0.0** | 0.0 | **0.007** |
| **`ours`** | 4,429.2 | **0.0** | 0.0 | **0.007** |
| `random` | 4,429.2 | 2,394.1 | 1,077.4 | 0.072 |
| `round_robin` | 4,429.2 | 4,309.5 | 2,035.0 | 0.176 |
| `widest` | 4,429.2 | 3,950.3 | 0.0 | 0.030 |

Restricted to the **37 sets that actually moved the plan** — the rest start at
regret 0 and no strategy can do anything:

| strategy | R(0 h) | R(0.041 h) | R(2.1 h) | mean h to regret 0 |
| --- | ---: | ---: | ---: | ---: |
| `oracle` | 27,652.3 | **0.0** | 0.0 | **0.041** |
| **`ours`** | 27,652.3 | **0.0** | 0.0 | **0.041** |
| `random` | 27,652.3 | 14,947.2 | 4,484.2 | 0.452 |
| `round_robin` | 27,652.3 | 26,905.0 | 11,210.4 | 1.097 |
| `widest` | 27,652.3 | 24,662.9 | 0.0 | 0.189 |

**`ours` matches the oracle exactly, at every budget and on both slices**, and
reaches zero regret in **0.041 h against `random`'s 0.452 and `round_robin`'s
1.097** on the sets where anything is at stake. Every strategy reaches 0
eventually — the pool is 11 items and the budget axis runs to 6.3 h — so the
claim is about *rate*, not reachability.

**Why `ours` can match an oracle here, and why that is less impressive than it
looks.** The pool has 11 items and on this fixture exactly one of them —
`sim_error:domain`, at 0.041 h, the cheapest — carries essentially all the
decision regret. Any rule that ranks it first matches the oracle. `ours` does so
because ΔR/cost puts it first; `random` gets there 1-in-k of the time;
`round_robin` visits `link_bw` before `sim_error` alphabetically and `widest`
prefers the links' wide range. A pool with two or three items that each matter
would separate the strategies more sharply and is not available on this fixture.

## What the truth is now, and why the numbers moved

The work order sets truth as "calibration = Stage A의 도메인 yaml". On the
reconciled tree that is `rngd_card_edf.yaml` itself. D33 **dropped**
`rngd_card_edf.domain.yaml`, the E-A2 artifact this experiment first ran against,
because it paired each real run with a simulation at the same *requested*
concurrency — its sim side ran at served 71–189 against real 15–107, the D32
mis-pairing — and the nine-point D32 domain supersedes it. So the truth here is a
**different and better-founded** domain than the one E-B1 first measured against,
and the two runs' absolute regrets are not comparable.

Truth, under that domain: **50 of 324 candidates feasible**, winner
`cuda-a40-node_a40a-tp4-dp1-s128-t8192` at −73,560 J.

## Scalar mode had to be redefined, and getting it wrong produced a false result

The third degradation is "도메인을 스칼라 모드로" — one error for the whole
workload. It used to be done at **file** granularity, `rngd_card_edf.yaml` being
the pre-domain artifact the domain file was built from, so the contrast was a
real pair of artifacts. D33 merged them and the pair no longer exists.

It is now synthesised by **collapsing each domain to the single point it was
fitted at**, so `widen_error_bars` has no slope and every margin is that one
value held flat. That is exactly the state `a40.accuracy.yaml`'s own header
describes the A40 domain as having been in before it gained a second and third
point.

**The first attempt passed no domains at all, and that was wrong in a way worth
recording.** The scalar fallback is a fitted *bucket* error, and neither
`a40.yaml` nor `rngd_card_edf.yaml` carries one for this canonical bucket
(`in_lt1024-out_ge512-rps_lt20`), so all 324 candidates came back `unmeasured`
and the degraded search had **nothing feasible at all**. `analyze` then returns
`ΔR = None` for every item — "no recommendation to flip" — and its sort key drops
them all into one bucket ordered by `input_id`, where `link_bw:…` precedes
`sim_error:…`. `ours` therefore bought the inert links first and came out as the
**worst** strategy, tying `round_robin` to the digit:

| strategy | R(0.041 h) | mean h to 0 |  | (the discarded run) |
| --- | ---: | ---: | --- | --- |
| `oracle` | 0.0 | 0.041 | | |
| `ours` | 195,436.6 | 1.663 | ← | identical to `round_robin` |
| `random` | 92,388.2 | 0.506 | | |

That table is **not a result** and is kept here only because its shape is a
useful alarm: `ours` exactly equalling a baseline means the ranking has no
signal, not that the invention lost.

## Reproduce

```bash
# the truth cache already exists (24 m 26 s at --workers 8; see ea1_margin_modes.md)
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb1_regret_vs_budget.py \
    --cache-dir outputs/uncertainty/ea1/cache --exhaustive --k 1 2 3
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb1_report.py
```

`eb1_report.py` prints the tables and writes the figure and
`outputs/uncertainty/eb1/eb1_summary.json`; **this document is written by hand
around those numbers** — its `--out` argument writes nothing and is dead.

## Two harness facts a re-runner needs

**The E-A1 truth cache carries no operating points.** All 162 entries predate
operating-point caching, and main's margin policy reads the operating point off
the `SimResult`, so every candidate came back `unmeasured` and there was no truth
to degrade. `_IslandOperatingPoint` — E-A1's own subclass, which fills the point
from the cached per-island served concurrency — is imported rather than copied,
so the two experiments cannot disagree about what a cached run's operating point
is. The evaluation's `operating_points` are also passed into `PerturbContext`,
which the old call omitted.

**`policy.decide` takes four arguments now** (D33 §3): the policy is handed the
`SimResult` its operating point comes from as well as the metrics.
