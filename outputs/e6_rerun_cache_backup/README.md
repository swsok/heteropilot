# The E6 re-run's replay cache — a deliberate exception to the outputs/ rule

`.gitignore` keeps planner caches out of the tree because simulations are
deterministic given the committed code, profiles and seeds. This archive breaks
that rule on purpose, and the reason is a claim made elsewhere.

`outputs/e6_rerun/*/cache/` is what makes re-ranking E6 under a **different
accuracy domain** a replay instead of a re-simulation: the cache key covers the
candidate and the trace digest, not the domain, so changing a calibration file
and re-ranking reuses every simulation that succeeded. `docs/deviations.md` D32
and PR #74 both say so about the RNGD follow-up. That statement is only true
while the cache exists, and it lived on the A40 node alone — regenerating it
costs **13.6 h** (4.18 h + 9.39 h measured, on 64 cores).

> **"costs seconds" was wrong — corrected 2026-09-11 while doing it.** The cache
> stores only the simulations that SUCCEEDED. Every candidate that dies in the
> simulator's KV allocator is re-run on every replay, and there are 114 of those
> per point at 10 rps and **291** at 1 rps. Measured: the 10 rps row replayed in
> **2.7 minutes** with 243 of 357 served from cache; the 1 rps row was still
> simulating after 10 minutes, against 150 minutes for its original run. Read the
> claim as *seconds to hours, depending on how complete the corpus is*. Both ends
> were measured on 2026-09-11 while re-ranking the card fixture
> (`experiments/results/d32_card_recheck.md`): the 3.3 rps rows, where the cache
> covered **318 of 318** candidates, replayed in **0.0 min**; the 1 rps rows, where
> 36 and 34 candidates still had to be simulated and fail, took **44.3** and
> **30.0 min**, because a failure at a low arrival rate is a long simulated span
> before it fails. A full eight-row re-rank of one fixture came to **1.26 h**.
> `experiments/results/rngd_perpe_accuracy_domain.md` re-ranked one cell and
> argued the other four could not move for the same reason.

787 entries, 217 KB compressed. Restore with:

```bash
tar xzf outputs/e6_rerun_cache_backup/e6_rerun_cache.tar.gz
```

The `work/` directories (1.6 GB of per-candidate simulator output) are NOT here
and are not needed for replay — only `cache/` and the sweep summaries are.

Written 2026-09-11, moving off the A40 node.
