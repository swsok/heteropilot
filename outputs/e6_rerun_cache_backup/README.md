# The E6 re-run's replay cache — a deliberate exception to the outputs/ rule

`.gitignore` keeps planner caches out of the tree because simulations are
deterministic given the committed code, profiles and seeds. This archive breaks
that rule on purpose, and the reason is a claim made elsewhere.

`outputs/e6_rerun/*/cache/` is what makes re-ranking E6 under a **different
accuracy domain** a replay instead of a re-simulation: the cache key covers the
candidate and the trace digest, not the domain, so changing a calibration file
and re-ranking costs seconds. `docs/deviations.md` D32 and PR #74 both say so
about the RNGD follow-up. That statement is only true while the cache exists, and
it lived on the A40 node alone — regenerating it costs **13.6 h** (4.18 h + 9.39 h
measured, on 64 cores).

787 entries, 217 KB compressed. Restore with:

```bash
tar xzf outputs/e6_rerun_cache_backup/e6_rerun_cache.tar.gz
```

The `work/` directories (1.6 GB of per-candidate simulator output) are NOT here
and are not needed for replay — only `cache/` and the sweep summaries are.

Written 2026-09-11, moving off the A40 node.
