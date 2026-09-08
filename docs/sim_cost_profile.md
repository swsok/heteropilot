# Where simulation wall time goes

*`WORK_ORDER_rps_aware.md` STEP 1. Measured 2026-09-08 on the NPU node, `main` =
`de4d235` plus STEP 0. Candidate: the R2 anchor
(`P[cuda:tp4] D[cuda:tp4] -s256-t8192`), 300 requests. Artifacts:
`outputs/perf/`.*

Two things are measured here, and only the first was asked for:

1. **the breakdown**, which decides D27 by the work order's 30 % rule;
2. **the low-RPS cost multiplier**, because E6a's 60 h estimate rests on a
   per-point figure taken at 10 rps, and STEP 0.1 had already found a low-rate
   penalty at 20 requests. Approved as an addition 2026-09-08.

## The breakdown

`pyinstrument` 5.1.3, one run per rate. Profiler overhead is negligible — 341 s
against 327 s unprofiled at 10 rps (4.3 %), 860 s against 859 s at 3.3 rps (0.1 %) —
so the shares below are not an artefact of measuring.

| | 10 rps (338.7 s) | | 3.3 rps (857.1 s) | |
| --- | ---: | ---: | ---: | ---: |
| `controller.read_wait` — waiting on ASTRA-Sim | 182.6 s | **53.9 %** | 442.4 s | **51.6 %** |
| **`generate_graph` — the Chakra subprocess** | 97.8 s | **28.9 %** | 253.7 s | **29.6 %** |
| `main` self | 15.7 s | 4.6 % | 44.0 s | 5.1 % |
| `generate_trace` | 10.7 s | 3.2 % | 26.1 s | 3.0 % |
| `parse_output` + `write_flush` | 16.8 s | 5.0 % | 50.4 s | 5.9 % |
| `Scheduler.schedule` | 5.1 s | 1.5 % | 13.6 s | 1.6 % |

The shares are stable across a 3× change in arrival rate, which is worth knowing on
its own: the cost structure does not shift with load, only the total does.

Inside `read_wait`, 159.6 s of the 182.6 s is `[self]` — the `readline` loop itself.
That is the frontend waiting on the child, not burning CPU on its own work: the run
used 165.5 s of CPU against 338.7 s of wall.

## What the 29 % actually is

The 30 % rule implicitly treats the Chakra share as the amount D27 could save. It is
not. Timing the converter on a real 295-row trace, 20 calls each:

| | per call |
| --- | ---: |
| subprocess (today) | **46.0 ms** |
| in-process | **7.4 ms** |

Re-measured after the D27 edit landed, 20 calls, median: **47.2 ms** against
**5.2 ms**. Same conclusion, slightly better ratio (0.89).

**The conversion work is 7.4 ms; the other 38.6 ms is fork, exec, interpreter start
and imports.** So 84 % of the Chakra share is recoverable, and D27 would return
roughly **24 % of wall time** — not 29 %, and not the ~0.4 % that a naive reading of
`fork_exec` alone (1.267 s) would suggest, because most of the fixed cost is the
child's own startup and sits inside `waitpid`.

Two prerequisites for D27 were checked while the numbers were being taken:

- **Byte-identical output.** The in-process path produces `.et` content equal to the
  subprocess path (`4fd83a7f968e980e`, 4 files).
- **No contamination across calls.** The work order flags module-level state as a
  risk. Three consecutive in-process calls give the same digest, and the 20-call
  benchmark ran without error. `converter.py` has no module-level mutable state;
  `main()` reads `sys.argv` through `argparse`, so a caller must set `sys.argv` and
  `chdir` to the chakra directory (it writes `debug.log` relative to cwd).

## The low-RPS multiplier

Same candidate, same 300 requests, same seed — only the arrival times differ.

| RPS | arrival span | wall | vs 10 rps | progress ticks |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 30.3 s | 327 s | 1.00× | 51 |
| 3.3 | 91.7 s | 859 s | **2.63×** | 104 |
| 1 | 302.5 s | 3376 s | **10.32×** | 305 |

STEP 0.1 saw only a 7–19 % penalty at **20** requests. At 300 it is **2.63×**. The
effect grows with request count, which is exactly the caveat STEP 0.1 attached to
its own result, and it is the reason this point was measured rather than assumed.

**D22 §4.4's mechanism was half right.** *"A slower arrival rate gives the decode
scheduler less to batch, which lowers throughput, which lengthens the simulated
drain"* — that is real and now quantified. What was wrong is *"unbounded in
practice"*: 2.63× is large and finite, nothing like the 0-of-36 that entry recorded.
That part was D26.

### What it does to E6a's budget

The work order estimates *"~1.7 h per point × 6 RPS × 2 TTFT × 3 fixtures ≈ 60 h"*.
The 1.7 h is a 10 rps figure, so the RPS axis is not 6 equal points:

| RPS | multiplier |
| ---: | ---: |
| 1 | **10.32× (measured)** |
| 2 | ~5.2× (interpolated) |
| 3 | 2.63× (measured) |
| 5 | ~1.7× (interpolated) |
| 10 | 1.00× (measured) |
| 20 | ~0.8× (extrapolated) |

**The axis weighs 21.4×, not 6×.** So `1.7 h × 21.4 × 2 TTFT × 3 fixtures = 218 h`,
not 60 h — and on the reduced `{1, 3, 10, 20}` axis (weight 14.75×), **151 h**. The
interpolated and extrapolated rows are labelled as such; they are budget arithmetic,
not envelope points, and A6's `extrapolation: refuse` applies to the latter only.

The 10.32× is the number that moves the decision. A single RPS=1 point costs as much
as ten RPS=10 points, so a six-point RPS axis is not six times one point — it is
twenty-one, and half of that is the bottom two rows.

## Decision — D27 done, against the rule as written

**The 30 % rule says no.** 28.9 % and 29.6 % are both below the threshold. D27 was
done anyway, on the user's direction, for a reason the rule could not see: the rule
exists to stop optimisation work that will not pay, and it assumed top-k pruning was
the cheaper lever. It is not available on this fixture.

| lever | wall | outcome |
| --- | ---: | --- |
| `full` (180 candidates) | 2579 s | **FEASIBLE** — `hp-00077`, 2.963 tok/J |
| `topk20` (12 simulated) | **320 s (8.1×)** | **INFEASIBLE** |

top-k is 8× faster and it drops the optimum. The planner says so itself — *"surrogate
top-K was active and dropped candidates heuristically — this 'infeasible' may be
surrogate error"*. This contradicts `SLIDE_OUTLINE.md`'s *"N=78, regret 0.000 at every
K down to K=1"*: that experiment was aggregated-heavy, and this fixture has P/D and
heterogeneous candidates the roofline surrogate cannot rank. **STEP 2 adds asymmetric
P/D, widening exactly the class it gets wrong**, so the surrogate must be fixed and
measured for regret before E6a can lean on it (STEP 4). Until then D27 is the only
lever that costs nothing in accuracy, because it changes no bytes.

Also worth stating: the rule's 30 % is a threshold on the *share*, and the share is
not the saving. Here the ratio is 0.84 — the conversion itself is 5–7 ms of a 46–47 ms
call, and the rest is fork, exec, interpreter start and imports.

## What D27 actually returned

Same candidate, same three traces, same seed. `main` = `2c373b9` plus the D27 edit.

| | before | after | speedup | saved |
| --- | ---: | ---: | ---: | ---: |
| 10 rps | 327 s | **204 s** | 1.60× | 37.6 % |
| 3.3 rps | 859 s | **500 s** | 1.72× | 41.8 % |
| 1 rps | 3376 s | **2181 s** | 1.55× | 35.4 % |

**All three CSVs are byte-identical to their pre-D27 counterparts**, as are all four
R1/R2 anchors (`dd0eca3f…`, `7d0ff3ce…`, `a0563bc7…`, `fff63c22…`) — seven independent
byte comparisons, three of them on different arrival patterns.

### The saving is larger than predicted, and only part of it is explained

35–42 % against a predicted 24 %, so the run was re-profiled at 10 rps (226 s
profiled, same sha) and the frame table diffed:

| frame | before (338.7 s) | after (225.6 s) | Δ |
| --- | ---: | ---: | ---: |
| **`generate_graph`** | 97.8 s | **17.6 s** | **−80.2 s** |
| `Controller.read_wait` | 182.6 s | 155.3 s | **−27.3 s** |
| `generate_trace` | 10.7 s | 7.1 s | −3.7 s |
| `parse_output` + `write_flush` | 16.8 s | 16.6 s | −0.2 s |
| `Scheduler.schedule` | 5.1 s | 4.7 s | −0.4 s |
| **total** | **338.7 s** | **225.6 s** | **−113.1 s** |

**The prediction was right about what it predicted.** `generate_graph` fell 80.2 s —
**23.7 % of wall**, against the 24 % forecast from the 46.0/7.4 ms decomposition.

**The other 27.3 s is `read_wait`, and this profile does not explain it.** The
frontend's own CPU time barely moved (165.5 s → 155.9 s), so the run is
wait-dominated and the frontend was never the constraint; what changed is that the
ASTRA-Sim child's turnaround improved by 15 % once the frontend stopped forking a
copy of itself four times per iteration. That is a plausible mechanism — a fork storm
against a large address space is not free for anything else on the machine — but it
is not tested here, and the two profiles were taken four and a half hours apart on a
shared node. It is recorded as observed and unattributed. Nothing rests on it: the
decision stands on the 23.7 % that is explained.

### E6a's budget after D27

Applying the measured per-rate factors (0.62, 0.58, 0.65) to the axis weights:

| axis | before D27 | after D27 |
| --- | ---: | ---: |
| full `{1, 2, 3, 5, 10, 20}` | 218 h | **~140 h** |
| reduced `{1, 3, 10, 20}` | 151 h | **~95 h** |

Still far above the work order's 60 h. **D27 alone does not make the full axis
affordable** — the reduced axis plus a working surrogate is what E6a needs, and the
surrogate is STEP 4's problem.
