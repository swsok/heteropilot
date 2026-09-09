# E6a time estimate, written before the run

*`WORK_ORDER_rps_aware.md` rev 2 asks for the estimate first and the measurement
beside it. This is the estimate; `e6_rps_sweep.md` carries both.*

Measured on the card fixture, 2026-09-09: a 10 rps point with 92 of 318 candidates
uncached took **485.6 s**, and the second TTFT point on the same cache took
**26.7 s** — confirming the two SLO points share every simulation.

Scaling a fully-uncached point from that: 318/92 × 485.6 ≈ **28 min** at 10 rps.
Applying STEP 1's measured RPS multipliers (1 rps 10.3×, 3.3 rps 2.6×, 10 rps 1×,
20 rps ~0.7× extrapolated):

| rps | multiplier | estimate |
| ---: | ---: | ---: |
| 10 | 1.0 | cached, ~8 min |
| 20 | ~0.7 | ~20 min |
| 3.3 | 2.6 | ~73 min |
| 1 | 10.3 | **~4.8 h** |
| | | **~6.4 h per fixture** |

Two fixtures: **~12.8 h**. Above the work order's ~10 h and below its 24 h ceiling,
so the contingency (trimming the 1 rps RNGD knobs) does not fire.
