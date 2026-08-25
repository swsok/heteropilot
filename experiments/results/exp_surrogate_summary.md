# Surrogate top-K accuracy (simulation, 2026-08-25)

Work order §5.4 stage 6. A cheap analytical roofline ranks all candidates; only
the top-K are fully simulated. This experiment MEASURES (never asserts) how much
optimality that trades for the N/K simulation speedup.

One command:

```bash
./experiments/scripts/run_exp_surrogate.sh    # exp2-local-lab, 120 requests, seed 42
```

**Everything below is an LLMServingSim prediction.** Method: simulate-once /
replay-select — every candidate is simulated exactly once (concurrently, see
below), then the oracle optimum and each K's surrogate pick are read off the
shared cache. `recall@K` = the true optimum survived the top-K; `regret@K` =
goodput/J lost vs the oracle; `speedup@K` = N/K sims.

## Result (exp2-local-lab A5000+RTXPRO6000, N=78, oracle goodput/J = 1.660)

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 0 | **0.000** | 78.0x | no |
| 2 | 0 | 0.000 | 39.0x | no |
| 3 | 0 | 0.000 | 26.0x | no |
| 5 | 0 | 0.000 | 15.6x | no |
| 10 | 0 | 0.000 | 7.8x | no |
| 20 | 1 | 0.000 | 3.9x | no |
| 40 | 1 | 0.000 | 1.9x | no |
| 78 | 1 | 0.000 | 1.0x | no |

## Findings

1. **Recall and regret tell different stories — and both matter.** The roofline
   surrogate does NOT place the exact oracle-optimum *candidate id* in its top-10
   (recall 0), yet **regret is 0.000 at every K, down to K=1 (78x speedup).** The
   surrogate picks a *different* candidate that achieves the *identical* goodput/J.
2. **Why: the objective has ties.** On this cluster many candidates reach the same
   SLO-goodput/J (e.g. a single RTXPRO6000 with different vLLM knobs). The roofline
   proxy and the exhaustive ranker break those ties differently, so the exact id
   differs (recall 0) while the *value* is identical (regret 0). This is exactly
   why the design measures both — recall alone would look like a failure; regret
   shows the surrogate loses nothing here.
3. **Honest scope.** This is one workload. A workload with a *unique* optimum and
   no ties would show non-zero regret at small K, and the driver would report it
   (and set `false_infeasible` if a small K dropped every feasible candidate). The
   surrogate top-K is a HEURISTIC, not a sound bound — it can drop the optimum;
   that loss is what this table measures, never a correctness bug. The exhaustive
   oracle remains the yardstick, and a `plan --top-k` output carries a caveat and
   a "re-run with --oracle" suggestion.

## Note — this run was ~5-7x faster (concurrent candidate simulation)

All 78 simulations ran concurrently (default ~half the CPUs, capped 32) via
`planner/util/parallel.predict_all`: each candidate is an isolated `python -m
serving` subprocess (unique `--run-id`), so they parallelize with no locking. The
78-candidate simulate-once pass finished in ~8 minutes wall instead of the
~40-60 minutes a sequential loop would take. The same `predict_all` now backs
`plan`'s `evaluate_candidates` (`plan --workers N`), so the parallelism is
byte-identical to sequential — it only changes wall time, not the result.

## Honesty caveats (absolute rule 3)

- All predictions; A5000 power measured, host/RTXPRO6000/link numbers
  placeholder/vendor_spec.
- No accuracy number is hardcoded anywhere; the recall/regret curve above is the
  only accuracy claim, measured on this named spec/cluster.
- Full provenance (git commit, spec hashes, command, versions) in
  `experiments/results/surrogate.json`.
