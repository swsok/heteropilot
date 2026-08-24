# Router baselines — RR / RAND / LOAD (simulation, 2026-08-24)

Work order §12 router baselines. Routing policy is a **simulator input**, so —
unlike the optimizer/resource/architecture baselines, which replay one cached sim
(`exp_baselines.py`) — this axis **re-simulates** the same deployment once per
policy.

One command:

```bash
./experiments/scripts/run_exp_router.sh    # exp2-local-lab, 120 requests, seed 42
```

**Everything below is an LLMServingSim prediction, not a live measurement.** The
deployment swept is the most-replica aggregated candidate on exp2-local-lab: a
**heterogeneous 4-replica mix** (A5000 ×2 + RTXPRO6000 ×2) — precisely where the
router matters, because it balances requests across replicas of very different
speed. Router choice is a **no-op for single-replica** deployments (nothing to
balance), so those are excluded by construction.

## Result (Llama-3.1-8B, 120 requests, 4 heterogeneous replicas)

| policy | p50/p99 TTFT (ms) | p50/p99 TPOT (ms) | throughput (tok/s) | SLO attain | goodput (rps) | tok/J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LOAD** | **67 / 314** | **15 / 34.2** | **5,453** | 1.00 | **3.75** | 1.244 |
| RR | 79 / 344 | 22 / 35.3 | 5,340 | 1.00 | 3.68 | 1.238 |
| RAND | 103 / 644 | 31 / 37.0 | 5,169 | 1.00 | 3.56 | 1.249 |

## Findings

1. **LOAD wins on every SLO-facing metric.** It has the lowest TTFT (p50 and p99),
   the lowest TPOT, the highest throughput, and the highest SLO goodput. Balancing
   by live load keeps the slow A5000 replicas from becoming the tail.
2. **RAND has the worst tail — ~2× LOAD's p99 TTFT** (644 vs 314 ms). Random
   assignment ignores replica speed, so it routinely piles requests onto a slow
   A5000 replica while a fast RTXPRO6000 idles. This is the clearest argument for
   load-aware routing on a heterogeneous fleet.
3. **RR sits in between** — even shares are better than random but still ignore the
   A5000/RTXPRO6000 speed gap.
4. **Energy (tok/J) is essentially flat** (1.238–1.249): all three run the same 4
   devices for a similar wall time, so the router moves *latency/goodput*, not
   efficiency. The RAND value is marginally highest only because its lower
   throughput slightly changes the active/standby power mix — not a real win.
5. **All policies meet the SLO at this load** (attainment 1.00); the differentiator
   is tail-latency headroom, which is exactly what degrades first as load rises.

## Honesty caveats (absolute rule 3)

- All predictions; A5000 power is measured, host/RTXPRO6000/link numbers are
  placeholder/vendor_spec (see the cluster and profile files).
- The result is specific to a **heterogeneous** replica set — the regime where the
  router matters most. On a homogeneous fleet RR and LOAD converge.
- The predictor's routing policy defaults to `LOAD` (Phase-2 behavior, byte-
  identical); only this experiment varies it. `CUSTOM` (SLO-aware routing) is a
  Phase-5+ hook and out of scope here.
- Full provenance (git commit, spec hashes, command, versions) in
  `experiments/results/router_baselines.json`.
