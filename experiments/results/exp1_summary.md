# Exp 1 — same-GPU TP=1/2/4 sweep (simulation, 2026-08-21)

Work order §12 Exp 1: validate the planner pipeline across tensor-parallel
degrees on one real accelerator class (A40), reporting TTFT/TPOT/power.

One command:

```bash
./experiments/scripts/run_exp1.sh          # TP=1,2,4; 300 requests; seed 42
# = experiments/scripts/exp1_tp_sweep.py on
#   experiments/configs/services/llama31-8b-goodputj.yaml
#   experiments/configs/clusters/exp1-a40-tp-sweep.yaml
```

**Everything below is an LLMServingSim prediction, not a live measurement.** It
runs on the **measured** A40 profile bundle (`profiler/perf/A40/meta-llama/Llama-3.1-8B/`,
dummy-weight layerwise profiling; TP=1/2 profiled 2026-08-19, **TP=4 profiled
2026-08-21** for this experiment). Prediction-error band ~1-3% class (A40 is
Ampere sm_86); prefix caching disabled throughout (D12).

## Result (Llama-3.1-8B, 300 requests @ 10 rps offered)

| TP | devices | feasible | p50/p99 TTFT (ms) | p50/p99 TPOT (ms) | throughput (tok/s) | SLO attain | avg/peak W | energy (J) | tok/J | J/req |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | no | 6,898 / 106,109 | 160.2 / 225.0 | 2,496 | 0.00 | 564 / 724 | 102,630 | 1.860 | 342.1 |
| 2 | 2 | no | 989 / 35,230 | 89.8 / 100.3 | 5,008 | 0.00 | 831 / 855 | 75,290 | 2.536 | 251.0 |
| 4 | 4 | no | 165 / 4,533 | 43.5 / **52.1** | 8,842 | 0.94 | 1,292 / 1,386 | 66,320 | **2.879** | 221.1 |

SLOs (goodput service spec): TTFT p99 ≤ 25,000 ms, TPOT p99 ≤ 50 ms.
Violations: TP=1 fails both; TP=2 fails both; **TP=4 clears TTFT and misses TPOT
p99 by 4%** (52.1 vs 50 ms), attaining 94%.

## Findings

1. **TP scaling is monotonic on every latency/throughput axis, as it must be.**
   TTFT p99 drops 106 s → 35 s → 4.5 s and TPOT p99 225 → 100 → 52 ms as TP goes
   1 → 2 → 4; throughput rises 2.5k → 5.0k → 8.8k tok/s. The planner pipeline
   reproduces the expected TP curve end-to-end (enumerate → compile → simulate →
   feasibility), which is what Exp 1 is meant to validate.

2. **More TP is more energy-efficient here, not less.** tokens/J improves
   1.860 → 2.536 → 2.879 and J/request falls 342 → 251 → 221, even though average
   power nearly doubles per step (564 → 831 → 1,292 W). Under this saturating
   load the faster completion outweighs the higher instantaneous power — the
   opposite of the light-load Exp 2 regime, where the smallest config won on
   tokens/J. Efficiency ordering is load-dependent, and the pipeline captures both.

3. **A single 4-GPU island saturates at 10 rps / 300 requests.** No TP clears the
   TPOT SLO here; TP=4 comes within 4%. At the lighter 100-request offered load
   used to smoke-test the driver, TP=2 was already feasible — so this is a
   capacity result, not a modeling failure. The offered load, not TP, is the
   binding constraint at TP=4.

4. **Power scales ~linearly with device count**, as expected from the per-A40
   measured power block (active ~298 W): 564 → 831 → 1,292 W for 1/2/4 devices
   (plus host base). Peak tracks average closely (no long standby tail in a
   saturating run).

Planner metrics (§12): 7 candidates generated, 7 survived, 0 pruned (prune ratio
0.0) — the size-4 island admits few placements and none hit a bound. Per-TP
simulator wall time is recorded in `exp1_tp_sweep.json`.

## Honesty caveats (absolute rule 3)

- **Power mixes measured and placeholder.** The A40 accelerator power is measured
  (`profiles/accelerators/a40.yaml`), but the host/node power components are
  `source: placeholder` (the Exp-1 node has no measured wall/IPMI power), so the
  absolute Watt/Joule figures include default host power. The **relative** trend
  across TP is trustworthy; the absolute energy is not a measured node figure.
- **TP=2 is conservative.** The size-4 A40 island's intra bandwidth is the PCIe
  bottleneck (64 GB/s, vendor_spec — `island_interconnect` takes the island min),
  so the TP=2 row is charged PCIe, not the 112.5 GB/s NVLink pair a production
  TP=2 engine would use (the `a40x8.yaml` model). A dedicated NVLink-pair TP=2
  would be somewhat faster than shown here.
- **TP=4 intra is cross-pair PCIe by physical necessity.** On this box NVLink
  (NV4) bonds only GPU pairs; a 4-way TP group must cross the PCIe host bridge, so
  the TP=4 collectives are correctly charged PCIe. This is realistic for the
  hardware, not a pessimistic assumption.
- Full provenance (git commit, spec hashes, seed, command, versions) is in
  `experiments/results/exp1_tp_sweep.json`.
