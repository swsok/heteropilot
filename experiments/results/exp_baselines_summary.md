# Baselines + ablation (simulation, 2026-08-24)

Work order §12: comparison baselines (optimizer / resource / architecture) and
the Phase-5 ablation study. One command:

```bash
./experiments/scripts/run_exp_baselines.sh    # exp2-local-lab, 120 requests, seed 42
```

**Everything below is an LLMServingSim prediction, not a live measurement.**
Cluster: A5000 (node0) + RTXPRO6000 (node1), both measured/vendor profiles; the
Ascend island is not generated at all (its profile supports only Qwen, not this
Llama service). Prefix caching off (D12). Objective: **SLO-goodput/J**
(`(completed_tokens × slo_attainment) / energy`).

## Method — simulate-once, replay-select

One oracle-mode pass simulated every structurally-valid, simulatable candidate
(13 candidates, 8 feasible; 36 min of sim); each baseline/ablation is then a
`(subset → selection-rule → objective)` decision over that shared cache, scored
on its pick's **true** metrics. `regret = (oracle_value − strategy_value) /
oracle_value`. The real optimizer (`exhaustive`/`pareto`/`feasibility`) is
untouched, so oracle-agreement is preserved — nothing here is a pruning stage.

## Result (oracle goodput/J = 3.138)

| group | strategy | goodput/J | regret | devices | note |
| --- | --- | ---: | ---: | ---: | --- |
| optimizer | exhaustive-oracle | 3.138 | 0.000 | 1 | reference optimum (RTXPRO6000 tp1, 1 GPU) |
| optimizer | **proposed** | 3.138 | **0.000** | 1 | sim-guided pruned search — matches the oracle |
| optimizer | greedy | 3.138 | 0.000 | 1 | analytical roofline proxy, no sim |
| resource | fastest-only | 3.138 | 0.000 | 1 | RTXPRO6000 (max bandwidth) |
| resource | most-efficient-only | — | n/a | — | A5000 (max bw/W) has **no feasible** candidate at this SLO |
| resource | least-device | 3.138 | 0.000 | 1 | |
| resource | simulator-blind | 2.112 | **0.327** | 2 | over-provisions (most-replicas rule) |
| architecture | aggregated | 3.138 | 0.000 | 1 | |
| architecture | homogeneous-P/D | 2.100 | **0.331** | 2 | P/D does not pay at this scale |
| architecture | heterogeneous-P/D | — | n/a | — | impossible: both islands are `cuda` (cross-backend needs an NPU) |
| ablation | No-PD-Specialization | 3.138 | 0.000 | 1 | |
| ablation | No-Energy | 1.662 | **0.470** | 4 | energy-blind → worst over-provisioning |
| ablation | No-Topology | 3.138 | 0.000 | 1 | transfer cost doesn't change the aggregated winner |
| ablation | No-Calibration | — | n/a | — | calibration is not in the plan path → today's default already is no-calibration |
| ablation | No-Uncertainty | — | n/a | — | robust margins default to 0 (need Phase-4 margins) |
| ablation | Static | — | n/a | — | no-replanning is a Phase-6 concept |

## Findings

1. **The proposed planner equals the oracle (regret 0).** On this space the
   sim-guided pruned search finds the true optimum — the oracle-agreement
   invariant, observed on a real (non-mock) run.
2. **Energy-awareness is the biggest differentiator.** Dropping energy from the
   objective (No-Energy) costs **47%** of goodput/J: it picks a 4-GPU P/D
   over-provision that maximizes raw goodput but burns far more joules. This is
   the strongest argument for the per-J objective.
3. **Naive provisioning over-provisions.** The simulator-blind "ops heuristic"
   (most replicas on the fastest class) costs **33%** — it buys a second GPU the
   workload doesn't need. Same lesson as Exp 2's light-load regime.
4. **P/D does not pay at this scale (regret 33%).** The best P/D split
   (A5000 prefill + RTXPRO6000 decode) is worse than a single RTXPRO6000
   aggregated engine — consistent with the Exp 5 finding that P/D wins only once
   one card misses the SLO. No-Topology's zero regret confirms the KV-transfer
   term doesn't change the (aggregated) winner here.
5. **Greedy matches the oracle *here* — honestly, not always.** The optimum
   (one RTXPRO6000) is analytically obvious, so the roofline proxy finds it
   without simulating. This is a property of this spec, not a general claim: on a
   tighter-SLO / feasibility-subtle workload the analytical proxy can pick an
   infeasible or worse candidate (the harness records `regret=1.0` +
   `selected_infeasible` in that case). A tighter-SLO run is the natural follow-up
   to exhibit the gap.
6. **Two rows are honestly N/A, not zero.** `most-efficient-only` = A5000, whose
   every candidate is infeasible at this SLO; `heterogeneous-P/D` needs two
   backends and this cluster is all-CUDA. Neither is fabricated.

Planner metrics: 13 generated (oracle) = 13 (pruned), **prune ratio 0** (the
generator's analytical bounds reject none here; feasibility is decided post-sim,
8/13 feasible), selection wall 0.013 s over the shared cache.

## Honesty caveats (absolute rule 3)

- All predictions; A5000 power is measured, host components + RTXPRO6000/link
  numbers are placeholder/vendor_spec (see the cluster and profile files).
- `fastest` / `most-efficient` use `memory_bandwidth_gbps` and active power as
  proxies — `AcceleratorProfile` has no compute-speed field, and decode is
  memory-bound. Labeled as proxies.
- No-Calibration / No-Uncertainty / Static are **N/A in the current phase** and
  emitted as labeled rows: the objective (goodput/J) reads simulator
  `slo_attainment`/energy, never TTFT/TPOT, so calibration and margins move only
  the feasibility boundary — they cannot change a feasible plan's objective value.
  A meaningful contrast needs Phase-4 non-identity calibration (margins) and
  Phase-6 replanning.
- Router baselines (RR/RAND/LOAD) are a separate experiment: routing is a
  *simulator input*, so it needs re-simulation, not replay (planned as
  `feat/router-baselines`).
- Full provenance (git commit, spec hashes, command, versions) in
  `experiments/results/baselines.json`.
