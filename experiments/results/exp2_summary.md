# Exp 2 — heterogeneous resource selection (simulation, 2026-08-15)

One command: `python -m planner plan --service examples/service_specs/llama31-8b-light.yaml
--cluster experiments/configs/clusters/exp2-local-lab.yaml --num-requests 300 --seed 42`.
54 candidates generated, 54 simulated (cold cache, 0 hits), 47 feasible.
Full output with provenance: `outputs/plans/exp2-llama31-8b-light.yaml`.

**Everything below is a simulator prediction, not a measurement.** Expected error
band ~9% on A5000-class numbers, ~1-3% on RTXPRO6000 (docs/phase0_bench_plan.md);
prefix caching disabled throughout (D12).

## Best candidate per placement class, by SLO-goodput/J

| class | best candidate | gpj (tok/J) | energy | peak W | TTFT p99 | attainment |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTXPRO6000-only | tp1-dp1-s128 (1 GPU) | **1.697** | 112.5 kJ | 887 | 118 ms | 100% |
| A5000-only | tp1-dp2-s256 (2 GPUs) | 1.634 | 116.9 kJ | 952 | 405 ms | 100% |
| mixed (1+1) | a5000-dp1 + rtxpro-dp1 | 1.043 | 183.0 kJ | 1,418 | 367 ms | 100% |

## Findings

1. **Right-sizing beats scale-out on tokens/J.** At 2.5 rps every class can meet
   the SLO, and adding hardware only adds energy: the mixed placement burns both
   nodes' power for the whole run while the workload never needs the combined
   capacity - 39% worse goodput/J than the single-GPU winner. Heterogeneous
   mixing pays when demand exceeds one class's capacity, not before.
2. **Two small GPUs nearly match one big one.** A5000 dp=2 lands within 3.7% of
   the RTXPRO6000's goodput/J (1.634 vs 1.697). The planner correctly prefers
   the big card, but the margin is inside the A5000 prediction-error band - on
   real hardware this comparison could flip, which is precisely what the A40
   bring-up can measure.
3. **A5000 single-GPU is infeasible, dp=2 is fine.** All 6 SLO violations are
   A5000 tp1-dp1 (TTFT p99 94-110 s vs a 15 s budget): one 24 GB card saturates
   at 2.5 rps, two carry it comfortably. The light spec achieved its purpose -
   the hardware choice is a real decision, and the planner's answer changes
   with placement class.
4. **Caveat on the mixed penalty's size.** Two nodes genuinely pay two hosts'
   base power, but the host-side components (base 60 W, CPU, DRAM, NIC...) are
   upstream placeholder values on every node. The *direction* of finding 1 is
   robust; its magnitude partially rests on placeholders. Measuring host power
   on the A40 nodes will firm it up.

## Objective behavior check

maximize_slo_goodput_per_joule now differentiates (unlike the 10 rps run where
it degenerated to 1/E): attainment varies across candidates here, and the
recommended plan is not the minimum-energy plan among all classes.
