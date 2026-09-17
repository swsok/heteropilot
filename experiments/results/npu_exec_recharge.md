# E-N3 — what each bucket rule costs, priced offline

`WORK_ORDER_npu_exec_model_spike.md` STEP A.3. Post-processing of STEP A.1's runs:
no simulator re-run, no hardware. Prices come from the same perf DB the run used,
grid from `profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml (artifact d6ae6a43-6ce0-4864-aaca-eeac4340234c)`. **First-order only** — re-pricing a step does not change which
requests the scheduler would have put in the next one.

Percentage points on the sum of DECODE step cost (the TPOT proxy); the prefill column
is the TTFT proxy. `R-c1` is the control and must read 0.00.

| point | steps | model/sim (median) | R-pad (artifact) | R-pad (pow2) | R-attn (menu grid) | R-attn (decode-edge grid) | R-attn naive (double count) | R-128 (prefill) | R-c1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1p0 | 182269 | 1.000 | +0.00 | +0.00 | -4.46 | -4.47 | +0.06 | +6.71 | +0.00 |
| c1p99 | 90705 | 1.000 | -0.01 | -0.01 | -1.18 | -2.65 | +1.31 | +6.70 | +0.00 |
| c3p98 | 45796 | 1.000 | -0.36 | -0.36 | +3.91 | -2.53 | +7.33 | +6.67 | +0.00 |
| c7p88 | 24063 | 1.000 | +0.47 | +0.47 | +13.35 | -3.63 | +17.90 | +6.63 | +0.00 |
| c15p3 | 13383 | 0.999 | +0.70 | +0.70 | +25.83 | -3.75 | +30.92 | +6.56 | +0.00 |
| c15p59 | 13054 | 0.999 | +0.85 | +0.85 | +26.47 | -3.70 | +31.60 | +6.56 | +0.00 |
