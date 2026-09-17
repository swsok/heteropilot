# E-N1 — what the simulator's steps look like, counted

`WORK_ORDER_npu_exec_model_spike.md` STEP A.1. Every number here is a property of a
**simulated** run; nothing was measured on hardware. The grid the padding and grouping
ladders come from is `profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml (artifact d6ae6a43-6ce0-4864-aaca-eeac4340234c)`.

| point | steps | mean decode bs | mixed steps | mixed cycle share | prefill steps bs=1 | decode pad (artifact ladder) | decode pad (pow2) | KV groups: menu / decode edges / uniform 1024 | D17 measured @ this batch | prefill 128-pad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1p0 | 182269 | 1.07 | 137 (0.1 %) | 0.4 % | 100.0 % | 0.0 % | 0.0 % | 1.05 / 1.05 / 1.05 | 1.95 | 7.2 % |
| c1p99 | 90705 | 2.16 | 299 (0.3 %) | 1.4 % | 100.0 % | 11.3 % | 11.3 % | 1.82 / 1.63 / 1.63 | 2.00 | 7.2 % |
| c3p98 | 45796 | 4.27 | 299 (0.7 %) | 2.6 % | 100.0 % | 29.9 % | 29.9 % | 2.87 / 2.12 / 2.14 | 2.43 | 7.2 % |
| c7p88 | 24063 | 8.13 | 299 (1.2 %) | 4.6 % | 100.0 % | 38.3 % | 38.3 % | 4.15 / 2.40 / 2.45 | 2.80 | 7.2 % |
| c15p3 | 13383 | 14.61 | 299 (2.2 %) | 7.4 % | 100.0 % | 25.7 % | 25.7 % | 5.32 / 2.63 / 2.72 | 3.02 | 7.2 % |
| c15p59 | 13054 | 14.97 | 299 (2.3 %) | 7.6 % | 100.0 % | 30.3 % | 30.3 % | 5.37 / 2.64 / 2.73 | 3.03 | 7.2 % |

D17 measured, on the card, attention executions per layer of
1.95 / 2.40 / 2.87 / 3.03 / 3.08 at mean batch 1.95 / 3.91 / 8.91 / 15.16 / 29.09 --
a curve that SATURATES near 3. The three KV-group columns are the candidate grids.
None of them is the answer: the kernelwise menu keeps growing where the measurement
flattens, and the two coarse grids track the shape but sit low. So grouping is not
'one attention execution per distinct compiled bucket in the batch', and STEP C.3
has to measure the rule rather than pick one of these.
