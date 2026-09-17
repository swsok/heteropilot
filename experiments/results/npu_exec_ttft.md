# STEP B.2 — TTFT under the prototype, simulator against simulator

`/tmp/claude-1004/-home-swsok-heteropilot/0908fd49-67e1-4f82-9e91-7471a42f4f49/scratchpad/vllm_all9` (**vllm**) against `outputs/npu_spike/b_aot_strict` (**bucketed_aot/strict**).

**Both columns are simulated.** The measured envelope's TTFT is closed-loop
and the simulator replays an arrival process, so the two are not comparable
(D19); this table scores the prototype against the untouched step policy and
nothing else. D17's measured TTFT error of -32.6 % is quoted elsewhere for
scale and is not what is being tested here.

| point | requests | vllm p50 | bucketed_aot/strict p50 | Δ p50 | vllm p95 | bucketed_aot/strict p95 | Δ p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1p0 | 300 | 77.0 | 75.8 | -1.6 % | 171.0 | 179.3 | +4.9 % |
| c1p99 | 300 | 80.3 | 82.8 | +3.0 % | 179.4 | 191.7 | +6.8 % |
| c3p98 | 300 | 84.5 | 81.7 | -3.3 % | 188.1 | 194.2 | +3.2 % |
| c7p88 | 300 | 86.6 | 84.1 | -2.9 % | 183.5 | 188.5 | +2.7 % |
| c15p3 | 300 | 87.2 | 86.7 | -0.5 % | 187.2 | 192.3 | +2.7 % |
| c15p59 | 300 | 87.1 | 84.3 | -3.2 % | 187.7 | 192.7 | +2.7 % |
| c29p3 | 300 | 87.9 | 85.9 | -2.3 % | 188.8 | 198.4 | +5.1 % |
| c59p2 | 300 | 93.5 | 90.9 | -2.8 % | 187.9 | 191.7 | +2.0 % |
| c107p2 | 300 | 95.7 | 90.3 | -5.7 % | 185.3 | 198.4 | +7.1 % |

Times in milliseconds.
