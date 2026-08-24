# Router baselines (candidate `mix(cuda-rtx-a5000-node0-tp1-dp2+cuda-rtxpro6000-node1-tp1-dp2)-s256-t2048`)

| policy | p50/p99 TTFT (ms) | p50/p99 TPOT (ms) | throughput (tok/s) | SLO attain | goodput (rps) | tok/J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RR | 78.9 / 344.3 | 22.3 / 35.3 | 5,340.0 | 1.00 | 3.68 | 1.238 |
| RAND | 102.7 / 644.3 | 31.3 / 37.0 | 5,168.9 | 1.00 | 3.56 | 1.249 |
| LOAD | 67.2 / 314.3 | 15.1 / 34.2 | 5,452.9 | 1.00 | 3.75 | 1.244 |
