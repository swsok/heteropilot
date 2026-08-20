| fabric BW (GB/s) | recommended arch | recommended id | rec p99 TTFT (ms) | best-P/D feasible | best-P/D p99 TTFT (ms) | P/D xfer p99 (ms) |
| ---: | --- | --- | ---: | :---: | ---: | ---: |
| 400 | pd_split | pd(cuda-rtxpro6000-node0-tp1-dp1 P + cuda-rtxpro6000-node1-tp1-dp1 D)-s128-t2048 | 134.4 | yes | 134.4 | 1.0 |
| 200 | pd_split | pd(cuda-rtxpro6000-node0-tp1-dp1 P + cuda-rtxpro6000-node1-tp1-dp1 D)-s128-t2048 | 135.3 | yes | 135.3 | 1.9 |
| 100 | pd_split | pd(cuda-rtxpro6000-node0-tp1-dp1 P + cuda-rtxpro6000-node1-tp1-dp1 D)-s128-t2048 | 137.2 | yes | 137.2 | 3.8 |
| 25 | pd_split | pd(cuda-rtxpro6000-node0-tp1-dp1 P + cuda-rtxpro6000-node1-tp1-dp1 D)-s128-t2048 | 148.5 | yes | 148.5 | 15.1 |
| 10 | aggregated | mix(cuda-rtxpro6000-node0-tp1-dp1+cuda-rtxpro6000-node1-tp1-dp1)-s128-t2048 | 108.8 | no | - | - |
| 1 | aggregated | mix(cuda-rtxpro6000-node0-tp1-dp1+cuda-rtxpro6000-node1-tp1-dp1)-s128-t2048 | 108.8 | no | - | - |
