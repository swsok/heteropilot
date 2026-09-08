# Surrogate top-K accuracy (N=492 candidates, oracle goodput/J = -64430.000)

Oracle optimum: `cuda-a40-node_a40a-tp4-dp1-s256-t8192`.

Replayed from `outputs/.hp-pd-slo/cache` -- the corpus, not a fresh sweep, so this speaks about the workload that sweep ran.

## ranker `roofline`  (shipped)

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 98.4x | yes |
| 10 | 0 | 1.000 | 49.2x | yes |
| 20 | 0 | 1.000 | 24.6x | yes |
| 30 | 1 | 0.000 | 16.4x | no |
| 50 | 1 | 0.000 | 9.8x | no |
| 492 | 1 | 0.000 | 1.0x | no |

## ranker `floor`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 98.4x | yes |
| 10 | 0 | 1.000 | 49.2x | yes |
| 20 | 0 | 0.142 | 24.6x | no |
| 30 | 0 | 0.142 | 16.4x | no |
| 50 | 1 | 0.000 | 9.8x | no |
| 492 | 1 | 0.000 | 1.0x | no |

## ranker `floor_then_tpj`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 98.4x | yes |
| 10 | 0 | 1.000 | 49.2x | yes |
| 20 | 0 | 0.142 | 24.6x | no |
| 30 | 0 | 0.142 | 16.4x | no |
| 50 | 1 | 0.000 | 9.8x | no |
| 492 | 1 | 0.000 | 1.0x | no |
