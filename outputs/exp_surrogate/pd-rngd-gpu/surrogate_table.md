# Surrogate top-K accuracy (N=324 candidates, oracle goodput/J = -64430.000)

Oracle optimum: `cuda-a40-node_a40a-tp4-dp1-s256-t8192`.

Replayed from `outputs/perf/topk/cache_full` -- the corpus, not a fresh sweep, so this speaks about the workload that sweep ran.

## ranker `roofline`  (shipped)

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 64.8x | yes |
| 10 | 0 | 1.000 | 32.4x | yes |
| 20 | 0 | 1.000 | 16.2x | yes |
| 30 | 1 | 0.000 | 10.8x | no |
| 40 | 1 | 0.000 | 8.1x | no |
| 60 | 1 | 0.000 | 5.4x | no |
| 324 | 1 | 0.000 | 1.0x | no |

## ranker `floor`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 64.8x | yes |
| 10 | 0 | 0.142 | 32.4x | no |
| 20 | 1 | 0.000 | 16.2x | no |
| 30 | 1 | 0.000 | 10.8x | no |
| 40 | 1 | 0.000 | 8.1x | no |
| 60 | 1 | 0.000 | 5.4x | no |
| 324 | 1 | 0.000 | 1.0x | no |

## ranker `floor_then_tpj`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 64.8x | yes |
| 10 | 0 | 0.142 | 32.4x | no |
| 20 | 1 | 0.000 | 16.2x | no |
| 30 | 1 | 0.000 | 10.8x | no |
| 40 | 1 | 0.000 | 8.1x | no |
| 60 | 1 | 0.000 | 5.4x | no |
| 324 | 1 | 0.000 | 1.0x | no |

## ranker `tpj_then_floor`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 64.8x | yes |
| 10 | 0 | 1.000 | 32.4x | yes |
| 20 | 0 | 1.000 | 16.2x | yes |
| 30 | 1 | 0.000 | 10.8x | no |
| 40 | 1 | 0.000 | 8.1x | no |
| 60 | 1 | 0.000 | 5.4x | no |
| 324 | 1 | 0.000 | 1.0x | no |
