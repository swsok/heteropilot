# Surrogate top-K accuracy (N=468 candidates, oracle goodput/J = -60350.000)

Oracle optimum: `mix(furiosa-rngd-card-node_rngd0-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192`.

Replayed from `outputs/.hp-reval-margin18-pd-rngd-gpu-card/cache` -- the corpus, not a fresh sweep, so this speaks about the workload that sweep ran.

## ranker `roofline`  (shipped)

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 0.001 | 93.6x | no |
| 10 | 1 | 0.000 | 46.8x | no |
| 20 | 1 | 0.000 | 23.4x | no |
| 30 | 1 | 0.000 | 15.6x | no |
| 50 | 1 | 0.000 | 9.4x | no |
| 468 | 1 | 0.000 | 1.0x | no |

## ranker `floor`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 93.6x | yes |
| 10 | 0 | 1.000 | 46.8x | yes |
| 20 | 0 | 0.219 | 23.4x | no |
| 30 | 0 | 0.068 | 15.6x | no |
| 50 | 0 | 0.068 | 9.4x | no |
| 468 | 1 | 0.000 | 1.0x | no |

## ranker `floor_then_tpj`

| K | recall@K | regret@K | speedup | false-infeasible |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0 | 1.000 | 93.6x | yes |
| 10 | 0 | 1.000 | 46.8x | yes |
| 20 | 0 | 0.219 | 23.4x | no |
| 30 | 0 | 0.068 | 15.6x | no |
| 50 | 0 | 0.068 | 9.4x | no |
| 468 | 1 | 0.000 | 1.0x | no |
