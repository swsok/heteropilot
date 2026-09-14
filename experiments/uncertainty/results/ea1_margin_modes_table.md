# E-A1 — scalar margin vs per-candidate margin (STEP A5)

Fixture: `experiments/configs/clusters/pd-rngd-gpu-card.yaml` + `examples/service_specs/llama31-8b.yaml`, 300 requests, seed 42.
Canonical bucket: `in_lt1024-out_ge512-rps_lt20`; shape `in_lt1024-out_ge512`.
Domains (profiles/calibration/*.yaml): `A40` [4.043, 170.6] (3 pts, widen_error_bars), `RNGD` [1.832, 79.03] (9 pts, widen_error_bars), `RNGD-CARD` [1.02, 76] (9 pts, widen_error_bars).

One set of simulations serves all four conditions - only the verdict rule
differs - so the comparison is of decision rules, not of predictions.
Operating points filled from the cached per-island served concurrency: 324 candidates; from the run-level figure: 0 (see the script docstring).

## Headline

| condition | feasible | recommended | tokens/J | rejected |
| --- | ---: | --- | ---: | --- |
| (a) no margin | 70 | `mix(furiosa-rngd-card-node_rngd0-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 3.1635 | slo_violated 254 |
| (b) global 18 % (D22) | 10 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 2.5954 | slo_violated 314 |
| (c) accuracy domain (committed policy: widen) | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 2.5954 | slo_violated 274 |
| (d) accuracy domain, outside_domain: refuse | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 2.5954 | outside_calibration_domain 30, slo_violated 244 |

## (b) vs (c): candidates the two conditions judge differently

40 of 324 candidates.

| candidate | served L | b_global18 verdict | tpot margin | c_accuracy_domain verdict | tpot margin |
| --- | ---: | --- | ---: | --- | ---: |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 28.09 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 28.12 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp3)-s32-t2048` | 28.09 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp3)-s32-t8192` | 28.12 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 23.52 | slo_violated | 18.00 % | feasible | 0.41 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 23.34 | slo_violated | 18.00 % | feasible | 0.41 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp2)-s32-t2048` | 28.09 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp2)-s32-t8192` | 28.12 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp3)-s32-t2048` | 23.52 | slo_violated | 18.00 % | feasible | 0.41 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp3)-s32-t8192` | 23.34 | slo_violated | 18.00 % | feasible | 0.41 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s128-t2048` | 20.98 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s128-t8192` | 20.41 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s256-t2048` | 20.98 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s256-t8192` | 20.41 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 19.60 | slo_violated | 18.00 % | feasible | 0.38 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 19.40 | slo_violated | 18.00 % | feasible | 0.38 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp1)-s32-t2048` | 28.09 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp1)-s32-t8192` | 28.12 | slo_violated | 18.00 % | feasible | 0.44 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp2)-s32-t2048` | 23.52 | slo_violated | 18.00 % | feasible | 0.41 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp2)-s32-t8192` | 23.34 | slo_violated | 18.00 % | feasible | 0.41 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s128-t2048` | 20.98 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s128-t8192` | 20.41 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s256-t2048` | 20.98 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s256-t8192` | 20.41 | slo_violated | 18.00 % | feasible | 0.39 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s32-t2048` | 19.60 | slo_violated | 18.00 % | feasible | 0.38 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s32-t8192` | 19.40 | slo_violated | 18.00 % | feasible | 0.38 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s128-t2048` | 17.27 | slo_violated | 18.00 % | feasible | 0.36 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s128-t8192` | 17.51 | slo_violated | 18.00 % | feasible | 0.37 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s256-t2048` | 17.27 | slo_violated | 18.00 % | feasible | 0.36 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s256-t8192` | 17.51 | slo_violated | 18.00 % | feasible | 0.37 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 17.59 | slo_violated | 18.00 % | feasible | 0.37 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 17.72 | slo_violated | 18.00 % | feasible | 0.37 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s128-t2048` | 50.28 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s128-t8192` | 50.23 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s256-t2048` | 50.28 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s256-t8192` | 50.23 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s128-t2048` | 50.28 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s128-t8192` | 50.23 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s256-t2048` | 50.28 | slo_violated | 18.00 % | feasible | 0.59 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s256-t8192` | 50.23 | slo_violated | 18.00 % | feasible | 0.59 % |

## (c) vs (d): what refusing extrapolation changes

30 of 324 candidates.

| candidate | served L | c_accuracy_domain verdict | tpot margin | d_refuse verdict | tpot margin |
| --- | ---: | --- | ---: | --- | ---: |
| `cuda-a40-node_a40a-tp1-dp1-s128-t2048` | 172.36 | slo_violated | 1.43 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp1-s256-t2048` | 197.88 | slo_violated | 1.61 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp1-s256-t8192` | 192.40 | slo_violated | 1.57 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s256-t2048` | 188.62 | slo_violated | 1.54 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s256-t8192` | 186.88 | slo_violated | 1.53 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s128-t2048` | 172.36 | slo_violated | 1.43 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s256-t2048` | 197.88 | slo_violated | 1.61 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s256-t8192` | 192.40 | slo_violated | 1.57 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s256-t2048` | 188.62 | slo_violated | 1.54 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s256-t8192` | 186.88 | slo_violated | 1.53 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2048` | 144.61 | slo_violated | 37.87 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192` | 143.87 | slo_violated | 37.66 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s256-t2048` | 177.56 | slo_violated | 47.42 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s256-t8192` | 175.41 | slo_violated | 46.79 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s32-t2048` | 139.84 | slo_violated | 36.49 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s32-t8192` | 139.81 | slo_violated | 36.48 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s128-t2048` | 144.61 | slo_violated | 37.87 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s128-t8192` | 143.87 | slo_violated | 37.66 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s256-t2048` | 177.56 | slo_violated | 47.42 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s256-t8192` | 175.41 | slo_violated | 46.79 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s32-t2048` | 139.84 | slo_violated | 36.49 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s32-t8192` | 139.81 | slo_violated | 36.48 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t2048` | 87.32 | slo_violated | 21.28 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t8192` | 85.96 | slo_violated | 20.89 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t2048` | 100.60 | slo_violated | 25.13 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 99.21 | slo_violated | 24.72 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t2048` | 87.32 | slo_violated | 21.28 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t8192` | 85.96 | slo_violated | 20.89 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t2048` | 100.60 | slo_violated | 25.13 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 99.21 | slo_violated | 24.72 % | outside_calibration_domain | 0.00 % |

## Notes and suggestions emitted

**(c) accuracy domain (committed policy: widen)**
- 10 candidate(s) put A40 at served concurrency 172.36-197.88, outside its measured accuracy domain; those margins are EXTRAPOLATED, not measured
- 20 candidate(s) put RNGD-CARD at served concurrency 85.96-177.56, outside its measured accuracy domain; those margins are EXTRAPOLATED, not measured

**(d) accuracy domain, outside_domain: refuse**
- 30 candidate(s) could not be judged: their operating point lies outside every measured accuracy domain (or their hardware has none), so no margin applies. They are undecidable, not infeasible (outside_calibration_domain).
- the strongest of them by minimize_energy is furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192 (5.005e+04) at served concurrency 143.9 - measure the envelope there to decide it, or set outside_domain: widen_error_bars to accept an extrapolation. Reason: served concurrency 143.9 on RNGD-CARD outside its accuracy domain [1.02, 76] (policy refuse); measure the envelope at c>=144 or set outside_domain: widen_error_bars to accept an extrapolation

## Provenance

- cache: {'hits': 2592, 'misses': 0}
- git: f0b6b3d9abfba65d4e6ce6e41b0145a244c8e4e1
- node: {'cuda': None, 'rngd_cards': None, 'atom_devices': 0}

