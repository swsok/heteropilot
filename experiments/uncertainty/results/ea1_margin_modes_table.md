# E-A1 — scalar margin vs per-candidate margin (STEP A5)

Fixture: `experiments/configs/clusters/pd-rngd-gpu-card.yaml` + `examples/service_specs/llama31-8b.yaml`, 300 requests, seed 42.
Canonical bucket: `in_lt1024-out_ge512-rps_lt20`; shape `in_lt1024-out_ge512`.
Domains (profiles/calibration/*.yaml): `A40` [4.043, 170.6] (3 pts, widen_error_bars), `RNGD` [1.832, 79.03] (9 pts, widen_error_bars), `RNGD-CARD` [1.02, 76] (9 pts, widen_error_bars).

One set of simulations serves all four conditions - only the verdict rule
differs - so the comparison is of decision rules, not of predictions.
Operating points filled from the run-level served concurrency: 324 candidates (see the script docstring).

## Headline

| condition | feasible | recommended | tokens/J | rejected |
| --- | ---: | --- | ---: | --- |
| (a) no margin | 70 | `mix(furiosa-rngd-card-node_rngd0-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 3.1635 | slo_violated 254 |
| (b) global 18 % (D22) | 10 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 2.5954 | slo_violated 314 |
| (c) accuracy domain (committed policy: widen) | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 2.5954 | slo_violated 274 |
| (d) accuracy domain, outside_domain: refuse | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 2.5954 | outside_calibration_domain 154, slo_violated 120 |

## (b) vs (c): candidates the two conditions judge differently

40 of 324 candidates.

| candidate | served L | b_global18 verdict | tpot margin | c_accuracy_domain verdict | tpot margin |
| --- | ---: | --- | ---: | --- | ---: |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 137.82 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 137.22 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp3)-s32-t2048` | 137.82 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp3)-s32-t8192` | 137.23 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 132.20 | slo_violated | 18.00 % | feasible | 1.16 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 130.51 | slo_violated | 18.00 % | feasible | 1.14 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp2)-s32-t2048` | 137.82 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp2)-s32-t8192` | 137.23 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp3)-s32-t2048` | 132.20 | slo_violated | 18.00 % | feasible | 1.16 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp3)-s32-t8192` | 130.51 | slo_violated | 18.00 % | feasible | 1.14 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s128-t2048` | 139.69 | slo_violated | 18.00 % | feasible | 1.21 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s128-t8192` | 138.20 | slo_violated | 18.00 % | feasible | 1.20 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s256-t2048` | 139.69 | slo_violated | 18.00 % | feasible | 1.21 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s256-t8192` | 138.20 | slo_violated | 18.00 % | feasible | 1.20 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 132.48 | slo_violated | 18.00 % | feasible | 1.16 % |
| `mix(cuda-a40-node_a40a-tp1-dp3+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 129.96 | slo_violated | 18.00 % | feasible | 1.14 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp1)-s32-t2048` | 137.82 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp1)-s32-t8192` | 137.22 | slo_violated | 18.00 % | feasible | 1.19 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp2)-s32-t2048` | 132.20 | slo_violated | 18.00 % | feasible | 1.16 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp2)-s32-t8192` | 130.51 | slo_violated | 18.00 % | feasible | 1.14 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s128-t2048` | 139.69 | slo_violated | 18.00 % | feasible | 1.21 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s128-t8192` | 138.20 | slo_violated | 18.00 % | feasible | 1.20 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s256-t2048` | 139.69 | slo_violated | 18.00 % | feasible | 1.21 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s256-t8192` | 138.20 | slo_violated | 18.00 % | feasible | 1.20 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s32-t2048` | 132.48 | slo_violated | 18.00 % | feasible | 1.16 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp3)-s32-t8192` | 129.96 | slo_violated | 18.00 % | feasible | 1.14 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s128-t2048` | 134.69 | slo_violated | 18.00 % | feasible | 1.17 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s128-t8192` | 134.90 | slo_violated | 18.00 % | feasible | 1.17 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s256-t2048` | 134.69 | slo_violated | 18.00 % | feasible | 1.17 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s256-t8192` | 134.90 | slo_violated | 18.00 % | feasible | 1.17 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s32-t2048` | 133.88 | slo_violated | 18.00 % | feasible | 1.17 % |
| `mix(cuda-a40-node_a40a-tp1-dp4+cuda-a40-node_a40b-tp1-dp4)-s32-t8192` | 133.85 | slo_violated | 18.00 % | feasible | 1.17 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s128-t2048` | 146.95 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s128-t8192` | 146.74 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s256-t2048` | 146.95 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp1+cuda-a40-node_a40b-tp2-dp2)-s256-t8192` | 146.74 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s128-t2048` | 146.95 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s128-t8192` | 146.74 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s256-t2048` | 146.95 | slo_violated | 18.00 % | feasible | 1.26 % |
| `mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s256-t8192` | 146.74 | slo_violated | 18.00 % | feasible | 1.26 % |

## (c) vs (d): what refusing extrapolation changes

154 of 324 candidates.

| candidate | served L | c_accuracy_domain verdict | tpot margin | d_refuse verdict | tpot margin |
| --- | ---: | --- | ---: | --- | ---: |
| `cuda-a40-node_a40a-tp1-dp1-s128-t2048` | 172.36 | slo_violated | 1.43 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp1-s256-t2048` | 197.88 | slo_violated | 1.61 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp1-s256-t8192` | 192.40 | slo_violated | 1.57 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s128-t2048` | 173.46 | slo_violated | 1.44 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s128-t8192` | 171.85 | slo_violated | 1.43 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s256-t2048` | 204.37 | slo_violated | 1.65 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s256-t8192` | 200.56 | slo_violated | 1.63 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s128-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s128-t8192` | 177.97 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s256-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s256-t8192` | 177.97 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s256-t2048` | 188.62 | slo_violated | 1.54 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s256-t8192` | 186.88 | slo_violated | 1.53 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s128-t2048` | 172.36 | slo_violated | 1.43 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s256-t2048` | 197.88 | slo_violated | 1.61 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s256-t8192` | 192.40 | slo_violated | 1.57 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s128-t2048` | 173.46 | slo_violated | 1.44 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s128-t8192` | 171.85 | slo_violated | 1.43 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s256-t2048` | 204.37 | slo_violated | 1.65 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s256-t8192` | 200.56 | slo_violated | 1.63 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s128-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s128-t8192` | 177.97 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s256-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s256-t8192` | 177.97 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
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
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp1)-s128-t2048` | 173.05 | slo_violated | 1.44 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp1)-s128-t8192` | 174.02 | slo_violated | 1.44 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp1)-s256-t2048` | 206.97 | slo_violated | 1.67 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp1)-s256-t8192` | 199.89 | slo_violated | 1.62 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp2)-s128-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp2)-s128-t8192` | 177.96 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp2)-s256-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+cuda-a40-node_a40b-tp1-dp2)-s256-t8192` | 177.96 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd0-tp1-dp1)-s128-t2048` | 132.05 | slo_violated | 34.23 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd0-tp1-dp1)-s128-t8192` | 131.28 | slo_violated | 34.01 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd0-tp1-dp1)-s256-t2048` | 151.42 | slo_violated | 39.85 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd0-tp1-dp1)-s256-t8192` | 148.93 | slo_violated | 39.13 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd0-tp1-dp1)-s32-t2048` | 121.64 | slo_violated | 31.22 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd0-tp1-dp1)-s32-t8192` | 117.67 | slo_violated | 30.07 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t2048` | 132.05 | slo_violated | 34.23 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t8192` | 131.28 | slo_violated | 34.01 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t2048` | 151.42 | slo_violated | 39.85 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 148.93 | slo_violated | 39.13 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s32-t2048` | 121.64 | slo_violated | 31.22 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s32-t8192` | 117.67 | slo_violated | 30.07 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp1)-s128-t2048` | 178.77 | slo_violated | 1.48 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp2+cuda-a40-node_a40b-tp1-dp1)-s128-t8192` | 177.96 | slo_violated | 1.47 % | outside_calibration_domain | 0.00 % |
... 94 more (see the JSON)

## Notes and suggestions emitted

**(c) accuracy domain (committed policy: widen)**
- 40 candidate(s) put A40 at served concurrency 171.85-206.97, outside its measured accuracy domain; those margins are EXTRAPOLATED, not measured
- 114 candidate(s) put RNGD-CARD at served concurrency 117.67-177.56, outside its measured accuracy domain; those margins are EXTRAPOLATED, not measured

**(d) accuracy domain, outside_domain: refuse**
- 154 candidate(s) could not be judged: their operating point lies outside every measured accuracy domain (or their hardware has none), so no margin applies. They are undecidable, not infeasible (outside_calibration_domain).
- the strongest of them by minimize_energy is furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192 (5.005e+04) at served concurrency 143.9 - measure the envelope there to decide it, or set outside_domain: widen_error_bars to accept an extrapolation. Reason: served concurrency 143.9 on RNGD-CARD outside its accuracy domain [1.02, 76] (policy refuse); measure the envelope at c>=144 or set outside_domain: widen_error_bars to accept an extrapolation

## Provenance

- cache: {'hits': 2592, 'misses': 0}
- git: bb56cf725bcfdb43c19748c6b9ba284b4be9c25e
- node: {'cuda': None, 'rngd_cards': None, 'atom_devices': 0}

