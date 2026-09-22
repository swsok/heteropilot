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
| (c) accuracy domain (committed policy: widen) | 0 | `NONE` | n/a | calibration_condition_mismatch 276, slo_violated 48 |
| (d) accuracy domain, outside_domain: refuse | 0 | `NONE` | n/a | calibration_condition_mismatch 276, outside_calibration_domain 26, slo_violated 22 |

## (b) vs (c): candidates the two conditions judge differently

276 of 324 candidates.

| candidate | served L | b_global18 verdict | tpot margin | c_accuracy_domain verdict | tpot margin |
| --- | ---: | --- | ---: | --- | ---: |
| `cuda-a40-node_a40a-tp1-dp2-s128-t2048` | 88.96 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s128-t8192` | 88.03 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s256-t2048` | 103.62 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s256-t8192` | 102.03 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s32-t2048` | 69.53 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp2-s32-t8192` | 71.30 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s128-t2048` | 62.31 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s128-t8192` | 61.58 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s256-t2048` | 62.31 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s256-t8192` | 61.58 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s32-t2048` | 47.05 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp3-s32-t8192` | 47.01 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp4-s128-t2048` | 42.04 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp4-s128-t8192` | 41.61 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp4-s256-t2048` | 42.04 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp4-s256-t8192` | 41.61 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp4-s32-t2048` | 34.14 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp4-s32-t8192` | 34.52 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s128-t2048` | 154.93 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s128-t8192` | 153.57 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s256-t2048` | 188.62 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s256-t8192` | 186.88 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s32-t2048` | 133.43 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp1-s32-t8192` | 133.41 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp2-s128-t2048` | 77.50 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp2-s128-t8192` | 77.45 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp2-s256-t2048` | 80.18 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp2-s256-t8192` | 80.08 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp2-s32-t2048` | 59.46 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp2-dp2-s32-t8192` | 59.43 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp4-dp1-s128-t2048` | 127.28 | feasible | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 125.26 | feasible | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp4-dp1-s256-t2048` | 157.85 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp4-dp1-s256-t8192` | 153.84 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp4-dp1-s32-t2048` | 115.75 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40a-tp4-dp1-s32-t8192` | 115.65 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s128-t2048` | 88.96 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s128-t8192` | 88.03 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s256-t2048` | 103.62 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s256-t8192` | 102.03 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s32-t2048` | 69.53 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp2-s32-t8192` | 71.30 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s128-t2048` | 62.31 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s128-t8192` | 61.58 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s256-t2048` | 62.31 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s256-t8192` | 61.58 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s32-t2048` | 47.05 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp3-s32-t8192` | 47.01 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp4-s128-t2048` | 42.04 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp4-s128-t8192` | 41.61 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp4-s256-t2048` | 42.04 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp4-s256-t8192` | 41.61 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp4-s32-t2048` | 34.14 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp4-s32-t8192` | 34.52 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s128-t2048` | 154.93 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s128-t8192` | 153.57 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s256-t2048` | 188.62 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s256-t8192` | 186.88 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s32-t2048` | 133.43 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
| `cuda-a40-node_a40b-tp2-dp1-s32-t8192` | 133.41 | slo_violated | 18.00 % | calibration_condition_mismatch | 0.00 % |
... 216 more (see the JSON)

## (c) vs (d): what refusing extrapolation changes

26 of 324 candidates.

| candidate | served L | c_accuracy_domain verdict | tpot margin | d_refuse verdict | tpot margin |
| --- | ---: | --- | ---: | --- | ---: |
| `cuda-a40-node_a40a-tp1-dp1-s128-t2048` | 172.36 | slo_violated | 1.45 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp1-s256-t2048` | 197.88 | slo_violated | 1.63 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40a-tp1-dp1-s256-t8192` | 192.40 | slo_violated | 1.60 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s128-t2048` | 172.36 | slo_violated | 1.45 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s256-t2048` | 197.88 | slo_violated | 1.63 % | outside_calibration_domain | 0.00 % |
| `cuda-a40-node_a40b-tp1-dp1-s256-t8192` | 192.40 | slo_violated | 1.60 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2048` | 144.61 | slo_violated | 60.96 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192` | 143.87 | slo_violated | 60.41 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s256-t2048` | 177.56 | slo_violated | 90.18 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s256-t8192` | 175.41 | slo_violated | 87.95 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s32-t2048` | 139.84 | slo_violated | 57.46 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd0-tp1-dp1-s32-t8192` | 139.81 | slo_violated | 57.44 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s128-t2048` | 144.61 | slo_violated | 60.96 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s128-t8192` | 143.87 | slo_violated | 60.41 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s256-t2048` | 177.56 | slo_violated | 90.18 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s256-t8192` | 175.41 | slo_violated | 87.95 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s32-t2048` | 139.84 | slo_violated | 57.46 % | outside_calibration_domain | 0.00 % |
| `furiosa-rngd-card-node_rngd1-tp1-dp1-s32-t8192` | 139.81 | slo_violated | 57.44 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t2048` | 87.32 | slo_violated | 27.03 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t8192` | 85.96 | slo_violated | 26.40 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t2048` | 100.60 | slo_violated | 33.56 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40a-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 99.21 | slo_violated | 32.84 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t2048` | 87.32 | slo_violated | 27.03 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s128-t8192` | 85.96 | slo_violated | 26.40 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t2048` | 100.60 | slo_violated | 33.56 % | outside_calibration_domain | 0.00 % |
| `mix(cuda-a40-node_a40b-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192` | 99.21 | slo_violated | 32.84 % | outside_calibration_domain | 0.00 % |

## Notes and suggestions emitted

**(c) accuracy domain (committed policy: widen)**
- 6 candidate(s) put A40 at served concurrency 172.36-197.88, outside its measured accuracy domain; those margins are EXTRAPOLATED, not measured
- 20 candidate(s) put RNGD-CARD at served concurrency 85.96-177.56, outside its measured accuracy domain; those margins are EXTRAPOLATED, not measured
- relax the TTFT SLO from 25000ms to at least 36849ms, or add faster prefill capacity
- relax the TPOT SLO from 50ms to at least 88ms, or raise tensor parallelism to cut per-token latency
- lower the admitted request rate from 10.0 to about 1.8 rps
- 276 candidate(s) could not be judged: no margin applies to them, so they are undecidable, not infeasible.
- 276 of them differ from every accuracy domain on a REQUIRED APPLICATION CONDITION (dp (228), islands (132), tp (66)), so no domain may be consulted for them at all (calibration_condition_mismatch). What would settle them is a measurement AT THEIR CONFIGURATION, not a wider load range.
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 1, "pp": 1, "tp": 2}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 1, "pp": 1, "tp": 4}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 1}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 2}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 4}
-   ... and 23 further configuration(s)
- the strongest of them by minimize_energy is mix(furiosa-rngd-card-node_rngd0-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192 (6.035e+04) at served concurrency 74.75 - measure the envelope there to decide it, or set outside_domain: widen_error_bars to accept an extrapolation. Reason: RNGD-CARD: the accuracy domain was measured under different conditions (islands) - islands: domain 1, candidate 2. It may not be consulted here, so this candidate is unmeasured AT ITS OWN CONFIGURATION, not infeasible (S1, D110)

**(d) accuracy domain, outside_domain: refuse**
- relax the TTFT SLO from 25000ms to at least 31759ms, or add faster prefill capacity
- relax the TPOT SLO from 50ms to at least 102ms, or raise tensor parallelism to cut per-token latency
- lower the admitted request rate from 10.0 to about 1.7 rps
- 302 candidate(s) could not be judged: no margin applies to them, so they are undecidable, not infeasible.
- 276 of them differ from every accuracy domain on a REQUIRED APPLICATION CONDITION (dp (228), islands (132), tp (66)), so no domain may be consulted for them at all (calibration_condition_mismatch). What would settle them is a measurement AT THEIR CONFIGURATION, not a wider load range.
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 1, "pp": 1, "tp": 2}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 1, "pp": 1, "tp": 4}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 1}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 2}
-   measure at: {"binding": "unknown", "dp": 1, "hardware": "A40", "islands": 2, "pp": 1, "tp": 4}
-   ... and 23 further configuration(s)
- 26 of them have a domain that MAY be consulted, but their operating point lies past the end of its measured load axis, or their hardware carries no calibration at all (outside_calibration_domain).
- the strongest of them by minimize_energy is furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192 (5.005e+04) at served concurrency 143.9 - measure the envelope there to decide it, or set outside_domain: widen_error_bars to accept an extrapolation. Reason: served concurrency 143.9 on RNGD-CARD outside its accuracy domain [1.02, 76] (policy refuse); measure the envelope at c>=144 or set outside_domain: widen_error_bars to accept an extrapolation

## Provenance

- cache: {'hits': 2592, 'misses': 0}
- git: 0b5961428d3027e09747d16d058ce135c9965011
- node: {'cuda': {'count': 1, 'model': 'NVIDIA RTX A5000'}, 'rngd_cards': None, 'atom_devices': 0}

