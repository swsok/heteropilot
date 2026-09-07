# STEP 0 — the D26 retrospective check

*`WORK_ORDER_rps_aware.md` STEP 0. Run 2026-09-07 on the NPU node, `main` =
`de4d235`. Simulation and code reading only; no code changed. Artifacts:
`outputs/rps_step0/`, `outputs/.hp-reval-pd-combo/`.*

Three results in one sentence each, as §STEP 0's completion condition asks:

1. **D22 §4.4's "RNGD candidates never terminate at 3.3 rps" was a D26 artifact** —
   the same candidates complete in 69–82 s, and 3.3 rps costs no more than 10 rps.
2. **Exp 5's 4-combo table survives the D26 retrospective check** — all five rows
   reproduce with zero differing fields.
3. **The `tok/J` 2.206 vs 2.2051 difference is not planner-side** — the two records
   come from fixtures with different `link_bw` (35.0 against the measured 35.2).

## 0.1 — the 3.3 rps re-test

D22 §4.4 tried to bring per-card concurrency down to the profile's validated range
by lowering the arrival rate, and recorded:

| class | attempted | completed |
| --- | ---: | ---: |
| RNGD P/D | 24 | **0** |
| cross-vendor | 12 | **0** |
| `cuda ↔ cuda` | 84 | 60 |

after 3.7 hours, with the mechanism given as *"a slower arrival rate gives the
decode scheduler less to batch, which lowers throughput, which lengthens the
simulated time needed to drain 300 requests… unbounded in practice"*.

That distribution — everything with an RNGD instance stuck, most `cuda ↔ cuda`
through — is D26's shape, so it was re-tested. One RNGD P/D candidate and one
cross-vendor candidate, each at both rates, 20 requests, under `livelock_watch.sh`:

| candidate | rate | exit | wall | rows | progress ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNGD P/D (`rngd0 tp4 P + rngd1 tp4 D`) | **3.3 rps** | 0 | **74 s** | 21 | 32 |
| same | 10 rps | 0 | 69 s | 21 | 31 |
| cross-vendor (`a40 tp4 P + rngd0 tp4 D`) | **3.3 rps** | 0 | **82 s** | 21 | 30 |
| same | 10 rps | 0 | 69 s | 21 | 29 |

**All four complete**, and the two claims fall separately:

- *"never terminate"* — refuted. These are the classes that went 0-for-36.
- *"the drain is unbounded at a lower rate"* — not supported either. 3.3 rps costs
  **7 % and 19 %** more wall time than 10 rps, and one tick more of simulated time.
  If a lower arrival rate lengthened the drain without bound, 3× the arrival span
  would not land within a fifth of the same cost.

**The caveat, stated because it is real.** This used **20 requests**; the abandoned
run used 300. A drain effect that grows with the request count is not excluded by
this test, and the work order chose 20 deliberately as a cheap check. What is
excluded is a structural non-termination, and the recorded mechanism predicts
trouble at any request count.

**Consequence for E6.** The low-RPS axis is available. §8's contingency — *"raise
the RPS floor to a value that terminates and say so"* — does not need to fire. E6a
should still record wall time per RPS point, because a 7–19 % penalty at 20 requests
says nothing about 300.

D22 §4.4 gets a note above it, not a rewrite (A2).

## 0.2 — Exp 5, re-run

The committed command from `pd_4combo.json`'s own provenance, unchanged —
`--num-requests 120 --seed 42`, same service and cluster specs, same knobs. 23
minutes.

| row | result |
| --- | --- |
| GPU-P + GPU-D | identical |
| GPU-P + NPU-D | identical |
| NPU-P + GPU-D | identical |
| NPU-P + NPU-D | identical |
| aggregated (GPU baseline) | identical |

**Zero differing fields** across `feasible`, `p99_ttft_ms`, `p99_tpot_ms`,
`slo_attainment`, `tokens_per_joule`. The four P/D rows are two-instance
candidates, so they sat in D26's sensitive class and could have moved; they did
not. §8's contingency for a differing Exp 5 does not fire, and no other 8/25–31
multi-instance result needs listing on this account.

Worth keeping in view separately: three of the four P/D rows report the *same*
`tokens_per_joule` to 16 digits, because `ascend-sim-proxy` is a **placeholder**
profile (`npu_is_sim_proxy: true` in the provenance). That degeneracy is a property
of the fixture, not of D26, and it is why this table was never evidence about NPU
P/D.

## 0.3 — 2.206 against 2.2051

The work order asks for a field-by-field diff on the premise that *"the CSV is
identical (R2), so the difference is planner-side — the power block or `link_bw`
(D18)"*. **The premise does not hold.** Diffing the two `recommended` records:

| field | `.hp-pd-slo` (8/26) | reval tight (9/7) |
| --- | --- | --- |
| `plan_id` / `arch` / `backend_mix` / `accelerators` | identical | identical |
| `p99_tpot_ms` | 37.32096584 | 37.271541989999996 |
| `p99_ttft_ms` | 372.3817571399998 | 371.2368359827271 |
| `average_power_w` | 1675.4127450980386 | 1675.8868627450981 |
| `slo_goodput_rps` | 5.784922155199647 | 5.785662871904691 |
| `tokens_per_joule` | 2.205637707948244 | 2.2051282051282053 |

**Everything that differs is a simulation output.** Nothing planner-side moved. And
the two runs used different compiled inputs:

```
outputs/.hp-pd-slo/work/pd_cuda…tp4_P___cuda…tp4_D_-s256-t8192/cluster.json   link_bw = 35.0
outputs/.hp-reval-tight-pd-rngd-gpu/work/<same candidate>/cluster.json         link_bw = 35.2
```

35.2 is the measured composed A40↔A40 value from
`experiments/results/gpu_host_bandwidth.md:156` (`1/(1/70.47 + 1/70.47)`), the same
document recording *"the link went up, 31.2 → 35.2"*. The 8/26 sweep predates it.

**Why "the CSV is identical" looked true.** The R2 anchor is the `.hp-pd-slo`
candidate, at 35.0 — so R2 confirms *that* fixture reproduces, not that the two
sweeps agree. The margin18 candidate has no `sim1.csv` at all: it timed out in the
committed tight run, which is what STEP 3.3 of the previous work order fixed. The
two records were never the same simulation.

So there is nothing to explain beyond a re-measured link bandwidth moving every
simulated metric by 0.02–0.13 %.

## Pulled forward from STEP 3 — the utilisation field exists

§2.1 and §8 flag as an open question whether `furiosa-smi` exposes utilisation, with
a contingency that records `util_pct: null` and states A5(c) as a limitation. It does
not need to fire. `furiosa-smi info --format json` has no utilisation field, but
**`furiosa-smi status --format json` has one per PE**:

```json
{"arch":"rngd","device":"npu0","liveness":"alive",
 "memory":{"DRAM":{"used_size":0,"total_size":51002736640,"used_ratio":0.0}},
 "pe_utilizations":[{"pe_core":0,"pe_occupancy":false,"pe_utilization":0.0}, … 8 entries]}
```

So `power_sampler.sh` polls **both** subcommands in its loop and records a timestamp
for each — `info` for `power`, `status` for `pe_utilizations` and DRAM
`used_ratio`. A5(c) is satisfiable as written, and §4.6's piecewise-in-util power
model has an axis to fit.

Two measurement facts fall out of the probe:

- **Power is quantised to 1 W** (`"38.00 W"`, `"40.00 W"`), which §8 already
  anticipated as a source of repeat variance at low load. At an idle of ~38 W a
  single quantum is 2.6 %, so the 5 % repeat threshold is roughly two quanta.
- **Utilisation is per-PE, not per-card.** A card-as-device measurement with TP=8
  inside has 8 numbers per sample. The envelope's `util_pct` should record the mean
  and the spread, because a partly-idle card and a uniformly-loaded one at the same
  mean are not the same operating point.

## A node fact that STEP 3 must not hardcode

Device names re-enumerated between 2026-09-04 and 2026-09-07 on this node:

| | 2026-09-04 | 2026-09-07 |
| --- | --- | --- |
| sysfs `rngd_mgmt` | `npu0`, `npu1`, **`npu3`** | `npu0`, `npu1`, **`npu2`** |
| card count | 3 | 3 |

`furiosa-smi`, sysfs and `/dev/rngd/` all agree on `npu0/1/2` now. The count did not
change; the **names** did — which is the re-enumeration `docs/nodes/npu.md` warns
about ("the card count has gone 4 → 3 → 4").

So a device-pinned measurement must select on something stable. `furiosa-smi info`
gives `device_sn` (e.g. `RNG26040100181Q`) and `pci_bdf` (`0000:03:00.0`); both
survive re-enumeration, `dev_name` does not. STEP 3's harness should record all
three and select on `device_sn`.
