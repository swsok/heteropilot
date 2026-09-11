# The per-PE RNGD accuracy domain — built without touching the hardware

**Conclusion (i): the five `unknown` cells are closed, and no device was used.**
`docs/HANDOVER.md` §2.2, `docs/CLAIMS.md` §2 and `docs/PAPER_OUTLINE.md` all said
a per-PE RNGD accuracy domain "needs the NPU node". It does not. Per-PE and card
are not two devices; they are two simulator models of **one physical card at
TP=8**, so the measured half already existed and only the simulated half was
missing.

**Conclusion (ii): the domain rejects D22's retracted headline.** E6's per-PE
winner at 10 rps — `agg[furiosa:tp8]` at **4.956 tok/J**, which
`e6_rps_sweep.md` identifies as D22's headline reproduced exactly — is
**infeasible** once its own operating point is priced. Re-ranked, that cell's
winner is `agg[cuda:tp4]` at 2.595 tok/J.

*Measured 2026-09-11 on the NPU node, in simulation only. Nine simulator runs,
300 requests each. Artifacts: `outputs/perpe_lowload_300/`,
`outputs/perpe_highload_300/`, `outputs/e6_perpe_domain/`. Domain:
`profiles/calibration/rngd_perpe.yaml`. Tests:
`tests/test_rngd_perpe_accuracy_domain.py`.*

## Why this needed no hardware

The reasoning that said it did went: the five `unknown` cells sit on the per-PE
fixture, the per-PE profile has no accuracy domain, an accuracy domain is built
from measurements, measurements of RNGD need the NPU node.

The third step is where it fails. An accuracy domain is built from **a measured
curve and a simulated one**, and for RNGD the measured curve was already
committed: `profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml`,
nine points from served concurrency 1.00 to 107.2, measured 2026-08-31 and
2026-09-08. Its `RNGD-CARD` name is the **simulator abstraction it was first used
for**, not a different piece of silicon: it is one card running `furiosa-llm` at
`tensor_parallel_size: 8`, which is exactly what the per-PE fixture
(`experiments/configs/clusters/rngd-llama31-8b-tp8.json`, 8 accelerators at tp8)
models. `experiments/results/rngd_card_vs_pe_model.md` is the precedent and does
the same thing in the other direction — it fits *both* abstractions to one real
furiosa-llm run and compares their errors.

So what was missing was the simulator side under the per-PE fixture. That is
`python -m serving`, which needs no accelerator.

## Method: served-concurrency matching, and why the old recipe could not be used

The RNGD-CARD and A40 domains were built by offering each envelope point's own
arrival rate to the simulator and then checking that both sides landed at the
same **served** concurrency, within 20 %. On those two fixtures the check passed
(gaps −0.8 % to −19 %) and the two pairings coincide.

On the per-PE fixture it fails everywhere. At the rate that puts the hardware at
served 1.00, the per-PE model sits at **1.83** — it is slow enough to back up, so
matching the offered rate cannot also match the operating point, and every point
is refused as `comparable: false`. The recipe measures nothing.

`lowload_sim_error.py --match served` pairs against the measured curve
**interpolated at the simulator's own served concurrency** instead. Two reasons,
and the second is the better one:

- it is the only pairing that yields a point at all on a fixture this far off;
- it is the axis the domain is actually read back on. `_auto_margins` looks the
  error up at the concurrency the **simulation** reports
  (`planner/util/operating_point.py`), not at the one the hardware would have
  reached. It also charges the model for its TPOT error alone instead of summing
  that with its throughput error.

`validity.extrapolation: refuse` reaches up through the pairing: a simulated
point outside the measured range gets `tpot_ref_ms: null` and
`comparable: false`, never an invented reference. All nine points here are inside
it.

## The domain

| offered rps | hardware point | **C_sim** | sim TPOT | measured curve at C_sim | **error** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0967 | c1.00 | 1.832 | 28.91 | 17.72 | **+63.16 %** |
| 0.1687 | c1.99 | 3.221 | 29.26 | 19.02 | +53.86 % |
| 0.3079 | c3.98 | 6.801 | 34.28 | 21.32 | +60.74 % |
| 0.5468 | c7.88 | 12.063 | 34.87 | 24.26 | +43.73 % |
| 0.8978 | c15.3 | 19.471 | 35.21 | 27.75 | +26.86 % |
| 0.9168 | c15.59 | 19.885 | 35.27 | 27.92 | +26.34 % |
| 1.3925 | c29.3 | 52.440 | 71.86 | 41.88 | +71.58 % |
| 1.9571 | c59.2 | 70.843 | 72.27 | 50.59 | +42.85 % |
| 2.2579 | c107.2 | 79.028 | 72.55 | 54.67 | +32.70 % |

**The sign is the first result.** Every point is positive: the per-PE simulator
is **pessimistic**, predicting a longer TPOT than the card delivers, at every
load measured. `AccuracyDomain.margin_from_error` is one-sided by construction —
`max(0, -err)` — so this hardware is charged **nothing** inside the domain. Four
of the five `unknown` cells therefore keep margin 0.00 %; what changes is that
the zero is now a measurement rather than the absence of one.

**The magnitude falls with load**, +63 % at 1.83 to +26 % at 19.9. That is what a
per-step overhead charged too often looks like as more tokens per forward
amortise it, and `rngd_card_vs_pe_model.md` already named the suspect: ASTRA-Sim
appears to charge ~340 µs/layer for the TP=8 reduction against a measured 115 µs.

**Two features are kept rather than smoothed.** The interval 19.885 → 52.440
crosses the simulator's saturation knee with nothing measured inside it —
simulated TPOT nearly doubles, 35.27 → 71.86 ms, while the hardware curve rises
27.92 → 41.88. And the top three points show the model hitting a throughput wall
the hardware does not: more offered load moves simulated TPOT only 71.86 → 72.55
ms while served concurrency creeps 52.4 → 79.0, so the simulator saturates near
1.4 rps where the card reaches 2.26.

### An independent check that is not a point

The per-PE model was first validated in `rngd_sim_vs_real_summary.md` against a
real furiosa-llm run of 20 requests. Recomputed here from
`outputs/rngd_bench/real_tp8.json` and `outputs/envcheck/rngd_verify_tp8.csv`:

```
real : served conc 17.738   TPOT p50 28.664 ms   (wall 23.612 s, 20/20 ok)
sim  : served conc 17.342   TPOT p50 35.809 ms   →  +24.93 %
```

Interpolating the nine points at 17.34 gives about **+30 %**, so two campaigns on
two datasets and two harnesses agree to ~5 pp. It is recorded as
`fitted_at_concurrency` and as a check, **not** as a tenth point: it was measured
on `outputs/envcheck/rngd20.jsonl`, a different token distribution from the
envelope's sharegpt workload, and putting two workloads on one interpolation axis
is precisely what `lowload_sim_error.py --dataset` refuses.

## What the five cells become

The RNGD operating points E6 recommends on the per-PE fixture, read off
`outputs/e6_rerun/pd-rngd-gpu/pd_slo_sweep.json`:

| cell | RNGD served conc | in domain | margin | outcome |
| --- | ---: | --- | ---: | --- |
| 1 rps / 64 s | 19.634 | yes | 0.00 % | `unknown` → **`measured`**, plan unchanged |
| 1 rps / 8 s | 19.634 | yes | 0.00 % | `unknown` → **`measured`**, plan unchanged |
| 3.3 rps / 64 s | 73.679 | yes | 0.00 % | `unknown` → **`measured`**, plan unchanged |
| 3.3 rps / 8 s | 73.679 | yes | 0.00 % | `unknown` → **`measured`**, plan unchanged |
| **10 rps / 64 s** | **139.367** | **no** | **42.12 %** | **winner rejected** |

**The four that only needed a label were not re-simulated, and do not need to
be.** Their margin is 0, and a margin can only *inflate* a predicted TPOT — it
removes candidates from the feasible set and never promotes one. With the
recommended plan itself untouched, the top of the ranking cannot move.

**The fifth was re-ranked, and the winner changed.** At served concurrency 139.4
the leg is far above the 79.03 where the simulator was last measured, so
`widen_error_bars` extrapolates down a negative slope and charges 42.12 %:

```
48.355 ms  x  1.4212  =  68.7 ms     against a 50 ms p99 TPOT SLO
```

| 10 rps / 64 s, per-PE fixture | before | after |
| --- | --- | --- |
| recommended | `agg[furiosa:tp8]` | **`agg[cuda:tp4]`** |
| tok/J | **4.956** | **2.595** |
| accelerators | 8 | 4 |
| p99 TPOT | 48.355 ms | 35.582 ms |
| validity | `unknown` | **`measured`** |
| margin | 0.00 %, source `manual` | 1.11 %, source `accuracy_domain` |

The rejected candidate was **evaluated, not skipped**: its cache entry
(`furiosa-rngd-node_rngd0-tp8-dp1-s128-t2048`, p99 TPOT 48.355 ms, 4.956 tok/J,
RNGD@139.367) was served from the replay cache in the re-run, given its margin,
and failed the TPOT SLO. Its TTFT, 29.5 s against a 64 s bound, is not the
binding constraint.

**This is E5's pattern reaching the half of E6 that E5 could not.** E5 showed the
planner self-rejecting D22's winner on the *card* fixture. The per-PE fixture
reproduced the same 4.956 tok/J headline and survived only because nothing could
size its margin. It no longer does.

### What this does not say

The domain measures **TPOT**, and only TPOT — the bench side of the envelope is
closed-loop and the simulator replays an arrival process, so TTFT is not
comparable (D19). It carries **no** statement about the energy axis. A reader
tempted to run the sign backwards — "the simulator is pessimistic, so real RNGD
is better than 4.956 tok/J" — is making an energy claim this file does not
support. `docs/CLAIMS.md` §3 stands: D22's headline is retracted, and it is now
retracted by the planner as well as by hand.

## A correction that came out of it — the replay cache does not cost seconds

`outputs/e6_rerun_cache_backup/README.md`, `docs/deviations.md` D32 and PR #74
all say that re-ranking E6 under a changed accuracy domain is "a replay costing
seconds". Measured while doing exactly that, it is not.

The cache stores **successful** simulations only. Every candidate that dies in
the simulator's KV allocator is re-run on every replay, and there are 114 of them
per point at 10 rps and **291** at 1 rps. A replay of the 10 rps row with a warm
cache took **2.7 minutes** (243 of 357 served from cache); an attempted replay of
the 1 rps row was still simulating after 10 minutes and its original run took
150. The cache is real and it is worth its 217 KB — it is what made the 10 rps
re-rank affordable — but the claim should read *minutes to hours, not seconds*,
and a full eight-row re-rank of one fixture is a multi-hour job.

This is why only the one cell that could change was re-ranked, and why the
argument above that the other four cannot change is load-bearing rather than a
convenience.

## D32's RNGD half stops being unchecked

D32 recorded that the four committed RNGD low-load points were taken at the
script's 20-request default, that two further points had been discarded at a
"40 % below the hardware" gap, and that the split between a real throughput error
and the drain-tail artifact "has not been measured" on the RNGD side.

The same sweep run both ways measures it:

| hardware point | C_sim @ 20 req | C_sim @ 300 req | difference |
| ---: | ---: | ---: | ---: |
| c15.3 | 10.82 | **19.47** | +80 % |
| c15.59 | 10.92 | **19.88** | +82 % |

Nothing physical differs between the columns. A 20-request run at 0.9 rps spans
22 s against a ~20 s mean latency, so the drain tail is half the wall — the same
mechanism D32 measured at −31.7 % on the A40, here worth a factor of 1.8. Every
point in the per-PE domain is from the 300-request runs;
`outputs/perpe_lowload_shape/` keeps the 20-request sweep as the evidence for
this section and is the basis of nothing.

**This does not by itself rehabilitate the two points D32 says may have been
discarded for the wrong reason** — those are on the *card* fixture and remain
untested. But it removes the last reason to doubt that the artifact is large
enough to explain them.
