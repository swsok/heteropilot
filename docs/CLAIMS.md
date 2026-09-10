# What HeteroPilot can claim, as of 2026-09-04

Input to the paper outline. **No new numbers**: every figure here points at a
committed artifact, and every claim carries the label that artifact earns.

Written at `main` = `8bb2f6f`, after the consolidation sprint
(`WORK_ORDER_consolidation.md` STEP 5).

**Labels, used strictly.**

| label | means |
| --- | --- |
| **measured** | read off real silicon on a node `scripts/whichnode.sh` listed at the time |
| **sim-on-measured** | LLMServingSim, driven by a bundle whose latencies are measured |
| **analytical** | LLMServingSim, driven by a datasheet-derived (Tier 0) bundle — never a measurement |

**The one-line summary a reviewer will ask for first — updated 2026-09-09.** A
heterogeneous configuration now wins, **in simulation, at one arrival rate, with an
extrapolated margin**: on `pd-rngd-gpu-card` at 3.3 rps the recommended plan is
`P[cuda:tp1] D[furiosa:tp1]` — an A40 prefill with an RNGD-card decode — at
1.833 tok/J. At 1 rps an NPU-only plan wins (`agg[furiosa:tp1]`, 1.074 tok/J) and
that is the **one cell of sixteen** whose margin rests on a measured accuracy
domain. Above 10 rps the GPU wins everywhere.

So the honest one-liner is now two: *there is a crossover, and almost none of the
curve is measured.* §1.2b and §2 carry both halves; `experiments/results/e6_rps_sweep.md`
has the tables.

> The previous summary read *"no experiment in this repository currently shows a
> heterogeneous configuration winning"*, and was true until E6. It is kept here
> because what changed is the experiment, not the standard of evidence — the
> retracted result §3 describes is still retracted.

---

## 1. Established

Reproducible from the committed artifacts named in each row.

### 1.1 Tier 0/1 — planning on hardware we do not own

| claim | number | label | artifact |
| --- | --- | --- | --- |
| A datasheet-derived profile ranks candidates almost like a measured one | Kendall **τ 0.914** / **0.902** | analytical vs sim-on-measured | `outputs/tier_validation/e1/e1_plan_agreement.json` |
| Exact top-1 agreement is **not** achieved, but the disagreement is cheap | **0.4 %** of the true objective (`llama31-8b`), **11.3 %** (`llama31-8b-light`) | analytical | same |
| Attention is where Tier 0's error lives | overall **38.9 % MAPE** with no anchors | analytical | `docs/tier0_calibration.md` §E2 |
| Anchors spent on attention are what pay | 200 attention anchors → **29.5 %** | calibrated | same |
| GEMM transfers on one scalar efficiency; attention does not | gemm **9.5–11.3 %** MAPE | analytical | same |
| Shape grids are a property of the model, not the hardware | same model, different hardware: **100 %** key overlap | — | `outputs/tier_validation/e3/e3_shape_overlap.json` |
| …so a cross-model shape cache would save almost nothing | normalized overlap **0.8 %** (Qwen3-32B vs Llama-3.1-8B), 2.6 %, **16 % max** | — | same |
| Ascend datasheet uncertainty does not change this cluster's decision | 4 parameters × ±30 %, 13 steps: **the plan never flips** | analytical | `outputs/tier_validation/e4/e4_sensitivity.json` |

**What this licenses.** A Tier 0 plan is a **shortlist, not an oracle**. It may be
presented as "we can rank deployment candidates on hardware we do not own, and the
ranking error costs single-digit percent of the objective" — never as a
performance prediction for that hardware. Every plan built on one carries
`profile_tier: analytical` and a mandatory caveat (D21).

### 1.2 RNGD on real silicon

| claim | number | label | artifact |
| --- | --- | --- | --- |
| Per-PE board power is measured, not assumed | `board = 38.01 + 32.71 × PEs` W, **R² 0.996** | **measured** | `docs/PROJECT_REPORT.md` §4.8.2 |
| The concurrency envelope reaches far past what was previously tested | eff **15.3 → 107.2**, output **585.8 → 1473.3** tok/s per card, zero failures | **measured** | `experiments/results/rngd_concurrency_envelope.md`, `outputs/rngd_envelope/edf/real_c*.json` |
| Scaling flattens sharply inside that range | throughput exponent **0.675** (c16→c32) → **0.241** (c64→c128) | **measured** | same |
| TPOT rises with it | **25.71 → 67.88 ms** across the same range | **measured** | same |
| The simulator is optimistic where the sweeps place RNGD | **1.31×** on throughput (1767 vs 1346) and **18 %** on TPOT at eff 76 | measured vs sim-on-measured | same |
| Rebuilt from the vendor's own profiler, decode prediction is accurate at the fitted concurrency | TPOT **−3.1 %** (was +25.7 %) | measured vs sim-on-measured | `docs/PROJECT_REPORT.md` §4.8.4 |
| TTFT agrees once the arrival patterns match | **−5.1 %** on the mean (was −71.3 %) | measured vs sim-on-measured | §4.8.4, D19 |
| The on-package all-reduce is measured, not inferred | **115 µs** per decoder layer at TP=8 | **measured** | §4.8.5 |
| The envelope now starts at concurrency 1, with power | served **1.00 → 15.59**, 63.1 → 598.2 tok/s, 3000 requests, **zero failures** | **measured** | `experiments/results/rngd_lowload_envelope.md`, `outputs/rngd_envelope_lowload/` |
| Energy per token collapses at low load | **0.418 → 3.941 tok/J**, a **9.4×** span | **measured** | same |
| …and it is throughput, not power, that moves | power spans **1.085×** across a **9.5×** throughput range (139.9–151.8 W) | **measured** | same |
| Utilisation does not explain this card's power | falls 92.1 → 84.7 % while power falls then rises; **r = +0.24** | **measured** | same, `deviations.md` D31 |
| A loaded-but-idle card draws what an empty one draws | **40.0 W** with the model resident, against `idle_power` 39.35 measured at 0 PEs | **measured** | same |
| The two halves of the envelope meet | c16 served **15.3** (2026-08-31) vs **15.59** (2026-09-08), throughput +2.1 % | **measured** | both files above |

### 1.2b The planner prices its own predictor

| claim | number | label | artifact |
| --- | --- | --- | --- |
| The planner rejects D22's committed winner **unaided**, with no manual margin | operating point **74.75** → domain **−17.69 %** → robust TPOT **56.97 ms** > 50 → SLO_VIOLATED | measured domain over a simulated run | `experiments/results/e5_self_rejection.md`, `tests/test_e5_self_rejection.py` |
| …and the recommendation it falls back to | `agg[cuda:tp4]`, **2.595 tok/J** | sim-on-measured | same |
| The margin is read from the run, not set by hand | 0.00 % at 1 rps, 3.05 % at the RNGD card's own operating point, 12.52 % for the cross-vendor P/D plan | measured domain | `experiments/results/e6_rps_sweep.md` |
| Held in CI without a simulator | 199 committed records replayed through a mock predictor | — | `tests/data/e5_sim_records.json` |

### 1.2c Asymmetric TP per phase (`slab3d`, D28)

| claim | number | label | artifact |
| --- | --- | --- | --- |
| The simulator never required uniform instance sizes | `_compute_network_dims` did; `topology_mode: slab3d` expresses `tp_d = 2·tp_p` with **no idle rank** | — | `deviations.md` D28, `docs/d14_spike.md` |
| The encoding changes nothing where it overlaps the old path | R1 ×3, R2 and **R3** (colocated tp4×2, `auto` vs `slab3d`) byte-identical | — | `experiments/scripts/slab3d_anchors.sh` |
| Uncorrected, the 3-D encoding under-predicts TPOT | **−24.4 %** at tp8, **−13.4 %** at tp4 | sim-vs-sim | `docs/slab3d_calibration.md` |
| One dim-1 latency factor removes it | 4 for `[4,2]`, 2 for `[2,2]`, 1 for `[1,2]`; residual **≤ 0.008 %** | fitted | `profiles/calibration/slab3d_latency.yaml` |
| …and it does not depend on bandwidth | same factor at **8 bandwidths over a 13× range** (7.7–100 Gbps) | fitted | same |
| Asymmetric P/D wins where TTFT is tight | `P[cuda:tp2] D[cuda:tp4]`, `P[cuda:tp1] D[cuda:tp2]` at 10 and 20 rps | sim-on-measured | `experiments/results/e6_rps_sweep.md` |

**What this does not license.** Absolute TPOT only at a `(split, link_bw)` in the
table — outside it the compiler refuses (`OUTSIDE_CALIBRATION_DOMAIN`). Not TTFT or
throughput: only TPOT p50 was fitted. And not P/D specifically — the fit is
single-instance and colocated by construction, while a real P/D run also crosses
dim 1 with the prefill compute→sender `COMM_SEND`.

### 1.3 The cross-vendor KV path

| claim | number | label | artifact |
| --- | --- | --- | --- |
| Both legs of the fabric are measured, sustained (not peak) | GPU leg 25.71 GB/s single / 82.63 GB/s ×8; NPU leg 3.77–26.27 GB/s at 1–8 streams | **measured** | `experiments/results/gpu_host_bandwidth.md`, `experiments/results/rngd_parallel_bandwidth.md` |
| Composed fixture links are measured, and the previous value was badly optimistic | **12.6–13.0 GB/s** against a 35 GB/s placeholder — **2.7–4.5× too optimistic** | **measured** | `docs/PROJECT_REPORT.md` §6 |
| …and it changed no prediction | all 16 SLO-sweep winners identical; TPOT moves **+0.012 %** between 35 and 13 GB/s | sim-on-measured | D18 |

That last row is a real result and an uncomfortable one: **these sweeps cannot see
fabric bandwidth**, for three recorded reasons, so they must never be cited as
evidence that it does not matter (D18, and the directive in its commit).

### 1.4 What the SLO sweeps now say

| claim | number | label | artifact |
| --- | --- | --- | --- |
| Under the measured TPOT error, **every** RNGD configuration is rejected at loose TTFT | winner becomes `agg[cuda:tp4]` at **2.595 tok/J** | sim-on-measured | `experiments/results/pd_slo_sweep_margin.md` |
| The result does not depend on the size of the margin | the committed winner cleared the 50 ms TPOT SLO by **1.59 ms**, so **any margin above 3.3 %** rejects it | sim-on-measured | same |
| …nor on how an RNGD accelerator is modelled | both fixtures (card-as-device and 8-PE) converge on the same A40 plan | sim-on-measured | same |
| Those three re-validate on a harness with D25 + D26 fixed | **every reported field identical to the last decimal**, 0 timeouts where there were 71 | sim-on-measured | `docs/d23_revalidation.md` §2 |
| The **tight** TTFT regime is feasible, and P/D disaggregation wins it | all four points FEASIBLE; `P[cuda:tp4] D[cuda:tp4]` at **2.2051 tok/J**, p99 TPOT 37.27, and `agg[cuda:tp2]` at the tightest card point | sim-on-measured | `docs/d23_revalidation.md` §3 |
| RNGD is rejected at tight TTFT too, now measured rather than unevaluated | **0 of 45** RNGD-only and **0 of 18** mixed candidates pass (tp4 fixture); 0 of 12 and 0 of 25 (card) | sim-on-measured | same |
| Tightening TTFT costs energy efficiency | 2.60 tok/J at 4 accelerators (loose) → 1.48–2.21 at 8 (tight) | sim-on-measured | same |

**What this does not license.** The tight winners are **homogeneous** — both P/D
halves are `cuda`. This is the first evaluated statement this repository has about
P/D disaggregation, and it is not a statement about heterogeneity; §2's first row
stands. And every one of these numbers is simulator output calibrated on the
measured TPOT error, not a hardware measurement.

---

## 2. Not established — and why

**E6's crossover is real in simulation and now measured over most of its
range — upgraded 2026-09-11, and the upgrade did not change it.** On
`pd-rngd-gpu-card` the recommended backend goes RNGD → cross-vendor P/D → A40 as
the rate rises from 1 to 10 rps, at both TTFT points. **Nine of sixteen switchover
cells are labelled `measured`**, up from one, after the A40 accuracy domain went
from a single point at served concurrency 170.56 to three
(`experiments/results/a40_lowload_envelope.md`). Every winner and every tok/J is
**unchanged to full precision** — so what the extrapolated cells asserted turned
out to be right, which is a stronger result than the labels alone.

Seven cells still are not measured, and neither group is waiting on effort:

  - **five are `unknown`** on the per-PE `pd-rngd-gpu` fixture, whose RNGD rows
    carry margin **0.00 %** for want of any domain at all — including a 4.956
    tok/J at 10 rps, which is D22's retracted headline reproduced exactly and is
    **not** rehabilitated by appearing here. Only a per-PE RNGD domain fixes
    these, and that needs the NPU node.
  - **two are `extrapolated`** on the card fixture at 3.3 rps, whose winner's A40
    leg is a *prefill role at served concurrency 0.499*. An almost-idle server is
    not an operating point a bench can hold, so this is not measurable by this
    method rather than not yet measured.

**Carried with it, and it bears on any future tight-TTFT claim:** the simulator
is **~18 % optimistic on TTFT at served concurrency 4–11**, where the single
saturated point had the domain declaring 1.97 % everywhere. It changes no E6
outcome because no recommended plan comes within 25 % of its TTFT SLO there —
TPOT binds first — but it is the number a tight-TTFT result would inherit.
`experiments/results/e6_rps_sweep.md`.

**"tokens/J is unimodal in concurrency" — not observed, and the low-end mechanism
is wrong.** `docs/rps_aware_planning_design.md` §1 derives it: idle power dominates
at low load, throughput roll-off starves the numerator at high load, so there is an
interior optimum. On the RNGD card, tokens/J is **monotonically increasing across
the whole measured range [1, 107.2]** — and that survives the most pessimistic power
assumption the hardware permits, since even at the 293 W maximum ever observed the
top point yields 5.03 tok/J against 3.94 at c15.59. The low-end penalty is real and
larger than predicted, but not for the stated reason: the card is not mostly idle at
concurrency 1, it draws **151 W**, within 0.5 % of its draw at c15.59. The design's
deliverable is unaffected — a crossover needs two curves to cross, not a peak on
either — but "each device has a sweet spot" does not hold here.
`experiments/results/rngd_lowload_envelope.md`.

Stated as plainly as §1, because these are the rows a reviewer will find anyway.

**Superseded 2026-09-09 by E6 — see the note at the top of this file and §2.**
On the RPS axis a heterogeneous plan wins at 3.3 rps and an NPU-only plan at 1 rps,
both in simulation and only the latter on a measured margin. The paragraph below
described the state when every sweep ran at a single arrival rate of 10 rps, where
the GPU does still win. It is kept because its two reasons are still the reasons,
and because the second of them was resolved rather than removed.

**No heterogeneous configuration is shown to win.** The GPU wins wherever the
sweeps can speak — which, since 2026-09-07, is everywhere they were asked. Two
separate reasons, and neither is "we did not look":

- The loose-TTFT RNGD energy win was **retracted** — §3.
- The tight-TTFT regime, where P/D disaggregation was the whole argument, is
  **undetermined**: every `pd_*`/`mix_*` candidate livelocks (D23). Not a timeout —
  52,903 progress ticks with prefill pinned at 1 running request, decode never fed,
  memory flat at 9 %, at 1080 / 1800 / 3600 s alike. The same candidate completed
  in **280.6 s** in an earlier committed run, so it is a regression with an open
  cause. Evidence:
  `outputs/pd_slo_sweep_margin18/tight/retry3600_livelock_evidence.txt`.
  **Updated 2026-09-04:** the candidates do *not* livelock — one completes alone in
  343 s at N=300. ASTRA-Sim races on a fixed cwd-relative `tmp__mem/*.json` (13 of
  64 bare processes fail) and the frontend spins forever on the dead child. Both
  bugs are unfixed upstream. The regime is still undetermined, but the reason is
  now a harness fault, not a property of the candidates. D23,
  `docs/d23_spike.md`.

  **Determined 2026-09-07, and it flipped.** The root cause was neither of those:
  `graph_generator.py` invoked the Chakra converter as bare `python`, so the
  workload graph could be built by a `chakra` with the wrong protobuf, and a
  multi-instance run then never finished its first prefill batch (D26). Fixed, the
  tight sweep runs with **0 timeouts** where it had 71 and 126, and **all four tight
  points are FEASIBLE** — homogeneous `cuda` P/D wins three of them
  (`P[cuda:tp4] D[cuda:tp4]`, 2.2051 tok/J, p99 TPOT 37.27). So this regime is no
  longer undetermined, and the sentence this bullet supports needs its second reason
  restated: **it is not that the tight regime cannot speak, but that when it speaks
  it picks a homogeneous configuration.** Filtering all 424 cached candidates
  against the SLO: **0 of 45 RNGD-only and 0 of 18 mixed** pass on the tp4 fixture,
  0 of 12 and 0 of 25 on the card fixture. `docs/d23_revalidation.md` §3.

  Two corrections travel with that. D23's "every `pd_*`/`mix_*` candidate" was wrong
  in both directions — of 197 timeouts **none** was single-instance and **18 were
  `aggregated dp2`**; the discriminator is instance count. And the "regression from
  280.6 s" was not a regression: that run and 17 others reproduce the committed CSV
  byte for byte whenever the converter runs under the right interpreter.

**The shape the industry recommends is not enumerated — but it is no longer
unsimulable.** `A40 tp4 prefill + RNGD tp8 decode` needs asymmetric TP per phase,
and our compiler's topology inference requires uniform instance sizes (D14).
Card-as-device sidesteps it by folding TP=8 inside the device; it does not lift the
constraint. So "heterogeneous P/D does not pay" is still **not** a claim this
repository can make — it has not tested the configuration most likely to pay.

*Updated 2026-09-04.* A spike (`docs/d14_spike.md`) ran that exact configuration to
completion — 86 s, 21 rows, TTFT 2.21× the A40 standalone and TPOT 0.73× the RNGD
standalone. Three things follow, and the third is why this claim stays in §2:

- The constraint is **ours**, in two sites of `serving/core/config_builder.py`, not
  ASTRA-Sim's. Under `auto` the same fixture compiles to a **15-rank topology for
  16 ranks** and hangs the harness rather than erroring — worse than D14 recorded.
- The accuracy cost of the fix is measured: **24.4 %** on TPOT, closing to
  **0.008 %** with a per-dim `link_latency` the config already supports.
- **The prototype is not merged** and the 4× correction is a constant fitted at one
  bandwidth and one split, with no calibration domain. So nothing here licenses a
  number. What has changed is the *reason* the claim is unavailable: from "the
  simulator cannot express it" to "we have not productionised or calibrated it".

**RNGD's low-load regime is unmeasured.** The envelope starts at eff 15.3. The
crossover the design document expects — where the NPU's efficiency advantage
survives a latency constraint — would live below that, and measuring it was
deliberately out of scope for the consolidation sprint. It belongs to
`WORK_ORDER_rps_aware.md`, which is not yet written.
Design: `docs/rps_aware_planning_design.md`.

**The power crossover is a hypothesis, not a measurement.** It is argued in the
design document above; nothing here measures it.

**ATOM is present, partly measured, and not profiled** (D20). Memory (15.047 GiB
largest allocation) and power (idle 19.44 W, active 68.73 W at 95.1 % utilisation)
are measured — `experiments/results/atom_device_facts.md` — but there is no perf
bundle, because host I/O exceeds the kernels and the device tracer's schema is
undocumented. ATOM is excluded from candidate generation and from Exp 4.
`experiments/results/atom_layerwise_blocked.md`.

**Tier 0 is validated for bf16 at the TP degrees tested, and not beyond.** fp8 and
TP ≥ 4 extrapolation is the S5 follow-up in `WORK_ORDER_tiered_profiles.md` and has
not been started.

**D12 remains open and blocks Phase 2.** Prefix-cache memory grows monotonically
until the run dies. Two attempted fixes were wrong and were reverted; `serving/` is
pristine.

---

## 3. Retracted, with the correction

The project's own discipline produced these; each states what was claimed, what
was measured, and what changed.

**D18 — NPU multi-stream bandwidth.** *Claimed:* the NPU leg's parallel-transfer
figures, quoted as the fabric rate. *Measured:* they were best-of-N **peaks** where
a **sustained** rate was needed. *Changed:* the scaling law held, the levels came
down ~25 %, and both fixtures were recomputed from sustained figures on both legs.
The sweeps' numbers did not move — for three reasons that make them blind to
fabric bandwidth, which is itself the finding.

**D19 — the −71 % card TTFT error.** *Claimed:* the card profile mispredicts TTFT
by −71.3 %, attributed to a scheduler difference. *Measured:* the bench harness
ignores `arrival_time_ns` and fires everything at once while the simulator replays
the trace — a burst compared against a spread arrival process, with the whole
difference landing in TTFT. *Changed:* matched arrivals give **−5.1 %**; both TTFT
calibrations were refitted (card fit error 2.34 "unusable" → 0.103).

**D22 — the headline result.** *Claimed:* RNGD beats the A40 on energy by
**1.67×** at loose TTFT (4.956 vs 2.963 tok/J), and "c32 is the highest concurrency
ever run on RNGD". *Measured:* the c32 point was a 24-request pool running at
**effective concurrency 21.2**, and the exponent fitted across it read a
pool-capped ×1.74 interval as a doubling. The envelope actually reaches eff 107.2 —
but at the eff 76 the winner runs at, the simulator is 1.31× optimistic on
throughput and **18 %** on TPOT. *Changed:* re-run with that 18 % as a feasibility
margin, **every RNGD configuration is rejected on both fixtures**. The committed
winner is **infeasible, not merely optimistic**, and the loose-TTFT half of the
three-regime answer does not survive.

**D30 — "regret 0 at every K down to K=1".** *Claimed:* the stage-6 surrogate
top-K costs no optimality — regret 0.000 at every K, 78× fewer simulations — offered
as a general scaling property of the planner. *Measured:* that curve came from a
single fixture (N=78) whose candidates are aggregated-heavy. On three P/D and
heterogeneous corpora (N=324/492/468) the shipped ranker is **false-infeasible at
K=20 on two of three** — it reports no plan where the oracle has one. The cause is
structural, not statistical: the proxy tok/J is **algebraically invariant to TP and
DP** (throughput and power both scale with `tp·dp`, so the ratio cancels), leaving
the ranker blind to the axis that decides feasibility, while its one
parallelism-sensitive term fires for 0 of 324 candidates because the roofline floor
underestimates simulated TPOT by a median 2.92×. *Changed:* **the K=1 claim is
withdrawn as a general property** and holds only of its own fixture; `--top-k` is not
used as a cost lever for P/D or heterogeneous sweeps. **No ranker change was made** —
the obvious repair (order by the roofline floor) fixes two corpora and breaks the
third, which is the same one-fixture error again. `docs/surrogate_topk_regret.md`,
`deviations.md` D30.

**D26 — the retrospective check that came back clean.** *Claimed (implicitly):*
results produced before D26 was found might be contaminated, since a wrong-version
`chakra` silently converted the workload graph and only multi-instance candidates
were affected. *Measured:* the Exp 5 four-combination table reproduces with **zero
differing fields** across all five rows, and D22 §4.4's "RNGD candidates never
terminate at 3.3 rps" was itself the D26 artifact — the same candidates complete in
69–82 s. *Changed:* nothing needed re-running. The check is recorded because a
retrospective that finds nothing is only worth something if it was actually run:
`docs/rps_step0_retro.md`.

**The pattern is worth stating in the paper.** All four were caught by internal
discipline — provenance labels, sustained-vs-peak hygiene, applying a measured
model error as a feasibility margin, and re-measuring an accuracy claim on fixtures
it had not been measured on — not by an external reviewer. The retracted
text is kept in place and marked, never overwritten, because it records what was
reasonable to believe at the time.

---

## 4. Where to look next

`docs/HANDOVER.md` §2 has the priority order. The first item is a work order, not
a run: `WORK_ORDER_rps_aware.md`, which should carry the low-load envelope
measurement, the power crossover, and the D23 diagnosis that currently blocks the
only claim P/D disaggregation had.
