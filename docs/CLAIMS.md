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

**The one-line summary a reviewer will ask for first: no experiment in this
repository currently shows a heterogeneous configuration winning.** §3 explains
what happened to the result that used to.

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

---

## 2. Not established — and why

Stated as plainly as §1, because these are the rows a reviewer will find anyway.

**No heterogeneous configuration is shown to win.** The GPU wins wherever the
sweeps can speak at all. Two separate reasons, and neither is "we did not look":

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

**The pattern is worth stating in the paper.** All three were caught by internal
discipline — provenance labels, sustained-vs-peak hygiene, and applying a measured
model error as a feasibility margin — not by an external reviewer. The retracted
text is kept in place and marked, never overwritten, because it records what was
reasonable to believe at the time.

---

## 4. Where to look next

`docs/HANDOVER.md` §2 has the priority order. The first item is a work order, not
a run: `WORK_ORDER_rps_aware.md`, which should carry the low-load envelope
measurement, the power crossover, and the D23 diagnosis that currently blocks the
only claim P/D disaggregation had.
