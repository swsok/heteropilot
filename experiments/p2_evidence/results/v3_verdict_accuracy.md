# V3 — verdict accuracy: what the hardware says about the four rules

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V3. **Deployment has not run.**
This file exists now because one of its results does not need a deployment, and
because the candidates and their caveats are fixed before the GPU is touched —
the work order requires the selection to be confirmed first, and it was
(2026-09-16). The measured sections are marked PENDING and are filled by the
deployment run.*

## Status

| section | state |
| --- | --- |
| §1 side result — what holding costs | **complete**, needs no hardware |
| §2 the candidates and their predictions | **complete** (selection confirmed, predictions re-verified) |
| §3 measured verdicts vs rules | **PENDING** — needs A40 GPUs |
| §4 limits | **complete** — two of them found by trying to deploy |
| §5 out of scope | **complete** |
| §6 P1 deployment + SLO sweep | rule side **complete**; measured column **PENDING** |

**Scope closed 2026-09-16.** The deploy backend cannot launch P2 or P3 as
selected (§4①), so V3 measures **P1 only**, extended by an SLO sweep (§6.2) that
recovers the disagreement P3 was chosen to show. P4 was dropped earlier; its
finding is §1.

---

## 1. Side result: on this fixture, holding costs nothing

E-A1's `refuse` policy holds **30** candidates as `outside_calibration_domain`
rather than judging them. The disclosure's §6 asks what that costs — whether the
policy withholds candidates that would have been recommended.

**All 30 are already rejected under rule (a), with no margin applied at all.**

So on this fixture the policy never withholds a candidate the unmargined
simulator would have accepted: every held candidate is one that was going to be
rejected anyway, and holding changes the *reason* recorded, not the outcome.
Their composition is 10 A40-only, 12 RNGD-only and 8 mixed.

The ten A40-only ones sit at served concurrency **172.4 to 197.9**, all above the
A40 domain's top measured point at **170.56**, with predicted p99 TPOT of **90 to
229 ms** against a 50 ms SLO.

**What this does and does not establish.** It establishes that `refuse` is cheap
here — a reader asking "how many good candidates does the refusal throw away?"
gets the answer **zero, on this fixture**. It does not establish that refusing is
*right*, because no rule wanted these candidates; a fixture in which a held
candidate is feasible unmargined would be needed for that, and none exists in the
committed corpora. It is also fixture-specific: the 30 are held because their
operating points exceed the domains' top measured points, which is a property of
where the sweep lands, not of the policy.

**This is why P4 was dropped from the deployment** (user decision, 2026-09-16):
measuring one held candidate would record what the hardware does at an unmeasured
operating point — useful for the domain, and V2's extra `L > 170` stage now picks
that up more cheaply — but it could not have shown the refusal to be wrong.

## 2. The candidates, and the predictions V3 will compare against

Selection and its reasoning: `v3_candidate_selection.md`. SLOs are the fixture's
own: **TPOT 50.0 ms, TTFT 25 000 ms**.

| group | candidate | GPUs | predicted p99 TPOT / TTFT | L | deciding metric | separation |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| P1 | `cuda-a40-node_a40a-tp4-dp1-s128-t2048` | 4 | 36.79 / 16 159 ms | 127.3 | — (all four agree) | — |
| P2 | `mix(a40a-tp2-dp1+a40b-tp2-dp2)-s32-t8192` | 6 | 30.57 / 24 064 ms | 39.8 | TTFT | +3 111.6 ms over |
| P3 | `mix(a40a-tp1-dp1+a40b-tp1-dp4)-s32-t2048` | 5 | 49.16 / 14 755 ms | 28.1 | TPOT | (b) +8.009 over, (c) 0.623 under |

**D40 was checked, not assumed.** Both mix candidates were re-simulated on their
own with the cache disabled, so a key collision was impossible rather than
unlikely — `v3_resimulation.json`, produced by `v3_resimulate.py`. Result below
in §2.1.

**P3's result must be recorded beside V2's run-to-run p99 spread** (user
instruction). Rule (b) is wrong by 8.009 ms if the hardware comes in under 50 ms;
rule (c) is right by only **0.623 ms**. The first claim survives any plausible
repeatability; the second does not survive a spread above ~0.6 ms, and V3 must
say which case it is rather than reporting a bare pass.

**P2 must be driven open loop** at the arrival rate matching its predicted
operating point, not by a closed-loop burst. D19: a closed-loop TTFT is inflated
by queueing and does not transfer, and P2 is decided *on TTFT*.

### 2.1 Re-simulation result — D40 did not fire

Both candidates re-simulated to **byte-identical** metrics, on every compared
field (`p99_tpot_ms`, `p99_ttft_ms`, `p50`/`p95` TPOT, served concurrency,
throughput, energy, tokens/J): worst |Δ| = **0.000000 %** for each.

| group | p99 TPOT cached → resim | p99 TTFT cached → resim | served L |
| --- | --- | --- | ---: |
| P3 | 49.16014071 → 49.16014071 | 14 754.56842926 → 14 754.56842926 | 28.0938 |
| P2 | 30.57248391 → 30.57248391 | 24 063.81209391 → 24 063.81209391 | 39.7532 |

**What this establishes, exactly.** The cached entry each candidate was served is
the one its own configuration produces, so **the predictions V3 compares against
are these candidates' own** — which is all V3 needs. It does **not** establish
that D40 is harmless in general: the mirrors may still share the key, in which
case the entry is this candidate's and it is the *mirror's* cached value that is
wrong. D40 stays open.

Method: `v3_resimulate.py` calls `predict()` directly with **the cache disabled**,
so a key collision is impossible rather than unlikely. Same cluster, same service
spec, same 300-request trace at seed 42 as E-A1. Raw: `v3_resimulation.json`;
run log `experiments/p2_evidence/results/v3_resim.log` (the work dir itself is
regenerable and untracked).

## 3. Measured verdicts against the four rules

**PENDING — needs A40 GPUs**, and reduced in scope: **P1 only** (§4①). The table
is §6.2's, filled from the measurement; §6.1 says what runs.

Per the work order, **these are cases, not statistics** — one candidate read at
six thresholds is one measurement, not a rate, and the specification must present
it as a case table without generalising.

**P2 and P3 are not measured.** Their rows in §2 stay as the record of what was
selected and why it could not be run; nothing in this file should be read as a
measured verdict for either.

## 4. Limits — what V3 cannot settle, and why

Four limits, agreed 2026-09-16 after the deploy attempt. The first two were found
by trying, not by reasoning.

### ① The deploy backend cannot launch P2 or P3 as selected

`planner/deploy/vllm_cuda.py` refuses two shapes outright:

```
multi-island launch (a plan with more than one assignment) needs a router
and is out of scope for this increment; launch a single-island plan

launching dp_replicas>1 or pp_size>1 needs multi-engine orchestration and is
out of scope for this increment
```

Every A40 candidate was scanned against those two conditions:

| group | A40 candidates | deployable | what blocks the rest |
| --- | ---: | ---: | --- |
| P1 | 10 | **4** | — |
| P2 | 8 | 2 | the selected TTFT-decided candidate is `mix(...)` |
| **P3** | **40** | **0** | **all forty are `mix(...)`** |
| P4 | 10 | 10 | (excluded by decision) |

**P3 — the "invention rescues a candidate" case, and the one with the cleanest
separation at 8.009 ms — cannot be deployed at all.** P2's two deployable
candidates are TPOT-decided and separate by 0.05 ms, which ② explains is
unusable. So the deployment reduces to P1.

**This is a property of the Phase 4 increment, not of the invention.** §5 records
it as a work-order candidate.

**The deploy attempt itself is recorded** rather than only its conclusion. P1's
plan dry-ran clean (`cuda-a40-node_a40a` → CUDA 0,1,2,3, TP=4) and the launch
then failed with `Engine core initialization failed`, because another job —
`pretrain_gpt.py`, Megatron-DeepSpeed, running as root — took all eight GPUs
while the server was starting. Nothing of ours was left running; the log is
`outputs/p2_evidence/v3/p1_failed_init_vllm.log`. The measurement is deferred,
not abandoned: **it runs only when the user approves, after 09:00 tomorrow**
(decision 2026-09-16).

### ② On the A40 the margin is smaller than the simulator's own error

The per-point TPOT margin across all 210 A40 candidates runs **0.37–1.63 %**. V2
measured the simulator's TPOT error on this hardware at **1–3 %**
(`v2_openloop_concurrency.md` §5). **The correction is smaller than the thing it
corrects**, so no hardware measurement can separate rule (a) from rule (c) here:
the gap between their verdicts (0.05 ms on the deployable P2) is an order of
magnitude below the prediction error (0.5–1.5 ms at these magnitudes).

It is worth being precise about what this is evidence of. It is **not** a failure
of the per-point margin — it is the margin behaving correctly on a predictor that
is already accurate at these operating points. A margin policy that charged more
than the measured error here would be over-correcting, and the disclosure's
one-sidedness exists to prevent exactly that. **The A40 is the case where the
invention should do almost nothing, and it does almost nothing.** What that case
cannot do is demonstrate the invention's value, which is why ③ matters.

### ③ The verdict-changing cases are on RNGD, and are built from V0 and V1

RNGD-CARD's margin at the sweep's operating point is **~21 %** against the A40's
~1 %. Every case where a per-point margin flips a verdict — including the
disclosure's own §5.6 candidate B, predicted 48.41 ms and robust 59.04 — lives
there. There is no RNGD card on this node.

So the §6 numbers for that regime come from **V0** (the measured error and its
aggregation basis) and **V1** (the registration procedure and the re-judged
worked example), not from V3. `patent2_evidence_map.md` §0 states this, and
**V3-R** — one RNGD candidate each of the P2 and P3 shapes — is left as a
selection step for when an RNGD node exists.

### ④ Holding costs nothing on this fixture

All 30 candidates `refuse` holds are **also rejected under rule (a), with no
margin at all** — §1 above. The policy never withheld a candidate the unmargined
simulator would have accepted, so its cost here is **zero**. It does not follow
that refusing is *right*; no rule wanted those candidates, and no committed
fixture contains a held candidate that is feasible unmargined.

## 5. Out of scope, recorded as a work-order candidate

**A router and multi-engine orchestration** would make P2 and P3 measurable and
would unblock every `mix(...)` and `dp_replicas > 1` candidate — 40 of 40 in P3
alone. `planner/deploy/vllm_cuda.py` names both as deliberate omissions of this
increment.

This work order does **not** cover it and nothing here should be read as starting
it. It is noted so the next planning round has the cost in front of it: without a
router, Phase 4 can deploy only single-island `dp=1` plans, which is a minority of
what the planner recommends.

## 6. What V3 does deliver: P1, and the SLO sweep that makes one deployment count

**PENDING the measurement** — the GPUs are held by another job. Scripts, plans and
the rule side are ready; §6.2's measured column is the only thing outstanding.

### 6.1 The deployment

`cuda-a40-node_a40a-tp4-dp1-s128-t2048`, TP=4 on CUDA devices 0–3, driven **open
loop** at the trace's own arrival rate — the service spec's `arrival_rate_rps:
10`, which is what the simulator was given and therefore what produced the
predicted **L = 127.28**. Three repeats, for the spread. D19 requires the open
loop: P1's TTFT is judged against a 25 s SLO and a closed-loop burst inflates it.

`--ignore-eos` is not optional (V2 §1): without it the engine stops at EOS and
the two sides run different workloads.

Plan `outputs/plans/p2ev-v3-p1.yaml`, runner `experiments/p2_evidence/v3_run_p1.sh`.
Both dry-run validated. A **fresh plan id** is used, because reusing one
overwrites the committed deployment record — the trap V2 hit.

A second deployable P1 candidate, `…-tp4-dp1-s128-t8192`
(`outputs/plans/p2ev-v3-p1b.yaml`, L = 125.26), runs if GPU time allows. **It is
not a different operating point.** All four deployable P1 candidates sit at
L ≈ 125–127; the widest spread among them is **2.0 (1.6 %)**. What the second run
buys is an independent sample at a second knob setting, not a second load.

### 6.2 The SLO sweep

One candidate at one SLO is one cell, and four rules that agree on it say
nothing. Sweeping the TPOT SLO turns the single deployment into a grid: the
rules' robust values are fixed, the threshold moves, and the measurement says
which side the hardware landed on.

**The SLO is a post-hoc parameter here.** E-A1's 50 ms aggregate is **not**
re-scored — this asks what the same four rules would have decided at other
thresholds, which is a sensitivity analysis. Nothing in `outputs/uncertainty/ea1/`
is touched.

P1's robust values (`v3_slo_sweep.py`, raw in `v3_slo_sweep.json`):

| rule | TPOT margin | robust TPOT | robust TTFT |
| --- | ---: | ---: | ---: |
| (a) no margin | 0.0000 % | **36.79452 ms** | 16 159.5 |
| (b) global 18 % | 18.0000 % | **43.41754 ms** | 19 068.2 |
| (c) per-point | 1.1347 % | **37.21204 ms** | 17 206.1 |
| (d) per-point + refuse | 1.1347 % | 37.21204 ms | 17 206.1 |

**TTFT never binds**: the largest margin any rule applies puts it at 19 068 ms
against a 25 000 ms SLO, so TPOT alone decides every cell. That is asserted in
the code, not assumed.

| TPOT SLO | (a) | (b) | (c) | (d) | measured ≤ SLO? |
| ---: | --- | --- | --- | --- | --- |
| 38 | feasible | **rejected** | feasible | feasible | PENDING |
| 40 | feasible | **rejected** | feasible | feasible | PENDING |
| 42 | feasible | **rejected** | feasible | feasible | PENDING |
| 44 | feasible | feasible | feasible | feasible | PENDING |
| 46 | feasible | feasible | feasible | feasible | PENDING |
| 50 | feasible | feasible | feasible | feasible | PENDING |

**The three cells at SLO 38/40/42 are what this recovers.** There the global
18 % rejects and the per-point margin passes — the same disagreement P3 was
selected to show, on a candidate that can actually be deployed. If the hardware
comes in at or under 42 ms, rule (b) produced a **false rejection** in three
cells and rule (c) did not; if it comes in above 43.42, rules (a), (c) and (d)
produced **false passes** and (b) was right.

Either way V3 gets §6 material — an exclusion rate for a candidate that really
satisfies, or a pass rate for one that really violates — from one deployment
rather than the six the original design needed. **As cases, not as statistics**:
one candidate across six thresholds is one measurement read six ways, and the
specification must present it as such.
