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
| §6 P1 deployment + SLO sweep | **complete** — measured 2026-09-17, 3 repeats |
| §6.4 the TP=1 / TP=4 contrast | **complete** — the finding V3 actually produced |

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

**Measured 2026-09-17**, reduced in scope to **P1 only** (§4①). The table is
§6.2's; §6.3 has the measurement and §6.4 what it means.

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
one-sidedness exists to prevent exactly that. **Where the domain covers the
candidate, the A40 is the case where the invention should do almost nothing, and
it does almost nothing.** What that case cannot do is demonstrate the invention's
value, which is why ③ matters.

**AMENDED 2026-09-17 by the measurement.** The "1–3 %" above is V2's figure and
it holds **at TP=1, where every point in the domain was measured**. V3 deployed a
TP=4 candidate and found the error is **−44.6 %** at nearly the same served
concurrency. So this limit is narrower than it was written: the margin is smaller
than the simulator's error *within the domain's coverage*, and outside it the
margin is not small but simply **uninformed**. §6.4 has the evidence and what it
implies for the schema.

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

**MEASURED 2026-09-17** on the A40 node, all eight GPUs idle and confirmed so
before starting. Three repeats, 300/300 requests each, server stopped and GPUs
released afterwards. Raw `outputs/p2_evidence/v3/p2ev-v3-p1_r{0,1,2}.json`.

> **The result runs against the invention on this candidate, and §6.4 says why
> that is a finding rather than a refutation.** The simulator predicted a p99
> TPOT of 36.5 ms and the hardware delivered **66.0 ms** — a −44.6 % error,
> against a per-point margin of 1.13 %. Across the SLO sweep the per-point rule
> produced **six false passes and zero correct rejections**, while the global
> 18 % produced **three correct rejections**. The margin that would have covered
> the measurement is **~80 %**.
>
> The mechanism is not a failure of per-point margining. At almost the same
> served concurrency the same simulator on the same hardware is accurate to
> **−1.05 %** at TP=1 (V2) and wrong by **−44.6 %** at TP=4 (here). **The A40
> accuracy domain is fitted entirely on TP=1 and `AccuracyDomain` has no
> parallelism axis at all.**

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
Measured p99 TPOT is **66.012 ms**, so the candidate really violates at every
threshold in the grid:

| TPOT SLO | (a) no margin | (b) global 18 % | (c) per-point | (d) per-point + refuse | measured ≤ SLO? |
| ---: | --- | --- | --- | --- | --- |
| 38 | FALSE PASS | **correct rejection** | FALSE PASS | FALSE PASS | no |
| 40 | FALSE PASS | **correct rejection** | FALSE PASS | FALSE PASS | no |
| 42 | FALSE PASS | **correct rejection** | FALSE PASS | FALSE PASS | no |
| 44 | FALSE PASS | FALSE PASS | FALSE PASS | FALSE PASS | no |
| 46 | FALSE PASS | FALSE PASS | FALSE PASS | FALSE PASS | no |
| 50 | FALSE PASS | FALSE PASS | FALSE PASS | FALSE PASS | no |

**Tally: (a) 6 false passes, (b) 3 correct rejections and 3 false passes, (c) and
(d) 6 false passes each.** On this candidate the global rule strictly dominated
the per-point one. **As cases, not statistics** — one candidate read at six
thresholds is one measurement, and the specification must present it that way.

### 6.3 The measurement

| | predicted (matched trace) | measured (3 repeats) | error `e` |
| --- | ---: | ---: | ---: |
| served concurrency | 127.872 | **163.383** | **−21.73 %** |
| p99 TPOT | 36.547 ms | **66.012 ms** | **−44.63 %** |
| p99 TTFT | 16 716.4 ms | **53 176.1 ms** | −68.56 % |
| p50 TPOT | 30.483 ms | 56.552 ms | −46.10 % |

Repeatability is excellent and is not the issue: p99 TPOT spread **0.047 ms
(0.072 %)** over three runs, `L_meas` spread 0.067 (0.041 %). Every run delivered
195 753 output tokens, the trace's exact total.

**The workload confound was removed, not argued away.** P1's committed prediction
comes from E-A1, which ran a trace synthesised from the spec's length
distributions; the hardware ran the sharegpt file. The two differ by 2–4 % on
mean lengths and offered rate, so the simulator was re-run on **the exact file
the server was driven with** (`v3_sim_matched.py`, cache disabled). It returned
L 127.872 / TPOT 36.547 / TTFT 16 716.4 against E-A1's 127.280 / 36.795 /
16 159.5 — under 1 % apart. The table above uses the matched values.

**D32's ±20 % operating-point guard fails here** at −21.73 %. The prediction and
the measurement are not at the same load, and that is itself the finding: the
simulator did not know how much work this configuration would be carrying.

**Launch accuracy**: worst 76.08 ms against the driver's 20 ms contract, as in
V2's 10 rps stages and for the same reason — 300 streamed responses in flight at
~10 rps saturate the event loop. At a p99 TTFT of 53 s that is 0.14 % of a
request's residency.

### 6.4 Why this is a finding about the domain's schema, not about margining

The natural reading — "the per-point margin failed and the cruder global rule did
better" — is wrong, and the evidence that it is wrong is V2's.

| | TP | served concurrency | p99 TPOT error |
| --- | ---: | ---: | ---: |
| V2, 10 rps stage | **1** | 170.46 | **−1.05 %** |
| V3, P1 | **4** | 163.38 | **−44.63 %** |

Same hardware, same model, same workload, same arrival process, nearly the same
served concurrency. The simulator is accurate to 1 % at TP=1 and wrong by 45 % at
TP=4. Measured tensor-parallel scaling is **1.78× of TP=1 throughput for four
times the GPUs**; the simulator credits far more.

**Every point in `a40.accuracy.yaml` and in V2's open-loop file was measured at
TP=1**, and `AccuracyDomain`'s scoping fields are

```
fitted_at_concurrency, points, outside_domain, source, note,
workload_shape, model, variant, arrival_process
```

— **there is no parallelism axis.** So a domain fitted on single-card runs is
consulted for a four-way tensor-parallel candidate with nothing in the schema to
say that is extrapolation. The margin it returns (1.13 %) is a correct answer to
a question about TP=1, applied to a configuration it never saw.

**This is constructive for the disclosure rather than damaging.** Its own
`refuse` contract — decline to judge an operating point no measurement covers —
is exactly the right response here, and it did not fire because the domain has no
way to represent the axis along which this candidate is out of range. **The
scoping conditions in §2.4.1 (model, precision, token mix, arrival process) are
incomplete: parallelism belongs among them.** The specification can state that as
a condition on the verification information rather than discovering it later.

What V3 therefore establishes, on A40:

1. With a domain that does not cover the candidate's parallelism, **the per-point
   margin under-corrects by a factor of ~70** (1.13 % charged, ~80 % needed) and
   produces false passes at every threshold tested.
2. A **fixed global margin is not a fix** — 18 % also under-corrects, and its
   three correct rejections come from being arbitrarily larger, not from being
   better informed. At SLO 44 and above it fails too.
3. **Neither result transfers to a domain that does cover the candidate.** V2's
   TP=1 points, where the domain does cover it, agree with the hardware to
   ~1 %.
