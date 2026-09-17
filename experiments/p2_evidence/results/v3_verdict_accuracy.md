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

**A parallelism / placement axis on `AccuracyDomain`**, and a `refuse` when a
candidate's configuration does not match the one the domain was fitted on, is the
second. §6.4 is the measured case for it: a domain fitted at TP=1 answered a TP=4
query with a 1.13 % margin where ~80 % was needed, and nothing in the schema
could say that was extrapolation.

**It is also out of scope here, because it is a code change**, and a consequential
one: adding a scoping field makes every committed domain under-specified until it
is filled in, and turning on `refuse` for a mismatch would change the verdict of
every candidate whose parallelism differs from its domain's — which on this
fixture is most of them. It belongs in its own PR with its own regression
evidence. Proposed, not started.

**A third, smaller one**: measure `link_bw:pcie-a40a-02`. Seven minutes of
exclusive GPU (§6.3.2), and §6.3.3 makes it the leading explanation for the whole
of V3's error. It needs no code at all — it is the measurement the plan already
wanted and could not rank. **Done 2026-09-17 — Appendix A.**

**A fourth, for Stage B+**: give each uncertainty **grade** a default sourced
range, so an input like this one can be *ranked* instead of shelved.

`link_bw:pcie-a40a-02` carried `grade: VENDOR_SPEC` and
`range: lo=None, hi=None`. `--measurement-plan` sweeps an input across its
**sourced** range to compute ΔR, so an input with no range is listed as
*undecidable* and gets **no rank at all** — correctly, because inventing a range
would be inventing a number (absolute rule 3). The consequence is that the one
input which turned out to explain a 44 % prediction error sat below every ranked
item, at a measurement cost of 0.114 h.

**Appendix A supplies the evidence such a default would need.** Measured, the
vendor figure was optimistic by **7.3×** for the quantity the simulator reads
(64.0 against a TP=4 effective 8.8) and by **2.6×** for the nearest physical
analogue (64.0 against a 25.0 GB/s device-to-device copy). A `vendor_spec` grade
whose default range were, say, `[nominal/8, nominal]` would have ranked this
input rather than shelving it — and the range would have had a source: this
measurement, on this class of path.

**One measurement is not a distribution**, so the figure above is a data point
for that work and not the default itself. Recorded here because Stage B+ owns the
grades table and because the case that motivates it is now measured. Out of scope
for this work order.

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
(d) 6 false passes each.**

**How to state this, and how not to.** The global rule came out ahead **on this
one candidate, at SLO 38–42 only, and by an accident of margin size** — 18 % is
arbitrarily larger than 1.13 %, and at SLO 44 and above it fails exactly as the
others do. It was not better informed; it was bigger. Nothing here supports
"a global margin beats a per-point one", and §6.4 shows the comparison is not
even between two margin policies but between a domain that covers the candidate
and one that does not.

**As cases, not statistics** — one candidate read at six thresholds is one
measurement, and the specification must present it that way.

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

#### 6.3.1 There is no steady state — both sides are transients, and the run is in overload

Instantaneous concurrency, counted as requests that have arrived and not yet
completed:

| | peak concurrency | at | arrival span | window | middle-half slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| simulator | 217 | 29.2 s | 29.1 s | 58.8 s | −1.34 req/s |
| **measured** | **300** | 29.3 s | 29.2 s | 101.4 s | **−3.23 req/s** |

Neither side is flat. Both are triangles: the queue accumulates through the
arrival window, peaks as the last request lands, and drains. **On the hardware
the peak is 300 — every request in the trace is resident at once**, and with
`max_num_seqs = 128` that means 172 of them are queued rather than running. The
makespan ratio is **1.724×**.

**So the conclusion is stated as overload, not as an operating point.** `L`
= 163.383 is the time-average of a triangle, not a concurrency the server
settled at; the simulator's 127.872 is the time-average of a *different* triangle
with the same base. Part of the −21.73 % gap is therefore a difference in drain
shape rather than in load, and neither figure should be read as "the operating
point this candidate runs at". A steady-state measurement would need an arrival
window long enough for the queue to stabilise, which 300 requests at ~10 rps
against a ~3 rps server cannot provide.

#### 6.3.2 The topology the simulator was given, against the one that exists

`nvidia-smi topo -m` on this node, for the four GPUs `cuda-a40-node_a40a` maps to:

```
        GPU0  GPU1  GPU2  GPU3
GPU0     X    NV4   NODE  NODE
GPU1    NV4    X    NODE  NODE
GPU2    NODE  NODE   X    NV4
GPU3    NODE  NODE  NV4    X
```

Two NVLink pairs (0–1, 2–3) bridged by PCIe. The cluster spec has the same
*shape* — `nvlink-a40a-01` and `nvlink-a40a-23` at 112.5, `pcie-a40a-02` at 64.0
— so the structure is right. **But `pcie-a40a-02` carries `source: vendor_spec`,
and TP=4 spans both pairs, so every all-reduce crosses exactly that link.**

The uncertain-input registry already says so. Of its six entries for this
fixture, the first is:

```
id      link_bw:pcie-a40a-02        grade   VENDOR_SPEC
nominal 64.0 gbps                   range   lo=None, hi=None
affects ['cuda-a40-node_a40a']      cost    0.114 h, exclusive
```

**It affects exactly P1's island, it is the only non-measured input that does,
and it has no sourced range** — so `--measurement-plan` lists it as *undecidable*
rather than scoring it, and it receives **no ΔR rank at all**. Measuring it costs
**seven minutes**.

#### 6.3.3 Is that link the cause? One parameter, three metrics

Re-simulated with only `pcie-a40a-02` changed (`v4_pcie_perturbation.json`):

| PCIe bw | `L_pred` | p99 TPOT | p99 TTFT |
| ---: | ---: | ---: | ---: |
| 64.0 (vendor spec) | 127.872 | 36.547 | 16 716 |
| 32.0 | 134.625 | 40.301 | 20 936 |
| 16.0 | 146.187 | 47.525 | 29 822 |
| 6.4 | 170.584 | 70.909 | 57 868 |
| **measured** | **163.383** | **66.012** | **53 176** |

Interpolating each metric separately for the bandwidth that reproduces it:

| metric | bandwidth that matches | vs vendor spec |
| --- | ---: | ---: |
| served concurrency | 8.39 gbps | 7.6× lower |
| p99 TPOT | 7.75 gbps | 8.3× lower |
| p99 TTFT | 7.46 gbps | 8.6× lower |

**Three metrics, one parameter, agreement within ±6 %.** A single unmeasured
input, set to about an eighth of its vendor figure, reproduces the whole of a
44.6 % TPOT error and a 21.7 % concurrency error at the same time.

**This is evidence, not proof.** A TP compute model that is wrong in a way that
mimics a slow link would fit too; what makes the link the better hypothesis is
that three metrics of different kinds move together under one knob, and that the
knob is the one input the registry independently flagged for this island.
**Settling it costs seven minutes of GPU** — the measurement the plan could not
rank because the input has no sourced range.

**Launch accuracy**: worst 76.08 ms against the driver's 20 ms contract, as in
V2's 10 rps stages and for the same reason — 300 streamed responses in flight at
~10 rps saturate the event loop. At a p99 TTFT of 53 s that is 0.14 % of a
request's residency.

#### 6.3.4 Sanity: patent 1's lower bound is not violated

Patent 1's pruning stages S1–S5′ are the candidate generator's; the numeric one
is **S5, the memory-roofline TPOT floor**. For P1 at tp=4 with 128 active
sequences:

```
weights 4 015 529 984 B + 128 x 32 768 B  =  4 019 724 288 B
/ 696 GB/s                                =  5.7755 ms
```

| | ms | ratio to floor |
| --- | ---: | ---: |
| S5 roofline floor | **5.7755** | 1.00× |
| simulator p99 TPOT | 36.547 | 6.33× |
| **measured p99 TPOT** | **66.012** | **11.43×** |

**The bound holds** — a lower bound that exceeded a measurement would be wrong by
construction, and this one does not. It is also very loose, which is expected: it
charges one pass of weights and live KV through HBM and nothing for compute,
collectives or queueing. The 11.4× gap is where those live, and §6.3.3 argues a
large part of it is the unmeasured link.

#### 6.3.5 The lookup coordinate does not matter here

Both coordinates are **inside** the A40 domain's measured range [4.043, 170.56]:
`L_pred` = 127.872 and `L_meas` = 163.383. The margins they select are 1.1347 %
and 1.3896 %, giving robust values of 37.21 ms and 37.31 ms against a measured
66.01 ms. **Neither covers, and the verdict is identical either way.**

So this case adds nothing for or against claim 2's coordinate choice — consistent
with V2, which measured the two coordinates differing by under 3 % and selecting
margins within 0.008 pp. What decides this case is not *where* the domain is
consulted but that **the domain does not cover the candidate's parallelism at
all** (§6.4).

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

---

# Appendix A — the link, measured

*Added 2026-09-17. §6.3.3 named an unmeasured link as the leading explanation for
V3's 44.6 % error and said seven minutes of GPU would settle it. This is that
measurement, plus the discriminator that rules out the competing hypothesis, plus
one thing nobody was looking for. All eight GPUs were confirmed idle before each
run and released after.*

> **Three independent routes reach the same number.** V4's perturbation had to
> assume **7.46–8.39** to reproduce the measurement; the link measures
> **8.78–8.81 GB/s** effective for a TP=4 all-reduce; and re-simulating with 8.8
> collapses the TP=4 error from **−43.4 % to −7.0 %** on TPOT and from **−21.2 %
> to −0.7 %** on served concurrency.
>
> The same substitution makes the **TP=2** candidate *worse* — −5.2 % to
> **+18.2 %** — and that is the appendix's second finding, not its failure. A
> single static link value cannot be right for both, because the quantity the
> simulator reads there is not a property of the path but of **which devices the
> candidate occupies**.

## A.1 What was measured

`nccl-tests` could not be used: the only build on this host needs GLIBC 2.34 and
Ubuntu 20.04 has 2.31. The probe is `experiments/p2_evidence/link_probe.py`,
torch 2.10 + NCCL 2.27.5 under `torchrun`, sweeping 8 KiB to 64 MiB — the range a
TP group actually sends, since Llama-3.1-8B at hidden 4096 in bf16 carries 8 KiB
per token per all-reduce, so a 128-way decode step is 1 MiB and a 2048-token
prefill chunk is 16 MiB.

Run on **both** GPU groups, which are topologically symmetric (0↔1 and 2↔3 are
`NV4`, the pairs bridged by `NODE`, 0–3 to 4–7 is `SYS`) but sit on different
NUMA nodes:

| case | 1 MiB busbw | **64 MiB busbw** | GPUs 0–3 | GPUs 4–7 |
| --- | ---: | ---: | ---: | ---: |
| `tp4` — all four | 7.07 / 7.60 | **8.81 / 8.78** | 8.81 | 8.78 |
| `tp4` with `NCCL_P2P_DISABLE=1` | 5.70 / 5.76 | 6.49 / 6.47 | 6.49 | 6.47 |
| `pair` inside an NVLink pair | 13.11 / 13.45 | **39.11 / 39.10** | 39.11 | 39.10 |
| `pair` across the bridge | 9.51 / 9.73 | **19.29 / 19.33** | 19.29 | 19.33 |
| the same, `NCCL_P2P_DISABLE=1` | 4.68 / 4.70 | 6.81 / 6.78 | 6.81 | 6.78 |
| direct p2p copy, NVLink pair | — | — | 52.21 | 52.20 |
| direct p2p copy, across the bridge | — | — | 25.01 | 25.01 |

**Every case agrees across the two groups to within 0.5 %.** That is a
reproduction on independent hardware, not a repeat.

**Against what the cluster spec says:**

| link | spec | measured p2p | measured pair all-reduce | TP=4 effective |
| --- | ---: | ---: | ---: | ---: |
| `nvlink-a40a-01` | 112.5 | 52.2 | 39.1 | — |
| `pcie-a40a-02` | **64.0 `vendor_spec`** | 25.0 | 19.3 | **8.8** |

**Peer-to-peer is enabled on both paths** (`can_device_access_peer` is true for
0↔2), and disabling it costs the bridge pair 2.85× (19.33 → 6.78), so it is not
merely available but load-bearing. **ACS could not be read** — `lspci`'s ACS
control register needs root here — so the functional check stands in for it: a
host where ACS blocked peer-to-peer would not show those numbers.

## A.2 The discriminator: is it the link, or the TP compute model?

§6.3.3 was explicit that a TP compute model wrong in a way that mimics a slow
link would fit the same data. The test is to run the same candidate at **TP=2
inside one NVLink pair**, which never touches the bridge, and compare the
simulator's error.

Both bound to NUMA 1 (see A.3), simulator on the same sharegpt trace, cache
disabled, three repeats each:

| | sim p99 TPOT | measured p99 | error | sim `L` | measured `L` | error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TP=4**, crosses the bridge | 36.547 | 64.616 | **−43.44 %** | 127.87 | 162.26 | −21.19 % |
| **TP=2**, inside the NV4 pair | 63.831 | 67.337 | **−5.21 %** | 153.77 | 156.06 | −1.47 % |

**An 8.3× difference in error, from removing the bridge from the path.** The TP
compute model is exonerated: shrink the group so it never crosses the slow hop
and the simulator is accurate to ~5 %.

## A.3 What nobody was looking for: NUMA placement is worth 1.93×

The first attempt to reproduce the P1 measurement on GPUs 4–7 came back at half
the throughput. Engine initialisation was **identical** — same 35.79 GiB KV
cache, same 1 172 624 tokens, same 8.95× maximum concurrency, same model load —
and so were clocks, power limits and temperatures across all eight GPUs.

GPUs 4–7 sit on NUMA node 1 and nothing binds the server to it:

| | p99 TPOT | p99 TTFT | output tok/s |
| --- | ---: | ---: | ---: |
| 0–3, TP=4, unbound | 66.012 | 53 176 | 1 930.6 |
| 4–7, TP=4, unbound | 66.520 | **139 180** | **1 018.5** |
| 4–7, TP=2, unbound | 68.494 | 142 391 | 990.2 |
| **4–7, TP=4, `numactl --cpunodebind=1 --membind=1`** | **64.616** | **51 357** | **1 970.7** |

**Binding recovers 1.93× of throughput and 0.37× of TTFT**, and lands within
2–3 % of the 0–3 baseline. TPOT barely moves in any of them: what NUMA costs is
prefill and queueing, not decode.

Three things follow.

1. **The TP hypothesis was already dead here.** Unbound, TP=4 and TP=2 on 4–7 are
   indistinguishable (1 018 against 990); bound, TP=4 jumps to 1 971. The
   variable was never TP.
2. **V3's headline survives.** Recomputed against the NUMA-bound measurement the
   simulator's TPOT error is **−43.44 %** where §6.3 reported −44.63 %. It was
   not a NUMA artefact.
3. **There is a methodology debt.** Every A40 measurement before this one — V2's
   eight ladder stages, V3's P1 — ran **unbound**, and GPUs 0–3 happened to land
   well. The deploy backend has no affinity control at all. V2's stages were TP=1
   on a single card, where the exposure is smallest, but **that is an argument,
   not a check.** Future measurements bind explicitly and say so.

## A.4 Re-simulating with the measured value — and why one value cannot serve

The cross-pair links are set to **8.8, `source: measured`**, in a **copy** of the
fixture (`experiments/configs/clusters/pd-rngd-gpu-card-measured-pcie.yaml`,
rule A3). The committed cluster is untouched.

| candidate | metric | measured | sim @ 64.0 | error | **sim @ 8.8** | **error** |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **TP=4** | p99 TPOT | 64.62 | 36.55 | −43.44 % | **60.09** | **−7.01 %** |
| | served `L` | 162.26 | 127.87 | −21.19 % | **161.07** | **−0.73 %** |
| | p99 TTFT | 51 357 | 16 716 | −67.45 % | 44 866 | −12.64 % |
| **TP=2** | p99 TPOT | 67.34 | 63.83 | −5.21 % | 79.60 | **+18.21 %** |
| | served `L` | 156.06 | 153.77 | −1.47 % | 166.49 | +6.69 % |
| | p99 TTFT | 52 613 | 48 447 | −7.92 % | 67 342 | +28.00 % |

**For TP=4 the substitution all but removes the error** — served concurrency to
within 0.7 %, which is the metric claim 2 is about.

**For TP=2 it introduces one, and changes its sign.** The reason is visible in
the fact that it changed at all: the hardware's TP=2 runs inside the NVLink pair
and **never crosses the bridge**, so a link it does not use could not have moved
its prediction by 25 %. **The simulator routes TP=2 over the cross-pair link
anyway** — it has no representation of which two of an island's four devices a
TP=2 group occupies.

**So 8.8 is the effective bandwidth of a TP=4 candidate, not a property of the
path.** Three values describe that path and they bracket rather than compete:

| value | what it is | where it applies |
| ---: | --- | --- |
| **25.0 GB/s** | a direct device-to-device copy | the link with no collective on it |
| **8.8 GB/s** | TP=4 all-reduce busbw | **the value this re-simulation uses** |
| 7.5–8.4 | what V4's perturbation had to assume | inferred, not measured |

A static per-link field cannot hold a quantity that depends on how many devices a
candidate spans and which ones. **That is the same axis as A.3's NUMA finding and
as §6.4's missing parallelism scope**, seen a third time:

| # | what the model cannot express | measured cost |
| --- | --- | ---: |
| 1 | which NUMA node hosts the island's server | **1.93×** throughput |
| 2 | which devices inside an island a TP group occupies | 5.2 % → 18.2 % error |
| 3 | the parallelism degree a domain was fitted at | 1.13 % margin where ~80 % was needed |

## A.5 What this changes, and what it does not

**Unchanged.** V3's finding stands in full: the per-point margin under-corrected
by ~70× on P1, and §6.4's reading — that this is a missing scoping axis rather
than a failure of margining — is now supported by measurement rather than by
inference.

**Strengthened.** §6.3.3 offered the link as a hypothesis with three metrics
fitting one parameter. It is now measured directly, reproduced on two GPU groups,
and confirmed by a discriminator that removes the competing explanation.

**New.** Device placement is not one condition but at least three, and the
largest of them — NUMA — was invisible to every measurement this work order has
taken so far.

**Still not established.** Whether the same holds on RNGD, where the margin is
~21 % rather than ~1 %, is untested and unreachable from this node.
