# Spike — how much of the simulator's RNGD error is the execution model?

**Work order:** `WORK_ORDER_npu_exec_model_spike.md` (2026-09-15). **D-block:** D90–D99.
**Experiment tag:** `E-N*`. This document is the spike's running record; STEP D turns it into the
decomposition table and the conclusion.

**The question.** The RNGD card's accuracy domain shows a TPOT error that changes sign with load —
+11 % at served concurrency 2, ~0 at 16, −18 % at 76 (`profiles/calibration/rngd_card_edf.yaml`),
with TTFT at −32.6 % (D17, `rngd_sim_vs_real_summary.md`). How much of that is the structural
difference between what LLMServingSim simulates (a vLLM-style continuous-batching scheduler) and
what Furiosa-LLM is (a static, ahead-of-time-compiled bucket executor)? "Most" means a follow-up
work order for an opt-in execution model; "some" means formalising only the mechanisms that carry
weight and leaving the rest in the accuracy domain.

This spike **decomposes**. Finishing an execution model is a later work order's job.

---

## §0 — Setup (STEP 0)

### Baseline

| item | value |
| --- | --- |
| branch point | `main` at `f6ee919` ("update WORK_ORDER_p2_regular_spec_evidence.md: V0 rewritten on D70") |
| work branch | `spike/npu-exec-0-setup` |
| date | 2026-09-16 |
| node | **NPU** (`scripts/whichnode.sh`: 3 RNGD cards, 4 ATOM devices, no NVIDIA reachable, 96 cores / 1511 GiB) |

`whichnode.sh` on this node reports: planner + analytical simulation YES; real vLLM bench NO (no
NVIDIA driver); CUDA layerwise profiler NO; RNGD profiling YES via `furiosa.torch` under the system
python3, not `.venv`; ATOM present but the vendor install is broken (`docs/nodes/npu.md`).

STEP C of this work order is therefore runnable on this node. STEP 0 and STEP A need no device.

### Gates

At the branch point (`f6ee919`, before anything in this table existed):

```
pytest   904 passed, 1 skipped in 185.00s
ruff     All checks passed!
mypy     Success: no issues found in 47 source files
```

With STEP 0 applied:

```
pytest   912 passed, 1 skipped in 184.51s
ruff     All checks passed!
mypy     Success: no issues found in 47 source files
```

The eight new tests are the seven in `tests/test_artifact_buckets.py` and the `D90` case added to
`test_claude_md_still_documents_the_blocks`. Nothing in `serving/` or `planner/` changed, so no
golden output moved.

### What STEP 0 committed

| path | what |
| --- | --- |
| `experiments/scripts/furiosa_artifact_buckets.py` | reads `artifact.json`, classifies buckets, `--json` / `--yaml` |
| `experiments/scripts/serve_log_hit_rates.py` | tabulates `Wire pipeline hit rate` from committed serve logs |
| `experiments/results/rngd_artifact_buckets.md` | the grid of all three artifacts, and what follows from it |
| `experiments/results/rngd_pipeline_hit_rate.md` | the composed-vs-kernelwise measurement, H-a/H-b left open |
| `profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml` | the machine-readable grid for `d6ae6a43` |
| `docs/deviations.md` D90 | the kernelwise-path finding, and the D17 annotation it resolves |
| `tests/test_artifact_buckets.py` | the grid's invariants, including D90's mixed-step claim |
| `tests/test_deviations_numbering.py` | block range now read from CLAUDE.md instead of hardcoded |

Both scripts were written during the 2026-09-15 investigation and were living in a session
scratchpad, not in the repo; STEP 0 committed them and re-ran both from scratch on this checkout.
Every number in §1.1 and §1.2 of the work order reproduced exactly (see the two results files).

`CLAUDE.md` already carried the `D90–D99` block row and the `E-N*` tag row from 2026-09-15; both
were confirmed present and **not** re-added.

**A note on absolute rule 1.** `artifact_buckets.yaml` lands under `profiler/perf/`, which is an
upstream *directory* (the pinned upstream ships an `RTXPRO6000` bundle there). The `RNGD-CARD`
subtree under it is entirely fork-added (`4aac7a5`), and the new file is data beside the fork's own
bundle, not a change to upstream code. Work order A7 requires this location: the grid is a property
of the served artifact, not of the card, so it cannot live in `profiles/accelerators/*.yaml`.

### One stale guard, and two sessions fixing it at once

`tests/test_deviations_numbering.py` rejected D90 with *"claim a block in CLAUDE.md
in the same commit"* — the one instruction that had already been followed on
2026-09-15. Its allowed range was the literal `BLOCK_RANGE = (37, 89)`, written
when D80–D89 was the last row, so claiming a block could never be enough on its
own.

**`WORK_ORDER_p2_regular_spec_evidence.md` hit the same wall and fixed it first**,
in the work that became `main`'s `f0a3a84`, and its fix is the better one: it also
widens `_REFERENCE` from `D\d{1,2}` to `D\d{1,3}`, without which no reference to
D90–D99 was being checked for existence at all. This spike's own version of the
fix was dropped in favour of theirs on rebase; all that is left here is the `D90`
and `D100` cases in the parametrised presence check.

That is the D34 collision the numbering rule exists to prevent, arriving through a
*test* rather than through `deviations.md` — two streams, the same stale constant,
no textual overlap in the entries themselves. Worth noting that the block table
did its job: D90–D99 and D100–D109 never collided, only the guard did.

### Two corrections to the work order's §1.1

Both were found by re-running the extraction, and both change rules that STEP A and STEP B are
written in terms of. Neither is settled by measurement yet, so both are carried forward as open
items rather than adopted.

**1. The kernelwise attention grid is not a uniform 1024 tokens.** §1.1 derives a 1024-token bucket
width from `131072 / 128`. The 128 is not a division result: pipeline 0's menu holds exactly
`8 prefill + 74 extend + 46 decode = 128` buckets and is the **union** of the compiled buckets
(verified as a set; the 46 composed pipelines are a strict subset). Its spacing is 128 tokens up to
1024 and then powers of two, and the decode edges are `{1024, 2048, …, 131072}`.

*Consequence.* The "1024-unit KV bucket" of R-attn (STEP A.3) and P3 (STEP B.1) is not the grid the
artifact actually has. Grouping on the real edges puts a ~29-sequence sharegpt batch into about 2–4
groups, which is what D17 measured (1.95 → 3.08 attention executions per layer); a uniform 1024-token
grid would predict many more. **STEP A.1 counts distinct groups both ways** and compares each with
D17's 1.95 / 2.40 / 2.87 / 3.03 / 3.08. Nothing here measures the runtime's grouping — only what it
was compiled for.

**2. The batch-padding ladder is not powers of two throughout.** The kernelwise `tokenwise_buckets`
ladder is 1, 2, 4, 8, 16, 32, 64, 128, 256, **384**, 512, 1024. "Round the batch up to the next
power of two" (R-pad in A.3, P2 in B.1) is correct at or below 256 only; above it the target is the
next entry of this ladder.

### Open items carried out of STEP 0

| item | where it is decided |
| --- | --- |
| H-a vs H-b — why the composed hit rate collapses | STEP C.1 |
| KV-group unit: real decode edges vs uniform 1024 | STEP A.1 census, both ways, against D17 |
| batch padding above 256 (the 384 rung) | STEP A.1 census / A.3 R-pad |
| KV usage at 100 % even at c1 — bucket-granular reservation or gauge definition | STEP B.1 P5 (an open investigation item there) |
| does the `INFO` batch log carry per-request ids and token counts | STEP A.1 (fallback: `--no-cleanup-inputs` trace `input_size`) |
| does `lowload_sim_error.py` accept scheduler knobs | STEP A.2 (fallback: a cluster-json copy) |
| is there attention-stage shape in the raw EDF CSV | STEP A.3 R-attn (fallback: D17 median, marked provisional, replaced by C.3) |

---

## §A — decomposition without code changes (STEP A)

Nothing in `serving/` or `planner/` changes in this step. Two things the work order
left as "needs investigation" were settled first, and one of them changes what A.2
can be.

### The step structure is already in the log — no debug print needed

The work order expected to have to add a read-only debug print and record it as an
A3 exception, because `scheduler.py:296-335` logs a batch id and nothing else. It
does not need one. `trace_generator.py:1308` already logs, at INFO, one line per
scheduled step carrying the requests in it:

```
Batch #7: model=meta-llama/Llama-3.1-8B num_reqs=4 total_len=4 req_ids=[0, 1, 2, 3]
```

and `Controller` logs a **cumulative** cycle count per iteration, whose first
difference is that step's duration. (Reading the count itself as a duration makes
every step look monotonically longer — a mistake that still produces a plausible
table.) Together with the trace the run was given, which holds each request's input
length, that is enough to replay every step: a request with fewer computed tokens
than its input is prefilling and consumes some of the step's token budget, any
other is decoding and consumes exactly one token over `computed` tokens of KV.

The replay is self-checking, because the prefill chunks must sum to
`total_len - n_decode` exactly. `experiments/scripts/npu_exec_step_census.py`
counts a step that does not balance and `--strict` makes it an error; on the
pilot run **3610 of 3610 steps balanced**, so the reconstruction is the
scheduler's own allocation and not an approximation of it.

### The work order's A.2 knob set cannot run this workload

A.2 asks for `--no-enable-chunked-prefill --prioritize-prefill
--max-num-batched-tokens 1024 --max-num-seqs 128`. Those flags do not compose on
sharegpt. With chunked prefill off, `max_num_batched_tokens` stops being a step
budget and becomes a cap on input length: `scheduler.py:164-180` drops requests
from the batch until the total fits, and when the batch empties it prints
`[WARNNING] Cannot load the request to batch due to max_num_batched_tokens
limitation` and returns `None`. A request longer than 1024 tokens can then never
be scheduled. Half the workload is longer than that — 10 of the first 20 sharegpt
requests, up to 3452 tokens — so the run livelocks. Measured, not inferred: the
probe emitted **80,821** of those warnings and was killed at the timeout
(`outputs/npu_spike/a2_asspec_probe/`, exit 124).

This is not a detail of the flags. The runtime chunks too — its 74 `extend`
buckets are exactly `chunk <= 1024` against non-zero KV — so "no chunked prefill"
was never the right description of it. What the runtime does is chunk at
`bs = 1` and keep prefill steps exclusive of decode, and §1.4 of the work order
already says no knob combination reproduces that. So A.2 runs **two** one-sided
approximations instead of one, and each is labelled with what it does not capture:

| variant | knobs | captures | misses |
| --- | --- | --- | --- |
| `a2_exclusive` | `--no-enable-chunked-prefill --prioritize-prefill --max-num-seqs 128` | prefill steps exclusive of decode | the 1024-token chunk ceiling; prefill batches are not forced to `bs=1` |
| `a2_chunk1024` | `--max-num-batched-tokens 1024 --long-prefill-token-threshold 1024 --max-num-seqs 128` | the 1024-token chunk ceiling, one prefill chunk per step | exclusivity — decode is still scheduled first and prefill fills the remainder |

`--max-num-batched-tokens 8192` is kept in the first so that a 3452-token prompt
still fits in one step; the second keeps chunked prefill on, which is what makes
1024 a budget again rather than a ceiling on input length.

`lowload_sim_error.py` hardcoded the scheduler knobs and the log level, so it grew
a `--sim-arg` passthrough (appended to the simulator's command line, last
occurrence wins). Omitting it leaves every committed invocation byte-identical.

### A.1 — what the simulator's steps look like (E-N1)

Six envelope points, 300 requests each, current knobs, `--log-level INFO`. The runs
reproduce the committed accuracy domain to the second decimal (+2.25 / +11.00 /
+10.79 / +7.33 / +3.26 / +2.47 against `outputs/card_lowload_300/`), so INFO logging
changed nothing. **369,270 steps replayed, 0 unbalanced.** Full table:
`experiments/results/npu_exec_step_census.md`.

| point | steps | mean decode bs | mixed steps | mixed cycle share | prefill bs=1 | decode pad | prefill 128-pad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1.0 | 182269 | 1.07 | 0.08 % | 0.4 % | 100 % | 0.0 % | 7.2 % |
| c1.99 | 90705 | 2.16 | 0.33 % | 1.4 % | 100 % | 11.3 % | 7.2 % |
| c3.98 | 45796 | 4.27 | 0.65 % | 2.6 % | 100 % | 29.9 % | 7.2 % |
| c7.88 | 24063 | 8.13 | 1.24 % | 4.6 % | 100 % | 38.3 % | 7.2 % |
| c15.3 | 13383 | 14.61 | 2.23 % | 7.4 % | 100 % | 25.7 % | 7.2 % |
| c15.59 | 13054 | 14.97 | 2.29 % | 7.6 % | 100 % | 30.3 % | 7.2 % |

Three readings, and the first is not the one the step-share column suggests.

**Mixed steps are a small share of steps and almost all of the prefills.** The raw
share looks negligible — 0.08 % to 2.3 % of steps, 0.4 % to 7.6 % of time — but that
is because decode steps outnumber prefill steps by three to four orders of magnitude.
Counted the other way: there are exactly 300 prefill steps at every point, and above
c1 **299 of them are mixed**. The simulator performs a pure prefill step exactly once
per run, for the first request, and shares every subsequent prefill with running
decodes. (At c1 it manages 163 pure ones, because the previous request has usually
finished.) So on the TTFT path the thing the compiled grid has no bucket for is not
rare at all: it happens to all but one request. Its small share of *time* is why it is
nearly invisible in TPOT, and its universality on prefill is a candidate for the
TTFT gap D17 measured at −32.6 %, which this spike has not otherwise touched.

**Prefill steps are already `bs = 1`, without being asked.** Every one of the 300 has
a single prefilling request, so the grid's `bs = 1` prefill constraint is never
violated at these loads — and none of them is chunked either, since 300 prefill steps
for 300 requests means each prompt fit in one step under the 8192-token budget. The
work order expected `bs = 1` to be a mechanism; at c1–c16 on sharegpt it has zero
magnitude. It would bind only where two prefills arrive close enough to coincide.

**The batch-padding ladder never leaves its lower rungs.** Mean decode batch tops out
at 15, so the artifact ladder and the powers of two are the same ladder here and the
384 rung STEP 0 found never fires. The padding *amount* is large — up to 38 % of
decode lanes are padding — which makes its cost (below) the surprise.

### The KV-group grid: none of the three candidates is the rule (E-N1, open)

The work order asked for the group count "both ways", the artifact's real decode edges
against a uniform 1024-token grid. Building it turned up a third candidate that has a
better claim than either: above c1 the runtime is on the **kernelwise** pipeline (D90),
whose attention menu is the union of all 128 compiled buckets — 15 distinct sizes,
128-spaced to 1024 and powers of two above. That is what a kernelwise step can
actually address.

| point | mean decode bs | kernelwise menu | decode edges | uniform 1024 | **D17 measured** |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1.0 | 1.07 | 1.05 | 1.05 | 1.05 | 1.95 |
| c1.99 | 2.16 | 1.82 | 1.63 | 1.63 | 2.00 |
| c3.98 | 4.27 | 2.87 | 2.12 | 2.14 | 2.43 |
| c7.88 | 8.13 | 4.15 | 2.40 | 2.45 | 2.80 |
| c15.3 | 14.61 | 5.32 | 2.63 | 2.72 | 3.02 |
| c15.59 | 14.97 | 5.37 | 2.64 | 2.73 | 3.03 |

**The answer is that none of them is the rule.** D17's measured executions per layer
*saturate*: 1.95 at batch 2, 3.08 at batch 29, i.e. it stops splitting at about three
however ragged the batch gets. The kernelwise menu does the opposite — it keeps
splitting, reaching 5.37 at batch 15 and diverging further above. The two coarse grids
do track the saturation, and they are indistinguishable from each other on sharegpt
(2.64 vs 2.73 at c15), but both sit about 0.4 below the measurement.

So the runtime is not issuing one attention execution per distinct compiled bucket in
the batch. Something caps it near three. **This is STEP C.3's question and it is now a
sharper one than the work order posed**: not "which of these two grids", but "what caps
the execution count at three". The prediction in STEP 0 — that the uniform grid would
"predict many more" groups than the real edges — is **wrong**: on sharegpt the two are
within 0.1 of each other. It is the menu, which STEP 0 did not consider, that explodes.

### A.2 — the step policy's contribution (E-N2)

Both variants, nine envelope points, 300 requests, `--match offered`, against the
committed 300-request baseline (`outputs/card_lowload_300/`, `card_highload_300/`).

| env conc | current err % | exclusive | chunk 1024 | Δ exclusive (pp) | Δ chunk (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | +2.25 | +2.41 | +2.25 | +0.16 | +0.00 |
| 1.99 | +11.00 | +11.28 | +10.99 | +0.28 | −0.01 |
| 3.98 | +10.79 | +11.50 | +10.71 | +0.71 | −0.08 |
| 7.88 | +7.33 | +8.81 | +7.08 | +1.48 | −0.25 |
| 15.3 | +3.26 | +6.00 | +2.79 | +2.74 | −0.47 |
| 15.59 | +2.47 | +5.40 | +1.97 | +2.93 | −0.50 |
| 29.3 | −3.28 | **+1.28** | −4.01 | +4.56 | −0.73 |
| 59.2 | −25.17 | −19.81 | −25.94 | +5.37 | −0.77 |
| 107.2 | −48.59 | −44.23 | −49.21 | +4.36 | −0.62 |

**The 1024-token chunk ceiling contributes nothing**: |Δ| ≤ 0.8 pp at every point,
and the sign is consistently negative, so it slightly deepens the high-load optimism
rather than relieving it.

**Exclusivity contributes a real but one-sided amount.** It grows monotonically with
load, +0.16 pp at c1 to +5.37 pp at c59, which is the right *direction* for the
high-load optimism — it takes c29.3 from −3.28 % to +1.28 %, across zero. But it is
the wrong direction everywhere the simulator is already pessimistic: at c3.98 the
error goes from +10.79 % to +11.50 %. So the step policy cannot be the explanation for
the low-load half of the curve, and at best it is a third of the high-load half.

**The simulator's throughput ceiling is not a step-policy artifact.** At the c107.2
arrival rate the sim settles at served concurrency 44.47 under current knobs and 47.77
under exclusivity — still less than half the measured 107.2. Whatever the sim is
missing at high load survives both step policies.

### A.3 — what each rule costs (E-N3)

The rebuilt step-cost model reproduces the simulator's own per-step cycles to a median
ratio of **0.999–1.000** (p05–p95 within 0.0003 at every point), so the re-pricing is
being done with the simulator's own arithmetic. Percentage points on the sum of decode
step cost, which is the TPOT proxy; full table in
`experiments/results/npu_exec_recharge.md`.

| point | current err % | R-pad | R-attn (menu) | R-attn (decode edges) | R-128 (prefill) | R-c1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c1.0 | +2.25 | +0.00 | −4.46 | −4.47 | +6.71 | +0.00 |
| c1.99 | +11.00 | −0.01 | −1.18 | −2.65 | +6.70 | +0.00 |
| c3.98 | +10.79 | −0.36 | +3.91 | −2.53 | +6.67 | +0.00 |
| c7.88 | +7.33 | +0.47 | +13.35 | −3.63 | +6.63 | +0.00 |
| c15.3 | +3.26 | +0.70 | +25.83 | −3.75 | +6.56 | +0.00 |
| c15.59 | +2.47 | +0.85 | +26.47 | −3.70 | +6.56 | +0.00 |

**R-pad costs essentially nothing, and that is a result rather than a bug.** Up to
38 % of decode lanes are padding, and re-pricing the step at the padded batch moves it
by less than 1 pp. Two reasons, and both matter for STEP B. The bundle's dense table is
nearly flat from 1 to 256 tokens — the card is latency-bound there, `qkv_proj` costs
45 µs at one token and 51 µs at eight — so rounding the batch up buys almost nothing.
And the table's points *are* the bucket points: it was measured on the card, which was
already padding, so the padding is inside the measurement. Charging it again is
double-counting a cost that is already paid. **P2 of STEP B.1 should not be built** on
this bundle; the measured ladder already absorbs it.

**R-128 is a flat +6.6 to +6.7 pp on prefill, at every load.** It does not vary because
it is a property of the token-length distribution, not of the batching, and it sits
against D17's +10.9 %. It is the only mechanism here with a stable, load-independent
magnitude, and it applies to TTFT rather than TPOT.

**R-attn cannot be quantified yet, and the range is the finding.** Depending on which
grouping grid is assumed, the same rule on the same steps reads **−3.7 pp or +26.5 pp**
at c15. Since A.1 showed that neither grid is the rule, neither number is the
contribution. Two things are nevertheless established:

1. *The grouping is already in the bundle, once.* `meta.yaml` states each decode row's
   time is "total decode-attention device time over (forwards × 32)" on sharegpt at
   that concurrency, so D17's 1.95–3.08 executions are inside the number the simulator
   looks up. The obvious reading of the work order's R-attn — replace the lookup with a
   sum over groups — multiplies it in a second time. That miscomputation is reported in
   the table as "R-attn naive" (+31.6 pp at c15) rather than deleted, because its size
   is what makes it easy to make and easy to believe.
2. *What the simulator can be wrong about is the residual diversity*: how ragged this
   run's batches are versus the sharegpt batches the row was measured on. That is what
   the two R-attn columns price, and it is the only part of the mechanism that survives
   a change of workload — which is exactly what STEP B.3's hold-out tests.

### A.4 — the decomposition so far, and the prediction it falsifies

STEP 0 predicted "low-load pessimism (c2–c8) is dominated by R-pad, high-load optimism
(c25–c76) by R-attn and the step policy". The first half is **falsified**: R-pad is
worth under 1 pp everywhere and the low-load +11 % is untouched by it. The second half
is **not yet testable** — A.2 shows the step policy is worth at most +5.4 pp at c59 and
the sign of R-attn is not known.

What can be said with the six measured points:

| mechanism | magnitude | shape | verdict |
| --- | --- | --- | --- |
| step policy, exclusivity | +0.2 to +5.4 pp | grows with load | real but small; **wrong sign at low load** — it makes the +11 % worse |
| step policy, 1024 chunk ceiling | ≤ 0.8 pp | flat | negligible |
| R-pad, batch padding | < 1 pp | flat | **already in the bundle**; do not model |
| R-128, prefill padding | +6.6 pp | flat | real, TTFT only |
| R-attn, KV diversity | −3.7 to +26.5 pp | grows with load | **undecided until C.3** |
| mixed steps | 0.4–7.6 % of time, but 299/300 prefills | grows with load | not yet priced; needs B's P1. The TTFT candidate |

The +11 % pessimism at c2–c4 is explained by **none of them**. Every mechanism with a
settled magnitude is either flat (R-128, which is TTFT) or under 1 pp (R-pad), and the
one that grows with load has the wrong sign there. That is the residual the work order
said not to drive to zero, and at c2–c4 it is currently the whole error.

Two things follow for STEP B, both narrowing it. **P2 (decode quantisation) should not
be built**: its cost is already in the bundle and re-charging it double-counts. **P1
(step policy) is worth building, but for TTFT rather than TPOT** — A.2 priced its TPOT
contribution at under 5.4 pp with the wrong sign below c16, while A.1 shows it changes
the shape of 299 of 300 prefills. That reframes what the prototype is for, and the
work order's risk row for "no mechanism exceeds 5 pp" — build P1 minimally and close —
is the row that currently applies.

---

## §B — the prototype (STEP B)

Spike branch `spike/npu-exec-b-prototype`, **do not merge**. Opt-in through the
cluster JSON:

```json
{"execution_model": "bucketed_aot", "prefill_priority": "strict"}
```

### B.1 — scope, cut down by what STEP A measured

| rule | built? | why |
| --- | --- | --- |
| **P1** step policy | **yes** | prefill/extend steps carry one request, `chunk ≤ 1024`, exclusive of decode |
| P2 decode quantisation | **no** | A.3: the padding is already inside the measured bundle, and re-charging it double-counts (D91) |
| P3 group attention | **no** | A.3: the grouping is also already in the bundle, and the grid it groups on is unknown until C.3 |
| P4 c1 path | unchanged | the control, as specified |
| P5 KV reservation | **no** | untouched by STEP A; still open |

So the prototype is P1 alone, which is what the work order's own risk row
prescribes when no mechanism clears 5 pp. One guard was added rather than a
fallback: `execution_model: bucketed_aot` together with prefix caching **raises**,
because the rule is implemented in `schedule_base` only and falling through to
`schedule_with_prefix` would report the vLLM step policy as if it were the
bucketed one.

`prefill_priority` is `strict` (drain the prefill queue) or `alternate` (take
turns). C.2 is supposed to decide it; both are run here.

### B.2 — equivalence, and what the prototype changes (E-N4)

**Equivalence holds twice over.** The R1/R2 anchors
(`experiments/scripts/d23fix_anchors.sh after_npu_exec_b`) reproduce all four
committed SHA-256 sums from `after_d28`. And the six low-load points re-run
*without* `execution_model` are **byte-identical CSVs** to the pre-edit runs of
§A.1. Absent the key, nothing moved.

**The prototype does change the step structure it was built to change.** Censused
at 300 requests, c15.59, against the same point under the current policy:

| | vLLM | `bucketed_aot` |
| --- | ---: | ---: |
| steps | 13054 | 13063 |
| mixed steps | 299 | **0** |
| prefill steps | 300 | 401 |
| prefill steps at `bs = 1` | 100 % | 100 % |
| prefill tokens | 257,239 | 257,239 |
| mean decode batch | 14.97 | 15.44 |

No step mixes prefill with decode any more, the 300 prompts now take 401 steps
because they chunk at 1024, and the token total is unchanged — the work is the
same work, differently shaped.

TPOT error, nine points, 300 requests, `--match offered`:

| env conc | current | `strict` | `alternate` | served: current → `strict` | measured |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | +2.25 | +2.41 | +2.38 | 1.02 → 1.02 | 1.0 |
| 1.99 | +11.00 | +11.36 | +11.34 | 2.19 → 2.20 | 1.99 |
| 3.98 | +10.79 | +11.58 | +11.57 | 4.31 → 4.35 | 3.98 |
| 7.88 | +7.33 | +9.03 | +9.03 | 8.21 → 8.34 | 7.88 |
| 15.3 | +3.26 | +6.43 | +6.43 | 14.83 → 15.28 | 15.3 |
| 15.59 | +2.47 | +5.79 | +5.80 | 15.21 → 15.68 | 15.59 |
| 29.3 | −3.28 | +1.94 | +1.94 | 25.18 → 26.49 | 29.3 |
| 59.2 | −25.17 | −19.28 | −19.29 | 37.67 → 40.47 | 59.2 |
| 107.2 | −48.59 | −43.73 | −43.73 | 44.47 → 48.23 | 107.2 |

The two comparison criteria the work order fixed in advance:

**(a) Residual span.** Current `[−48.59, +11.00]`, width 59.59 pp. Prototype
`[−43.73, +11.58]`, width 55.31 pp. That is a **7 % narrowing**, against the
"one third of current" the work order names for conclusion (i). It is not close.

**(b) Does the sign flip go away?** **No — it moves.** Under the current policy the
error crosses zero between served 15.21 and 25.18; under the prototype it crosses
between 26.49 and 40.47. The vLLM fingerprint is displaced, not removed.

Two further readings.

**`strict` and `alternate` are indistinguishable.** The largest difference across
all nine points is **0.03 pp**. So the indirect estimate the work order's risk row
proposed for C.2 — run both, report whichever is closer to the measurement —
**cannot discriminate**, and C.2 must be decided by direct observation or left
open. This is a result about the experiment, not about the runtime.

**Queue reproduction improves where it was already close and not where it matters.**
At c15.3 the simulated served concurrency goes 14.83 → 15.28 against a measured
15.3, which is near exact. At the c107.2 arrival rate it goes 44.47 → 48.23
against a measured 107.2. The simulator's throughput ceiling is not the step
policy.

### B.2 — TTFT (E-N4), and the STEP A hypothesis it falsifies

Simulator against simulator, because the measured side is closed-loop (D19). Full
table: `experiments/results/npu_exec_ttft.md`.

| env conc | vLLM p50 | AOT p50 | Δ | vLLM p95 | AOT p95 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 77.0 | 75.8 | −1.6 % | 171.0 | 179.3 | +4.9 % |
| 1.99 | 80.3 | 82.8 | +3.0 % | 179.4 | 191.7 | +6.8 % |
| 3.98 | 84.5 | 81.7 | −3.3 % | 188.1 | 194.2 | +3.2 % |
| 7.88 | 86.6 | 84.1 | −2.9 % | 183.5 | 188.5 | +2.7 % |
| 15.3 | 87.2 | 86.7 | −0.5 % | 187.2 | 192.3 | +2.7 % |
| 15.59 | 87.1 | 84.3 | −3.2 % | 187.7 | 192.7 | +2.7 % |
| 29.3 | 87.9 | 85.9 | −2.3 % | 188.8 | 198.4 | +5.1 % |
| 59.2 | 93.5 | 90.9 | −2.8 % | 187.9 | 191.7 | +2.0 % |
| 107.2 | 95.7 | 90.3 | −5.7 % | 185.3 | 198.4 | +7.1 % |

**§A's TTFT hypothesis is falsified.** A.1 found the simulator mixes prefill with
decode on 299 of 300 prefill steps and proposed that as a candidate for D17's
−32.6 % TTFT gap. Removing the mixing entirely moves simulated TTFT p50 by between
−5.7 % and +3.0 %, and p95 by +2.0 % to +7.1 %. Nothing of that size closes a
32.6 % gap, and p50 mostly moves **down** where it would need to move up.

An intermediate number that looked much better is worth recording so it is not
believed later: the same comparison on a **20-request** run read +14.0 % on p50
and +10.3 % on p95. That is a small-sample artifact — the same drain-tail effect
D32 recorded — and it is the number this spike would have quoted had it stopped at
the smoke test.

### B.3 — hold-out (E-N5): not run

The hold-out workload needs a measured counterpart from STEP C.4, and STEP C has
not run: `npu0`, the card every committed RNGD measurement was taken on, is held
by another tenant's pod (`docs/nodes/npu.md`). Per the work order's own completion
condition this is recorded as **undecided, awaiting measurement** rather than
substituted. Nothing in §B should be read as evidence that the prototype
generalises off sharegpt — that is exactly the question B.3 exists to answer, and
it is open.

---

## §C — hardware (STEP C), on `npu2`

**Which card, and why it matters.** Every committed RNGD measurement — the perf
bundle, the envelope, the accuracy domain, D17's traces — was taken on `npu0`
(PCI `03:00.0`). Throughout this step `npu0` was held by another tenant's
`rngd_pd.serving.cluster --role prefill --backend rngd-full --chip 0` pod, and
`npu1` by its decode counterpart. Only `npu2` (PCI `45:00.0`) was free, and that
is a **different physical card** — the four-card inventory in `docs/nodes/npu.md`
called it `npu3`. So every number below is `npu2`'s, and a result that is a
property of the *card* cannot be merged into the `npu0` bundle without a
cross-card check. A result that is a property of the *artifact* carries.

(The three vendor-level ways of checking whether a card is free all said it was.
That is recorded in `docs/nodes/npu.md`; the process list is what answers it.)

### C.1 — the composed path is single-KV-bucket, and the batch is padded (E-N6)

Artifact `d6ae6a43`, one card at `tp=8`, `workloads/fixedlen-512in-128out-64.jsonl`
— every prompt exactly 512 tokens and every completion 128, so every sequence's KV
stays inside `(0, 1024]`, one decode attention bucket, for the whole run.

| workload | concurrency | wire (composed) hit rate |
| --- | ---: | ---: |
| sharegpt, variable length | 4 | **12.0 %** |
| fixed 512 / 128 | 4 | **97.5 %** |
| fixed 512 / 128 | 3 | **96.0 – 98.1 %** |

**H-b, and H-a is excluded.** Holding the concurrency fixed and removing only the
KV diversity takes the hit rate from 12.0 % to 97.5 %. Under H-a — the batch size
rarely matching a compiled `bs` — changing the prompt-length distribution could
not have done that. A `composed` pipeline is compiled for one
`(batch_size, attention_size)` pair and serves a step only when every sequence in
the batch shares that attention bucket.

**And the batch size is rounded up rather than matched.** Three is not a compiled
decode batch size, yet c3 holds 96–98 %; an exact-match rule would read ~0 %.

This is a property of the artifact's pipeline-selection rule, so it carries from
`npu2` to `npu0`; the percentages themselves are `npu2`'s.

**What it changes.** It corrects the description of the `composed` path recorded
in STEP 0: it is not "batch-1 only", it is "single-KV-bucket only", and sharegpt
simply never presents such a batch above c1. It does **not** move the
decomposition — the prototype leaves the c1 path alone (P4) and real traffic is
still on the kernelwise path at every load the planner cares about.

It does sharpen C.3. The runtime demonstrably *can* run four sequences as one
fused execution when their KV agrees, so whatever caps D17's attention executions
near three is not a per-sequence cost. C.3 should look for a cap on the number of
distinct **groups**.

### C.3 — what one attention execution costs, and what the grid is (E-N8)

Two probe workloads, `npu2`, artifact `d6ae6a43`, c4 / c8 / c16 each: 900-in /
100-out so every sequence's KV stays in `[900, 1000]`, and 1900-in / 100-out so it
stays in `[1900, 2000]`. An EDF stage is named by its compiled bucket, so
`batch_size` **is** the group size that execution covered. Full table:
`experiments/results/npu_exec_attention_groups.md`.

| group size | µs at `attention_size` 1024 | µs at 2048 |
| ---: | ---: | ---: |
| 1 | 48.3 | 38.8 |
| 2 | 54.5 | 54.9 |
| 4 | 55.0 | 76.7 |
| 8 | 68.8 | 108.8 |
| 16 | 96.5 | 171.1 |
| **least squares** | **45.1 + 3.15·n** | **37.1 + 8.54·n** |

**A decode attention execution is mostly fixed cost.** ~40 µs to issue one at all,
plus a per-sequence term that grows with context length. So splitting a batch into
`g` groups costs about `40·g` µs whatever the split; the per-sequence work is paid
either way.

**The cards agree on the structure.** Dividing the committed `npu0` bundle's
per-layer totals by D17's measured executions per layer gives an independent
estimate — **43.1 µs fixed + 7.15 µs per sequence** — from a different card, a
different workload and a different derivation. Against `npu2`'s directly measured
37.1 + 8.54 that is 14 % on the fixed term and 19 % on the slope. The *structure*
transfers; the exact constants are `npu2`'s and are not merged into the `npu0`
bundle.

One thing not explained: at group size 1 the longer bucket is **cheaper**
(38.8 µs at 2048 against 48.3 at 1024), tightly so at both (p05–p95 spans of 1.4
and 1.5 µs). Recorded as observed.

#### The grouping grid is the decode buckets — A.1's open item, closed

Every decode attention execution in the `[900, 1000]` runs used
`attention_size = 1024`, and every one in the `[1900, 2000]` runs used `2048`.
Not one used 896, 768, 640 or any of the kernelwise menu's 128-spaced rungs. Those
finer rungs exist for prefill and extend; **decode groups on the powers of two
from 1024.**

So of A.1's three candidates, the **decode-edge** grid is the runtime's, the
kernelwise menu is excluded, and the uniform-1024 grid was only ever a coincidence
of sharegpt's range.

#### What caps the executions near three: nothing does

D17's saturation needs no cap. The decode ladder is geometric, so the bucket
`[2048, 4096)` covers a 2:1 span of KV; a workload whose KV distribution is
bounded — sharegpt, mean ≈ 2200 — occupies two or three buckets however large the
batch grows. A.1's own census measured exactly that on this grid: 1.05 → 2.64
groups as the mean batch went 1.07 → 14.97. The runtime is not declining to split;
there is nothing left to split into.

This is testable and currently unmeasured: a workload with a **wide** KV spread
should show more groups and more attention executions per layer. That is the
experiment a follow-up work order should run, and it is the same axis B.3's
hold-out was meant to probe.

#### The whole of §C ran without NUMA binding, and that is now checked

Every measurement above was taken with placement left to the kernel scheduler —
no `numactl`, no affinity, and nothing in the artifacts saying so. The repo had
already measured that leaving it to chance is worth **1.93× of throughput** on
this host (`p2_regular_spec_evidence` Appendix A.3); that work landed on `main`
during this session and this spike did not carry it over. **D94** records the debt
and the fix: `--numa-bind` on both harnesses, defaulting to `auto`, plus the
affinity mask in `provenance`.

The check, on the same card and workload, server mask `0-23,48-71` against `0-95`:

| group size | unbound | bound | Δ |
| ---: | ---: | ---: | ---: |
| 1 | 48.3 | 48.3 | +0.02 % |
| 2 | 54.7 | 54.8 | +0.12 % |
| 4 | 50.3 | 50.2 | −0.24 % |
| 8 | 67.6 | 67.6 | −0.05 % |
| 16 | 96.5 | 96.6 | +0.05 % |
| **fit** | **43.7 + 3.20·n** | **43.7 + 3.20·n** | 0.05 % / 0.07 % |

**The cost model stands.** It survives because these are EDF *device* cycles and
NUMA governs host memory and DMA staging, and because `tp=8` here is two fused
quads inside one card — from the host, one engine. That was the argument before
the re-measurement; it is now a result. What is **not** cleared: the wall-clock
half of `measure_envelope.py`, and the committed envelope and accuracy domain,
which earlier sessions took unbound.

#### R-attn, finally a number

With the measured cost model in place of a ratio of group counts — re-grouping
changes only the fixed term, not the per-sequence work — and with the grid
settled, A.3's R-attn stops being a range:

| point | R-attn before C.3 | **R-attn after C.3** |
| --- | --- | ---: |
| c1.0 | −4.47 … −4.46 | **−4.10** |
| c1.99 | −2.65 … −1.18 | **−2.27** |
| c3.98 | −2.53 … +3.91 | **−1.96** |
| c7.88 | −3.63 … +13.35 | **−2.43** |
| c15.3 | −3.75 … +25.83 | **−2.05** |
| c15.59 | −3.70 … +26.47 | **−2.00** |

**−2.0 to −4.1 pp, and flat in load.** The simulator over-charges decode attention
slightly, because it pays for the group structure of the sharegpt batches the
table was measured on while its own batches are marginally less ragged. Two of the
three things that made this look big were errors of method — the double count
(D91) and scaling the per-sequence term along with the fixed one — and the third,
the grid, is now measured.

---

## §D — the decomposition, and the conclusion (STEP D)

### The table

TPOT, sharegpt, 300 requests, `--match offered`. Contributions are percentage
points on the decode-cost sum; the step-policy column is measured by re-simulation
(A.2) and the rest by offline re-pricing (A.3) under C.3's measured cost model.

| served | current err | step policy | R-pad | R-attn | explained | **residual** | prototype err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.02 | +2.25 | +0.16 | +0.00 | −4.10 | −3.94 | **+6.19** | +2.41 |
| 2.19 | +11.00 | +0.28 | −0.01 | −2.27 | −2.00 | **+13.00** | +11.36 |
| 4.31 | +10.79 | +0.71 | −0.36 | −1.96 | −1.61 | **+12.40** | +11.58 |
| 8.21 | +7.33 | +1.48 | +0.47 | −2.43 | −0.48 | **+7.81** | +9.03 |
| 14.83 | +3.26 | +2.74 | +0.70 | −2.05 | +1.39 | **+1.87** | +6.43 |
| 15.21 | +2.47 | +2.93 | +0.85 | −2.00 | +1.78 | **+0.69** | +5.79 |
| 25.18 | −3.28 | +4.56 | — | — | +4.56 | **−7.84** | +1.94 |
| 37.67 | −25.17 | +5.37 | — | — | +5.37 | **−30.54** | −19.28 |
| 44.47 | −48.59 | +4.36 | — | — | +4.36 | **−52.95** | −43.73 |

(The three high-load rows have no A.1 census, so R-pad and R-attn are not priced
there; the work order's completion condition allows a cell to read "undecided"
and these do.)

TTFT is a separate table (§B.2, simulator against simulator) and the mechanism
there is **falsified**, not quantified: removing every mixed step moves TTFT p50
by −5.7 % to +3.0 % against a measured gap of −32.6 %. R-128, the prefill
128-padding, is a flat +6.6 to +6.7 pp and is the only TTFT mechanism with a
settled magnitude.

### Conclusion: **(ii) — some of it, and a smaller some than the question assumed**

Not (i). The prototype narrows the nine-point error span from 59.59 pp to
55.31 pp, a **7 %** narrowing against the one-third the work order set as the bar.
The sign flip does not go away; it moves. Below c16 the prototype makes the error
worse.

Not (iii) either. Three mechanisms have settled, non-zero magnitudes and two of
them are worth formalising:

| mechanism | magnitude | formalise? |
| --- | --- | --- |
| step policy, exclusivity | +0.2 to +5.4 pp, grows with load | **yes** — real and load-dependent, but it is not a TPOT fix; it is what makes the simulator's queue behave, and it took served concurrency at c15.3 from 14.83 to 15.28 against a measured 15.3 |
| R-128, prefill 128-padding | +6.6 pp, flat | **yes** — the largest single settled number in the spike, and the one that lands on TTFT where the gap is −32.6 % |
| R-attn, KV-group diversity | −2.0 to −4.1 pp, flat | **only as the measured cost model** — `g × 43 µs + N × 7 µs`, never as a ratio of group counts, and only once a wide-KV workload has tested it |
| R-pad, batch padding | < 1 pp | **no** — already inside the bundle; modelling it double-counts |
| P3 as the work order wrote it | +31.6 pp | **no** — that number is a double count, not a mechanism |
| mixed steps as a TTFT cause | −5.7 to +3.0 % on TTFT | **no** — falsified by B.2 |

Everything that remains is the residual, and at the low-load end the residual
**is** the error: at served 2.19 the measured error is +11.00 pp and the three
mechanisms together account for −2.00. Nothing in the execution model explains why
the simulator is 11 % pessimistic at concurrency 2.

### The relationship to the accuracy domain

The domain's job is unchanged by this spike, and that is the finding. It carries
the error curve as measurement precisely because the curve is not yet derivable
from mechanism: at the low-load end 100 % of it is residual, and at the high-load
end the prototype recovers 5.4 of 25 pp. Re-measuring the domain under the
prototype would move the low-load points by +0.2 to +3.3 pp in the **wrong**
direction and shift the zero crossing upward from ~16 to ~30 served, so a
prototype-based domain would charge *more* margin below c16 and less above it.
That is not an improvement to buy with a `serving/` change.

`rps_aware`'s E5 winner was not re-evaluated here: the prototype's effect on a
plan is bounded by its effect on the error, and at the operating point that plan
sits at the effect is a few percentage points in the direction that widens
margins. Recorded as **not re-run** rather than estimated.

### What a follow-up work order should and should not do

**Should**: formalise the step policy and the 128-padding as opt-in rules, with
the measured constants; run the wide-KV workload that C.3's saturation explanation
predicts; and take the hold-out measurement B.3 is still waiting on.

**Should not**: build P2 or P3 as the work order describes them, treat the 7 %
narrowing as a case for merging the prototype, or quote the +31.6 pp,
the +14.0 % TTFT or the +26.5 pp R-attn — each is recorded here with the method
error that produced it.

**Open, and cheap**: C.2 (does the runtime alternate prefill and decode steps, or
drain the prefill queue) is still unobserved, and B.2 showed the two knob values
differ by 0.03 pp, so it cannot be settled indirectly. C.5's staircase was not run.
