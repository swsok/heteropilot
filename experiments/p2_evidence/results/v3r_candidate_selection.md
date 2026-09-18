# V3-R — which single-card RNGD candidate could decide a rule, and which cannot

*`WORK_ORDER_domain_scoping.md` STEP S7.0, rev 2. Run 2026-09-18 on the **NPU
node** (`whichnode.sh` → `npu`; RNGD npu0 `03:00.0`, npu1 `04:00.0`, npu3
`45:00.0`). **No card was used and nothing was deployed** — every number here is
the simulator's, produced by `experiments/p2_evidence/v3r_select_candidates.py`.
The work order requires this table to be confirmed before the card is touched,
which is the rule V3's own CPU half followed.*

Generated tables: `outputs/p2_evidence/v3r/v3r_tables.md`. Machine-readable
record: `v3r_candidates.json`. Run log: `outputs/p2_evidence/v3r/select.log`.

> **Read this before approving.** Two of the findings run against the work
> order's own expectations, and one of them removes half of what S7 was written
> to deliver.
>
> 1. **The P2-shape — the candidate the invention CATCHES — is measurable here,
>    and comfortably.** `s128-t2048` at **3.5 rps** sits at served concurrency
>    **74.50** with a predicted p99 TPOT of **46.691 ms**: rule (a) passes it
>    with 3.309 ms of headroom and rule (c) rejects it by **6.640 ms**, at the
>    fixture's own 50 ms SLO. On the A40 the same disagreement was 0.3 ms wide
>    and V3 called it unmeasurable.
> 2. **The P3-shape — the candidate the invention RESCUES — is measurable too,
>    but NOT at the fixture's own SLO.** At 50 ms it exists only at 3.0 rps and
>    its separations there are 0.588 / **0.236 ms**, tighter than the A40 P3 V3
>    already called noise-bound. Lowering the TPOT SLO moves the case down the
>    TPOT curve to a smaller `L`, where the per-point margin is smaller and the
>    gap to the global 18 % is wider: at **40 ms** the 2.0 rps candidate
>    separates by **1.748 / 1.963 ms** with a margin of 7.51 %. §3 has the
>    mechanism and the cost of using a post-hoc SLO.
> 3. **Both candidates are margined by a line between one measured point and one
>    interpolated one.** §4. This is a requirement handed to S7.3, not a caveat
>    to be noted and forgotten.
> 4. **Rule (e) holds all 72 rows**, at every rate, on `arrival_process`. The
>    S7.1 gate is confirmed by measurement rather than by reading S4's file.

## The fixture, and why it is not a new one

`experiments/configs/clusters/pd-rngd-gpu-card.yaml` with
`examples/service_specs/llama31-8b.yaml`, 300 requests, seed 42 — E-A1's and
V3's own corpus, so the rows here are comparable with theirs. Only
`traffic.arrival_rate_rps` is varied, through `spec.model_copy(deep=True)`, the
same mechanism `plan --rps` uses (`planner/__main__.py`).

**Candidate scope: `tp=1, pp=1, dp=1`, single island, on `node_rngd0`.** Both
halves are required. The card domain was fitted at that configuration, so
anything else mismatches on `dp`/`islands` too and would hide what an
arrival-process refusal did; and there is **no furiosa deploy backend**
(`planner/deploy/` has cuda, an ascend stub and kubernetes), so S7.4 drives one
engine by hand and a multi-island candidate would need the router Phase 4 does
not have. That leaves six knob settings — `s{32,128,256}` × `t{2048,8192}` —
and `node_rngd1`'s six are their cache mirrors (§5).

**The four rules are E-A1's, imported not re-implemented.** `_conditions` comes
from `experiments/uncertainty/ea1_margin_modes.py`, so (a)–(d) carry
`condition_mismatch="warn"` and (e) carries `refuse`, exactly as S4 built them.
This experiment cannot drift from the one the disclosure quotes.

## 1. Why P2 and P3 are read as verdict shapes here

V3's P2 and P3 are `mix(cuda-a40-node_a40a…+cuda-a40-node_a40b…)` candidates.
There is no NVIDIA driver on this node, so neither can be run here and absolute
rule 3 forbids reporting anything as if it had been. What transfers is not the
topology but **what each group was for**:

| shape | definition | what a measurement would show |
| --- | --- | --- |
| **P2-shape** | (a) no margin passes · (c) per-point rejects | the invention **catches** an optimistic candidate |
| **P3-shape** | (b) global 18 % rejects · (c) per-point passes | the invention **rescues** a candidate a blunt margin excluded |

On the A40 both happened to be mixes because that is where any separation
existed at all — the per-point margin there runs 0.37–1.63 %, so the (a)/(c)
band is at most 0.80 ms wide (`v3_candidate_selection.md` §1). On RNGD the
margin runs to ~21 %, and §2 shows single-card candidates reach both shapes.

## 2. The selection

Full sweep in `v3r_tables.md` T1. The rows that matter:

Verdicts in this table are at the fixture's own **TPOT SLO of 50 ms**. The `shape`
column is therefore what each row exhibits *at 50 ms*; §3 shows the P3-shape moves
to a different row once the threshold is allowed to move.

| rps | knob | L | in dom | p99 TPOT | p99 TTFT | m(c) % | a | b | c | d | e | shape @50 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| **2.0** | **s128-t8192** | **37.975** | yes | **35.380** | **368** | **7.51** | pass | pass | pass | pass | held | — (**P3 at 40 ms**, §2.3) |
| 2.5 | s128-t2048 | 49.527 | yes | 38.480 | 450 | 11.52 | pass | pass | pass | pass | held | — |
| 3.0 | s128-t2048 | 61.677 | yes | 42.871 | 471 | 16.08 | pass | **fail** | pass | pass | held | P3-shape (0.236 ms, §2.2) |
| **3.5** | **s128-t2048** | **74.498** | yes | **46.691** | **582** | **21.31** | pass | fail | **fail** | fail | held | **P2-shape**, §2.1 |
| 4.0 | s128-t2048 | 87.440 | **NO** | 50.384 | 626 | — | fail | fail | fail | fail | held | — |

The `in dom` column is load-bearing: at 4.0 rps the operating point has passed the
domain's top measured point (76.0), so no margin there is a lookup at all — T4's
largest separations all sit in that region and none of them is selectable (§3).

**TTFT never binds on the selected rows** — 368 ms, 471 ms and 582 ms against a
25 000 ms SLO — so TPOT alone decides them, which is what makes them tests of the
TPOT margin. That is not true of the `s32` knob at any rate above 1.5 rps: it caps the
batch at 32, the queue grows, and its p99 TTFT runs 5.250 s at 1.5 rps to 105.698 s at
4 rps (T1). Those rows are rejected **on TTFT**, and a measurement of one would
be measuring the wrong thing.

### 2.1 The recommended P2-shape candidate

```
furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2048   at arrival_rate_rps 3.5

served concurrency L         74.498      (domain measured range [1.02, 76])
predicted p99 TPOT           46.691 ms
own per-point margin m       21.308 %
robust TPOT, rule (c)        46.691 x 1.21308 = 56.640 ms

(a) passes  by 50 - 46.691                      =  3.309 ms
(c) rejects by 56.640 - 50                      =  6.640 ms
```

**A measurement can resolve this.** If the hardware comes in at or below 50 ms,
rule (c) rejected a candidate that worked and the per-point margin over-corrected
at this operating point. If it comes in above 50, (a) passed a candidate that
violates and the margin caught it. Either way the answer is 3.3–6.6 ms away from
the threshold, against a p99 spread that V3 measured on the A40 at well under
1 ms and that S7.3's three repeats will establish for RNGD.

Second choice, for the alternates the work order asks for in advance:
`s128-t8192` at 3.5 rps (L = 74.226, TPOT 46.442, m = 21.192 %, seps 3.558 / 6.284).
`s256-*` at 3.5 rps duplicate `s128-*` exactly — the batch cap is not reached at
this load, so `s256` is not an independent knob setting here and buys no second
sample.

### 2.2 The P3-shape at the fixture's own SLO — a near miss

```
furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2048   at arrival_rate_rps 3.0

served concurrency L         61.677
predicted p99 TPOT           42.871 ms
own per-point margin m       16.078 %
robust, rule (b) global 18 % 42.871 x 1.18    = 50.588 ms
robust, rule (c) per-point   42.871 x 1.16078 = 49.764 ms

(b) rejects by 50.588 - 50                    =  0.588 ms
(c) passes  by 50 - 49.764                    =  0.236 ms
```

**0.236 ms of headroom is not a verdict a measurement can confirm.** It is
tighter than V3's A40 P3 (0.623 ms), which the work order already recorded as
needing to be read against repeatability rather than on its own. §3 shows what
does fix it — a lower TPOT SLO, which moves the case to `L = 37.98` and widens
the separation to 1.748 ms — and what using a post-hoc threshold costs.

### 2.3 The recommended P3-shape candidate, at a 40 ms TPOT SLO

```
furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192   at arrival_rate_rps 2.0
judged against a TPOT SLO of 40 ms  (POST-HOC -- not the fixture's 50 ms, §3.1)

served concurrency L         37.975      (domain measured range [1.02, 76])
predicted p99 TPOT           35.380 ms
predicted p99 TTFT              368 ms   (vs the 25 000 ms SLO: TPOT decides)
own per-point margin m        7.511 %
robust, rule (b) global 18 %  35.380 x 1.18     = 41.748 ms
robust, rule (c) per-point    35.380 x 1.07511  = 38.037 ms

(b) rejects by 41.748 - 40                      =  1.748 ms
(c) passes  by 40 - 38.037                      =  1.963 ms
```

**A measurement can resolve this.** If the hardware comes in at or below 40 ms,
the global 18 % excluded a candidate that worked and the per-point margin was
right to keep it — the disclosure's rescue case, with 1.748 ms of room. If it
comes in above 40, the per-point margin was too small at this operating point and
the blunt one was accidentally right; that is a finding against the invention and
it is 1.963 ms away, so the measurement can say so.

Alternate: `s128-t2048` at 2.0 rps (L = 37.965, TPOT 35.332, m = 7.507 %, seps
1.692 / 2.016). Larger-margin alternate, if a margin nearer the headline is
wanted: `s128-t8192` at 2.5 rps (L = 49.462, m = 11.499 %, SLO 44, seps
1.293 / 1.202) — a wider margin bought with half the separation.

## 3. Why the P3-shape needs a different SLO, and what that costs

At the fixture's own 50 ms the P3-shape is a near-miss, and the reason is that
the two rules converge exactly where the candidate arrives. Write `T` for the
predicted p99 TPOT and `m` for the candidate's own margin. A P3-shape needs

```
(1 + m) T  <=  SLO  <  1.18 T
```

so its whole window is `T x (0.18 - m)` wide and the best either separation can
be is half of that. The window closes as `m` approaches 18 %, which the
committed domain reaches at **L = 66.52**. On this fixture `T` and `m` both grow
with `L`, and they grow at rates that put `T` inside the window only after `m`
has nearly closed it:

| L | m % | T | P3 window `T(0.18-m)` | where T sits |
| ---: | ---: | ---: | ---: | --- |
| 27.00 | 3.96 | 33.056 | 4.64 ms | 9.3 ms **below** the window (at SLO 50) |
| 37.97 | 7.51 | 35.332 | 3.71 ms | 7.0 ms below |
| 49.53 | 11.52 | 38.480 | 2.49 ms | 3.9 ms below |
| **61.68** | **16.08** | **42.871** | **0.82 ms** | **inside, 0.236 ms of headroom** |
| 74.50 | 21.31 | 46.691 | none (m > 18 %) | unreachable |

**Moving the SLO is what opens it, and it is legitimate.** A lower threshold
meets the TPOT curve at a lower `L`, where `m` is smaller and `0.18 - m` is
therefore wider. T4 sweeps every row against every TPOT SLO from 20 to 100 ms in
0.5 ms steps — arithmetic only, no simulation, because a row's `T` and its own
`m` are fixed and only the threshold moves. Restricted to rows whose operating
point is **inside** the domain's measured range (the `in domain` column; outside
it the margin is an extrapolation and rules (d)/(e) hold the candidate anyway),
the best P3-shapes are:

| rps | knob | L | m % | SLO | (b) rejects by | (c) passes by | min sep |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | s128-t8192 | 7.325 | **0.00** | 29.0 | 2.330 | 2.449 | 2.330 |
| 1.5 | s128-t8192 | 27.005 | 3.96 | 36.5 | 2.499 | 2.142 | 2.142 |
| **2.0** | **s128-t8192** | **37.975** | **7.51** | **40.0** | **1.748** | **1.963** | **1.748** |
| 2.5 | s128-t8192 | 49.462 | 11.50 | 44.0 | 1.293 | 1.202 | 1.202 |
| 3.0 | s128-t2048 | 61.677 | 16.08 | 50.0 | 0.588 | 0.236 | 0.236 |

**The widest separation is not the best case, and this is the choice to make
deliberately.** The 0.5 rps row separates by 2.330 ms but its margin is
**0.00 %** — at `L = 7.3` the simulator is *pessimistic* (`e = +2.25 %`) and the
one-sided rule correctly charges nothing. That is a real P3-shape and it does
demonstrate the global 18 % over-rejecting, but rule (c) is then identical to
rule (a), so a measurement there tests that the domain charges nothing where
nothing is due — not that it charges the right amount where something is.

**So the recommended P3-shape candidate is 2.0 rps at a 40 ms TPOT SLO**: a
margin of 7.51 % that is plainly neither 0 nor 18 %, and 1.748 ms of separation,
which is 7x V3's A40 P3 headroom and an order above any p99 spread the RNGD
repeats are likely to show. `s128-t2048` at the same rate is the alternate
(1.692 / 2.016). `s256-*` duplicate `s128-*` at this load.

### 3.1 What using a post-hoc SLO costs, stated rather than hidden

The 40 ms threshold is **not the fixture's SLO**, and three things follow that
S7.4 and any write-up must carry:

* **E-A1's 50 ms aggregate is not re-scored.** Nothing under
  `outputs/uncertainty/ea1/` is touched. This is the discipline V3 §6.2 set when
  it swept the SLO over one deployment, and it is the same move here.
* **The P2 and P3 cases are then measured at different thresholds** — P2 at the
  fixture's 50 ms, P3 at 40 ms — so they are two cases, not one grid. S7.4
  should report each with its threshold named in the row, never a bare pass.
* **The ≈21 % margin and the measurable P3-shape do not coexist.** The handover
  and the work order both motivate S7 by "the margin is ≈21 % only in the RNGD
  region"; that margin belongs to `L ≈ 74.5`, which is P2 territory. The P3 case
  lives at 7.51 %. The disclosure should not quote 21 % as the margin that
  rescued a candidate, because it is not.
* **A third option exists and is cheaper to describe**: run the 3.0 rps
  candidate at 50 ms anyway and report that (c) passed by 0.236 ms *inside* the
  measured spread. That is an honest negative result about resolution, not a
  demonstration, and it is not what §6 needs.

## 4. What both candidates rest on, and what S7.3 must therefore measure

All three selected operating points fall in the interval **(25.181, 76.0)**, and
the committed domain has **no point inside it**:

```
  ... 16.6   tpot_err_pct  -3.1    EDF-rebuilt bundle fit, "Not re-measured here"
      25.181 tpot_err_pct  -3.28   MEASURED, "First measured point between the two anchors"
      ------ L = 37.975 (P3-shape, §2.3), 61.677 (P3 at 50 ms, §2.2)  ------
      ------ and 74.498 (P2-shape, §2.1) all land in here                ------
      76.0   tpot_err_pct -18.0    "interpolated c64/c128 ... Not re-measured here"
```

So the margins of 7.51 %, 16.08 % and 21.31 % that decide every verdict above are
read off a straight line drawn from **one measured point to one interpolated
point**, across an interval containing no measurement. The candidate that makes
the invention look best is the one sitting where the domain is thinnest — 74.498
is 98.0 % of the way to the top anchor, and that anchor was never measured.

**This is a requirement for S7.3, stated as one:** the open-loop refit must place
measured points inside `L ∈ [30, 76]` — at or above 70 for the P2 case, and near
38 for the P3 one — or those margins will still be interpolations dressed as
lookups after the refit. Two or three points in that interval is the difference
between S7.4 testing a margin and S7.4 testing a straight line.

## 5. Controls

**The rules reproduce E-A1 at the fixture's own rate.** The sweep's 10 rps rows
are E-A1's operating points, and they agree with the committed record: `s32-t2048`
L = 139.838 / p99 TPOT 32.794, `s128-t2048` L = 144.609 / 55.723 — the same values
`outputs/uncertainty/ea1/cache` carries. **T6 checks all six rows** against that
cache and is regenerated with the tables, so the control is an artifact rather
than an assertion; all six agree to under 5e-3.

**The D40 mirror control.** `node_rngd1`'s candidates are served `node_rngd0`'s
cache entries, because the two islands are one card model on one profile and the
key collides by design. D40 establishes that dedup is sound for single-island
aggregated candidates and unsound for mirrored `mix(...)` ones, so the sound case
is the one relied on here — but sound is not verified, so `--verify-mirror`
re-simulates one `node_rngd1` candidate with **the cache disabled** and compares,
the same control V3 §2.1 ran on its own two. Result in T5.

**The margin arithmetic agrees with the code.** `WORK_ORDER_domain_scoping.md`
§S7.B derives the margin at a given `L` by interpolating the domain's points and
applying `m = -e/(1+e)`. Independently, `calibration.AccuracyDomain` reports
7.507 % at L = 37.97 and 13.828 % at L = 55.80; the §S7.B formula gives 7.508 %
and 13.828 %. The work order's table and the planner are the same function.

## 6. Reproducing

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/p2_evidence/v3r_select_candidates.py \
    --rps 0.5,1,1.5,2,2.5,3,3.5,4,5,6,8,10 --workers 6 --verify-mirror \
    --out experiments/p2_evidence/results/v3r_candidates.json

.venv/bin/python experiments/p2_evidence/v3r_tables.py \
    --json experiments/p2_evidence/results/v3r_candidates.json \
    --out outputs/p2_evidence/v3r/v3r_tables.md
```

About 40 s per candidate-rate at 6 workers; the low rates are the slow ones,
because 300 requests at 0.5 rps is 600 s of simulated arrival time. The runs are
bounded by the predictor's own `timeout_s`; `livelock_watch.sh` is not used
because it reads one simulator's progress lines and this driver spawns many short
ones.

## 7. What this does not settle

- **Nothing is measured.** Every figure is the simulator's prediction. The whole
  point of S7.4 is that these predictions have no RNGD ground truth at an
  open-loop operating point.
- **Rule (e) holds every row**, so the planner's default policy recommends
  nothing here. That is S7.1's gate and it is why S7.2/S7.3 come before S7.4.
  The (a)–(d) columns above exist because `condition_mismatch="warn"` was used to
  produce them, exactly as S4 had to for E-A1.
- **The 300-request trace is short for L ≈ 74.** A steady-state interval and a
  warm-up exclusion are S7.3/S7.4's requirement (V3's transient lesson); nothing
  here checks that the simulated window has one.
- **The P3 case needs a threshold the fixture does not declare** (§3.1), so it is
  a sensitivity result and must be reported as one. The P2 case does not — it
  stands at the fixture's own 50 ms.
- **The two cases are two deployments**, at 3.5 rps and 2.0 rps. S7.4's budget is
  two open-loop runs x 3 repeats, not one.
- **`node_rngd0` is a fixture island, not a card.** Which physical card S7.4
  runs on is S7.3's decision — npu0 at BDF `0000:03:00.0`, because every
  committed RNGD measurement is on that silicon. The `npuN` labels move: npu.md
  recorded `45:00.0` as `npu2` on 2026-09-17 and `furiosa-smi` calls it `npu3`
  today.
