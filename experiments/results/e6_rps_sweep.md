# E6 — the crossover exists, and only one point on the curve is measured

**Conclusion (i): there is a crossover.** On the card fixture the recommended
backend goes RNGD → cross-vendor P/D → A40 as the arrival rate rises from 1 to 10
rps, at both TTFT points. **But exactly one of the sixteen switchover cells is
labelled `measured`** — the rest are extrapolated or, on the per-PE fixture,
unknown. The headline and that caveat are the same sentence: *the crossover is
real in simulation, and the only cell standing on a measured accuracy domain is
the RNGD card at 1 rps.*

Conclusion (iii) also lands: **asymmetric P/D wins wherever TTFT is tight**, which
is D28 appearing in an answer rather than in a test.

**Knob fixing cost nothing at 1 rps** — regret 0.0000 %, same plan — but for a
reason that limits the reassurance: the winner there is an RNGD-only shape, and
the policy leaves those open by rule. It never restricted the candidate that won.

*`WORK_ORDER_rps_aware.md` rev 2 STEP 5. Run 2026-09-09 on the NPU node. 300
requests, seed 42, 64 workers, `--accuracy-domain`, `--enable-pd`, **no
`--top-k`** (D30), knob fixing from the 10 rps results. Artifacts:
`outputs/e6/`. Figure: `experiments/figures/e6_rps_sweep.png`.*

## Method

| | |
| --- | --- |
| fixtures | `pd-rngd-gpu-card` (RNGD as one card), `pd-rngd-gpu` (RNGD per PE) |
| RPS | 1, 3.3, 10, 20 (whole fleet) |
| TTFT SLO | 64 s and 8 s |
| TPOT SLO | 50 ms, with the margin each candidate's own operating point earns (D29) |
| knobs | **`knob: fixed@10rps`** — see below |
| top-K | **not used.** D30: it is false-infeasible on two of three P/D corpora |

**Knob fixing, and what it assumes.** The RPS axis costs 14.6× a single 10 rps
point, so for each `(arch, backend_mix)` the sweep keeps the one knob combination
that was feasible at 10 rps with the best tok/J. That assumes the knob winning at
10 rps also wins at 1 — the load-independence this work order exists to distrust.
Two things bound it: **shapes with no feasible candidate at 10 rps keep all six
knobs** (every RNGD-only shape is in that bucket, and which knob revives them at
low load is the question), and the regret is measured, not asserted (below).

9 of 17 shapes fixed on the card fixture, 10 of 22 on the other;
`outputs/e6/knob_policy_*.json` carries both tables.

## Time: estimated, then measured

`outputs/e6/time_estimate.md`, written before the run, said **~12.8 h**.
Measured: **5.0 h** — 1.70 h for the card fixture and 3.31 h for the other. The
estimate was 2.6× pessimistic because it assumed every rate started cold; seeding
each fixture's cache from the committed 10 rps results made that point nearly free
(26 s against an estimated 8 min), and the two TTFT points shared every simulation
(the second cost 0.1–0.2 s where the first cost up to 7288 s), which the estimate
allowed for but under-counted.

## Switchover — `pd-rngd-gpu-card`

| rps | TTFT ≤ 64 s | tok/J | TTFT ≤ 8 s | tok/J | margin | validity |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 1 | `agg[furiosa:tp1]` | **1.074** | `agg[furiosa:tp1]` | **1.074** | 3.05 % | **measured** |
| 3.3 | `P[cuda:tp1] D[furiosa:tp1]` | 1.833 | `P[cuda:tp1] D[furiosa:tp1]` | 1.833 | 12.52 % | extrapolated |
| 10 | `agg[cuda:tp4]` | 2.595 | `P[cuda:tp2] D[cuda:tp4]` | 2.400 | 1.42 % | extrapolated |
| 20 | `agg[cuda:tp4]` | 2.762 | `P[cuda:tp1] D[cuda:tp2]` | 2.320 | 1.42 % | extrapolated |

**Crossovers** (both TTFT points, both estimated — nothing was run at the crossing
rate): `furiosa` → `cuda+furiosa` between **1 and 3.3 rps**, then
`cuda+furiosa` → `cuda` between **3.3 and 10 rps**.

**The 3.3 rps winner is cross-vendor P/D, and D16(b) requires that be stated
precisely:** `P[cuda:tp1] D[furiosa:tp1]` — an A40 prefill and an RNGD-card decode,
**both at TP degree 1**, so a shared degree exists and it is 1. It is a symmetric
(`auto`) placement, not `slab3d`. Cross-vendor P/D is out of scope to *build*
before Phase 5 (CLAUDE.md); the generator enumerating and simulating it is
established behaviour and this is a prediction, not a deployment recommendation.

**It also carries the largest margin in the table, 12.52 %,** because its decode
side sits high in the RNGD card's accuracy domain where the simulator is most
optimistic. That is the margin machinery working on the plan it matters most for.

## Switchover — `pd-rngd-gpu`, and why its RNGD rows are not quotable

| rps | TTFT ≤ 64 s | tok/J | TTFT ≤ 8 s | tok/J | margin | validity |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 1 | `agg[furiosa:tp4]` | 1.602 | `agg[furiosa:tp4]` | 1.602 | **0.00 %** | **unknown** |
| 3.3 | `agg[furiosa:tp8]` | 3.477 | `agg[furiosa:tp8]` | 3.477 | **0.00 %** | **unknown** |
| 10 | `agg[furiosa:tp8]` | **4.956** | `P[cuda:tp2] D[cuda:tp4]` | 2.398 | 0.00 / 1.42 % | unknown / extrapolated |
| 20 | `agg[cuda:tp4]` | 2.762 | `P[cuda:tp2] D[cuda:tp4]` | 2.657 | 1.42 % | extrapolated |

**The 4.956 at 10 rps is D22's retracted headline, reproduced exactly.** That is
not a coincidence and not a rehabilitation. This fixture uses the **per-PE** RNGD
profile, which has **no measured accuracy domain**, so every RNGD row here carries
margin **0.00 %** and validity **unknown** — they are the unmargined numbers the
retraction was about. `docs/CLAIMS.md` §3 still stands and nothing in this table
changes it.

The contrast with the card fixture is the point of the machinery: the *same*
device, profiled two ways, is margined where a domain exists and refused a margin
where none does. E5 shows what the margin then does — it rejects the card
fixture's committed winner outright.

## What is not evaluated, and why the counts are here

Per point: **68 `sim_error` on the card fixture, 114 on the other**, plus 4
`memory_infeasible`. These are not timeouts — D25-b removed those — they are the
simulator's KV allocator refusing a candidate
(`[MemoryModel] tried to load 92.00MB but only 25.49MB is available`), which is a
deterministic, reproducible failure of a candidate the generator's memory bound
let through.

They are printed because a sweep row without them cannot be read: this driver used
to report INFEASIBLE identically whether a candidate was rejected or never
evaluated, and on 2026-09-03 that hid 71 of 222 timeouts including the committed
winner (`HANDOVER.md` §3). **Every "best" in the tables above is the best of what
evaluated**, over 318 of 528 generated candidates on the card fixture and 357 of
the other's, after knob fixing.

## Why almost everything says "extrapolated"

Not because the plans are shaky — because the **A40 accuracy domain has exactly one
point**, at served concurrency 170.56, and every plan here runs far below it. A
single point carries no slope, so its 1.42 % margin is held flat and
`in_calibration_domain` is correctly false. The one `measured` cell is the RNGD
card at 1 rps, where the operating point falls inside the domain's measured
[1.09, 76].

**The fix is a second A40 measurement at a different load, and this node has no
NVIDIA GPU.** Until then the label is doing its job: it says the margin is being
reused outside where it was fitted, which is exactly what is happening.

## E6b — the measured half

tokens/J against arrival rate from the STEP 3 silicon curve, one RNGD card:

| served conc | rps/card | **tok/J** |
| ---: | ---: | ---: |
| 1.00 | 0.097 | 0.418 |
| 1.99 | 0.169 | 0.783 |
| 3.98 | 0.308 | 1.436 |
| 7.88 | 0.547 | 2.492 |
| 15.59 | 0.917 | 3.941 |

The accuracy domain reproduces its own points when queried at the simulation-side
concurrency, which is a self-consistency check on how it was recorded, not
independent confirmation.

**The asymmetry must survive into any sentence written from this.** RNGD is
measured silicon; the A40 side exists only in simulation, because
`scripts/whichnode.sh` reports no NVIDIA GPU on this node. A crossover claim reads
**"RNGD measured against A40 simulated"** and never "RNGD against A40".

## E7 — how much of this rests on any one calibration point

The work order frames E7 against the *envelope*. E6a does not use the envelope —
`--envelope-prefilter` was off, because rejecting candidates before simulating them
would remove the very rows the switchover table is made of. What E6a does use is
the **accuracy domain**, so that is what was perturbed. The substitution is noted
rather than silent. Pure replay from the sweep's cache: nothing was re-simulated,
so the domain is the only thing that changes between variants.

**Removing any single domain point never changes the winning backend** — 0 of 24
leave-one-out variants across four rates. The domain's shape does move the margin:
dropping the 76 point widens the 3.3 rps margin from 12.52 % to **38.01 %**,
because `widen_error_bars` then extrapolates from a shorter baseline with a steeper
slope, and dropping 16.6 *lowers* it to 10.35 % (and to 0.00 % at 1 rps) by moving
where the interpolation crosses zero. Conservative in the first direction and
permissive in the second, both as designed.

**Removing the domain entirely flips one cell, and it is the interesting one.** At
3.3 rps the winner goes from `P[cuda:tp1] D[furiosa:tp1]` at 1.833 tok/J to
`agg[furiosa:...]` at **3.037 tok/J**. With no margin an RNGD-only plan wins and
looks 66 % more efficient; with the measured margin it does not win at all.

So the answer to "how sensitive is this?" is: **not to any one point, but entirely
to having a domain.** That is the same shape as D22 — no individual number was
wrong there either; what was missing was the correction as a whole.

`outputs/e6/e7_domain_sensitivity.json`.

## Knob-fixing regret at 1 rps: zero, and the rule is why

The 1 rps point re-run on the card fixture with **every** knob — 528 candidates
against the fixed set's 318:

| TTFT | | evaluated | winner | tok/J | margin |
| --- | --- | ---: | --- | ---: | ---: |
| 64 s | fixed | 318 | `agg[furiosa:tp1]` | 1.0739 | 3.05 % |
| 64 s | **all knobs** | **528** | `agg[furiosa:tp1]` | **1.0739** | 3.05 % |
| 8 s | fixed | 318 | `agg[furiosa:tp1]` | 1.0739 | 3.05 % |
| 8 s | **all knobs** | **528** | `agg[furiosa:tp1]` | **1.0739** | 3.05 % |

**Regret 0.0000 %** — the same plan, the same tok/J to four decimals, the same
margin.

That is not luck, and it is not evidence that knob fixing is generally safe. The
1 rps winner is `agg[furiosa:...]`, an **RNGD-only shape**, and RNGD-only shapes
are exactly the ones the policy leaves open because none was feasible at 10 rps.
Knob fixing never restricted the candidate that won. The rule protected the
answer, and the measurement confirms the rule rather than the heuristic:

> Shapes with no feasible candidate at 10 rps keep all six knobs, because which
> knob revives them at low load is what E6 asks.

**What this does not license.** Zero regret on one fixture at one rate for a shape
the policy never touched says nothing about the cuda shapes it did fix. Those
remain a labelled heuristic — every cuda row in the tables above carries
`knob: fixed@10rps` — and measuring their regret would mean re-running a rate where
a cuda plan wins with all six knobs restored.

## Reading the cache-hit counts

A row's `cache_hits` can exceed what was written at that rate, and that is
correct: `planner/envelope.py:key_for` keys on **placement**, not candidate id, so
two candidates that compile to the same placement and knobs share one simulation.
At 1 rps, 318 candidates resolved to 121 distinct keys and 197 hits. It is the same
mechanism behind the renderer's "N other candidate(s) predict exactly this
outcome".

---

# Re-run 2026-09-10/11 — the crossover survives the domain gaining a slope

*Everything above is the 2026-09-09 run on the NPU node and is left unchanged.
This section is the same sweep re-run on the **A40 node** after
`profiles/calibration/a40.accuracy.yaml` went from one point to three
(`docs/HANDOVER.md` §2.2). Same service spec, same two cluster fixtures, same
four rates, same two TTFT points, same knob policy files, same seed, same 300
requests. **The only input that changed is the A40 accuracy domain.** Artifacts:
`outputs/e6_rerun/`.*

## The answer, in one line

**Seven of sixteen switchover cells move from `extrapolated` to `measured`. Every
winner and every tok/J is unchanged to full precision.** The crossover was right;
it is now standing on a measured accuracy domain over most of its range instead
of on one point reused outside where it was fitted.

| validity | before | after |
| --- | ---: | ---: |
| `measured` | 2 | **9** |
| `extrapolated` | 9 | **2** |
| `unknown` | 5 | 5 |

Seven was also the **ceiling**, computed from the committed operating points
before the measurement was taken, and the run hit it exactly. The two cells that
stay `extrapolated` and the five that stay `unknown` cannot be fixed by measuring
the A40 harder, for reasons given below.

## What did not move, and why that is the interesting part

Not one winner changed. Both mechanisms that could have moved one were checked:

**TTFT margins rose sharply and still do not bind.** The new domain says the
simulator is ~18 % optimistic on TTFT at low load rather than 1.97 % everywhere,
so the TTFT margin at these operating points went from a flat 1.97 % to 2.0–9.6 %.
It changes nothing because **no recommended plan is anywhere near its TTFT SLO** —
the worst headroom in the table is 74.6 % of the SLO and most cells sit under
25 %. TTFT is simply not the binding constraint in this regime; TPOT is.

**TPOT margins got slightly SMALLER, not larger.** Interpolating between the
low-load points (where the simulator is pessimistic, +0.59 %) and 170.56
(−1.42 %) gives magnitudes below the flat 1.42 % the single point imposed:

| fixture | rps / TTFT | TPOT margin before | after | winner's TPOT vs 50 ms SLO |
| --- | --- | ---: | ---: | ---: |
| card | 10 / 64 s | 1.42 % | **1.11 %** | 71.2 % |
| card | 10 / 8 s | 1.42 % | **1.20 %** | **93.7 %** |
| card | 20 / 64 s | 1.42 % | **1.33 %** | 72.0 % |
| card | 20 / 8 s | 1.42 % | **0.87 %** | **92.9 %** |
| per-PE | 10 / 8 s | 1.42 % | **1.08 %** | 74.5 % |
| per-PE | 20 / 64 s | 1.42 % | **1.33 %** | 72.0 % |
| per-PE | 20 / 8 s | 1.42 % | **1.42 %** | 85.7 % |

So the domain became marginally more permissive exactly where it became measured,
and the plans that already passed still pass. **The tight-TTFT cells are the ones
to watch**: their winners run at 92.9 % and 93.7 % of the TPOT SLO, so a TPOT
margin above about **6.7 %** would reject them. The measured domain gives ~1 %.
That is a much stronger statement than the same conclusion drawn from an
extrapolated 1.42 %, and it is the difference this work bought.

## The two cells that stay `extrapolated`, permanently

Both are the card fixture at 3.3 rps, and both are held there by the same thing:
the winner is `P[cuda:tp1] D[furiosa:tp1]`, whose A40 leg is a **prefill role at
served concurrency 0.499**. The domain now starts at 4.043. Closing that gap
would mean measuring a server that is idle 99 % of the time, which is not an
operating point a bench can hold — so this cell is not "not yet measured", it is
**not measurable by this method**, and the label is doing exactly its job. Its
12.52 % margin comes from the RNGD-CARD decode leg, which is in-domain, so the
cell is better founded than the label alone suggests.

## The five `unknown` cells are untouched, as expected

Every one is a per-PE fixture cell whose winner is RNGD-only. The per-PE RNGD
profile has **no accuracy domain at all**, so those plans carry margin 0.00 % and
validity `unknown` — including the `agg[furiosa:tp8]` at 4.956 tok/J, which is
D22's retracted headline reproduced exactly. Nothing here rehabilitates it.
`docs/CLAIMS.md` §3 still stands. An A40 measurement cannot reach these cells;
only a per-PE RNGD domain could, and that needs the NPU node.

## Cross-node reproduction

The simulation half reproduced on different hardware: **`generated` and
`evaluated` are identical at all sixteen cells** (528/318 on the card fixture,
616/357 on the other) and the recommended plan's tok/J is identical to full
precision everywhere. Two differences, neither substantive:

- **`rejected_summary` is richer here.** The committed run recorded nothing at
  the 1 and 3.3 rps cells — the persistence gap `docs/HANDOVER.md` §3 describes —
  while this run records `sim_error` counts there. Where both recorded, at 10 and
  20 rps, the counts are **identical**: 68/68 on the card fixture, 114/114 and
  118/118 on the other. That agreement is the evidence that the simulator itself
  behaves the same on both nodes; the 1 and 3.3 rps figures were simply missing
  before.
- **Wall time is 2.5–2.8× longer**: 4.18 h and 9.39 h against 1.70 h and 3.31 h.
  Partly 64 cores against 96, and partly that failing candidates are not cached,
  so the second TTFT point re-attempts them instead of running nearly free.
