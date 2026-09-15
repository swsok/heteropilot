# E-B2 on F2 — every false positive is one defect, and the recall collapses where the plan has no incumbent

`WORK_ORDER_uq_stage_b_plus.md` STEP C3. Two things at once: §2.4's α/β/γ
classification of the false positives, and the same flip detection re-run on the
F2 corpus STEP C1 built (P/D on, TTFT SLO 8 000 ms, D40 mirrors collapsed).

Both corpora were re-run here, because the committed E-A1 result predated D70 —
the accuracy-domain margin was `-e` where the error's denominator is the
measurement — and D70 moves the margins that decide every flip. The numbers in
`eb2_flip_detection.md` are the regenerated ones; the pre-D70 values are kept
below for the comparison and nowhere else.

```bash
# E-A1 corpus (regenerated on the D70 margin)
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb2_flip_detection.py \
    --cache-dir outputs/uncertainty/ea1/cache --truth-sweep \
    --out-json outputs/uncertainty/eb2/eb2_flip_detection.json

# F2 corpus
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb2_flip_detection.py \
    --fixture experiments/uncertainty/fixtures/f2.json \
    --cache-dir outputs/uncertainty/f2/cache \
    --work-dir outputs/uncertainty/eb1_f2/work --truth-sweep \
    --out-json outputs/uncertainty/eb2_f2/eb2_f2_flip_detection.json
```

Raw: `outputs/uncertainty/eb2/eb2_flip_detection.json`,
`outputs/uncertainty/eb2_f2/eb2_f2_flip_detection.json` — per-case rows, so every
false positive and false negative can be read individually.

> **Cache replay, no measurement.** Both runs re-judge simulations already in the
> envelope caches; neither launches the simulator and neither touches an
> accelerator. Provenance stamps the node this ran on (one RTX A5000) because
> that is what `provenance.collect` records, **not** because any number here was
> measured on it. The measured inputs are the committed RNGD-CARD and A40
> domains, taken on the nodes that own them.

## The answer, both corpora

*616 cases per grid, 1,848 per run. m = 3, 5 and 9 agree case for case in both
corpora, so one row stands for all three.*

| corpus | n | TP | FP | FN | TN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E-A1, pre-D70 (superseded) | 616 | 58 | 17 | 1 | 540 | 0.773 | 0.983 |
| **E-A1, on the D70 margin** | 616 | 102 | **19** | 1 | 494 | **0.843** | **0.990** |
| **F2** | 616 | 36 | **15** | **97** | 468 | **0.706** | **0.271** |

D70 did not merely shift the margins: it raised the true positives from 58 to
102. Restoring an input changes the winner far more often once the margin is
computed with the measurement in the denominator, which is the direction D70
predicted and is why the old artifact could not be kept.

**F2's recall is 0.271 against E-A1's 0.990.** That is the result of this step,
and §5's four words for it are below.

## §2.4's classification: there is no α, no β and no γ

| corpus | false positives | α structural | β approximation | γ equivalence | δ (added) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E-A1 | 19 | 0 | **0** | 0 | **19** |
| F2 | 15 | 0 | **0** | 0 | **15** |

Every false positive in both corpora is `sim_error:domain`, and every one is the
fourth category the data forced. `δ` is: the sweep places a strict crossing **at
the truth value itself**, and restoring the input to that same value produces no
flip. Two computations of one point cannot disagree unless they are computing
different functions — so this says nothing about where the crossing is. It says
the sweep and the restore disagree about what the input *means*. **D72** records
the mechanism: the sweep spends a point of `[0, 0.4725]` as a flat one-sided
margin of `v × 100 %`, so truth (`v = 0`) is *no margin at all*, while the restore
puts the **measured domain** back, whose margin is +11.6 % at served concurrency
3.9 and −18 % at 76 and is nowhere zero.

**This is what the classification was for.** Read undifferentiated, E-B2 says the
closed-form rules cry wolf about one flip in six. Separated, **β is 0 in both
corpora** — no rule placed a crossing wrongly — which is what §2.4 predicted for
it, and the whole false-positive budget is one modelling mismatch in one item.

§2.4 expected α to be the common case. It never occurs: no crossing was ever
found inside an item's range but outside the interval the restore walks.

## Per kind — and the link that still decides nothing

E-A1, on the D70 margin:

| kind | n | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `link_bw` | 336 | 0 | 0 | 0 | – | – |
| `profile` | 224 | 66 | 0 | 0 | **1.000** | **1.000** |
| `sim_error` | 56 | 36 | 19 | 1 | 0.655 | 0.973 |

F2:

| kind | n | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `link_bw` | 336 | 0 | 0 | **0** | – | – |
| `profile` | 224 | 14 | 0 | **82** | 1.000 | **0.146** |
| `sim_error` | 56 | 22 | 15 | 15 | 0.595 | 0.595 |

**`link_bw` produces 336 true negatives on F2 and not one predicted flip.**
`eb1_f2.md` closed by predicting the opposite — that `link_bw` would generate α
false positives *by construction*, because on F2 the link is finally priced into
54 candidates including the winner, so a crossing should exist in its range even
though the decision never follows. It does not: the sweep finds no crossing at
all, so there is nothing to be a false positive. The prediction was wrong and the
weaker statement survives — on F2 the link is charged to the winner and the
decision is insensitive to it, at both ends of its range rather than only at
truth.

§2.3 required `link_bw` and `profile` to be non-zero for flip detection. `profile`
is; **`link_bw` is still zero**, now for a third distinct reason across three
experiments (E-A1: structurally inert, no P/D candidates · E-B1 on F2: priced in,
ΔR = 0 · E-B2 on F2: priced in, no crossing anywhere in range).

## Where the 97 false negatives come from: no incumbent, no signal

| | E-A1 | F2 |
| --- | ---: | ---: |
| degradation sets | 231 | 231 |
| sets with **nothing feasible while degraded** | **0** | **87 (37.7 %)** |
| — by k (1 / 2 / 3) | 0 / 0 / 0 | 2 / 17 / 68 |
| cases inside those sets | 0 | 240 |
| — of which `delta_regret` is `None` | – | **240 (all)** |
| — of which true positives | – | **0** |
| false negatives | 1 | 97 |
| — inside those sets | – | **82 (84.5 %)** |
| — where restoring produced a real winner | – | **82 (all of them)** |

`analyze` short-circuits when the degraded search finds nothing feasible — *"no
recommendation to flip: the search found nothing feasible"* — and returns
ΔR = `None` and `flip = False` for **every** input in the set. So the detector is
silent by construction on 240 of F2's 1,848 cases, and in 82 of them restoring
that single input brings a real winner back. Those are not near-misses: the
measurement would have restored feasibility outright, and the plan said nothing.

The remaining 15 false negatives are all `sim_error`, outside those sets.

**E-A1 has no such set at all**, which is the entire difference between recall
0.990 and 0.271. E-A1's corpus is aggregated-only with a 25 s TTFT budget, so no
single degradation rejects everything; F2 degrades an A40 profile by the Tier 0
multiplier at an 8 000 ms budget and the whole 223-candidate corpus goes
`slo_violated`.

This is the same gap `eb1_f2.md` found from the other side — there it made `ours`
lose to `random` at intermediate budgets, because with all-`None` ΔR the ranking
falls through to alphabetical order. Measured here, the statement is sharper:
**"which measurement would make something feasible again?" is not a question the
closed-form sensitivity asks.** It ranks by how far a measurement moves an
existing recommendation, and where there is no recommendation the value of
measuring is highest and the reported signal is exactly zero.

### One consequence for the approximate/exact split

§2.3 asks for the approximate rules' false-positive rate separately. On E-A1 that
split is clean. On F2 it is not, and the table has to say so:

| corpus | class | n | TP | FP | FN | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E-A1 | approximate | 224 | 66 | **0** | **0** | **1.000** | **1.000** |
| E-A1 | exact | 392 | 36 | 19 | 1 | 0.655 | 0.973 |
| F2 | approximate | **90** | 14 | **0** | **0** | **1.000** | **1.000** |
| F2 | exact | 526 | 22 | 15 | 97 | 0.595 | 0.185 |

F2's approximate row covers 90 cases, not the 224 `profile` cases in the corpus.
The other 134 are exactly the profile cases in no-incumbent sets: `analyze`
returns before any rule runs, so `approximation` keeps its default `False` and
the case is counted as *exact*. **F2's "exact" row therefore contains 134 cases
where no rule ran at all**, and its recall of 0.185 is not a statement about exact
rules. The approximate rows are sound in both corpora, and both are perfect.

The finding that survives from E-A1 holds: **the approximate rules make no errors
and the exact one makes all of them**, which is the opposite of the work order's
expectation.

## Criterion (2), and what it does not cover

§2.4 asks for a second scoring: not "did the winner move at the one value truth
turned out to have" but "is the flip anywhere in the item's range at all". That
is the detector's own claim, and criterion (1) is partly a statement about where
this fixture's truth happened to sit.

| corpus | n scored | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E-A1 | 560 | 66 | **0** | **0** | **1.000** | **1.000** |
| F2 | 560 | 14 | **0** | 82 | **1.000** | 0.146 |

**On E-A1 the detector is perfect under criterion (2)** — over 560 cases of
`link_bw` and `profile`, every predicted flip is real and no real one is missed.

**56 of the 616 cases per grid are not scored**, and they are every `sim_error`
case: D72. Criterion (2) restores an input to each point of its range, which only
means something when "restore to v" and "sweep to v" are the same function. For
`profile`, `link_bw`, `link_lat` and `power` they are — both go through `perturb`,
so the two implementations agreeing is a control. For `sim_error` they are not,
and the first implementation mapped its range onto the restore's two policy states
by a midpoint test, which made criterion (2) return criterion (1)'s answer and
report it as independent evidence. It agreed on all 1,848 cases for that reason.
`realisable` is now `None` there, and `realisable_by_kind` in the payload shows
the coverage per kind.

**So the consequence has to travel with the number: on both corpora every false
positive is `sim_error`, and `sim_error` is exactly what criterion (2) does not
cover.** The perfect precision above is a statement about the link and profile
rules. It is not a precision figure for the detector as a whole, and must not be
quoted as one.

Criterion (2) also does not rescue F2's recall. `profile`'s 82 false negatives are
identical under both criteria — the flip is not merely absent at truth, it is
absent everywhere in the item's range, because in those sets the degraded state
has no recommendation for any value of the input to flip.

## Grid density

m = 3, m = 5 and m = 9 agree case for case in both corpora, as they did in the
pre-D70 run. A flip is decided by whether the interval *contains* the crossing,
not by how finely it is sampled, so density affects only the ΔR magnitude. The
work order asks for this comparison expecting a density effect; across three runs
and 5,544 cases there is none, and §2.3 notes that this is favourable to the
invention description.

## What this means for C4 and C5

- **C4's Spearman is safe from one hazard and exposed to another.** F2 has more
  than one active item (`eb1_f2.md`: `sim_error` plus two RNGD profiles), so the
  ≥ 3 rule can be evaluated rather than assumed. But `sim_error` is not
  re-simulable by definition and `link_bw` never flips, so the LINK_BW identity
  control §2.5 wants as a second gate has nothing to fire on for this corpus.
- **The δ mechanism is upstream of C4's magnitude question.** E-B3 found the
  closed form 30.5 % low on ΔR and could not separate that from D40. D72 adds a
  second confound for any `sim_error` term: the sweep and the restore do not
  price the same input. C4 should report `sim_error` magnitudes separately or not
  at all.
- **The no-incumbent gap belongs in C5's limits, not in its effects.** It is not a
  harness fault and not a tuning choice: the verdicts are real (`slo_violated`,
  not `unmeasured`), and it costs the detector 82 of 97 false negatives on F2 and
  cost `ours` its intermediate-budget curve in C2. Two experiments now measure the
  same missing question from opposite directions.
