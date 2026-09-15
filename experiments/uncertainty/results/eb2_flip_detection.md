# E-B2 — how well the plan predicts that measuring an input changes the winner

**What this is.** A measurement plan does not only rank inputs; it says of each
one whether restoring it would **flip** the recommendation. E-B2 checks that
claim against what actually happens when the input is restored, over every
(degraded set × input) pair, at three grid densities.

> **Re-run 2026-09-15 on the D70 margin** (`WORK_ORDER_uq_stage_b_plus.md` STEP
> C3). The previous run used the margin `1 − e`, which D70 corrected to
> `m = −e/(1+e)` — the error's denominator is the measurement. That moves every
> margin and therefore every flip, so the pre-D70 numbers do not carry over; they
> are kept for comparison in `eb2_f2.md` and nowhere else. The truth is unchanged:
> the nine-point D32 RNGD-CARD domain, as in E-B1.

## The answer

*616 cases per grid, 1,848 total. `slo_penalty` 198,990.*

| grid `m` | n | TP | FP | FN | TN | precision | recall | FPR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 616 | 102 | 19 | 1 | 494 | 0.843 | **0.990** | 0.037 |
| 5 | 616 | 102 | 19 | 1 | 494 | 0.843 | **0.990** | 0.037 |
| 9 | 616 | 102 | 19 | 1 | 494 | 0.843 | **0.990** | 0.037 |

**Recall 0.990 — one missed flip in 103.** A plan that says "measuring this will
not change your mind" is right 99 % of the time, which is the direction that
matters: a missed flip is a measurement the operator skips and should not have.

**Precision 0.843 — 19 false alarms in 121.** The plan over-predicts flips, which
costs money rather than correctness. All 19 have one cause; see below.

**The grid does not matter.** m=3, m=5 and m=9 agree case for case. The work
order asks for this comparison expecting a density effect; there is none on this
fixture, because the flip is decided by whether the interval *contains* the
crossing, not by how finely it is sampled.

## Where the error lives, and it is not where the work order expected

The work order asks for the false-positive rate of the **approximate** rules to
be reported separately, on the assumption that approximation is where the error
would be. It is the opposite:

| rule class | n | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **approximate** | 224 | 66 | **0** | **0** | **1.000** | **1.000** |
| exact | 392 | 36 | 19 | 1 | 0.655 | 0.973 |

**Every false positive and the single false negative come from an EXACT rule.**
The approximate rules are perfect over 224 cases. Per input kind, at m=3:

| kind | n | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `link_bw` | 336 | 0 | 0 | 0 | – | – |
| `profile` | 224 | 66 | 0 | 0 | 1.000 | 1.000 |
| `sim_error` | 56 | 36 | 19 | 1 | 0.655 | 0.973 |

So the whole error budget sits in `sim_error` — the one input whose perturbation
moves the **margin** rather than the prediction — and `link_bw` never flips
anything on this fixture at all, which is the same inertness E-B1 sees.

That concentration is worth stating plainly: **E-B2's headline precision is a
statement about one input kind on one fixture**, not about the detector in
general. The 336 `link_bw` cases contribute 336 true negatives and inflate any
accuracy-style figure computed over the whole table.

## All 19 false positives are one defect (§2.4, and D72)

The α/β/γ classification STEP C3 added puts **0 in α, 0 in β and 0 in γ**. All 19
land in `delta`, a fourth category the data forced: the sweep places a strict
crossing *at the truth value itself*, and restoring to that same value produces no
flip. Two computations of one point cannot disagree unless they compute different
functions, and these do — the sweep spends a point of the item's `[0, 0.4725]`
range as a flat margin of `v × 100 %`, so truth (`v = 0`) means *no margin*, while
the restore puts the measured accuracy domain back, which is nowhere zero.
**D72** has the mechanism.

So β = 0: no closed-form rule placed a crossing in the wrong place. Read
undifferentiated, the 19 would have accused the rules of erring on one flip in
six.

## Criterion (2): the possible transition

§2.4's second scoring asks whether the flip exists anywhere in the range rather
than only at truth. Over the 560 cases it covers — `link_bw` and `profile` — the
detector is **perfect: precision 1.000, recall 1.000**.

The 56 `sim_error` cases are **not scored** (D72): the restore does not walk that
item's declared range, it swaps a policy, so the criterion is not defined for it.
Since every false positive above is `sim_error`, the perfect score is a statement
about the link and profile rules only and must not be quoted as the detector's
precision.

## Reproduce

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb2_flip_detection.py \
    --cache-dir outputs/uncertainty/ea1/cache --truth-sweep
```

Raw: `outputs/uncertainty/eb2/eb2_flip_detection.json` (per-case rows, so the
19 false positives can be read individually). The same detector on the F2 corpus,
where recall falls to 0.271, is `eb2_f2.md`.
