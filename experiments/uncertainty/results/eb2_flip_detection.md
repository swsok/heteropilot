# E-B2 — how well the plan predicts that measuring an input changes the winner

**What this is.** A measurement plan does not only rank inputs; it says of each
one whether restoring it would **flip** the recommendation. E-B2 checks that
claim against what actually happens when the input is restored, over every
(degraded set × input) pair, at three grid densities.

> **Run 2026-09-14 on the reconciled architecture** (PRs #78–#80, D33), against
> the same truth as E-B1 — the nine-point D32 RNGD-CARD domain, not the E-A2
> domain the 2026-09-11 run used. Numbers do not carry over between the two runs.

## The answer

*616 cases per grid, 1,848 total. `slo_penalty` 198,990.*

| grid `m` | n | TP | FP | FN | TN | precision | recall | FPR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 616 | 58 | 17 | 1 | 540 | 0.773 | **0.983** | 0.031 |
| 5 | 616 | 58 | 17 | 1 | 540 | 0.773 | **0.983** | 0.031 |
| 9 | 616 | 58 | 17 | 1 | 540 | 0.773 | **0.983** | 0.031 |

**Recall 0.983 — one missed flip in 59.** A plan that says "measuring this will
not change your mind" is right 98 % of the time, which is the direction that
matters: a missed flip is a measurement the operator skips and should not have.

**Precision 0.773 — 17 false alarms in 75.** The plan over-predicts flips, which
costs money rather than correctness.

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
| **approximate** | 224 | 20 | **0** | **0** | **1.000** | **1.000** |
| exact | 392 | 38 | 17 | 1 | 0.691 | 0.974 |

**Every false positive and the single false negative come from an EXACT rule.**
The approximate rules are perfect over 224 cases. Per input kind, at m=3:

| kind | n | TP | FP | FN | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `link_bw` | 336 | 0 | 0 | 0 | – | – |
| `profile` | 224 | 20 | 0 | 0 | 1.000 | 1.000 |
| `sim_error` | 56 | 38 | 17 | 1 | 0.691 | 0.974 |

So the whole error budget sits in `sim_error` — the one input whose perturbation
moves the **margin** rather than the prediction — and `link_bw` never flips
anything on this fixture at all, which is the same inertness E-B1 sees.

That concentration is worth stating plainly: **E-B2's headline precision is a
statement about one input kind on one fixture**, not about the detector in
general. The 336 `link_bw` cases contribute 336 true negatives and inflate any
accuracy-style figure computed over the whole table.

## Reproduce

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb2_flip_detection.py \
    --cache-dir outputs/uncertainty/ea1/cache
```

Raw: `outputs/uncertainty/eb2/eb2_flip_detection.json` (per-case rows, so the
17 false positives can be read individually).
