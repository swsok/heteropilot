# V1 — the §5.4 two-stage procedure, run on the measured points

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V1. Run 2026-09-16 on the **A40
node**; no hardware used. Implementation `experiments/p2_evidence/v1_region.py`,
machine-readable output `v1_region.json`, tests `tests/test_p2ev_region.py`.
Nothing in `planner/` or `profiles/` is touched. Read
`v1_percentile_audit.md` first — it establishes which residual basis each result
below is computed on.*

> **Headline.** On the residuals the domain actually stores, the procedure
> registers **no interval at all**, under every one of the three placements. On
> the p99 basis it registers **exactly one**, `[14.832, 25.181]`, and only under
> the leave-one-out placement. Re-judged against that region the disclosure's
> §5.6 worked example does not reproduce: candidate **B (L = 76), the case the
> whole example exists to show, comes out `unmeasured` rather than
> `SLO_VIOLATED`**, and candidate A (L = 20) flips the other way, from feasible
> to `SLO_VIOLATED`. The work order's risk row for this outcome says to record it
> as it is, and that is what §5 does.

---

## 1. Thresholds, and why each one

Every threshold is a choice the disclosure leaves open. These are the values and
the reasoning; `v1_region.json` carries them so a reader can re-run against
others.

| threshold | value | why |
| --- | ---: | --- |
| `RATIO_MAX` — adjacent concurrency ratio | **2.0** | The disclosure's own figure. Two points more than a doubling apart cannot support a linear claim about what lies between them. |
| `N_MIN` — measured samples at an endpoint | **100** | The only committed point below it is c16.6, a **five-sample** bucket fit, which cannot support a distributional claim. Every genuine comparison has n ≥ 128. Setting it anywhere in (5, 128] gives the same registrations. |
| `TOLERANCE` — allowed exceedance rate | **0.05** | The disclosure's suggested 5 %. **See §2.1 — at these sizes it means zero.** |
| `SLACK_PP` — added to the margin | **0.0** | The *strictest* choice, and therefore the one worth reporting: a larger margin raises the corrected latency and can only reduce exceedances, so an interval that registers at slack 0 registers at any larger slack. Intervals that fail stage 2 on rate report the minimum slack that would register them instead of being handed one. |
| `MARGIN_QUANTILE` | **100** | §5.4 says "upper quantile of the calibration residuals". An interval is defined by its two ends, so the upper quantile of a two-element set is its maximum. |
| `SATURATION_CONC` — operating-mode boundary | **44.0** | D32 measured the simulator settling at served 37.67 and 44.47 when offered the rates that put the hardware at 59.2 and 107.2. That is its own throughput ceiling, so a point above it is in a different regime from one below. |

### 1.1 What the tolerance means at this sample size

With `k` validation points inside an interval, a 5 % allowance permits
`floor(0.05 k)` failures, which is **zero for every k < 20**. No interval here
has 20 interior validation points — the largest has three. **So the second stage
is exactly "no validation point may be left uncovered."** This is not a
criticism of the 5 %; it is what the disclosure's own figure reduces to on a
nine-point domain, and the specification should say so rather than imply a rate
is being estimated.

---

## 2. Results

Full detail in `v1_region.json`. `m` is the interval's margin; a rejected
interval names the condition that rejected it.

### 2.1 On the committed residuals — nothing registers

| placement | calibration points | intervals | registered |
| --- | --- | ---: | ---: |
| **A** all calibration | all 9 | 8 | **0** |
| **B** ends calibrate | 1.02, 76.0 | 1 | **0** |
| **C** leave-one-out | 1.02, 4.311, 14.832, 16.6, 76.0 | 4 | **0** |

Why each interval failed:

| placement | interval | m | rejected by |
| --- | --- | ---: | --- |
| A | [1.02, 2.194] | 0.00 % | ratio 2.151 > 2 |
| A | [2.194, 4.311] … [14.832, 15.212] | 0.00 % | **stage 2 vacuous** — no interior validation point |
| A | [15.212, 16.6], [16.6, 25.181] | 3.20 / 3.39 % | sample count 5 < 100 |
| A | [25.181, 76.0] | 21.95 % | ratio 3.018 > 2 **and** mode change |
| B | [1.02, 76.0] | 21.95 % | ratio **74.51** > 2 **and** mode change |
| C | [1.02, 4.311] | 0.00 % | ratio 4.227 > 2 |
| C | [4.311, 14.832] | 0.00 % | ratio 3.441 > 2 |
| C | [14.832, 16.6] | 3.20 % | sample count 5 < 100 |
| C | [16.6, 76.0] | 21.95 % | ratio 4.578, sample count 5, mode change |

**Placement A registering nothing is a confirmation, not a failure.** The work
order asked for it to be checked rather than assumed: with every point
calibrating, the second stage has nothing to test, and an interval that
registered here would be registering on the strength of the very points that
sized its margin. `tests/test_p2ev_region.py::test_c_all_calibration_registers_nothing`
pins it.

**Placement B cannot register, by construction.** Two calibration points make
exactly one interval, and here it spans 1.02 → 76 — a ratio of 74.5 against a
limit of 2. §5.7's variant placement is incompatible with §5.4's own adjacency
condition on any domain wider than a doubling. **This is a structural finding
about the disclosure, not about this data**, and it should be stated in the
specification: the two sections cannot both be applied to a wide domain.

### 2.2 On the p99 basis — one interval registers

The c16.6 bucket fit has no per-request pair, so it drops out and eight points
remain.

| placement | registered | region |
| --- | ---: | --- |
| A | 0 | — |
| B | 0 | — |
| **C** | **1** | **[14.832, 25.181]**, m = **13.63 %** |

The one registered interval has calibration points at 14.832 (`e` = −6.90 %,
margin 7.41 %) and 25.181 (`e` = −12.00 %, margin 13.63 %), and one interior
validation point at 15.212: measured p99 **28.47 ms** against a corrected
prediction of 27.58 × 1.1363 = **31.34 ms**, so it is covered with 2.87 ms to
spare. Every other interval fails stage 1 on the ratio or on a mode change.

**Margin function.** Over `[14.832, 25.181]` the margin is the constant
**13.63 %** — §5.4's "upper quantile of the interval's calibration residuals",
which for a two-ended interval is the larger of the two. The planner's own
behaviour differs here: `AccuracyDomain.errors_at` interpolates the *error*
across the interval and then converts, giving a margin that rises from 7.41 % at
the left end to 13.63 % at the right. The constant is the more conservative of
the two everywhere inside, and it is what the disclosure specifies.

---

## 3. The §5.6 worked example, re-judged

All three candidates share the predicted TPOT of **48.4097 ms** (the card
fixture's winner) against a **50 ms** SLO; §5.6 varies only the operating point.
The control is what `plan --accuracy-domain` returns on the committed domain
today.

| candidate | L | committed domain (control) | measured region (p99, placement C) |
| --- | ---: | --- | --- |
| **A** | 20 | margin 3.28 % → **49.9952 ms** → *feasible* | inside [14.832, 25.181]: margin 13.63 % → **55.01 ms** → **SLO_VIOLATED** |
| **B** | 76 | margin 21.95 % → 59.04 ms → SLO_VIOLATED | **`unmeasured`** — no registered interval covers L = 76 |
| **C** | 110 | margin 38.60 % → 67.09 ms → SLO_VIOLATED (**extrapolated**, outside the domain) | **`unmeasured`** |

Three things follow, and two of them are uncomfortable.

**(a) Candidate A passes today by 4.8 microseconds.** 49.9952 against a 50 ms
SLO. The disclosure's illustrative "this one is fine" case is decided in the
fifth significant figure of an interpolated margin, and it does not survive a
margin sized on p99. A specification should not rest an example on a 0.005 ms
gap; either the example's numbers or its SLO should be moved so the verdict is
robust to the basis.

**(b) Candidate B — the case the example exists for — becomes `unmeasured`.**
Its operating point is covered by the domain (the file has a point at exactly
76) but not by any *registered interval*, because the interval below it,
[25.181, 76.0], fails on both the ratio (3.018) and the change of operating mode.
The distinction is the disclosure's own and it is correct: **unmeasured is not
infeasible**, and refusing to judge is the honest answer when the two-stage
procedure has not validated the ground under the candidate. But it means the
specification cannot simultaneously (i) present §5.4's registration procedure as
the basis for the verdict and (ii) quote §5.6's B as a `SLO_VIOLATED` result. One
of the two has to give.

**(c) The gap between the two columns is the size of what §5.4 adds.** The
committed domain answers at every operating point, including L = 110, which is
outside its measured range and reached by `widen_error_bars` extrapolation. The
registered-region procedure answers at one interval and declines elsewhere. That
is the intended behaviour of the invention, and this table is the first
measurement of what it costs: on this domain, **coverage falls from "every
candidate" to "candidates between served concurrency 14.8 and 25.2."**

---

## 4. What would have to change to register more

Not a proposal — an inventory, so the specification can say what a domain has to
look like for the procedure to bite.

| blocking condition | how many intervals it blocks (committed / p99) | what would fix it |
| --- | ---: | --- |
| ratio > 2 | 4 / 3 | measured points at most a doubling apart. The domain's own lowest interval, [1.02, 2.194], misses by **7.5 %** — a point anywhere in (2.04, 2.194] would open it. |
| sample count < 100 | 3 / 0 | re-measure c16.6, or drop it. It is the only thin point and it blocks two intervals in placement A and one in C. |
| operating-mode change | 2 / 2 | a measured point between 25.181 and 44 (the simulator's saturation), which would let the sub-saturation interval close without crossing the regime. |
| stage 2 vacuous | 4 / 4 | a placement that interleaves. This is placement A's defining property, not a defect. |

**Nothing here is blocked by the exceedance test.** Not one interval that
reached stage 2 failed it — the coverage was never the problem; admissibility
was. On this data the second stage has never been the binding constraint, which
is worth saying because it is the stage the disclosure spends most of its text
on.

---

## 5. Where this contradicts the hypothesis, and what it means for the disclosure

Work-order rule 4 requires recording a result that runs against the invention.
Three do:

1. **The committed domain registers no validation region at all.** Under every
   placement. If §5.4 is presented as the operative procedure, the disclosure's
   own measured domain currently supports **no** automatic verdict.
2. **§5.7's endpoint placement is incompatible with §5.4's ratio condition** on
   any domain spanning more than a doubling. This is a specification-internal
   contradiction, independent of data.
3. **The worked example does not reproduce.** B becomes `unmeasured`; A flips to
   `SLO_VIOLATED` where it is judged at all.

None of this touches the **invention's core claim**, which is that a margin sized
from the candidate's own operating point beats one global number — V0's audit and
the E-A1 verdict comparison still support that. What it touches is **§5.4's
registration procedure as a gate**: on nine points spread over a 74× concurrency
range, the gate closes almost everywhere. The specification's options are to
present §5.4 as an embodiment whose preconditions this domain does not yet meet
(and say what a qualifying domain looks like — §4), or to relax a condition and
state the relaxation. **Choosing between them is the user's call, not this
step's**, and the work order forbids tuning the thresholds to the result.

---

## 6. A40 points

The three A40 points are **not** run through the procedure. Two of them (4.043,
10.800) are open-loop and the third (170.56) is saturated and at a different
statistic again (p95 absolute error signed by the mean), and the file's own
`SEAM 2` records that the interval between 10.8 and 170.56 "crosses that change
of regime with nothing measured inside it". With three points there is one
interior interval, it spans a ratio of **15.8**, and it changes operating mode —
so it fails stage 1 on both counts before any placement is chosen, and there is
nothing to report that the seam note does not already say. **V2 adds four
open-loop points**; the A40 domain is worth running through this procedure after
that, not before.
