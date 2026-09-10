# E5 — the planner rejects D22's winner on its own

*`WORK_ORDER_rps_aware.md` rev 2 STEP 5. Run 2026-09-09 on the NPU node. Fixture
`pd-rngd-gpu-card.yaml`, 300 requests, seed 42, TTFT ≤ 64 s, **no manual margin**,
`--accuracy-domain` on. Artifacts: `outputs/e5/pd_slo_sweep.json`. Regression test:
`tests/test_e5_self_rejection.py`.*

**Result: yes, and with the number nobody supplied.** The committed winner is
rejected on its own predicted operating point, and the recommendation becomes
`agg[cuda:tp4]` at **2.595 tok/J** — the value the work order named in advance.

## What D22 needed a human for

The card fixture's winner `hp-00323` — two RNGD cards, tp1 each — passed a 50 ms
TPOT SLO at a predicted **48.41 ms**. It was infeasible: the simulator is 18 %
optimistic at the load that plan implies, so the real figure is ~57 ms. Finding
that took someone noticing the envelope measurement, interpolating it to the
winner's concurrency, and re-running the sweep with `--tpot-margin-percent 18`.
Nothing in the pipeline could see it, because the 18 % was a fact about the
simulator that the simulator did not carry.

## What the pipeline does now

| step | value | where it comes from |
| --- | ---: | --- |
| predicted p99 TPOT | 48.41 ms | the committed simulation |
| **served concurrency** | **74.75** | Little's law over that run's own CSV |
| accuracy-domain error there | **−17.69 %** | interpolated between the domain's 16.6 and 76 points |
| **robust p99 TPOT** | **56.97 ms** | 48.41 × 1.1769 |
| verdict | **SLO_VIOLATED** | 56.97 > 50 |

and the recommendation moves to `agg[cuda:tp4]`, n=4, **2.595 tok/J**, p99 TTFT
15.07 s, goodput 5.28 rps.

**Every number above is read, not supplied.** The operating point comes from the
run, the error from the measured domain, the verdict from the two together.

## Where it differs from the prediction, and why that is fine

The work order expected an operating point of ≈76 and a robust TPOT of 57.1 ms;
the measured operating point is **74.75** and the robust TPOT **56.97 ms**. The
76 was itself an interpolation — D22 read it off the c64/c128 envelope points as
"the load the card-fixture winner implies" — and 74.75 is what the winner's own
simulation actually ran at. The 0.13 ms difference in the robust figure follows
from that and changes no verdict.

**The domain was not adjusted to close the gap.** The work order forbids it in as
many words, and the regression test asserts the margin with a ±0.5 pp tolerance
for exactly this reason: tightening it by moving a calibration point would be
fitting the instrument to the answer.

## Held permanently, without a simulator

`tests/test_e5_self_rejection.py` replays the 199 committed simulation records
(`tests/data/e5_sim_records.json`) through a mock predictor and asserts the whole
chain: the winner's operating point, the margin the domain gives it, that it is
rejected **for the SLO** and not for some unrelated reason, that the margin's
source is `accuracy_domain` rather than a manual floor, and that the A40 plan at
2.595 tok/J takes over.

The operating points in that file were computed from the same runs' CSVs by
`experiments/scripts/backfill_operating_point.py`, because those cache entries
predate STEP 4.3's decision to store the field. That is a derivation from the
run's own artifacts, not a new measurement.

## One thing this does not show

That the A40 plan is *correct* — only that it is what remains after the RNGD plan
is priced honestly. The A40 accuracy domain has a single point at served
concurrency 170.56, and the winner here runs far below that, so its own margin is
held flat rather than interpolated. A second A40 measurement at a different load
would be the way to close that, and none exists.
