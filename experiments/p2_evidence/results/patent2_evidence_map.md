# Patent 2 evidence map — which number comes from where

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V4. Started 2026-09-16, while V2
and V3's measurements are still pending, because two scope decisions had to be
recorded before the specification is drafted.*

> **Rule for the drafter: a number that is not in this table does not go into the
> specification.** That is the work order's instruction and it is the reason the
> file exists. Where a row says PENDING, the specification cannot yet make that
> claim.

## 0. Two scope facts that decide where §6's numbers come from

**(i) V3 is an A40-only case study.** There is no RNGD card on the node this work
runs on, so V3 deploys A40 candidates only. That is not a neutral restriction:
the A40 per-point TPOT margin is **0.37–1.63 %** across all 210 A40 candidates,
where RNGD-CARD's at the sweep's operating point is **~22 %**. The cases in which
a per-point margin visibly changes a verdict therefore live on RNGD — including
the disclosure's own §5.6 candidate B (predicted 48.41 ms, robust 59.04 ms).

**So the §6 numbers for the RNGD regime are built from V0 and V1**, which are
measurement-and-arithmetic results needing no hardware, and **not** from V3. V3
supplies A40 cases: one agreed baseline, one TTFT-decided catch, one rescue.

**(ii) V3-R is left as a selection step.** When an RNGD node is available, select
one RNGD candidate each of the P2 and P3 shapes and run the same comparison.
Until then the specification must not imply that the RNGD regime's verdicts have
been checked against hardware — V0 and V1 establish the *error* and the *margin*
there, not the *verdict*.

## 1. Section-by-section map

| disclosure § | what it needs | source | state |
| --- | --- | --- | --- |
| §3 prediction/measurement pair | predicted 48.41 ms p99 vs measured p99 **58.54 ms** at L = 76; `r` = +20.93 % | `v0_source_reconciliation.md` §2.4 | **established** (interpolated; see caveats) |
| §3, §5.6, §6, §8(3), fig. 4 | **57.1 ms must be removed** — it is `48.41 × 1.18`, not a measurement | `v0_source_reconciliation.md` §3, deviations D70 | **established** |
| §6 residual denominator | `e` (file) vs `r` (disclosure), `r = −e/(1+e)`; 12-point table | `v0_source_reconciliation.md` §4 | **established** |
| §6 aggregation basis | no committed point pairs p99 against p99; 5 are p50/p50, 3 p50/mean, 1 bucket mean | `v1_percentile_audit.md`, deviations D101 | **established** |
| §5.4 two-stage registration | applied to the measured points; committed residuals register **no** interval; p99 basis registers one | `v1_validation_region.md` §2 | **established** |
| §5.4 / §5.7 consistency | the endpoint placement cannot satisfy the ratio condition on a domain wider than a doubling | `v1_validation_region.md` §2.1 | **established** |
| §5.6 worked example | re-judged: B becomes `unmeasured`, A flips to SLO_VIOLATED | `v1_validation_region.md` §3 | **established** |
| §5.2/§5.3, claim 2 — lookup coordinate | `L_pred` vs `L_meas` divergence under open-loop load | `v2_openloop_concurrency.md` | **PENDING** (driver ready, PR #95) |
| §6 verdict accuracy, A40 cases | measured verdicts for P1/P2/P3 against the four rules | `v3_verdict_accuracy.md` §3 | **PENDING** (selection confirmed) |
| §6 cost of holding | all 30 held candidates are also rejected unmargined → holding costs nothing on this fixture | `v3_verdict_accuracy.md` §1 | **established** |
| §6 verdict accuracy, RNGD regime | — | **V3-R, not scheduled** | **not established** |
| fig. 2 | measured 9-point residual curve, sign change, zero-margin span, registered region | V4, from V0+V1 | **PENDING** |

## 2. Numbers cleared for use, with their caveats

| number | meaning | caveat that must travel with it |
| ---: | --- | --- |
| **48.4097 ms** | predicted p99 TPOT, card fixture winner `s256-t8192` | a prediction, never deployed |
| **58.54 ms** | measured p99 TPOT at L = 76 | **interpolated** between eff 59.185 and 107.192; measured **closed-loop** while the prediction is open-loop (D19 untested for TPOT — V2 measures this) |
| **+20.93 %** | `r` on the p99-vs-p99 pairing | the committed domain stores −18.0 on a p50-vs-mean pairing; a reader reproducing the planner gets 59.04, not 58.54 |
| **59.04 ms** | robust TPOT under the committed domain | the repo's number, on the committed basis; **not** a measurement |
| **0.37–1.63 %** | A40 per-point TPOT margin range | across all 210 A40-only candidates of the card fixture |
| **30 / 30** | held candidates also rejected unmargined | fixture-specific; see `v3_verdict_accuracy.md` §1 |

**Retired, and must not appear:** `57.1 ms` (a product, not a measurement — D70);
`56.97 ms` (the same product at L = 74.75); the D22 "RNGD wins on energy by
1.67×" headline (retracted, `CLAUDE.md`).

## 3. Open items before the specification is final

1. **V2** — claim 2's coordinate evidence, and the run-to-run p99 spread that
   V3's P3 result must be read against (its margin headroom is 0.623 ms).
2. **V3** — the A40 verdict table.
3. **The 58.54 ms caveats** above; V2's open-loop measurement is what settles
   whether a closed-loop TPOT transfers.
4. **V3-R** if an RNGD node becomes available.
