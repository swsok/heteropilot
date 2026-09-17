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

**(iii) The parallelism axis goes in a DEPENDENT claim, not the independent one**
(decision 2026-09-17). V3 measured a case where a domain fitted at TP=1 answered
a TP=4 query with a 1.13 % margin where ~80 % was needed
(`v3_verdict_accuracy.md` §6.4). The response the specification takes is to name
**the parallelism configuration and the device placement among the conditions
under which verification information may be applied** — as a dependent claim and
as a detailed-description embodiment. It does **not** enter the independent
claim: the invention's core is per-operating-point margining, and a scoping
condition is a refinement of it, not a precondition for it.

**(iv) V3's P1 result maps to §2 and §5.2, not to §6.** It is evidence for the
*problem* the invention addresses (a prediction that is confidently wrong, §2)
and for the *conditions* under which verification information transfers (§5.2).
It is **not** an effect of the invention and must not be cited in §6, because it
is a case where the invention did not help. **n = 1, and it is counterfactual**:
what it shows is what happens when the domain does not cover the candidate, not
what the margin achieves when it does.

**(v) V3's P1 measurement is a TRANSIENT time-average, so it is a qualitative
case only** (decision 2026-09-17). Neither side reaches steady state: both are
triangles, the hardware peaking at 300 concurrent — every request in the trace
resident at once — and draining (`v3_verdict_accuracy.md` §6.3.1). `L` = 163
is the mean of that triangle, not a concurrency the server settled at.

**The specification's worked embodiment must therefore be measured on a
steady-state interval**, not on this run: an arrival window long enough for the
queue to stabilise, with the interval's flatness reported. V3's numbers support
§2 and §5.2 as *cases*; they are not the measurement an embodiment quotes.

**(vi) Figure 2's registered region carries its own non-registration table.** The
figure marks `[14.832, 25.181]` as the one interval V1's §5.4 procedure
registers; every other interval is unregistered **for a stated reason**, and the
figure must be read against `v1_validation_region.md` §2.1, which names the
condition that rejected each one (ratio > 2, sample count < 100, a change of
operating mode, or a vacuous stage 2). A reader seeing one shaded band must be
able to ask why the rest is unshaded and get an answer per interval, not in
aggregate.

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
| **§2** problem definition — a prediction that is confidently wrong | P1: predicted p99 TPOT 36.5 ms, measured **66.0 ms** (−44.6 %); predicted L 127.9, measured 163.4 | `v3_verdict_accuracy.md` §6.3 | **established**, n=1 |
| **§5.2** application conditions — parallelism and placement | domain fitted at TP=1 answers a TP=4 query; `AccuracyDomain` has no parallelism axis; error −1.05 % at TP=1 against −44.6 % at TP=4 | `v3_verdict_accuracy.md` §6.4 | **established**, n=1, **counterfactual** |
| §5.2 supporting — an unmeasured input explains it | `link_bw:pcie-a40a-02` measures **8.8 GB/s** effective against a `vendor_spec` 64.0; substituting it drops the TP=4 error from −43.4 % to −7.0 % on TPOT and −21.2 % to −0.7 % on concurrency | `v3_verdict_accuracy.md` Appendix A.1, A.4 | **established by measurement**, reproduced on two GPU groups |
| §5.2 — the parallelism condition, discriminated | same candidate at TP=2 inside an NVLink pair errs **−5.2 %** where TP=4 across the bridge errs **−43.4 %**, an 8.3× difference from removing one hop | `v3_verdict_accuracy.md` A.2 | **established**, n=1 per arm |
| §5.2 — device placement, third instance | one static link value cannot serve both candidates: 8.8 fixes TP=4 and breaks TP=2 (−5.2 % → +18.2 %), because the simulator cannot express which devices a TP group occupies | `v3_verdict_accuracy.md` A.4 | **established** |
| §5.2 — device placement, NUMA | binding the server to the island's NUMA node is worth **1.93×** throughput and 0.37× TTFT, with identical engine init, clocks and power | `v3_verdict_accuracy.md` A.3 | **established**, and it invalidates nothing already measured (A.3 ③) |
| §6 verdict accuracy, A40 cases | — | **not measurable**: the deploy backend blocks P2 and P3 | `v3_verdict_accuracy.md` §4① | **not established** |
| §6 cost of holding | all 30 held candidates are also rejected unmargined → holding costs nothing on this fixture | `v3_verdict_accuracy.md` §1 | **established** |
| §6 verdict accuracy, RNGD regime | — | **V3-R, not scheduled** | **not established** |
| fig. 2 | measured residual curve, sign change, zero-margin span, registered region | `experiments/figures/patent2_fig2_measured.png`, from `v4_figure2.py` | **established** — 8 points on the **p99 basis the verdict reads** (D101); c16.6 is absent because it has no per-request pair |

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
