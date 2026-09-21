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

**(ii) V3-R RAN on 2026-09-21 (S7.4) and this paragraph's conclusion survives it,
for a different reason.** It was written as "left as a selection step"; the step
was taken, on the NPU node, against an open-loop domain measured for the purpose
(S7.3). What it produced is **not** a §6 verdict count:

* the one case whose verdict is rankable is **circular at the point that decides
  it** — the open-loop domain's robust value returns the measurement it was
  fitted from, by construction (`v3r_verdict_accuracy.md` §3);
* leave-one-out gives that domain **±0.63 ms** on held-out interior points, which
  is predictive accuracy, not a verdict count, and the **two boundary points
  cannot be leave-one-out validated at all** — one of them is this case.

**So the specification still must not imply that the RNGD regime's verdicts have
been checked against hardware.** V0 and V1 establish the *error* and the
*margin* there; V3-R adds a measured **counterfactual** — what a domain
consulted outside its application conditions does — which belongs to §5.2 and
§8(3), not to §6.

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
| **§5.2** application conditions — parallelism and placement | domain fitted at TP=1 answers a TP=4 query; error −1.05 % at TP=1 against −44.6 % at TP=4 | `v3_verdict_accuracy.md` §6.4 | **established**, n=1, **counterfactual** |
| **§5.2** — the condition axis, now IMPLEMENTED | `AccuracyDomain` carries `hardware` / `parallelism {tp,pp,dp}` / `placement {islands, device_binding}` / `arrival_process` / `variant`; `check_conditions()` compares them per island assignment and a mismatch is its own rejection stage carrying `mismatch_fields` and `required_measurement` | domain-scoping S1, deviations **D110**; `planner/predictor/calibration.py`, `planner/optimizer/margin.py` | **built**, default `refuse`, `profiles/calibration/index.yaml` lists every registered domain |
| **§5.2** — what the condition axis costs, measured | on E-A1's 324 candidates rule (e) holds **312**, leaves **0 feasible** and **no recommendation**. Per field: `dp` 228, `islands` 132, `tp` 66, `arrival_process` 42 | `ea1_s4_condition_refuse.md`, deviations **D113** | **established** on one fixture |
| **§5.2** — the axis is WIDER than parallelism | `tp` explains only **66 of 312** holds; `dp` alone is the largest single group at 108. Replication and placement are refused more often than parallelism is | `ea1_s4_condition_refuse.md` §2 | **established** on one fixture. **The specification must not present the axis as "parallelism degree" alone** |
| §5.2 supporting — an unmeasured input explains it | `link_bw:pcie-a40a-02` measures **8.8 GB/s** effective against a `vendor_spec` 64.0; substituting it drops the TP=4 error from −43.4 % to −7.0 % on TPOT and −21.2 % to −0.7 % on concurrency | `v3_verdict_accuracy.md` Appendix A.1, A.4 | **established by measurement**, reproduced on two GPU groups |
| §5.2 — the parallelism condition, discriminated | same candidate at TP=2 inside an NVLink pair errs **−5.2 %** where TP=4 across the bridge errs **−43.4 %**, an 8.3× difference from removing one hop | `v3_verdict_accuracy.md` A.2 | **established**, n=1 per arm |
| §5.2 — device placement, third instance | one static link value cannot serve both candidates: 8.8 fixes TP=4 and breaks TP=2 (−5.2 % → +18.2 %), because the simulator cannot express which devices a TP group occupies | `v3_verdict_accuracy.md` A.4 | **established** |
| §5.2 — device placement, NUMA | binding the server to the island's NUMA node is worth **1.93×** throughput and 0.37× TTFT, with identical engine init, clocks and power | `v3_verdict_accuracy.md` A.3 | **established**, and it invalidates nothing already measured (A.3 ③) |
| **§8(3)** grounds for withholding a verdict, distinguished | two refusals, not one: `OUTSIDE_CALIBRATION_DOMAIN` (a domain applies, the operating point is past the end of its load axis) and `CALIBRATION_CONDITION_MISMATCH` (no domain was measured under this configuration at all). They ask for **different measurements** — a wider load range against a measurement at the candidate's own configuration — and the planner prints which, per candidate | deviations **D110**, **D113**; `ea1_s4_condition_refuse.md` §3 | **established**. E-A1's 12 single-card RNGD candidates move from the first to the second when `arrival_process` is stated: same hold, different measurement requested |
| §6 verdict accuracy, A40 cases | — | **not measurable**: the deploy backend blocks P2 and P3 | `v3_verdict_accuracy.md` §4① | **not established** |
| §6 cost of holding | all 30 held candidates are also rejected unmargined → holding costs nothing on this fixture | `v3_verdict_accuracy.md` §1 | **established** |
| §6 verdict accuracy, RNGD regime | — | **V3-R RAN (S7.4, 2026-09-21) and did NOT establish this.** The one rankable case is **circular at the point that decides it**: the open-loop domain's robust 42.894 against a measured 42.895 is construction, not accuracy — that point's `tpot_err_pct` was computed from that measurement. Leave-one-out gives the domain ±0.63 ms on held-out interior points, which is predictive accuracy and **not a verdict count**; the two boundary points, one of which is this case, cannot be leave-one-out validated at all | `v3r_verdict_accuracy.md` §2–§3 | **not established** — and now for a stated reason rather than for want of a measurement |
| **§5.2 / §8(3)** — consulting a domain outside its application conditions, measured | the counterfactual D113 asserted as policy. Consulted anyway, the **closed-loop** domain calls the candidate feasible (robust 37.984 ms) and the hardware violates (measured p99 TPOT **42.895**, 3 runs, spread 0.116): a **false pass**, under-predicting by **4.911 ms = 42× the run-to-run spread**. Non-circular — that domain was fitted on a closed-loop burst (D19) with no input from this or any open-loop measurement | `v3r_verdict_accuracy.md` §2 | **established**, n=1, **counterfactual** |
| **§8(3)** — a hold that was right, measured end to end | P2 at sim L 74.498: `sim feasible` (unmargined 46.691 ms < 50) / `planner refuse` (far outside the domain's [11.730, 37.965]) / **`measured saturated`** (served concurrency 114.198 against the simulator's 74.498, p99 TPOT 89.111 ms, drift slope over threshold). The refusal was right and the unmargined simulator was wrong | `v3r_verdict_accuracy.md` §4 | **established**, n=1. A true positive of the HOLD, which is a different claim from a margin being the right size |
| §6 reporting discipline — when a verdict may be ranked at all | a separation below **3× the measured run-to-run spread at that operating point** is recorded as **"indistinguishable"**, a verdict rather than a failure to reach one. It bites on the *wider* band: sim L 32.475's inversion band is 3.213 ms — four times wider on its face — but its spread is 0.776 ms and 3× consumes it, while L 37.965's 4.910 ms band against a 0.116 ms spread is rankable at 21.2× | `v3r_verdict_accuracy.md` §1, `v3r_slo_inversion.json` | **established as a rule**. The spread is measured and **varies by operating point**; one value across the range would have mis-ranked L 32.475 |
| **§2, §5.2** — V3 P1's ENDING | under rule (e) P1 is **held at every TPOT SLO from 38 to 50 ms, identically** — a condition mismatch is decided before any margin exists, so no threshold can move it. Rules (a)–(d) made 6 / 3 / 6 / 6 false passes on that grid; (e) makes **none**, and none of the correct rejections either | `v3_addendum_s4.md` §1 | **established**, n=1. The case closes as "unmeasured at its own configuration", not as a verdict |
| §2 — and the prediction was recoverable where the margin was not | with the link priced from measurement (8.8 GB/s) the same candidate predicts **60.09 ms** p99 TPOT against **64.616** measured (−7.01 %) and `L` within **0.73 %** | `v3_addendum_s4.md` §3, `v5_resim_measured_link.json` | **established**. P1 needed a bandwidth that was not a datasheet number, not a 44 % correction |
| fig. 2 | measured residual curve, sign change, zero-margin span, registered region | `experiments/figures/patent2_fig2_measured.png`, from `v4_figure2.py` | **established** — 8 points on the **p99 basis the verdict reads** (D101); c16.6 is absent because it has no per-request pair |

## 1.5 The patent-3 candidate embodiment, and the one step of it that is not established

The work order asks for a one-line embodiment: *"registry names the input → 0.114 h
of measurement → cause confirmed (NUMA placement, effective bandwidth ≈ 1/8 of
spec) → previous measurements re-verified (PR #104, < 0.5 %)."* Three of those
four steps are established. **The first is not, and this repository's own
experiments are what refute it.**

| step | state | evidence |
| --- | --- | --- |
| 1. the registry names the input | **NOT established** | S2 predicted `link_bw:pcie-a40a-02` would rank first once it had a range. Measured: `inert` — worth nothing (`s2_default_ranges.md`). S3 re-keyed the item and it became `needs_resimulation` — "the sweep cannot price this" (`s3_link_effective_bw.md`). At no point did the ranking name it. A human investigating V3 named it |
| 2. 0.114 h of measurement | **established** | `profiles/uncertainty/costs.yaml`, the `link_bw` row |
| 3. cause confirmed | **established by measurement** | effective TP=4 all-reduce **8.8 GB/s** against a `vendor_spec` **64.0** — a ratio of **0.1375** — reproduced on two independent GPU groups (8.81 / 8.78, 0.5 % apart); and NUMA binding worth **1.93×** throughput (`v3_verdict_accuracy.md` A.1, A.3) |
| 4. previous measurements re-verified | **established** | V2's ladder re-run bound, all five rates: agrees with the unbound stages to **0.42 %** on throughput and **0.37 %** on served concurrency (A.3) |

**So the embodiment may be written as the diagnostic loop it was — measure the
suspected input cheaply, confirm the cause, re-verify what came before — but not
as one the registry initiated.** Claiming step 1 would be claiming an effect this
repository measured and did not find.

**Why it failed is itself the interesting part, and it is fixed in the code
without yet being demonstrated end to end.** The registry priced a link only
through the prefill→decode KV transfer, and the E-A1 corpus has no P/D
candidate, so the item could not move anything and scored exactly zero. The
error V3 measured came from the same link carrying a **TP all-reduce inside one
island**, which reaches a prediction through the simulator's own `link_bw` and
has no closed form at all (D112). Such an item is now reported as
`needs_resimulation` rather than `inert`, and `--resimulate-top` prices it —
but **no run has yet been done in which the ranking names this link first**. A
demonstration would need a corpus where that resimulation is performed; it is
not in this work order.

**`1/8` is a rounded policy value, not the measurement.** The measured ratio is
**0.1375**; `profiles/uncertainty/grades.yaml` rounds it DOWN to 1/8 for the
`vendor_spec` default range (D111). The specification should quote 8.8 against
64.0, or 0.1375 — not 1/8 as if it were measured.

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
   whether a closed-loop TPOT transfers. Since S4 the closed-loop/open-loop
   difference is a **stated condition** on `rngd_card_edf.yaml`
   (`arrival_process: closed_loop`, D113), so the planner now refuses that
   transfer rather than performing it silently — but refusing is not the same as
   measuring, and V2 is still what would settle it.
4. ~~**V3-R** if an RNGD node becomes available.~~ **Done 2026-09-21 (S7.4).**
   It did not close §6 — see §0(ii) — so what remains for §6 in the RNGD regime
   is a verdict case whose domain point was **not** fitted from the measurement
   that judges it. That needs either a hold-out design (fit the domain without
   the operating point under test, which the boundary points cannot support) or
   a second measurement at an operating point the domain already covers from
   other data.
5. **A run in which the registry itself names the V3 link** — §1.5 step 1. The
   mechanism that made it impossible is fixed (D112); the demonstration has not
   been done.
6. **The condition refusal is whole-domain.** `arrival_process` disagreeing
   withdraws the TTFT *and* the TPOT error together, while D19's evidence is
   that the difference "lands entirely in TTFT". A closed-loop domain's TPOT
   error may therefore survive into an open-loop deployment, and the refusal is
   conservative rather than correct. Per-metric condition scoping is not built.
