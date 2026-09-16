# V0 — where 57.1 ms came from, what aggregation the 18 % compares, and what the disclosure must say

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V0. Run 2026-09-16 on the **A40
node** (`scripts/whichnode.sh`: 8 x A40, 0 RNGD, 0 ATOM). No hardware was used:
every number below is read from committed artifacts or computed from them by
`experiments/p2_evidence/v0_convert.py`. Base `main` = `f6ee919`.*

> **Headline.** 57.1 ms is not a measurement, as D70 established — but the pair
> D70 put in its place is not measured like-for-like either. **52.7 ms is a
> MEAN and 43.2 ms is a p50**, and every other point in both accuracy-domain
> files pairs p50 against p50. Recomputed consistently the c76 error is
> **−20.05 %**, not −18.0, and D22's winner's robust TPOT is **60.55 ms**, not
> 59.04 and not 57.1. Compared on the percentile the feasibility check actually
> reads (p99 against p99) the error is **−17.31 %** and the robust value is
> **58.54 ms**, which is the measured p99 itself. The verdict — breach of the
> 50 ms TPOT SLO — is the same under all three, which is again why nobody
> noticed.

---

## 1. The two "조사 필요" items, answered

**(a) Which formula does the margin code use now?** The work order points at
`planner/optimizer/margin.py`; the formula does not live there. It is
`AccuracyDomain.margin_from_error` in **`planner/predictor/calibration.py:248`**,
and it uses the corrected `-e / (1 + e)`:

```python
fraction = err_pct / 100.0
if fraction <= -1.0:
    raise ValueError(...)          # a domain error implying a non-positive measurement
return -fraction / (1.0 + fraction) * 100.0
```

It is one-sided (`err_pct >= 0` returns 0.0), so a pessimistic simulator earns no
margin. `planner/optimizer/margin.py:318` is its **only** consumer — it takes the
worst margin per metric across a candidate's hardware — and defines no arithmetic
of its own. The multiplication happens in
`planner/optimizer/feasibility.py:56,67`: `robust = metric * (1 + margin/100)`.

**(b) What did the E-A1 / E6 regression do?** From D70's own body, which measured
rather than assumed it: **all sixteen E6 cells were re-scored from their recorded
operating points and zero verdicts change.** Margins rise (largest 42.12 → 72.78,
on a cell already rejected), and since a larger margin can only remove candidates
from the feasible set while every recommended plan survived, the ranking cannot
move. The default `plan` path applies no automatic margin, so golden output is
untouched. `tests/test_margin_policy.py:314` pins the c76 row at
`(76.0, 21.9512, 59.04)`, which reproduces exactly.

**(c) The instruction that was not carried out, and correctly so.**
`WORK_ORDER_uq_stage_b_plus.md` STEP C0 asked for the c76 point to be changed
from `−18.0` to `−15.2`, reading −18.0 as D22's `(57.1−48.41)/48.41` with the
sign flipped. D70 refused: −18.0 is `(43.2−52.7)/52.7`, the envelope's own pair,
correct under the file's declared convention. **That refusal stands** — §2 below
reopens the point on entirely different grounds (aggregation, not denominator).

---

## 2. What aggregation the 52.7 / 43.2 pair actually compares

### 2.1 The measured side is a MEAN

`experiments/results/rngd_concurrency_envelope.md` interpolates its **`TPOT avg`**
column between the c64 and c128 rows to the eff = 76 the card fixture's winner
implies. Recomputed from the raw per-request records
(`outputs/rngd_envelope/edf/real_c{64,128}.json`, n = 256 and 300, `eff` =
`Σ latency / wall`):

| measured aggregation | c64 @ eff 59.185 | c128 @ eff 107.192 | interpolated @ 76 |
| --- | ---: | ---: | ---: |
| **mean** | 44.5418 | 67.8762 | **52.7149** ← this is the 52.7 |
| p50 | 45.1225 | 70.3958 | 53.9748 |
| p95 | 47.6832 | 77.5055 | 58.1288 |
| p99 | 47.9586 | 78.1807 | 58.5442 |

52.7149 reproduces the committed 52.7 exactly. **The measured side is the
arithmetic mean of per-request TPOT.**

### 2.2 The simulator side is a p50

43.2 comes from the card fixture's committed winner. The run is identified
beyond doubt — `experiments/results/pd_slo_sweep_margin.md:41` names it
`s256-t8192`, and the cached metrics
(`outputs/uncertainty/ea1/cache/58d6c795….json`, candidate
`mix(furiosa-rngd-card-node_rngd0-tp1-dp1+furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192`)
reproduce three independent figures the results file quotes:

| figure in the results file | cached run |
| --- | --- |
| p99 TTFT 480 ms | 480.0921 |
| goodput 5.55 rps | 5.5519 |
| 1767 output tok/s per card | 190 920 tokens / 54.0 s makespan / 2 cards = **1767.78** |

That run's TPOT percentiles are **p50 = 43.1517**, p95 = 47.4974,
**p99 = 48.4097** at served concurrency **74.750**. So 43.2 is the **p50**, and
48.41 — the number the disclosure cites as "the prediction" — is the **p99 of the
same run**. The simulator emits no mean at all
(`planner/predictor/llmservingsim.py:600,668-670` produce p50/p95/p99 only), so a
mean-vs-mean comparison was never available.

### 2.3 Every other point compares p50 against p50

`experiments/scripts/lowload_sim_error.py` records `"compared_metric":
"tpot_p50"` and feeds `tpot_measured_ms = pt.tpot_p50` against
`tpot_sim_ms = m["tpot_p50_ms"]` (lines 198, 317-318). That script produced six
of the nine RNGD points and two of the three A40 points. So:

| basis | points |
| --- | --- |
| sim p50 vs measured p50 | 6 RNGD low-load + 2 A40 low-load = **8** |
| bucket mean error | RNGD c16.6 (EDF bundle fit) |
| p95 abs error signed by the mean diff | A40 c170.56 |
| **sim p50 vs measured MEAN** | **RNGD c76 — alone** |

The A40 file already **declares its seam** and quantifies it: *"that point's
magnitudes are p95 absolute errors signed by the mean; the new points are p50
errors… Under the p50 method the 170.56 point would read −1.32 % instead of
−1.42 %"* (`profiles/calibration/a40.accuracy.yaml`, METHODOLOGY SEAM). **The
RNGD c76 seam is declared nowhere** — its note says only "interpolated c64/c128,
rngd_concurrency_envelope.md. Not re-measured here."

### 2.4 What the point reads under each basis

Sim p50 43.1517 against the measured curve interpolated to 76, and the resulting
robust value for the winner's p99 of 48.4097:

| measured aggregation | ref @76 | file `e` % | disclosure `r` % | robust TPOT |
| --- | ---: | ---: | ---: | ---: |
| mean — **the committed basis** | 52.7149 | −18.14 | +22.16 | **59.14 ms** |
| p50 — **consistent with the other 8 points** | 53.9748 | −20.05 | +25.08 | **60.55 ms** |
| p95 | 58.1288 | −25.77 | +34.71 | 65.21 ms |
| p99 | 58.5442 | −26.29 | +35.67 | 65.68 ms |

And like-for-like on the metric the feasibility check reads — sim **p99** 48.4097
against measured **p99** 58.5442:

> `e` = **−17.31 %**, `r` = **+20.93 %**, and 48.41 × 1.2093 = **58.54 ms**,
> which reconstructs the measured p99 by construction.

**The verdict is unchanged under every row**: all exceed the 50 ms SLO. That is
the third time this family of errors has been invisible for the same reason.

### 2.5 What was NOT done here, and why

`profiles/calibration/rngd_card_edf.yaml` is **left at −18.0**. Changing it
invalidates the E-A1 corpus, `outputs/e6*/` and the pinned values in
`tests/test_margin_policy.py` / `tests/test_accuracy_domain.py`, and it is a
decision about what to re-run rather than a patch — the same reasoning D40 gives
for not re-keying the envelope cache. The work order's V0 scope is to *determine*
the basis, not to change the domain. **A follow-on work order should decide
between the p50 and p99 bases**; §5 records the recommendation.

---

## 3. Every occurrence of 57.1, and its current treatment

Matches of the D22 robust value only. Unrelated numerals containing the digits
(`peak_power_w 1557.1`, `docs/HANDOVER.md:264` operating point 157.1,
`docs/sim_cost_profile.md:21` 857.1, `eb1_table.md:31` 257.1,
`outputs/atom_profile/…:1071` io_baseline 57.1 µs, `tests/data/e5_sim_records.json`
1557.1) are excluded.

| # | site | what it says | treatment | action |
| --: | --- | --- | --- | --- |
| 1 | `docs/deviations.md:2288-2298` (D70) | the correction itself | **is the correction** | none |
| 2 | `docs/uncertainty_planner.md:27` | "48.41 × 1.18 = 57.1 ms *(pre-D70 arithmetic; corrected margin gives 59.04 — same verdict)*" | annotated by PR #87 | none |
| 3 | `docs/rps_aware_planning_design.md:198` | same annotation | annotated by PR #87 | none |
| 4 | `docs/PROJECT_REPORT.md:656` | same annotation | annotated by PR #87 | none |
| 5 | `profiles/calibration/rngd_card_edf.yaml:31-33` | "…57.1 is that product, not a measurement. The margin is −e/(1+e), so the corrected robust value is 59.04 (D70)" | annotated by PR #87 | none |
| 6 | `tests/test_accuracy_domain.py:8,147-159` | asserts the corrected 59.04 | annotated + pinned | none |
| 7 | `docs/deviations.md:2073` (**D22 body**) | "48.41 × 1.18 = 57.1 ms. The configuration was infeasible" — bare | **NOT annotated** | annotate → this PR |
| 8 | `planner/predictor/calibration.py:140` (`AccuracyDomain` docstring) | "48.41 x 1.18 = 57.1" — bare | **NOT annotated** (the sibling `margin_from_error` docstring is correct) | annotate → this PR |
| 9 | `docs/patent_future_ideas.md:50` | "48.41 ms 예측이 50 ms SLO를 통과했고 **실제로는 57.1 ms로 실행불가였다**" — asserts 57.1 as the *measured* value | **NOT annotated, and patent-facing** | rewrite → this PR |
| 10 | `experiments/results/e5_self_rejection.md:40` | "the work order expected … a robust TPOT of 57.1 ms; the measured operating point is 74.75 and the robust TPOT **56.97 ms**" | **NOT annotated**; 56.97 is the pre-D70 product | annotate, do not restate the recorded value → this PR |
| 11 | `WORK_ORDER_uncertainty_planner.md:317` | expectation text for the D22 reproduction test | historical instruction | leave (records what was asked) |
| 12 | `WORK_ORDER_rps_aware.md:310` | expectation text | historical instruction | leave |
| 13 | `WORK_ORDER_uq_stage_b_plus.md:108` | the C0 instruction D70 declined to carry out | historical instruction | leave (D70 records the refusal) |
| 14 | `outputs/uncertainty/ea1/ea1_margin_modes.json` (5 rejection reasons) | `p99_tpot_ms=57.1 vs target 50.0` | **run artifact** — what the pre-D70 code actually produced | **must not be edited**; A3 |

**Risk row from the work order §4** — "57.1이 다른 신고서·논문 초안에도 남아 있음":
patent 1's specification is in ETRI internal processing and is **not in this
repository**, so this table cannot cover it. The user must apply rows 1-14's
reasoning there and, if 57.1 appears, ask the attorney to correct it.

---

## 4. The 12-point convention table

Generated by `experiments/p2_evidence/v0_convert.py` (the work order forbids hand
arithmetic for these). `e` is the file convention `(sim − measured)/measured`;
`r` is the disclosure convention `(measured − sim)/sim = −e/(1+e)`; `margin` is
the one-sided value the planner applies (`max(0, r)`).

| hardware | conc (predicted) | source | aggregation basis | file `e` % | disclosure `r` % | margin % |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| RNGD-CARD | 1.02 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +2.25 | −2.20 | 0.00 |
| RNGD-CARD | 2.194 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +11.00 | −9.91 | 0.00 |
| RNGD-CARD | 4.311 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +10.79 | −9.74 | 0.00 |
| RNGD-CARD | 8.211 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +7.33 | −6.83 | 0.00 |
| RNGD-CARD | 14.832 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +3.26 | −3.16 | 0.00 |
| RNGD-CARD | 15.212 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +2.47 | −2.41 | 0.00 |
| RNGD-CARD | 16.6 | EDF bundle fit (PROJECT_REPORT §4.8.4) | bucket mean error | −3.10 | +3.20 | 3.20 |
| RNGD-CARD | 25.181 | 300-req arrival-matched comparison | sim p50 vs measured p50 | −3.28 | +3.39 | 3.39 |
| RNGD-CARD | 76 | envelope interpolation (c64/c128) | **sim p50 vs measured MEAN** | −18.00 | +21.95 | 21.95 |
| A40 | 4.043 | 300-req arrival-matched comparison | sim p50 vs measured p50 | +0.59 | −0.59 | 0.00 |
| A40 | 10.8 | 300-req arrival-matched comparison | sim p50 vs measured p50 | −0.32 | +0.32 | 0.32 |
| A40 | 170.56 | 300-req sharegpt validation | **p95 abs error, signed by mean diff** | −1.42 | +1.44 | 1.44 |

The two values the work order predicted are confirmed: the 76 point reads
`e = −18.0 %, r = +21.95 %` (the work order says 22.0) and the 16.6 point reads
`e = −3.1 %, r = +3.20 %` (the work order says 3.2).

**Sample counts.** The eight arrival-matched points are 300-request simulations
against the measured envelope curve; their measured side is an interpolation of
that curve, so `n` is a property of the curve, not of the point. c76's measured
side interpolates n = 256 (c64) and n = 300 (c128). c16.6 is a 5-sample bucket
fit (`sample_count: 5` in the same file). A40 c170.56 is the 300-request
sharegpt validation run. **No point below c76 has an independent high-load
sample**, which is why V1 must treat c76 as the only interior high-load evidence.

---

## 5. Correction list for disclosure v5

Delivered as a review copy to the user; **not applied to the repo**, which holds
no copy of the disclosure.

| § | current text | replace with |
| --- | --- | --- |
| §3 | "예측 48.41 / 실측 57.1 / 17.95 %" | "예측 48.41 ms(추천 후보의 p99 TPOT, 미배포). 동시성 76에서 실측 p99 **58.54 ms** 대 예측 p99 48.41 → `r` = **+20.9 %**. (57.1 ms는 측정값이 아니라 `48.41 × 1.18`이라는 계산값 — D70)" |
| §5.6 | 후보 B 마진 18 %, 보정 57.1 | 마진 **20.9 %**, 보정 **58.54 ms**. 전역 마진 비교도 20.9 % |
| §6 | 효과 첫 항목 + [수행한 측정과 판정 비교] 둘째 단락의 57.1 | 58.54 ms(실측 p99). "57.1"은 삭제 |
| §8(3) | 57.1 인용 | 58.54로 교체 |
| 도 4 | 예측/실측 막대 48.41 / 57.1 | 48.41(예측 p99) / 58.54(실측 p99) |
| 도 2 | 설명용 4점 | V1/V4에서 실측 9점으로 교체 (이 STEP 범위 밖) |
| E-A1 인용부 | "(b) 전역 마진 18 %" | 그대로 두되 **"종전 기록의 18 %"**로 표기 — 실험에서 실제로 쓴 규칙이므로 |

**Which basis the disclosure should use, and why p99.** The claim being supported
is a *feasibility* claim, and the feasibility check reads p99
(`planner/optimizer/feasibility.py:67`, `robust_tpot = p99_tpot × (1+m)`). A
margin fitted on p50 or on a mean and then applied to a p99 is comparing
distributions at different points; only the p99-against-p99 pairing makes
"보정 지연 ≥ 실측 지연" mean what the claim needs it to mean. It also lets the
disclosure quote a **measurement** (58.54 ms) instead of a product, which is what
D70's whole lesson is about.

**Three caveats the disclosure must carry with 58.54.**
1. It is an **interpolation** between eff 59.185 and eff 107.192, not a point
   measured at 76. Same status as the committed 52.7.
2. The envelope was measured **closed-loop** (a request pool), while the
   simulated side is an open-loop Poisson process at 9.9 rps. D19 establishes
   that closed-loop TTFT does not transfer; for TPOT the transfer is untested.
   **V2 measures exactly this**, so the disclosure should not be finalised before
   V2 reports.
3. The domain file still stores −18.0 (§2.5). Until a follow-on work order
   changes it, a reader reproducing the planner will get 59.14/59.04, not 58.54.
   The disclosure should cite the measurement, and the repo number is a separate
   record.

---

## 6. Raw material index

| what | path |
| --- | --- |
| measured envelope, per request | `outputs/rngd_envelope/edf/real_c{16,32,64,128}.json` |
| envelope write-up | `experiments/results/rngd_concurrency_envelope.md` |
| card winner's cached metrics | `outputs/uncertainty/ea1/cache/58d6c79533907b0576b730d7a3550a1e81f2bf82279f7fc01446ccd6939c18cf.json` |
| winner identified as `s256-t8192` | `experiments/results/pd_slo_sweep_margin.md:41` |
| sweep write-up (1767 / 43.2 / 76 row) | `experiments/results/pd_slo_sweep.md:190-194` |
| RNGD accuracy domain | `profiles/calibration/rngd_card_edf.yaml` |
| A40 accuracy domain (+ its declared seam) | `profiles/calibration/a40.accuracy.yaml` |
| low-load comparison script (`compared_metric: tpot_p50`) | `experiments/scripts/lowload_sim_error.py` |
| margin formula | `planner/predictor/calibration.py:248` |
| margin application | `planner/optimizer/feasibility.py:56,67` |
| this table generator | `experiments/p2_evidence/v0_convert.py` |
