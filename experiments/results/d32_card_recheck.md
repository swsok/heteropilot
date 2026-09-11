# D32 on the card fixture — two points were discarded for the wrong reason

**Conclusion (i): D32's suspicion is confirmed.** `rngd_card_edf.yaml` discarded
two envelope points because the simulator settled "40 % below the hardware" and
attributed that to throughput error. At 300 requests — same rates, same trace,
only the run length different — the same two points land at **−3.1 %** and
**−2.4 %**. The gap was the drain-tail artifact.

**Conclusion (ii): the artifact is not the whole story, and now the split is
measured.** At the rates matching measured c59.2 and c107.2 the simulator still
sits **−36.4 %** and **−58.5 %** below the hardware at 300 requests. That
residual is the card model's real throughput ceiling — it saturates near served
44 where the card reaches 107.2 — and is exactly what a 20-request run could not
separate from the tail.

**Conclusion (iii): nothing in E6 moves.** All eight card-fixture rows were
re-ranked under the rebuilt domain. **Every winner and every tok/J is identical
to full precision**; only the margins change, 3.05 → 2.88 % and 12.52 → 11.68 %.

*Measured 2026-09-11 on the NPU node, simulation only — the RNGD envelope is
committed, so this needed no device. Artifacts: `outputs/card_lowload_300/`,
`outputs/card_highload_300/`, `outputs/card_highload_20/`,
`outputs/e6_card_d32/`. Domain: `profiles/calibration/rngd_card_edf.yaml`.*

## The experiment

`lowload_sim_error.py` with its committed defaults — the card fixture, the
RNGD-CARD envelope, the sharegpt workload, `--match offered` — and **only**
`--num-reqs` changed from the script's default of 20 to 300, which is what the
hardware side used. That is D32's experiment exactly.

| measured | C_sim @20 | gap @20 | err @20 | ok | C_sim @300 | gap @300 | err @300 | ok |
| ---: | ---: | ---: | ---: | :-: | ---: | ---: | ---: | :-: |
| 1.00 | 1.09 | +9.5 % | +3.88 % | ✓ | 1.020 | +2.0 % | +2.25 % | ✓ |
| 1.99 | 2.19 | +10.1 % | +11.71 % | ✓ | 2.194 | +10.3 % | +11.00 % | ✓ |
| 3.98 | 3.92 | −1.4 % | +11.56 % | ✓ | 4.311 | +8.3 % | +10.79 % | ✓ |
| 7.88 | 6.41 | −18.7 % | +6.37 % | ✓ | 8.211 | +4.2 % | +7.33 % | ✓ |
| **15.3** | 9.14 | **−40.2 %** | −1.22 % | ✗ | **14.832** | **−3.1 %** | **+3.26 %** | **✓** |
| **15.59** | 9.26 | **−40.6 %** | −1.97 % | ✗ | **15.212** | **−2.4 %** | **+2.47 %** | **✓** |

**The sign changes, and that is the part with consequences.** At 20 requests the
two readmitted points read −1.22 % and −1.97 %, i.e. optimistic; at 300 they read
+3.26 % and +2.47 %, pessimistic. Only a negative error earns a margin, so
readmitting them **removes** a margin around served concurrency 15 rather than
adding one. A domain built on the 20-request numbers would have charged a plan
there for an error whose sign was an artifact of run length.

## The high end, where the artifact stops explaining everything

The three envelope points above 20 were never in the domain. Run both ways:

| measured | C_sim @20 | C_sim @300 | ratio | gap @300 | verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 29.3 | 11.45 | 25.18 | 2.20× | **−14.1 %** | passes the ±20 % guard → **new point, −3.28 %** |
| 59.2 | 13.01 | 37.67 | 2.90× | −36.4 % | refused |
| 107.2 | 13.59 | 44.47 | 3.27× | −58.5 % | refused |

At 29.3 the artifact accounted for essentially all of the gap. At 59.2 and 107.2
it did not: a 2.9–3.3× correction still leaves −36 % and −58 %. **That is the
measurement D32 says is missing** — the split between the tail artifact and a
real throughput error — and it is: artifact-only at 29.3, artifact *plus* a real
ceiling above it. The card simulator saturates at about served 44 and 35 ms TPOT
while the hardware goes to 107.2 and 67.88 ms.

The two refused points are refused by the script's own guard, not by judgement.
Their TPOT belongs to a different operating point than the one they are compared
against.

## The rebuilt domain

Nine points, 1.020 → 76.0:

```
1.020  +2.25     2.194 +11.00     4.311 +10.79     8.211  +7.33
14.832 +3.26    15.212  +2.47    16.6   −3.10 (EDF anchor, not re-measured)
25.181 −3.28    76.0   −18.00 (c64/c128 interpolation, not re-measured)
```

The error falls monotonically from +11 % at c2.2 to −18 % at c76, crossing zero
between 15.212 and 16.6, so the one-sided margin charges nothing below about 16
and grows above it.

**25.181 is the first measured point between the two anchors.** The interval
16.6 → 76.0 previously contained nothing, and 76.0 is itself an interpolation
from the c64/c128 envelope rather than a direct measurement. The new point reads
−3.28 % against the EDF anchor's −3.1 % at a nearby concurrency and by a
completely different method (open-loop replay against the envelope, versus a
bundle fit). That is a cross-check the domain did not have.

## What it does to E6 — nothing, and the check was necessary

The margins at the two operating points the card fixture's winners run at:

| operating point | before | after |
| --- | ---: | ---: |
| `RNGD-CARD@16.545` (1 rps winner) | 3.05 % | **2.88 %** |
| `RNGD-CARD@54.164` (3.3 rps decode leg) | 12.52 % | **11.68 %** |

**The margins went DOWN, so the cheap argument does not apply here.** When a
margin rises it can only remove candidates from the feasible set, and if the
recommended plan survives, the ranking cannot move — that is how
`rngd_perpe_accuracy_domain.md` avoided re-ranking four cells. A margin that
*falls* readmits candidates, and any readmitted candidate might outrank the
winner. Checking the cached corpus directly: **three candidates flip from
infeasible to feasible**, all `mix(a40 … + rngd-card …)` shapes with p99 TPOT
between 46.7 and 48.0 ms against the 50 ms SLO.

So all eight rows were re-ranked rather than argued about:

| rps | TTFT | before | after |
| ---: | ---: | --- | --- |
| 1 | 64 s / 8 s | `agg[furiosa:tp1]` 1.0739 tok/J | **identical**, margin 3.05 → 2.88 % |
| 3.3 | 64 s / 8 s | `P[cuda:tp1] D[furiosa:tp1]` 1.8329 | **identical**, margin 12.52 → 11.68 % |
| 10 | 64 s | `agg[cuda:tp4]` 2.5954 | **identical** |
| 10 | 8 s | `P[cuda:tp2] D[cuda:tp4]` 2.4000 | **identical** |
| 20 | 64 s | `agg[cuda:tp4]` 2.7618 | **identical** |
| 20 | 8 s | `P[cuda:tp1] D[cuda:tp2]` 2.3204 | **identical** |

**Zero of eight rows changed.** The three readmitted candidates draw 1023–2305 W
against winners that draw far less, and the objective is `minimize_energy`, so
they never came close. The result is that the card half of E6 now stands on a
domain whose low end is measured at the right run length — with the same answers.

## A refinement to the "replay costs seconds" correction

`rngd_perpe_accuracy_domain.md` corrected the cache README's claim to *minutes to
hours*, on the evidence that the cache stores only successful simulations. This
re-rank sharpens it in both directions:

| row | cache hits / evaluated | wall |
| --- | ---: | ---: |
| 3.3 rps, both TTFT points | **318 / 318** | **0.0 min** |
| 10 and 20 rps, all four | 250 / 318 | 0.3–0.4 min |
| 1 rps, 64 s | 264 / 318 | 44.3 min |
| 1 rps, 8 s | 283 / 318 | 30.0 min |

When the corpus covers every candidate the replay really is **seconds**. The cost
is entirely the candidates that die in the simulator and are therefore never
cached — 36 of them at 1 rps cost 44 minutes, because a failure at a low arrival
rate is a long simulated span before it fails. Whole re-rank: **1.26 h**.
