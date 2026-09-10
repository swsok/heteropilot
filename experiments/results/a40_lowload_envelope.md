# The A40 at low load — and the simulator is 18 % optimistic on TTFT there

*`docs/HANDOVER.md` §2.2. Measured 2026-09-10 on the **A40 node**
(`scripts/whichnode.sh`: a40, 8 × NVIDIA A40, driver 560.35.05, CUDA 12.8),
card 0 pinned by `CUDA_VISIBLE_DEVICES`, Llama-3.1-8B bf16 TP=1. Raw:
`outputs/a40_envelope_openloop/` and `outputs/a40_lowload_sim_error/`. Curve:
`profiles/envelopes/A40/meta-llama/Llama-3.1-8B/bf16/tp1.yaml`. Domain:
`profiles/calibration/a40.accuracy.yaml`.*

Three results, one sentence each:

1. **The A40 accuracy domain has three points instead of one**, so
   `widen_error_bars` has a slope to widen along and seven of E6's sixteen
   switchover cells stop reading `extrapolated`.
2. **The simulator's TTFT error at low load is ~18 %, not the 1.97 % the single
   point declared** — a factor of nine, and in the optimistic direction.
3. **Its TPOT error changes sign across the range** (+0.59 % at concurrency 4,
   −1.42 % at 170), so no single number describes this predictor and the
   one-sided margin correctly charges nothing at the low end.

## Why open-loop, and why that decides everything else

The domain's existing point came from `python -m bench run`, which replays an
arrival trace. The obvious route — port the closed-loop `measure_envelope.py` to
CUDA, which §2.2 specifies — would have produced a point measured under a
different load protocol, and putting two definitions of "served concurrency 8"
on one interpolation axis is the class of error D22 was. So the new points use
the same harness at lower offered rates.

The check that this actually put them on one axis: the driver, run over the
committed artifact that produced the 170.56 point, returns **170.5619** against
the recorded 170.56. Both halves of the domain are the same measurement of the
same quantity.

It also buys the TTFT column. Both sides replay the same arrival process, so
**D19 does not apply** and TTFT is comparable — on a closed-loop route it would
not be, and the largest finding here would have been invisible.

## The measurement

Two offered rates, two independent processes each, 300 requests per run,
**zero failures in 1200**.

| offered | **served** | tok/s | TPOT p50 | TPOT p99 | TTFT p50 | util % | **power W** | **tok/J** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 rps | **4.076** | 128.96 | 31.334 | 33.191 | 144.4 | 100.0 | 293.95 | **0.4387** |
| 0.5 rps | **11.272** | 314.41 | 35.494 | 38.244 | 154.9 | 99.9 | 294.98 | **1.0658** |

Repeat spread: served concurrency 0.008 % and 0.011 %, throughput 0.001 %, TPOT
p50 0.03 %, power 0.21 % and 0.66 %. Nothing approached the 5 % threshold, so no
third repeat was called for.

**Neither point is saturated**, and that is measured rather than assumed. A5(b)'s
pool rule is closed-loop and has no meaning here; what replaces it is the queue
delay slope, `d(scheduled_ts − queued_ts)/d(arrival)`. Both points read order
1e−9 s/s. The committed 170.56 run reads **+4.15 s/s** — it offered 10.3 rps and
completed 1.671, so its concurrency is a backlog depth rather than an occupancy.

> **The metric that was tried first and is wrong.** Completed-rps-against-offered
> looks like the natural open-loop stand-in for A5(b). It divides by a wall that
> includes the drain tail, so the first real run — 20 requests at 0.5 rps, every
> request served with zero waiting — reported 0.643 and was flagged saturated.
> Low-load points are short runs by construction, so that metric would have
> refused every point this work exists to take. It is still recorded in the
> artifacts; it no longer decides.

## Power is flat, and 3.3× worse than the RNGD card at matched concurrency

Power moves **293.95 → 294.98 W, a 0.35 % span, while throughput moves 2.44×**.
Same shape as the RNGD card (1.085× power across 9.5× throughput) and flatter
still. Two points cannot show a U, so the RNGD curve's shallow minimum is neither
confirmed nor contradicted here — that is an absence of evidence and is not
recorded as a finding.

At matched served concurrency ≈ 4 the two measured curves read:

| | served conc | tok/s | power W | **tok/J** |
| --- | ---: | ---: | ---: | ---: |
| RNGD card (2026-09-08, closed-loop) | 3.98 | 200.9 | 139.9 | **1.436** |
| A40 (2026-09-10, open-loop) | 4.076 | 129.0 | 293.9 | **0.4387** |

**A 3.27× raw energy-efficiency gap in RNGD's favour, and it must not be read as
D22 rehabilitated.** D22's retracted headline was "RNGD wins on energy by 1.67×",
an SLO-feasible claim that died when the measured 18 % TPOT margin made those
configurations infeasible — E5 rejects that winner outright and E6 reproduces the
rejection. This row is raw tokens per joule at one operating point with no SLO,
no margin and no feasibility test, measured under two different load protocols on
two different nodes. It is a weaker statement about a different quantity.
`docs/CLAIMS.md` §3 is unchanged by it.

Idle is **31.24 W**, one reading and not the mean of four: only the first
empty-card window is a cold card, the other three (39.84, 40.57, 41.81 W) begin
60 s after the previous run's process exited and are a cooling curve. It is also
not comparable to the RNGD envelope's 40.0 W, which is a *loaded* idle card —
`bench run` loads and unloads inside its own process, so no loaded-but-idle
window exists on this route and none is claimed.

## The simulator at the same operating points

Same offered rates, same 300-request trace, `python -m serving` against
`experiments/configs/clusters/a40-llama31-8b-tp1.json`:

| measured conc | **sim conc** | gap | TPOT meas → sim | **TPOT err** | TTFT meas → sim | **TTFT err** |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 4.076 | 4.043 | −0.8 % | 31.334 → 31.520 ms | **+0.59 %** | 144.4 → 117.5 ms | **−18.66 %** |
| 11.272 | 10.800 | −4.2 % | 35.494 → 35.379 ms | **−0.32 %** | 154.9 → 128.3 ms | **−17.15 %** |

**TPOT is nearly exact and changes sign.** At concurrency 4 the simulator is
*pessimistic*, so it earns no margin at all — `margin_from_error` is one-sided by
construction, and inflating there would make plans look worse than the hardware.

**TTFT is optimistic by ~18 %, where the one point at saturation said 1.97 %.**
Both readings are right about their own regime. At saturation TTFT is 40 s of
queueing, which the simulator models well; at low load it is ~150 ms of nearly
pure prefill, which it does not. Decomposed it is worse than 18 %: the simulator
charges **~16 ms of queueing** at these loads where the real engine measures
**8 microseconds**, so its prefill alone is optimistic by roughly 30 % and the
queueing it invents partly masks the gap.

### The request count is part of the method, not a detail

`lowload_sim_error.py` defaults to 20 requests. At that length the simulator's
served concurrency lands **−8.2 % and −31.7 %** from the measured points, and the
second would have been discarded as "not comparable" by the script's own ±20 %
guard. At 300 requests — matching the hardware — the gaps are **−0.8 % and
−4.2 %** and both points are comparable.

Nothing physical changed. A short run's drain tail is a large fraction of its
wall, and `served = Σlatency / wall` is depressed by exactly that fraction:
11.27 × 41.5/(41.5+23) ≈ 7.2, against the 7.70 the 20-request run reported. **No
point in the A40 domain rests on a 20-request run.** The same default was used
for the four RNGD low-load domain points and is worth revisiting there —
`docs/deviations.md` D32.

## What this changed downstream

`experiments/results/e6_rps_sweep.md`, re-run section: seven of sixteen
switchover cells go from `extrapolated` to `measured`, every winner and every
tok/J unchanged. The large TTFT correction changes no outcome because no
recommended plan comes within 25 % of its TTFT SLO — TPOT is the binding
constraint in this regime. It would matter in a tighter TTFT regime, which is
exactly the one `docs/HANDOVER.md` §2.5 still records as not quotable.

---

## The closed-loop curve, and how far the two protocols actually disagree

*Measured 2026-09-10/11 on the same card, same workload, same engine config
(`max_num_seqs` 128, `max_num_batched_tokens` 2048, bf16, seed 42) — so the only
thing that differs from the open-loop points above is **how load is offered**.
`experiments/scripts/measure_envelope.py --backend cuda`, STEP 3 protocol: 45 s
settle, 60 s idle, power and utilisation from the same 1 Hz samples, two
independent processes per point. Raw: `outputs/a40_envelope_closedloop/`.*

| requested | **served** | ratio | pool | tok/s | TPOT p50 | TPOT p99 | TTFT p50 | util % | **power W** | **tok/J** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **1.00** | 1.000 | 60 | 34.85 | 28.319 | 31.448 | 174.6 | 99.8 | 284.17 | **0.1226** |
| 2 | **1.99** | 0.996 | 120 | 64.97 | 30.226 | 32.600 | 178.9 | 99.9 | 284.87 | **0.2281** |
| 4 | **3.98** | 0.994 | 300 | 125.52 | 31.194 | 35.956 | 184.5 | 99.9 | 290.21 | **0.4325** |
| 8 | **7.88** | 0.985 | 300 | 234.55 | 33.091 | 37.225 | 192.5 | 99.8 | 292.34 | **0.8023** |
| 16 | **15.57** | 0.973 | 300 | 408.75 | 37.488 | 43.765 | 214.8 | 99.7 | 290.83 | **1.4055** |

Every point clears A5(b): `served/requested ≥ 0.973`, pool ≥ 30× the concurrency
everywhere, **no point is pool-bound and none carries a note**. Power repeat
spread ≤ 0.45 %. The c1 and c2 pools are 60 and 120 rather than 300 — still 60×
and 60× their concurrency, so far inside the rule, and 300 at c1 would have cost
~100 min per repeat for no statistical gain. The pool is recorded per point.

**These land on the RNGD card's served concurrencies almost exactly** — 1.00,
1.99, 3.98, 7.88, 15.57 against 1.00, 1.99, 3.98, 7.88, 15.59 — because the same
closed-loop protocol with the same pools produces the same occupancy. That makes
the cross-device comparison point-to-point rather than interpolated.

### The measured value of the protocol mismatch

`a40.accuracy.yaml` carries a warning that comparing an open-loop concurrency
axis against a closed-loop one needs care. This is that warning measured, on one
piece of hardware, with everything but the load protocol held fixed. Closed-loop
values interpolated to the open-loop points' served concurrency:

| served conc | TPOT open | TPOT closed | **diff** | TTFT p50 open | TTFT p50 closed | **diff** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.076 | 31.334 ms | 31.243 ms | **+0.29 %** | 144.4 ms | 185.2 ms | **−22.0 %** |
| 11.272 | 35.494 ms | 35.032 ms | **+1.32 %** | 154.8 ms | 200.6 ms | **−22.8 %** |

**TPOT is protocol-independent to within 1.3 %; TTFT is not, by about 22 %.**
The mechanism is arrival clustering: closed-loop admits a new request only as an
old one finishes, so arrivals bunch at completion instants and prefills queue
behind decode batches. At the *same* served concurrency that costs ~22 % of
TTFT. Decode steps, once a request is running, do not care how it arrived.

**This retroactively decides the route choice, by measurement rather than
argument.** Had the domain been built from closed-loop hardware, the simulator's
TTFT error at concurrency 4 would have read **(117.5 − 185.2)/185.2 = −36.6 %**
instead of −18.66 %. That would have been the wrong number: `python -m serving`
replays an arrival trace, so the condition it must be calibrated against is the
open-loop one. §2.2 prescribed the closed-loop route; taking it would have put a
36.6 % TTFT error into the domain and called it measured.

The practical rule, and it is now a measurement rather than a caution: **a TPOT
figure may cross the protocol boundary; a TTFT figure may not.** That is also why
`lowload_sim_error.py` compares TPOT only — on the RNGD side, where the bench is
closed-loop and the simulator open-loop, D19 forbids the TTFT comparison
outright. The A40's open-loop route is what made its TTFT column legal at all.

### A40 against the RNGD card, both closed-loop, matched concurrency

Same protocol, same workload, same statistic, no interpolation:

| served conc | A40 tok/s | RNGD tok/s | ratio | A40 W | RNGD W | A40 tok/J | RNGD tok/J | **ratio** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 34.85 | 63.1 | 1.81× | 284.2 | 151.1 | 0.1226 | 0.418 | **3.41×** |
| 1.99 | 64.97 | 110.1 | 1.69× | 284.9 | 140.6 | 0.2281 | 0.783 | **3.43×** |
| 3.98 | 125.52 | 200.9 | 1.60× | 290.2 | 139.9 | 0.4325 | 1.436 | **3.32×** |
| 7.88 | 234.55 | 356.8 | 1.52× | 292.3 | 143.2 | 0.8023 | 2.492 | **3.11×** |
| 15.57 | 408.75 | 598.2 | 1.46× | 290.8 | 151.8 | 1.4055 | 3.941 | **2.80×** |

The RNGD card is 1.46–1.81× the A40's throughput and 2.80–3.43× its tokens per
joule across the measured range, and **its advantage narrows as concurrency
rises** on both axes.

**This is not D22 and must not be quoted as if it were.** D22's retracted claim
was "RNGD beats the A40 on energy by 1.67×" as an *SLO-feasible* result; it died
because the measured 18 % TPOT margin made those configurations infeasible, and
E5/E6 reproduce that rejection. The table above is raw tokens per joule at fixed
occupancy with **no SLO, no feasibility test and no predictor margin**, and the
two halves were measured on two different nodes eight days apart. It says the
RNGD card is the more efficient device at these operating points; it says nothing
about which the planner should choose, which is the question D22 got wrong.
`docs/CLAIMS.md` §3 is unchanged.

**Why this curve is not loaded as an envelope.** `profiles/envelopes/A40/...`
holds the open-loop points, because those are the ones the accuracy domain and
the planner's arrival-replay predictor are calibrated on. Publishing a second
A40 envelope for the same model and TP degree would give the loader two curves
for one device differing by a protocol the schema has one boolean for, and the
22 % TTFT gap above is exactly the size of mistake that invites. The closed-loop
data lives in `outputs/a40_envelope_closedloop/` and in this table.

## E6b now has an A40 half

`e6b_measured_curve.py --hardware A40` → `outputs/e6/e6b_measured_a40.json`:

| served conc | sim conc | rps/card | **tok/J** | domain declares | observed | agrees |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 4.08 | 4.04 | 0.1976 | 0.439 | +0.59 % | +0.59 % | ✓ |
| 11.27 | 10.80 | 0.4818 | 1.066 | −0.32 % | −0.32 % | ✓ |

The agreement is a **self-consistency check on how the domain was recorded, not
independent confirmation** — these are the points the domain was fitted from, so
it reproduces them by construction. It catches a domain written on the wrong
axis or with a transposed sign, which is what it is for.

What changes is the sentence E6b's crossover has to be written in. It no longer
reads "RNGD **measured** against A40 **simulated**" — both sides are measured.
It must instead say **which protocol each side was measured under**: the RNGD
curve is closed-loop, the A40 curve open-loop, and the table above shows that
gap is worth ~22 % on TTFT and ~1 % on TPOT. Every row of both artifacts carries
`closed_loop` so the distinction cannot be lost downstream.
