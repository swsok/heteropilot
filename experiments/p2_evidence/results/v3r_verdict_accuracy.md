# V3-R — which domain was right, and which half of the answer is circular

*`WORK_ORDER_domain_scoping.md` STEP S7.4 rev 4. NPU node, accelerator set
`6fe246ed1abf`, card npu0 `0000:03:00.0`. **No new measurement was taken**: the
verdict is read from the three runs S7.3 already made at this operating point,
because a fourth run at the same rate and the same candidate would produce no
new information. Record: `v3r_slo_inversion.json`, `outputs/s74/reselect_*.json`,
`experiments/results/s73_npu_6fe246ed1abf/`.*

> **The headline is one finding and one non-finding, and they must not be
> merged.** Consulting the **closed-loop** domain for an open-loop candidate
> produces a **false pass** — measured, non-circular, and 42× the run-to-run
> spread. That the **open-loop** domain gets the same candidate right is
> **circular**: that domain's point at this operating point was fitted from this
> very measurement. The second statement is not evidence and is not claimed as
> any.

## 1. The case, and how the SLO was chosen

rev 4 expected sim L 37.965 to invert at the fixture's 50 ms — closed-loop
feasible, open-loop infeasible. **It does not.** The margin does triple, 7.51 %
→ 21.40 %, but a predicted 35.332 ms × 1.2140 is 42.89 ms and 42.89 < 50. Both
domains pass it at 50 ms.

The inversion is real; the threshold is elsewhere. A candidate inverts over
exactly `[T(1+m_closed), T(1+m_open))`, and the sweep (`v3r_slo_inversion.py`,
post-hoc arithmetic — **E-A1's 50 ms aggregate is not re-scored**) finds three
bands among the six in-domain points:

| sim L | m closed | m open | inversion band (SLO ms) | spread | rev 4 verdict |
| ---: | ---: | ---: | --- | ---: | --- |
| 27.002 | 3.96 % | 6.34 % | [34.365, 35.151) | **unknown** | single run — may not be ranked |
| 32.475 | 5.70 % | 15.07 % | [36.225, 39.438) | 0.776 | **INDISTINGUISHABLE** |
| **37.965** | **7.51 %** | **21.40 %** | **[37.984, 42.894)** | **0.116** | **rankable** |

**rev 4's 3× rule bites, and on the wider band.** L 32.475's band is 3.213 ms —
four times wider than it needs to be, on the face of it — but its spread is
0.776 ms, so 3× takes 2.328 ms off each side and the usable interval is empty.
It is recorded as *indistinguishable*, which is a verdict and not a failure to
produce one. Had one spread been used across the range, this point would have
been mis-ranked; S7.3's finding that the spread **varies by operating point and
is larger at lower load** is what prevents that.

**SLO 40.439 ms** is the midpoint of the rankable band: closed-loop passes by
2.455 ms, open-loop fails by 2.455 ms, each **21.2×** the measured spread.

## 2. The verdict

Three runs at 2.0 rps, 300 requests each: p99 TPOT **42.834 / 42.895 / 42.950
ms**, median **42.895**, spread **0.116 ms**.

| | robust value | verdict at SLO 40.439 | truth |
| --- | ---: | --- | --- |
| closed-loop domain (`rngd_card_edf.yaml`, 7.51 %) | 37.984 | **feasible** | **FALSE PASS** |
| open-loop domain (`…openloop.yaml`, 21.40 %) | 42.894 | infeasible | correct — **but see §3** |
| measured | 42.895 | — | violates by 2.456 ms |

**The false pass is the result.** The closed-loop domain under-predicts the
hardware by **4.911 ms at this operating point — 42× the run-to-run spread** —
and would have admitted a candidate that misses its SLO. It is non-circular:
that domain was fitted on a closed-loop burst against `bench_furiosa_endpoint.py`
(D19), with no input from this measurement or any open-loop one.

**This is D113's refusal vindicated by measurement.** S4 made an
arrival-process mismatch a refusal on the argument that a closed-loop domain
cannot answer for an arrival-trace replay. That was a policy argument. This is
the number: consulted anyway, it passes a candidate that violates.

## 3. What is circular, stated rather than buried

The open-loop domain's robust value is **42.894 ms** against a measured
**42.895 ms** — 0.001 ms apart. That is not accuracy, it is **construction**:
the domain's point at sim L 37.965 carries `tpot_err_pct: −17.63`, which was
computed as `(35.332 − 42.895) / 42.895`. Feeding the margin back through
`sim × (1 + m)` necessarily returns the measurement.

**So "the open-loop domain is right here" is a tautology and is not offered as
evidence.** The disclosure must not quote 42.894 vs 42.895 as a validation.

### What is non-circular about the open-loop domain

**Leave-one-out on the interior points.** Refit the domain without one point and
ask it to predict that point:

| held out (sim L) | measured err | LOO prediction | Δ | robust error vs measured |
| ---: | ---: | ---: | ---: | ---: |
| 16.545 | −0.76 % | −0.04 % | +0.72 pp | −0.216 ms |
| 21.668 | −2.25 % | −3.31 % | −1.06 pp | +0.355 ms |
| 27.002 | −5.96 % | −7.60 % | −1.64 pp | +0.624 ms |
| 32.475 | −13.10 % | −11.79 % | +1.31 pp | −0.586 ms |

The domain interpolates to within **±0.63 ms** on data it did not see. That is
5.4× the spread — not tight — but it is **eight times smaller than the 4.911 ms
by which the closed-loop domain misses**. The comparison the disclosure may make
is that one, not the tautological one.

**The two boundary points cannot be leave-one-out validated at all**: remove
11.730 or 37.965 and the target becomes an extrapolation, which `refuse`
declines. The domain's edges are therefore unvalidated by construction, and
sim L 37.965 — the case above — is one of them.

## 4. The other candidate rev 4 named

**P2 (sim L 74.498) behaves exactly as rev 4 predicted**, and needs no SLO
sweep: `sim feasible / planner refuse / measured saturated`.

* unmargined, the simulator passes it (predicted p99 TPOT 46.691 ms < 50);
* the open-loop domain **refuses** it — 74.498 is far outside [11.730, 37.965],
  policy `refuse`, so no margin is formed and no verdict is offered;
* the hardware at that offered rate (3.5 rps) is **saturated**: measured served
  concurrency 114.198 against the simulator's 74.498, p99 TPOT 89.111 ms, and
  the TTFT drift slope crosses the threshold.

So the refusal was right and the unmargined simulator was wrong. It is a **true
positive of the hold**, which is a different claim from a margin being the right
size, and it is the only one this candidate supports.

## 5. What this does not establish

* **Nothing about the open-loop domain at its own edges.** §3.
* **Nothing at 50 ms.** At the fixture's own SLO both domains pass this
  candidate; the inversion needs a threshold in [37.984, 42.894). The SLO is a
  post-hoc parameter here and is reported as one.
* **Nothing about TTFT.** Every verdict above is decided on TPOT; the measured
  p99 TTFT at this operating point is 420 ms against a 25 000 ms SLO.
* **Nothing about other hardware, parallelism or placement.** One card, `tp=1`,
  one island, NUMA-bound.
* **Nothing from a fourth run.** The verdict uses S7.3's three runs; no new
  measurement was taken, and the spread quoted is theirs.

## 6. Reproducing

```bash
# the two arms — same script, same cache, differing only in --accuracy-domain
PYTHONPATH=$PWD .venv/bin/python experiments/p2_evidence/v3r_select_candidates.py \
    --rps 0.75,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5 \
    --cache-dir outputs/p2_evidence/v3r/cache \
    --out outputs/s74/reselect_closedloop.json
PYTHONPATH=$PWD .venv/bin/python experiments/p2_evidence/v3r_select_candidates.py \
    --rps 0.75,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5 \
    --accuracy-domain profiles/calibration/openloop/rngd_card.accuracy.openloop.yaml \
    --cache-dir outputs/p2_evidence/v3r/cache \
    --out outputs/s74/reselect_openloop.json

# the inversion band and the 3x rule
PYTHONPATH=$PWD .venv/bin/python experiments/p2_evidence/v3r_slo_inversion.py \
    --closed outputs/s74/reselect_closedloop.json \
    --open   outputs/s74/reselect_openloop.json \
    --out    experiments/p2_evidence/results/v3r_slo_inversion.json
```

Warm cache, no simulation, no card.
