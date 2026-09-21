# V3 addendum — P1's ending under rule (e)

**What this is.** `WORK_ORDER_domain_scoping.md` STEP S4, the V3 half.
`v3_verdict_accuracy.md` ends with P1 deployed, measured and mis-margined, and
§6.4 reading that as a missing scoping axis rather than a failure of margining.
S1 (D110) built the axis. This is the sentence that closes the case.

**V3's own file is not edited.** Its numbers, its four rules and its tally stay
as published; this is the addendum §6.4 asked for.

Record: `outputs/uncertainty/s4/v3_slo_sweep_s4.json`, from
`experiments/p2_evidence/v3_slo_sweep.py --measured 66.012 --ea1
outputs/uncertainty/s4/ea1_s4_stated.json --out-json …`. Rules (a)–(d)
reproduce V3's table exactly; (e) is the new column.

---

## 1. The ending

P1 is `cuda-a40-node_a40a-tp4-dp1-s128-t2048`: one A40 island at **tp=4**. The
only A40 domain, `a40.accuracy.yaml`, is fitted at **tp=1**. Under rule (e) the
domain may not be consulted at all, and the planner says which field and both
values:

> unmeasured: A40: the accuracy domain was measured under different conditions
> (tp) — tp: domain 1, candidate 4. It may not be consulted here, so this
> candidate is unmeasured **AT ITS OWN CONFIGURATION**, not infeasible (S1, D110)

| TPOT SLO | (a) no margin | (b) global 18 % | (c) per-point | (d) + refuse | **(e) + condition refuse** | measured ≤ SLO? |
| ---: | --- | --- | --- | --- | --- | --- |
| 38 | FALSE PASS | correct rejection | FALSE PASS | FALSE PASS | **held** | no |
| 40 | FALSE PASS | correct rejection | FALSE PASS | FALSE PASS | **held** | no |
| 42 | FALSE PASS | correct rejection | FALSE PASS | FALSE PASS | **held** | no |
| 44 | FALSE PASS | FALSE PASS | FALSE PASS | FALSE PASS | **held** | no |
| 46 | FALSE PASS | FALSE PASS | FALSE PASS | FALSE PASS | **held** | no |
| 50 | FALSE PASS | FALSE PASS | FALSE PASS | FALSE PASS | **held** | no |

Measured p99 TPOT is **66.012 ms**, so the candidate really violates at every
threshold. V3's tally was (a) 6 false passes, (b) 3 correct rejections and 3
false passes, (c) and (d) 6 false passes each.

**Two P1 measurements exist and this table uses V3's published one.** 66.012 ms
is GPUs 0–3, **unbound** (A.3); the NUMA-bound three-repeat run on GPUs 4–7 is
**64.616 ms** and is what §3 compares against. Both exceed every threshold in
the grid, so no cell changes — but they are different runs and are not
interchangeable, which is the whole of A.3's finding.

**(e) makes zero false passes and zero correct rejections.** It is not a better
score on the same scale; it is off that scale. The column is identical at every
threshold because a condition mismatch is decided **before any margin exists** —
there is no robust value to compare against an SLO, so the SLO cannot move the
answer. The other four columns move with the grid and this one cannot, which is
the clearest statement of what changed: the rule stopped guessing.

## 2. Why "held" is the right answer and not an evasion

A held verdict costs something real — P1 is the only plan V3's backend could
deploy, and (e) leaves the run with no recommendation at all. Three things make
it the honest answer rather than a refusal to decide:

* **The margin it replaces was wrong by about 70×.** (c) and (d) charged
  **1.1347 %** where the measured error at P1's configuration was **−44.63 %**.
  They did not fail to be conservative; they answered a question they had no
  measurement for.
* **The same domain is right where it was fitted.** At tp=1 it is accurate to
  −1.05 %. Nothing is wrong with the domain — it was applied outside itself.
* **It names the measurement.** A held candidate carries `mismatch_fields` and
  `required_measurement`, and the run prints the latter verbatim:
  `measure at: {"binding": "unknown", "dp": 1, "hardware": "A40",
  "islands": 1, "pp": 1, "tp": 4}` — the disclosure's additional-measurement
  condition, generated rather than written. `binding: unknown` is what this
  fixture's nodes say, so the request is under-specified on the axis worth
  1.93× of throughput here; S6(i) should record its binding.

And (b)'s three correct rejections stay what V3 said they were: **18 % is
arbitrarily larger than 1.13 %, not better informed.** At SLO 44 and above it
fails exactly as the others do. Nothing here supports a global margin.

## 3. The measurement beside the prediction, since PR #104 supplies it

The work order asks for one line if the NUMA-bound P1 measurement exists. It
does — `v3_verdict_accuracy.md` A.2/A.3, GPUs 4–7, three repeats, `numactl
--cpunodebind=1 --membind=1` — and S3 (D112) supplies the other half by giving
the simulator the measured effective link bandwidth instead of the datasheet
one:

| | measured (3 repeats, bound) | sim @ 64.0 `vendor_spec` | **sim @ 8.8 measured** |
| --- | ---: | ---: | ---: |
| p99 TPOT | 64.616 ms | 36.55 (−43.44 %) | **60.09 (−7.01 %)** |
| served `L` | 162.26 | 127.87 (−21.19 %) | **161.07 (−0.73 %)** |

**The prediction was recoverable and the margin was not.** With the link priced
from measurement the simulator is within 7 % on TPOT and 0.73 % on concurrency
— so P1 never needed a 44 % correction, it needed a bandwidth that was not a
datasheet number. That is the S3 half of the same story: (e) correctly refuses
to margin P1, and D112 removes most of what the margin would have had to cover.

**What still is not established.** The residual −7.01 % is not itself margined
by anything: no A40 domain is fitted at tp=4, which is exactly what (e) says.
S6(i) is the measurement that would close it.

## 4. Mapping to the disclosure

| disclosure | this file |
| --- | --- |
| §2, §5.2 — application conditions of a calibration | §1: the tp mismatch, the field named, both values |
| §8(3) — distinguishing the grounds for withholding a verdict | §1: `calibration_condition_mismatch`, distinct from `outside_calibration_domain`, which is what E-A1's RNGD candidates get |
| §6 — verdict-flip numbers | **not from here.** The margin is ≈21 % only in the RNGD region; S7 on the RNGD node is the only source, and it has not run |

The last row is the standing limit and this addendum does not change it.

> **Addendum, 2026-09-21: S7 has now run, and the limit still stands.** S7.3
> measured an open-loop RNGD domain and S7.4 used it. The ≈21 % margin is real
> and was reproduced — 21.40 % at sim L 37.965 — but the case whose verdict it
> decides is **circular**: that domain point was fitted from the very
> measurement that judges it, so `sim × (1+m)` returns the measurement by
> construction. What S7.4 *did* establish is a **counterfactual** belonging to
> the rows above, not to §6: consulted outside its arrival-process condition,
> the closed-loop domain **false-passes** a candidate that violates, by 4.911 ms
> = 42× the run-to-run spread. See `v3r_verdict_accuracy.md` and
> `patent2_evidence_map.md` §0(ii).
