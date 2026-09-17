# V2 — open loop: does the predicted served concurrency diverge from the measured one?

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V2. Measured 2026-09-16 on the
**A40 node** (`scripts/whichnode.sh`: 8 × A40), single card, TP=1, real vLLM
0.19.0 deployed by `planner deploy`. Raw client records
`outputs/p2_evidence/v2/rps*_r*.json` (committed — measurements are not
regenerable); simulated side `v2_sim_side.json`; the joined comparison
`v2_comparison.json`.*

> **Headline.** Claim 2's direction is confirmed and its magnitude is not. The
> simulator is optimistic, so by Little's law `L_pred < L_meas` at **every** one
> of the five stages — but by **0.64 % to 2.88 %**, under the 5 % the work order
> set as the threshold for the lookup coordinate to matter. Looking the margin up
> at `L_pred` rather than `L_meas` moves it by **0.0008–0.0078 pp** and changes
> **no** verdict. The work order's risk row for this outcome fired, and §4 says
> what the specification should do about it.
>
> Two things came out that V2 was not looking for. The **L > 170 stage landed at
> L = 170.466 against the committed domain's 170.56 anchor** — 0.05 % apart, by a
> completely different harness. And the committed A40 domain, confirmed at both
> anchors, is **wrong in the interval between them** by up to 5.5×, in the
> under-conservative direction — exactly where its own SEAM 2 note said it was
> carrying an untested assumption.

---

## 1. What ran

`deploy` launched `vllm serve NousResearch/Meta-Llama-3.1-8B` on one A40, TP=1,
`max_num_seqs 128`, `max_num_batched_tokens 2048`, prefix caching off. Load came
from `replay_to_endpoint.py --open-loop --ignore-eos`, 300 requests per stage,
the trace rescaled by a constant factor per rate. The simulator ran the same
candidate on **the same rescaled arrival times**, not merely the same mean rate.

**The rates are not the work order's 2 / 5 / 10 / 15.** A capacity probe measured
this deployment at 682.8 output tok/s at concurrency 36 and the workload's mean
output is 652.5 tokens, so every rate the work order named looked saturating. The
ladder keeps the work order's intent — below saturation, climbing — at the
measured capacity. (At the top stage the server actually reached **1 086 tok/s**,
so true saturation is ≈1.66 rps and the ladder was more conservative than it
needed to be. The range it covers, L = 11.5 to 170.5, is what matters.)

**A trap worth naming.** `planner deploy` writes its handle and server log to
`outputs/deployments/<plan_id>/`, and reusing a plan id **overwrites whatever is
there**. This run reused `a40-live-llama-tp1` and silently replaced the committed
record of the 2026-08-19 first live deploy loop (3 302 changed lines of
`vllm.log`). It was caught at `git status` and restored from HEAD; V2's own
handle and server log are kept as
`outputs/p2_evidence/v2/{deployment_handle,vllm_server}.log`. **A future run
should use a fresh plan id**, or `outputs/deployments/` should stop being
tracked.

**Two fixes were needed before the first stage could be trusted**, and both
changed what was being measured:

* **`--ignore-eos`.** The simulator generates exactly the trace's `output_toks`;
  a real engine stops at EOS. The 2026-08-19 live run delivered 347.5 output
  tokens per request against the trace's 652.5 — the two sides were running
  different workloads. With `--ignore-eos` every stage delivered **195 753**
  output tokens, the trace's exact total.
* **Chunk counting.** Counting only chunks with non-empty text lost 1–4 tokens
  per request (27 035 of 27 268, 99.15 %) and biased the TPOT denominator. A
  token that renders as `""` is still a decode step. After the fix the delivered
  count matches the requested count exactly.

## 2. Measured, and the simulator's answer for the same arrivals

| rps | `L_pred` | `L_meas` | Δ | TPOT p99 sim / meas | TTFT p99 sim / meas | runs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 11.353 | 11.464 | **−0.97 %** | 40.44 / 41.14 ms | 451 / 567 ms | 1 |
| 0.75 | 18.574 | 18.941 | **−1.94 %** | 44.65 / 45.18 ms | 482 / 593 ms | 1 |
| 0.90 | 23.218 | 23.701 | **−2.04 %** | 46.49 / 47.78 ms | 499 / 619 ms | 3 |
| 1.00 | 26.606 | 27.394 | **−2.88 %** | 48.31 / 49.87 ms | 516 / 682 ms | 1 |
| 10.0 | 169.369 | 170.464 | **−0.64 %** | 116.02 / 117.26 ms | 107 476 / 110 132 ms | 2 |

All eight runs completed 300/300 with zero failures.

**Launch accuracy.** The driver's contract is 20 ms. The four sub-saturation
stages met it (worst 15.58 ms). **The two 10 rps stages did not — 49.28 ms and
29.40 ms.** At 100 ms inter-arrivals with 300 streamed responses in flight the
event loop is the bottleneck. It is reported rather than waved away, and its
size is worth stating: at that stage the p99 TTFT is 110 *seconds*, so a 49 ms
launch error is **0.04 %** of a request's residency and does not perturb the
arrival process in any way the comparison can see.

## 3. Claim 2: the coordinate does not change the answer here

`L_pred < L_meas` at every stage, which is the direction claim 2 assumes — an
optimistic predictor under-states residency, and Little's law carries that into
the concurrency. But:

| rps | margin at `L_pred` | margin at `L_meas` | Δ | verdict differs? |
| ---: | ---: | ---: | ---: | --- |
| 0.50 | 0.3249 % | 0.3256 % | −0.0008 pp | no |
| 0.75 | 0.3749 % | 0.3775 % | −0.0025 pp | no |
| 0.90 | 0.4072 % | 0.4105 % | −0.0034 pp | no |
| 1.00 | 0.4307 % | 0.4361 % | −0.0055 pp | no |
| 10.0 | 1.4320 % | 1.4398 % | −0.0078 pp | no |

**Max |ΔL| = 2.88 %, against the work order's 5 % threshold.** The work order's
risk row prescribes the response: *"청구항 2는 유지하되 명세서에서 '차이가 작은
경우 어느 좌표든 같은 판정' 서술."* That is what §4 records.

It also says to push one stage closer to saturation to see whether the gap
widens. It does not, monotonically: the gap grows 0.97 → 1.94 → 2.04 → 2.88 %
through the sub-saturation ladder and then **falls to 0.64 %** at the saturated
stage. At saturation both sides are queue-bound and the concurrency is set by the
backlog, which both compute the same way; the divergence lives in the middle,
where the service rate still matters. That is a finding about *where* the
coordinate could matter, and it is the opposite of where one would look.

## 4. What the specification should say

1. **Keep claim 2.** Its direction is confirmed on real hardware: the optimistic
   predictor lands below the measurement at every load tested.
2. **State the magnitude honestly.** On this hardware and workload the two
   coordinates differ by under 3 % and select the same margin to within 0.008 pp,
   so **either coordinate gives the same verdict**. The claim's value is that it
   specifies *which* coordinate, not that the choice changed an outcome here.
3. **Do not generalise from A40.** The A40's margin is ~0.3–1.4 % over this
   range. A hardware whose margin is steep in concurrency — RNGD-CARD's rises
   from ~3 % to ~22 % between L 16.6 and 76 — would turn the same 3 % coordinate
   gap into a materially different margin. **That case is not measured**, and the
   specification should say so rather than imply the A40 result settles it.

## 5. The committed A40 domain: anchors confirmed, interior wrong

Compared like-for-like with the committed file (p50 against p50, D101):

| `L_pred` | measured TPOT `e` | committed domain reads | ratio |
| ---: | ---: | ---: | ---: |
| 11.353 | **−0.26 %** | −0.32 % | ✓ agrees |
| 18.574 | −1.38 % | −0.37 % | 3.7× |
| 23.218 | −1.21 % | −0.41 % | 3.0× |
| 26.606 | **−2.38 %** | −0.43 % | **5.5×** |
| 169.369 | **−1.44 %** | −1.41 % | ✓ agrees |

**Both anchors are confirmed by a different harness**, which is strong evidence
that the committed points are sound. **The interior is not.** `a40.accuracy.yaml`
warns in its own SEAM 2 that the 10.8-to-170.56 interval "is interpolated across
a change of regime (unsaturated to saturated) with nothing measured inside it.
Any margin read in that interval is carrying this assumption." Measured inside
it, the assumption fails in the under-conservative direction: the TPOT error is
**not monotone** — it deepens to −2.38 % in the middle and comes back to −1.44 %
at the top, where a straight line between the anchors reads −0.43 %.

Recorded as new points in
**`profiles/calibration/openloop/a40.accuracy.openloop.yaml`** — a separate file
per rule A3, and in a subdirectory per **D102**: the loader refuses two domains
for one hardware label, and that refusal is right. The committed `a40.accuracy.yaml` is unchanged.

## 6. Client-side transport, separated

Measured against the server's own `/metrics` over a 60-request stage:

| | client | engine (`/metrics`) | difference |
| --- | ---: | ---: | ---: |
| TPOT (mean) | 36.64 ms | 36.62 ms | **+0.021 ms (+0.06 %)** |
| TTFT (mean) | 234.26 ms | 214.00 ms | **+20.26 ms (+9.5 %)** |

**TPOT is transport-free**, so every TPOT number in this file is engine-faithful
and directly comparable with the simulator. **TTFT is not**: the client measures
from its own launch, which includes serialising up to 3 998 input token ids and
the SSE first-chunk path.

Removing the 20.26 ms still leaves the measured TTFT error at −24.6 % to −30.1 %
where the committed domain reads −15.7 % to −17.1 %. **Transport explains part of
the gap and not all of it**; the rest is unexplained and is most likely the
harness (in-process `AsyncLLM` against a deployed HTTP server). The open-loop
file stores **raw client-side** TTFT errors with the offset recorded, not
subtracted, and they must not be merged with the committed file's.

**This bears directly on V3's P2**, which is decided on TTFT by 3 111.6 ms — three
orders of magnitude above the transport offset, so the verdict is safe. It is
recorded because a smaller TTFT case would not be.

## 7. Run-to-run spread — the number V3 needs

Three repeats at 0.9 rps, two at 10 rps, same deployment and session:

| stage | metric | values | spread |
| --- | --- | --- | ---: |
| 0.9 rps ×3 | **TPOT p99** | 47.783 / 47.773 / 47.788 | **0.0151 ms (0.032 %)** |
| 0.9 rps ×3 | `L_meas` | 23.701 / 23.704 / 23.699 | 0.0052 (0.022 %) |
| 0.9 rps ×3 | TPOT p50 | 43.764 / 43.760 / 43.788 | 0.0277 ms |
| 0.9 rps ×3 | TTFT p99 | 613.1 / 623.5 / 620.6 | 10.33 ms (1.67 %) |
| 10 rps ×2 | TPOT p99 | 117.271 / 117.251 | 0.0200 ms (0.017 %) |

**V3's P3 question is answered.** Its margin headroom is **0.623 ms** and the
repeatability of the metric that decides it is **0.015 ms** — a factor of 40. So
rule (c)'s pass is **not** inside the noise, and V3 may report both that the
global 18 % over-rejected by 8.009 ms *and* that the per-point margin was right.

**One limit on that number.** These are repeats within one deployment and one
session. A restart, a different card, or a different driver would add variance
this does not capture, so **0.015 ms is a lower bound** on true repeatability. It
is the right bound for V3, which compares two verdicts on one deployment.

## 7.1 NUMA re-check — the ladder is validated, not merely argued

*Added 2026-09-17, after Appendix A.3 of `v3_verdict_accuracy.md` found that
leaving the server's NUMA binding to chance is worth **1.93×** of throughput on
this host. Every stage above ran **unbound**, so the ladder rested on GPUs 0–3
having happened to land on the right node — an argument, not a check.*

The five rates were run again with `numactl --cpunodebind=0 --membind=0`
(GPU 0 is on NUMA node 0), one repeat each, fresh plan id, same workload and
driver. Against the unbound stages above:

| rps | `L` Δ | p99 TPOT Δ | p99 TTFT Δ | output tok/s Δ |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | −0.26 % | −0.46 % | −3.45 % | **+0.00 %** |
| 0.75 | −0.24 % | −0.16 % | +3.28 % | **+0.01 %** |
| 0.90 | −0.30 % | −0.24 % | +0.15 % | **+0.01 %** |
| 1.00 | −0.37 % | −0.26 % | −4.08 % | **+0.00 %** |
| 10.0 | −0.09 % | −0.39 % | −0.56 % | **+0.42 %** |

**Throughput agrees to 0.42 % at worst**, against the 93 % that binding was worth
on the TP=4 deployment. Served concurrency, which is what claim 2 rests on,
agrees to **0.37 %**; p99 TPOT to **0.46 %**.

**So every conclusion in this file stands**, including the two that would have
been most at risk: `L_pred < L_meas` at 0.64–2.88 % (§3), and the L > 170 stage
landing at 170.46 against the committed domain's 170.56 anchor (§2).

**One column is not as clean and is reported rather than smoothed.** p99 TTFT
differs by up to **4.08 %**. That is larger than the 1.67 % run-to-run spread
measured at 0.9 rps over three repeats (§7), but each rate here is a **single**
run on each side, so the comparison cannot separate a real difference from TTFT's
own noise — and TTFT is the noisiest metric in the ladder by an order of
magnitude. It is not resolvable with the data taken, and nothing in this file
turns on a 4 % TTFT difference.

**Why the exposure was small here, now that it is measured rather than assumed.**
The ladder is TP=1 on one card: there is no collective, and the host-side traffic
is one engine's. The TP=4 deployment that exposed the effect drives four workers
whose all-reduce and KV movement cross the host bridge on every step. The
difference is not that TP=1 is immune — it is that it asks far less of the path
that NUMA placement governs.

## 8. Raw material

| what | path |
| --- | --- |
| measured client records, 8 stages | `outputs/p2_evidence/v2/rps*_r*.json` |
| simulated side | `experiments/p2_evidence/results/v2_sim_side.json` |
| joined comparison | `experiments/p2_evidence/results/v2_comparison.json` |
| new open-loop domain | `profiles/calibration/openloop/a40.accuracy.openloop.yaml` (D102) |
| driver | `experiments/scripts/replay_to_endpoint.py` |
| ladder | `experiments/p2_evidence/v2_run_ladder.sh` |
| simulator side | `experiments/p2_evidence/v2_sim_side.py` |
| run logs | `outputs/p2_evidence/v2/{ladder,simside,deploy}.log` |
| NUMA-bound re-check (§7.1) | `outputs/p2_evidence/v2_numa/rps*_numa.json`, runner `experiments/p2_evidence/v2_ladder_numa.sh`, log `outputs/p2_evidence/v2_numa_ladder.log` |
