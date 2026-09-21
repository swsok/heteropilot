# S7.3 open-loop measurements — NPU node `6fe246ed1abf`

*Measured 2026-09-18 to 2026-09-21. **These belong to one machine and must not be
combined with measurements from another** — see "Why this directory is named
after a hash" below. `WORK_ORDER_domain_scoping.md` STEP S7.3;
`docs/deviations.md` D115 (the refit) and D80 (the identity).*

**This is S7.3's dataset, and it is finished here.** It was briefly not: the card
was released mid-measurement for another user and the work was to move to a
second RNGD machine, so an earlier revision of this file described the contents
as preserved relics. **That move did not happen** — the second machine's NUMA
topology turned out to be unsuitable for measurement — so the remaining runs were
taken on this card and these points are the real thing rather than a record of an
abandoned attempt.

The fingerprint in the directory name is *not* vestigial. It is what lets this
dataset be continued rather than restarted: the later runs were confirmed to come
from the same accelerator set (`6fe246ed1abf`) before being merged with the
earlier ones, which is exactly the check D80 exists to make possible.

> **Where the domain files live.** The two here are the pipeline's raw output for
> this dataset. The installed domain is
> `profiles/calibration/openloop/rngd_card.accuracy.openloop.yaml`; if the two
> ever disagree, the installed one is authoritative and this pair is the record
> of what produced it.

## Why this directory is named after a hash

`6fe246ed1abf` is `provenance.accelerator_set.fingerprint` — a hash over the
RNGD and ATOM serials present. This machine is

```
RNG26040100105Q + RNG26040100181Q + RNG26040100187Q     (3 x RNGD)
0000000022502229 + 0000000022502247
  + 0000000022507041 + 0000000022507045                 (4 x ATOM)
```

and the measured card is **npu0, `0000:03:00.0`, serial `RNG26040100181Q`**,
NUMA-bound to node 0.

Before D80 nothing in an artifact distinguished two RNGD nodes: the provenance
block recorded `{"rngd_cards": 3}` and the hostname is not discriminating. A
point measured on the second machine would have appended to the same domain file
under the same `hardware: RNGD-CARD` label with nothing to separate them. That is
why the identity is in the directory name and in every JSON here.

## What is here

| file | what |
| --- | --- |
| `envelope.json` | the nine measured points, assembled from the two runs |
| `raw/replay_*.json` | per-request client-side timestamps, 300 requests each |
| `sim_extra_rates.json` | the simulated side for the four rates S7.0 never ran |
| `openloop_sim_error.json` | the pairing, with full provenance incl. the fingerprint |
| `rngd_card.accuracy.openloop.yaml` | the domain the pipeline produced, p99 basis |
| `rngd_card.accuracy.openloop.p50.yaml` | the same measurements on the p50 basis |

`raw/` is kept because the per-request timings cannot be regenerated without this
card, and the p50/p99 basis question (D101) is live — a later reader may need to
re-derive a statistic these summaries do not carry.

## The measurement

Harness `measure_envelope.py --mode open` (D114), driver
`replay_to_endpoint.py --open-loop --ignore-eos`, pairing
`openloop_sim_error.py --match offered`. 300 requests per run, **15 runs**:
three each at 1.75 and 2.0 rps where S7.4 reads the domain, two at 2.25, one
elsewhere. Every run delivered the trace's 195 753 output tokens to within
0.01 %.

**Run-to-run spread is recorded per point and is not uniform.** At sim L 37.965
the p99 TPOT spread over three runs is **0.116 ms (0.27 %)**; at L 32.475 it is
**0.776 ms (1.97 %)** — seven times larger at the lower load. That is why the
repeats were concentrated rather than spread evenly, and why a single-run point
says *"single run, so no run-to-run spread is known here"* instead of reporting
a spread of zero.

| offered | runs | L meas | L sim | gap | p99 TPOT meas | sim | **err** | paired |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.75 | 1 | 11.309 | 11.730 | +3.72 % | 27.548 | 28.106 | **+2.02 %** | yes |
| 1 | 1 | 16.169 | 16.545 | +2.32 % | 30.113 | 29.884 | **-0.76 %** | yes |
| 1.25 | 1 | 21.711 | 21.668 | -0.20 % | 32.292 | 31.567 | **-2.25 %** | yes |
| 1.5 | 1 | 28.713 | 27.002 | -5.96 % | 35.152 | 33.056 | **-5.96 %** | yes |
| 1.75 | 3 | 36.381 | 32.475 | -10.74 % | 39.437 | 34.271 | **-13.10 %** | yes |
| 2 | 3 | 44.506 | 37.965 | -14.70 % | 42.895 | 35.332 | **-17.63 %** | yes |
| 2.25 | 2 | 60.550 | 43.689 | -27.85 % | 59.862 | 36.715 | **-38.67 %** | **no** |
| 2.5 | 1 | 73.997 | 49.527 | -33.07 % | 65.422 | 38.480 | **-41.18 %** | **no** |
| 3 | 1 | 99.967 | 61.677 | -38.30 % | 89.422 | 42.871 | **-52.06 %** | **no** |
| 3.5 | 1 | 114.198 | 74.498 | -34.76 % | 89.111 | 46.691 | **-47.60 %** | **no** (saturated) |

## Three findings that should survive the move

**1. The error changes sign, so no scalar can express it.** It runs **+2.02 % at
sim L 11.73 to −17.63 % at 37.97**, crossing zero between **L 11.73 and 16.55** (interpolated ≈ 15.0). The simulator is *pessimistic* at low load and
increasingly optimistic above it. That is D22's lesson restated on the open-loop
axis, and it is why `outside_domain: refuse` is kept: a curve that changes sign
inside its own range says nothing about outside it.

**2. The pairing cannot be formed above sim L 37.965, and not because of
saturation.** The bound is measured, not inferred: 2.25 rps was run twice for
this purpose and lands at a **−27.85 %** concurrency gap, already outside the
±20 % guard, so the domain stops at the 2.0 rps point below it. At 2.25, 2.5 and
3.0 the card is *not* saturated — the queue is flat — it simply serves far more
concurrency at far worse latency than the model predicts. Only 3.5 saturates,
and there the measured p99 TPOT *plateaus* (89.42 ms at 3.0, 89.11 at 3.5) as
the backlog absorbs the load rather than the per-token cost rising further.

The four unpaired rates are kept in the domain's `provenance.unpaired_points`,
keyed by **offered rps** because that is the only axis the two sides still share
up there. Their p99 under-prediction runs **−38.67 %, −41.18 %, −52.06 %,
−47.60 %** at 2.25 / 2.5 / 3.0 / 3.5.

**3. Both of S7.0's selected candidates are affected, and S7.4 must reselect.**

* **P2 (sim L 74.498) is unmeasurable open-loop on this class of card** —
  roughly twice the highest concurrency at which a pair can be formed.
* **P3 (sim L 37.965) inverts.** Its margin moves from the closed-loop domain's
  **7.51 %** to **21.40 %**. A P3-shape needs the per-point margin *below* the
  global 18 % so that (b) over-rejects where (c) does not; at 21.40 % the
  per-point rule is the stricter of the two, so the candidate becomes a
  **P2-shape** — the invention catching an optimistic candidate rather than
  rescuing a wrongly-rejected one.

Findings 1 and 2 are properties of the simulator against RNGD-CARD hardware and
are **expected to reproduce** on the second machine. Finding 3 follows from them.
None of the three is a number the new machine may inherit — they are hypotheses
it should confirm.

## Two bases, and why both files exist

At the same operating point the two disagree, and at low load they disagree in
**sign**:

| sim L | p99 | p50 |
| ---: | ---: | ---: |
| 11.730 | +2.02 % | +7.82 % |
| 16.545 | **−0.76 %** | **+5.25 %** |
| 37.965 | −17.63 % | −14.88 % |

At L 16.5 the p50 basis says "pessimistic, charge nothing" and the p99 basis says
"optimistic, charge something". Since the margin is applied to a p99 (D101), a
p50-fitted domain would charge **zero where a margin is due**. That is the
concrete case for fitting where the value is used, and for keeping the p50
sibling only as comparison material against `rngd_card_edf.yaml`.

## What was thrown away before this, and why it is not here

The first runs of S7.3 are **not** in this directory. `replay_to_endpoint` caps
completions at `min(output_toks, --max-tokens-cap)` and that cap defaulted to
**512** against a trace whose p50 is 632 and max 1021 — 299 of 300 rows over. The
card delivered **153 900 tokens against 195 753**, 78.6 % of the simulator's
work, and the shortfall read as the simulator over-predicting concurrency by +18
to +37 %. One of those pairs passed the ±20 % concurrency guard. Every point in
this directory was taken after the cap was resolved from the trace and carries
`max_tokens_cap: 1021`, `cap_truncated_rows: 0`. See D115.

## Reproducing the pairing (no card needed)

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/openloop_sim_error.py \
    --real experiments/results/s73_npu_6fe246ed1abf/envelope.json \
    --sim experiments/p2_evidence/results/v3r_candidates.json \
          experiments/results/s73_npu_6fe246ed1abf/sim_extra_rates.json \
    --sim-cache outputs/p2_evidence/v3r/cache \
    --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl --num-reqs 300 \
    --out-dir /tmp/s73check
```

`--sim-cache` is this repository's `outputs/p2_evidence/v3r/cache`, which is
untracked; without it the p50 column cannot be formed and the run says so rather
than deriving a p50 from a p99.
