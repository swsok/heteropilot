# S7.3 — the RNGD-CARD open-loop refit, and where it stops

*`WORK_ORDER_domain_scoping.md` STEP S7.3; `docs/deviations.md` D115. Measured
2026-09-18 on the **NPU node** (`whichnode.sh` → `npu`), RNGD **npu0**, PCI
`0000:03:00.0`, NUMA-bound to node 0. No holder pod was present — read from the
process list, not `alloc_status`, which reads all-zero while a pod holds a card
(`docs/nodes/npu.md`).*

Harness `measure_envelope.py --mode open` (S7.2, D114); driver
`replay_to_endpoint.py --open-loop --ignore-eos`; pairing
`openloop_sim_error.py --match offered`. Simulated side: S7.0's sweep
(`v3r_candidates.json`) plus four rates it never ran
(`experiments/results/s73_npu_6fe246ed1abf/sim_extra_rates.json`, committed
beside the data), same fixture, seed 42, 300
requests, candidate `furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t2048`.

> **Read this before using the domain.**
>
> 1. **It is fitted on p99, unlike every domain before it.** The margin is
>    applied to a p99 (D101) and S7.4's verdicts are p99, so the loaded file is
>    fitted where it is used. The `.p50.yaml` sibling carries the same
>    measurements on the old basis, for comparison with `rngd_card_edf.yaml`.
> 2. **The two bases disagree in SIGN at TWO of the six points.** At sim L
>    16.545 the p50 basis reads +5.25 % and the p99 basis −0.76 %; at L 21.668
>    they read +2.25 % and −2.25 % — equal and opposite. A positive error is
>    pessimism and earns **no margin** by construction, so on the p50 basis both
>    points charge nothing while on the p99 basis both are due a margin. A
>    p50-fitted domain applied to a p99 therefore charges **zero where a margin
>    is due**, over a third of the domain's range. That is D101 producing a
>    wrong answer, not an inelegant one.
> 3. **The domain stops at sim L ≈ 38, and the stop is evidenced.** Above it the
>    two sides sit at different operating points; those measurements are kept in
>    `provenance.unpaired_points` rather than discarded.
> 4. **The first runs of this step were invalid and were thrown away.** §2.

## 1. Why a refit exists at all

S4 stated `arrival_process: closed_loop` on `rngd_card_edf.yaml` (D113), and
`python -m planner plan` always replays an arrival trace. Under the default
`condition_mismatch: refuse` no RNGD candidate may consult that domain — **72 of
72 rows** in S7.0's sweep were held on that single field. A wider load range does
not answer them. A measurement under the arrival process the planner performs
does, and that is a new file, not more points on the committed one: two protocols
on one interpolation axis is the class of error D22 was.

## 2. The first measurements were invalid, and how that was caught

`replay_to_endpoint` sets `max_tokens = min(row["output_toks"], cap)` with `cap`
defaulting to **512**. The committed trace has p50 632 and max 1021 — **299 of
its 300 rows exceed 512**. The first runs therefore delivered **153 900 tokens
against the trace's 195 753**, 78.6 % of the work the simulator did, and the
shortfall read as the simulator over-predicting served concurrency by +18 % to
+37 %. One of those pairs passed the ±20 % concurrency guard.

It is the failure `--ignore-eos` exists to prevent (V2 §1), arriving through a
different door, and `openloop/a40.accuracy.openloop.yaml` had already written
down the invariant that catches it: *"The simulator generates exactly the trace's
output_toks … every stage delivered 195 753 output tokens, the trace's exact
total."*

Three guards now stand where one did:

| guard | what it catches |
| --- | --- |
| `--max-tokens-cap auto`, resolved from the trace | truncation by default |
| explicit cap → truncated rows counted and reported | an operator's own cap |
| delivered-token ratio in the pairing | the end-to-end invariant |

The third is the one that matters: a short workload lands at a *plausible*
concurrency, not an obviously wrong one, so the concurrency guard cannot see it.
Every point below delivered the trace's tokens to within 0.01 %, and the run
block records `max_tokens_cap: 1021`, `cap_truncated_rows: 0` so the artifact
says so itself.

## 3. Method, and the guard that decides whether a point exists

The same offered rate goes to both sides (`--match offered`, the committed
convention). Served concurrency is then an **outcome** on both, and a pair whose
two concurrencies differ by more than 20 % is excluded: the simulator's p99 there
is a different operating point's p99. The x axis is the **simulator's** served
concurrency, because that is what the planner knows when it consults the table.

**The guard is not independent of what is being measured**, and that is worth
stating. `L = λ·W` and `W ≈ TTFT + N·TPOT`, so at unsaturated load a TPOT error
propagates almost 1:1 into a concurrency error — at 1.5 rps the gap is −5.96 %
and the TPOT error −5.96 %. The guard can therefore only reject a point whose
TPOT error already approaches 20 %, which is precisely the regime the domain most
needs. It is a sanity check on whether the two sides are at the same operating
point, not an independent check on the error.

## 4. The domain

| sim L (x axis) | `tpot_err_pct` p99 | `_p50` | margin | runs | p99 spread |
| ---: | ---: | ---: | ---: | ---: | --- |
| 11.730 | **+2.02 %** | +7.82 % | 0.00 % | 1 | — single run |
| 16.545 | **-0.76 %** | +5.25 % | 0.77 % | 1 | — single run |
| 21.668 | **-2.25 %** | +2.25 % | 2.30 % | 1 | — single run |
| 27.002 | **-5.96 %** | -5.56 % | 6.34 % | 1 | — single run |
| 32.475 | **-13.10 %** | -10.86 % | 15.07 % | 3 | 0.776 ms (1.97 %) |
| 37.965 | **-17.63 %** | -14.88 % | 21.41 % | 3 | 0.116 ms (0.27 %) |

The error runs from **pessimistic at low load to sharply optimistic at high**,
crossing zero between sim L 11.730 and 16.545. No scalar expresses that, which
is D22's lesson
restated on the open-loop axis; it is also why `outside_domain: refuse` is kept
rather than `widen_error_bars`, since a curve that changes sign inside its own
range says nothing about what happens outside it.

Repeat counts differ by point and each records its own (user decision,
2026-09-18): **3 runs at 1.75 and 2.0 rps**, where S7.4 reads the domain, **2 at
2.25**, **1 elsewhere**. A single-run point says so rather than reporting a
spread of `0.0000 ms`, which would read as measured agreement rather than as
absence of data. The precedent is `a40.accuracy.openloop.yaml`, whose notes carry
`1 run`, `2 runs` and `3 runs` side by side.

## 5. Where it stops, and the evidence for the stop

| offered rps | L real | L sim | gap | p99 meas | p99 sim | under-prediction | saturated | runs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2.25 | 60.550 | 43.689 | -27.85 % | 59.862 | 36.715 | **-38.67 %** | no | 2 |
| 2.5 | 73.997 | 49.527 | -33.07 % | 65.422 | 38.480 | **-41.18 %** | no | 1 |
| 3 | 99.967 | 61.677 | -38.30 % | 89.422 | 42.871 | **-52.06 %** | no | 1 |
| 3.5 | 114.198 | 74.498 | -34.76 % | 89.111 | 46.691 | **-47.60 %** | **yes** | 1 |

These are **measurements, kept** (user direction). Each is a real observation of
the card at a real offered rate; what none of them is, is a calibration point,
because the simulator sat at a different served concurrency. They live under
`provenance.unpaired_points`, keyed by **offered rps** — the one axis both sides
still share up there — and they are what turns "the domain ends at L ≈ 38" from
an assertion into a bounded claim.

**Only the 3.5 rps point is saturated.** 2.5 and 3.0 are non-saturated steady
state: the card is keeping up, it is simply serving far more concurrency at far
worse latency than the model predicts. At 3.5 the queue does grow, and the
measured p99 TPOT *plateaus* (89.42 ms at 3.0, 89.11 ms at 3.5) because the
backlog absorbs the extra load rather than the per-token cost rising further.

## 6. The two bases

| sim L | p99 basis | p50 basis | same sign? |
| ---: | ---: | ---: | --- |
| 11.730 | +2.02 % | +7.82 % | yes |
| 16.545 | -0.76 % | +5.25 % | **NO** |
| 21.668 | -2.25 % | +2.25 % | **NO** |
| 27.002 | -5.96 % | -5.56 % | yes |
| 32.475 | -13.10 % | -10.86 % | yes |
| 37.965 | -17.63 % | -14.88 % | yes |

`compared_metric` states the basis on both files, on the domain and in
`index.yaml`. `AccuracyDomainMargin` warns when a domain explicitly declares
`tpot_p50`, so the mismatch appears in plan output rather than only in a
docstring. Silence earns no warning: every domain committed before today is p50
in fact and says nothing, and warning on silence would fire on every run while
telling nobody anything new. Migrating those two files is left as a separate
step (D101 addendum, D115).

## 7. What this changes for S7.4

**S7.0's P2 candidate cannot be measured open-loop on this card.** It sits at
sim L = 74.498 — roughly twice the highest concurrency at which a pair can be
formed. No amount of repeats fixes that; it is a property of where the card and
the model diverge.

**S7.0's P3 candidate is inside the domain, and its verdict shape inverts.** At
sim L = 37.965 the closed-loop domain gave a margin of 7.51 %; the open-loop one
gives **21.40 %**. A P3-shape needs the per-point margin *below* the global 18 %
so that (b) over-rejects where (c) does not; at 21.40 % the per-point rule is the
stricter of the two, so the candidate becomes a P2-shape — the invention catching
an optimistic candidate rather than rescuing a wrongly-rejected one.

Re-selecting against this domain is **S7.4's** work and is deliberately not done
here. What S7.3 establishes is that the selection must be redone, and why.

## 8. What this does not establish

* **Nothing about the card above sim L ≈ 38.** The unpaired points bound the
  divergence; they do not calibrate it.
* **Nothing about TTFT beyond the raw client-side figure.** Over HTTP the
  client's TTFT includes request serialisation and the SSE first-chunk path,
  measured at +20.26 ms on the A40, and nothing is subtracted here.
* **Nothing about other parallelism or placement.** This is one card, `tp=1`,
  one island, NUMA-bound. A candidate differing on any of those is a
  `condition_mismatch` against this domain exactly as it was against the old one.
* **Nothing about the closed-loop domain's correctness.** `rngd_card_edf.yaml`
  is untouched and remains right for what it measured; it simply cannot answer
  for an arrival-trace replay.

## 9. Reproducing

**The pairing and both domain files, from committed data, no card needed:**

```bash
D=experiments/results/s73_npu_6fe246ed1abf
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/openloop_sim_error.py \
    --real $D/envelope.json \
    --sim experiments/p2_evidence/results/v3r_candidates.json $D/sim_extra_rates.json \
    --sim-cache outputs/p2_evidence/v3r/cache \
    --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl --num-reqs 300 \
    --out-dir /tmp/s73check --write-domain /tmp/s73check/rngd_card.accuracy.openloop.yaml
```

`--sim-cache` is untracked. Without it the run still works but the `_p50` column
cannot be formed, and it says so rather than deriving a p50 from a p99.

**The measurements themselves, which need the card** — and which must not be
appended to this dataset unless `whichnode.sh` prints the same `accel serials`
(D80):

```bash
bash scripts/whichnode.sh     # accel serials must be RNG26040100105Q+...181Q+...187Q
ps -eo pid,etime,cmd | grep "[r]ngd_pd.serving.cluster"   # the only reliable holder check
ART=~/.cache/huggingface/hub/models--furiosa-ai--Llama-3.1-8B-Instruct/snapshots/231d94fbc03cdd66aaeb2411697064a45f008ec7
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/measure_envelope.py \
    --backend furiosa --mode open --rps 0.75,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5 \
    --num-reqs 300 --artifact "$ART" \
    --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
    --card 0 --port 8020 --out outputs/s73_rerun
```

The driver runs in `.venv`; the client runs in `/usr/bin/python3`, which is where
the vendor stack and `openai` live — that split is the `bench_python` column of
the `BACKENDS` table, not a choice made here. **Do not pass `--max-tokens-cap`**:
its `auto` default resolves the cap from the trace, and a number below the
trace's longest row is what invalidated the first attempt (§2).

**Regenerating the index after installing a domain:**

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/rebuild_domain_index.py --check
```

