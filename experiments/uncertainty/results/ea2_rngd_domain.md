# E-A2 — RNGD-CARD accuracy domain (uncertainty work order STEP A5)

> **SUPERSEDED 2026-09-11 (deviations D33).** The domain this experiment produced,
> `profiles/calibration/rngd_card_edf.domain.yaml`, is **not** carried onto
> `main` and the file is removed. Two reasons, both found after this document was
> written:
>
> 1. **It is the D32 mis-pairing.** Each real run (requested c16/32/64/128, served
>    15.3–107.2) was paired with a burst simulation capped at the same
>    `--max-num-seqs`. The sim side's own served concurrency — recorded in the
>    `note` of every point below — is **71.0 / 76.2 / 154.2 / 189.3**: two to
>    three times the real side's. The two runs were therefore at different
>    operating points, and the "error" column compares a simulator at L=154 with
>    hardware at L=59. D32 shows what this does: at matched rates the simulator
>    saturates near served 44, so its TPOT above ~30 is a different operating
>    point's TPOT. The monotone +1 % → +47 % shape below is largely that
>    mismatch, not a property of the card model at the hardware's load.
> 2. **`main` already carries the measured curve this was reaching for.** PR #72
>    and #77 fitted `RNGD-CARD`'s accuracy domain from matched-rate 300-request
>    runs: nine points over served 1.02–76.0 in `profiles/calibration/rngd_card_edf.yaml`
>    (`accuracy_domain:` block, D32), with the two points whose sim side could not
>    reach the hardware's load *refused rather than fitted*. That domain is what
>    E-A1 now uses.
>
> What survives from this experiment: the method for reconstructing the exact
> request sets, the `--no-enable-prefix-caching` finding (D12), the TTFT
> exclusion and its reason (D19), and the served-concurrency reproduction of
> D22's table, which `tests/test_accuracy_domain.py` still asserts. The
> `fit-accuracy-domain` command it used has been rewritten to emit `main`'s
> `accuracy_domain:` schema and to print the sim/real concurrency gap for every
> pair, so the mismatch above is visible at fit time. The rest of this document
> is kept as written, as the record of what was done.


**What this is.** The simulator's TPOT error against real FuriosaAI RNGD hardware
at four served concurrencies, so the error can be treated as a function of the
operating point instead of one scalar per workload. This table is the §7 evidence
for the patent's 도 3.

**Provenance.** Produced 2026-09-10 at commit `3fd9770` on the **a5000** node
(`scripts/whichnode.sh`). Simulation only — no accelerator was used and none was
needed. The real side is committed measurement from the **NPU** node
(`outputs/rngd_envelope/edf/real_c{16,32,64,128}.json`, measured 2026-08-31 on
npu0, TP=8, per docs/deviations.md D22); it is neither re-run nor relabelled here
(absolute rule 3).

## The domain

`profiles/calibration/rngd_card_edf.domain.yaml`, entry
`sharegpt-llama31-8b-300-closed-loop-c16-128`, domain **[15.32, 107.19]**.

| requested | served L | decode-time L | real TPOT | sim TPOT | **TPOT error** | n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 15.32 | 15.07 | 25.71 ms | 26.32 ms | **+1.25 %** | 128 |
| 32 | 29.35 | 28.37 | 31.18 ms | 30.45 ms | **+4.90 %** | 128 |
| 64 | 59.19 | 56.89 | 44.54 ms | 36.83 ms | **+20.36 %** | 256 |
| 128 | 107.19 | 99.83 | 67.88 ms | 45.59 ms | **+47.25 %** | 300 |

Error is `(real - sim) / sim`, the `ErrorStats` convention: positive means the
simulator predicted faster than reality, so a robust plan inflates by that much.
Note this is the OPPOSITE denominator from D22's prose, which states optimism as
`(real - sim) / real`.

**The shape is the finding.** The simulator agrees to within a few percent at the
concurrency the profile was fitted at and is optimistic by nearly half at the top
of the measured envelope, rising monotonically. That is D22's qualitative claim -
"the decode model is accurate at the concurrency it was fitted on and degrades
above it" - quantified as a curve a planner can interpolate.

**Both halves reproduce committed artifacts exactly.** Served L computes to
15.32 / 29.35 / 59.19 / 107.19 against D22's published 15.3 / 29.3 / 59.2 / 107.2,
and real TPOT to 25.71 / 31.18 / 44.54 / 67.88 against D22's table to the last
digit. `tests/test_accuracy_domain.py` asserts the first of these.

**Estimator.** The committed errors use the same estimator the existing
calibrations were fitted with - `write_summary` -> `parse_validation_summary` ->
`compute_error_stats`, pooling the five statistic rows (mean, median, P90/95/99).
Comparing the two distributions' means alone gives -2.29 / +2.37 / +20.94 /
+48.87 %: the same shape, and the same conclusion. The pooled estimator is used
because STEP A3 falls back to the bucket's scalar `mean_error` outside the
domain, and a point fitted by a different estimator would make the margin jump
at the domain edge.

## Method

The real runs are **closed-loop**: `bench_furiosa_endpoint.py` ignores
`arrival_time_ns` and holds N requests in flight (D19). A burst trace alone does
not reproduce that once the request set exceeds N, so each sim run pairs a burst
with `--max-num-seqs N`, which caps running requests exactly as vLLM does
(`serving/core/scheduler.py:85`). The `decode-time L` column confirms it worked:
15.07 / 28.37 / 56.89 / 99.83 against caps of 16 / 32 / 64 / 128.

The request sets are reconstructed exactly, not approximated: each real run used
rows `0..n-1` of `workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl` (128 / 128 /
256 / 300), and every `input_toks` / `output_toks` matches the committed
per-request record.

```bash
# per point: burst trace, then simulate with the matching concurrency cap
python -m serving \
  --cluster-config experiments/configs/clusters/rngd-card-llama31-8b-tp1.json \
  --dtype bfloat16 --block-size 16 --no-enable-prefix-caching \
  --dataset outputs/uncertainty/ea2/burst_c<N>.jsonl \
  --output outputs/uncertainty/ea2/sim_c<N>.csv --max-num-seqs <N> --run-id ea2np<N>

python -m planner fit-accuracy-domain \
  --real outputs/rngd_envelope/edf/real_c{16,32,64,128}.json \
  --sim  outputs/uncertainty/ea2/sim_c{16,32,64,128}.csv \
  --hardware RNGD-CARD --label-only \
  --label sharegpt-llama31-8b-300-closed-loop-c16-128 \
  --shape in_lt1024-out_ge512 --arrival-process closed_loop --metric tpot \
  --base profiles/calibration/rngd_card_edf.yaml \
  --out profiles/calibration/rngd_card_edf.domain.yaml
```

**`--no-enable-prefix-caching` is not optional.** The first attempt ran with the
simulator's default (prefix caching ON) and c64 and c128 died with
`[MemoryModel] tried to load 2.00MB but only 1.49MB is available` at 99.8 % NPU
memory - docs/deviations.md **D12**, prefix-cache memory growing until the run
dies. The planner disables prefix caching for every candidate, so the domain must
describe that configuration anyway; c16 and c32 were re-run for consistency
rather than kept.

**`outputs/envcheck/rngd_verify_card_edf.csv` is NOT one of these four points**
(the work order's 조사 필요). It has 20 rows on `outputs/envcheck/rngd20.jsonl`,
while these are 128-300 rows on `sharegpt-llama-3.1-8b-300-sps10.jsonl`. It could
not be reused.

## TTFT is excluded, and why

The domain carries **TPOT only**. Sim TTFT under a burst is not comparable to the
real closed-loop client's:

| requested | real TTFT | sim TTFT (burst) |
| ---: | ---: | ---: |
| 16 | 288.5 ms | 65 134 ms |
| 32 | 725.0 ms | 34 464 ms |
| 64 | 1 173.5 ms | 41 192 ms |
| 128 | 3 260.6 ms | 28 309 ms |

A burst puts every request in the queue at t=0, so the ones the real client had
not yet sent accumulate waiting time the real run never paid - a 226x inflation
at c16. There is nothing honest to compare, so `--metric tpot` records no TTFT
point, `arrival_process: closed_loop` marks why, and the margin policy reports
TTFT as `unmeasured` rather than falling back to a scalar. D19 is the reason this
is a TPOT-only result and not a failure: fixing the arrival handling moved the
TTFT calibration substantially (alpha 2.089 -> 1.241) and TPOT by about 1 %
(0.025 -> 0.019), so the arrival process is a TTFT effect.

## Comparison against D22's interpolated figure

D22 interpolated its measured curve to eff 76 and reported the simulator **18 %**
optimistic on TPOT (52.7 ms measured against 43.2 ms). This domain interpolates
to **+29.8 %** at L=76, which is 22.9 % in D22's denominator - larger.

They are not the same comparison and neither supersedes the other. D22's sim
number came from `pd_slo_sweep.py`'s winning configuration, a different trace,
arrival process and batch cap; the four points here are a controlled pairing -
same request set, matching concurrency cap, prefix caching off on both. The
controlled pairing is the one a per-candidate margin should use, and it is the
more pessimistic of the two.

## A40: one point, and a caveat that matters more

`outputs/phase0_bench/A40/vllm/` is a **single** run, so its "5 validation
points" are five statistics of one measurement, not five operating points: the
domain would have zero width. `a40.domain.yaml` is therefore **not committed**
(the work order marks A40 optional) - but the reason is worth more than the
artifact would have been.

| run | in-system L | waiting | decoding |
| --- | ---: | ---: | ---: |
| A40 validation (open loop, saturated) | 170.6 | **67.8** | 102.8 |
| RNGD c16 (closed loop) | 15.32 | 0.25 | 15.07 |
| RNGD c32 | 29.35 | 0.98 | 28.37 |
| RNGD c64 | 59.19 | 2.30 | 56.89 |
| RNGD c128 | 107.19 | 7.36 | 99.83 |

**`sum(latency) / wall` equals in-SERVICE occupancy only when queueing is
negligible.** On the closed-loop RNGD runs waiting is 1.6-6.9 % of the total, so
D22's statistic is the service occupancy and the domain axis is sound. The A40
run replayed arrivals at ~10 rps against a server completing 1.67 rps, so it
saturated: 40 % of its 170.6 is queue, not service. Putting that number on the
same axis as the RNGD points would compare two different quantities.

Consequence for Stage C, where the A40 c-sweep belongs: an open-loop A40 sweep
must record the decode-time occupancy, or run below saturation, or the operating
point it reports will not mean what the RNGD points mean.

## Files

- `profiles/calibration/rngd_card_edf.domain.yaml` — the domain (committed)
- `outputs/uncertainty/ea2/burst_c{16,32,64,128}.jsonl` — the sim traces
- `outputs/uncertainty/ea2/sim_c{16,32,64,128}.csv` — the sim runs
- `outputs/uncertainty/ea2/sim_c*.log` — simulator stdout, including the D12 failure
