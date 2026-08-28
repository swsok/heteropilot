# The −71 % TTFT gap was the validation harness, not the scheduler

*Investigated 2026-08-28 on the A40 server. No NPU was needed: the real-side data
(`outputs/rngd_bench/real_tp8.json`) is the committed 2026-08-26 measurement and
is unchanged. Only the simulator side was re-run.*

## The claim being tested

`docs/PROJECT_REPORT.md` §4.8.4 and `rngd_edf_bundle_notes.md` recorded the card
profile's TTFT error as a **scheduler** difference:

> The residual TTFT error is a scheduler artefact, not a profile error. −36.6 % is
> present with no queuing at all; the stretch to −71.3 % is upstream's scheduler
> queuing ~2.2× less than furiosa-llm's.

The open question was *which* knob — `max_num_seqs`, chunked-prefill admission, or
the P/D interleave. **It is none of them.**

## Root cause: one side honours the arrival timestamps, the other ignores them

The validation compares two runs over `outputs/envcheck/rngd20.jsonl`:

- **Simulator.** `python -m serving` replays the trace's `arrival_time_ns` column.
  Those arrivals are spread over **1.78 s** (46.9 ms → 1823.9 ms, 20 distinct).
- **Real.** `experiments/scripts/bench_furiosa_endpoint.py` fires every row under
  `asyncio.Semaphore(args.concurrency)` with `concurrency=64` against 20 requests.
  All 20 acquire immediately. **The string `arrival` does not appear anywhere in
  that file** — the column is silently ignored, and the real run is a burst.

So the simulator was given a 1.78-second-long arrival process and the real server
was given an instantaneous one. The simulator queued less because *less arrived at
once*, which is not a scheduler property.

## The test

Same trace, same profile, same everything, with `arrival_time_ns` set to 0 for all
20 requests (`outputs/envcheck/rngd20_burst.jsonl`) so the simulator sees the burst
the bench actually produced. `queuing_delay` is a column the simulator already
emits, so TTFT decomposes without instrumentation.

| profile / arrivals | TTFT | err | queue | prefill | TPOT | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| card-EDF / spread *(committed)* | 403.0 | **−71.3 %** | 253.3 | 149.7 | 27.55 | −3.1 % |
| **card-EDF / burst** | **1332.9** | **−5.1 %** | 1126.5 | 206.3 | 27.60 | −3.0 % |
| per-PE tp8 / spread *(committed)* | 946.9 | −32.6 % | 655.5 | 291.5 | 35.70 | +25.5 % |
| **per-PE tp8 / burst** | **2004.8** | **+42.8 %** | 1699.9 | 304.9 | 35.69 | +25.5 % |
| **real** | **1404.1** | — | ~1246 | ~158 | 28.44 | — |

**TPOT does not move** (27.55 → 27.60, 35.70 → 35.69). That is the control: the
change touches queueing only, exactly as intended, so the TTFT movement is not an
artefact of having perturbed something else.

### The distributions match, not just the means

A mean can be hit by accident. The whole shape lines up:

| | sorted per-request TTFT (ms) |
| --- | --- |
| real | 402 · 480 · 550 · 557 · 637 · 732 · 898 · 908 · 1067 · 1075 · 1241 · 1254 · 1535 · 1614 · 2018 · 2188 · 2274 · 2867 · 2888 · 2897 |
| sim, burst | 432 · 432 · 638 · 638 · 638 · 638 · 836 · 1044 · 1044 · 1044 · 1244 · 1244 · 1444 · 1676 · 1676 · 2122 · 2348 · 2348 · 2570 · 2603 |
| sim, spread *(committed)* | 41 · 115 · 133 · 145 · 161 · 175 · 177 · 203 · 205 · 222 · 237 · 239 · 268 · 429 · 449 · 824 · 861 · 950 · 1108 · 1119 |

Real and burst-sim are the same ramp over the same range — the signature of a
burst drained in sequence. The committed spread run is a different distribution
that starts near zero and never reaches the real maximum.

### Percentiles, card-EDF with matched arrivals

| metric | real | sim | diff |
| --- | ---: | ---: | ---: |
| TTFT mean | 1404.1 | 1332.9 | **−5.1 %** |
| TTFT median | 1157.9 | 1143.9 | −1.2 % |
| TTFT p90 | 2869.5 | 2370.2 | −17.4 % |
| TTFT p95 | 2888.1 | 2571.8 | −11.0 % |
| TTFT p99 | 2895.4 | 2596.7 | −10.3 % |
| latency mean | 20941.3 | 20242.5 | −3.3 % |

Against **−61 % to −80 % across percentiles** before. What remains is a tail
under-prediction of 10–17 %, not a systematic collapse.

## What this overturns

**1. The scheduler diagnosis is withdrawn.** There is no evidence upstream's
scheduler queues 2.2× less than furiosa-llm's. The apparent difference was the
input. `max_num_seqs`, chunked-prefill admission and the P/D interleave were never
implicated, and the scheduler comparison the report called for is not needed.

**2. The per-PE profile is not the better TTFT model — the opposite.** Its
−32.6 % looked better than the card's −71.3 %, and
`docs/PROJECT_REPORT.md` §4.8.6 recommends it "for TTFT-feasibility decisions" on
that basis. With arrivals matched it is **+42.8 %** against the card's −5.1 %.
Its −32.6 % was **two errors cancelling**: a queue 2.2× too short multiplied by a
prefill cost 93 % too high (304.9 ms against a real ~158 ms). The card-EDF
profile's prefill is 206.3 ms, and its 149.7 ms in the spread run was within 5 %
of real isolated prefill.

**3. Both TTFT calibrations were fitting a harness bug.** Refitted from the burst
runs with the same `compare_rngd_sim_vs_real.py --calibration-out` path:

| | TTFT α | TTFT β | TTFT fit error |
| --- | ---: | ---: | ---: |
| `rngd_card_edf.yaml` — before | 2.089 | +646.0 | **2.340** (recorded "unusable") |
| `rngd_card_edf.yaml` — after | 1.241 | −242.1 | **0.103** |
| `rngd.yaml` (per-PE) — before | 1.336 | +183.0 | 0.473 |
| `rngd.yaml` (per-PE) — after | 0.851 | −302.5 | −0.263 |

**The card profile's TTFT calibration goes from unusable to usable — 23× better —
without touching the model, the bundle or the hardware.** TPOT barely moves
(0.025 → 0.019 card, −0.204 → −0.206 per-PE), which is the same control as above.

Consequence for anything quoted off the card fixture: the warning in
`docs/HANDOVER_A40.md` to apply `real = 2.089·sim + 646` before any TTFT claim is
superseded. The 480 ms p99 winner in `pd_slo_sweep.md` was calibrated to ~1649 ms
under the old fit; under the new one it is `1.241 × 480 − 242 ≈ 354 ms`. **Neither
figure should be used yet** — `pd_slo_sweep.py` generates its own arrival process,
so whether that sweep has the same mismatch is a separate question this
investigation did not settle.

## What is still open

- **A 10–17 % tail under-prediction** remains at p90–p99 with arrivals matched.
  Small enough to be bucket quantisation (`rngd_sim_vs_real_summary.md` measures
  +10.9 % aggregate charged-vs-actual prefill tokens, which the simulator cannot
  see), but that is a hypothesis, not a measurement.
- **The −36.6 % unloaded-prefill gap is untouched by this.** It came from a
  separate sparse-arrival comparison of 6 requests and is a genuine prefill-cost
  difference; `rngd_edf_bundle_notes.md` attributes ~11 % of it to bucket
  quantisation and the rest to server-side work the device trace cannot see.
- **Whether other sweeps share the mismatch.** Any comparison that pairs
  `python -m serving` against `bench_furiosa_endpoint.py` inherits it. The A40
  bench (`bench/`) is a different harness and was not examined here.

## Reproduce

```bash
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"   # chakra needs the venv on PATH

# burst trace: the committed one with every arrival_time_ns zeroed
python -c "import json;[print(json.dumps({**json.loads(l),'arrival_time_ns':0})) for l in open('outputs/envcheck/rngd20.jsonl')]" \
    > outputs/envcheck/rngd20_burst.jsonl

.venv/bin/python -m serving \
    --cluster-config experiments/configs/clusters/rngd-card-llama31-8b-tp1.json \
    --dtype bfloat16 --block-size 16 --dataset outputs/envcheck/rngd20_burst.jsonl \
    --output outputs/envcheck/rngd_verify_card_edf_burst.csv --run-id burstcard

.venv/bin/python experiments/scripts/compare_rngd_sim_vs_real.py \
    --sim-csv outputs/envcheck/rngd_verify_card_edf_burst.csv \
    --real-json outputs/rngd_bench/real_tp8.json \
    --out-dir outputs/rngd_bench --prefix rngd-card-edf-burst \
    --hardware RNGD-CARD --bucket sharegpt-llama31-8b-20 \
    --calibration-out profiles/calibration/rngd_card_edf.yaml
```

Swap `rngd-llama31-8b-tp8.json` / `--hardware RNGD` /
`--bucket sharegpt-llama31-8b-20-tp8` for the per-PE row.
