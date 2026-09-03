# The card fixture's RNGD winner does not survive its own measured error

*Swept 2026-09-01/02 on the NPU node. `experiments/scripts/pd_slo_sweep.py` at the
original 10 rps with `--tpot-margin-percent 18`, the TPOT optimism D22 measured at
the concurrency these plans run at. Artifacts:
`outputs/pd_slo_sweep_margin18/{pd-rngd-gpu-card,pd-rngd-gpu}.json`.
This is the re-run D22 §4.4 asks for.*

## The short answer

**Every RNGD configuration on the card fixture is rejected once the measured
decode-model error is applied.** The committed winner is not merely optimistic —
it is infeasible, and the "RNGD wins on energy at loose TTFT" half of the
three-regime answer does not survive.

The result does not depend on the size of the margin: the winner clears the 50 ms
TPOT SLO by 1.59 ms, so **anything above 3.3 % rejects it**, and the profile's own
agreement at its fitted concurrency is −3.1 %.

## What changed

| | committed (no margin) | with 18 % TPOT margin |
| --- | --- | --- |
| card fixture, TTFT ≤ 64 s | `agg[furiosa:tp1]` n=2, **3.164 tok/J**, p99 TTFT 480 ms | `agg[cuda:tp4]` n=4, **2.595 tok/J**, p99 TTFT 15,070 ms |
| tp4 fixture, TTFT ≤ 64 s | `agg[furiosa:tp8]` n=8, **4.956 tok/J**, p99 TTFT 29,548 ms | `agg[cuda:tp4]` n=4, **2.595 tok/J**, p99 TTFT 15,070 ms |

Both fixtures converge on the same A40 plan, because in both the RNGD arm is
rejected and the same surviving CUDA configuration is left. **The conclusion is
fixture-independent**, which is worth more than either fixture alone: it does not
depend on whether an RNGD accelerator is modelled as a card or as 8 PEs.

## Why every RNGD configuration fails

Computed directly from the completed simulations, p99 over the repo's own
`planner.util.percentile`. SLOs: TPOT p99 ≤ 50 ms, TTFT p99 ≤ 25 s.

**Two RNGD cards:**

| config | p99 TPOT | ×1.18 | p99 TTFT | verdict |
| --- | ---: | ---: | ---: | --- |
| s256-t8192 ← **the committed winner** | 48.41 | **57.12** | 480 | reject |
| s256-t2048 | 49.58 | 58.50 | 673 | reject |
| s128-t8192 | 47.62 | 56.19 | 4,589 | reject |
| s128-t2048 | 49.07 | 57.90 | 4,307 | reject |
| s32-t8192 | 32.83 | 38.74 | **56,766** | reject |
| s32-t2048 | 32.78 | 38.68 | **57,972** | reject |

**One RNGD card** fails the same way, more sharply: s32 clears TPOT at 32.79 ms
but takes 148,739 ms to first token.

**The arm is squeezed between the two SLOs.** Raising per-instance concurrency
(s128, s256) meets TTFT and breaks TPOT; lowering it (s32) meets TPOT and breaks
TTFT by 2.3×. There is no setting in between that satisfies both, and that is a
property of the device's measured throughput/latency curve, not of the margin:
the s32 rows fail TTFT at **any** margin, including zero.

## What this does NOT show

**The tight-TTFT rows are not determined by this run, and must not be read as
infeasible.** The sweep printed INFEASIBLE at TTFT ≤ 8 s and ≤ 500 ms, but that is
an artifact of a timeout I lowered, not a result:

- I set `--timeout 1080` after measuring that every successful card-fixture
  simulation finished within 14.9 minutes. That was true of the card fixture and
  **wrong for the tp4 fixture**, where all 72 `pd_cuda-a40-tp4` candidates timed
  out against the committed run's 1800 s.
- One of those is the committed tight-TTFT winner, `P[cuda:tp4] D[cuda:tp4]`, at
  p99 TPOT **37.27 ms**. 37.27 × 1.18 = 43.98 ms, which **passes**. So that regime
  is probably unchanged by the margin — but this run cannot say so, because the
  candidate was never evaluated.

Settling the tight-TTFT regime needs a re-run at `--timeout 1800` or higher. The
loose-TTFT finding above is unaffected: **zero RNGD candidates timed out** on the
card fixture (54 attempted, 0 timeouts), so the rejection rests entirely on
completed simulations.

## Method, and what was reduced

| | committed run | this run |
| --- | --- | --- |
| arrival rate | 10 rps | 10 rps (unchanged — see D22 on why lowering it fails) |
| TTFT points | 8 | **3** (500 / 8,000 / 64,000) |
| `--timeout` | 1800 s | **1080 s** — see above, too low for tp4 |
| TPOT margin | none | **18 %** |

The TTFT grid was cut from 8 points to 3 because timeouts are not cached, so every
timing-out candidate is retried once per point; 3 points brackets the tight,
middle and loose regimes at 3/8 of the retry cost. The committed card-fixture
table has the same winner at all 8 points, so the reduction does not hide a
transition there.

The 18 % is not a guess. It is the TPOT optimism measured at the concurrency the
card-fixture winner runs at (D22, `rngd_concurrency_envelope.md`). Applying it
uniformly is **conservative for the A40 arm**, which is validated to ~2 % — so the
A40 plan that wins here wins despite being penalised as if it shared RNGD's error.

## Reproduce

```bash
for fx in pd-rngd-gpu-card pd-rngd-gpu; do
  PYTHONPATH=$PWD .venv/bin/python experiments/scripts/pd_slo_sweep.py \
      --service examples/service_specs/llama31-8b.yaml \
      --cluster experiments/configs/clusters/$fx.yaml \
      --ttft-ms 500,8000,64000 --num-requests 300 --seed 42 --workers 32 \
      --timeout 1800 \
      --tpot-margin-percent 18 \
      --output-dir outputs/.hp-slo-margin18-$fx
done
```

Note `--timeout 1800`, not the 1080 this run used — see "What this does NOT show".
