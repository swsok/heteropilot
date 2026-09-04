# The RNGD concurrency envelope, measured to c128

*Measured 2026-08-31 on the NPU node, RNGD npu0 (TP=8, card-as-device), by
`experiments/scripts/rebuild_rngd_bundle_from_edf.py collect`. Raw:
`outputs/rngd_envelope/edf/real_c{16,32,64,128}.json`. Executes
`docs/npu_concurrency_envelope_work_order.md`.*

## The short answer

**The card serves well beyond c76 — but not as fast as the simulator thinks, and
its TPOT at that load is far worse than predicted.** Outcome **2** of the work
order's three, with the correction factor smaller than it guessed on throughput
and a second, unguessed error on latency.

And the premise needed fixing first: **the committed c1–c32 curve was
request-pool-limited, not hardware-limited.** Its top point never ran at 32.

## The measured envelope

Pool sized to keep itself out of the way; `eff` is average in-flight concurrency
by Little's law (`Σ latency / wall`), which is what the card actually served:

| requested | pool | wall (s) | **eff. conc** | **output tok/s** | TTFT avg | TTFT p99 | TPOT avg | TPOT p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | 146.7 | 15.3 | 585.8 | 288.5 | 1925.1 | 25.71 | 29.46 |
| 32 | 128 | 94.5 | 29.3 | 908.6 | 725.0 | 3848.6 | 31.18 | 35.52 |
| 64 | 256 | 130.8 | 59.2 | 1277.0 | 1173.5 | 6901.1 | 44.54 | 47.99 |
| 128 | 300 | 132.8 | **107.2** | **1473.3** | 3260.6 | 14009.5 | 67.88 | 78.22 |

All 812 requests succeeded; zero failures at any level.

Marginal scaling exponent, on **effective** concurrency:

| interval | concurrency | throughput | exponent |
| --- | ---: | ---: | ---: |
| c16 → c32 | ×1.92 | ×1.55 | 0.675 |
| c32 → c64 | ×2.02 | ×1.41 | 0.485 |
| c64 → c128 | ×1.81 | ×1.15 | **0.241** |

The curve is flattening decisively. It had *not* flattened by the old top point,
which is why extrapolation from there was unsafe in both directions.

## Two errors in the simulator, at the load the sweep actually uses

Interpolating between the c64 and c128 points to the eff = 76 the card-fixture
winner implies:

| at eff 76 | measured (interpolated) | simulator | simulator is |
| --- | ---: | ---: | --- |
| output tok/s per card | **1346** | 1767 | **1.31× optimistic** |
| TPOT | **52.7 ms** | 43.2 ms | **18 % optimistic** |

**Throughput: 1.31×, not the 1.6× the work order predicted.** Its estimate came
from a 0.598 exponent computed over a concurrency ratio the pool made impossible
(see below); the honest figure from properly-offered load is 1.31×.

**TPOT is the finding the work order asked for and did not expect to get.** It
predicted the measured trend would extrapolate to ~37 ms at c76 and said that if
c64/c128 diverged, "the decode model — currently the accurate half of the card
profile at −3.1 % — has a limit that has not been found yet." It diverges:
measured TPOT is **44.54 ms at eff 59** — already past the simulator's c76
prediction of 43.2 — and **67.88 ms at eff 107**. The decode model is accurate at
the concurrency it was fitted on and degrades above it. That limit is now found.

## The old curve's top point was never c32

The committed points all used a **24-request pool** while requesting up to 32
concurrent, so the pool bound the experiment:

| committed point | requested | pool | **eff. conc** | tok/s |
| --- | ---: | ---: | ---: | ---: |
| c16 | 16 | 24 | 12.2 | 427.4 |
| c32 | 32 | 24 | **21.2** | 646.4 |

So "the highest concurrency ever run on RNGD is 32" was **~21**, and the two
figures the work order builds on are both low. Re-measured at a non-binding pool:

| point | committed (pool 24) | re-measured (pool 128) | change |
| --- | ---: | ---: | ---: |
| c16 | 427.4 tok/s | 585.8 | **+37 %** |
| c32 | 646.4 tok/s | 908.6 | **+40 %** |

TPOT is essentially unchanged at c32 (30.14 → 31.18 ms), so this is real
throughput the harness was leaving on the table, not a bookkeeping difference.
TTFT *falls* with the larger pool (1894 → 725 ms at c32) for the same reason:
with a 24-request pool every request is admitted at once and their prefills
contend, while a 128-request pool holds in-flight at the requested 32.

Consequently the work order's exponent is wrong twice over: it read the c16 → c32
interval as a concurrency doubling when the pool made it ×1.74, and both endpoints
were depressed. Recomputed on effective concurrency at a proper pool it is 0.675
over that interval, not 0.598.

## What these numbers cannot be used for

**TTFT here is not comparable to the sweep's p99 TTFT.** The bench ignores
`arrival_time_ns` and fires the whole pool at once (deviations D19) — a closed-loop
saturation probe, which is the right instrument for a throughput envelope and the
wrong one for an open-loop arrival process. The sweep offers Poisson arrivals at
9.9 rps. So the 14 s p99 TTFT at c128 above does **not** refute the card fixture's
480 ms winner; it is a different experiment. Throughput and TPOT are the valid
comparisons, and they are the two in the table above.

**Only npu0 was measured**, TP=8 card-as-device, one artifact
(`furiosa-ai/Llama-3.1-8B-Instruct`), one dataset
(`workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl`). The 300-line dataset caps the
pool, so c128 ran at 2.3× headroom and reached eff 107 rather than 128 — the one
point where the pool is still slightly binding. A larger trace would push it
further; the flattening exponent suggests there is not much left to find.

## What this changes

1. **The card-fixture results are optimistic on throughput by ~1.31× and on TPOT
   by ~18 %** at the concurrency they assume. Per work order §4.4 these are **not**
   silently rescaled — the discrepancy is recorded here and in deviations, and
   `pd_slo_sweep.py` should be re-run at a defensible load before its numbers are
   quoted again.

   **That re-run is still open, and the obvious way to do it does not work.**
   Lowering the arrival rate to bring per-card concurrency down was attempted on
   2026-09-01 at 3.3 rps and abandoned after 3.7 hours: **no RNGD-involving
   candidate terminated** (0 of 24 RNGD P/D and 0 of 12 cross-vendor, against 60 of
   84 CUDA-only), because a slower arrival rate gives the decode scheduler less to
   batch, which lowers throughput, which lengthens the simulated time needed to
   drain the trace. It would also not have produced the intended load: the
   `minimize_active_accelerators` objective shrinks the fleet as load falls, pushing
   per-card concurrency back up. **Hold the rate at 10 rps and constrain the fleet
   instead** — give the RNGD arm enough cards that per-card concurrency lands in the
   directly measured band (eff 15.3–59.2 above). Deviations D22 has the full
   account.
2. **The envelope itself is no longer the largest open risk.** The card does reach
   eff 107 with zero failures, so c76 is inside what the hardware serves. The risk
   moved from "can it?" to "at what cost?", and the cost is now measured.
3. **The correction is a throughput/latency model error, not a calibration
   offset**, exactly as the work order anticipated — it varies with concurrency
   (exponent 0.675 → 0.241), so a single scalar cannot express it. It belongs as a
   profile-level caveat with the curve attached, which is this file.

## Reproduce

```bash
ART=~/.cache/huggingface/hub/models--furiosa-ai--Llama-3.1-8B-Instruct/snapshots/231d94fbc03cdd66aaeb2411697064a45f008ec7
for spec in "16 128" "32 128" "64 256" "128 300"; do set -- $spec
  PYTHONPATH=$PWD /usr/bin/python3 experiments/scripts/rebuild_rngd_bundle_from_edf.py collect \
      --artifact "$ART" --card 0 --port 8020 --concurrency $1 --num-reqs $2 \
      --out outputs/rngd_envelope
done
```

System `python3`, not `.venv`. `--out` is deliberately **not**
`outputs/rngd_edf_bundle`: `collect` writes `real_c<N>.json` by concurrency alone,
so re-running c16 or c32 there would overwrite the committed points.

Pre-flight, because the device set has now changed four times (4 → 3 → 4 → 3
cards): `bash scripts/whichnode.sh`, then check `furiosa-smi info` for which cards
exist and `alloc_status` for free PEs. `--card N` is the **physical** npu number
(`--devices npu:N:*`), so with npu2 absent `--card 2` is invalid.
