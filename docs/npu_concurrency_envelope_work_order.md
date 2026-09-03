# Work order — measure the RNGD concurrency envelope beyond c32

*Written 2026-08-28 on the A40 server, to be executed on the **NPU server**.
Branch: `fix/scaling-curve-provenance-and-npu-envelope`. Self-contained: every
number you need is inlined.*

> **EXECUTED 2026-08-31 on the NPU node. Results:
> `experiments/results/rngd_concurrency_envelope.md`, deviations D22.**
>
> Both of this document's premises turned out to be wrong in the same direction,
> and D22 records the retraction: "the highest concurrency ever run on RNGD
> hardware is 32" was a 24-request pool running at an **effective** concurrency of
> 21.2, and the ~1.6x margin below came from an exponent fitted over an interval
> that same pool capped at x1.74. The envelope was then measured to c128: the card
> serves eff 107.2 at 1473 output tok/s with zero failures, so the sweep's 76 is
> inside the hardware's range — but at eff 76 the simulator is 1.31x optimistic on
> throughput and 18 % optimistic on TPOT. Re-running the sweeps with that 18 % as a
> TPOT feasibility margin rejects every RNGD configuration in the loose-TTFT regime
> (`experiments/results/pd_slo_sweep_margin.md`). The tight-TTFT half is still
> undetermined — every `pd_cuda-a40-tp4` candidate timed out — and finishing it is
> STEP 2 of `WORK_ORDER_consolidation.md`.
>
> This file is kept as the historical statement of the gap. Read the two result
> documents above for what is true now; nothing below this banner has been edited.

---

## 1. The gap, stated precisely

Every card-fixture result in `experiments/results/pd_slo_sweep.md` — the three-regime
answer, the 480 ms p99 TTFT winner, the 5.55 rps goodput, the tokens/J that decide
whether heterogeneous P/D pays — rests on the simulator running each RNGD card at
**~76 concurrent sequences**.

**The highest concurrency ever run on RNGD hardware is 32.**

| | output tok/s per card | TPOT | implied concurrent |
| --- | ---: | ---: | ---: |
| real, validation run (burst of 20 @ conc 64) | 584 | 28.4 ms | 16.6 |
| real, highest ever tested (c32) | 646.9 | 30.14 ms | ~32 |
| **sim, the sweep's winner** | **1767** | 43.2 ms | **76** |

The measured scaling curve is committed and solid —
`outputs/rngd_edf_bundle/edf/real_c{1,2,4,8,16,32}.json`, 24 requests each:

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wall (s) | 255.2 | 156.9 | 93.3 | 55.1 | 38.6 | 25.5 |
| output tok/s | 64.6 | 105.1 | 176.8 | 299.4 | 427.4 | **646.9** |
| TTFT mean (ms) | 158.4 | 205.0 | 260.3 | 406.7 | 895.5 | 1894.4 |
| TPOT mean (ms) | 15.26 | 18.59 | 21.41 | 24.14 | 27.24 | 30.14 |

Its marginal scaling exponent from c16 → c32 is **0.598** (throughput ×1.513 for
concurrency ×2). Extrapolated to c76 that gives ~1090 output tok/s per card, where
the simulator assumes 1767 — **~1.6× optimistic, at 2.4× beyond the measured
envelope.**

**Whether that extrapolation or the simulator is right is unknown, and only the
hardware can say.** The curve has not flattened by c32 (still gaining 1.51× per
doubling), so the simulator is not obviously wrong — it is unvalidated.

> **Not a provenance gap.** An earlier draft of `pd_slo_sweep.md` claimed this
> curve was prose-only. It is not; the six JSON files above are committed and
> reproduce the table exactly. That claim was retracted. The gap here is real
> measurement beyond c32, not missing records of what was already measured.

---

## 2. What to run

Extend the existing concurrency sweep to **c64 and c128**, with the same
harness, same trace and same request count so the new points drop straight into
the table above.

### 2.0 Pre-flight — device availability changes

```bash
for n in 0 1 2 3; do for m in 0 1 2 3 4 5 6 7; do
  f="/sys/class/rngd_mgmt/rngd!npu${n}pe${m}/alloc_status"
  [ -e "$f" ] && echo "npu${n}pe${m}: $(cat "$f" 2>/dev/null || echo FREE)"
done; done
```

You need **all 8 PEs of one card** free (the artifact is TP=8). Note from the last
session: **npu2 left the PCI bus entirely** and torch renumbers densely over the
cards that remain, so `rngd:24` may not exist — `rngd:16` was npu3. `card_of()`
now resolves through live sysfs, so artifacts get stamped correctly, but pick
`--card` from what `furiosa-smi info` actually lists.

### 2.1 The run

`experiments/scripts/rebuild_rngd_bundle_from_edf.py collect` already starts and
stops `furiosa-llm serve` once per concurrency and writes `edf/real_c<N>.json`:

```bash
PYTHONPATH=$PWD /usr/bin/python3 experiments/scripts/rebuild_rngd_bundle_from_edf.py collect \
    --artifact ~/.cache/huggingface/hub/models--furiosa-ai--Llama-3.1-8B-Instruct/snapshots/<id> \
    --card <N> --port 8020 --concurrency 64,128 --num-reqs 24 \
    --out outputs/rngd_edf_bundle
```

**Keep `--num-reqs 24`.** The committed points all use 24, and changing it changes
what "concurrency" means relative to the request pool — at c64 with 24 requests the
pool is already exhausted, so **c64 and c128 will both behave as "all 24 at once"
and should give the same wall time.** That is not a wasted run: it establishes the
saturation point directly, and if c64 ≈ c128 ≈ c32 then the card saturates at or
below 32 and the simulator's c76 assumption is refuted outright.

**If they differ from c32, you need a larger pool.** Re-run with `--num-reqs 128`
at concurrency 32, 64, 128 so each level is actually offered that many in-flight
requests. Say in the results file which pool size each point used — a c64 point
taken with a 24-request pool is not comparable to one taken with 128.

### 2.2 The direct check, if the collect path is awkward

`bench_furiosa_endpoint.py` is the simpler instrument and needs no EDF profiling:

```bash
PYTHONPATH=$PWD python3 experiments/scripts/bench_furiosa_endpoint.py \
    --dataset outputs/envcheck/rngd20.jsonl --num-reqs <pool> \
    --concurrency 64 --out outputs/rngd_bench/real_c64.json
```

**It ignores `arrival_time_ns` and fires everything at once** (deviations D19) —
which is exactly what you want here, since the question is a closed-loop
saturation curve. Note it in the results file anyway.

---

## 3. What to do with the answer

Compute output tok/s per card at each new point and extend the table. Then one of
three things is true:

1. **Throughput saturates at or below ~650 tok/s.** The simulator's 1767 at c76 is
   wrong by ~2.7×, every card-fixture goodput and tokens/J figure is optimistic,
   and `pd_slo_sweep.md`'s three-regime answer needs re-deriving at a load the card
   can actually serve. This is the outcome the 0.598 exponent predicts least — the
   curve was still climbing at c32 — but it is the one with the largest
   consequence, so check it first.
2. **Throughput keeps scaling near the 0.598 exponent.** ~1090 tok/s at c76. The
   simulator is ~1.6× optimistic; fit the measured curve and record the correction
   factor as a profile-level caveat rather than a calibration (it is a throughput
   model error, not a latency offset).
3. **Throughput reaches ~1767 tok/s.** The simulator is validated at its own
   operating point and the card-fixture numbers can be quoted with a real envelope
   behind them for the first time.

In every case, **also record TPOT at the new points.** The simulator predicts
43.2 ms at c76; the measured curve gives 30.14 at c32 and a log-ish trend that
extrapolates to ~37 ms. If measured TPOT at c64/c128 diverges from that trend, the
decode model — currently the accurate half of the card profile at −3.1 % — has a
limit that has not been found yet.

---

## 4. Acceptance

1. `outputs/rngd_edf_bundle/edf/real_c64.json` (and `real_c128.json`) committed,
   or `outputs/rngd_bench/real_c64.json` if you used the direct bench.
2. `experiments/results/rngd_concurrency_envelope.md` written: the extended table,
   the pool size used at each point, which of the three outcomes above holds, and
   the recomputed marginal exponent.
3. `experiments/results/pd_slo_sweep.md` and `docs/PROJECT_REPORT.md` §4.8.7
   updated — both currently carry an envelope caveat pointing at this work order.
4. If outcome 1 or 2: **do not silently rescale the committed sweep results.**
   Record the discrepancy the way D18 and D19 did — state what was claimed, what
   was measured, and what it changes — then re-run
   `experiments/scripts/pd_slo_sweep.py` at a defensible load.
5. Gates before the PR: `pytest` (284), `ruff check .`, `mypy planner/`.

---

## 5. Context

| Doc | What it gives you |
| --- | --- |
| `experiments/results/pd_slo_sweep.md` | The results this envelope question hangs over; the rewritten "why the 480 ms winner must not be quoted" block |
| `experiments/results/rngd_ttft_gap_resolved.md` | D19 — the harness bug that made the card profile look 14× worse at TTFT than it is |
| `experiments/results/rngd_edf_bundle_notes.md` | Where the c1–c32 curve came from and how the bundle was built |
| `docs/deviations.md` D18, D19 | The two retractions this project has already made, and the format for a third |
| `docs/HANDOVER_NPU.md` §2 | Environment: system `python3` for vendor runtimes, `.venv` for the planner. Do not cross them |
| `docs/hardware_roadmap.md` "Who holds the NPUs" | Device ownership; re-check, it has changed twice |
