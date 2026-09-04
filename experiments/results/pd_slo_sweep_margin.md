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

> **The re-run was done 2026-09-03 and the diagnosis above is wrong** — not about
> the loose-TTFT finding, which stands, but about the cause. `--timeout` was never
> the binding constraint: these candidates *livelock*. See
> "Tight-TTFT regime" below. The paragraph is kept because it records what was
> reasonable to believe from a 1080 s run.

## Tight-TTFT regime (re-run 2026-09-03, `--timeout 1800`)

*Artifacts: `outputs/pd_slo_sweep_margin18/tight/`. Ran 06:10:10Z-10:47:08Z on the
NPU node, both fixtures, `--ttft-ms 500,8000` only (the loose point was already
settled above), `--workers 64`, same service spec, seed and 300-request trace.*

**The tight-TTFT regime is still undetermined, and now for a reason no timeout can
fix.** All four points printed INFEASIBLE, and all four must again be read as
"not evaluated" rather than "rejected".

| fixture | TTFT SLO | verdict | simulations | completed | **timed out** |
| --- | ---: | --- | ---: | ---: | ---: |
| `pd-rngd-gpu-card` | ≤ 8 s | INFEASIBLE | 222 | 151 | **71** |
| `pd-rngd-gpu-card` | ≤ 500 ms | INFEASIBLE | 222 | 151 | **71** |
| `pd-rngd-gpu` (tp4) | ≤ 8 s | INFEASIBLE | 252 | 126 | **126** |
| `pd-rngd-gpu` (tp4) | ≤ 500 ms | INFEASIBLE | 252 | 126 | **126** |

Full lists: `tight/timeouts_pd-rngd-gpu-card.txt` (71 entries),
`tight/timeouts_pd-rngd-gpu.txt` (126). These are work-dir signatures, not
candidate ids — the simulator dedups candidates sharing an (island placement,
`max_num_seqs`, `max_num_batched_tokens`) signature, so one line can stand for
several of the 468/496 enumerated candidates. Timeouts are not cached and are
retried once per TTFT point.

Timeouts by family:

| family | card | tp4 |
| --- | ---: | ---: |
| `pd_cuda-a40` | 29 | 30 |
| `mix_cuda-a40` | 24 | 36 |
| `pd_furiosa-rngd*` | 12 | 24 |
| `mix_furiosa-rngd` | — | 24 |
| `cuda-a40` (aggregated) | 6 | 6 |
| `furiosa-rngd` (aggregated) | — | 6 |

### The committed winner did not complete, at any timeout

The question this re-run existed to answer: does `P[cuda:tp4] D[cuda:tp4]`, the
committed tight-TTFT winner, still win once its p99 TPOT of 37.27 ms is inflated
by the measured 18 %? **It cannot be answered, because the candidate has still
never been simulated to completion.** All six of its knob variants timed out at
1800 s. The variant that is the committed winner is `-s256-t8192`, identified by
p99 TTFT 371 ms.

Work order §7 permits exactly one escalation — **1 candidate, 3600 s, single
retry** — and it was taken. It also failed: `exit=124`, elapsed 3601 s, no CSV.

### It is a livelock, not a slow simulation

That hour is the useful part of the result. The simulator printed **52,903
progress ticks**, so its simulated clock was advancing the whole time. What never
advanced is the work:

```
Instance[0] (prefill): 1 reqs running at EVERY tick, Waiting 7 -> 299
Instance[1] (decode) : 0 reqs running, 0 waiting, for the entire hour
memory               : flat at 9.304 % / 9.234 %, both instances
```

The prefill instance admits exactly one request and never retires it; the decode
instance is never handed anything; the arrival queue fills with the whole
300-request trace. Evidence, including every distinct state either instance
reported: `tight/retry3600_livelock_evidence.txt`.

This is **not** the memory-saturation deadlock of deviations D12 — memory is flat
at 9 %, and this run had prefix caching disabled. It is a distinct failure, and it
explains the whole history: `--timeout 1080` "timing out all 72 `pd_cuda-a40-tp4`
candidates", 1800 s not helping, and 3600 s not helping either. Raising the
timeout further would not help, so per §7 it was not raised again.

### The same candidate used to finish in 280 seconds

`outputs/.hp-pd-slo/` — an earlier committed sweep of this fixture — has the same
candidate completed and cached:

| run | `link_bw` | `-s256-t8192` |
| --- | ---: | --- |
| `.hp-pd-slo` | 35.0 | **completed in 280.6 s**, p99 TPOT 37.32, p99 TTFT 361.6, 2.206 tok/J |
| this re-run | 35.2 | livelocked past 3600 s |

So this is a regression of at least 12.8x, into non-termination, and it postdates
that run. **The cause is not settled.** The only difference found in the compiled
simulator input is `link_bw` 35.0 -> 35.2 GB/s from the D18 fabric recomputation,
which is a 0.6 % change and an unconvincing explanation on its own; the
intermediate `pd_slo_sweep_measured_fabric` run completed the same candidate, but
its work directory has been cleaned, so its `link_bw` cannot be read back. No
further experiment was run, because this sprint makes no new numbers (work order
§0.3).

This is worth stating plainly against D18's own summary, "the fixtures were
recomputed, and the sweeps cannot see it": that is true of the sweeps' **numbers**
and appears to be false of the simulator's **runtime**.

### Conclusion

Of the three the work order allows — (i) tight regime holds, (ii) tight regime
overturned, (iii) still undetermined — this is **(iii)**, with the reason
upgraded from "the timeout was too short" to "the simulator does not terminate on
these configurations". The committed three-regime table's sub-second row is
neither confirmed nor refuted, and cannot be until the livelock is diagnosed.

### Runtime, predicted and actual (work order instruction 3)

Predicted before the run: 1.5-2.5 h, from a model that priced only the timeout
tail. Actual: **4 h 37 min** (card 1 h 45 m, tp4 2 h 52 m). The estimate was wrong
because it counted timeouts and treated the successful simulations as free; the
candidate population dominates. The card fixture reran with 151 results already
cached and still took 1 h 45 m against the cold run's 2 h 02 m — 86 % of its wall
clock is candidates that never finish.

An earlier attempt was killed 30 minutes into the second fixture because it was
started as a session-owned background job rather than detached; work order
instruction 1 says `nohup`/`tmux` for exactly this reason. Its 84 completed
simulations were lost: the envelope cache is written when a TTFT point joins, so
a run killed mid-point caches nothing.

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
