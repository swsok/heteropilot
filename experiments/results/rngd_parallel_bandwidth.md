# The NPU leg of the cross-vendor KV path, measured across a card's 8 PEs

*Measured 2026-08-27 on the NPU server, npu3 (`rngd:16..23`), by
`experiments/scripts/rngd_device_facts.py --parallel-bandwidth`. Raw data:
`outputs/rngd_profile/parallel_bandwidth.json`. Closes the gap stated in
`docs/npu_parallel_bandwidth_work_order.md`.*

## The short answer

**The scaling law reproduces. The absolute level does not — it was quoted as a
peak, and a peak is the wrong statistic for a KV handoff.**

| streams | H2D sustained (GB/s) | trial min–max | scaling vs ideal | previously quoted | ratio |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 1 | 3.77 | 3.76 – 4.14 | 1.000 | 5.06 | 0.745 |
| 2 | 7.60 | 7.44 – 8.16 | 1.008 | 10.39 | 0.731 |
| 4 | 15.36 | 15.02 – 16.03 | 1.019 | 19.10 | 0.804 |
| 8 | **26.27** | 26.13 – 27.09 | **0.871** | 35.47 | 0.741 |

Median of 3 independent trials, 256 MB per PE, 5 s sustained window, pageable
host memory. `h2d` is the measurement; see "d2h is not a link figure" below.

Two things to read off this table:

1. **Near-linear scaling across a card's 8 PEs is confirmed.** 87.1 % of ideal at
   8 streams, against the 88 % that was claimed. That the two agree to under a
   point, from an independent implementation and a different statistic, is the
   strongest evidence that the original figures were real measurements of a real
   effect.
2. **Every absolute figure was ~25 % high**, uniformly across all four stream
   counts. That uniformity is itself the clue: a hardware or placement change
   would not scale all four points by the same factor.

## The 1-stream cross-check failed, and that is the finding

Acceptance criterion §5.2 of the work order asks the `streams: 1` row to land near
the committed single-stream figure of 5.03 GB/s, and says that if it does not,
that is a finding about the method rather than a number to paper over. It did not:
**3.77 against 5.03, a 25 % shortfall.**

It is not the hardware. Re-running the *committed* best-of-N method on the same
device on the same day reproduces the old number exactly:

```
256 MB  H2D  5.06 GB/s (best)  5.03 (median)
```

(`outputs/rngd_profile/host_bandwidth_recheck.json`, committed as evidence.) That
is 5.06 and 5.03 against the 5.06 in the prose and the 5.03 in
`outputs/rngd_profile/host_bandwidth.json` — the link is unchanged.

So the gap is entirely in how the number is taken. Decomposing it, one PE, 256 MB:

| variant | best | median | mean |
| :--- | ---: | ---: | ---: |
| A — committed method: `.to()` timed, `.cpu()` between each | 5.04 | 5.01 | 5.01 |
| B — back-to-back `.to()` only, no intervening transfer | 5.09 | **4.15** | 4.14 |
| C — sustained back-to-back over a 5 s window | — | — | **3.67** |

Two independent effects, each worth about 20 %:

- **A → B: the gap between transfers.** A's `.cpu()` costs ~160 ms at 256 MB, and
  the device-side buffer free and recycle happen inside it. Remove that idle time
  and the *typical* transfer drops 5.01 → 4.15, while the *best* one is unchanged
  at ~5.05. The card can still hit 5 GB/s; it just cannot hit it back to back.
- **B → C: best-of-N versus sustained.** Taking the minimum of 7 timings picks the
  luckiest transfer. Over a 5 s window the slow tail is included and the aggregate
  falls to 3.67.

**For a KV handoff, C is the right statistic.** A prefill→decode KV transfer is a
sustained bulk copy with nothing interleaved, which is variant B/C, not variant A.
The work order argued this on first principles before the measurement; the
measurement now quantifies what it costs to get it wrong: **26 %**.

## What this does and does not change

**Does not change the qualitative conclusion.** 26.27 GB/s across a TP=8 decode
island is still far above the ~10 GB/s P/D adoption crossing that Exp 3 found, so
the NPU leg is still not the bottleneck, and near-linear PE scaling still holds.
Nothing in `pd_slo_sweep.md`'s three-regime answer depends on 35 vs 26.

**Does change every composed number.** `docs/npu_parallel_bandwidth_work_order.md`
§1 records that both P/D fixtures' `fabric-*` links were to be composed as
`1/(1/gpu + 1/npu)` using the GPU leg measured on branch
`feat/gpu-host-bandwidth`. With the corrected NPU leg those compositions move:

| fixture | link | old composition | new composition | GB/s: old → new |
| --- | --- | --- | --- | ---: |
| `pd-rngd-gpu-card.yaml` (tp1) | rngd ↔ a40 | 1/(1/26.03 + 1/35.47) | 1/(1/26.03 + 1/26.27) | 15.0 → 13.07 |
| | rngd ↔ rngd | 1/(1/35.47 + 1/35.47) | 1/(1/26.27 + 1/26.27) | 17.7 → 13.13 |
| `pd-rngd-gpu.yaml` (tp4) | rngd ↔ a40 | 1/(1/62.42 + 1/19.10) | 1/(1/62.42 + 1/15.36) | 14.6 → 12.33 |
| | rngd ↔ rngd | 1/(1/19.10 + 1/19.10) | 1/(1/15.36 + 1/15.36) | 9.6 → 7.68 |

**These have not been applied, and the reason is that the branch they belong to
does not exist here.** `feat/gpu-host-bandwidth` is not on `origin`, and
`experiments/results/gpu_host_bandwidth.md` is not in the tree. On this branch both
fixtures still carry `bandwidth_gbps: 35, source: placeholder` — the state that
predates the GPU leg. The GPU figures above (26.03 / 62.42 / 79.98) are quoted
from the work order's prose and are themselves uncommitted here, so composing
against them would repeat exactly the error this branch exists to fix.

The last row is the one to watch: `rngd ↔ rngd` at tp4 falls from 9.6 to **7.68
GB/s**, which crosses below the ~10 GB/s P/D adoption threshold. Whoever merges
the GPU leg should re-run both SLO sweeps rather than assume the regimes hold.

## d2h is not a link figure

Recorded for continuity only: 1.61 / 3.52 / 7.10 / 14.04 GB/s at 1 / 2 / 4 / 8
streams. The `d2h` loop calls `.cpu()`, which allocates a fresh pageable
destination every iteration; above ~16 MB PyTorch's CPU allocator stops reusing a
cached block and mmaps instead, so the figure is allocator-bound rather than
link-bound. The same artifact is documented on the GPU side of this path. Do not
quote it as a link bandwidth.

## Method

One process per PE, not threads: nothing establishes that `furiosa.torch` allows
several PE contexts in one interpreter, and subprocess-per-PE is the pattern
`start_load()` already proves on this hardware. Workers allocate a contiguous
pageable bfloat16 buffer, warm once, report `READY`, then spin until an absolute
`time.time()` instant broadcast by the parent — `time.time()` is comparable across
processes and `perf_counter` is not. Aggregate is total bytes across all workers
over the concurrent window, first start to last finish, which is the definition
the GPU leg uses so the two compose. Workers are respawned per trial because host
buffer NUMA placement is fixed at allocation and is not controlled.

**Concurrent 8-PE access from separate processes works.** This was the failure mode
the work order §6 flagged as most likely, since the code had never run on RNGD
hardware. All 8 workers reached `READY` and `furiosa-smi ps` showed 8 live
contexts on npu3. No trial was dropped at any stream count.

### Device enumeration changed, and the script was mislabelling the card

The work order says to run on `rngd:24` (npu3) and that npu0/1/2 are held by
another tenant's pods. Neither is true as of 2026-08-27 07:30 UTC:

- **npu2 (PCI `44:00.0`) is gone from the PCI bus entirely** — no sysfs node, no
  `furiosa-smi` row. Three cards remain: npu0, npu1, npu3.
- **No pods hold anything.** Every PE on all three cards reports an empty
  `alloc_status`, and `furiosa-smi ps` is empty.
- **Torch numbers rngd devices densely over the cards that are present.** With
  npu2 missing, `rngd:16..23` is npu3 and `rngd:24` does not exist — it fails with
  `Expected allocator != nullptr`. Confirmed by pinning a load to `rngd:16` and
  reading `furiosa-smi ps`, which reported `npu3:0`.

`card_of()` computed the card as `index // 8`, which would have stamped this run's
artifact `npu2` — a card that is not in the machine. It now resolves through the
live sysfs enumeration instead, and falls back to the arithmetic only when sysfs
is unreadable. All three present cards are on NUMA node 0, as was npu2, so the
old and new runs are not separated by placement.

## Reproduce

```bash
PYTHONPATH=$PWD python3 experiments/scripts/rngd_device_facts.py \
    --device rngd:16 --parallel-bandwidth --streams 1,2,4,8 \
    --parallel-size-mb 256 --parallel-duration-s 5 --parallel-trials 3 \
    --out outputs/rngd_profile/parallel_bandwidth.json
```

System `python3`, not `.venv` — the vendor runtime lives in the system
site-packages. Adjust `--device` to the first PE of whichever card is free and
re-check the enumeration first; it has changed once already.
