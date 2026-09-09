# Calibrating `slab3d`'s dim-1 link latency

*`WORK_ORDER_rps_aware.md` rev 2 STEP 2.2. Measured 2026-09-09 on the NPU node,
simulation only, `main` + D28. Raw: `outputs/slab3d_calib/`. Table:
`profiles/calibration/slab3d_latency.yaml`. Driver:
`experiments/scripts/slab3d_calibrate.py`.*

**Result in one line:** the factor is **4 for `[4,2]`, 2 for `[2,2]` and 1 for
`[1,2]`, independent of bandwidth across five values spanning 13×**, and it removes
an uncorrected error of **−22.6 to −24.4 %** (tp8) and **−12.6 to −13.8 %** (tp4)
down to **≤ 0.008 %**.

## What needs correcting, and why it is only the latency

`slab3d` turns a flat ring allreduce over `tp` ranks into a hierarchical
`tp/2 × 2`. `WORK_ORDER_spikes.md` §1.2 works out that the **bandwidth term is
identical** in both — reduce-scatter plus all-gather is `2·(7/8)·size/bw` flat, and
`3/4 + (1/4)(1/2) = 7/8` hierarchically — so only the **latency term** moves, from
7 ring steps to 4 at tp=8.

That matters because the RNGD tp8 profile's `link_latency` was fitted against a
flat ring (`PROJECT_REPORT.md` §4.8.5, 115 µs per decoder layer). Encoding the same
group in 3-D therefore makes decode look faster than the hardware is, and the error
is not small.

## Method

One colocated instance, 20 requests, same model and same trace on both sides. The
only difference is the topology:

| | dims | `tp_dim` | allreduce |
| --- | --- | --- | --- |
| flat | `[tp]` | — | one ring of `tp` |
| split | `[tp/2, 2]` | `[T, T]` | `tp/2` in dim 0, then 2 in dim 1 |

`topology_mode: split2` produces the split side. **It is a calibration instrument,
not a deployment mode**: STEP 2.1 deliberately left it out of `main` because it
describes no real placement, and 2.2 needs it back because splitting one TP group
is the only way to isolate the term under test — a real `slab3d` placement changes
the instance mix as well as the split, so the difference would not be attributable.
It refuses anything but one colocated non-MoE instance, and
`tests/test_slab3d_config.py` asserts `planner/` never emits it.

Dim 0 keeps the calibrated per-hop latency; dim 1 carries the factor under test.
Six `(split, link_bw)` points, factors {1, 2, 4, 8}, 30 runs.

## The full sweep

TPOT p50 of the split configuration against the flat reference:

| split | link_bw | flat p50 ms | f=1 | f=2 | f=4 | f=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[4,2]` tp8 | 7.7 | 35.1389 | −21.94 % | — | **−0.01 %** | — |
| `[4,2]` tp8 | 12.6 | 34.0727 | −22.55 % | — | **−0.01 %** | — |
| `[4,2]` tp8 | 16.0 | 33.6279 | −24.43 % | −16.81 % | **−0.01 %** | +30.44 % |
| `[4,2]` tp8 | 35.2 | 32.3079 | −22.84 % | −14.92 % | **−0.01 %** | +32.59 % |
| `[4,2]` tp8 | 100.0 | 32.1967 | −23.68 % | −15.73 % | **−0.01 %** | +31.61 % |
| `[2,2]` tp4 | 7.7 | 20.8556 | −12.09 % | **+0.00 %** | — | — |
| `[2,2]` tp4 | 12.6 | 19.9994 | −12.60 % | **+0.00 %** | — | — |
| `[2,2]` tp4 | 16.0 | 19.7750 | −12.95 % | **+0.00 %** | +25.23 % | +76.76 % |
| `[2,2]` tp4 | 35.2 | 18.8832 | −13.57 % | **+0.00 %** | +27.17 % | +81.10 % |
| `[2,2]` tp4 | 100.0 | 18.5235 | −13.79 % | **+0.00 %** | +27.37 % | +82.68 % |

Bold is the fitted factor; the 12.6 rows show only the two cells that matter (the
full sweep is in `outputs/slab3d_calib/fits.json`). Residuals: 0.0064–0.0075 % for
`[4,2]`, 0.0009–0.0016 % for `[2,2]`.

`[1,2]` (tp2) is the third split, measured at all five bandwidths: **factor 1,
residual exactly 0.0000**. Flat tp2 costs `2(2−1) = 2` hops and the split costs
`2(1−1) = 0` in dim 0 plus `2` in dim 1, so there is nothing to correct — and
`tp/2 = 1` predicts precisely that.

**Three of the fifteen points were not in the work order's list, and had to be.** The named bandwidths are 16 (the
spike), 35.2 (the composed A40↔A40 value) and 100 (NVLink-class), and the named
splits are `[4,2]` and `[2,2]`. Running `plan --enable-pd` on
`pd-rngd-gpu.yaml` refused candidates for three points none of them covered:

| point | what it is | candidates refused |
| --- | --- | ---: |
| `[4,2]`, `[2,2]` @ **12.6** | `fabric-rngd0-a40*`, the A40↔RNGD cross-vendor link | all 84 asymmetric |
| `[4,2]` @ **7.7** | `fabric-rngd0-rngd1`, RNGD↔RNGD — the asymmetric NPU P/D case D14/D16(b) was written about | 12 |
| **`[1,2]`** @ 35.2 | an A40 tp1 prefill with a tp2 decode | 24 |

35.2 is the A40↔A40 value; the cross-vendor path is 12.6. That is the refusal doing
its job — the fix is to measure each point, never to widen the domain by fiat.

**The spike is reproduced.** It reported 4× at `[4,2]` / bw 16 and a 24.4 %
accuracy cost; this run gives factor 4 and **−24.43 %** uncorrected at that exact
point. That is the regression the work order asked for.

## Three things the table says

**1. The factor does not depend on bandwidth.** 4 at all five bandwidths for
`[4,2]`, 2 for `[2,2]`, 1 for `[1,2]`, over a **13× range** (7.7 to 100). This is a
prediction of §1.2 rather than a surprise — the latency term contains no bandwidth
— but it is worth having measured, because *one* bandwidth-independent factor
zeroing the gap at five bandwidths is also empirical evidence that the **bandwidth
terms are equal**, which §1.2 asserted analytically and nothing had yet checked.

**2. The factor does depend on the split**, and that is why this is a table. 4 for
`[4,2]`, 2 for `[2,2]`, 1 for `[1,2]`.

**3. All three values equal `tp/2`, and that is derivable — which is not the same
as measured.** A ring allreduce over N ranks costs `2(N−1)` hops. Flat tp=8 is 14; the
split is `2(4−1) = 6` in dim 0 plus `2(2−1) = 2` in dim 1, so dim 1 must carry
`(14 − 6)/2 = 4` hops' worth. At tp=4: 6 flat against `2 + 2`, so `(6 − 2)/2 = 2`.
At tp=2: 2 flat against `0 + 2`, so `(2 − 0)/2 = 1`. In general
`factor = (tp−1) − (tp/2−1) = tp/2`.

The arithmetic explains all three entries exactly, and the near-zero residuals say the
model is a hop count and nothing else. **It is still not a licence.** The work
order is explicit — *if the factor varies with (split, bw), write a table, not a
law* — and it does vary with the split. `validity.extrapolation: refuse` covers
every combination not in the table, so a `tp16` split (predicted 8) must be
measured before it is used, not assumed.

## What this licenses, and what it does not

**Licensed:** a `slab3d` plan at one of the three measured splits and one of the
five measured bandwidths may quote absolute TPOT, because the corrected residual is ≤ 0.008 % —
two orders of magnitude below the RNGD decode profile's own −3.1 % error, which
remains the binding term.

**One mapping is assumed, not measured.** The fit is done at **two** dims
(`[tp/2, 2]`, a single instance) and applied at **three** (`[g, 2, n_slabs]`, a P/D
pair), with dim 1 — the 2-wide axis — taken to be the same axis in both. The slab
axis keeps the base latency. That correspondence is what the encoding means, and
the hop-count derivation supports it, but no measurement here distinguishes it from
alternatives.

**Not licensed:**

- any other split or bandwidth. STEP 2.3 rejects those as
  `OUTSIDE_CALIBRATION_DOMAIN` rather than simulating them with a guessed factor.
- **TTFT and throughput.** Only TPOT p50 was fitted. Prefill collectives differ in
  size and count, and nothing here measures them.
- **P/D configurations.** The fit is single-instance and colocated by construction.
  A real `slab3d` P/D run also carries the prefill instance's compute→sender
  COMM_SEND across dim 1, which this experiment deliberately excludes so the
  allreduce could be isolated. Whether the same factor serves that path is
  **unmeasured**, and `docs/d14_spike.md`'s caveat on absolute tok/J for asymmetric
  P/D stands until STEP 2.3 checks it.
