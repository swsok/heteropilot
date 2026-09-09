# Calibrating `slab3d`'s dim-1 link latency

*`WORK_ORDER_rps_aware.md` rev 2 STEP 2.2. Measured 2026-09-09 on the NPU node,
simulation only, `main` + D28. Raw: `outputs/slab3d_calib/`. Table:
`profiles/calibration/slab3d_latency.yaml`. Driver:
`experiments/scripts/slab3d_calibrate.py`.*

**Result in one line:** the factor is **4 for `[4,2]` and 2 for `[2,2]`,
independent of bandwidth**, and it removes an uncorrected error of **−24.4 %**
(tp8) and **−13.4 %** (tp4) down to **≤ 0.008 %**.

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
| `[4,2]` tp8 | 16.0 | 33.6279 | −24.43 % | −16.81 % | **−0.01 %** | +30.44 % |
| `[4,2]` tp8 | 35.2 | 32.3079 | −22.84 % | −14.92 % | **−0.01 %** | +32.59 % |
| `[4,2]` tp8 | 100.0 | 32.1967 | −23.68 % | −15.73 % | **−0.01 %** | +31.61 % |
| `[2,2]` tp4 | 16.0 | 19.7750 | −12.95 % | **+0.00 %** | +25.23 % | +76.76 % |
| `[2,2]` tp4 | 35.2 | 18.8832 | −13.57 % | **+0.00 %** | +27.17 % | +81.10 % |
| `[2,2]` tp4 | 100.0 | 18.5235 | −13.79 % | **+0.00 %** | +27.37 % | +82.68 % |

Bold is the fitted factor. Residuals: 0.0067–0.0075 % for `[4,2]`, 0.0009–0.0016 %
for `[2,2]`.

**The spike is reproduced.** It reported 4× at `[4,2]` / bw 16 and a 24.4 %
accuracy cost; this run gives factor 4 and **−24.43 %** uncorrected at that exact
point. That is the regression the work order asked for.

## Three things the table says

**1. The factor does not depend on bandwidth.** 4 at all three bandwidths for
`[4,2]`, 2 at all three for `[2,2]`. This is a prediction of §1.2 rather than a
surprise — the latency term contains no bandwidth — but it is worth having
measured, because *one* bandwidth-independent factor zeroing the gap at three
bandwidths is also empirical evidence that the **bandwidth terms are equal**, which
§1.2 asserted analytically and nothing had yet checked.

**2. The factor does depend on the split**, and that is why this is a table. 4 for
`[4,2]`, 2 for `[2,2]`.

**3. Both values equal `tp/2`, and that is derivable — which is not the same as
measured.** A ring allreduce over N ranks costs `2(N−1)` hops. Flat tp=8 is 14; the
split is `2(4−1) = 6` in dim 0 plus `2(2−1) = 2` in dim 1, so dim 1 must carry
`(14 − 6)/2 = 4` hops' worth. At tp=4: 6 flat against `2 + 2`, so `(6 − 2)/2 = 2`.
In general `factor = (tp−1) − (tp/2−1) = tp/2`.

The arithmetic explains both entries exactly, and the near-zero residuals say the
model is a hop count and nothing else. **It is still not a licence.** The work
order is explicit — *if the factor varies with (split, bw), write a table, not a
law* — and it does vary with the split. `validity.extrapolation: refuse` covers
every combination not in the table, so a `tp16` split (predicted 8) must be
measured before it is used, not assumed.

## What this licenses, and what it does not

**Licensed:** a `slab3d` plan at `[4,2]` or `[2,2]` and one of the three measured
bandwidths may quote absolute TPOT, because the corrected residual is ≤ 0.008 % —
two orders of magnitude below the RNGD decode profile's own −3.1 % error, which
remains the binding term.

**Not licensed:**

- any other split or bandwidth. STEP 2.3 must reject those as
  `OUTSIDE_CALIBRATION_DOMAIN` rather than simulate them with a guessed factor.
- **TTFT and throughput.** Only TPOT p50 was fitted. Prefill collectives differ in
  size and count, and nothing here measures them.
- **P/D configurations.** The fit is single-instance and colocated by construction.
  A real `slab3d` P/D run also carries the prefill instance's compute→sender
  COMM_SEND across dim 1, which this experiment deliberately excludes so the
  allreduce could be isolated. Whether the same factor serves that path is
  **unmeasured**, and `docs/d14_spike.md`'s caveat on absolute tok/J for asymmetric
  P/D stands until STEP 2.3 checks it.
