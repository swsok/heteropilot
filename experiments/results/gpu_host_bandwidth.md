# The GPU leg of the cross-vendor KV path, measured

*A40 server (`s8`, 8 × NVIDIA A40), 2026-08-27. Driver 560.35.05, torch 2.10.0+cu128,
CUDA 12.8. Raw data: `outputs/a40_profile/host_bandwidth.json`. Method:
`experiments/scripts/gpu_host_bandwidth.py`.*

This closes the one measurement `docs/HANDOVER_A40.md` §1 said only an NVIDIA host
could produce. A GPU-prefill / NPU-decode split has no device-to-device route
between vendors, so the KV cache goes **GPU → host → NPU**: two copies. The NPU
leg was measured on the NPU server; the GPU leg is measured here. Both P/D
fixtures can now stop carrying the NPU leg as a stand-in for the whole path.

**The relevant direction is D2H** — the GPU is the prefill side, so its
contribution is device-to-host. H2D is reported too.

## Method

Deliberately identical to `measure_host_bandwidth()` in
`experiments/scripts/rngd_device_facts.py`: contiguous bfloat16 buffer, host-side
`perf_counter`, best and median of 7 reps after one warm copy, sizes
1/4/16/64/256 MB. Two legs measured differently do not compose.

Three CUDA-specific departures, each forced:

- **`torch.cuda.synchronize()` inside the timed region.** CUDA copies can complete
  asynchronously; timing the enqueue would report bandwidth the scheduler never
  gets. RNGD's `.to(device)` blocks, so this mirrors its semantics rather than
  diverging from them.
- **Pinned host memory measured as well as pageable.** The NPU figures are
  pageable; on CUDA the difference is 8×, so one number without the qualifier
  would be meaningless.
- **The pinned path reuses a pre-allocated destination.** A real KV handoff uses a
  persistent staging buffer, never a fresh pinned allocation per transfer. The
  pageable path keeps `.cpu()` exactly as the RNGD script has it.

Parallel groups are repeated over **5 independent buffer allocations** and the
median trial is the headline. A single trial is not reproducible: host buffers are
not NUMA-bound, their placement is fixed at allocation, and two early runs
disagreed by 38 % on the 4-GPU pinned D2H figure (84.4 vs 60.9 GB/s) with the
same-node / cross-node ordering reversing between them. Trial spreads are reported
so no one reads these as tighter than they are.

## Single stream (GB/s, best of 7)

| size | pageable H2D | pageable D2H | pinned H2D | pinned D2H |
| ---: | ---: | ---: | ---: | ---: |
| 1 MB | 4.38 | 3.4 | 4.19 | 5.05 |
| 4 MB | 5.74 | 4.79 | 20.65 | 23.66 |
| 16 MB | 20.08 | 14.18 | 24.86 | 25.16 |
| 64 MB | 23.2 | 3.17 | 25.37 | 25.72 |
| 256 MB | 23.55 | 3.1 | **25.5** | **25.77** |

Pinned saturates at **~25.77 GB/s**, which is PCIe 4.0 ×16 line rate — the A40's slot,
so the measurement is bounded by the link and not by anything above it.

> **Pageable D2H is allocator-bound, not link-bound — do not read it as a link
> figure.** It peaks at 16 MB (14.18) and falls ~4.5x at 64 MB and beyond (3.1),
> because `.cpu()` allocates a fresh destination per rep and PyTorch's CPU
> allocator reuses a cached block only up to ~16 MB, mmap-ing above it so every rep
> pays page faults. The medians sit far below the bests for the same reason. The
> pageable column exists for like-for-like comparison against the pageable NPU leg;
> **read the pinned rows for the link.**

## Parallel, 256 MB, pinned (GB/s aggregate, median of 5 trials)

| streams | GPUs | H2D | D2H | D2H trial spread |
| ---: | --- | ---: | ---: | --- |
| 2 | 0,1 (same node, NVLink pair) | 44.56 | 47.08 | 46.98–47.15 |
| 2 | 0,4 (cross node) | 50.82 | 50.83 | 50.7–50.92 |
| 4 | 0,1,2,3 (same node) | 57.42 | 66.27 | 66.26–66.46 |
| 4 | 0,1,4,5 (cross node) | 72.66 | 72.06 | 72.0–72.15 |
| 8 | all | 80.73 | 77.69 | 77.52–77.79 |

Pageable, same groups, D2H: 4.4 / 4.4 / 6.33 / 6.09 / 7.44 — again allocator-bound.

### The GPU leg does not scale the way the NPU leg does

Both columns are the **sustained** figures, so the comparison is like-for-like.
The NPU numbers are the corrected ones (D18), not the retracted peaks.

| | 4 streams | 8 streams |
| --- | ---: | ---: |
| NPU leg (host → RNGD PE), sustained | 15.36 GB/s = 102 % of ideal | 26.27 GB/s = **87 %** of ideal |
| GPU leg (A40 → host), same node, sustained | 70.47 GB/s = 69 % of ideal | 82.63 GB/s = **40 %** of ideal |

The host path saturates around **83 GB/s** and adding GPUs past four buys little.
This is the answer to the open question in `docs/HANDOVER_A40.md` §1 ("the NPU leg
scaled almost linearly and the question is whether the GPU leg does too"): **it
does not** — and the retraction of the NPU figures did not change that, because it
scaled all four NPU points by the same factor and left the scaling law intact.

Spreading a group across both NUMA nodes is consistently *better* than packing it
onto one (72.06 cross-node against 66.27 same-node at 4 GPUs;
50.83 against 47.08 at 2) — two memory controllers beat one, and the host
buffers are unbound so they land on both. An earlier single-trial run showed the
opposite ordering; that was noise, and it is why the trial repetition exists.

## Sustained, and why it was measured second

The NPU leg's remeasurement (deviations **D18**,
`experiments/results/rngd_parallel_bandwidth.md`) found that **best-of-N
overstates a bulk copy by 25 %** on RNGD: 5.06 GB/s per PE peak against 3.77
sustained. Roughly half of that is the idle between timed transfers letting the
device recycle its buffer, half is best-versus-typical.

That raised an obvious worry about this file's own numbers, which were all
best-of-N: if the GPU leg carried the same upward bias, every composed fabric
value would be flattered on both sides. **It does not.** Measured with the NPU
leg's statistic — a 5 s window, back-to-back copies, aggregate over the
concurrent region, 3 trials — pinned D2H at 256 MB:

| configuration | best-of-N | sustained | sustained / peak |
| --- | ---: | ---: | ---: |
| 1 stream | 25.77 | 25.71 | 0.998 |
| 2 GPUs, same node (0,1) | 47.08 | 47.95 | 1.018 |
| 4 GPUs, same node (0–3) | 66.27 | 70.47 | 1.063 |
| 4 GPUs, cross node (0,1,4,5) | 72.06 | 80.04 | 1.111 |
| 8 GPUs | 77.69 | 82.63 | 1.064 |

**The A40 sustains its peak** — 0.998 at one stream, and *above* 1.0 in the
parallel groups, where best-of-N pays a per-rep barrier that a sustained window
amortises. Against RNGD's 0.745 this is a real difference between the two
devices, not a measurement quirk: CUDA's dedicated copy engine plus PyTorch's
caching device allocator mean back-to-back pinned copies pipeline with nothing to
recycle, which is exactly the cost RNGD was found to pay.

**The hypothesis that prompted this measurement was refuted by it.** The GPU leg
does not need the correction the NPU leg needed. It was still worth measuring:
the fixtures now compose two legs taken with the same statistic, which is what
makes `1/(1/gpu + 1/npu)` meaningful.

Sustained scaling vs ideal, pinned D2H: 93 % at 2 same-node, 100 % at 2
cross-node, 69 % at 4 same-node, 78 % at 4 cross-node, **40 % at 8**. The
saturation conclusion is unchanged and slightly sharpened — the ceiling is
~83 GB/s. Pageable sustained D2H (3.20 / 4.39 / 6.00 / 7.27 at 1/2/4/8) remains
allocator-bound and is not a link figure.

## What this means for the P/D question

Composed as a **serialised** handoff, `1/(1/gpu + 1/npu)` — the two copies are not
assumed to overlap, because nothing in this stack pipelines them. A pipelined
implementation would give `min(gpu, npu)` instead. **Both legs are sustained**, so
the composition mixes like with like.

Per link, at the TP degree where each fixture's cross-vendor P/D actually exists.
A scalar link cannot vary with the candidate's TP degree, so each fixture is
pinned to its own degree and says so.

| fixture | link | composition | GB/s |
| --- | --- | --- | ---: |
| `pd-rngd-gpu-card.yaml` (tp1) | rngd ↔ a40 | 1/(1/25.71 + 1/26.27) | **13.0** |
| | rngd ↔ rngd | 1/(1/26.27 + 1/26.27) | 13.1 |
| | a40 ↔ a40 | 1/(1/25.71 + 1/25.71) | 12.9 |
| `pd-rngd-gpu.yaml` (tp4) | rngd ↔ a40 | 1/(1/70.47 + 1/15.36) | **12.6** |
| | rngd ↔ rngd | 1/(1/15.36 + 1/15.36) | 7.7 |
| | a40 ↔ a40 | 1/(1/70.47 + 1/70.47) | 35.2 |

All twelve links across the two fixtures are `source: measured`. `measured` is the
honest label available: both legs are measurements and the composition is
arithmetic over them under a stated handoff model. The enum
(`planner/inventory.py`) has no `derived` value; the assumption is recorded in the
fixture comment and here.

### Against the placeholder, and against the interim values

| link | placeholder | interim (peak legs) | final (sustained legs) |
| --- | ---: | ---: | ---: |
| card, rngd ↔ a40 | 35 | 15.0 | **13.0** |
| card, rngd ↔ rngd | 35 | 17.7 | 13.1 |
| per-PE, rngd ↔ a40 | 35 | 14.6 | **12.6** |
| per-PE, rngd ↔ rngd | 35 | 9.6 | 7.7 |

**The placeholder was 2.7–4.5× too optimistic.** It was the NPU leg's *peak*
used as an upper bound on the whole two-copy path — wrong in three ways at once:
one leg standing for two, a peak standing for a sustained rate, and a retracted
figure standing for a measured one.

Note how little the *second* correction moved the cross-vendor links: 15.0 → 13.0
and 14.6 → 12.6, both under 15 %. **The NPU leg dominates the composition**, so
the GPU leg's statistic barely matters. That is itself the finding — for this
hardware pair the fabric bandwidth is set on the NPU side, and effort spent
pinning down the GPU number precisely has low leverage.

Three things to carry forward:

1. **At tp1 the two legs are balanced, not GPU-bound.** The card fixture's
   cross-vendor P/D exists only at tp1, where the GPU contributes 25.71 GB/s
   against the card's 26.27 across 8 PEs. The interim reading — that the GPU leg
   was the bottleneck there — came from composing a peak NPU leg; with both legs
   sustained they are within 2 % of each other.
2. **The per-PE fixture's RNGD↔RNGD link is below the crossing** — 7.7 GB/s
   against Exp 3's ~10. Same-vendor P/D across two RNGD cards is priced worse
   than cross-vendor P/D in that fixture, and worse than it was at the interim
   value of 9.6.
3. **The cross-vendor links sit just above the crossing** at 12.6–13.0 GB/s, where
   the placeholder implied 35. The SLO sweeps were re-run at these values; the
   headroom is thin enough that the regimes should not be assumed to carry over.

## Provenance

Both legs are now committed and reproducible from committed code:

| leg | artifact | producer |
| --- | --- | --- |
| GPU (A40 → host) | `outputs/a40_profile/host_bandwidth.json` | `experiments/scripts/gpu_host_bandwidth.py` |
| NPU (host → RNGD) | `outputs/rngd_profile/parallel_bandwidth.json` | `experiments/scripts/rngd_device_facts.py --parallel-bandwidth` |

The gap this file originally reported — that the NPU leg's multi-stream figures
existed only as prose — was closed on the NPU server on 2026-08-27, and closing it
retracted them (D18). The single-stream NPU run
(`outputs/rngd_profile/host_bandwidth.json`) remains valid as a best-of-N
measurement of the same link.
