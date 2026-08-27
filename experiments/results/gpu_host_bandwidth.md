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
| 1 MB | 4.33 | 1.67 | 4.40 | 5.29 |
| 4 MB | 5.72 | 3.19 | 20.79 | 18.60 |
| 16 MB | 19.78 | 15.63 | 24.80 | 25.50 |
| 64 MB | 23.17 | 3.23 | 25.34 | 25.75 |
| 256 MB | 23.55 | 3.18 | **25.49** | **26.03** |

Pinned saturates at **~26 GB/s**, which is PCIe 4.0 ×16 line rate — the A40's slot,
so the measurement is bounded by the link and not by anything above it.

> **Pageable D2H is allocator-bound, not link-bound — do not read it as a link
> figure.** It peaks at 16 MB (15.63) and falls 5× at 64 MB and beyond (3.2),
> because `.cpu()` allocates a fresh destination per rep and PyTorch's CPU
> allocator reuses a cached block only up to ~16 MB, mmap-ing above it so every rep
> pays page faults. The medians sit far below the bests for the same reason. The
> pageable column exists for like-for-like comparison against the pageable NPU leg;
> **read the pinned rows for the link.**

## Parallel, 256 MB, pinned (GB/s aggregate, median of 5 trials)

| streams | GPUs | H2D | D2H | D2H trial spread |
| ---: | --- | ---: | ---: | --- |
| 2 | 0,1 (same node, NVLink pair) | 44.42 | 47.17 | 47.1–49.2 |
| 2 | 0,4 (cross node) | 50.78 | 50.86 | 50.7–51.9 |
| 4 | 0,1,2,3 (same node) | 57.49 | 62.42 | 59.8–65.4 |
| 4 | 0,1,4,5 (cross node) | 73.85 | 68.50 | 66.5–68.6 |
| 8 | all | **81.88** | **79.98** | 76.6–80.2 |

Pageable, same groups, D2H: 4.43 / 4.46 / 6.30 / 6.10 / 7.73 — again allocator-bound.

### The GPU leg does not scale the way the NPU leg does

| | 4 streams | 8 streams |
| --- | ---: | ---: |
| NPU leg (host → RNGD PE) | 19.10 GB/s = 94 % of ideal | 35.47 GB/s = **88 %** of ideal |
| GPU leg (A40 → host), same node | 62.42 GB/s = 60 % of ideal | 79.98 GB/s = **38 %** of ideal |

The host path saturates around **80 GB/s** and adding GPUs past four buys little.
This is the answer to the open question in `docs/HANDOVER_A40.md` §1 ("the NPU leg
scaled almost linearly and the question is whether the GPU leg does too"): **it
does not.**

Spreading a group across both NUMA nodes is consistently *better* than packing it
onto one (4 GPUs: 68.50 cross-node vs 62.42 same-node; 2 GPUs: 50.86 vs 47.17) —
two memory controllers beat one, and the host buffers are unbound so they land on
both. An earlier single-trial run showed the opposite ordering; that was noise, and
it is why the trial repetition exists.

## What this means for the P/D question

Composed as a **serialised** handoff, `1/(1/gpu + 1/npu)` — the two copies are not
assumed to overlap, because nothing in this stack pipelines them. A pipelined
implementation would give `min(gpu, npu)` instead. Both are reported; the
qualitative answer does not turn on the choice.

| configuration | GPU leg | NPU leg | serialised | pipelined |
| --- | ---: | ---: | ---: | ---: |
| single stream each side | 26.03 | 5.06 | 4.24 | 5.06 |
| widest parallel each side | 79.98 (8 GPU) | 35.47 (8 PE) | 24.57 | 35.47 |

**The placeholder was 35 GB/s — the NPU leg alone, used as an upper bound on the
whole path. Composed properly it is ~15 GB/s on the cross-vendor links, so the
placeholder was ~2.3× too optimistic.** It still clears the ~10 GB/s P/D adoption
crossing Exp 3 found, so P/D remains viable — but with far less headroom than the
old number implied.

Two findings worth carrying forward:

1. **At tp1 the GPU leg is the bottleneck, not the NPU leg.** The card fixture's
   cross-vendor P/D exists only at tp1, where the GPU contributes a single 26.03
   GB/s stream against the card's 35.47 GB/s across 8 PEs. The handover's framing
   ("the NPU leg is not the bottleneck") is right only when the GPU side is wide.
2. **The per-PE fixture's RNGD↔RNGD link falls *below* the crossing** — 9.6 GB/s,
   composed from two 4-PE legs. Same-vendor P/D across two RNGD cards is priced
   worse than cross-vendor P/D in that fixture.

### Values written into the fixtures

Per link, at the TP degree where each fixture's cross-vendor P/D actually exists.
A scalar link cannot vary with the candidate's TP degree, so each fixture is pinned
to its own degree and says so.

| fixture | link | composition | GB/s |
| --- | --- | --- | ---: |
| `pd-rngd-gpu-card.yaml` (tp1) | rngd ↔ a40 | 1/(1/26.03 + 1/35.47) | **15.0** |
| | rngd ↔ rngd | 1/(1/35.47 + 1/35.47) | 17.7 |
| | a40 ↔ a40 | 1/(1/26.03 + 1/26.03) | 13.0 |
| `pd-rngd-gpu.yaml` (tp4) | rngd ↔ a40 | 1/(1/62.42 + 1/19.10) | **14.6** |
| | rngd ↔ rngd | 1/(1/19.10 + 1/19.10) | 9.6 |
| | a40 ↔ a40 | 1/(1/62.42 + 1/62.42) | 31.2 |

All six links in both fixtures move from `source: placeholder` to `source: measured`.
`measured` is the honest label available: both legs are measurements and the
composition is arithmetic over them under a stated handoff model. The enum
(`planner/inventory.py`) has no `derived` value; the assumption is recorded in the
fixture comment and here.

## Provenance gap noticed while doing this

The NPU leg's **multi-stream** figures (10.39 / 19.10 / 35.47 GB/s at 2 / 4 / 8
streams) exist only as prose in `docs/HANDOVER_A40.md`, `docs/PROJECT_REPORT.md` and
the two fixture comments. The committed
`outputs/rngd_profile/host_bandwidth.json` holds **only the single-stream run**
(`rngd:16`, peak H2D 5.03). Every composed number above that uses a parallel NPU
leg therefore rests on an uncommitted measurement. Re-running
`rngd_device_facts.py` with a parallel mode on the NPU server would close it; it
cannot be closed from here.
