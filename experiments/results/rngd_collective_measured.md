# RNGD: the on-package all-reduce, measured

*Measured 2026-08-26 on one RNGD card. Replaces two earlier estimates, both of
which were too large — one by 40×. The headline: **the intra-card reduction is
cheap**, and the claim that RNGD decode is "collective-bound" was wrong.*

## The number

Row-parallel layers (`o_proj`, `down_proj`) are the ones an all-reduce follows:
each rank holds a slice of the input dimension and produces a partial sum over
the full output, so the partials must be reduced. Both layers reduce the same
4096-dim bf16 activation — 8 KB at one token.

| group size N | `down_proj` | `o_proj` | per decoder layer |
| ---: | ---: | ---: | ---: |
| 1 (control) | **0.22 µs** | **−0.01 µs** | — |
| 2 | 3.67 | 1.98 | 5.65 |
| 4 | 13.51 | 10.48 | 23.99 |
| **8** | **49.97** | **65.11** | **115.08** |

So at the deployed TP=8, the two all-reduces in a decoder layer cost about
**115 µs**, against a total per-layer decode cost of 507 µs measured on the real
compiled graph.

**Growth is superlinear in N** — ×3.7 then ×3.7 for `down_proj`, ×5.3 then ×6.2
for `o_proj` — where a ring all-reduce of fixed message size should grow roughly
linearly. The EDF traces give the likely mechanism: `tensor_parallel_size: 8` is
realised as **two fused 4-PE quads** (`rngd_vendor_profiler_vs_layerwise.md`), so
an 8-PE reduction crosses a quad boundary that 2- and 4-PE reductions never do.
That is a hypothesis the numbers are consistent with, not something this probe
isolates.

## How it was measured, and the bug that had to be fixed first

`furiosa.torch` exposes no `all_reduce` API — collectives live inside the compiled
EDF — so device fusion is the only way to make the compiler emit one. For each
group size N, in two separate processes (`set_fusion()` is process-global and
one-shot):

```
SHARD.  set_fusion(1); Linear(in/N, out) on ONE PE
        -> exactly one rank's compute, and no reduction, because there is no
           group to reduce across
FULL.   set_fusion(N); Linear(in, out) on the fused N-PE device
        -> the same per-rank compute done N-ways in parallel, PLUS the reduction
```

`collective = FULL − SHARD`.

**The first version of this probe ran both legs on the fused device**, so the
"per-rank" leg was a 1/N-sized layer that the compiler *also* sharded and *also*
reduced. The reduction largely cancelled in the subtraction and what remained was
mostly the extra weight traffic of an N-times-larger layer. It reported
**139 / 107 / 95 µs** for `down_proj` at N = 2 / 4 / 8 — plausible-looking, wrong
by up to 40×, and *decreasing* in N, which should have been the tell.

The fusion=1 control could not catch it: at N=1 the two legs are the same graph
whichever device they run on, so it returned ~0 for the wrong reason. Hence a
second check that the old design structurally lacked:

| validity check | requirement | observed |
| --- | --- | --- |
| **control** (N=1) | `FULL − SHARD ≈ 0` | 0.22 / −0.01 µs |
| **scaling** | `N × SHARD(N) ≈ SHARD(1)` | 1.011–1.130 (N = 2, 4, 8) |

The scaling check is what makes the subtraction meaningful: it verifies the SHARD
leg really is one rank's worth of work. Both pass on every row here.

`time_us` is the union of device spans (`Renegade::*`, `DMA`, `Task`), the same
definition the perf bundle uses.

## What this corrects

**1. The retracted 202 µs per-collective figure** — already withdrawn in
`rngd_card_vs_pe_model.md` for dividing a per-token residual by a per-forward
collective count. The measured value is 115 µs *per decoder layer* for both
reductions together, i.e. ~58 µs each.

**2. "Decode is collective-bound" (commit `4aac7a5`) is wrong.** That commit
attributed the card-harness model's −45.5 % TPOT error — 12.92 ms per token at
concurrency ~64 — to the missing intra-card collective. Measured, the collective
is 115 µs/layer × 32 layers = **3.68 ms per forward**, and at ~4.3 tokens per
forward that is **0.86 ms per token, about 7 % of the 12.92 ms gap**. The
dominant cause was the layerwise harness under-measuring per-layer decode by
1.5–1.7×, which the EDF rebuild established independently
(`rngd_edf_bundle_notes.md`). Two separate lines of evidence now agree, and the
collective was a red herring.

**3. The 212 µs/layer "unaccounted" gap is now decomposed.**
`rngd_vendor_profiler_vs_layerwise.md` measured 507 µs/layer on the real graph
against 295 µs in the harness bundle and called the 212 µs difference an upper
bound on the reduction. It is: the reduction is **115 µs, or 54 % of it**. The
remaining ~97 µs is the other work the compiler fused into that Tokenwise stage —
residual adds, the second norm, KV-cache writes.

**4. It explains why the per-PE model over-predicts decode, and gives a
calibration target.** The per-PE bundle's +25.7 % TPOT error must come from
ASTRA-Sim's analytical collective model, since the harness *under*-measures the
compute. Working backwards from the TPOT ratio (35.7 / 28.4 = 1.257) and treating
attention as common to both, ASTRA-Sim appears to charge roughly **340 µs/layer
against a measured 115 µs — about 3× too much**. That arithmetic assumes per-layer
cost scales with the TPOT ratio, so treat it as an estimate; the 115 µs is the
measurement, and it is the number to calibrate `link_bw` / `link_latency` in
`experiments/configs/clusters/rngd-llama31-8b-tp8.json` against.

**5. 8-PE fusion nodes do exist.** Both RNGD profiles asserted that `/dev/rngd`
exposes fusion nodes for 2 and 4 PEs but none for 8. `set_fusion(8)` enumerates
and works: device counts go 24 / 12 / 6 / 3 at fusion 1 / 2 / 4 / 8 (three cards
visible to this process), and the 8-PE device gives a real 7.2× per-rank speedup
on `o_proj` (174.76 → 24.39 µs). The claim has been corrected in both profiles.
It was never the load-bearing argument for `max_tp_size` anyway — that rests on
no multi-card artifact having been built or served here, and `MeshKind: Single`.

## Limits

- **One token, one layer shape each.** The reduction of an 8 KB activation is the
  decode case, which is the one that matters, but nothing here measures how the
  reduction scales with batch or with a larger message.
- **`FULL − SHARD` is an upper bound**, not a pure communication cost: the fused
  execution also carries whatever scheduling and DMA overhead differs between a
  one-PE and an N-PE plan. The controls bound that at well under 1 µs at N=1, but
  it is not guaranteed to stay there as N grows.
- **This is `furiosa.torch`, not `furiosa-llm`.** The compiler emits the reduction
  for a hand-written row-parallel `Linear`; the served artifact's fused
  `Tokenwise` stage may reduce differently. The EDF-derived 212 µs is the
  cross-check, and 115 of 212 µs is consistent.
- **`o_proj` at N=8 has the loosest scaling ratio** (1.117) and the largest
  collective relative to its compute (65 µs against 24 µs of per-rank work), so
  it is the row to re-measure first if this is ever load-bearing.

## Reproducing

```bash
PYTHONPATH=$PWD /usr/bin/python3 experiments/scripts/rngd_collective_probe.py \
    --fusions 1,2,4,8 --layers down_proj,o_proj --tokens 1 --reps 5 \
    --out outputs/rngd_profile/collective_probe.json
```

Needs the vendor interpreter (`/usr/bin/python3`), not the planner venv —
`furiosa.torch` lives in the user site-packages with torch 2.10. Per-worker
stdout/stderr lands in `outputs/rngd_profile/collective_probe_logs/`, which is
what made the earlier silent failures diagnosable.
