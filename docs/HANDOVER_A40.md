# Handover — continuing on the A40 server

*Written 2026-08-26, **revised 2026-08-27 after everything merged to `main`**, for
a Claude Code session that will `git pull` this repo on the **A40 server** and
continue. Read this first, then `experiments/results/rngd_edf_bundle_notes.md` and
`docs/PROJECT_REPORT.md` §4.8.*

> The NPU server work is done and **merged to `main`** (PRs #14, #15, #16 — there
> are no feature branches left to check out; `git pull` on `main` is enough). What
> remains needs a **GPU**, and there is no NVIDIA GPU on the NPU server
> (`nvidia-smi` cannot reach a driver). This file says exactly what to measure
> there and why it matters.

> **One correction to read before anything else.** An earlier version of §2 said
> RNGD's per-watt advantage was a *lower* bound. **That was backwards on the
> compute axis** — see the correction box in §2. It changes where mixing might pay,
> so do not build on the old reading.

---

## 1. The measurement that was blocked here — DONE 2026-08-27

> **Resolved on this server.** Both legs of the cross-vendor KV path are now
> measured and the two P/D fixtures carry composed values at `source: measured`.
> Full result: `experiments/results/gpu_host_bandwidth.md`; raw data
> `outputs/a40_profile/host_bandwidth.json`; method
> `experiments/scripts/gpu_host_bandwidth.py`. Headlines:
>
> - **GPU leg (A40 → host, pinned): 26.03 GB/s single stream, 79.98 GB/s across
>   8 GPUs.** Single-stream is PCIe 4.0 ×16 line rate. Pageable D2H is
>   allocator-bound, not link-bound — do not quote it as a link figure.
> - **The GPU leg does NOT scale like the NPU leg** — 38 % of ideal at 8 streams
>   against the NPU's 88 %. The host path saturates ~80 GB/s. That answers the
>   open question below.
> - **Composed serialised, the cross-vendor links are ~15 GB/s, not 35.** The old
>   placeholder was ~2.3× too optimistic. It still clears Exp 3's ~10 GB/s
>   crossing, so P/D stays viable with less headroom.
> - **At tp1 the GPU leg is the bottleneck, not the NPU leg** (26.03 against
>   35.47), which inverts the framing in the text below. The "NPU leg is not the
>   bottleneck" claim holds only when the GPU side is wide.
>
> The rest of this section is kept as the record of what was asked for and why.

## 1b. The original statement of the problem

**GPU → host transfer bandwidth**, so the cross-vendor Prefill/Decode KV handoff
path can be priced end to end.

A GPU-prefill / NPU-decode split has no device-to-device route between vendors.
The KV cache goes **GPU → host → NPU**: two copies. Only the NPU leg is
measurable on the NPU server, and it was measured:

| streams | aggregate host → RNGD PE |
| ---: | ---: |
| 1 | 5.06 GB/s |
| 2 | 10.39 GB/s |
| 4 | 19.10 GB/s |
| 8 | **35.47 GB/s** (median 31.36) |

A single stream is only 5 GB/s, which is *below* the ~10 GB/s P/D adoption
crossing Exp 3 found — but a TP=8 decode island shards its KV across 8 PEs, so
the handoff uses the parallel path and sustains ~35 GB/s at 88 % of ideal
scaling. Raw data: `outputs/rngd_profile/host_bandwidth.json`.

So the NPU leg is **not** the bottleneck. Whether the whole path clears the
crossing depends on the GPU leg, which is what the A40 server has to answer.

### What to run there

```bash
# 1. GPU -> host, single stream and parallel across GPUs, mirroring the NPU method:
#    contiguous bfloat16 buffer, host-side perf_counter, best and median of N reps
#    after one warm copy, sizes 1/4/16/64/256 MB.
#    The NPU-side implementation to copy is measure_host_bandwidth() in
#    experiments/scripts/rngd_device_facts.py -- keep the method identical or the
#    two legs are not comparable.
# 2. Pinned vs pageable host memory BOTH ways. cudaHostAlloc changes this by
#    several x on CUDA and the NPU figures above are pageable, so report both and
#    say which the comparison uses.
# 3. Parallel across 2 and 4 GPUs, because the NPU leg scaled almost linearly and
#    the question is whether the GPU leg does too.
```

Then set `bandwidth_gbps` on the `fabric-*` links in **both** P/D fixtures to
`1 / (1/gpu_leg + 1/npu_leg)` for a serialised handoff, or `min(gpu_leg,
npu_leg)` if the implementation pipelines the two copies — and say in the file
which one was assumed. Right now those links carry **35 GB/s labelled
`placeholder`** in both, i.e. the NPU leg used as an upper bound on the whole path.

**There are now two fixtures and they are not interchangeable:**

| fixture | RNGD accelerator | TP set | use it for |
| --- | --- | --- | --- |
| `pd-rngd-gpu.yaml` | one **PE**, 6.25 GB | {4, 8} | TTFT-feasibility questions (its TTFT error is −32.6 %) |
| `pd-rngd-gpu-card.yaml` | one **card**, 47.5 GB, tp1 | {1} | decode / energy / goodput, and **the only one where cross-vendor P/D actually simulates** |

Under the per-PE fixture, `A40 prefill → RNGD decode` — the direction that matters
— **crashes 6/6** on RNGD decode-KV exhaustion. The card fixture is what unblocked
it. Run both, and expect only the card one to have a cross-vendor row.

With a real bandwidth number, re-run:

```bash
./experiments/scripts/run_exp_pd.sh          # Exp 3 network sweep + Exp 5 combos
for fx in pd-rngd-gpu pd-rngd-gpu-card; do
  PYTHONPATH=$PWD .venv/bin/python experiments/scripts/pd_slo_sweep.py \
      --service examples/service_specs/llama31-8b.yaml \
      --cluster experiments/configs/clusters/$fx.yaml \
      --ttft-ms 500,1000,2000,4000,8000,16000,32000,64000 \
      --num-requests 300 --seed 42 --workers 32 \
      --output-dir outputs/.hp-slo-$fx
done
```

**Before quoting any TTFT number off the card fixture, apply
`profiles/calibration/rngd_card_edf.yaml` (`real = 2.089·sim + 646 ms`).** Its raw
sweep declares a 480 ms p99 TTFT winner; calibrated that is ~1649 ms, and the real
hardware does 1404 ms at concurrency ~20. Decode, energy and goodput need no such
correction (TPOT fit error 0.025). The per-PE fixture's calibration is
`profiles/calibration/rngd.yaml`.

Baseline to compare against — the sweeps as run on the NPU server with placeholder
bandwidth, so a real GPU leg can only move these *down*:
`experiments/results/pd_slo_sweep.md`.

---

## 2. State of the answer to "can mixed GPU+NPU P/D ever win?"

Established on the NPU server, from measurements:

**Per watt, RNGD wins both phases, so mixing cannot win on efficiency alone.**
Derived the same way from every committed bundle (tp1 `dense.csv`, 1 token =
bandwidth-bound, 2048 tokens = compute-bound):

| | achieved GB/s | TFLOP/s | W | GB/s per W | GFLOP/s per W |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNGD card (8 PE) | 1750 | 240 | 299.7 | **5.84** | **801** |
| A40 | 569 | 127 | 297.8 | 1.91 | 427 |
| A5000 | 694 | 104 | 227.6 | 3.05 | 457 |
| RTXPRO6000 | 1473 | 417 | 600 | 2.45 | 696 |

If one device dominates both axes, the efficiency optimum is homogeneous and
adding a GPU strictly lowers tokens/J. **Mixing can only pay when a constraint
binds**, and the candidates are:

1. **TTFT SLO.** Single-request prefill latency is capped by one island's
   compute, and RNGD is far behind per device: 2048-token `down_proj` takes
   8513 µs on one PE against 2010 µs on an A40. Below some TTFT there is no
   feasible all-RNGD plan and prefill must move to the GPU. Finding that
   threshold is `experiments/scripts/pd_slo_sweep.py`.
2. **Bucket quantisation, which taxes RNGD prefill only.** The vendor artifact is
   compiled for fixed `tokenwise_buckets` (1, 2, 4, …, 256, 384, 512, 1024), so a
   prompt pays for the buckets covering it: 399 tokens runs the 512 bucket.
   Measured over the 20-request trace: 27,408 charged tokens against 24,725
   actual, **+10.9 % aggregate, +20.7 % mean per request**. Decode always runs
   `input_size=1`, so it pays nothing. The vendor's own compilation model
   penalises exactly the phase a mixed split would move to the GPU.
3. **A hard power cap**, where the optimum is "buy the minimum GPU prefill
   capacity that meets TTFT, spend every remaining watt on RNGD decode".

> ### CORRECTION, 2026-08-27 — the error direction was backwards on compute
>
> The paragraph that stood here said the RNGD figures come from a harness that is
> "~26 % pessimistic on decode", so the per-watt advantage was a **lower bound**.
> That conflated two different things:
>
> - the **simulator** was pessimistic on decode (predicted TPOT 35.7 ms against a
>   real 28.4 ms), because ASTRA-Sim over-charges the intra-card collective;
> - but the **harness itself under-measures per-layer time**. FuriosaAI's own EDF
>   profiler puts the real per-layer decode cost at 507 µs where the harness
>   accounts for 290–307 µs, and the measured vendor/harness ratio runs ×1.16 at
>   1024 tokens to ×1.61 at 2 tokens
>   (`outputs/rngd_edf_bundle/edf_vs_harness_dense.csv`).
>
> **Times that are too small make derived throughput too large.** So the
> `GFLOP/s per W` column is an **upper** bound, not a lower one. Correcting the
> RNGD row by the measured prefill-range ratios:
>
> | GFLOP/s per W | value |
> | --- | ---: |
> | RNGD as tabulated above | 801 |
> | ÷1.16 (the ×1.16 ratio at 1024 tokens) | **691** |
> | ÷1.50 (the ×1.50 ratio at 512 tokens) | **534** |
> | A40 | 427 |
> | RTXPRO6000 | 696 |
>
> **What survives and what does not.** Against the A40, RNGD still wins the
> compute axis comfortably (534–691 against 427), so §2's argument holds *for this
> hardware pair* and the A40 server can proceed on it. Against **RTXPRO6000 (696)
> the advantage disappears**, so the sentence "if one device dominates both axes,
> the efficiency optimum is homogeneous" must not be generalised beyond the
> RNGD/A40 pair.
>
> **The bandwidth axis is unaffected.** 5.84 GB/s/W comes from a direct
> measurement — 8 PEs sustaining 218.8 GB/s each, 104 % scaling — not from
> `dense.csv` timings, so it stands as measured.
>
> A better source now exists for anything per-layer: `profiler/perf/RNGD-CARD/`,
> rebuilt from the vendor's own EDF traces, predicts real decode to −3.1 % against
> the per-PE bundle's +25.7 %. **Recompute this table from that bundle** rather
> than patching the numbers above, if the per-watt comparison is going into the
> paper.

Absolute latencies carry ±30 % error, so the TTFT threshold in (1) must come from
simulation with calibration applied, not from these numbers.

---

## 3. A structural blocker found while building the experiment — do not lose this

**Cross-vendor P/D is unrepresentable whenever the two backends' feasible TP
degrees do not overlap**, and for this hardware pair they naturally do not.

`planner/candidate_generator.py` requires `tp_p == tp_d` for a P/D pair
(deviations D14: the simulator infers its topology as
`[npus_per_group, num_instances]` by integer division over the total device
count, so unequal instance sizes are unrepresentable). And:

- RNGD reaches only **tp ∈ {4, 8}** for Llama-3.1-8B, because a PE holds 6.25 GB
  against 14 GB of weights. tp=1 and tp=2 are correctly rejected on memory.
- An **NVLink-pair** A40 island offers only **tp ∈ {1, 2}**.

Those sets are disjoint, so the first run of the experiment emitted **no mixed
P/D candidate at all** — 3 representatives instead of 5, with both mixed combos
silently absent. Nothing errored; the combos just did not exist.

The fix in `experiments/configs/clusters/pd-rngd-gpu.yaml` is to bridge the
NVLink pairs over PCIe into **size-4** A40 islands, exactly as
`experiments/configs/clusters/exp1-a40-tp-sweep.yaml` already does, giving
tp ∈ {1, 2, 4} and making **tp=4 the shared degree** where GPU-P + NPU-D exists.

**Consequence for the paper, and for any future hardware pair:** a heterogeneous
P/D deployment is only expressible if the island sizes can be chosen to share a
TP degree. That is a real planner-level constraint on heterogeneity, not a
fixture detail, and it deserves to be stated wherever cross-vendor P/D is
claimed. If it becomes limiting, lifting it means addressing D14 — teaching the
compiler to emit non-uniform instance sizes — which is a simulator-side change.

### Update 2026-08-27 — a second route exists, and the tp=4 bridge is no longer the only fix

Two things changed after this section was written:

1. **`LinkType` had no on-package fabric at all.** Added as `ONPACKAGE`
   (deviations **D16**), which is what lets an RNGD card be one island rather than
   8 isolated devices. Before that, the disjoint-TP problem was compounded by the
   PEs not forming an island in the first place.
2. **Card-as-device sidesteps the shared-degree problem entirely.** With one
   accelerator = one 47.5 GB card at tp1, the RNGD TP set is **{1}**, which
   overlaps an A40 island's {1, 2, 4}. So `pd-rngd-gpu-card.yaml` needs **no
   PCIe-bridging fixture hack** — cross-vendor P/D exists at tp1 naturally. It
   also removes the decode-KV exhaustion that made the promising direction crash.

**But be clear about what that is and is not.** Card-as-device does not lift D14;
it *folds TP=8 inside the device* so the constraint has nothing to bind on. That
works for RNGD specifically, because the vendor's only deployable configuration
*is* TP=8 on one card. It is **not a general fix**, and the shape the literature
recommends — `A40 tp4 prefill + RNGD tp8 decode`, big TP on the memory-bound
phase (NVIDIA Dynamo, AWS Neuron) — remains unrepresentable.

Also new: **deviation D17.** The §3.7 attention grid cannot express the vendor
runtime's per-layer attention cost, which tracks the batch's KV **diversity**
rather than its size (1.95 attention executions per layer at batch 2, 3.08 at
batch 29). The card bundle works around it with total-preserving rows, which
**ties its decode-attention axis to sharegpt-like traffic at mean KV ≈ 2200.** If
the A40 work introduces a workload with a very different KV spread, that axis
needs its own collection pass on the NPU server — it cannot be re-derived from a
GPU.

---

## 4. What is already measured and needs no repeat

Do not re-measure these on the A40 server; they are committed.

| thing | where | note |
| --- | --- | --- |
| RNGD perf bundle, tp1/2/4/8 | `profiler/perf/RNGD/` | measured via `furiosa.torch`; tp8 is the weakest (40 CPU fallbacks vs 1 elsewhere) |
| RNGD per-PE power | `profiles/accelerators/furiosa_rngd.yaml` | `board = 38.01 + 32.71 × PEs`, R² 0.996, from a 0..8 loaded-PE sweep |
| RNGD sim-vs-real | `experiments/results/rngd_sim_vs_real_summary.md` | TTFT −32.6 %, TPOT +25.5 %; **opposite signs** |
| RNGD calibration | `profiles/calibration/rngd.yaml` | scoped to one workload bucket; do not extrapolate |
| host → RNGD bandwidth | `outputs/rngd_profile/host_bandwidth.json` | the NPU leg of the KV path |
| A40 profile + calibration | `profiles/accelerators/a40.yaml`, `profiles/calibration/a40.yaml` | measured on the A40 server previously; α 1.017 / 1.011 |
| **RNGD perf bundle from the vendor's own profiler** | `profiler/perf/RNGD-CARD/` | **the accurate one** — decode −3.1 % against real, vs +25.7 % for the per-PE bundle. Built from 6 `furiosa-llm serve` passes, 1.74 M stage executions |
| **RNGD card profile** | `profiles/accelerators/furiosa_rngd_card.yaml` | card-as-device, 47.5 GB at tp1, card-total power measured |
| **RNGD card calibration** | `profiles/calibration/rngd_card_edf.yaml` | TPOT fit error **0.025**; TTFT fit error **2.34 — recorded as unusable, do not lean on it** |
| **On-package all-reduce** | `experiments/results/rngd_collective_measured.md` | **115 µs per decoder layer at TP=8**, measured directly. Retracts two earlier estimates, one wrong by 40× |
| **P/D SLO sweeps, both fixtures** | `experiments/results/pd_slo_sweep.md` | 8 TTFT points each, with placeholder bandwidth — the baseline your real number replaces |

**The A40 numbers already in the repo are valid measurements of that server.**
Re-running the A40 profiler is not required for the P/D question and would only
be worth it if you want to refresh the bundle for a newer vLLM.

---

## 5. Environment on the A40 server

Follow `docs/HANDOVER_NPU.md` §2 for `.venv` (planner + analytical simulator);
it is machine-independent except for the NPU-specific §2b. Two traps that cost
time on the NPU server and will repeat:

- **`pydantic`, `pytest`, `ruff`, `mypy` are missing from the documented install
  list.** Without pydantic, `tests/conftest.py` fails at import and nothing
  collects. `CLAUDE.md` §Environment now includes them.
- **`astra-sim` is a submodule**; a plain clone leaves it empty and
  `scripts/compile.sh` fails. `git submodule update --init --recursive`, and
  `protoc` + `libprotobuf-dev` are system prerequisites.

For the profiler on CUDA you also need the `.venv-vllm` described in
`docs/HANDOVER_NPU.md` §2b — that part *is* CUDA-specific and applies on the A40
server, unlike on the NPU server where the vendor stacks live in the system
interpreter.

Expect `pytest` to report **284 passed** — unconditionally now that
`fix/calibration-test-artifact-path` is on `main` (PR #14). If it reports
280 passed / 4 failed in `tests/test_calibration.py`, you are on a stale checkout;
`git pull`.

**Nothing NPU-specific will break on the A40 server.** `furiosa` imports are
confined to `experiments/scripts/`, and `pyproject.toml` sets
`testpaths = ["tests"]`, so no NPU code is imported at collection time. The
NPU-only scripts (`profile_rngd.py`, `rngd_device_facts.py`,
`rngd_collective_probe.py`, `rebuild_rngd_bundle_from_edf.py`,
`bench_furiosa_endpoint.py`) simply will not run there, which is expected — do not
try to make them work.

Upstream pin to verify after cloning: `UPSTREAM_COMMIT` says `2c2042ce`, submodule
`astra-sim` at `f82fb3d`.

---

## 6. Where the work lives — all merged, 2026-08-27

**There are no feature branches to check out. `git checkout main && git pull` is
the whole story.** All three branches from the NPU server are merged:

| PR | branch (deleted) | what |
| ---: | --- | --- |
| #14 | `fix/calibration-test-artifact-path` | stale test path; restores 284 passing |
| #15 | `docs/npu-server-env-setup` | env bring-up log, NPU inventory, who holds the NPUs |
| #16 | `feat/rngd-profiling` | everything RNGD: two profiling instruments, both bundles, power, sim-vs-real, the measured all-reduce, both P/D fixtures, both sweep drivers |

Merge state at handover: `bcb3498`, `main` green (284 passed, `ruff` clean, `mypy`
clean over `planner/`).

One PR remains open and is unrelated to this work: **#13** (`docs/slide-deck-ko`).

## 7. Suggested order of work on the A40 server

1. **Environment + `git pull`**, confirm 284 passed. (§5)
2. **Read the §2 correction box before using any per-watt number.** It inverts the
   error direction the old text claimed.
3. **Measure the GPU→host leg** — the one thing only that machine can do. (§1)
4. **Set `bandwidth_gbps` on the `fabric-*` links in BOTH fixtures**, and record in
   each file whether a serialised or pipelined handoff was assumed.
5. **Re-run both SLO sweeps** and diff against `experiments/results/pd_slo_sweep.md`.
   Apply the card calibration before any TTFT claim.
6. **Optional, and only if the per-watt table goes in the paper:** recompute it
   from `profiler/perf/RNGD-CARD/` rather than the per-PE bundle.

What is *not* worth doing there: re-measuring the A40 profile (the committed one is
a valid measurement of that server), and anything RNGD — the hardware is on the
other machine and every RNGD quantity this project needs is now measured.
