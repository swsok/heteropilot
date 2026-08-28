# HeteroPilot — Integrated Project Report

*Status snapshot: 2026-08-27. Basis for the presentation deck. Every quantitative
result below is reproducible from the committed artifacts named in each section.*

> **Honesty note (project absolute rule 3).** Except where a row is explicitly
> labelled **measured**, every number here is an **LLMServingSim prediction**, not
> a live hardware measurement. These labels are carried through every results file
> and figure. The label situation changed materially on 2026-08-25/26 and now
> differs per accelerator:
>
> | accelerator | status |
> | --- | --- |
> | **FuriosaAI RNGD** | **MEASURED on real silicon on this host**, twice over and by two instruments — a `furiosa.torch` layerwise harness and FuriosaAI's own EDF profiler on the served graph. Power, memory, achieved bandwidth, per-layer latency and the on-package all-reduce are all measured. §4.8. |
> | Rebellions ATOM | **placeholder stub** — broken vendor install and no free device. `sim_hardware: null`, empty `supported_models`, so it fails loud and stays out of candidate generation. |
> | NVIDIA A40 / RTX A5000 | **measured, but on OTHER machines.** This host has no NVIDIA GPU (`nvidia-smi` cannot reach a driver). Those artifacts stay valid and must never be re-run or relabelled here. |
> | RTXPRO6000 | vendor-spec profile. |
>
> **SIM-PROXY is retired for RNGD.** Results predating 2026-08-25 that stood an
> NPU in with a GPU compute model (notably Exp 5's original run) are superseded;
> where a section still carries SIM-PROXY rows it says so explicitly.

---

## 1. What HeteroPilot is

HeteroPilot is a **control plane for LLM serving on heterogeneous GPU/NPU
clusters**, built as a fork of `casys-kaist/LLMServingSim` (upstream pinned at
`2c2042ce`). Given:

- a **ServiceSpec** — model + traffic distribution + TTFT/TPOT SLOs + power cap, and
- a **ClusterSpecV2** — accelerator inventory + topology graph,

the planner **enumerates deployment candidates, predicts each one's performance and
energy via LLMServingSim, and emits a DeploymentPlan** that maximizes
**SLO-goodput per joule** under a power cap, plus Pareto alternatives.

**Central abstraction — the execution island:** a set of accelerators sharing one
runtime backend (`cuda`, `ascend`, …), mutually reachable by collectives, able to
host one vLLM engine. TP/PP live **only inside** an island; heterogeneity is
exploited at **replica** or **Prefill/Decode-role** granularity, never inside a TP
group (no cross-vendor TP).

**Core metric:** `SLO-goodput/J = tokens of SLO-satisfying requests / total joules`.
Percentiles (P50/P95/P99) are the headline latencies, never means.

---

## 2. System architecture

```
ServiceSpec + ClusterSpecV2
  → detect_islands()                     (planner/inventory.py)
  → candidate enumeration + pruning      (planner/candidate_generator.py)
  → [stage 6] surrogate top-K            (planner/optimizer/surrogate.py)   ← opt-in
  → compile to configs/cluster/*.json    (planner/predictor/llmservingsim.py)
  → `python -m serving` subprocess       (concurrent, planner/util/parallel.py)
  → feasibility (hard constraints)       (planner/optimizer/feasibility.py)
  → lexicographic ranking + Pareto       (planner/optimizer/pareto.py)
  → PlannerOutput (recommended + alternatives + rejected_summary)
```

**Four planes:** Control (`planner/`, the new work) · Simulation (upstream
`serving/` + ASTRA-Sim) · Data (real vLLM instances) · Profiling (`profiler/`,
`bench/`).

**Two design invariants that gate correctness:**
1. **Bound pruning is a relaxation of feasibility** — stages 1–5 may reject a
   candidate only when the most optimistic arithmetic already misses a constraint,
   so they never drop the optimum. (The stage-6 surrogate is explicitly *not* a
   sound bound — see §4.7.)
2. **Optimization is lexicographic, never weighted-sum** — feasibility → primary
   objective → tie-breaks (fewest accelerators → least fragmentation → …).

The **exhaustive oracle** (`planner/optimizer/exhaustive.py`, never deleted) is the
yardstick that separates surrogate error from search error.

---

## 3. Implementation status by phase

| Phase | Scope | Status |
| --- | --- | --- |
| **0** Baseline | upstream reproduction, format archaeology | ✅ done (`docs/phase0_formats.md`, `phase0_bench_plan.md`) |
| **1** Spec / inventory / islands | ServiceSpec, ClusterSpecV2, island detection | ✅ done (`inspect-cluster`) |
| **2** Offline planner (MVP) | enumerate → simulate → feasibility → Pareto; oracle | ✅ done (`plan`, oracle-agreement + reproducibility + golden tests) |
| **3** Heterogeneous profiles | 2nd GPU profile, profiler contract, NPU CSV importer | ✅ done (6 accelerator profiles, `CsvProfileImporter`) |
| **4** Real deployment + calibration | vLLM launcher, monitor, sim-vs-real | ✅ done (CUDA; A40 live loop + calibration). NPU launcher is a stub |
| **5** Topology-aware P/D | KV-transfer cost, P/D placement, Level-2 topology | ✅ core done (sim-level + planner-side P/D transfer, per-dim topology). network-aware routing deferred |
| **6** Online replanning | workload estimator, replan triggers | ⛔ not started (requires explicit user approval) |

**Real hardware used.** The project has moved machines twice, so this is per-host:

- **Current host — the NPU server (since 2026-08-25).** 4 × FuriosaAI RNGD
  (47.5 GiB, 8 PEs each) + 4 × Rebellions ATOM, 96 cores, 1.5 TB RAM, and **no
  NVIDIA GPU at all**. Shared Kubernetes node; another tenant's
  `rngd_pd.serving.cluster` pods hold most devices, so availability must be
  re-checked per run (`/sys/class/rngd_mgmt/rngd!npu<N>pe<M>/alloc_status`).
- **Earlier hosts.** 8 × NVIDIA A40 (46 GB, measured) and 2 × RTX A5000 (measured).
  Their profiles and bench artifacts are retained as measured artifacts of *those*
  machines. RTXPRO6000 is vendor-spec.

Consequence: **real-vLLM `bench/` runs on CUDA are no longer possible here**, and
the one measurement the heterogeneous P/D question still needs — the GPU→host leg
of the cross-vendor KV path — is blocked on this host by design, not by effort
(`docs/HANDOVER_A40.md`).

---

## 4. Key results

All figures live in `experiments/figures/` (regenerate with
`experiments/scripts/make_figures.py`); every result has a `*_summary.md` and a
committed JSON under `experiments/results/`.

### 4.1 Exp 1 — same-GPU TP=1/2/4 sweep (measured A40 profile)
`experiments/figures/exp1_tp_sweep.png` · `exp1_summary.md`

Validates the planner pipeline across tensor-parallel degrees on one real A40
class (Llama-3.1-8B, 300 req @ 10 rps). **TP=4 was profiled on real hardware for
this experiment** (dummy-weight layerwise; TP emulated by shrinking layer dims).

| TP | p99 TTFT | p99 TPOT | throughput | tok/J | avg power |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 106,109 ms | 225 ms | 2,496 tok/s | 1.860 | 564 W |
| 2 | 35,230 ms | 100 ms | 5,008 tok/s | 2.536 | 831 W |
| 4 | 4,533 ms | **52.1 ms** | 8,842 tok/s | **2.879** | 1,292 W |

**Headline:** monotonic TP scaling on every axis; under this saturating load higher
TP is *also* more energy-efficient (tok/J 1.86→2.88). A single 4-GPU island
saturates at this offered load — TP=4 is the only config clearing TTFT and misses
TPOT p99 by just 4 % (52.1 vs 50 ms).

### 4.2 Exp 2 — heterogeneous resource selection
`experiments/figures/exp2_selection.png` · `exp2_summary.md`

Best SLO-goodput/J per placement class (A5000 + RTXPRO6000, light load):

| class | goodput/J | devices |
| --- | ---: | ---: |
| RTXPRO6000 | **1.697** | 1 |
| A5000 | 1.634 | 2 |
| mixed | 1.043 | 2 |

**Headline:** **right-sizing beats scale-out** — one big GPU edges two small ones,
and mixing both is worst when demand fits a single class. Heterogeneous mixing pays
only when demand exceeds one class's capacity.

### 4.3 Exp 3 — network-bandwidth sensitivity (P/D)
`experiments/figures/pd_network_sweep.png` · `exp_pd_summary.md`,
`pd_sim_network_sweep_table.md`

The §5.9 P/D adoption crossing — the fabric bandwidth where a P/D split stops
paying — reproduced **two ways**: (a) planner-side (analytical KV-transfer cost),
crossing near 10 GB/s for the chosen SLO band; and (b) **sim-level** (deviations
D15): the first sanctioned upstream `serving/` edit makes the simulator's *own* P/D
latency/TPOT bandwidth-sensitive (`--pd-transfer-model bandwidth`), so latency/TPOT
rise monotonically as bandwidth drops while TTFT stays flat.

### 4.4 Exp 5 — heterogeneous P/D 4-combo
`experiments/figures/pd_4combo.png` · `pd_4combo_table.md`

GPU/NPU × prefill/decode, four combos vs an aggregated baseline:

| config | tok/J | feasible? | provenance |
| --- | ---: | :---: | --- |
| GPU-P + GPU-D | 1.081 | yes | sim (RTXPRO6000) |
| GPU-P + NPU-D / NPU-P + GPU-D / NPU-P + NPU-D | 1.081 | yes | **SIM-PROXY** |
| aggregated | 1.655 | **no** | sim |

**Headline:** the aggregated baseline is more efficient (1.655 tok/J) but
**infeasible** (misses SLO), so P/D pays *here by meeting the SLO*. The four P/D
combos are byte-identical because the NPU is a relabelled GPU (SIM-PROXY) — that
identity is the proof, not a result.

> **SUPERSEDED for the GPU-vs-NPU comparison, 2026-08-26.** This table's NPU rows
> are SIM-PROXY and stay here only as the record of that identity check. RNGD is
> now measured, and **§4.8.7** answers the same question properly — across 8 TTFT
> SLO points, two device abstractions, and with the cross-vendor directions
> enumerated rather than stood in for. The re-run also found that a
> *quantitative* 4-combo comparison is not the right instrument: three of its
> combos are unsimulatable at dp=1 because RNGD decode runs out of KV, which is a
> finding about the fixture, not about P/D. The SLO sweep is the instrument.

### 4.5 Baselines + ablation
`experiments/figures/baselines_regret.png` · `exp_baselines_summary.md`

Regret vs the exhaustive oracle (oracle goodput/J = 3.138), simulate-once /
replay-select:

| strategy | regret | note |
| --- | ---: | --- |
| proposed (sim-guided) | **0.000** | matches the oracle — pruning is sound |
| greedy (analytical) | 0.000 | optimum is obvious here |
| **No-Energy** ablation | **0.470** | energy-blind → over-provisions (biggest lever) |
| homogeneous-P/D | 0.331 | P/D doesn't pay at this scale |
| simulator-blind | 0.327 | naive provisioning over-provisions |

**Honest N/A rows (not fabricated):** most-efficient-only (A5000 infeasible at this
SLO), heterogeneous-P/D (all-CUDA cluster), and No-Calibration / No-Uncertainty /
Static (the goodput/J objective reads sim attainment/energy, not latency, so
calibration/margins move only feasibility — meaningful only after Phase-4
calibration / Phase-6 replanning).

### 4.6 Router baselines (RR / RAND / LOAD)
`experiments/figures/router_baselines.png` · `exp_router_summary.md`

On a heterogeneous 4-replica deployment (A5000×2 + RTXPRO6000×2):

| policy | p99 TTFT | goodput |
| --- | ---: | ---: |
| **LOAD** | **314 ms** | 3.75 rps |
| RR | 344 ms | 3.68 rps |
| RAND | 644 ms | 3.56 rps |

**Headline:** load-aware routing wins the tail; RAND is ~2× worse because it
overloads the slow A5000 replicas. Energy is flat — the router moves latency, not
efficiency.

### 4.7 Surrogate top-K + parallel simulation (scaling)
`experiments/figures/surrogate.png` · `exp_surrogate_summary.md`

**Stage-6 surrogate top-K** — a roofline ranker scores all candidates, only the
top-K are simulated. Measured recall/regret vs the oracle (N=78 candidates):

| K | recall@K | regret@K | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0 | **0.000** | 78× |
| 20 | 1 | 0.000 | 3.9× |

**Headline:** regret is 0 even at K=1 (78× fewer sims) although recall reaches 1
only at K=20 — the objective has ties, so the surrogate picks a *different candidate
of equal value*. Measuring **both** recall and regret is what reveals this;
accuracy is measured, never asserted. The surrogate is a **heuristic, not a sound
bound** — it can drop the optimum, which is why the oracle remains the yardstick.

**Parallel candidate simulation** — each candidate is an isolated subprocess, so
they parallelize with no locking. Result is **byte-identical** to sequential (sim
parallel, assembly in candidate order); `plan --workers N`.

| | sequential | parallel (32-wide) |
| --- | ---: | ---: |
| 78-candidate run | ~40–60 min | **~8 min** |
| CPU load (64 cores) | ~2.7 | ~21 |

---

---

## 4.8 RNGD on real silicon — the measured NPU arm

*Added 2026-08-27, covering the 17 commits on `feat/rngd-profiling`. This is the
first measured NPU work in the project and it replaces the SIM-PROXY arm.*

RNGD has **no vLLM platform plugin**, so the A40 route (vLLM's own layerwise
profiler) does not exist here. Two independent instruments were built instead, and
the disagreement between them turned out to be the most useful result.

### 4.8.1 Instrument 1 — the `furiosa.torch` layerwise harness

`experiments/scripts/profile_rngd.py` compiles one canonical layer of
`profiler/models/llama.yaml` at a time through
`torch.compile(backend=furiosa.torch.backend)` inside an `RNGDProfiler` context and
records device spans. `time_us` is the **union** of spans, not their sum, because
tensor units run concurrently and DMA overlaps compute (measured on
`down_proj@256`: TuExec 1603 µs + DMA 964 µs summed, but 1250 µs union — equal to
the timeline's full extent).

Imported to `profiler/perf/RNGD/meta-llama/Llama-3.1-8B/bf16/` at tp1/2/4/8 via the
Phase-3 `CsvProfileImporter`. **Parallelised one worker per PE: 315 s against
40–60 min serial.**

### 4.8.2 Measured device facts

| quantity | value | how |
| --- | --- | --- |
| per-PE addressable memory | 6.25 GB | single-allocation bisect |
| card memory | 47.5 GiB | `furiosa-smi` |
| achieved HBM read, 1 PE | ~219 GB/s | timed weight reads |
| achieved HBM read, card | **~1750 GB/s** | 8 PEs concurrently sustain 218.8 GB/s **each** — 104 % scaling, no degradation |
| board power | **`38.01 + 32.71 × PEs` W**, R² 0.996 | 0..8 loaded-PE sweep |
| card active / idle | 290.93 / 39.35 W | same sweep; the 8th PE adds only +17.4 W where PEs 1–4 add ~31 W, so 291 W is a plateau |
| host↔PE transfer, sustained | 3.77 GB/s single, **26.27 GB/s** at 8 streams | 87.1 % of ideal; corrected 2026-08-27, see below |
| **on-package all-reduce, TP=8** | **115 µs per decoder layer** | §4.8.5 |

A single-PE power reading understated the card by **4×** (68 W against 285.5 W);
that is why the sweep exists.

The host↔PE row was **corrected on 2026-08-27**. It previously read 5.06 GB/s
single and 35.47 GB/s at 8 streams, but only the single-stream figure had ever
been committed; the multi-stream ones existed as prose alone. Measured from
committed code (`outputs/rngd_profile/parallel_bandwidth.json`) the near-linear
PE scaling holds — 87.1 % of ideal against the claimed 88 % — while every
absolute figure drops ~25 %, because the old numbers were peak single transfers
and a KV handoff is a sustained bulk copy. The old method still reproduces 5.06
GB/s on the same card, so nothing about the hardware changed.
`experiments/results/rngd_parallel_bandwidth.md` has the decomposition.

### 4.8.3 Instrument 2 — FuriosaAI's own EDF profiler, and what it revealed

The documented `furiosa.runtime.profiler` / `FURIOSA_PROFILER_OUTPUT_PATH` path does
not exist on the installed stack. What works — found by reading strings in
`furiosa/native_runtime.*.so` — is `EDF_PROFILER_OUTPUT_PATH` gated by
`TUC_PROFILE_LEVEL=info` + `RUST_LOG=span::tuc=info`. Output is **CSV, not JSON**
(`leader_device,name,cycle`), and the clock falls out as 1.6 GHz (total device
cycles ÷ wall time = 1599.9 MHz, 0.006 % off a round figure — consistent only with
a saturated card, which 5 concurrent requests on one card is).

Three structural facts about the vendor runtime, none of them documented:

1. **`tensor_parallel_size: 8` is two fused 4-PE quads, not eight ranks.**
   `leader_device` is `npu0pe0-3`, and the serve log confirms
   `DpId(0) → [npu0pe0-3, npu0pe4-7]`. A per-PE `tp8` bundle therefore modelled a
   rank granularity the hardware does not use.
2. **The runtime compiles two plans for the same model.** Batch 1 runs a fully
   fused `Composed` graph — 98.8 % of device cycles at concurrency 1, and only
   1.7 % at concurrency 4, which is why a single-concurrency look missed it
   entirely. Batch ≥ 2 runs per-layer `Tokenwise` + `Attention`, exactly 32
   `Tokenwise` per forward. Consequence: **no `input_size: 1` bucket exists
   anywhere in 1.74 M stage executions.**
3. **Our harness under-measures per-layer decode by 1.5–1.72×** — 507 µs on the
   real graph against 290–307 µs synthetic. Both instruments agree decode is
   **flat in batch size** (vendor 502.2 µs at bucket 4, 495.8 at 8; harness 306.6 /
   290.4 / 295.0 / 304.9 µs at 1/4/8/16), which is the memory-bound signature
   confirmed rather than inferred.

### 4.8.4 Sim-vs-real, and the bundle rebuilt from the vendor's own measurements

Against a real `furiosa-llm serve` run, 20 sharegpt requests, token-for-token
(13,787 generated against 13,787 requested):

| bundle | TTFT mean | TPOT mean | latency mean |
| --- | ---: | ---: | ---: |
| **real** furiosa-llm TP=8 | 1404.1 ms | **28.4 ms** | 20941 ms |
| per-PE harness, 8 acc @ tp8 | −32.6 % | **+25.7 %** | +21.5 % |
| card-as-device fed harness numbers | −77.9 % | **−45.5 %** | — |
| **card-as-device fed EDF stage times** | −71.3 % | **−3.1 %** | −7.9 % |

**The headline of the first row pair is the sign, not the size: the simulator is
optimistic on prefill by a third and pessimistic on decode by a quarter.** Quoting
total latency alone (+21.5 %) would have hidden both.

**The rebuild (`profiler/perf/RNGD-CARD/`) is the main result.** Card-as-device
initially failed at −45.5 %, and the diagnosis at the time — "decode is
collective-bound" — was wrong. The real mechanism: a `tp1` instance is charged no
collective by the simulator, and an **EDF stage time already contains the
intra-card reduction**, so feeding EDF times pays for the communication exactly
once instead of never. **The abstraction was never the problem; the input was.**

TPOT is now within 3.4 % at every percentile, and **+2.4 % on an unloaded
sparse-arrival run** (15.43 ms against a measured 15.07 ms) — two operating points
a factor of two apart in absolute cost, which is what makes it a cost model rather
than a fit to one point. Self-consistency across a 32× concurrency range: +0.0 /
+0.1 / +0.5 / −4.7 / −5.2 / +0.2 %.

**What in that bundle is measured and what is inherited** (stated because no single
row should be misquoted): absolute per-decoder-layer latency, decode and prefill
attention, and the head are **measured**; the split of a layer's stage time across
`qkv_proj` / `o_proj` / … and the head's scaling with sequence count are
**inherited from the harness**, because the compiler fuses a whole decoder layer
into one stage. The simulator only ever *sums* the per-layer lookups, so the sum is
what has to be right. Full account: `experiments/results/rngd_edf_bundle_notes.md`.

> **RETRACTED 2026-08-28 — it was the validation harness, not the scheduler.**
> This paragraph read: "the residual TTFT error is a scheduler artefact... the
> stretch to −71.3 % is upstream's scheduler queuing ~2.2× less than
> furiosa-llm's." **There is no such scheduler difference.** `python -m serving`
> replays the trace's `arrival_time_ns` (spread over 1.78 s) while
> `bench_furiosa_endpoint.py` fires all 20 requests at once under
> `Semaphore(concurrency=64)` and never reads that column. Matched arrivals give
> **−5.1 %**, and the per-request distributions line up, not just the means.
> Deviations **D19**, `experiments/results/rngd_ttft_gap_resolved.md`.

The −36.6 % unloaded-prefill gap is a separate and still-genuine finding: it comes
from a sparse-arrival comparison of 6 requests where nothing queues, and ~11 % of
it is bucket quantisation. What was wrongly attributed to the scheduler is only
the stretch from there to −71.3 %.

### 4.8.5 The on-package all-reduce, measured

`furiosa.torch` exposes no `all_reduce` API — collectives live inside the compiled
EDF — so device fusion is the only way to make the compiler emit one. Per group
size N, in two separate processes (`set_fusion()` is process-global and one-shot):
`SHARD` = `Linear(in/N, out)` on **one** PE (one rank's compute, no group to reduce
across); `FULL` = `Linear(in, out)` on the fused N-PE device (same compute N-ways
in parallel, plus the reduction).

| N | `down_proj` | `o_proj` | per decoder layer |
| ---: | ---: | ---: | ---: |
| 1 (control) | 0.22 µs | −0.01 µs | — |
| 2 | 3.67 | 1.98 | 5.65 |
| 4 | 13.51 | 10.48 | 23.99 |
| **8** | **49.97** | **65.11** | **115.08** |

**This corrects two earlier estimates, both too large — one by 40×.** The first
version of the probe ran both legs on the fused device, so the reduction cancelled
in the subtraction and what remained was mostly the extra weight traffic of an
N-times-larger layer; it reported 139/107/95 µs and *decreased* in N, which should
have been the tell. The fusion=1 control could not catch it (at N=1 the two legs
are the same graph on any device), so a second check was added: `N × SHARD(N) ≈
SHARD(1)`, now 1.011–1.130 on every row.

What it settles:

- **"Decode is collective-bound" was wrong.** 115 µs/layer × 32 = 3.68 ms per
  *forward*, i.e. ~0.86 ms/token at ~4.3 tokens/forward — about **7 %** of the
  12.92 ms gap it was blamed for, not 45 %.
- **The 212 µs/layer "unaccounted" gap is decomposed**: 115 µs is the reduction
  (54 %), ~97 µs is other work the compiler fused into that stage.
- **The per-PE model's +25.7 % TPOT error has a quantified suspect**: ASTRA-Sim
  appears to charge ~340 µs/layer against a measured 115 — roughly 3× too much.
  That figure is *derived* from the TPOT ratio, not measured; the 115 µs is the
  measurement and the calibration target.
- **8-PE fusion nodes do exist**, contradicting what both RNGD profiles asserted.
  `set_fusion(8)` enumerates and works (24/12/6/3 devices at fusion 1/2/4/8, 7.2×
  per-rank speedup on `o_proj`). Corrected in both profiles; it was never the
  load-bearing argument for `max_tp_size`.

### 4.8.6 Which RNGD profile to use — neither dominates

| | `furiosa_rngd.yaml` (PE-as-device) | `furiosa_rngd_card.yaml` (card-as-device, EDF) |
| --- | --- | --- |
| accelerator | one PE, 6.25 GB, `max_tp 8` | one card, 47.5 GB, `max_tp 1` |
| TPOT error | +25.7 % (conservative) | **−3.1 %** |
| TTFT error, harness-matched arrivals | +42.8 % | **−5.1 %** |
| TTFT error, as first reported | −32.6 % | −71.3 % (both invalid — D19) |
| fitted calibration error | TPOT 0.204, TTFT −0.263 | **TPOT 0.019, TTFT 0.103** |
| use it for | nothing it is uniquely better at | **everything, TTFT included** |

> **The recommendation in this table was inverted before D19.** It read "use the
> per-PE profile for TTFT-feasibility decisions" because −32.6 % looked better than
> −71.3 %. Both figures came from the mismatched harness. With arrivals matched the
> card profile is better on **both** axes, and the per-PE profile's −32.6 % turns
> out to have been two errors cancelling: a queue 2.2× too short times a prefill
> cost 93 % too high (304.9 ms against a real ~158 ms).

The card profile also unblocks two structural problems: 47.5 GB removes the
decode-KV exhaustion that crashed P/D simulations, and its TP set {1} overlaps an
A40 island's {1,2,4} so cross-vendor P/D is expressible without the PCIe-bridging
fixture that deviation D16 describes.

### 4.8.7 Does heterogeneous RNGD+GPU P/D ever pay?

> **Envelope caveat (D19 follow-up, 2026-08-28).** The card fixture's winner runs
> each RNGD card at ~76 concurrent sequences, against 16.6 in the validation run
> and 32 the highest ever tested. Extrapolating the measured scaling curve puts the
> card ~1.6x below what the simulator assumes there. The *ordering* of the regimes
> below is unaffected -- it is driven by energy and TTFT feasibility, not by that
> throughput margin -- but no absolute TTFT figure from the card rows should be
> quoted. `experiments/results/pd_slo_sweep.md`. Settling it needs a c64/c128 run
> on the hardware: `docs/npu_concurrency_envelope_work_order.md`.

Two 8-point TTFT-SLO sweeps (300 requests, seed 42), one per fixture.
`experiments/results/pd_slo_sweep.md`.

**Three regimes, per-PE fixture:**

| TTFT p99 SLO | recommended | tok/J | goodput | p99 TTFT |
| ---: | --- | ---: | ---: | ---: |
| ≤ 64 s, ≤ 32 s | `agg[furiosa:tp8]` | **4.956** | 3.75 | 29547 ms |
| ≤ 16 / 8 / 4 s | `agg[cuda:tp4]` | 2.963 | **5.92** | 2972 ms |
| ≤ 2 / 1 / 0.5 s | `P[cuda:tp4] D[cuda:tp4]` **pd_split** | 2.206 | 5.78 | **372 ms** |

So RNGD wins on energy by **1.67×** where TTFT can be loose (481 W against
1283 W), and **P/D disaggregation is what buys the sub-second end** — 8× lower p99
TTFT for 25 % of the energy efficiency. That is a clean demonstration of what P/D
buys, on the arm whose profile is trustworthy to ~2 %.

**But "heterogeneous P/D does not pay" is not a supported conclusion**, and the
write-up says so three ways:

1. Of the two cross-vendor directions, **the promising one crashed 6/6** under the
   per-PE profile — `A40 prefill → RNGD decode` died on RNGD decode-KV exhaustion
   (39 MB wanted, 3.49 MB free). It was never evaluated, so its absence from the
   ranking is not evidence.
2. The direction that did run is the **inverted** one and is dominated — yet it has
   the **best p99 TPOT of any family** (28.2 ms against 49.4), so pairing the
   vendors does help decode; putting prefill on RNGD at tp4 ruins it.
3. The shape the literature recommends — `A40 tp4 prefill + RNGD tp8 decode`, big
   TP on the memory-bound phase (NVIDIA Dynamo, AWS Neuron) — is **unrepresentable
   under D14's `tp_p == tp_d`**, and tp8 holds 4× the KV of tp4-dp1 on exactly the
   side that needs it.

**Card-fixture re-run: the blocker is gone.** `A40 tp1 P → RNGD-card tp1 D` now
simulates at 2.779 tok/J — **88 %** of the winner's 3.164, against 45 % under the
per-PE profile — and now loses on **SLO attainment (0.74)** rather than being
dominated on every axis. Its raw ranking flip (a 480 ms p99 TTFT winner) is an
artefact of the card profile's prefill error; calibrated (`real = 2.089·sim +
646 ms`) that becomes **~1649 ms**, so every card-sweep row at SLO ≤ 1 s is an
artefact and the winner clears ~2 s, not 500 ms.

> **Do not read the simulator's ~558 W as a card power figure.** It prices the
> whole **node** — accelerator plus base, CPU, DRAM, link, NIC, storage — against
> the card's measured 290.93 W active. That is the right `tokens/joule`
> denominator per work order §3, and both fixtures use it, so cross-fixture tok/J
> stays comparable.

**No RNGD P/D result is a deployment claim.** FuriosaAI's own llm-d documentation
states Furiosa-LLM does not support prefill/decode disaggregation at all today.

## 5. Engineering & discipline highlights

- **Sim-vs-real calibration (A40, measured).** Linear fit `real = α·sim + β`:
  TTFT α=1.017 β=111 ms, TPOT α=1.011 β=−0.02; nominal mean abs error ~1.3–2 %.
  (`profiles/calibration/a40.yaml`.)
- **Provenance on every result** — git commits, versions, spec hashes, seed, full
  command line (`planner/util/provenance.py`), so any figure traces to its inputs.
- **Reproducibility** — same spec + seed ⇒ byte-identical plan; enforced by tests.
- **Deviations ledger** (`docs/deviations.md`, **D1–D17**) records every place
  upstream reality diverges from the work order and how HeteroPilot adapts — e.g.
  D10 (memory model over-estimates KV by +71 % → explicit derating), D3/D15 (no
  link graph in the sim config → Level-2 per-dimension bandwidth + sim-level P/D
  transfer), D16 (no on-package `LinkType`; cross-vendor P/D needs a shared TP
  degree), **D17** (the §3.7 attention grid cannot express the vendor runtime's
  per-layer attention, which tracks the batch's KV *diversity* rather than its
  size — 1.95 executions/layer at batch 2, 3.08 at batch 29).
- **Sim-vs-real calibration (RNGD, measured).** `profiles/calibration/rngd.yaml`
  (per-PE) and `rngd_card_edf.yaml` (card). The card fit's TPOT mean relative error
  is **0.025**; its TTFT error is **2.34**, and that is recorded as *unusable*
  rather than quietly shipped.
- **Retractions are written inline, not deleted.** Three numbers were withdrawn
  during this work — a ≤202 µs per-collective figure (a per-token residual divided
  by a per-forward count), a 139/107/95 µs all-reduce reading (a probe whose two
  legs shared a device, so the reduction cancelled), and a "decode is
  collective-bound" diagnosis (worth ~7 % of the gap, not 45 %). Each retraction
  sits next to the claim it replaces, with the arithmetic that was wrong.
- **Two instruments beat one.** Every significant RNGD finding came from a
  *disagreement* between the layerwise harness and the vendor's EDF profiler, not
  from either alone. The 1.72× per-layer gap, the two compiled plans, and the fused
  4-PE quads were all invisible to a single instrument.

---

## 6. What remains (and why)

| Item | Status | Blocker |
| --- | --- | --- |
| **Measured NPU profiles** — RNGD | ✅ **done**, twice | Layerwise harness (`profiler/perf/RNGD/`, tp1/2/4/8) *and* vendor EDF rebuild (`profiler/perf/RNGD-CARD/`). §4.8 |
| **RNGD per-PE power split** | ✅ **done 2026-08-25** | `board = 38.01 + 32.71 × PEs`, R² 0.996, from a 0..8 loaded-PE sweep. This was the Exp-4 blocker and it is gone |
| **RNGD tokens/J** | ✅ **unblocked** | Power, memory, bandwidth and latency all measured; the node-level denominator is the simulator's, documented in §4.8.7 |
| **Exp 4 — GPU vs NPU island (SLO-goodput/J)** | superseded in substance | The SLO sweeps of §4.8.7 answer the same question across 8 SLO points and two device abstractions. A dedicated Exp 4 run would add a figure, not a finding |
| **GPU→host leg of the cross-vendor KV path** | ✅ **done 2026-08-27** (A40 server) | Sustained pinned D2H 25.71 GB/s single stream, 82.63 GB/s across 8 GPUs; saturates ~83 GB/s at 40 % of ideal against the NPU leg's 87 %. Unlike RNGD the A40 sustains its peak (0.998), so no D18-style correction was needed — but both legs are now composed from sustained figures. Cross-vendor fixture links land at 12.6–13.0 GB/s against a 35 GB/s placeholder that was 2.7–4.5× too optimistic; still above Exp 3's ~10 GB/s crossing. `experiments/results/gpu_host_bandwidth.md` |
| **Measured NPU profiles** — ATOM | placeholder stub | `rebel-compiler` 0.11.0 vs `vllm_rbln`/`optimum-rbln` expecting 0.10.2, **and** all four ATOMs held by another tenant's pods |
| **The −71 % TTFT gap (card profile)** | ✅ **resolved 2026-08-28** (retraction) | Not a scheduler difference. The simulator replayed spread arrivals against a bench that fires everything at once and ignores the trace's `arrival_time_ns`. Matched arrivals: **−5.1 %**. Both TTFT calibrations refitted (card fit error 2.34 → 0.103). Deviations D19, `experiments/results/rngd_ttft_gap_resolved.md` |
| **ASTRA-Sim collective over-charge** | target measured | 115 µs/layer measured against ~340 µs apparently charged. Calibrating `link_bw`/`link_latency` in `rngd-llama31-8b-tp8.json` should shrink the per-PE model's +25.7 % TPOT error |
| **Asymmetric TP per phase** (D14) | structural | `A40 tp4 prefill + RNGD tp8 decode` — the industry-recommended shape — cannot be enumerated. Card-as-device *sidesteps* it by folding TP=8 inside the device; it does not lift the constraint |
| **D12 prefix-cache memory growth** | open, blocks Phase 2 | Two attempted fixes were wrong and reverted; `serving/` is pristine. Read D12 before retrying |
| **Ablation extensions** (No-Calibration / No-Uncertainty / Static) | labelled N/A | Need Phase-4 calibration in the plan path / Phase-6 replanning |
| **Learned (xgboost) surrogate** | analytical shipped | Needs a real multi-hardware training corpus |
| **Network-aware routing** (`_custom_select`) | deferred | Uncertain value; several upstream+planner edits |
| **Phase 6 online replanning** | not started | **Requires explicit user approval** |

**Update 2026-08-25 — the hardware arrived and RNGD is now measured.** The
project moved to the NPU server (4 × Rebellions ATOM, 4 × FuriosaAI RNGD, no
NVIDIA GPU). RNGD has no vLLM plugin, so it was profiled through
`furiosa.torch`'s torch.compile backend instead and imported via the Phase 3
`CsvProfileImporter`: `profiler/perf/RNGD/meta-llama/Llama-3.1-8B/bf16/` at
tp1/2/4/8, verified by a 20-request simulation. Details and caveats:
`docs/hardware_roadmap.md` "First access".

**Update 2026-08-27 — the power blocker is resolved and the bundle was rebuilt.**
The paragraph that stood here said Exp 4 was blocked on power attribution: board
power reads per *card* while one accelerator is one PE, so an even 8-way split was
an assumption and the block stayed `source: placeholder`. **That sweep has been
run** — `board = 38.01 + 32.71 × PEs`, R² 0.996 — so the per-PE cost is measured
rather than assumed and both RNGD profiles now carry `source: measured` power. A
single-PE reading had understated the card by 4×, which is why the sweep was
necessary rather than a formality.

Since then the perf bundle has also been **rebuilt from FuriosaAI's own EDF
profiler**, taking decode prediction from +25.7 % to −3.1 % against the real
`furiosa-llm` run, and the on-package all-reduce has been **measured directly**
(115 µs per decoder layer at TP=8), which retracted an earlier "decode is
collective-bound" diagnosis. All of it is in **§4.8**.

ATOM is separately blocked: `rebel-compiler` resolves to 0.11.0 while
`vllm_rbln`/`optimum-rbln` expect 0.10.2, and all four ATOMs were occupied by
another tenant's serving pods.

---

## 7. Figure index (for the deck)

| Figure | Slide topic |
| --- | --- |
| `exp1_tp_sweep.png` | TP scaling: latency ↓, efficiency ↑ |
| `exp2_selection.png` | Right-sizing beats scale-out |
| `pd_network_sweep.png` | P/D pays only above a bandwidth threshold |
| `pd_4combo.png` | P/D vs aggregated; **NPU rows were SIM-PROXY — superseded by §4.8.7's measured sweeps** |
| `baselines_regret.png` | What the sim-guided planner buys (energy-awareness = biggest lever) |
| `router_baselines.png` | Load-aware routing wins the tail |
| `surrogate.png` | Surrogate top-K: 78× fewer sims, 0 objective loss |

No figure exists yet for §4.8. The RNGD material is currently tables and prose in
`experiments/results/rngd_*.md` and `pd_slo_sweep.md`; the three-regime SLO sweep
and the sim-vs-real bundle comparison are the two that would carry a slide.

*Continuation on the NPU server: `docs/HANDOVER_NPU.md`. The one blocked
measurement and its recipe: `docs/HANDOVER_A40.md`.*
