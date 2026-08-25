# HeteroPilot — Integrated Project Report

*Status snapshot: 2026-08-25. Basis for the presentation deck. Every quantitative
result below is reproducible from the committed artifacts named in each section.*

> **Honesty note (project absolute rule 3).** Except where a row is explicitly
> labelled **measured**, every number here is an **LLMServingSim prediction**, not
> a live hardware measurement. NPU numbers are either **SIM-PROXY** (an NPU stood
> in with a GPU compute model) or placeholder — no NPU hardware has been available.
> These labels are carried through every results file and figure.

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

**Real hardware used:** 8 × NVIDIA A40 (46 GB, measured) on the current server;
2 × RTX A5000 on the original dev box (measured profile retained). RTXPRO6000 is a
vendor-spec profile. **No NPU hardware has been available** (see §6).

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
identity is the proof, not a result. **A real GPU-vs-NPU efficiency story needs NPU
hardware (§6).**

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

## 5. Engineering & discipline highlights

- **Sim-vs-real calibration (A40, measured).** Linear fit `real = α·sim + β`:
  TTFT α=1.017 β=111 ms, TPOT α=1.011 β=−0.02; nominal mean abs error ~1.3–2 %.
  (`profiles/calibration/a40.yaml`.)
- **Provenance on every result** — git commits, versions, spec hashes, seed, full
  command line (`planner/util/provenance.py`), so any figure traces to its inputs.
- **Reproducibility** — same spec + seed ⇒ byte-identical plan; enforced by tests.
- **Deviations ledger** (`docs/deviations.md`, D1–D15) records every place upstream
  reality diverges from the work order and how HeteroPilot adapts — e.g. D10
  (memory model over-estimates KV by +71 % → explicit derating), D3/D15 (no link
  graph in the sim config → Level-2 per-dimension bandwidth + sim-level P/D transfer).

---

## 6. What remains (and why)

| Item | Status | Blocker |
| --- | --- | --- |
| **Exp 4 — GPU vs NPU island (SLO-goodput/J)** | not run | NPU latency is now measured, but the **per-PE power split** is not, and Exp 4's objective is per joule |
| **Measured NPU profiles** — RNGD | ✅ **done 2026-08-25** | Llama-3.1-8B bf16 at tp1/2/4/8, `profiler/perf/RNGD/`; the first measured NPU number in this project |
| **Measured NPU profiles** — ATOM | placeholder stub | Broken vendor install (`rebel-compiler` 0.11.0 vs `vllm_rbln` 0.10.2) and no free ATOM |
| **RNGD tokens/J** | blocked | Board power is per *card*; the per-PE split in the profile is not measured (`docs/hardware_roadmap.md`) |
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

**The remaining blocker for Exp 4 is now narrow and specific: power
attribution.** Board power reads per card (idle 39.0 W, active 285.5 W with all
8 PEs loaded), but the simulator applies the profile's power block per NPU
instance, and one accelerator here is one PE. An even 8-way split is an
assumption, so that block stays `source: placeholder` and no RNGD tokens/J
figure should be quoted yet. Fitting the marginal per-PE cost from a 1..8 loaded
-PE sweep is what unblocks Exp 4 and the Exp-5 NPU rows.

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
| `pd_4combo.png` | P/D vs aggregated; NPU rows SIM-PROXY |
| `baselines_regret.png` | What the sim-guided planner buys (energy-awareness = biggest lever) |
| `router_baselines.png` | Load-aware routing wins the tail |
| `surrogate.png` | Surrogate top-K: 78× fewer sims, 0 objective loss |

*Continuation on the NPU server: see `docs/HANDOVER_NPU.md`.*
