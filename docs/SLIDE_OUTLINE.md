# HeteroPilot — Presentation Slide Outline

*A slide-by-slide skeleton to build the PowerPoint from. Each slide lists its
title, the bullets to show, the visual to drop in (from `experiments/figures/`),
and a one-line speaker note. Source of every number: `docs/PROJECT_REPORT.md`.
Reminder for every data slide: label numbers **prediction / measured / SIM-PROXY**
(absolute rule 3) — put a small "LLMServingSim prediction" footer on results slides.*

Suggested length: ~18 content slides + 3 backup. Narrative arc: **problem →
approach → method/trust → results → scaling → status/limits → contributions.**

---

## Slide 1 — Title
- **HeteroPilot: SLO-goodput/J-optimal LLM serving on heterogeneous GPU/NPU clusters**
- Subtitle: a simulator-in-the-loop control plane (fork of LLMServingSim)
- Author / affiliation / date
- *Note:* one sentence — "we plan *where and how* to place LLM replicas across
  mixed accelerators to maximize SLO-satisfying throughput per joule."

## Slide 2 — The problem
- LLM serving clusters are increasingly **heterogeneous** (GPU generations, NPUs).
- Operators must choose: which accelerator, tensor-parallel degree, replica count,
  aggregated vs Prefill/Decode split — under **TTFT/TPOT SLOs and a power cap**.
- The search space is huge and the trade-offs (latency vs energy vs SLO) are
  non-obvious; picking wrong wastes energy or misses SLOs.
- *Visual:* a simple cluster cartoon (GPU + NPU islands). *Note:* motivate "energy
  per SLO-satisfying token," not raw throughput.

## Slide 3 — What HeteroPilot does (one slide)
- **Input:** ServiceSpec (model + traffic + SLOs + power cap) + ClusterSpecV2
  (inventory + topology).
- **Output:** a DeploymentPlan maximizing **SLO-goodput/J** under the power cap,
  plus Pareto alternatives, plus a diagnosis when infeasible.
- Method: **enumerate candidates → predict each via LLMServingSim → rank
  lexicographically.**
- *Visual:* input→planner→output arrow diagram. *Note:* "goodput/J is the objective,
  percentiles are the SLO gate."

## Slide 4 — Key abstraction: the execution island
- An **island** = accelerators sharing one backend, mutually reachable by
  collectives, hosting one vLLM engine.
- **TP/PP only inside an island**; heterogeneity is exploited across **replicas** or
  **P/D roles** — never cross-vendor TP.
- Why: cross-backend collectives are impractical; this keeps candidates realizable.
- *Visual:* island diagram (NVLink pair = one island; two islands bridged by fabric).
- *Note:* this is the unit of placement and the reason the space is enumerable.

## Slide 5 — System architecture
- Pipeline: detect islands → enumerate + prune → (opt.) surrogate top-K → compile
  to sim config → **parallel `python -m serving`** → feasibility → Pareto rank.
- Four planes: Control (planner) · Simulation (LLMServingSim + ASTRA-Sim) · Data
  (real vLLM) · Profiling.
- *Visual:* the pipeline block diagram from `PROJECT_REPORT.md` §2.
- *Note:* highlight the **exhaustive oracle** kept as ground truth.

## Slide 6 — Why you can trust the numbers (method & honesty)
- **Simulator-in-the-loop**, calibrated: A40 sim-vs-real fit `real=α·sim+β`
  (TTFT α=1.02/β=111 ms, TPOT α=1.01), **~1.3–2 % mean error (measured)**.
- **Provenance on every result** (git, versions, spec hash, seed, command).
- **Honesty rule:** unmeasured hardware is labelled placeholder / SIM-PROXY, never
  presented as measured.
- **Reproducible:** same spec+seed ⇒ byte-identical plan (tested).
- *Note:* pre-empt "is the sim trustworthy?" — calibrated + labelled + reproducible.

## Slide 7 — Result: TP scaling validates the pipeline (Exp 1)
- Same A40, TP=1/2/4, Llama-3.1-8B (TP=4 **profiled on real hardware**).
- Monotonic: p99 TTFT 106 s→35 s→**4.5 s**; p99 TPOT 225→100→**52 ms**; tok/J
  1.86→2.54→**2.88**.
- Under saturation, more TP is *also* more efficient; TP=4 is closest to SLO.
- *Visual:* **`exp1_tp_sweep.png`**. *Note:* the pipeline reproduces the expected
  physics end-to-end.

## Slide 8 — Result: right-sizing beats scale-out (Exp 2)
- Heterogeneous A5000 + RTXPRO6000; best goodput/J per class:
  **RTXPRO6000 1.697 > A5000 1.634 > mixed 1.043**.
- Mixing pays only when demand exceeds one class's capacity — not before.
- *Visual:* **`exp2_selection.png`**. *Note:* the planner correctly prefers one big
  card here; heterogeneity is a tool, not a default.

## Slide 9 — Result: when does P/D pay? (Exp 3 + Exp 5)
- **Exp 3 (network sweep):** the P/D-adoption crossing — bandwidth below which a
  split stops paying (reproduced both planner-side and **sim-level**, D15).
- **Exp 5 (4-combo):** aggregated is more efficient (1.655 tok/J) but **infeasible**;
  P/D pays *by meeting the SLO* (1.081, feasible).
- *Visual:* **`pd_network_sweep.png`** (+ `pd_4combo.png` as inset/next slide).
- *Note:* **NPU combos are SIM-PROXY** — say it out loud; the four identical rows
  are the proxy proof, not a GPU-vs-NPU result.

## Slide 10 — Result: what the sim-guided planner buys (baselines + ablation)
- Regret vs the exhaustive oracle (oracle goodput/J = 3.138):
  - **proposed = 0.000** (pruning is sound), greedy = 0.000 (obvious optimum here)
  - **No-Energy ablation = 0.470** (energy-blind over-provisions — the biggest lever)
  - simulator-blind / homogeneous-P/D ≈ 0.33
- Honest N/A rows (infeasible class, all-CUDA, calibration not yet in path).
- *Visual:* **`baselines_regret.png`**. *Note:* the headline is **energy-awareness
  is worth ~47 % of goodput/J**.

## Slide 11 — Result: routing on a heterogeneous fleet (router baselines)
- 4 replicas (A5000×2 + RTXPRO6000×2): **LOAD** p99 TTFT **314 ms** vs RAND **644 ms**.
- RAND overloads the slow replicas; energy is flat — routing moves latency, not J.
- *Visual:* **`router_baselines.png`**. *Note:* load-aware routing wins the tail on
  mixed hardware.

## Slide 12 — Scaling 1: surrogate top-K
- A cheap roofline ranker scores all candidates; only the top-K are simulated.
- Measured (N=78): **regret 0.000 at every K down to K=1 (78× fewer sims)**; recall
  reaches 1 only at K=20 — the objective has ties, so an equal-value candidate is
  chosen. Heuristic, so the **oracle stays the yardstick**.
- *Visual:* **`surrogate.png`**. *Note:* accuracy is *measured, never asserted* —
  that's why we plot both recall and regret.

## Slide 13 — Scaling 2: parallel candidate simulation
- Each candidate is an isolated subprocess → parallelize with no locking.
- **~5–7× wall speedup** (78 sims ~8 min vs ~40–60 min; 64-core load 2.7→21).
- **Byte-identical** to sequential (sim parallel, assembly in candidate order).
- *Visual:* a small before/after bar (sequential vs 32-wide) + CPU-load callout.
- *Note:* surrogate cuts sim *count*; parallelism cuts sim *wall time* — complementary.

## Slide 14 — Implementation status (vs the work order)
- Phases 0–5 done (Phase 4 CUDA); Phase 6 gated on approval.
- Table: Phase → status (from `PROJECT_REPORT.md` §3).
- 10 merged PRs; 284 tests green; ruff + mypy clean.
- *Note:* the system is end-to-end runnable today on real GPU hardware.

## Slide 15 — Limitations & honesty
- **No NPU hardware yet** → Exp 4 (GPU vs NPU) not run; Exp-5 NPU rows are SIM-PROXY.
- Analytical backend can't model per-flow link contention (needs ns3).
- Prefix caching off (upstream memory bug, D12); calibration not yet in the plan path.
- *Note:* these are stated in every figure — credibility through disclosure.

## Slide 16 — Future work
- **Measure the NPUs** (Rebellions ATOM ×4, FuriosaAI RNGD ×4): converts Exp 4 and
  the Exp-5 NPU rows from SIM-PROXY to **measured** — the biggest credibility win.
- NPU sim-vs-real calibration → unblocks the No-Calibration/No-Uncertainty ablations.
- Learned (xgboost) surrogate once a real corpus exists; network-aware routing;
  Phase 6 online replanning.
- *Note:* the handover doc (`docs/HANDOVER_NPU.md`) already sequences this.

## Slide 17 — Contributions
1. An **execution-island** formulation that makes heterogeneous placement enumerable.
2. A **calibrated, simulator-in-the-loop** planner optimizing **SLO-goodput/J** with
   an exhaustive oracle for ground truth.
3. **Topology- and P/D-aware** candidate evaluation (sim-level bandwidth sensitivity).
4. **Scalable search** (surrogate top-K + parallel sim) with measured accuracy/speed.
5. A disciplined, **reproducible, provenance-tracked** artifact.

## Slide 18 — Summary / takeaways
- Heterogeneity is a *tool*: it pays only past a capacity/bandwidth threshold.
- **Energy-awareness is the dominant lever** (~47 % goodput/J).
- The planner matches the oracle while a surrogate makes it cheap.
- Next: real NPU numbers close the last honesty gap.

---

## Backup slides
- **B1 — Core metric definitions:** SLO-goodput/J, SLO attainment, J/token, avg/peak W.
- **B2 — Deviations ledger highlights:** D10 (memory +71 % over-estimate → derating),
  D3/D15 (no link graph → per-dim bandwidth + sim-level P/D transfer), D12 (prefix
  cache off).
- **B3 — Reproduce everything:** one-command runners
  (`experiments/scripts/run_exp*.sh`), `make_figures.py`, committed JSON + provenance.

---

*Optional: I can also render this as a self-contained HTML slide deck (viewable in a
browser, exportable) if you want a visual draft before PowerPoint.*
