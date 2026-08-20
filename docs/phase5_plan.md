# Phase 5 plan — topology-aware Prefill/Decode (work order §5.3, §5.9, §7, §12)

Planning document (not implementation). Decomposes Phase 5 into increments,
marks exactly what is planner-only vs what requires the first upstream `serving/`
edits, and records the cross-vendor / NPU constraints and the experiment design.

## What already exists (survey, 2026-08-19)
Phase 5 is further along than the phase label suggests:
- **Topology graph + paths + contention-aware bandwidth**: `planner/topology.py`
  has `TopologyGraph.path()`, `effective_bandwidth_gbps()` (splits a link's
  bandwidth by `contention_group` flow count), `path_latency`, and Level-1
  representative values. The Level-2 *policy* (apply real paths to top-K) is what
  is missing, not the graph machinery.
- **P/D data model**: `plan.py` has `Role.{PREFILL,DECODE,AGGREGATED}` (with
  `pd_type` mapping), `ServingArch.PD_SPLIT`, `RoutingPolicy.PD_SPLIT`.
- **Compiler already emits P/D**: `planner/predictor/llmservingsim.py` writes
  `pd_type` per instance and sets a single representative `link_bw`/`link_latency`
  via `topology.reduce_for_simulator()` (Level-1).
- **The simulator already models P/D natively**: `serving/core/config_builder.py`
  validates `pd_type ∈ {prefill,decode,None}`, gives a prefill instance 2× "sender"
  NPUs to model KV send, and pairs prefill↔decode. Example configs exist
  (`single_node_moe_pd_instance.json`, `single_node_pd_per_instance_config.json`).
- **KV sizing**: `planner/util/memory.py` exposes `kv_bytes_per_token`.

**Consequence**: basic P/D (generate → compile → simulate → rank) needs NO
upstream edit. Only *Level-2 path-aware* network modelling and *network-aware
routing* touch `serving/`. This lets us keep `serving/` pristine for the first,
highest-value increment.

## The gap the sim network model has (deviations D3)
The sim cluster JSON carries only a single `link_bw` + `link_latency` (or a flat
per-node list) — no path graph, no `contention_group`. `reduce_for_simulator()`
lossily compresses the ClusterSpecV2 graph to one representative value. So today
every P/D KV transfer is charged the same representative link, regardless of which
islands the prefill and decode sides sit on. Level-2 fixes this and is the reason
§7 unlocks `config_builder.py` at Phase 5.

## Increment 1 — P/D candidate generation (PLANNER-ONLY, no upstream edit) ★ start here
Goal: the planner enumerates and evaluates P/D-split deployments; the existing
compile+sim path scores them.
1. `planner/candidate_generator.py`: emit `ServingArch.PD_SPLIT` candidates —
   a prefill assignment on island A (`Role.PREFILL`) + a decode assignment on
   island B (`Role.DECODE`), each with its own tp/dp. Enumerate the four role×
   backend combos **that the cluster's islands allow**:
   `GPU P + GPU D` (realizable on A40), `NPU P + NPU D`, `GPU P + NPU D`,
   `NPU P + GPU D` (the last three need NPU islands ⇒ simulator-only, §12).
   Keep it behind the existing fixed-order pruning; P/D candidates go through the
   same memory / parallelism / compatibility filters.
2. `planner/topology.py` (or a small `planner/util/kv_transfer.py`): a KV-transfer
   estimator — `kv_bytes = kv_bytes_per_token × prompt_tokens`, transfer time =
   `kv_bytes / effective_bandwidth(path)` + `path_latency`, transfer energy =
   Σ `energy_per_bit` on the path. Uses the *already-built* topology path between
   the prefill and decode islands.
3. Adoption gate (§5.9): only surface / prefer a P/D split when
   `Benefit_of_split > KV_transfer_latency + KV_transfer_energy + queueing_penalty`.
   In increment 1 this is an **analysis annotation** on the candidate (so the
   oracle still simulates everything and the gate never prunes the optimum), not a
   hard pruning stage — a pruning stage must be a relaxation of feasibility (the
   two-invariants rule), and §5.6 declares no P/D constraint.
4. Tests: P/D candidate enumeration; KV-transfer math; **oracle-agreement must
   still hold** with P/D candidates in the space (pruned optimum == exhaustive
   optimum); reproducibility + a golden P/D plan for a small synthetic
   GPU+NPU cluster (simulator-only, clearly labelled).
Risk: candidate-space growth. P/D roughly squares the island-pairing space; this
is where the deferred §5.4 stage-6 **surrogate predictor + top-K** may finally be
needed (currently every candidate is fully simulated). Increment 1 should
`log()` the candidate count and, if it explodes, gate full simulation behind a
cheap analytical top-K before committing to the surrogate.

## Increment 2 — Level-2 path-aware network — SPIKED, DEFERRED (2026-08-19)

**Correction to the original framing.** Increment 2 was scoped as "the FIRST
upstream edit to `config_builder.py`". That was wrong on two counts:
1. **It needs no upstream edit.** The ASTRA-sim analytical backend already accepts
   a per-dimension `link_bw`/`link_latency` LIST (`config_builder._create_network_config`
   → `_normalize_network_dim_values`; upstream commit `72955ea2`, predates the pin;
   D3 documents the array form; `dual_node_moe_dp_ep_intra_inter_instance.json`
   uses `link_bw: [128,16]`). So "Level-2" = the planner emitting a per-dimension
   `[intra, cross]` list where the cross-instance value is the path-aware,
   contention-adjusted bandwidth from `TopologyGraph.path()` +
   `effective_bandwidth_gbps()`. Pure planner change; `serving/` stays pristine.
   The graph-with-per-pair-contention that D3 calls impossible needs ns3, and was
   never the achievable Level-2.

**Decisive spike (verification step 4) — the analytical backend does NOT charge
P/D KV transfer to the cross-instance link.** Ran the P/D config
`single_node_pd_per_instance_config.json` (prefill+decode, isolated `--run-id`,
network.yml verified as `bandwidth: [900, X]`) sweeping the cross-instance
dimension X ∈ {400, 100, 25} GB/s:

| run | network.yml bandwidth | Median/P99 TTFT | Mean TPOT | output |
| --- | --- | --- | --- | --- |
| cross=400 | [900, 400] | 46.56 / 248.59 | 21.89 | — |
| cross=100 | [900, 100] | 46.56 / 248.59 | 21.89 | **byte-identical** |
| cross=25  | [900, 25]  | 46.56 / 248.59 | 21.89 | **byte-identical** |

A 16× swing on the cross-instance dimension changes *nothing* (identical CSVs);
crushing the intra dimension to 10 GB/s at tp1 is likewise inert. The backend is
NOT network-blind, though: a TP=2 control (`single_node_4_instance_2TP.json`,
`link_bw` 900 vs 1) moves P99 ITL 14.27 → 17.25 ms with different output — so it
*does* charge TP-collective cost. It just does not charge the prefill→decode KV
transfer to the cross-instance dimension.

**Root cause (2026-08-20, decisive).** It is NOT a config issue and NOT
specifically an "analytical backend" issue — the simulator does not model the P/D
KV transfer *cost* at all. At `serving/__main__.py:597`, when prefill requests
finish, `router.transfer_prefill_request()` → `scheduler.add_decode()`
(`serving/core/scheduler.py:829`) simply appends the request to the decode
scheduler and `allocate()`s its KV on the decode NPU — **zero simulated-time
delay, no send/recv collective emitted to the Chakra/ASTRA graph**. The prefill→
decode handoff is instantaneous. The `config_builder` "sender NPU" doubling is
memory/topology accounting, not a costed transfer. So no `link_bw` on any
dimension can move a P/D run — the transfer that Level-2 would price simply
isn't simulated.

**Decision: the "feed the sim a path-aware link_bw" form of Level-2 is DEAD** (the
sim ignores it). But the finding reopens increment 2 in a better, planner-only
form — see below.

## Increment 2 (redirected) — planner-side P/D KV-transfer cost
Since the simulator omits the P/D KV-transfer cost, the *planner* adds it, which
is squarely HeteroPilot's job (work order §5's "KV transfer estimator") and needs
no upstream edit:
- `planner/util/kv_transfer.py` already computes `(time_ms, energy_j)` for a KV
  transfer over a topology path (path-aware, `contention_group`-adjusted). Promote
  it from "informational" to the actual P/D transfer-cost term.
- For a `ServingArch.PD_SPLIT` candidate, after the sim returns compute metrics,
  add the KV-transfer penalty to the predicted first-token path (TTFT += transfer
  time; total_energy += transfer energy) using the prefill→decode island path and
  a workload-derived prompt-token count. Apply it identically in oracle and pruned
  modes so oracle-agreement is untouched (it is a post-predict adjustment of a
  selected candidate, not a pruning stage).
- This makes the §5.9 adoption gate computable (`Benefit_of_split > KV_transfer_
  latency + energy + queueing_penalty`) and makes the increment-4 network sweep
  reproducible **at the planning level**: sweeping the fabric bandwidth moves the
  planner's P/D TTFT via `kv_transfer.py`, even though the sim itself is flat.
- Honesty: every P/D plan must carry a caveat that the transfer cost is a
  planner-side analytical add-on, not simulated (the sim models it as free), with
  the bandwidth/path/flow assumptions recorded in provenance.
Open alternative (bigger, later): teach the simulator to charge the transfer
(add a delay in `transfer_prefill_request` or emit a send/recv over the
cross-instance dim) — a real upstream serving/ change; only worth it if a
sim-level (not planner-level) transfer model is needed.

**Status: DONE (planner-side, 2026-08-20).** `planner/optimizer/exhaustive.py::
apply_pd_transfer_cost` adds the KV-transfer penalty to PD_SPLIT candidates'
predicted metrics inside `evaluate_candidates` (the path shared by `search()` and
`oracle()`), using `planner/util/kv_transfer.py` over the prefill→decode topology
path. Per-percentile: `p{50,95,99}_ttft += xfer_ms(input_tokens.p{50,95,99})`;
`total_energy += energy_per_req·completed_requests`; TPOT/power untouched. A
disconnected pair falls back to `topology._inter_island` (interconnect class
default, byte-consistent with `reduce_for_simulator`, never an optimistic 0), and
a recommended plan relying on that default gets a prominent caveat. Verified:
oracle-agreement holds with non-zero transfer (post-predict, feasibility-only,
identical in both modes — it only *strengthens* feasibility, so it can never
prune the optimum); a P/D whose transfer blows the p99 TTFT SLO becomes
infeasible; the **network sweep is reproducible at the planning level** (p99 TTFT
rises monotonically as fabric bandwidth drops); and the **§5.9 adoption crossing**
is observable (P/D recommended at high bandwidth, aggregated at low). pytest 239 /
ruff / mypy clean; serving/ untouched.

This unblocks increment 4 (the network-sweep headline) **at the planning level**
without the ns3 backend: the sweep now moves the planner's P/D TTFT via the
transfer term, even though the simulator itself is flat on P/D transfer. A
sim-level transfer model (upstream) remains the only way to make the *simulator's*
own P/D numbers bandwidth-sensitive.

**Implication for increment 4 (the headline network sweep).** The §5.9 experiment
— "sweep 25/100/200/400G and find the bandwidth where the P/D benefit vanishes" —
**cannot be reproduced on the analytical backend**, because the P/D KV-transfer
cost that the sweep is meant to move is not modelled there. Options to revisit
before promising that result: (a) build/enable the **ns3 backend** (packet-level,
models the transfer — large effort, currently not built, `scripts/compile.sh` has
it commented out); (b) confirm whether a different P/D config makes the sim route
KV transfer over a costed dimension (the "sender NPU" mechanism in
`config_builder` may need specific wiring); or (c) reframe Phase 5's contribution
around what the analytical backend *does* model (TP/replica placement, memory,
TP-collective cost) rather than P/D-transfer-vs-bandwidth. This is a material
finding for the Phase 5 paper story and should be resolved before increment 4.

## Increment 3 — network-aware routing (UPSTREAM: router.py, §2.2)
- `serving/core/router.py`: add an SLO-aware `_custom_select()` hook (the work
  order calls this Phase 5+). Additive; default routing unchanged.

## Increment 4 — experiments (§12 Exp 3 + Exp 5)
- A `experiments/scripts/run_exp_pd.sh` evaluating the 4 P/D combos vs aggregated
  on a synthetic cluster; cross-vendor combos flagged simulator-only.
- Network sweep `25/100/200/400G` (+1/10G stress): reproduce the bandwidth point
  where the P/D benefit disappears (the §5.9 adoption condition crossing). This is
  the headline Phase 5 paper result.

**Status: DONE (planner-side, 2026-08-20).** Both experiments built, run, and
documented; `serving/` untouched; pytest 239 / ruff / mypy clean.

- Driver `experiments/scripts/pd_network_sweep.py` + runner `run_exp_pd.sh`.
  Efficiency requirement met: the simulator is bandwidth-invariant for P/D transfer
  (increment 2), so it simulates each candidate **once**, caches the raw metrics,
  then re-runs only the planner-side transfer cost + feasibility + ranking per
  bandwidth (the envelope cache is bypassed because its key bands the network
  class). One 5-candidate simulation served all six bandwidths.
- **The crossing is observed.** On two RTXPRO6000 islands (Llama-3.1-8B, 2.5 rps,
  TTFT p99 budget 155 ms, objective goodput/J), the recommendation is `pd_split` at
  400/200/100/25 GB/s and flips to aggregated (mixed 1+1) at 10 and 1 GB/s —
  **crossing bandwidth 10 GB/s**. Mechanism: the planner-side transfer term
  (~378/BW ms for the p99 prompt) pushes P/D's p99 TTFT over budget below ~17 GB/s.
  The crossing is TTFT-driven (energy moves negligibly) and its bandwidth is a clean
  function of the SLO slack, `378/(SLO-133.4)`, with a crossing existing only for a
  TTFT budget in `(133.4, 165.8) ms` (below: P/D never meets it; above: the single
  card meets it and wins on tok/J). Full analysis + provenance:
  `experiments/results/exp_pd_summary.md`, figure
  `experiments/figures/pd_network_sweep.png`.
- **4-combo comparison** (`pd_combo_compare.py`): GPU-P+GPU-D runs end-to-end on the
  real RTXPRO6000 profile; the three NPU-touching combos are SIM-PROXY (the NPU
  islands borrow the RTXPRO6000 compute model, `ascend-sim-proxy.yaml`), so every NPU
  row is byte-identical to the GPU row by construction and is labeled as such — no
  NPU numbers are invented (absolute rule 3). What a real NPU result needs is
  documented in the results doc; the planner already enumerates all four combos
  structurally.
- Two simulator facts that shaped the fixtures: (1) the P/D **decode** path has no KV
  admission control and hard-crashes on overflow, so a 24 GB A5000 decode is
  unusable for this workload and RTXPRO6000 (96 GB) is used; (2) the crossing lives
  in a narrow TTFT window because a single strong card is very energy-efficient, so
  P/D is only preferred once a single card misses the SLO.

## Cross-vendor / NPU constraint
No NPU hardware exists here, and the ATOM/RNGD stub profiles have empty
`supported_models` so they are excluded from candidate generation by design. The
three NPU-touching P/D combos are therefore **simulator-only** and need a
dedicated cluster spec with NPU islands whose profiles declare `supported_models`
(a simulator-only fixture, clearly labelled `source: placeholder`, not presented
as measured). GPU P + GPU D is the only combo runnable end-to-end on the A40 (and
even then the live server has no P/D transport — the *simulator* evaluates P/D;
real cross-instance KV transport is out of scope for this phase).

## Absolute-rule / scope guards
- Increment 1 keeps `serving/` pristine. Increments 2-3 are the first sanctioned
  upstream edits (§7) — additive, back-compatible, on their own branches, with the
  pin re-recorded.
- Still out of scope (stop and report): cross-vendor TP, live migration, RL, k8s,
  online replanning (Phase 6, needs explicit approval), full switch-level
  congestion (the contention model stays the simple per-group split).
- `planner/optimizer/exhaustive.py` stays the oracle; every increment must keep
  oracle-agreement green.

## Recommended sequence
1 → (measure candidate blow-up; add surrogate top-K if needed) → 4 (experiments
on the Level-1 model to see if P/D even helps) → 2 (Level-2) only if Level-1
ranking proves link-sensitive → 3 (routing). Stop after increment 1 for review
before the first upstream edit.
