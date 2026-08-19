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

## Increment 2 — Level-2 path-aware network (FIRST UPSTREAM EDIT: config_builder.py)
Goal: charge each P/D KV transfer the *actual* path bandwidth/latency, with
`contention_group` sharing, for the top-K candidates only.
- `serving/core/config_builder.py`: extend the ClusterSpecV2→ASTRA adapter so the
  sim can represent per-instance-pair network cost (not one global `link_bw`).
  This is the D3 resolution. **This is the first edit to `serving/` — do it on a
  dedicated branch, keep the change additive/back-compatible with the existing
  flat `link_bw` configs (all current tests + bench must still reproduce
  byte-identically at the pin), and re-pin/record per absolute rule.**
- `planner/predictor/llmservingsim.py`: for top-K only, emit the Level-2 network
  block instead of the single representative value; Level-1 stays the default for
  bulk scoring.
- `topology.py`: `reduce_for_simulator` gains a Level-2 path-aware mode; provenance
  flips `path_aware: true` for those runs.
Gate: this increment is only worth doing once increment 1 shows P/D candidates
that are close enough that the representative-link error changes the ranking.

## Increment 3 — network-aware routing (UPSTREAM: router.py, §2.2)
- `serving/core/router.py`: add an SLO-aware `_custom_select()` hook (the work
  order calls this Phase 5+). Additive; default routing unchanged.

## Increment 4 — experiments (§12 Exp 3 + Exp 5)
- A `experiments/scripts/run_exp_pd.sh` evaluating the 4 P/D combos vs aggregated
  on a synthetic cluster; cross-vendor combos flagged simulator-only.
- Network sweep `25/100/200/400G` (+1/10G stress): reproduce the bandwidth point
  where the P/D benefit disappears (the §5.9 adoption condition crossing). This is
  the headline Phase 5 paper result.

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
