# Exp — Prefill/Decode network sweep + 4-combo comparison (Phase 5 increment 4, simulation)

Reproduce: `./experiments/scripts/run_exp_pd.sh` (docs/phase5_plan.md increment 4;
work order §12 Exp 3 network sweep + Exp 5 P/D combos).

**Everything below is a simulator prediction, never a measurement.** Two honesty
facts frame the whole result and are repeated in provenance:

1. **The P/D KV-transfer cost is a planner-side analytical add-on.** The simulator
   models the prefill→decode handoff as *free* (it charges no transfer time or
   energy; docs/phase5_plan.md increment 2, root-caused in `serving/__main__.py` /
   `scheduler.py`). So the *simulator's* P/D numbers are bandwidth-invariant; the
   entire bandwidth effect below comes from HeteroPilot's own transfer term
   (`planner/optimizer/exhaustive.py::apply_pd_transfer_cost` over
   `planner/util/kv_transfer.py`). A sim-level (ns3) transfer model is the only way
   to make the *simulator itself* bandwidth-sensitive; that is out of scope here.
2. **NPU numbers are simulator-only proxy data.** No NPU hardware exists here and
   the real Ascend stub deliberately fails loudly (`sim_hardware: null`). The three
   NPU-touching combos use an experiment-only proxy that borrows the RTXPRO6000
   compute model (`experiments/configs/clusters/ascend-sim-proxy.yaml`), so every
   NPU row is byte-identical to the GPU row by construction. They are NOT NPU
   measurements and no NPU performance conclusion may be drawn from them.

Provenance (both result JSONs carry the full §3.8 block): git
`632a2772`, upstream pin `2c2042ce`, seed 42, 120-request synthetic ShareGPT-like
trace, prefix caching off (D12), `gpu_memory_utilization=0.90` with the D10 derate.

## How the sweep avoids re-simulating per bandwidth

A full `plan --enable-pd` re-simulates every candidate; at ~2–4 min/candidate that
is hours, and re-running it for six bandwidths would be pointless because the
simulator ignores the fabric bandwidth for P/D transfer (fact 1). Instead
`pd_network_sweep.py`:

1. generates the candidate set once, simulates each candidate **once** with the real
   `LLMServingSimPredictor`, and caches the raw (un-adjusted) metrics to disk;
2. for each bandwidth, rebuilds the cluster with that fabric bandwidth and re-runs
   **only** the planner-side transfer cost + feasibility + ranking
   (`evaluate_candidates` + `pareto`), replaying the cached metrics.

The envelope cache is bypassed on purpose: its key bands the network class
(`planner/envelope.py::network_class`), so sweeping across bands would force a
re-simulation. One simulation pass (5 candidates) served all six bandwidths. Knobs
were restricted to `max_num_seqs=128, max_num_batched_tokens=2048` (single knob
point); the crossing does not need the full knob grid.

## Fixture and regime

`experiments/configs/clusters/pd-network-sweep.yaml`: two single-GPU RTXPRO6000
islands (96 GB each) joined by one declared fabric link, `fabric-node0-node1`, whose
bandwidth is swept. RTXPRO6000 (not the local 24 GB A5000) is used because the
simulator's P/D **decode** path has no KV admission control and hard-crashes when the
decode instance overflows — a 24 GB A5000 decode overflows this workload, a 96 GB
card does not. Two identical cards keep P/D and mixed-aggregated on the same two
devices (equal hardware), so the choice between them is decided by latency/energy and
the swept link is what moves it.

Service (`experiments/configs/services/pd-sweep-llama31-8b.yaml`): Llama-3.1-8B,
2.5 rps, objective `maximize_slo_goodput_per_joule` then `minimize_active_accelerators`,
**TTFT p99 budget 155 ms**, TPOT p99 budget 100 ms. Baseline simulated metrics (all
feasible at 2.5 rps, 100% attainment):

| placement | devices | p99 TTFT (ms) | tok/J | note |
| --- | ---: | ---: | ---: | --- |
| single RTXPRO6000 (aggregated) | 1 | 165.8 | **1.655** | best tok/J, but misses a <165.8 ms TTFT budget |
| mixed 1+1 (aggregated) | 2 | 108.8 | 0.838 | lowest TTFT, worst tok/J |
| P/D split (prefill→decode) | 2 | 133.4 | 1.081 | between the two; beats mixed on tok/J |

The 155 ms budget is chosen to sit in the **adoption window** this regime creates: a
single card *misses* it (165.8 ms) so two devices are required, and between the two
two-device options P/D is the more energy-efficient (1.081 vs 0.838 tok/J) because
its prefill engine idles during decode. This is exactly the §5.9 setting where the
P/D-vs-aggregated decision is real.

## Result 1 — the network sweep (headline)

`outputs/.hp-pd-sweep/pd_network_sweep_table.md`, figure
`experiments/figures/pd_network_sweep.png`.

| fabric BW (GB/s) | recommended arch | rec p99 TTFT (ms) | P/D feasible | P/D p99 TTFT (ms) | P/D xfer p99 (ms) |
| ---: | --- | ---: | :---: | ---: | ---: |
| 400 | **pd_split** | 134.4 | yes | 134.4 | 1.0 |
| 200 | **pd_split** | 135.3 | yes | 135.3 | 1.9 |
| 100 | **pd_split** | 137.2 | yes | 137.2 | 3.8 |
| 25  | **pd_split** | 148.5 | yes | 148.5 | 15.1 |
| 10  | aggregated (mixed) | 108.8 | no | — | — |
| 1   | aggregated (mixed) | 108.8 | no | — | — |

**Crossing bandwidth: 10 GB/s.** Above it the planner recommends the disaggregated
P/D split (lower energy per SLO-token); at and below it the planner-side transfer
term pushes P/D's p99 TTFT over the 155 ms budget, P/D becomes infeasible, and the
planner falls back to replica-parallel aggregation (mixed 1+1). This is the §5.9
adoption condition made concrete: **P/D is worth it only while the fabric is fast
enough that the KV transfer keeps p99 TTFT under budget.**

Mechanics: the transfer term is `latency + prompt_KV / bandwidth`, with prompt KV =
p99 prompt (2884 tok) × 131072 B/tok ≈ 378 MB, so xfer_p99(ms) ≈ 378/BW(GB/s). P/D is
adopted while `133.4 + 378/BW < 155`, i.e. `BW > ~17.5 GB/s`; the highest swept point
below that is 10 GB/s, hence the flip there. Energy moves negligibly (per-request
transfer energy ≈ 0.015 J on the 5 pJ/bit link, ~1.8 J over the run vs ~70 kJ total),
so the crossing is TTFT-driven, not energy-driven.

### SLO sensitivity (the crossing is a function of the TTFT budget)

The crossing bandwidth is `378 / (SLO_ms − 133.4)`, and a crossing exists **only**
for a TTFT budget in the window `(133.4 ms, 165.8 ms)` — below 133.4 ms even P/D
never meets it (never adopted), above 165.8 ms the single card meets it and wins on
tok/J (P/D never preferred):

| TTFT p99 budget (ms) | crossing bandwidth (GB/s) |
| ---: | ---: |
| 140 | 57 |
| 150 | 23 |
| 155 | 17.5 (reported: flips at the 10 GB/s point) |
| 160 | 14 |
| 165 | 12 |

This dependence is itself the finding: with a generous TTFT budget the ≤378 ms
transfer term never bites and P/D is always adopted; with a tight budget it is never
adopted; the interesting adoption boundary lives in a narrow SLO band, and its
bandwidth location is what the sweep pins down.

## Result 2 — the four P/D combos (§12 Exp 5)

`outputs/.hp-pd-combo/pd_4combo_table.md`, fixture
`experiments/configs/clusters/pd-4combo-sim.yaml` (two RTXPRO6000 GPU islands + two
SIM-PROXY NPU islands, all 96 GB, full fabric mesh at 400 GB/s).

| combo | provenance | feasible@155ms | p99 TTFT (ms) | p99 TPOT (ms) | attainment | energy (J) | tok/J | P/D xfer p99 (ms) |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU-P + GPU-D | sim (RTXPRO6000, vendor_spec) | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| GPU-P + NPU-D | SIM-PROXY (RTXPRO6000 model) | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| NPU-P + GPU-D | SIM-PROXY (RTXPRO6000 model) | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| NPU-P + NPU-D | SIM-PROXY (RTXPRO6000 model) | yes | 134.4 | 16.0 | 0.99 | 70,250 | 1.081 | 1.0 |
| aggregated (single, baseline) | sim (RTXPRO6000, vendor_spec) | no | 165.8 | 18.8 | 0.98 | 45,860 | 1.655 | — |

All four P/D combos land on identical numbers (134.4 ms p99 TTFT, 1.081 tok/J) — the
signature of the proxy: the NPU islands run the same RTXPRO6000 compute model as the
GPU islands, so an NPU-touching combo is arithmetically a GPU combo. The single-card
aggregated baseline is infeasible at the 155 ms budget (165.8 ms), which is why a
two-device P/D split is the operative choice in this regime.

**Only GPU-P + GPU-D is a real (RTXPRO6000) result.** The three NPU-touching combos
are SIM-PROXY: the NPU islands run the RTXPRO6000 compute model, so their rows are
identical to the GPU rows by construction — that identity is the proof that they are a
GPU stand-in, not NPU data. What a genuine NPU combo needs (and the proxy fakes):

- a real `profiler/perf/<HW>/` bundle for the NPU (attention/dense/per_sequence/skew
  CSVs) — Phase 3's `CsvProfileImporter` (deviations D4) is the intended path;
- `sim_hardware` pointing at that bundle, measured `memory_bandwidth_gbps` and
  `power`, and `supported_models` reflecting real kernel coverage;
- a cross-vendor P/D transport in the deployer for a *real* (non-simulated) run — out
  of scope for this phase (the simulator evaluates cross-vendor P/D; the live server
  has no KV transport for it).

The planner already **enumerates all four role×backend combos** structurally
(candidate generation admits them the moment an NPU island declares `supported_models`
and a `sim_hardware` bundle); the only missing piece for real NPU numbers is the
profile bundle, not planner support.

## Caveats (read before citing any number)

- The bandwidth effect is entirely planner-side analytical (fact 1); the simulator's
  own P/D numbers do not move with bandwidth.
- All NPU rows are simulator-only proxy (fact 2); no NPU performance claim follows.
- Knobs were restricted to a single point (`max_num_seqs=128`,
  `max_num_batched_tokens=2048`); the crossing does not depend on the full grid but
  absolute TTFT/energy values would shift with other knobs.
- Host-power components are upstream placeholders on every node, so energy magnitudes
  (and thus tok/J) inherit placeholder host power; the *direction* of the P/D-vs-mixed
  energy gap (prefill idles during decode) is robust, the exact kJ is not.
- RTXPRO6000 stands in for the local hardware because the 24 GB A5000 P/D decode
  overflows the simulator; on real 24 GB cards P/D would need a smaller
  `max_num_seqs` or KV offload, which this study does not model.
- Reproducibility: fixed seed (42) + the cached single simulation make the sweep
  table and crossing deterministic across re-runs.
