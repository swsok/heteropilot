# Deviations — work order spec vs. upstream reality

Per the work order's closing instruction: when this document's spec and the real upstream code
conflict, **the real code wins**; the difference is recorded here and work continues.

Each entry: what the work order assumes, what upstream actually does, and how HeteroPilot adapts.
Format evidence is in `phase0_formats.md`. Pin: `2c2042ce`.

Status legend — **Resolved** (adaptation decided, no user input needed) ·
**Open** (needs a decision before the phase it blocks).

---

## D1 — There is no `sim.csv` · Resolved

**Work order §5.5** says `parse_results(output_dir)` reads "`sim.csv` 등 출력".

**Upstream**: the per-request CSV path is whatever `--output` says; the literal `{run_id}` in the
path is substituted with the run id. Without `--output` results go to stdout only. No fixed filename
exists anywhere.

**Adaptation**: the predictor chooses the output path itself, so nothing needs discovering. Use
`--output <tmpdir>/{run_id}.csv` and read back the resolved path.

---

## D2 — Power and energy are stdout-only · **Decided 2026-08-07** (parse stdout in Phase 2)

**Work order §5.5** expects `parse_results` to extract "TTFT/TPOT percentile, throughput,
power/energy" from output files. §3.6 wants `average_power_w`, `peak_power_w`, `total_energy_j`,
`tokens_per_joule` in the envelope DB, and §5.6 makes `PeakPower(x) <= slo.max_cluster_power_w` and
`TokensPerJoule(x) >= slo.min_tokens_per_joule` hard constraints.

**Upstream**: the per-request CSV has no power, energy, or memory column. Energy appears **only** as
Rich-formatted stdout (`Total energy consumption (kJ)`, a per-node component tree, and
`Power per N sec (W): [...]`). Confirmed by running the power example — see `phase0_formats.md` §3.1.

This is a genuine tension: two Phase 2 hard constraints depend on numbers that only exist as
console text, and `serving/core/power_model.py` is not editable until Phase 4 under §7.

**Options**

| | Approach | Cost |
| --- | --- | --- |
| A | Parse the stdout power block with pinned regexes | No upstream change; brittle against Rich formatting and log-level changes |
| B | Pull Phase 4's `power_model.py` API exposure forward to Phase 2 | Clean and machine-readable; breaks the §7 phase gate early |
| C | Recompute energy in the planner from the power config + CSV timings | Duplicates upstream's state machine — violates the "call, don't copy" rule of §5.4 |

**Recommendation: A for Phase 2, B at Phase 4.** Parsing keeps the Phase 5 no-touch rule intact,
and the risk is contained: isolate every regex in one module (`planner/predictor/_power_parse.py`)
with a golden-text fixture test, so a format change fails loudly in CI rather than silently
producing wrong energy. Switch to B when §7 opens `power_model.py` anyway.

Two findings make A far less brittle than it first looks:

- **Upstream already does this.** `bench/core/validate.py::_load_sim_log` parses the simulator's
  stdout with regexes (`_TS_RE`, `_TPUT_RE`, `_INST_RE`) and `bench validate` takes `sim.log`
  alongside `sim.csv` as a first-class input. Log parsing is the established upstream contract for
  metrics absent from the CSV, not a workaround we invented.
- **Redirected output is clean plain text.** Verified on the captured run: zero ANSI escape
  sequences (Rich detects the non-TTY). The only non-ASCII is stable UTF-8 box drawing (`├─`, `└─`)
  in the per-component tree. Anchor regexes on the label text, not the tree glyphs.

Lines to parse (exact, from the reproduction run):

```text
Total energy consumption (kJ):                                      1.42
Node 0 total energy consumption (kJ):                               1.42
├─ NPU energy consumption (J):                                      972.14
Power per 1.0 sec (W): [845.91]
```

Note also that `bench/examples/<model>/outputs/` names its files `sim.csv` / `sim.log`. That is a
bench-example convention, not a simulator-fixed name (D1 stands) — but it is the naming the
validation tooling expects, so the predictor should adopt it for calibration runs in Phase 4.

**Caveat regardless of option**: `Power per N sec (W)` resolution equals `--log-interval`, so
`peak_power_w` is an interval average, not a true instantaneous peak. Enforcing a power cap against
it is optimistic. Pick the interval deliberately and record it in provenance.

---

## D3 — The cluster config has no topology graph · **Decided 2026-08-07** (Level-1 now, compare later)

**Work order §3.2.2** defines a rich `links` graph: per-link `src`/`dst`, `type`, `bandwidth_gbps`,
`latency_ns`, `energy_per_bit_pj`, `duplex`, `contention_group`. §5.3 builds a `TopologyGraph` on it
and computes bottleneck bandwidth by dividing each link's bandwidth by its `contention_group` flow
count.

**Upstream**: the cluster config carries exactly two topology fields — `link_bw` (GB/s) and
`link_latency` (ns) — each a scalar or a per-dimension array matching `network.yml::npus_count`.
There is no notion of a named link, an endpoint pair, a contention group, or per-link energy.

**Consequences**

- `ClusterSpecV2` **cannot round-trip**. Compilation is lossy by construction: an arbitrary graph
  collapses to a per-dimension bandwidth/latency vector.
- The Level 1 fast model (§5.3) is the *only* model expressible through the stock config. Level 2
  path-aware evaluation needs the `config_builder.py` work that §7 unlocks at Phase 5 — it is not
  an optimization, it is a prerequisite.
- `energy_per_bit_pj` has no simulator consumer today. Link energy comes from the `power` block's
  `link.energy_per_bit`, which is **per node, not per link**.

**Adaptation for Phase 2**: keep the full graph in `ClusterSpecV2` — it is still the planner's own
reasoning substrate for pruning and for the topology lower-bound filter. At compile time, reduce it
to `link_bw`/`link_latency` by taking the bottleneck along the relevant path, and record the
reduction in provenance so results are never mistaken for path-aware ones. Do not silently average.

**Phase 5 update — Level-2 shipped as opt-in `--topology-level 2` (2026-08-21).** The stock ASTRA-Sim
config accepts a *per-dimension* `link_bw`/`link_latency` list (`config_builder.py::
_normalize_network_dim_values`), where dim 0 is the intra-group (TP) FullyConnected dimension and
dim 1 is the cross-instance dimension. The Level-2 compile emits `[intra_bottleneck,
cross_bottleneck]` instead of the single global scalar, so a fast intra-island interconnect is no
longer dragged down by a slow cross-instance fabric (or vice-versa — the `heterogeneous-lab` example
has PCIe islands on a faster InfiniBand fabric, so Level 2 raises the *cross* dim from 64 → 400
GB/s). Key points:

- **No `config_builder.py` edit was needed.** The stock config already consumes lists; the planner
  compiler (`planner/predictor/llmservingsim.py`) sizes the list by reusing the pinned
  `serving.core.config_builder._compute_network_dims`, so the "adapter §7 unlocks in
  config_builder" is satisfied *functionally, planner-side* — no upstream `serving/` change, no D12
  exposure. The work order's literal file location is deliberately not followed.
- **Default stays Level 1**, so this is the D3 "compare the two" mechanism, not a silent flip.
  Level 2 changes predictions *only for multi-island placements*; single-island and same-island
  P/D compile byte-identically (the intra value serves the one dimension).
- **Still not per-flow path-aware.** ASTRA-Sim's dimensional model cannot represent
  `contention_group` sharing, so it is dropped exactly as in Level 1 — the cross dim is single-flow.
  Provenance records `model_level: 2, resolution: per-dimension, path_aware: false,
  contention_modeled: false` so a Level-2 result is never mistaken for a per-flow one.
- **Oracle-agreement is untouched**: the reduction feeds only the sim config and the top-level
  provenance summary, never a pruning stage (stage-4 uses `island_interconnect` directly), and the
  mock predictor ignores it. The envelope cache key folds in `topology_level` so Level-1 and
  Level-2 results cannot collide (Level-1 keys unchanged). Verified by `tests/test_topology_perdim.py`.

---

## D4 — Only one hardware profile exists · **Resolved 2026-09-02 (Tier 0 synthetic-bundle path)**

**Work order §2.1 / §3.3 / Phase 3** assume `h100.yaml`, `rtxpro6000.yaml`, `ascend_target.yaml`,
and a Phase 3 exit criterion of "2–3 accelerator classes appearing as candidates".

**Upstream**: `profiler/perf/` contains **`RTXPRO6000` only**, for three models
(`meta-llama/Llama-3.1-8B`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `Qwen/Qwen3-32B`).
`configs/cluster/single_node_single_instance_H100.json` ships but has no `profiler/perf/H100/`
behind it, and upstream validation rejects a `hardware` value with no profile directory.

Compounding it: this machine has 2 × RTX A5000 and no NPU, so we cannot profile an H100 or an
Ascend part ourselves.

**Adaptation**: Phase 3's `CsvProfileImporter` (work order V1 path) stops being a convenience and
becomes the critical path — every non-RTXPRO6000 accelerator must arrive as imported CSV under the
`profiler/CONTRACT.md` schema. A5000 is profilable locally and is the honest second class if a
real second class is needed before external data lands.

**Needs a decision at Phase 3**: where H100 / Ascend performance data comes from — published
benchmarks, a collaborator's measurements, or vendor specs. Whatever the source, it is
`source: placeholder` or `source: measured` with attribution, never silently synthesized
(absolute rule 3, §11 "실물 hardware 부족").

**Update 2026-08-14 — the decision resolved itself with real hardware.** The user confirmed
access from the week of 2026-08-17 to A40x8 GPU nodes (up to 8), 4x Rebellions ATOM and
4x FuriosaAI RNGD. The concrete NPU targets are therefore ATOM/RNGD rather than Ascend (the
work order's `backend` enum is explicitly extensible; `rbln` and `furiosa` identifiers added),
and the second-large-GPU class is A40 rather than imported H100 data. Profiles become
*measured* instead of imported; the CsvProfileImporter remains the V1 entry path for NPU
latency data. Plan and bring-up order: `docs/hardware_roadmap.md`. Stubs with all-placeholder
fields and empty `supported_models` (= excluded until verified):
`profiles/accelerators/{a40,rbln_atom,furiosa_rngd}.yaml`.

---

## D5 — No `--seed` flag · Resolved

**Work order §9** requires "same spec + seed run twice ⇒ byte-identical plan output", and §5.5 says
"random_seed 필수".

**Upstream**: `python -m serving` has no seed flag. It does not need one — given a fixed input trace
and a deterministic routing policy it is a deterministic discrete-event simulation. Verified: two
identical invocations produced byte-identical CSVs.

**Adaptation**: the seed belongs to *our* workload generator (`planner/util/workload.py`), which
turns the `ServiceSpec.traffic` distribution into the JSONL trace. Seed there, record it in
provenance (§3.8), and avoid `--request-routing-policy RAND` (and `RAND` expert routing) in any
reproducible run. Reproducibility is then a property of trace generation, not of the simulator.

---

## D6 — Committed example outputs are stale · Resolved

**Not a work order assumption** — a trap worth recording, since §9 asks for golden-output tests and
`outputs/example_*_run.csv` looks like a ready-made reference set.

**Upstream**: those CSVs predate the current `main`. Verified across all 10 rows of the power
example against `workloads/example_trace.jsonl`: the committed files record `output = input_toks +
output_toks` (total length) while current `main` records `output = output_toks` (decode only).
Latencies moved as well — request 3 TPOT 25.54 ms → 11.21 ms — consistent with post-artifact
accounting fixes in `serving/`. Current behavior matches
`docs/docs/simulator/reading-output.md`; the committed CSVs do not.

**Adaptation**: generate all golden fixtures ourselves at the pinned commit. Never diff against
`outputs/example_*`. If upstream is ever re-pinned, regenerate the goldens as part of that change.

---

## D7 — Accelerator profile schema does not map onto the simulator's power block · Resolved

**Work order §3.3** defines a profile with `memory_bandwidth_gbps`, `tdp_w`, `idle_power_w`.

**Upstream** needs, per node and keyed by hardware string: `idle_power`, `standby_power`,
`active_power`, `standby_duration` (ns), plus node-level `base_node_power`, `cpu`, `dram`, `link`,
`nic`, `storage` sub-blocks.

Two fields have no source in the work order schema: **`standby_power`** and **`standby_duration`**,
and `tdp_w` is not the same quantity as `active_power`.

**Adaptation**: extend the profile schema with an explicit `power:` block mirroring the simulator's
field names one-to-one, rather than deriving them at compile time. Anything not measured is
`source: placeholder` and says so. Node-level components (`cpu`, `dram`, `nic`, `storage`,
`base_node_power`) are properties of the *node*, not the accelerator, so they belong in
`ClusterSpecV2` under `nodes[i]`, not in `profiles/accelerators/*.yaml`.

---

## D8 — DP replicas are separate instances; `dp_group` means something else · Resolved

**Work order §5.4** lists `dp_replicas` as an enumerated decision variable, and §3.4 gives each
`DeploymentPlan.instances[i]` a `tp_size`/`pp_size` but no replica count.

**Upstream**: data-parallel replication is expressed by listing N sibling entries in
`nodes[i].instances`, with cross-instance load spread handled by `--request-routing-policy`. The
`dp_group` field is **not** data parallelism — it is a MoE expert-sharing group whose members
exchange tokens via cross-instance ALLTOALL and must agree on `ep_size` and `tp_size`.

**Adaptation**: compile `dp_replicas = N` into N instance objects. Leave `dp_group` `null` for dense
models; set it only for MoE expert parallelism. Naming the planner's variable `dp_replicas` while
upstream's `dp_group` means something unrelated is a live footgun — comment it at the compile site.

---

## D9 — "heterogeneous" upstream means P/D, not mixed hardware · Resolved

`configs/cluster/single_node_heterogeneous.json` is a prefill/decode split with **both instances on
`RTXPRO6000`**. It is not an example of hardware heterogeneity and must not be cited as a starting
point for `examples/clusters/heterogeneous-lab.yaml`.

---

## D10 — The simulator's memory model has no utilization or overhead reserve · **Open (affects Phase 2 §5.4 stage 2)**

**Work order §5.4** makes memory feasibility the second pruning stage and requires calling
`serving/core/memory_model.py` rather than reimplementing it, so whatever that model computes
becomes the planner's definition of "fits".

**Upstream**: `MemoryModel` computes `mem_for_kv = npu_mem - weight` (`memory_model.py:60`), where
`npu_mem` is the config's `npu_mem.mem_size` in GB. There is **no `gpu_memory_utilization` factor
and no reserve for activations or CUDA graphs** — grep for a utilization factor in `serving/core/`
returns nothing.

Real vLLM reserves `gpu_memory_utilization` (default 0.9) of total VRAM and then subtracts weights,
activation peak, and CUDA-graph capture.

**Measured on this machine** (RTX A5000 24 GB, Llama-3.1-8B bf16, TP=1, `enforce_eager=True`,
`max_model_len=2048`) — vLLM reports `Available KV cache memory: 5.29 GiB` / `GPU KV cache size:
43,296 tokens`. Independently derived: KV/token = 2 × 32 layers × 8 kv-heads × 128 head-dim × 2 B
= 128 KiB, and 43,296 × 128 KiB = 5.29 GiB. The two agree exactly.

| | KV budget | KV tokens |
| --- | ---: | ---: |
| Simulator, `mem_size: 24` (24 GiB − 14.96 GiB weights) | 9.04 GiB | 74,075 |
| Planner with `gpu_memory_utilization=0.9` applied | 6.64 GiB | 54,415 |
| Real vLLM on the same card | 5.29 GiB | 43,296 |
| | | **raw sim over-estimates by +71%** |

Applying the 0.9 utilization factor alone leaves +26%. Closing the rest needs an
explicit activation reserve; measured at **1.35 GiB** for this model/device under
`enforce_eager=True`. That figure will grow once CUDA graphs are captured, so it
is an input, not a constant.

**Why this was invisible until now**: upstream's validation ran on a 96 GB RTX PRO 6000 where the
workload never approached the KV limit, so the missing reserve changed nothing. On a 24 GB card it
becomes first-order — the sharegpt-300 workload at `max_num_seqs=128` wants ~177,664 concurrent KV
tokens (21.7 GiB), far above *both* budgets, so KV pressure is the binding constraint and a 71%
error in it drives preemption and queueing behavior directly.

**Consequence for the planner**: the memory feasibility filter will over-estimate KV capacity on
every real device by roughly the utilization factor, admitting candidates that cannot actually be
served. Worse, the error scales *inversely* with card size — negligible on 96 GB, severe on 24 GB —
so a planner validated on large cards silently degrades on small ones. Any Phase 4 deployment built
on this filter would OOM or thrash.

**Adaptation**: keep calling upstream's model (§5.4's "call, don't copy" rule stands), but apply an
explicit, recorded derating in the planner before the call — `effective_mem_size = mem_size *
gpu_memory_utilization - activation_reserve` — with both terms named in `ClusterSpecV2` rather than
hard-coded, and defaulting to vLLM's own defaults. Do not silently bake 0.9 in.

**Empirically confirmed 2026-08-07.** Both configurations were run against the same real A5000
measurement (`phase0_bench_plan.md` §2b). Correcting only the memory accounting moved mean
\|error\| from **22.54% to 9.26%** (−13.28pp, 13 of 15 metrics improved); TTFT Mean went −32.0% →
−7.7%. On a memory-constrained device this is the largest single error term, larger than profile
quality and far larger than grid density. The derating in `planner/util/memory.py` is therefore
load-bearing, not defensive.

Measured under the real bench engine settings (`max_model_len=8192`, CUDA graphs on) vLLM reports
`Available KV cache memory: 5.85 GiB` / `47,920 tokens`, against the simulator's 9.04 GiB / 74,075
— **+55%**. The earlier +71% figure was measured with `enforce_eager=True`; the over-estimate
varies with engine settings, which is another reason to treat the reserve as an input.

**Open question for the A5000 validation**: whether to run the comparison at nominal `mem_size: 24`
or at a KV-matched `mem_size` (~21.3 GB = 16 GiB weights + 5.29 GiB observed KV). Running **both**
separates memory-accounting error from profile error, which is the entire point of the exercise;
that is the plan unless it proves too costly. Re-measure the KV figure under the actual bench
engine settings first — the 5.29 GiB above was captured with `enforce_eager=True`, and CUDA-graph
capture in the real bench run will reduce it.

---

## D11 — `meta.yaml` under-describes the profile grid, and grid density costs ~2.2pp of accuracy · Resolved (quantified)

**Not a work order assumption** — discovered while trying to make the A5000 bundle comparable to
the shipped RTXPRO6000 one.

**Finding 1 — `meta.yaml` is not a faithful description of the bundle.** It records
`attention_grid: {chunk_factor: 2.0, kv_factor: 2.0, chunks: "0, 16-2048 x2"}`, which yields 9
`prefill_chunk` values. The actual `attention.csv` contains **20**:

```
meta.yaml (x2)  : 0, 16, 32, 64, 128, 256, 512, 1024, 2048
actual CSV      : 0, 16, 24, 32, 36, 54, 64, 81, 122, 128, 182, 256, 273, 410, 512, 615, 923, 1024, 1384, 2048
```

The extra points are a ×1.5 sequence (16·1.5ⁿ, and 512·1.5ⁿ on the kv axes). The bundle is the
**union of at least two runs at different factors**, accumulated through the profiler's default
resume mode, while `meta.yaml` is overwritten with only the most recent run's settings.
`skew.csv` shows no such densification — its `pc` axis has the 9 values `meta.yaml` claims.

Consequence: **never infer a bundle's grid from `meta.yaml`.** Read the CSV keys. Phase 3's
`CsvProfileImporter` and `profiler/CONTRACT.md` must record measured coverage, not declared factors.

**Finding 2 — density is worth ~2.2pp of end-to-end accuracy.** Controlled experiment: the
RTXPRO6000 `attention.csv` was subset to exactly the 8,643 keys present in the locally measured
A5000 grid (`profiler/perf/RTXPRO6000X2/`, a derived artifact — every other file copied verbatim),
then the same simulation and `bench validate` were re-run against the same real-vLLM data. Only
attention-grid density changed.

| Metric | full grid (19,364) | ×2 subset (8,643) | penalty |
| --- | ---: | ---: | ---: |
| TTFT P99 | +1.0% | +4.4% | +3.40pp |
| TPOT P99 | +2.1% | +3.8% | +1.70pp |
| Latency P99 | +0.8% | +2.8% | +2.00pp |
| **mean over 15 metrics** | | | **+2.21pp** |

Range +1.50 to +3.40pp, and **every one of the 15 is positive** — coarser interpolation makes the
simulator uniformly more pessimistic, never optimistic.

**Consequences**

- The "systematic positive bias" recorded in `phase0_bench_plan.md` is partly an artifact of
  interpolation, not purely a simulator property. On the full grid the tail bias is ~+1–2%; the
  reference bundle happens to be densely sampled.
- **The A5000 comparison must subtract this penalty before attributing anything to hardware or
  profile quality.** The A5000 bundle is at ×2 density, so its expected error floor is roughly the
  ×2-subset column above, not the full-grid column. That is exactly why the control was run.
- For the planner: profile density is a first-class accuracy/cost knob. A denser bundle costs GPU
  hours once and buys ~2pp of prediction accuracy permanently — relevant to §5.8's robust margin,
  since a 2pp tighter margin admits materially more candidates.

**Why the A5000 bundle was not densified instead**: matching the reference grid would have required
39,831 additional shots (~4.4 h for TP=1 attention alone, 12 h+ with TP=2 and skew) against ~10 min
of CPU simulation for the control. The control also isolates density better, since it holds
hardware and real-measurement data fixed.

---

## D12 — Prefix-cache memory grows monotonically until the run dies · **Open (blocks Phase 2)**

**Not a work order assumption** — found while running the A5000 sim-vs-real validation, and the
most consequential finding so far.

**Symptom**: simulating the bundled sharegpt-300 workload on a 24 GB device
(`hardware: A5000`, `mem_size: 24`, `max_num_seqs: 128`, prefix caching at its default of ON) dies
partway through:

```
[13.0s] ... Each NPU Memory Usage 24534.51 MB (99.831 % Used)
Traceback (most recent call last):
  serving/core/scheduler.py:800    in add_done            -> self.memory.cache_unfinished_req(...)
  serving/core/memory_model.py:454 in cache_unfinished_req -> self.apply_kv_cache_events()
  serving/core/memory_model.py:608 in apply_kv_cache_events -> self.allocate(npu_byte_alloc, Device.NPU)
  serving/core/memory_model.py:252 in allocate
RuntimeError: NPU: tried to load 2.00MB but only 1.49MB is available.
```

Both the nominal (`mem_size: 24`) and KV-matched (`mem_size: 20.81`) configurations fail; the
tighter one fails sooner.

**This is not a missing-preemption problem.** `scheduler.py:194` implements exactly that — it
preempts decode requests one at a time and spills their KV to CPU until the batch fits. The failing
call is on a different path: **prefix-cache bookkeeping** (`cache_unfinished_req` →
`apply_kv_cache_events` → `allocate`), which allocates unconditionally and has no eviction
fallback. Confirmed by construction: the identical run with `--no-enable-prefix-caching` completes
cleanly (exit 0).

**Why upstream never saw it**: their validation runs on a 96 GB RTX PRO 6000, where this workload
peaks far below capacity. It needs a device small enough for NPU memory to actually reach ~100%.
Same root cause as D10 — the accounting only matters once memory is the binding constraint.

**Why it blocks Phase 2.** HeteroPilot exists to plan on memory-constrained heterogeneous hardware.
The candidates worth evaluating are precisely the tight ones, and the predictor crashing on them is
not a survivable failure mode:

- A crash is indistinguishable from a genuinely infeasible configuration unless the predictor
  inspects the traceback. Silently treating it as "infeasible" would discard candidates that real
  vLLM serves fine — vLLM handles this workload on the same card without incident.
- It is *selective*: it removes exactly the memory-tight candidates, which biases the planner
  toward over-provisioning in a way that looks like a legitimate optimizer decision.
- Prefix caching is on by default and `prefix_share_ratio` is a `ServiceSpec` field (§3.1), so
  "just turn it off" narrows the model the planner is allowed to reason about.

**Adaptation (Phase 2)**

1. The predictor must classify a non-zero simulator exit as a distinct outcome —
   `SIM_ERROR`, never silently folded into `slo_violated` or `memory_infeasible` — and surface the
   count in `rejected_summary`. A candidate set with many `SIM_ERROR` entries is a broken run, not
   a planning result.
2. Match the sim's prefix-caching setting to whatever the deployment will use, and record it in
   provenance. Comparing a prefix-cache-off simulation against a prefix-cache-on deployment is not
   a valid comparison.
3. Longer term this wants an upstream fix: `apply_kv_cache_events` should evict from the prefix
   cache rather than raise, mirroring what `evict_prefix_cache` already does elsewhere.
   `serving/core/memory_model.py` is not editable until Phase 4 under §7, so Phase 2 lives with
   the workaround.

### Two fix attempts, both wrong — 2026-08-10

The user authorized pulling an upstream fix forward (a documented exception to absolute rule 1).
**Both attempts failed and were reverted; `serving/` is pristine again.** Recorded here so nobody
repeats them.

**Attempt 1 — make the prefix-cache store best-effort** (`memory_model.py`). Reclaim evictable
blocks before allocating, and if the store still does not fit, skip it and roll back the
`_npu_cache_hashtolen` entries instead of raising. Passed the byte-identical RTXPRO6000 regression.

*Outcome: strictly worse.* The crash became a **silent deadlock** — NPU memory pinned at 99.994%,
0 requests running, 300 waiting, throughput zero, forever. A loud failure turned into one that
burns wall-clock and looks like a slow candidate to a planner. The reclaim found nothing to evict,
which is the real clue: essentially the whole cache was un-evictable.

**Attempt 2 — release prefix locks on preemption** (`scheduler.py`). The eviction site at line ~204
frees the request's KV and spills to CPU but never calls `unlock_prefix`; the *other* eviction site
(~494) does, but only `if is_prefill()`. Hypothesis: preempted decode requests keep their radix
nodes pinned forever, so the cache can never be evicted. Also passed the regression.

*Outcome: no change.* Still deadlocked at 99.994% with 0 running / 300 waiting. The hypothesis was
wrong, or at least incomplete.

### What the evidence actually says

- Memory climbs **monotonically** — 66% at t=1s, 80% at t=6s, 99.8% at t=13s, then pinned at
  99.994% — and never comes back down. This trajectory is identical with and without either fix;
  the crash was only ever the symptom of hitting the ceiling.
- At the deadlock, `Total CPU Memory Usage 0.00 MB`. Preemption never spilled anything, so the
  ~9.2 GB of non-weight NPU memory is prefix cache, not request KV.
- Prefix hit ratio stays low (3.02%), so the cache is retaining blocks that are not being reused.

The likely root cause is the accounting relationship flagged above: the scheduler allocates a
request's KV (`scheduler.py:248, 841`) *and* `apply_kv_cache_events` allocates again for the same
blocks when they are stored in the radix cache, with the request's own allocation only freed at
completion (`scheduler.py:790`). On a 96 GB card the resulting slack is invisible; on 24 GB it
compounds until the pool is gone. Confirming that requires understanding upstream's intended KV
lifecycle rather than guessing at it — two guesses have already been wrong.

### Recommended next step

Treat this as an upstream bug report, not a local patch: a minimal reproducer exists
(`experiments/configs/clusters/a5000-llama31-8b-tp1.json` + `sharegpt-llama-3.1-8b-300-sps10.jsonl`
at `max_num_seqs=128`), it needs no GPU, and it fails on any device small enough to saturate.

Until it is resolved, Phase 2 must either run memory-tight candidates with prefix caching disabled
(losing `ServiceSpec.prefix_share_ratio` coverage, §3.1) or restrict itself to devices with enough
headroom that the growth never reaches the ceiling — which excludes exactly the hardware
HeteroPilot is meant to plan for. **The predictor must in any case treat a non-zero simulator exit
and a wall-clock timeout as distinct `SIM_ERROR` outcomes**, never silently folded into
`memory_infeasible`, and must impose a timeout — attempt 1 showed the failure can present as a hang
rather than a crash.

---

## D13 — The §3.6 envelope key omits `dp_replicas`, and results collide · Resolved

**Work order §3.6** specifies the PerformanceEnvelope key as:

```
PerformanceEnvelope[model, dtype, accelerator, tp, pp, ep, pd_role,
                    scheduler_config_hash, network_class, workload_bucket]
```

Implemented literally. It is **missing the data-parallel replica count**, and our predictor
simulates the *whole deployment* — every replica — so a 2-replica run has roughly double the
throughput and a fraction of the queueing delay of a 1-replica one. Two candidates that differ only
in `dp_replicas` hashed to the same entry.

**Observed damage.** In a 30-candidate, 300-request run (`outputs/plans/llama31-8b-plan-300.yaml`,
2026-08-10) the generator emits dp=1 before dp=2 for each parallelism degree, so:

- 18 candidates were simulated and cached;
- **all 12 dp=2 candidates were then served the corresponding dp=1 entry** and never simulated;
- they were ranked, and in some cases rejected as `slo_violated`, on single-replica metrics.

The run's own provenance recorded `envelope_cache_hits: 12` on a cold cache — a cold cache should
produce zero hits, which is the signal that was there to be read.

**Why it is silent.** A wrong hit does not error. It returns plausible, well-formed metrics for a
configuration that was never run. Nothing downstream can tell the difference.

**Fix**: `dp` added to `EnvelopeKey`, with `tests/test_envelope.py` asserting distinct digests for
every field that changes the outcome (`dp`, `tp`, `pp`, `max_num_seqs`,
`max_num_batched_tokens`, `role`) plus a direct regression for the collision. The poisoned cache
was discarded and the run repeated.

**Where §3.6 would have been right**: if an envelope described *per-replica* performance that the
planner then composed arithmetically. That is a defensible design — it makes entries reusable across
replica counts — but it is not what this predictor produces, and the key has to match what is
actually stored. If the surrogate predictor of Phase 2's later stages ever moves to per-replica
envelopes, revisit this.

**General lesson for the cache**: any field that changes the simulated result must be in the key.
A cold-cache run reporting a non-zero hit count is a bug, not a nicety — worth an assertion.

---

## D14 — The simulator's topology inference requires uniform instance sizes · Resolved (constraint enumerated around)

> **PARTLY LIFTED 2026-09-08 (D28).** The uniformity is not the simulator's; it
> was `_compute_network_dims` plus a shared `local_dim`. `topology_mode: slab3d`
> now expresses `tp_d = 2 * tp_p` with no idle rank. Other ratios still need
> idle-rank padding and remain enumerated around. STEP 2.3 carries the planner side.


### Measured 2026-09-04 (`WORK_ORDER_spikes.md` STEP B) — the constraint is liftable, and worse than recorded

`docs/d14_spike.md`. Two corrections to what follows, both measured on
`A40 tp4 prefill + RNGD tp8 decode` — the configuration D16(c) names as
industry-recommended and unavailable:

**1. It is worse than "wrong numbers, not an error".** That configuration compiles
today to `npus_count: [5, 3]` — **15 ranks for instances that occupy 16**
(`total_npu` 16, `total_pp` 3, `16 // 3 = 5`). Rank 15 has nowhere to live, so the
collective never completes: of the four instance-boundary ranks only NPU[7] reports
`iteration 0`, the ASTRA-Sim child dies, and the frontend spins on EOF (D23). The
run emits **no progress line in 300 s**. It does not produce wrong numbers; it
produces none and hangs the harness.

**2. The constraint is two sites in `config_builder.py`, and both are liftable.**
Not only `_compute_network_dims`'s integer division, but also `_resolve_dp_groups`,
which hands *every* instance the same `local_dim`. With an opt-in
`topology_mode: "slab3d"` bypassing both, the same fixture compiles to
`npus_count: [4, 2, 2]`, all four boundary ranks report, and the run **completes**
(86 s, 21 rows). TTFT lands 2.21× the A40 standalone and TPOT 0.73× the RNGD
standalone — the right side of each, for the right reason.

The prototype is **spike-only** (`spike/d14-asym-tp`, not merged); `serving/` on
`main` is unchanged and the `auto` path is byte-identical (`sha256`-equal CSVs,
460 tests, golden included). Two costs were measured and are the real gate:

- splitting one TP group across two dims changes TPOT by **24.4 %** — but a
  per-dim `link_latency` of **4× the scalar**, which `_normalize_network_dim_values`
  already accepts, closes it to **0.008 %**. The 4× is a fitted constant at one
  bandwidth and one split; it needs a calibration domain before any absolute number
  is quoted from a slab3d plan.
- a 3-D collective tag `ALLREDUCE:1,1,0` is **exactly** the 15 characters of
  `utils.py::_FMT`'s `comm_type` column, so it abuts `comm_size`, and
  `generate_trace` re-reads its own fixed-width file with `re.findall(r'\S+')`
  (`trace_generator.py:1514`) and gets 10 fields instead of 11. An upstream bug,
  unreachable at ≤ 2 dims. Not fixed here (§7); recorded, with a runtime
  workaround in `experiments/scripts/b3_widecol_run.py`.

*Original entry follows, unchanged.*


**Work order §1.3 / Exp 2** exploit heterogeneity at replica granularity: replicas of the same
model on different islands, requests spread by the router. Nothing in the work order restricts the
per-island parallelism of those replicas.

**Upstream**: `serving/core/config_builder.py::_compute_network_dims` infers the ASTRA-Sim topology
for independent instances as `[npus_per_group, num_instances]` with
`npus_per_group = total_npu // num_instances` (or `// total_pp`) — **integer division over the
device total, which assumes every instance is the same size.** A tp=1 instance mixed with a tp=2
instance yields `3 // 2 = 1`, silently mis-scoping the tp=2 instance's collectives into a
1-wide dimension. There is no validation catching this; the run would produce wrong numbers, not
an error.

**Adaptation**: mixed candidates are enumerated only with **equal devices-per-replica across
islands** (`tp_a * pp_a == tp_b * pp_b`; pairs only for the MVP). Unequal mixes are structurally
unrepresentable, so — like cross-backend TP — they are excluded without per-candidate rejection
records; the constraint is asserted by `tests/test_mixed.py`. Locally this is no loss (the A5000
island is tp=1-only anyway); on the incoming A40 nodes uniform-tp mixes still cover Exp 2's
design space.

**Related Level-1 note**: for tp=1 mixed replicas there are *no* collectives at all, so the scalar
`link_bw` bottleneck reduction (D3) is irrelevant to them. For future uniform-tp>1 mixes the scalar
min() is pessimistic for intra-island collectives; the honest encoding is a per-dimension
`link_bw` array (the config supports it), but the array rank depends on `_compute_network_dims`
internals, so that is deferred to Phase 5 when `config_builder.py` opens for modification.

Two honesty guards shipped with the mixed feature. The envelope cache key now describes the
**entire placement** (every assignment, sorted — a mixed candidate can never hit a single-island
entry, extending the D13 fix). And a deployment whose nodes are only partially covered by
`power:` blocks reports **no** energy. On the second point upstream turns out to already be safe —
`config_builder.py:326` disables power modeling wholesale when any node lacks a power spec
(verified in a real mixed run: no power output at all) — so the predictor's `power_complete`
guard is defense-in-depth against that upstream behavior ever changing, not a live bug fix.
Practical consequence: mixed candidates have energy metrics only once *every* node they touch has
a measured power block, which makes the A5000 power measurement a prerequisite for energy-ranked
Exp 2.

## D15 — The simulator charges the P/D KV handoff as free; we make it bandwidth-sensitive (first sanctioned `serving/` edit) · Resolved (opt-in, default byte-identical)

**Context.** D12's earlier `serving/` edits were reverted, so `serving/` had stayed pristine. This is
the **first authorized upstream edit** (Phase 5, work order §7). Increment 2 (`docs/phase5_plan.md`)
root-caused that the simulator prices the prefill→decode KV transfer at **zero**: `__main__.py:597`
→ `router.transfer_prefill_request()` → `scheduler.add_decode()` allocates the decode KV with no
delay and emits no collective, so a cross-instance bandwidth sweep left simulator output
byte-identical. HeteroPilot had worked around it with a *planner-side* add-on
(`apply_pd_transfer_cost`); D15 adds the **simulator-side** model so the sim's own P/D numbers move
with bandwidth.

**Work order §7 constraint — scheduler.py is off-limits.** §7 marks `serving/core/scheduler.py`
"수정하지 않고 그대로 사용" (use as-is, do not modify), while `request.py` and `router.py` are the
files sanctioned for Phase 5+ P/D extension. The architect's first design edited the scheduler's
batch filter and `add_decode`; that was **reverted** to keep scheduler.py pristine. The shipped
design confines the change to the two orchestration files:

- `serving/core/router.py`: a **deferred-transfer queue**. In bandwidth mode
  `transfer_prefill_request` does *not* hand the request to the decode scheduler at prefill
  completion; it enqueues `(ready_time, req_id, req, decode_index)` with
  `ready_time = current + link_latency + KV_bytes / link_bw`, and exposes `pop_ready_transfers`,
  `has_pending_transfers`, `get_next_transfer_ready_time`.
- `serving/__main__.py`: the main loop **drains** ready transfers (calling `add_decode` only once
  the KV has "arrived", so the decode-side KV allocation is deferred too — physically it lands on
  arrival, not send); the idle **clock-advance** jumps to the next `ready_time` (the P/D counterpart
  of the agentic `get_next_pending_arrival`); and the **done-detection** condition gains
  `not router.has_pending_transfers()` so a run cannot exit while a request's KV is still in flight
  (that request is not in any scheduler yet and would otherwise be silently lost).

`serving/core/scheduler.py` and `serving/core/request.py` are **untouched**.

**Metric bucket (decided).** This simulator emits the first token on the *prefill* instance
(`request.py::set_ttft`), so a delayed decode start lands in end-to-end **latency, ITL[0] and
(smeared) TPOT — never TTFT**. We adopt this sim-honest bucket deliberately. It **differs from the
planner-side add-on** (increment 2), which charges the transfer to **TTFT** (a DistServe/Mooncake
client-TTFT convention). Consequence: the sim-level §5.9 adoption crossing is latency/TPOT-driven,
the planner-level one is TTFT-driven; they will not coincide. This is accepted and recorded rather
than reconciled, because aligning them would require moving where `set_ttft` fires (a large,
metric-corrupting change).

**Double-counting.** `LLMServingSimPredictor` does **not** pass `--pd-transfer-model`, so the
planner always simulates in `none` mode and its add-on remains the sole transfer price — no double
count today. If a future change makes the predictor use `bandwidth` mode, `apply_pd_transfer_cost`
**must** be disabled for that run.

**Back-compat.** New CLI flag `--pd-transfer-model {none,bandwidth}`, default `none`. In `none`
mode the queue is always empty, so the drain, clock-advance and done-guard are no-ops. Verified
byte-identical against the pinned baseline for a non-P/D config (bench-class) and for the P/D config
with no flag. `UPSTREAM_COMMIT` is unchanged (this is our own fork edit, tracked in git — not a
rebase onto newer upstream).

**Modeling limits (honesty).** `transfer_ns` is a hand-computed `KV_bytes / link_bw` over one
cross-instance link (the trailing network dim for a list `link_bw`); `KV_bytes` comes from the
decode scheduler's own `memory.get_total_kv`, which is **per-NPU** (the memory model divides by
`num_npus`), i.e. it models the TP shards moving in parallel over per-rank links. It does **not**
model contention between the KV transfer and concurrent TP collectives on a shared link — that
would need a real send/recv node in the Chakra/ASTRA graph (approach "B", rejected for now: the
analytical backend does not cost a P2P hop, so it would drag in the unbuilt ns3 backend). At
realistic P/D bandwidths (≥25 GB/s) and short prompts the transfer is sub-millisecond, consistent
with the increment-2 finding; the effect grows as `1/bw` and is clearly visible below ~4 GB/s.

**Verification.** `experiments/scripts/pd_sim_network_sweep.py` (latency & TPOT monotonic ↑ as bw
drops, TTFT flat, `none`-mode control flat — all PASS) and `tests/test_sim_pd_transfer.py`
(router-side queue arithmetic and clock behavior).

---

## Open items summary

| ID | Blocks | Status |
| --- | --- | --- |
| D2 | Phase 2 | **Decided** — parse stdout in Phase 2, switch to a `power_model.py` API at Phase 4 |
| D3 | Phase 5 | **Decided** — Level-1 compile for Phase 2; revisit after adding the link graph and compare the two |
| D4 | Phase 3 | **Resolved 2026-09-02** (Tier 0 synthetic-bundle path) — A5000/A40/RNGD measured locally; Ascend closed without external measurements via the datasheet-derived `ASCEND_TARGET-t0` bundle from `profiler.synth`, so Ascend islands survive candidate generation with `profile_tier: analytical`. Measured data still supersedes it whenever it arrives (D21) |
| D10 | Phase 2 | **Open** — derating factor for the memory feasibility filter; nominal vs KV-matched config for the A5000 comparison |
| D15 | Phase 5 | **Resolved** — sim-level P/D KV-transfer model, opt-in `--pd-transfer-model bandwidth`, default byte-identical; first sanctioned `serving/` edit (router.py + __main__.py only, scheduler.py pristine) |
| D18 | Phase 5 | **Resolved** (retraction) — NPU-leg multi-stream bandwidth remeasured; scaling law held, levels ~25 % lower. Fixtures recomputed once the GPU leg landed (all six links now `measured`); the SLO sweeps cannot see the change, for three recorded reasons |
| D19 | Phase 4 | **Resolved** (retraction) — the card profile's −71 % TTFT error was an arrival-pattern mismatch in the validation harness, not a scheduler difference; matched arrivals give −5.1 %. Both RNGD TTFT calibrations refitted |
| D20 | Phase 3 / Exp 4 | **Open** — ATOM layerwise profiling blocked: host I/O exceeds the kernels and the device tracer's schema is undocumented. Memory and power measured; no perf bundle, so ATOM stays out of candidate generation |
| D21 | Phase 3 | **Decided 2026-09-02** — Tier 0/1 synthetic profiles: `datasheet:` fields are vendor spec, never measurements. Generated bundles carry `tier: analytical`/`calibrated` and a `-t0`/`-t1` hardware-label suffix so they can never shadow a measured bundle; `PlannerOutput.profile_tier` propagates the weakest tier with a mandatory caveat. `flops_efficiency`/`mem_efficiency` stay empty until fitted against a measured bundle |
| D22 | Phase 4 | **Resolved** (retraction + measurement) — the c1–c32 curve's top point was a 24-request pool running at eff 21.2, not c32; envelope measured to eff 107.2. At eff 76 the simulator is 1.31× optimistic on throughput and 18 % on TPOT. The re-run is **done for the loose-TTFT regime**: with the measured 18 % margin every RNGD config is rejected on both fixtures and the winner becomes `agg[cuda:tp4]` at 2.595 tok/J — the committed winner is infeasible, not merely optimistic. Tight-TTFT rows still open — the 1800 s re-run kept them undetermined, and D23 explains why: those candidates livelock |
| D23 | Phase 4 / Phase 5 | **Diagnosed 2026-09-04, root cause open upstream** — the candidates do **not** livelock: one completes alone in 343 s at N=300. ASTRA-Sim races on a fixed cwd-relative `tmp__mem/*.json` (13 of 64 bare processes fail), and the frontend spins forever on the dead child (`controller.py` `read_wait` on EOF) with its stderr captured and never read. Unfixed at both upstream heads; `experiments/scripts/astra_isolated.sh` works around it, 64/64. `docs/d23_spike.md` |
| D24 | — | **Resolved** — the work order's layout lists `profiles/networks/`, but Level-1 interconnect-class values live inline in `planner/topology.py` and the YAMLs were an unread duplicate read only by ScenarioLab's cluster generator. Moved out with it (STEP 3.3); recoverable if Phase 5 ever wants them as data |

---

## D16 — `LinkType` has no on-package fabric, and cross-vendor P/D needs a shared TP degree · Resolved (one added type) + Open (the TP constraint)

> **(b) PARTLY LIFTED 2026-09-08 (D28).** "The simulator requires a shared TP
> degree" is false for the 2x case: `slab3d` encodes `tp_d = 2 * tp_p` directly.
> The size-4 island bridging workaround in the fixture still works and is kept,
> but is no longer required. STEP 2.3 carries the planner side.


Two problems surfaced together while building the first heterogeneous
RNGD + GPU P/D fixture (`experiments/configs/clusters/pd-rngd-gpu.yaml`).

### (a) No link type fits an on-package PE fabric — resolved

**Upstream / our schema** offered `NVLINK`, `PCIE`, `INFINIBAND`, `ETHERNET`,
`HCCS`. A FuriosaAI RNGD card is 8 PEs on one package sharing 47.5 GiB of HBM,
and an accelerator in our model is one PE (`furiosa-llm build -tp` counts PEs per
TP group), so the 8 PEs of a card must be joined by an **intra-island** link for
`detect_islands` to group them. None of the five fits: `NVLINK` and `HCCS` are
other vendors' names, and `PCIE` is simply false for elements that never leave
the package — it would also feed the Level-1 topology model the wrong
interconnect class.

**Adaptation**: added a vendor-neutral `LinkType.ONPACKAGE`, included it in
`INTRA_ISLAND_LINKS`, and ranked it first in `_dominant_link` so an all-PE island
reports `ONPACKAGE` rather than falling through to `None`. Existing cluster specs
are unaffected — nothing else uses it.

Its **bandwidth is `placeholder`** and must stay so until measured: PE-to-PE
on-package bandwidth was not measured. What *was* measured is HBM→PE read
(≈219 GB/s per PE, scaling to 8 PEs at 104 % efficiency, so a card sustains
≈1750 GB/s of reads), and that is a different quantity. It only feeds the TP
collective term, which ASTRA-Sim prices.

### (b) Cross-vendor P/D is unrepresentable unless the TP degrees overlap — open

`planner/candidate_generator.py::_pd_candidates` requires `tp_p == tp_d`, which
is D14's constraint: the simulator infers its topology as
`[npus_per_group, num_instances]` by integer division over the total device
count, so unequal instance sizes are unrepresentable. That interacts badly with
heterogeneity in a way neither D14 nor §5.4 anticipated:

- RNGD reaches only **tp ∈ {4, 8}** for Llama-3.1-8B, because a PE holds 6.25 GB
  (measured) against 14 GB of weights. tp=1 and tp=2 are correctly rejected on
  memory feasibility.
- An **NVLink-pair** A40 island offers only **tp ∈ {1, 2}**.

The two sets are disjoint, so **no mixed P/D candidate is generated at all** —
the first run of the re-done Exp 5 produced 3 representatives (GPU-P+GPU-D,
NPU-P+NPU-D, aggregated) with both mixed combos silently absent. Nothing errored;
the candidates simply did not exist, which is the dangerous failure mode: a
"heterogeneous P/D does not pay" conclusion could be drawn from a search that
never considered it.

**Adaptation** (fixture-level, not a code change): bridge the A40 NVLink pairs
over PCIe into **size-4** islands, exactly as
`experiments/configs/clusters/exp1-a40-tp-sweep.yaml` already does, giving
tp ∈ {1, 2, 4} so that **tp=4 is a shared degree**. Both mixed combos then
appear.

**Update 2026-09-04 (STEP B spike)**: `docs/d14_spike.md` measured that lifting
D14 is a `config_builder.py` change of two sites, and ran
`A40 tp4 prefill + RNGD tp8 decode` to completion under it. So "unrepresentable"
is now "unrepresentable **by the compiler we ship**", with a demonstrated path out
and a measured accuracy cost. The `tp_p == tp_d` guard in
`planner/candidate_generator.py` stays until that path is productionised — the
spike is not merged — but it is no longer a statement about the simulator.

**Why this stays open**: the workaround requires the island sizes to be
*choosable*. On hardware where the per-device memory forces disjoint TP sets and
the island sizes are fixed by the fabric, heterogeneous P/D remains
unrepresentable. Lifting it means addressing D14 itself — teaching the compiler
to emit non-uniform instance sizes — which is a simulator-side change and out of
scope here. Any claim about cross-vendor P/D must state which TP degree the two
backends shared, and whether one existed at all.

### (c) The constraint is ours, not the world's — verified 2026-08-26

`tp_p == tp_d` is an artifact of `_compute_network_dims`, not a property of P/D
serving. Asymmetric TP per phase is a **headline feature** of real disaggregated
systems, and the direction they recommend is the one we cannot express:

- **DistServe** makes independent per-phase parallelism a core contribution.
- **AWS Neuron** Disaggregated Inference documents `TP=4 for prefill attention
  and TP=1 for decode` explicitly.
- **NVIDIA Dynamo** supports differing prefill/decode TP first-class, transposing
  KV blocks into the receiver's layout between NIXL read and write, and
  recommends *"a larger TP for the memory-bound decoding phase while a smaller TP
  for the computation-bound prefill phase"*.
- **vLLM** `NixlConnector` supports heterogeneous TP (each decode worker computes
  which remote TP ranks to read, no extra copies); its `MooncakeConnector` does
  not, so the restriction is per-implementation, not fundamental.

The consequence for our results is concrete. Dynamo's recommendation — big TP on
decode — is exactly what RNGD needs: tp8 holds 246,079 KV tokens against 61,775
at tp4. So `GPU tp4 prefill + RNGD tp8 decode` is both the industry-recommended
shape and the one that uses RNGD's measured bandwidth advantage, and D14 forbids
it.

**Measured 2026-09-04**: that exact configuration has now been **simulated to
completion** on the spike branch (`docs/d14_spike.md`, 86 s, 21 rows, consistency
check passed). The shape is not unsimulable — it was unsupported by our compiler.
What the scoping sentence below must still say is that no *shipped* result uses
it, because the prototype is not merged. **Any "heterogeneous P/D does not pay" statement from this repo is therefore
scoped to the uniform-TP configurations D14 permits, and must say so.**

One further real-world limit, from FuriosaAI's own llm-d documentation:
**Furiosa-LLM does not support prefill/decode disaggregation at all** today. So
every RNGD P/D number here is simulator-only until that ships, independently of
D14.

---

## D17 — The §3.7 attention grid cannot express the vendor runtime's per-layer attention cost · Resolved (calibrated per workload, limit recorded)

**Found** 2026-08-26, while rebuilding the RNGD perf bundle from FuriosaAI's EDF
profiler (`experiments/results/rngd_edf_bundle_notes.md`).

**The work order and the simulator both assume** that per-layer attention cost is
a function of the batch's *shape*: `attention.csv` is keyed
`(prefill_chunk, kv_prefill, n_decode, kv_decode)` and `_lookup_attention()`
returns one number per layer for a given `n_decode` and mean KV.

**Furiosa-LLM's runtime groups a decode batch by KV bucket**, so one forward
issues as many attention executions per layer as the batch has distinct KV
buckets — measured over 1.74 M stage executions:

| sequences per forward | 1.95 | 3.91 | 8.91 | 15.16 | 29.09 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention executions per layer | 1.95 | 2.40 | 2.87 | 3.03 | 3.08 |
| per-layer attention (µs) | 88.3 | 131.3 | 198.0 | 254.2 | 329.7 |

Per-layer attention therefore depends on the batch's KV **diversity**, which is a
property of the traffic, not of `n_decode` and a mean. A single execution's cost
is *not* the per-layer cost: charging the `n_decode=16` bucket median (86 µs)
where reality runs three groups (254 µs) under-counts by ~3× at large batch.

**How we adapt.** Each concurrency pass contributes one total-preserving row:
`time_us` = total decode-attention device time ÷ (forwards × 32 layers). Totals
then close — predicted against measured wall time is within 5.2 % across a 32×
concurrency range — but the decode-attention axis is **calibrated to the traffic
it was measured on** (sharegpt, mean KV ≈ 2200, stable to ±1 % across all five
batched passes, which is why one calibration serves them all). A workload with a
very different KV spread would need its own pass.

**Why not extend the schema.** Adding a "KV-bucket count" axis would change
`serving/core/trace_generator.py`, which is upstream and pristine until Phase 5,
and it would be RNGD-specific: vLLM's paged attention runs one kernel per layer
regardless of KV spread, so the existing contract is right for every GPU profile
in the repo. The calibrated row is the honest local fix.

**Related, same source, recorded so it is not rediscovered:**

- **Mixed prefill+decode steps are absent from the traces.** Every attention
  bucket is pure prefill or pure decode, so the 4D grid has data only on the two
  axis planes and `_attn_slice_lookup()`'s nearest-slice fallback approximates
  the interior — which under-counts continuous-batching steps that do both.
  Whether the runtime genuinely never mixes them is **not decidable from these
  traces**: the EDF CSV carries durations, not timestamps, so co-occurrence
  within one forward is unobservable.
- **The vendor runtime compiles two plans for the same model.** Batch 1 runs a
  fully-fused `Composed` graph with no `Tokenwise` and no `Attention` stages at
  all; batch ≥ 2 runs the per-layer path. So there is no `input_size: 1` bucket
  anywhere in the traces, and a bundle keyed only on the bucketed path has no
  `tokens=1` row — the row decode needs most. The builder derives it from the
  fused graph instead, with the union / head / attention corrections
  `rngd_edf_bundle_notes.md` sets out.

---

## D18 — The NPU leg's multi-stream bandwidth was quoted as a peak where a sustained figure was needed · Resolved (retraction + remeasurement)

**What was wrong.** Four figures — host → RNGD PE aggregate of 5.06 / 10.39 /
19.10 / 35.47 GB/s at 1 / 2 / 4 / 8 streams, "88 % of ideal at 8" — appeared in
`docs/HANDOVER_A40.md` §1, `docs/PROJECT_REPORT.md` §4.8.2 and both P/D fixtures'
link comments, labelled as measured and citing
`outputs/rngd_profile/host_bandwidth.json`. **That file holds only the
single-stream run.** No committed code produced the 2 / 4 / 8-stream figures and
no artifact recorded them. The citation was false for three of the four numbers.

**What the remeasurement found** (2026-08-27, npu3, `--parallel-bandwidth`,
`outputs/rngd_profile/parallel_bandwidth.json`):

| streams | measured, sustained | previously quoted | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 3.77 | 5.06 | 0.745 |
| 2 | 7.60 | 10.39 | 0.731 |
| 4 | 15.36 | 19.10 | 0.804 |
| 8 | 26.27 (87.1 % of ideal) | 35.47 (88 % of ideal) | 0.741 |

**The scaling law was right; the levels were not.** 87.1 % against 88 % from an
independent implementation is a genuine confirmation that near-linear PE scaling
is real. But all four absolute figures were ~25 % high, and uniformly so.

**Why, and why it is not a hardware story.** Re-running the *committed* best-of-N
method on the same card the same day reproduces 5.06 GB/s exactly
(`outputs/rngd_profile/host_bandwidth_recheck.json`). The gap is entirely the
statistic. Decomposed at one PE, 256 MB: the committed method interleaves a
`.cpu()` between timed `.to()` calls, and the ~160 ms of idle that buys lets the
device free and recycle its buffer, so the *typical* transfer runs 5.01 GB/s;
remove the gap and back-to-back copies fall to 4.15; take a 5 s sustained window
instead of the best of 7 and it falls again to 3.67. A prefill → decode KV
handoff is a sustained bulk copy with nothing interleaved, so the sustained
figure is the one the planner should use.

**How we adapt.** The sustained figures replace the peak ones everywhere they
were quoted, each site carrying an explicit correction note. `card_of()` in
`experiments/scripts/rngd_device_facts.py` now resolves the physical card through
live sysfs enumeration rather than `index // 8`, because npu2 left the PCI bus and
torch renumbers densely over the cards that remain — under the old arithmetic this
run's artifact would have been stamped `npu2`, a card no longer in the machine.

**Left open deliberately at the time — CLOSED 2026-08-28.** When this entry was
written, `bandwidth_gbps: 35` still sat on the six `fabric-*` links of both P/D
fixtures, and the corrected NPU leg could not be composed into them because the
GPU leg existed only as prose: `feat/gpu-host-bandwidth` was not on `origin` and
both fixtures read `source: placeholder`. Composing against uncommitted figures
would have repeated exactly the error this entry retracts, so the recomputation
was deferred rather than guessed.

The GPU leg landed (PR #19) and the recomputation was done. All six links in both
fixtures now read `source: measured`, and the composed values match what
`experiments/results/rngd_parallel_bandwidth.md` predicted from the corrected NPU
leg to within rounding:

| fixture | link | predicted | committed |
| --- | --- | ---: | ---: |
| `pd-rngd-gpu.yaml` (tp4) | rngd ↔ a40 | 12.33 | 12.6 |
| | rngd ↔ rngd | 7.68 | **7.7** |
| `pd-rngd-gpu-card.yaml` (tp1) | rngd ↔ a40 | 13.07 | 13.0 |
| | rngd ↔ rngd | 13.13 | 13.1 |

The row this entry flagged to watch behaved as flagged: `rngd ↔ rngd` at tp4
falls to **7.7 GB/s**, below the ~10 GB/s P/D adoption crossing Exp 3 found.

**But the SLO sweeps could not see it**, and that is the more useful result. Both
were re-run at the measured fabric: all 16 winners unchanged, the per-PE fixture's
tight-TTFT row moving 372 → 371 ms, the card fixture byte-identical at all 8
points. Three reasons, none of them "the number does not matter": the simulator
prices the P/D KV handoff at **zero** by default (D15) and `pd_slo_sweep.py` does
not pass `--pd-transfer-model bandwidth`; `link_bw` acts on collectives and the
card fixture has none, every candidate there being tp1 on both sides; and the
planner-side penalty is a per-request latency term worth 0.13–0.36 % of the
cross-vendor candidate's p99. So these sweeps are not evidence that fabric
bandwidth is irrelevant — they are evidence that *this configuration of them
cannot answer the question*. See the PR #19 commit chain and
`experiments/results/pd_slo_sweep.md`.

---

## D19 — The RNGD TTFT gap was blamed on the scheduler; it was the validation harness · Resolved (retraction + refit)

**What was wrong.** `docs/PROJECT_REPORT.md` §4.8.4 and
`experiments/results/rngd_edf_bundle_notes.md` recorded the card profile's −71.3 %
TTFT error as "upstream's scheduler queuing ~2.2x less than furiosa-llm's", and
left "*which* knob -- `max_num_seqs`, chunked-prefill admission, or the P/D
interleave" as the open question. §6 carried it as "diagnosed, not fixed".

**The actual cause.** The two sides of the validation disagree about the arrival
process. `python -m serving` replays the trace's `arrival_time_ns` column, which in
`outputs/envcheck/rngd20.jsonl` spreads 20 requests over **1.78 s**.
`experiments/scripts/bench_furiosa_endpoint.py` fires every row under
`asyncio.Semaphore(concurrency=64)` against 20 requests, so all 20 start at once --
**the string `arrival` does not appear anywhere in that file.** The simulator queued
less because less arrived at once. That is not a scheduler property.

**Evidence** (`experiments/results/rngd_ttft_gap_resolved.md`). Same trace, same
profile, every arrival zeroed:

| profile / arrivals | TTFT | err | queue | prefill | TPOT | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| card-EDF / spread | 403.0 | -71.3 % | 253.3 | 149.7 | 27.55 | -3.1 % |
| card-EDF / burst | 1332.9 | **-5.1 %** | 1126.5 | 206.3 | 27.60 | -3.0 % |
| per-PE tp8 / spread | 946.9 | -32.6 % | 655.5 | 291.5 | 35.70 | +25.5 % |
| per-PE tp8 / burst | 2004.8 | **+42.8 %** | 1699.9 | 304.9 | 35.69 | +25.5 % |
| real | 1404.1 | -- | ~1246 | ~158 | 28.44 | -- |

TPOT does not move, which is the control. The per-request TTFT *distributions*
match too, not only the means: real spans 402 -> 2897 ms and burst-sim 432 ->
2603 ms in the same ramp, where the spread run spans 41 -> 1119 ms.

**Two further things it overturns.**

1. **The per-PE profile is not the better TTFT model.** Its -32.6 % was two errors
   cancelling -- a queue 2.2x too short times a prefill cost 93 % too high (304.9 ms
   against a real ~158). With arrivals matched it is +42.8 % where the card profile
   is -5.1 %. §4.8.6's recommendation to use per-PE "for TTFT-feasibility decisions"
   is inverted and has been corrected.
2. **Both TTFT calibrations were fitting the bug.** Refitted from burst runs through
   the same `--calibration-out` path: `rngd_card_edf.yaml` TTFT goes alpha
   2.089 -> 1.241, beta +646 -> -242, **fit error 2.340 -> 0.103** (it was recorded
   as "unusable"); `rngd.yaml` goes alpha 1.336 -> 0.851, beta +183 -> -302, error
   0.473 -> -0.263. TPOT is essentially unchanged in both (0.025 -> 0.019,
   -0.204 -> -0.206), the same control again.

**How we adapt.** The burst trace (`outputs/envcheck/rngd20_burst.jsonl`) and both
burst validation CSVs are committed, the calibrations are refitted, and
`bench_furiosa_endpoint.py` now states in its docstring that it ignores
`arrival_time_ns`. Every doc that carried the scheduler diagnosis is corrected in
place rather than quietly overwritten.

**`pd_slo_sweep.py` was checked and does NOT share the mismatch.** It builds its
own arrival process from the ServiceSpec (`planner/util/workload.py:generate_trace`,
Poisson at `arrival_rate_rps: 10`), so there is no bench to disagree with. But the
check found a different reason its card rows are not quotable: the sweep's winner
runs each card at **~76 concurrent sequences**, against 16.6 in the validation run
and 32 the highest ever tested on hardware, and assumes 1767 output tok/s per card
where extrapolating the measured c16->c32 scaling exponent (0.598) gives ~1090 --
**~1.6x optimistic, 2.4x outside the measured envelope.** Neither calibration may
be applied there either: both are scoped to the `sharegpt-llama31-8b-20` bucket.
`experiments/results/pd_slo_sweep.md` is rewritten accordingly.

**Left open.** A 10-17 % tail under-prediction remains at p90-p99 with arrivals
matched -- plausibly bucket quantisation (+10.9 % of charged prefill tokens), but
that is a hypothesis. Other comparisons that pair `python -m serving` with
`bench_furiosa_endpoint.py` still inherit the mismatch and have not been audited.
(An earlier version of this paragraph claimed the c1-c32 scaling curve was
prose-only with no committed artifact. **That was wrong** --
`outputs/rngd_edf_bundle/edf/real_c{1,2,4,8,16,32}.json` are committed and
reproduce the table exactly. The extrapolation rests on measured data.)

What the concurrency check *did* expose as genuinely unmeasured: the highest
concurrency ever run on RNGD is 32, and the sweep's operating point is ~76 per
card. Settling whether the simulator's high-concurrency throughput is valid needs
a c64/c128 run on the hardware -- see
`docs/npu_concurrency_envelope_work_order.md`.

---

## D20 — ATOM cannot be layerwise-profiled to contract fidelity: host I/O exceeds the kernels, and the device profiler is unreadable · **Open (blocks ATOM in Exp 4)**

**What the work order assumes.** `docs/HANDOVER_NPU.md` §3 says that once the
`rebel-compiler`/`vllm_rbln`/`optimum-rbln` versions are consistent in an
rbln-only venv, "the existing vLLM profiler works as-is". The packaging half of
that is now true — `.venv-rbln-vllm` carries a consistent 0.11.0 trio and
`RblnPlatform` activates. The conclusion does not follow.

**Why vLLM cannot drive it.** The profiler's `HOST_ENGINE_DEFAULTS` fixes
`load_format: "dummy"` and `enforce_eager: True`, both marked "should not be
changed". vllm-rbln's optimum path AOT-compiles a *real* checkpoint and rejects
dummy weights; its vLLM-native path accepts them but rejects eager unless
`VLLM_RBLN_USE_DEVICE_TENSOR=1`, which needs a torch device named `rbln` that
nothing on this machine registers (no `torch_rbln`, unlike `furiosa.torch`
registering `rngd:` as PrivateUse1). That path also registers only
deepseek_v2 / gpt_oss / minimax_m2 / qwen2 / qwen3 — not llama. Both fixes
require editing `profiler/`, pristine until Phase 5.

**Why the RNGD-style harness cannot either.** `experiments/scripts/profile_atom.py`
runs fine — 284 shots, zero compile failures — but cannot produce *device* time.
`rebel._C.profiler` emits protobuf traces with `comp_cycle`/`transfer` records
and no published schema, and no decoder ships with the stack, so wall clock is
the only instrument. On this card wall clock is dominated by transport:

| input | bytes | pure-I/O µs |
| ---: | ---: | ---: |
| 8 | 16 | 6.4 |
| 1,024 | 2 KB | 56.3 |
| 1,048,576 | 2 MB | 300.6 |
| 4,194,304 | 8 MB | 999.7 |

against RNGD device spans of 3–200 µs for the same layers. **The transport costs
more than the computation**, so the measurement is I/O with a kernel inside it.

**Three subtraction schemes, all defeated.** A constant floor (6.5 µs, calibrated
on a 1×8 tensor) inflated elementwise layers 8–25× and *produced a bundle that
passed contract validation* — it was deleted. A per-shot I/O baseline works for
single-input layers (`o_proj` at 0.83–1.09× RNGD) but its `sum()` over extra
inputs costs more than the layer for multi-input ones (27,919 µs on one attention
shape; 145/284 shots negative), and the cheap single-index variant lets the
compiler elide the unread tensors instead. A repetition slope is defeated by its
own accumulator. The invariant behind all three: **the transfer only happens if
the graph consumes the data, and consuming it costs compute.**

**How we adapt.** No `profiler/perf/ATOM/` bundle is shipped. A partial one is
not possible either — `attention.csv` is `required=True`, and attention is
precisely the case the subtraction cannot handle, since a decode shot carries
megabytes of K/V whose transfer is inseparable from its compute. The CSVs the
harness did produce are kept under `outputs/atom_profile/layerwise_attempt/`
with their sidecar, explicitly **not** as a bundle. Full evidence:
`experiments/results/atom_layerwise_blocked.md`.

**Consequence.** `profiles/accelerators/rbln_atom.yaml` keeps `sim_hardware:
null` and empty `supported_models`, so ATOM stays out of candidate generation and
out of Exp 4, even though its memory and power are now measured. Absolute rule 3:
present, idle, importable and partly measured is still not profiled.

**What would resolve it**, in order of expected effort: (1) the `.pb` trace
schema from Rebellions — the tracer already records what is needed, so this is a
documentation request; (2) a torch backend registering device `rbln`, which
enables the vLLM-native path; (3) a llama entry in vllm-rbln's native model
registry, after (2).

---

## D21 — Tier 0 introduction: `datasheet:` fields are vendor spec, not measurements · Decided 2026-09-02

**What.** The tiered-profile work (WORK_ORDER_tiered_profiles.md) adds a
`datasheet:` block to `AcceleratorProfile` so a roofline generator can emit
synthetic (`analytical`/`calibrated`) perf bundles for hardware we do not
own. Those numbers are copied from public vendor documents and are labelled
by `datasheet_source`; they are **never** measurements, and the bundles they
produce carry `tier: analytical` (or `calibrated`) plus a `-t0`/`-t1`
hardware-label suffix so they can never shadow a measured bundle.

**Discipline.** `flops_efficiency` / `mem_efficiency` are left empty until
they are fitted from a real measured bundle (STEP 8 of that work order):
pre-filling them would present an invented derating as usable. A profile
whose `sim_hardware` ends in `-t0`/`-t1` without a `datasheet:` block is
rejected at load time. The planner propagates the weakest bundle tier into
`PlannerOutput.profile_tier` with a mandatory caveat, so an analytical plan
can never be read as a measured result.

### D4 addendum (2026-09-02): resolved via the Tier 0 path

The tiered-profile work (WORK_ORDER_tiered_profiles.md, D21) closed the gap
without waiting for external measurements: `profiler.synth` generates a
datasheet-derived (`tier: analytical`) bundle under
`profiler/perf/ASCEND_TARGET-t0/` (regenerated deterministically by
`scripts/gen-tier0-bundles.sh`; synthetic bundles are gitignored so measured
and synthetic data never mix in the tree). `ascend_target.yaml` now points
`sim_hardware` at that label, so Ascend islands survive candidate generation
and full simulation; every resulting plan carries
`profile_tier: analytical` and the simulator-only caveat. Measured/imported
data (the original Phase 3 path) still supersedes this whenever it arrives.

**Coexistence note (STEP 10 item 5).** The candidate generator's stage-5
analytical bound (`candidate_generator._stage5_analytical_ok`, memory-BW
decode lower bound) and the Tier 0 `RooflineModel` remain two separate code
paths on purpose: stage-5 is a *pruning relaxation* whose only permitted
failure mode is under-rejection, while the Tier 0 model is a *bundle
generator* whose numbers feed the simulator. Unifying them (adding a compute
term to stage-5) is deferred to the optional S1 follow-up, which needs its
own golden-update plan.

---

## D22 — The RNGD scaling curve's top point was request-pool-limited, and the envelope beyond it is now measured · Resolved (retraction + measurement) · **basis re-validated 2026-09-07**

> **Re-validated under D25 + D26** (`docs/d23_revalidation.md` §2). The
> `pd_slo_sweep_margin18` rows this entry rests on were re-run on a harness with all
> three faults fixed, 32 workers, no isolation wrapper: **0 timeouts** against the
> original 71, 1 h 43 m against 4 h 37 m, and **every reported field identical to the
> last decimal** — winner `agg[cuda:tp4]`, 2.5954323001631323 tok/J, p99 TTFT
> 15070.10466225, p99 TPOT 35.58226791. The verdict is unchanged.
>
> It also holds in the **tight** regime, which was undetermined when this was
> written. All four tight points are now FEASIBLE (§3), and filtering all 424 cached
> per-candidate records against the SLO gives **0 of 45 RNGD-only and 0 of 18 mixed**
> candidates passing on the tp4 fixture, 0 of 12 and 0 of 25 on the card fixture. So
> "every RNGD candidate rejected" is now measured at the tight points rather than
> left unevaluated there.


**What was wrong.** `experiments/results/pd_slo_sweep.md`, `docs/PROJECT_REPORT.md`
§4.8.7 and `docs/npu_concurrency_envelope_work_order.md` all rested on two figures
from the committed c1–c32 curve: that **32 was the highest concurrency ever run on
RNGD** at ~648 output tok/s, and that the curve's marginal exponent was **0.598**
between c16 and c32. Both are artifacts of the harness.

Every committed point used a **24-request pool** while requesting up to 32
concurrent, so the pool bound the experiment. Average in-flight concurrency by
Little's law (`Σ latency / wall`):

| committed point | requested | pool | **actually served** | tok/s |
| --- | ---: | ---: | ---: | ---: |
| c8 | 8 | 24 | 7.4 | 299.4 |
| c16 | 16 | 24 | **12.2** | 427.4 |
| c32 | 32 | 24 | **21.2** | 646.4 |

So the top point was ~21, not 32, and the 0.598 exponent reads a pool-capped ×1.74
interval as a ×2 doubling. Re-measured at a non-binding pool the same levels give
**585.8 tok/s at c16 (+37 %)** and **908.6 at c32 (+40 %)**, with TPOT essentially
unchanged at c32 (30.14 → 31.18 ms) — real throughput the harness was leaving on
the table, not a bookkeeping difference.

**What the measurement found** (2026-08-31, npu0, TP=8;
`outputs/rngd_envelope/edf/real_c{16,32,64,128}.json`):

| requested | pool | eff. conc | output tok/s | TPOT avg |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | 15.3 | 585.8 | 25.71 |
| 32 | 128 | 29.3 | 908.6 | 31.18 |
| 64 | 256 | 59.2 | 1277.0 | 44.54 |
| 128 | 300 | **107.2** | **1473.3** | 67.88 |

Exponent decays 0.675 → 0.485 → **0.241**; the curve flattens decisively, which it
had not done by the old top point. All 812 requests succeeded.

**Two simulator errors at the load the card fixture uses.** Interpolated to
eff 76: measured **1346 tok/s** against the simulator's 1767 (**1.31× optimistic**,
not the ~1.6× the work order predicted from the bad exponent), and measured TPOT
**52.7 ms** against 43.2 (**18 % optimistic**). The TPOT divergence is the one the
work order asked for and did not expect: measured TPOT is already 44.54 ms at
eff 59, past the simulator's c76 prediction. **The decode model is accurate at the
concurrency it was fitted on and degrades above it** — its −3.1 % agreement does
not extend to this range.

**How we adapt.** The correction is a throughput/latency *model* error, not a
calibration offset: it varies with concurrency, so no scalar expresses it and it
does not belong in `profiles/calibration/`. It is recorded as a profile-level
caveat with the curve attached
(`experiments/results/rngd_concurrency_envelope.md`), and per the work order's
§4.4 the committed sweep results are **not** silently rescaled —
`pd_slo_sweep.py` must be re-run at a defensible load before its absolute numbers
are quoted again.

**The envelope is no longer the largest open risk.** The card reaches eff 107 with
zero failures, so c76 is inside what the hardware serves; the question moved from
"can it?" to "at what cost?", and the cost is measured.

**Limits of this measurement.** One card (npu0), one artifact, one dataset. The
300-line trace caps the pool, so c128 ran at 2.3× headroom and reached eff 107
rather than 128 — the only point where the pool still binds slightly. TTFT from
these runs is **not** comparable to the sweep's p99 TTFT: the bench fires the whole
pool at once (D19), a closed-loop saturation probe, while the sweep offers Poisson
arrivals. Throughput and TPOT are the valid comparisons.

### Reason 1 below was a D26 artifact — re-tested 2026-09-07

> `WORK_ORDER_rps_aware.md` STEP 0.1, `docs/rps_step0_retro.md`. The distribution
> recorded below — every RNGD-involving candidate stuck, most `cuda ↔ cuda` through
> — is **D26's shape**, and D26 was not known until 2026-09-07. Re-tested on the
> fixed harness, one RNGD P/D and one cross-vendor candidate at 3.3 rps, 20
> requests, under `livelock_watch.sh`:
>
> | candidate | 3.3 rps | 10 rps |
> | --- | ---: | ---: |
> | RNGD P/D `rngd0 tp4 P + rngd1 tp4 D` | **74 s, completed** | 69 s |
> | cross-vendor `a40 tp4 P + rngd0 tp4 D` | **82 s, completed** | 69 s |
>
> Both claims below fall, and separately. **"RNGD candidates never terminate"** is
> refuted: these are the classes that went 0-for-36. **"The drain is unbounded at a
> lower rate"** is not supported either — 3× the arrival span costs 7 % and 19 %
> more wall time, and one extra tick of simulated time.
>
> **Caveat**: this used 20 requests where the abandoned run used 300, so a drain
> effect that grows with request count is not excluded. What is excluded is
> structural non-termination, which the mechanism below predicts at any count.
>
> **Reason 2 below still stands** — it is an argument about the objective
> (lower load lets the planner satisfy the SLO with fewer accelerators, pushing
> per-card concurrency back up), and nothing about D26 touches it. Lowering the
> arrival rate is still not the way to control per-card concurrency; that is what
> the envelope and `accuracy_domain` of `WORK_ORDER_rps_aware.md` are for.
>
> Consequence: E6's low-RPS axis is available.

### The re-run this entry calls for cannot be done by lowering the arrival rate — attempted 2026-09-01 and abandoned

§4.4 asks for `pd_slo_sweep.py` at "a defensible load". The obvious reading is to
lower the offered load until the winner's per-card concurrency lands where the
profile was validated (~16.6) rather than at ~76. Per-card concurrency is an
*outcome* of the Poisson arrival rate, not an input, and the sweep exposes no
arrival-rate flag, so this was tried with a service spec at **3.3 rps** (9.9 × 25/76,
targeting ~25 per card) identical to `examples/service_specs/llama31-8b.yaml` in
every other field. **It does not work, for two independent reasons, and the run
was killed after 3.7 hours.**

**1. RNGD candidates never terminate at that rate.** After 3.7 hours, by candidate
class:

| class | attempted | completed |
| --- | ---: | ---: |
| `furiosa ↔ furiosa` (RNGD P/D) | 24 | **0** |
| cross-vendor (`cuda ↔ furiosa`) | 12 | **0** |
| `cuda ↔ cuda` | 84 | 60 |

Completions decayed 102 → 24 → 6 → 0 per hour as the sweep worked through the
CUDA-only candidates and reached the RNGD ones. The same candidates complete
normally at 10 rps — `agg[furiosa:tp8]` is a *winner* in the committed sweep. The
mechanism is that a slower arrival rate gives the decode scheduler less to batch,
which lowers throughput, which lengthens the *simulated* time needed to drain 300
requests. On a device whose per-token cost is already high that is unbounded in
practice. The naive cost model — "3.3 rps triples the 30 s window to 91 s, so
expect 3× runtime" — is wrong: the window is the arrival span, not the drain time.

**2. It would not have produced the intended load anyway.** The objective is
`minimize_energy` then `minimize_active_accelerators`. Lower offered load lets the
planner satisfy the SLO with *fewer* accelerators, which pushes per-card
concurrency back up. Reducing the arrival rate therefore does not control
per-card concurrency; it trades fleet size against it, and the two effects fight.

**So the arrival rate is the wrong knob: it changes the operating regime rather
than the load level, and the result would not have been comparable to the
committed sweep point-for-point even if it had finished.** The service spec used
for the attempt is deliberately **not** committed — a spec that cannot be
simulated to completion is a trap for whoever finds it.

**What was done instead — and the answer is stronger than "optimistic".** Neither
the arrival rate nor the fleet is the right knob. `exhaustive.search()` already
took `tpot_margin_percent`, which inflates predicted p-TPOT before the
feasibility check; that is exactly the shape of a measured model error, so the
re-run held 10 rps and applied **18 %**, the TPOT optimism measured at the
concurrency these plans run at.

**Every RNGD configuration on the card fixture is then rejected**, on both
fixtures, converging on the same A40 plan — so the result does not depend on
whether an RNGD accelerator is modelled as a card or as 8 PEs:

| | committed (no margin) | with the measured margin |
| --- | --- | --- |
| card fixture, TTFT ≤ 64 s | `agg[furiosa:tp1]` n=2, 3.164 tok/J | `agg[cuda:tp4]` n=4, **2.595 tok/J** |
| tp4 fixture, TTFT ≤ 64 s | `agg[furiosa:tp8]` n=8, 4.956 tok/J | `agg[cuda:tp4]` n=4, **2.595 tok/J** |

**The committed winner is not optimistic, it is infeasible.** It clears the 50 ms
TPOT SLO by 1.59 ms, so any margin above **3.3 %** rejects it — and the profile's
own agreement at its fitted concurrency is −3.1 %. The RNGD arm is squeezed
between the two SLOs: high per-instance concurrency (s128/s256) meets TTFT and
breaks TPOT at 56–58 ms; low concurrency (s32) meets TPOT at 38.7 ms and breaks
TTFT at 57 s, **at any margin including zero**. No setting satisfies both.

So the half of the three-regime answer that says RNGD wins on energy at loose
TTFT does not survive. Full table and per-candidate arithmetic:
`experiments/results/pd_slo_sweep_margin.md`.

**Still open: the tight-TTFT regime.** That run printed INFEASIBLE at TTFT ≤ 8 s
and ≤ 500 ms, and **that is an artifact, not a result** — `--timeout` was lowered
to 1080 s on card-fixture evidence (no successful sim exceeded 14.9 min) which
did not hold for tp4, where all 72 `pd_cuda-a40-tp4` candidates timed out. One of
them is the committed tight-TTFT winner at p99 TPOT 37.27 ms, which *passes* the
margin at 43.98. That regime needs a re-run at 1800 s. The loose-TTFT finding is
unaffected: zero RNGD candidates timed out on the card fixture.
---

## D23 — The P/D tight-TTFT candidates livelock: prefill pinned at one request, decode never fed · **Resolved 2026-09-07 (harness faults: D26 explains the symptom, D25 the crashes)**

> ### Root-caused 2026-09-07 — it was the Chakra converter's interpreter (D26)
>
> `WORK_ORDER_d23_fix_revalidation.md`. The symptom recorded below — prefill
> holding one request, decode never fed, memory flat, the simulated clock
> advancing — is produced by `serving/core/graph_generator.py` invoking the
> workload converter as bare **`python`**. Which interpreter builds the `.et`
> graph then depends on `PATH`, and on this node the two candidates carry
> different `chakra`/`protobuf` pairs (6.33.1 against the `>= 7.35.1` CLAUDE.md
> requires). The same trace converts to **different `.et` bytes**, and a P/D
> simulation built from the wrong ones never finishes its first prefill batch.
>
> Holding everything else constant and varying only `PATH`: **18 runs completed
> with the venv first — every CSV `sha256`-equal to the committed `sim1.csv` — and
> 6 hung with `~/.local` first.** No exceptions. Fixed by one token,
> `sys.executable`; see **D26** for the full record and for the five hypotheses
> refuted on the way, one of which was D25's own edit.
>
> **Two separate harness faults, not one.** D25's `tmp__mem` race is real and kills
> 13 of 64 concurrent processes, but it produces *crashes*, and the spike showed it
> was not masking this. D26 produces this. The heading below was wrong in both of
> its claims: the candidates do not livelock, and the failure is not a property of
> the candidates.
>
> **The description below is also wrong about *which* candidates.** Classifying the
> committed timeout lists: of 197 timeouts, **not one** was a single-instance
> candidate — and there were 144 `dp1` work directories available — while **18 were
> `aggregated dp2`**, which are not P/D at all. So "every `pd_*` and `mix_*`
> candidate" is wrong in both directions: not all of them, and not only them. The
> discriminator is **instance count**, which is D26's (a mis-converted `.et` graph
> breaks where collectives span more than one rank group) and not D25's. Every
> other measurement in this work order sorts on the same axis: the two
> single-instance `bench/examples` runs are byte-identical across the fixes, while
> the two-instance MoE example's hash flips with `PATH`.
>
> **Consequence for the results.** D22's verdict was re-run on the fixed harness and
> **holds exactly** — winner `agg[cuda:tp4]`, 2.5954323001631323 tok/J, every
> reported field equal to the last decimal, 0 timeouts against the original 4 h 37 m
> and 71 (`docs/d23_revalidation.md` §2). Completed past results stand: a run either
> reproduced the committed answer byte for byte or produced nothing. The tight-TTFT
> points are being re-run in STEP 3.3.


> ### Diagnosed 2026-09-04 — the heading below is wrong: they do not livelock
>
> `WORK_ORDER_spikes.md` STEP A. Full record: `docs/d23_spike.md`. Evidence:
> `outputs/d23/evidence/`.
>
> **The candidate completes alone.** With the sweep's own `cluster.json`
> (`link_bw` 35.2), a trace verified `sha256`-identical to the sweep's, and flags
> compared token-by-token against the recorded `command:` line, it finishes at
> every request count tried — **343 s at N=300**, the exact point that burned
> 1800 s and then 3600 s. H1, H2 and H5 (environment, `link_bw`, workload) are
> refuted; H3 and H4 were never reachable because there was nothing to inspect.
>
> **The real fault is a race in ASTRA-Sim on a fixed, cwd-relative temp path.**
> `astra-sim/.../congestion_unaware/main.cc:27` writes, reads and `std::remove`s
> `tmp__mem/<name>.json` with no pid and no run id, three times per start
> (`local_mem`, `remote_mem`, `cxl_mem`). Every concurrent process shares one cwd
> — `astra-sim/`, which the frontend chdirs into — so one removes the file another
> is opening. Launching 64 bare `AnalyticalAstra` processes on identical inputs,
> with no frontend, no planner and no Chakra involved, **13 of 64 fail**: five with
> `Unable to open file: tmp__mem/remote_mem.json` (exit 1) and eight with
> `terminate called without an active exception` (SIGABRT).
>
> **`--run-id` does not cover it.** That flag isolates the *input tree*;
> `tmp__mem/` is outside it and no flag reaches it. CLAUDE.md's "parallel
> simulations need no extra locking" is therefore **wrong** and is corrected there.
>
> **Two frontend bugs turn that crash into a four-hour timeout.**
> `serving/core/controller.py:14` loops forever when the child is gone — a dead
> `p.stdout.readline()` returns `""`, which matches neither exit condition, so the
> loop spins at 100 % CPU and grows its `out` list unbounded (measured: RSS
> 6.51 → 6.68 GB in 30 s, child a zombie at exit 1). And
> `serving/__main__.py:548` captures the child's stderr and never reads it, so the
> one line that explains everything dies with the pipe. There is no `poll()`, no
> `returncode` check, nowhere in `__main__.py`.
>
> **Where to fix, in the order the work order asks for.** The root cause is in
> `astra-sim/`, which absolute rule 1 and the spike's A3 both forbid touching —
> reported upstream instead (`docs/upstream_issues/`, both bugs unfixed at their
> heads as of 2026-08-23/28, and astra-sim has zero open issues). The frontend
> bugs are `serving/` edits of the D15 kind and belong to the next work order,
> opt-in and byte-identical by default, with `outputs/.hp-pd-slo/`'s completed
> candidate as the regression anchor. A planner-side workaround exists without
> touching either: give each candidate its own cwd, which needs the `../` path
> convention in `llmservingsim.py` reworked.
>
> **How to adapt, today — there is a working isolation.**
> `experiments/scripts/astra_isolated.sh` puts an unprivileged mount namespace and
> a private tmpfs over `astra-sim/tmp__mem`, touching neither `serving/` nor
> `astra-sim/`. Measured: the 64-way concurrent run that lost 4 instances twice
> now completes **64 of 64** (503–566 s), and the bare binary goes from 13 failures
> to none. It **refuses** (exit 2) where namespaces are unavailable rather than
> falling back — a silent unisolated fallback would restore a race whose failure
> mode is a four-hour timeout, not an error. Available on the NPU node; check
> per node, it is not a property of the repository.
>
> Passing `cwd=<per-run dir>` to the `Popen` at `serving/__main__.py:548` works too
> (64/64 measured) and is the better long-term fix because it cannot be forgotten;
> it is a `serving/` edit and belongs to the next work order. All four arguments
> the frontend passes are already absolute, so the child's cwd is used for nothing
> but `tmp__mem` — one argument is enough.
>
> And wrap runs in `experiments/scripts/livelock_watch.sh` regardless.
> `--log-level WARNING` hides none of this — the symptom is indistinguishable from
> slowness at any log level, which is why longer timeouts kept confirming the wrong
> theory. The watcher ends a stuck run in seconds and separates a tick-stall
> (exit 3) from a run that never reports or stops reporting (exit 4).
>
> **Still open, and the isolation did not change it.** D23's *original* symptom —
> 52,903 progress ticks with prefill pinned at one running request and memory flat
> — was **not** reproduced. The obvious question was whether the `tmp__mem` race
> had been masking it; it had not. With the race removed all 64 concurrent runs
> completed, each emitting ~102 ticks, the same as a solo run. Alone the candidate
> completes; concurrently it either completes or dies before emitting a tick, and
> neither is what D23 recorded. The untested difference is that the sweep ran 64
> *different* candidates, not 64 copies of one. The tight-TTFT regime therefore
> stays undetermined.

**What.** Every `pd_*` and `mix_*` candidate that the tight-TTFT sweeps need has
stopped terminating. The 2026-09-03 re-run (`--timeout 1800`, both P/D fixtures,
`experiments/results/pd_slo_sweep_margin.md` § Tight-TTFT regime) timed out 71 of
222 card-fixture simulations and 126 of 252 tp4-fixture simulations, and printed
INFEASIBLE at all four tight points. Those verdicts are *not evaluated*, not
rejected.

**It is not a timeout.** Work order §7 permits one escalation for the committed
winner alone — 1 candidate, 3600 s, single retry — and it was taken on
`P[cuda:tp4] D[cuda:tp4]` (`-s256-t8192`). It failed the same way, and the hour of
logs says why. The simulator emitted **52,903 progress ticks**, so its simulated
clock advanced continuously. What never advanced is the work:

```
Instance[0] (prefill): 1 reqs running at EVERY tick, Waiting 7 -> 299
Instance[1] (decode) : 0 reqs running, 0 waiting, for the entire hour
memory               : flat at 9.304 % / 9.234 %
```

The prefill instance admits exactly one request and never retires it, the decode
instance is never handed anything, and the whole 300-request trace piles up in the
arrival queue. Raising the timeout cannot help.

**Distinct from D12.** D12 is prefix-cache memory growing monotonically until the
run dies. Here memory is flat at 9 % on both instances and prefix caching was
disabled for the run (`--no-enable-prefix-caching`). The two share only the
symptom the predictor reports — `SimOutcome.TIMEOUT`.

**A regression, and the cause is open.** The same candidate is committed as
completed: `outputs/.hp-pd-slo/` has it at **280.6 s**, p99 TPOT 37.32, p99 TTFT
361.6, 2.206 tok/J. So this is a fall of at least 12.8x into non-termination. The
only difference found in the compiled simulator input is `link_bw` 35.0 -> 35.2
GB/s from the D18 fabric recomputation — a 0.6 % change, unconvincing on its own.
The intermediate `pd_slo_sweep_measured_fabric` run also completed this candidate,
but its work directory has been cleaned, so its `link_bw` cannot be read back. No
bisect was run: the consolidation sprint makes no new numbers (its §0.3).

**What this blocks.** The three-regime table's sub-second row
(`P[cuda:tp4] D[cuda:tp4]`, 372 ms p99 TTFT) is neither confirmed nor refuted, and
cannot be until this is diagnosed. D22's loose-TTFT retraction is unaffected: it
rests on completed simulations.

**How to adapt.** Do not read INFEASIBLE from a P/D sweep without checking the
timeout count first — the sweep driver does not persist
`PlannerOutput.rejected_summary`, so the count has to be recovered by counting
work directories with no `sim*.csv`. Both lists are committed:
`outputs/pd_slo_sweep_margin18/tight/timeouts_*.txt`, with the retry's tick-level
evidence in `tight/retry3600_livelock_evidence.txt`.

**What would resolve it**, cheapest first: (1) bisect `link_bw` between 35.0 and
35.2 on this one candidate — if the cliff is real it is a threshold bug, not a
bandwidth effect; (2) diff the simulator inputs of the `.hp-pd-slo` run against
this one beyond `cluster.json`, in particular the A40 perf bundle, which the Tier 0
merge touched; (3) instrument the prefill instance's admission path. All three
belong to the RPS-aware work order, not to this sprint.

---

## D24 — `profiles/networks/` is in the work order's layout but not in the planner's Level-1 path · Resolved (moved out with ScenarioLab)

**Work order §(repo layout)** lists `profiles/networks/{nvlink,pcie_gen5,ib_100g,ib_400g}.yaml`
under "[신규] 하드웨어/네트워크 profile catalog", alongside `profiles/accelerators/`.

**The real code puts those values somewhere else.** CLAUDE.md's two-level topology
model says Level 1 is "interconnect-class representative values", and that is
implemented inline in `planner/topology.py`:

```python
CLASS_DEFAULT_GBPS = {LinkType.INFINIBAND: 400.0, ...}    # ib_400g.yaml: bandwidth_gbps 400
CLASS_DEFAULT_LAT_NS = {LinkType.INFINIBAND: 5000.0, ...} # ib_400g.yaml: latency_ns 5000
```

The YAML files duplicate the same numbers and nothing in `planner/`, `examples/`,
`experiments/` or `tests/` reads them — verified by grep before removal. Their own
headers say what they are for: *"Inter-node 400G InfiniBand link class used by
ScenarioLab's random cluster generator"*, with every value `source: placeholder`,
copied verbatim from `examples/clusters/heterogeneous-lab.yaml`.

**How we adapt.** They moved to `swsok/heteropilot-scenariolab` with the rest of
ScenarioLab (`WORK_ORDER_consolidation.md` STEP 3.3), where the random cluster
generator is their only consumer and they are anchored at that repo's root rather
than at a heteropilot-relative path. `experiments/configs/lab/*.yaml`, their only
other referent, went with them.

This follows CLAUDE.md's standing rule for this class of divergence — *"When spec
and reality diverge, the real code wins. Record the difference in
`docs/deviations.md` and continue"* — rather than the consolidation work order's
§0.5 stop-and-report, because the layout list is not a functional requirement and
the planner never grew a code path that reads the files.

**If Phase 5 ever needs per-class profiles as data** rather than as constants, the
place to put them back is `profiles/networks/`, and the values are recoverable
from `planner/topology.py` or from the split repo. Nothing about this removal
forecloses that; it removes an unread duplicate, not a capability.

---

## D25 — ASTRA-Sim's cwd-relative temp file: the frontend gives each run its own working directory · Resolved (second sanctioned `serving/` edit)

**Context.** `WORK_ORDER_d23_fix_revalidation.md` STEP 1. D23's spike measured that
ASTRA-Sim's analytical backend writes, reads and removes `tmp__mem/<name>.json` at
a path relative to its working directory, with no pid and no run id
(`congestion_unaware/main.cc:28`, three times per start for `local_mem`,
`remote_mem`, `cxl_mem`). The frontend chdirs into `astra-sim/`
(`__main__.py:198-199`), so every concurrent simulation shared one such directory
and they deleted each other's file: **13 of 64** bare `AnalyticalAstra` processes
launched together died, five with `Unable to open file: tmp__mem/remote_mem.json`
and eight with SIGABRT. `--run-id` / `--inputs-root` isolate the *input tree*;
`tmp__mem/` is outside it and no flag reaches it.

**The edit** is one argument at `serving/__main__.py`'s `Popen`:
`cwd=run_paths.inputs_root`. `serving/core/controller.py` and everything else are
untouched. The root cause is in `astra-sim/`, which absolute rule 1 forbids
touching before Phase 5; it is reported upstream in
`docs/upstream_issues/astra-sim-tmp-mem-race.md` (unfixed at head, and astra-sim
has zero open issues).

**Why one argument is enough — with a correction to the work order.** §1.2 of the
work order says the child's cwd "is used for nothing except `tmp__mem`". That is
not quite true. Every *path argument* is indeed already absolute, and
`tests/test_astra_cwd.py` asserts it: the binary via `os.path.join(astra_sim, …)`,
and the four configurations via `run_paths.py`'s `abspath` (`astra_sim` itself is
computed at `__main__.py:198`, one line **before** the `chdir`, so it is absolute).
But ASTRA-Sim's logger also writes **cwd-relative** `log/log.log` and `log/err.log`
via spdlog rotating sinks, 10 MB × 10 files each (`common/Logging.cc:48-58`) —
which is why `astra-sim/log/` had grown past 100 MB. After this edit those logs
follow the run into its own tree and `--cleanup-inputs` removes them with it. So
the accurate statement is not "the cwd is used for nothing else" but **"the only
other use is the debug log, and moving it is harmless and in fact tidier"**.

**Regression anchors.** R1 — the three `bench/examples/` runs — is byte-identical
across the edit (`dd0eca3f…`, `a0563bc7…`, `0b548376…`). R2, the completed P/D
candidate, is byte-identical too, but only once D26 below was also fixed; before
that it could not be made to complete at all under an unlucky `PATH`.

**`experiments/scripts/astra_isolated.sh` is no longer required** and says so at
the top. It is kept as a belt-and-braces measure for a node where it is not certain
the running frontend carries D25, and because it is the only mechanism that also
covers a *stray* ASTRA-Sim process started outside the frontend.

**One consequence worth knowing.** `_cleanup_inputs_root` (`__main__.py:86-96`)
removes the tree only on the success path and only when `inputs_root` is under
`astra-sim/inputs/runs/`. A timed-out or crashed run therefore leaves its inputs —
and now its `tmp__mem/` and `log/` — behind. Measured on this node: **239 of 284**
leftover run directories, 15 GB, all from the timed-out `pd_*`/`mix_*` sweep
candidates. That is not new to D25, but D25 makes those directories slightly
larger.

## D26 — the Chakra converter ran under whatever `python` PATH found, and that is what D23 actually was · Resolved (third sanctioned `serving/` edit)

**This is D23's root cause.** `docs/d23_spike.md` closed with D23's original
symptom — 52,903 progress ticks, prefill pinned at one running request, decode
never fed, memory flat — recorded as **not reproduced and unexplained**. It is
explained here, and the explanation is not the `tmp__mem` race of D25.

**The defect.** `serving/core/graph_generator.py` built the workload-conversion
command as

```python
cmd = ['python', '-m', 'chakra.src.converter.converter', 'LLM', …]
subprocess.run(cmd, cwd=chakra, text=True, check=True)
```

`'python'` is resolved through `PATH`, so **which interpreter converts the
ASTRA-Sim workload graph depends on the environment of whoever launched the
simulator** — not on the interpreter the frontend is running under.

On this node the two candidates are not equivalent:

| resolved `python` | `chakra` | `protobuf` | CLAUDE.md requires `>= 7.35.1` |
| --- | --- | ---: | --- |
| `.venv/bin/python` | `.venv/lib/…/chakra` | 7.36.0 | satisfied |
| `~/.local/bin/python` | `~/.local/lib/…/chakra` | **6.33.1** | **not satisfied** |

CLAUDE.md already carries `uv pip install "protobuf>=7.35.1"` with the reason
"Chakra gencode 7.35.1 needs it". The second install silently violates it.

**Measured, end to end.** The same trace converted by the two interpreters produces
**different `.et` bytes** (`llm.0.et` … `llm.3.et` all differ). A P/D simulation
built from the wrong ones never completes its first prefill batch — prompt
throughput `0.0` at the very first tick, prefill holding one request, decode at
zero, memory flat, the simulated clock racing. Holding the input, the cluster
config, the binary and `serving/` constant and varying **only** `PATH`:

| `PATH` order | runs | outcome |
| --- | ---: | --- |
| `.venv/bin` first | 18 | **18 completed**, every CSV `sha256` = the committed `sim1.csv` |
| `~/.local/bin` first | 6 | **6 hung**, all with D23's signature |

No exceptions in either direction. The two-each controlled pair is
`outputs/d23fix/pathexp/`.

**Why it looked intermittent for a month.** Nothing in the failure mentions the
converter, the interpreter or protobuf. The frontend reports a slow simulation, and
the only visible remedy is a longer timeout. Whether a given run hit it depended on
the `PATH` of the shell, the script or the scheduler that launched it — so the same
candidate completed in one sweep and timed out in the next, which reads exactly
like a regression in the candidate. D23 recorded it as "a fall of at least 12.8×
into non-termination".

**The edit** is one token: `sys.executable` in place of `'python'`. The frontend is
already running under the interpreter that has the correct `chakra`, so it is the
one that must do the conversion. `tests/test_chakra_interpreter.py` asserts the
command names `sys.executable` and, separately, that this interpreter's `chakra`
and `protobuf` actually satisfy the requirement — because `sys.executable` is only
the right answer if the venv is correctly provisioned, and otherwise the fix would
move the failure rather than remove it.

**Scope.** The work order's rule A3′ permitted only the two edits of D25. This
third file was approved explicitly by the user on 2026-09-07 after the diagnosis,
because the two permitted edits do not fix D23 and D25's own R2 anchor could not be
established without it.

**What this changes elsewhere.**

- **D23's heading is wrong twice over.** The candidates do not livelock (the spike
  established that) *and* the tight-TTFT timeouts are not a property of the
  candidates at all.
- **The tight-TTFT regime's 71/126 and 126/252 timeouts are suspect as an
  environment artifact.** If those sweeps ran without the venv first on `PATH`,
  re-running them under D26 should simply complete. That is
  `WORK_ORDER_d23_fix_revalidation.md` STEP 3.3, and its character changes from
  "settle an open question" to "re-run and read off the answer".
- **Completed past results stay trustworthy.** Every one of the 18 completions
  reproduced the committed `sim1.csv` byte for byte, which is the evidence STEP 3.1
  was designed to obtain. A run either produced the right answer or produced none.
- **Upstream should have this.** `graph_generator.py` is upstream code; the bug is
  upstream's. Note upstream's own head has since replaced this subprocess with an
  in-process call (`fa6fbde`, "Run the Chakra converter in-process instead of
  per-batch subprocess"), which removes the defect as a side effect — so the fix
  to report is "you already fixed this; it bites everyone still on 2c2042ce".

**An independent confirmation, arrived at by way of a retraction.** STEP 0 of this
work order reported that the committed `bench/examples/Qwen3-30B-A3B` baseline was
stale, on the evidence that three reruns agreed with each other and disagreed with
it — and even offered a cause (upstream `c4edd0a` landed after the baseline
refresh, and the MoE example is the only one with two non-DP instances). **That was
wrong.** Under D26 the same example reproduces the committed `7d0ff3ce…` four times
out of four; the three agreeing runs agreed because none of them had `.venv/bin`
first on `PATH`. The MoE example is the only single-node example that goes through
the multi-instance path, so its hash flips with the interpreter exactly as the P/D
anchor's does — which makes it the sharpest of the three anchors and, incidentally,
the reason `docs/phase0_formats.md` §2.1 and CLAUDE.md's "safe regression anchor"
were right all along. The reasoning error is worth keeping: *reruns agreeing with
each other says nothing when they share an environment defect.*

**Five hypotheses were refuted before this one**, and they are recorded because
each cost a run: `link_bw` 35.0 vs 35.2 (both complete solo); `AnalyticalMemory`
racing its own `std::remove` (the constructor reads eagerly); **D25-a's own
`cwd=` edit** (reverting it changed nothing); orphaned ASTRA-Sim processes from a
killed launch (a clean process table still hung); and dependence on which run
preceded (`R1 → R2` completes).

## D27 — the Chakra converter is called in-process, not spawned · Resolved (fourth sanctioned `serving/` edit)

`WORK_ORDER_rps_aware.md` STEP 1 asked for a wall-time breakdown and gave a rule:
convert the Chakra subprocess to an in-process call if it exceeds 30 % of wall time.
It measured **28.9 %** at 10 rps and **29.6 %** at 3.3 rps — below the threshold — and
the edit was made anyway, on the user's direction, because the cheaper lever the rule
assumed was available turned out not to be. `docs/sim_cost_profile.md` carries the
decision and the numbers; this entry records the code change.

**The defect, such as it is.** `generate_graph()` spawned a fresh Python interpreter
once per instance per iteration to convert a 295-row trace:

```python
subprocess.run([sys.executable, '-m', 'chakra.src.converter.converter', 'LLM', …],
               cwd=chakra, text=True, check=True)
```

Timed on a real trace, 20 calls: **46–47 ms per call, of which 5–7 ms is the
conversion.** The other 40 ms is fork, exec, interpreter start and imports — paid
thousands of times per run to do 5 ms of work.

**The fix.** `chakra.src.converter.llm_converter.LLMConverter`, imported once on first
use and called directly:

```python
_llm_converter()(trace_path, output_path, num_npus, npu_offset,
                 enable_local_offloading).convert()
```

**`LLMConverter` and not `converter.main()`**, which is what the work order's sketch
suggested. `convert_llm()` is a three-line wrapper around this constructor, so nothing
is skipped by going one level down — and `main()` would bring two side effects into
the frontend's process:

- `setup_logging()` calls `logging.basicConfig(level=DEBUG, …)`, which reconfigures the
  **frontend's** root logger on every call;
- that same call defaults to a **cwd-relative `debug.log`**. The subprocess absorbed it
  because it ran with `cwd=` the chakra directory; in-process it would land in the repo
  root, once per instance per iteration.

`main()` also parses `sys.argv`, so a caller would have to swap argv and `chdir`.
Bypassing it removes all of that. The import is deferred to first use so that importing
`serving` stays cheap for the planner, which pulls in `serving.core.memory_model` for
its own feasibility arithmetic and never builds a graph.

**The work order's stated risk — module-level state surviving between calls — does not
exist.** `converter.py` has no module-level mutable state, and repeated conversions
produce identical bytes.

**Verification.** Seven independent byte comparisons, all equal:

| artifact | result |
| --- | --- |
| R1 ×3 (`Llama-3.1-8B`, `Qwen3-32B`, `Qwen3-30B-A3B`) | identical to `after_d25_d26` |
| R2 (`P[cuda:tp4] D[cuda:tp4] -s256-t8192`) | identical (`fff63c22…`) |
| the three STEP 1 rate runs (10 / 3.3 / 1 rps) | identical to their pre-D27 CSVs |

plus `tests/test_chakra_inprocess.py`, which compares against the **live subprocess
path** rather than a stored digest, so it stays true if the vendored chakra is updated,
and asserts the two `main()` side effects above are absent.

Measured saving: **1.55–1.72×** across the three rates. Of the 113 s saved at 10 rps,
80 s is `generate_graph` — 23.7 % of wall, matching the 24 % the breakdown predicted —
and 27 s is `read_wait` shrinking, which is observed but not attributed
(`docs/sim_cost_profile.md`).

**D26 is subsumed but stays in the ledger.** There is no longer an interpreter to
resolve, so the PATH hazard is gone by construction; what replaces it is an
`ImportError` at first conversion if the venv lacks chakra. That is the strictly better
failure — this venv has `ENABLE_USER_SITE = False` and no `~/.local` on `sys.path`, so
there is no wrong-version fallback to silently succeed with.

## D30 — the roofline surrogate's proxy is invariant to TP and DP, so top-K is not a cost lever on P/D or heterogeneous corpora · Open (measured, not fixed)

*D28 and D29 are reserved for `WORK_ORDER_rps_aware.md` STEP 2 and STEP 4.*

Full measurement in `docs/surrogate_topk_regret.md`; this entry records the
divergence and what it forbids.

**What the work order assumed.** §E6a planned to cut a 218 h sweep with
`--top-k 20`, citing §4.7's *"regret 0 at every K down to K=1"*. STEP 1 found the
same setting turning a FEASIBLE plan INFEASIBLE on `pd-rngd-gpu`.

**The divergence.** `AnalyticalRooflineRanker` orders by `greedy.estimate`'s proxy
tok/J, and that quantity cancels exactly across the parallelism axis:

```
throughput = Σ active/step_s × dp_replicas      step_s = (W/tp + a·K/tp) / BW
power      = Σ active_power × tp × dp
```

Both scale with `tp · dp`. Measured across seven tp/dp configurations on one
accelerator, the proxy tok/J spread is **0.000463** (s32) and **0.001848** (s128);
`dp1` through `dp4` agree to six decimals. The ranker discriminates on accelerator
and `max_num_seqs` and on nothing else — while on this fixture parallelism is what
decides feasibility (`tp4-dp1` 49.40 ms against a 50 ms TPOT SLO, `tp2-dp2`
53.47 ms).

The one parallelism-sensitive term, `roofline_tpot_ms`, feeds only a binary
`likely_infeasible` flag, and that flag fired for **0 of 324** candidates: the floor
underestimates simulated p99 TPOT by a median **2.92×** (1.77–11.26×, n=180). So the
ordering is decided by the fifth significant digit of a near-constant.

**What it forbids.** `--top-k` is not used as a cost lever on P/D or heterogeneous
sweeps. The shipped ranker is false-infeasible at K=20 on two of three corpora
(N=324, 492, 468). K=30 is clean on all three, but three fixtures do not license a
threshold and they were swept under different SLO margins.

**Why nothing was changed.** Ordering by the roofline floor instead fixes those two
corpora and **breaks the third**, where the shipped ranker is already perfect at
K=10. Adopting it would repeat precisely the error being corrected here — a ranker
justified on the fixtures where it happens to win. The alternative orderings live in
`exp_surrogate.py --rankers` so the comparison is reproducible, and none is the
default.

**Two harness defects fixed while measuring this**, both in
`experiments/scripts/exp_surrogate.py`:

- `--cache-dir` replays an existing `EnvelopeCache` corpus, so a past sweep's
  simulations can answer a regret question without re-running. A candidate absent
  from the corpus stays in the ranking and yields no plan — the cache stores only
  `result.ok`, so absent means simulated-and-failed, and it did consume a top-K slot.
  Dropping such candidates instead (the first attempt) deleted exactly the high-ranked
  ones that deliver nothing and reported the shipped ranker as fine at K=20,
  contradicting the observed run.
- **Regret was `None` for every minimisation objective.** `pareto.objective_value`
  negates minimisation objectives so callers can always maximise; the regret formula
  divided by the *signed* value behind an `oracle_value > 0` guard that was therefore
  never true. The denominator is now `abs(oracle_value)`. The published curve is
  unaffected — its oracle value is +1.660, which only
  `maximize_slo_goodput_per_joule` can be.

## D28 — asymmetric TP per phase: `topology_mode: slab3d` · Resolved (fifth sanctioned `serving/` edit)

*D29 is reserved for `WORK_ORDER_rps_aware.md` STEP 4.*

**What D14 and D16(b) said, and why it was wrong.** Both record that the
simulator "requires uniform instance sizes", so `tp_d = 2 · tp_p` was out of
reach and heterogeneous P/D had to be worked around. The `spike/d14-asym-tp`
investigation (`docs/d14_spike.md`) established that the simulator never required
it:

- ASTRA-Sim reads a per-collective `involved_dim` (`Workload.cc:275-296`) and
  supports up to five dimensions;
- the vendored Chakra converter writes an arbitrary-length `involved_dim`
  through to the ET (`llm_converter.py:226-233`);
- `tp_dim` is **already** a per-instance field, and the EP path already uses
  two-dimensional involvement.

The uniformity came from one function. `_compute_network_dims` folded every
instance into `[npus_per_group, num_instances]` with an integer division, and
`_resolve_dp_groups` then gave all of them the same `local_dim`.

**The edit.** A cluster config may set a top-level `topology_mode`. Absent, or
`"auto"`, is the pre-D28 path byte for byte. `"slab3d"` computes `[g, 2, n_slabs]`,
where `g` is the smallest compute TP: a tp=g instance occupies half a slab and a
tp=2g one a whole slab, so `tp_d = 2 · tp_p` costs no idle rank.
`A40 tp4 prefill + RNGD tp8 decode` is `[4, 2, 2]` — 16 ranks, none idle.

**`tp_dim` is keyed on the collective, not the footprint.** A prefill instance
occupies `2 · tp` ranks (compute + sender) but its TP group is still only `g`
wide, so it gets `[T, F, F]` while a full-slab decode gets `[T, T, F]`. Keying on
the footprint gives both `[T, T, F]`, which declares a 2g-rank allreduce for a
g-rank group; the prefill batch then waits for ranks that never join. That was
done once during the spike, presented **exactly as D23's signature**, and was
reported as "D23 reproduced" before it was found to be self-inflicted.
`tests/test_slab3d_config.py` pins it with a case where prefill and decode occupy
equal rank counts and must still differ.

**What is refused rather than reshaped**: odd half-slab counts, MoE/EP instances,
widths that are neither `g` nor `2g` (4× ratios would need idle-rank padding,
whose interaction with the frontend's iteration barrier is unverified), and a
full-slab instance that starts half a slab in — `[half, full, half]` straddles a
boundary because ranks are handed out as consecutive blocks.

**`_FMT` comm_type widened 15 → 24** (`serving/core/utils.py`). The row format
pads but does not truncate, so a three-dimensional tag runs into the next column
and the whitespace-splitting reader silently mis-assigns every field after it. At
width 15, `ALLREDUCE:1,1,0` and a comm_size of `4` merge into `ALLREDUCE:1,1,04`:
ten fields where there should be eleven. Eliminating the reparse is
`docs/upstream_issues/llmservingsim-trace-column-overflow.md`; this widens the
column only.

**Byte-identity.** Every trace row goes through `formatter`, so the widening moved
the *text* of every row from column 9 on. The parsed CSVs did not move: R1 ×3 and
R2 reproduce `after_d25_d26` exactly, and **R3** — colocated tp4×2 under `auto`
against the same under `slab3d` — is byte-identical between the two modes. R3 is
the strongest of the three, because two colocated tp4 instances are two half slabs
and therefore yield `[4, 2]` with `[T, F]`, exactly what `auto` computes: it says
the new path agrees with the old where they overlap, not merely that it stays out
of the way. `experiments/scripts/slab3d_anchors.sh`.

**The calibration is done (STEP 2.2, 2026-09-09).** The flat ring becoming
hierarchical makes decode look **24.4 %** faster at tp8 and **13.4 %** at tp4;
multiplying dim 1's `link_latency` by **4** for `[4,2]` and **2** for `[2,2]`
removes it to within **0.008 %**, at all three bandwidths measured. The factor is
bandwidth-independent, split-dependent, and recorded as a table rather than a law
in `profiles/calibration/slab3d_latency.yaml` with
`validity.extrapolation: refuse`. Both values equal `tp/2`, which hop counting
derives — `2(tp-1)` flat against `2(tp/2-1) + 2` split — but a derivation is not a
measurement and unmeasured splits stay refused. `docs/slab3d_calibration.md`.

So a `slab3d` plan at a **measured** `(split, link_bw)` may now quote absolute
TPOT. Three things it still may not: any other split or bandwidth, TTFT and
throughput (only TPOT p50 was fitted), and **P/D configurations** — the fit is
single-instance and colocated by construction, and a real P/D run also sends the
prefill compute→sender COMM_SEND across dim 1, which this experiment excluded on
purpose so the allreduce could be isolated.

**Still open.** The planner side (STEP 2.3), including the
`OUTSIDE_CALIBRATION_DOMAIN` rejection for candidates the table does not cover.

**`split2` came back as an instrument.** 2.1 was right to leave it out — it
describes no placement — but 2.2's calibration is defined as *single instance,
flat vs split*, and splitting one TP group is the only way to attribute the
difference to the allreduce rather than to a changed instance mix. It refuses
anything but one colocated non-MoE instance, is excluded from
`DEPLOYABLE_TOPOLOGY_MODES`, and a test asserts `planner/` never emits it.

## D31 — utilisation does not explain RNGD card power; the §3 power model is keyed on served concurrency instead · Resolved (schema adapted, measurement kept)

`docs/rps_aware_planning_design.md` §3 proposes
`power_model: {kind: piecewise_linear_in_util, ...}` and argues the case well: *"a
power figure without the utilisation it was taken at is not a measurement"*. That
principle stands and A5(c) keeps it. What does not transfer is the **functional
form**, and the design's own example says why.

**The measurement** (`WORK_ORDER_rps_aware.md` STEP 3.2, two repeats per point,
agreeing to within 0.72 %):

| served conc | util % | power W | tput tok/s |
| ---: | ---: | ---: | ---: |
| 1.00 | 92.1 | 151.1 | 63.1 |
| 1.99 | 88.0 | 140.6 | 110.1 |
| 3.98 | 86.5 | 139.9 | 200.9 |
| 7.88 | 85.3 | 143.2 | 356.8 |
| 15.59 | 84.7 | 151.8 | 598.2 |

**Utilisation falls monotonically while power falls and then rises.** No monotone
function of utilisation fits, piecewise-linear or otherwise; the correlation is
r = +0.24. The card sits between 84.7 % and 92.1 % across the whole range — a
7.4 pp span — so utilisation has almost no dynamic range to explain an 8.2 %
movement in power that reverses direction.

§3's example is an **ATOM** card at 36.2 / 64.4 / 95.1 % utilisation reading
44.3 / 54.9 / 68.7 W. There utilisation spans 59 pp and is monotone in power, and
the schema is the right one. The divergence is device- and range-specific, not a
flaw in the design.

**What was done.** `profiles/accelerators/furiosa_rngd_card.yaml` gains a
`power_model` with `kind: piecewise_linear_in_served_conc`, its five measured
points, and `validity.extrapolation: refuse` above conc 15.59 where power was
never measured. Each point still carries the utilisation it was taken at, because
A5(c) is about provenance rather than about the fit. The pre-existing scalar
`power:` block is untouched for back-compatibility.

**Why the shape is worth keeping in view.** The U is physical: at concurrency 1
every decode step streams the whole model for a single token, so the card is
memory-bandwidth-bound and draws 151 W to produce 63 tok/s; batching amortises
that traffic, and the minimum near c4 is where the memory-bound and compute-bound
terms cross. **Across a 9.5x throughput range the power moves 1.085x**, so energy
per token is set almost entirely by throughput — 0.418 tok/J at c1 against
3.941 at c15.59, a **9.4x** span. That is the quantitative form of the premise
behind this whole work order: at low load an accelerator is not merely slower, it
is dramatically less efficient, and a planner that treats performance as a scalar
cannot see it.

**What is NOT claimed.** Power above conc 15.59 is unmeasured. D22's four points
(conc 15.3 to 107.2) keep `power_w: null` and were not back-filled (A2).
