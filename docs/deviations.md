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

> **LIFTED 2026-09-09 for the 2x case (D28).** The uniformity is not the
> simulator's; it was `_compute_network_dims` plus a shared `local_dim`.
> `topology_mode: slab3d` expresses `tp_d = 2 * tp_p` with no idle rank, the
> generator enumerates it, and the compiler emits it with a measured dim-1 latency
> correction. Other ratios still need idle-rank padding, whose interaction with the
> frontend's iteration barrier is unverified, and remain enumerated around.


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
| D22 | Phase 4 | **Resolved** (retraction + measurement), **basis re-validated 2026-09-07**. The c1–c32 curve's top point was a 24-request pool at eff 21.2; the envelope now runs 1.00 → 107.2. At eff 76 the simulator is 1.31× optimistic on throughput and 18 % on TPOT, and with that margin every RNGD config is rejected — the committed winner is infeasible, not optimistic. **Both regimes are now determined**: the tight half was D26, not the candidates. And since 2026-09-09 the planner finds this unaided (E5) |
| D23 | Phase 4 / Phase 5 | **Resolved 2026-09-07.** Not a livelock and not the `tmp__mem` race (real, D25, but it crashes rather than hangs). The cause is **D26**: the Chakra converter ran under a PATH-resolved `python`, so a wrong-protobuf `chakra` could build the workload graph and a multi-instance run then never finished its first prefill batch. 18/18 complete with the venv first against 6/6 hanging without. `docs/d23_revalidation.md` |
| D24 | — | **Resolved** — the work order's layout lists `profiles/networks/`, but Level-1 interconnect-class values live inline in `planner/topology.py` and the YAMLs were an unread duplicate read only by ScenarioLab's cluster generator. Moved out with it (STEP 3.3); recoverable if Phase 5 ever wants them as data |
| D25 | Phase 4 | **Resolved** — ASTRA-Sim writes `tmp__mem/<name>.json` cwd-relative with no run id; the frontend now gives each child `cwd=run_paths.inputs_root`. Second sanctioned `serving/` edit. D25-b adds dead-child detection so a killed child reports instead of hanging |
| D26 | Phase 4 | **Resolved** — `sys.executable` for the Chakra converter. **This is what D23 actually was**; third sanctioned `serving/` edit. Subsumed by D27, kept because it explains the symptom |
| D27 | — | **Resolved** — the converter is called in-process rather than spawned. 1.55–1.72× faster, seven byte-identical comparisons. Fourth sanctioned `serving/` edit. Done against the 30 % rule, because the cheaper lever the rule assumed (`--top-k`) turned out to be unusable — D30 |
| D28 | Phase 5 | **Resolved** — `topology_mode: slab3d` expresses `tp_d = 2·tp_p` with no idle rank; fifth sanctioned `serving/` edit, R1/R2/**R3** byte-identical. Lifts the 2× case of D14 and D16(b). Its dim-1 latency correction is a 24-point measured table with `extrapolation: refuse`, not a law |
| D29 | Phase 4 | **Resolved** — the SLO margin comes from the candidate's own operating point, not a hand-set constant. One-sided; manual and automatic coexist with the larger winning; hardware without a domain gets 0 and a note. The A40 domain is a separate opt-in file so the default path stays byte-identical. **D29b**: `power_model` is parsed and deliberately unused by the planner's energy, because changing that definition would break every historical tok/J comparison |
| D30 | — | **Open (measured, not fixed)** — the roofline surrogate's proxy tok/J is algebraically invariant to TP and DP, so `--top-k` is false-infeasible at K=20 on two of three P/D corpora. The obvious repair (order by the roofline floor) fixes those two and breaks the third, so **no ranker change was made** and top-K is not used as a cost lever. `docs/surrogate_topk_regret.md` |
| D31 | — | **Resolved** — utilisation does not explain RNGD card power (falls 92.1 → 84.7 % while power is U-shaped, r = +0.24), so the §3 power model is keyed on served concurrency instead. Device- and range-specific; §3's ATOM example spans 59 pp of utilisation and the schema is right there |
| D33 | uncertainty planner (Stage A/B) | **Decided 2026-09-11** — two accuracy-domain implementations were built in parallel from 108e48a (rps STEP 4's `calibration.AccuracyDomain`, per hardware, `widen_error_bars`; the uncertainty stack's `predictor/accuracy_domain.py`, per bucket, `unmeasured` outside). One remains: main's curve, `refuse` by default, the uncertainty stack's per-candidate `MarginPolicy` on top, `UNMEASURED` absorbed into `outside_calibration_domain`, E-A2's domain dropped as the D32 mis-pairing |
| D34 | uncertainty planner (Stage B) | **Resolved 2026-09-14** — the register of which closed-form perturbation rules are approximate: `SIM_ERROR`, `LINK_BW`, `LINK_LAT` exact; `PROFILE`, `POWER` first-order. Writing it down found that `PROFILE`'s "energy unchanged" clause contradicted the simulator — `power_model.py:73` makes active energy linear in operator latency, and a ×1.38876 profile moved measured energy 162 260 → 225 190 J (×1.3878). The double-counting rationale was wrong: energy is watts × seconds, PROFILE moves the seconds and POWER the watts, so they compose rather than double-count. **Fixed** — `_scale_profile` scales energy with latency and recomputes tokens/J; ΔR on an energy-only decision goes 0.0000 → 375.0000. The one-line §2.7 amendment is the owner's |
| D35 | uncertainty planner (Stage B) | **Recorded 2026-09-14** — STEP B5 asks for a `README.md` paragraph and a `CHANGELOG.md` entry, but both files are pure upstream LLMServingSim with no fork content, and upstream's own README policy keeps CLI tables off it. The paragraph went to `CLAUDE.md` §Commands and the long form to `docs/uncertainty_planner.md`; the two upstream files are untouched. The same rule applies to any later work order asking for those files — cite this entry rather than opening a new one |
| D40 | uncertainty planner (Stage B) | **Open (measured, not fixed)** — `EnvelopeCache`'s placement key records each island's `hardware|arch|tp|pp|ep|dp` but not WHICH island got which share, so a pair differing only by mirroring the split collapses to one entry. Found by E-B3's identity control: `mix(a40a-tp2-dp2+a40b-tp2-dp1)` truly predicts p99 TTFT 518.95 ms / 108 750 J and its mirror 563.28 ms / 108 930 J — 8.5 % apart — and the cache serves the second to both. Either the key is too coarse or the simulator is order-sensitive for a logically identical configuration; which has not been established. The E-A1 headline winner is unaffected and re-simulates to its cached entry exactly |

**Reading order.** The entries below are in the order they were written, not
numerically: D30 and D31 precede D28 and D29 in the file because the surrogate and
power-model findings landed before the work orders that reserved those numbers.
This table is the index.

**Still open, in the order they bind:** **D20** (ATOM, blocks it from candidate
generation), **D10** (memory derating), **D30** (no usable surrogate for P/D
corpora), and the one this sprint created — **a second A40 accuracy-domain point**,
without which every cuda row in E6 reads `extrapolated`. That last needs an NVIDIA
node and this one has none.

---

## D16 — `LinkType` has no on-package fabric, and cross-vendor P/D needs a shared TP degree · Resolved (one added type) + Open (the TP constraint)

> **(b) LIFTED 2026-09-09 for the 2x case (D28).** "The simulator requires a
> shared TP degree" is false for it: `slab3d` encodes `tp_d = 2 * tp_p` directly,
> and `plan --enable-pd` now enumerates, compiles and ranks such candidates. The
> size-4 island bridging workaround in the fixture still works and is kept, but is
> **no longer required**. (c) is unaffected: cross-vendor P/D before Phase 5
> remains out of scope, and this changes only how a pair is *encoded*, not which
> pairs are permitted.


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

  **Resolved by the artifact buckets, 2026-09-15/16 — see D90.** The sentence
  above stays as written, because it is still true *of these traces*. The
  compiled grid answers it from the other side: every prefill and extend bucket
  is `batch_size = 1` and every decode bucket is `input_ids_size = 1`, so no
  compiled plan can hold a prefill chunk and a decode token at once. The runtime
  structurally never mixes them, and the nearest-slice fallback is approximating
  an interior the hardware never visits.
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

> **Erratum, 2026-09-21: the third sum above is a pre-D26 value and does not
> reproduce.** `0b548376…` is the MoE anchor `r1_Qwen3-30B-A3B-Instruct-2507`
> as this entry first measured it, and that measurement was taken under the bug
> D26 names: the trace was converted by a second `chakra` beside protobuf
> 6.33.1, which emits different `.et` bytes. Under D26 the same example
> reproduces the committed `7d0ff3ce…` four times out of four
> (`docs/d23fix_baseline.md`, "All of that was wrong";
> `outputs/d23fix/anchor/README.md`, "`0b548376` -> `7d0ff3ce` … that change is
> the whole of D26"). Nothing hashes to `0b548376…` in the tree today.
>
> **The committed artifact's own history is separate and has two values, neither
> of them this one**: `7547edf1…` at the v1.1.0 release, then `7d0ff3ce…` after
> upstream's `3723a94` "Refresh validation baselines". So a reader checking R1
> against `bench/examples/Qwen3-30B-A3B-Instruct-2507/outputs/sim.csv` should
> expect `7d0ff3ce…`.
>
> **D25's claim is unaffected.** R1 is byte-identical *across the D25 edit*, which
> is what this entry asserts; the two runs it compared were both taken under the
> same (then-unfixed) converter, so the edit moved nothing. Only the recorded
> value is from a configuration that no longer exists. The sums are left in place
> rather than rewritten — rule A3 — and this note is the correction.
>
> Found while verifying PR #116, which needed R1 untouched. Traced once with
> `git log -p -S 0b548376 --all`; not pursued further.

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

## D30 — the roofline surrogate's proxy is invariant to TP and DP, so top-K is not a cost lever on P/D or heterogeneous corpora · Resolved for K>=20 (2026-09-11); the 20 rps end remains open

> **Update 2026-09-11 — a ranker that fixes it without breaking a corpus now
> ships.** The entry below ends *"no ranker change is made"*, and the reason
> was that the only alternative measured, `floor`, repaired two corpora and
> broke the third. The fix was hiding in this entry's own evidence:
> `tpj_then_floor` measured **byte-identical to `roofline`**, because the
> proxy's TP/DP cancellation is algebraic but the arithmetic is floating
> point -- about one part in ten thousand survives, no two values are ever
> exactly tied, and `sorted()` reads that dust as a preference. The ranker
> does not ignore the parallelism axis; it lets rounding error pick for it.
>
> `BinnedRooflineRanker` makes the tie explicit -- group proxy tok/J within a
> relative tolerance, order inside the group by `roofline_tpot_ms` -- and
> leaves the coarse order (accelerator, `max_num_seqs`, which differ by
> factors) alone, which is why it does not break the corpus `floor` broke.
> Measured on **sixteen** corpora rather than three, including E6's two
> sweeps read one arrival rate at a time: **96 (corpus, K) cells, 11 strictly
> better, 0 worse, 85 identical**, and false-infeasibility at K=20 falls from
> 8 corpora to 2. It is now `plan --surrogate`'s default; `roofline` stays
> selectable so published runs reproduce. `docs/surrogate_topk_regret.md`.
>
> **The tolerance is not fitted**: 0.001, 0.01 and 0.05 give identical regret
> curves and identical top-K membership everywhere.
>
> **What stays open.** At **20 rps** every efficiency-ordered ranker is
> false-infeasible to K=50 on both fixtures, while `floor` finds a plan --
> feasibility there is decided by the TPOT floor alone. So `--top-k` is still
> not a general cost lever; the exception is now characterised (the top of
> the load axis) instead of unknown. K=5 and K=10 are unchanged. And sixteen
> corpora are still only **two cluster fixtures**.
>
> **Rank fusion was tried and is worse.** Round-robin merging `binned` with
> `floor` -- no fitted weight, each parent's top K/2 inside any top-K -- is
> false-infeasible at K=20 on 7 corpora against 2, because the depth it gives
> up costs more than the complementarity buys. Kept in
> `exp_surrogate.py --rankers`; does not ship.

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

**The planner side is done (STEP 2.3).** `_pd_candidates` enumerates
`tp_d in {tp_p, 2 * tp_p}` and marks the 2x pairs `topology_mode: slab3d`; the
compiler emits that mode plus a per-dim `link_latency` whose dim 1 carries the
measured factor. A candidate whose `(split, link_bw)` is not in the table is
**refused**, into its own bucket:
`SimOutcome.OUTSIDE_CALIBRATION_DOMAIN` ->
`RejectionStage.OUTSIDE_CALIBRATION_DOMAIN`. That is an epistemic category, kept
distinct from both a feasibility verdict and a crash, because "never measured" is
not "does not work" and a sweep full of refusals must not read as a sweep that
found nothing feasible.

**The refusal earned its keep immediately.** The first end-to-end
`plan --enable-pd` on `pd-rngd-gpu.yaml` refused candidates for three points the
work order's table did not cover: **12.6 Gbps** (the A40<->RNGD cross-vendor link;
35.2 is the composed A40<->A40 value, not this one), **7.7** (`fabric-rngd0-rngd1`,
which is the asymmetric NPU P/D case D14/D16(b) was actually written about), and the
**`[1,2]`** split. Measuring all three brought the table to 15 points -- factors
still 4 / 2 / 1, bandwidth-independent over a 13x range -- and every asymmetric
candidate on the fixture now compiles. Had the compiler fallen back to a factor of
1.0 instead, those runs would have produced 12.6-24.4 % optimistic TPOT on exactly
the candidates D28 exists to enable, and reported them as ordinary results.

**The end-to-end run also caught a defect fourteen unit tests missed.** The compiler
sized its per-dim list from `_compute_network_dims` called on instance dicts it had
not stamped with the mode, so it got `auto`'s dimension count while
`build_cluster_config` -- which stamps first -- got slab3d's. Every asymmetric
candidate died in `_normalize_network_dim_values` with *"'link_bw' must have exactly
3 value(s) ... but got 2"*, and the first run's 84 `sim_error`s were that, not the
pre-existing P/D failures they resembled. The tests checked the list's CONTENTS and
never its LENGTH against the function the simulator would use. Fixed by stamping the
mode before computing, and a test now runs the config through
`_normalize_network_dim_values` itself.

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

## D29 — the SLO margin comes from the candidate's own operating point, and the A40 domain is opt-in · Resolved

`WORK_ORDER_rps_aware.md` rev 2 STEP 4.2–4.3, implementing
`docs/rps_aware_planning_design.md` §5.

**What was wrong with a constant.** `pd_slo_sweep.py --tpot-margin-percent` is one
number applied to every candidate, and the simulator's error is not one number. On
the RNGD card it is **+11.6 %** at served concurrency 3.9 and **−18 %** at 76 — not
merely different in size but opposite in sign. A constant is therefore too loose
somewhere or too tight everywhere, and D22 is what "too loose at the top" looks
like: the winner passed a 50 ms TPOT SLO at a predicted 48.41 ms, and
48.41 × 1.18 = 57.1 ms. The configuration was infeasible, not optimistic, and
nothing in the pipeline could see it because the 18 % was a fact about the
simulator that the simulator did not carry.

*That product is the pre-D70 arithmetic. The corrected margin `-e/(1+e)` gives
**59.04 ms**, and 57.1 was never a measurement (D70). The 18 % itself compares a
simulated p50 against a measured MEAN, where every other point in the file
compares p50 against p50; like-for-like it is -20.05 %, and on the p99 the
feasibility check actually reads, -17.31 % against a measured p99 of 58.54 ms
(D100, `experiments/p2_evidence/results/v0_source_reconciliation.md`). The
verdict is the same under all of them, which is why none of it was visible.*

**What replaces it.** `planner/util/operating_point.py` reads the served
concurrency each hardware kind actually ran at, out of the simulation's own CSV by
Little's law. `AccuracyDomain.tpot_error_at(conc)` prices the simulator's error
there, and feasibility applies it.

**Four decisions, each of which could have gone the other way.**

1. **The margin is one-sided.** Where the simulator is *pessimistic* the prediction
   is left alone rather than deflated. Deflating would make plans look better than
   the hardware measured, which is the direction of the retraction.
2. **Manual and automatic coexist; the larger wins.** A hand-set
   `--tpot-margin-percent` is an explicit instruction not to go below a floor, so
   it is a floor. Both values are recorded in provenance, so a plan says what it
   was checked against and not only which check bound.
3. **Hardware with no accuracy domain gets margin 0 and a note.** Never a margin
   borrowed from a different device — that would be rule 3 with extra steps.
4. **The A40 domain lives in its own file and is opt-in.** `profiles/calibration/a40.yaml`
   is on the default planning path; a domain there would apply an automatic margin
   to every A40 candidate and move the frozen output for both `examples/` specs.
   So it is `profiles/calibration/a40.accuracy.yaml`, reached only by
   `plan --accuracy-domain`, and the default path is byte-identical.

**The A40 point had to be computed, not read.** Its `fitted_at_concurrency` is
**170.56**, derived from `outputs/phase0_bench/A40/vllm/requests.jsonl` — whose
mean latency, TTFT and TPOT reproduce the committed summary exactly, which is what
licenses the derivation. It counts requests *in the system*, queued as well as
running, and the engine's `max_num_seqs` was 128: that run offered ~10.3 rps to one
A40 and the card could not keep up. The RNGD envelope uses the same definition but
was measured closed-loop, where in-system and running nearly coincide. **Comparing
the two domains' concurrency axes therefore needs care**, and the file says so.

**A measurement that changed the answer.** The RNGD domain originally had two
points, 16.6 and 76, and `widen_error_bars` extrapolated below 16.6 — producing a
4.8 % optimistic-side margin at concurrency 10. Measuring the low end
(`outputs/lowload_sim_error/`, STEP 4.2) found the simulator is **pessimistic**
there, so the correct margin is **zero**. The extrapolation had the sign wrong, not
just the magnitude. Four of six attempted points are in the domain; the two nearest
16.6 are excluded because the simulator settled 40 % below the hardware's occupancy
at the same offered rate, which makes its TPOT a different operating point's TPOT.

## D29b — `power_model` is not wired into the planner's energy, on purpose

STEP 4.6. `profiles/accelerators/furiosa_rngd_card.yaml` carries a measured
`power_model` (D31) and `planner/inventory.py` parses it. The planner does **not**
use it to compute energy, and that is a decision rather than an omission.

Today's `total_energy_j` comes from the simulator's **node-level** power
(`PROJECT_REPORT.md` §4.8.7: the ~558 W figure is node power, not card power).
Swapping in a card-level curve would change what `tokens_per_joule` *means*, and
every tok/J in `CLAIMS.md`, `PROJECT_REPORT.md` and the sweep results was computed
under the old definition. The comparison across all of them would silently break —
new numbers would look better or worse than old ones for a reason that is not a
change in the hardware or the plan.

So `power_model` is used **only** to compute the measured curve E6b compares
against, where it is the right instrument because that side is a card measurement.
If the planner's energy definition is ever changed, it must be a deliberate,
documented migration that re-derives the historical numbers — not a side effect of
this file appearing.

---

## D32 — a 20-request simulation is not at the same operating point as a 300-request measurement · Resolved 2026-09-11

> **Closed on the card fixture too, and the suspicion was right.** This entry
> ended by noting that `rngd_card_edf.yaml` discards two points at a "40 %
> below the hardware" gap it attributes to throughput error, that a -40 % gap
> is also the signature of the tail artifact, and that **the split between
> them has not been measured**. It is now.
>
> Same rates, same trace, only `--num-reqs` 20 -> 300: the two discarded
> points move from **-40.2 % and -40.6 %** to **-3.1 % and -2.4 %**. They were
> discarded for the wrong reason and are back in the domain. Their TPOT error
> also changes SIGN, -1.22 / -1.97 % to +3.26 / +2.47 %, and only a negative
> error earns a margin -- so readmitting them REMOVES a margin near served
> concurrency 15 rather than adding one.
>
> **The artifact is not the whole story above 29.3, and that is the other half
> of the split.** At the rates matching measured c59.2 and c107.2 the
> simulator still sits **-36.4 %** and **-58.5 %** low at 300 requests, having
> moved 2.9x and 3.3x from its 20-request values. Those two stay refused. The
> residual is the card model's real throughput ceiling: it saturates near
> served 44 and 35 ms TPOT where the hardware reaches 107.2 and 67.88 ms. So
> the answer is artifact-only at 29.3 and artifact-plus-ceiling above it.
>
> The domain is rebuilt at 300 requests, nine points from 1.020 to 76.0,
> including a new **25.181** -- the first measured point between the 16.6 and
> 76.0 anchors, an interval that previously held nothing and whose upper end is
> itself an interpolation. It reads -3.28 % against the EDF anchor's -3.1 % by
> a different method.
>
> **All eight E6 card rows were re-ranked and none changed** -- every winner
> and every tok/J identical to full precision, margins 3.05 -> 2.88 % and
> 12.52 -> 11.68 %. The re-rank was necessary rather than precautionary: a
> FALLING margin readmits candidates, so the argument used for the per-PE
> domain does not apply, and three candidates did flip to feasible.
> `experiments/results/d32_card_recheck.md`.

> **Update 2026-09-11 — the RNGD half is no longer unchecked.** Building the
> per-PE accuracy domain ran the same sweep both ways on RNGD hardware's
> simulated counterpart. At the two top low-load points the 20-request runs
> report served concurrency **10.82 and 10.92**; at 300 requests the same
> points report **19.47 and 19.88** — a factor of 1.8, with nothing physical
> between the columns. At 0.9 rps a 20-request run spans 22 s against a ~20 s
> mean latency, so the drain tail is half the wall: the A40's −31.7 % is worth
> −45 % here. Every point in `profiles/calibration/rngd_perpe.yaml` is from the
> 300-request runs, and `outputs/perpe_lowload_shape/` keeps the 20-request
> sweep as the evidence. Write-up:
> `experiments/results/rngd_perpe_accuracy_domain.md`.
>
> **What this still does not settle** is the part below about the two points
> `rngd_card_edf.yaml` discards at "40 % below the hardware". Those are on the
> CARD fixture and were not re-run. The artifact is now shown to be large
> enough to explain them, which removes the last reason to doubt the
> suspicion — it does not replace the measurement.

**What the code does.** `experiments/scripts/lowload_sim_error.py` compares the
simulator against a measured envelope point by offering the same arrival rate to
both. Its `--num-reqs` defaults to **20**, while every hardware envelope point it
is compared against is **300 requests**. The script then computes each side's
served concurrency by Little's law and refuses the comparison if they differ by
more than 20 % (`comparable: false`).

**What that costs, measured.** Running the A40 points both ways, same rates, same
trace prefix, changing only the simulator's request count:

| offered | measured conc | sim conc @20 req | gap | sim conc @300 req | gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1976 rps | 4.076 | 3.74 | **−8.2 %** | 4.043 | −0.8 % |
| 0.4818 rps | 11.272 | 7.70 | **−31.7 %** | 10.800 | −4.2 % |

The second 20-request point fails the script's own ±20 % guard and would have
been dropped. Nothing physical differs between the two columns.

**Why.** `served = Σ latency / wall`, and `wall` spans the first arrival to the
last completion — so it includes a drain tail of roughly one request latency
after arrivals stop. Over a short run that tail is a large fraction of the wall
and depresses the answer by about `span/(span+latency)`. At 0.4818 rps, 20
requests span 41.5 s against a ~23 s mean latency:
`11.27 × 41.5/(41.5+23) ≈ 7.2`, against the 7.70 observed. The effect vanishes as
the run lengthens; at 300 requests the span is 15× longer and the residual gap is
the real throughput error.

**Adopted.** The A40 domain's two new points are measured with `--num-reqs 300`,
matching the hardware side. The default is left at 20 rather than changed,
because changing it would silently alter what a bare re-run of the committed
RNGD invocation means; the A40 command line in the script's docstring passes it
explicitly.

**What is NOT resolved, and it touches a committed artifact.** The four RNGD
low-load points in `profiles/calibration/rngd_card_edf.yaml` were taken with the
20-request default — their notes say so ("20-req sim at 0.0967 rps vs measured
c1.0"). Their sim/measured concurrency gaps are +9 %, +10 %, −1.5 % and −19 %,
and that file additionally **discards two further points** at a gap it describes
as "40 % below the hardware", attributing it to throughput error:

> "At offered rates matching the measured c15.3 and c15.59 the simulator settled
>  at served concurrency 9.1 and 9.3 — 40 % below the hardware — so its TPOT is a
>  different operating point's TPOT and the pair is not comparable. That
>  divergence is itself the throughput error the 76 point records."

A −40 % gap at the top of a 20-request sweep is also the exact signature of the
tail artifact above, which at −31.7 % on the A40 was entirely an artifact. **This
does not establish that the RNGD attribution is wrong** — RNGD genuinely does
have throughput error at high concurrency, and the two causes are additive — but
it does mean the split between them has not been measured, and two points may
have been discarded for the wrong reason.

Testing it needs no NPU hardware: the RNGD envelope is committed, so re-running
`lowload_sim_error.py` at `--num-reqs 300` against it is pure simulation and runs
on any node. It is not done here because it would change the RNGD accuracy
domain, and therefore E6's other half, beyond the scope of the A40 work order
this deviation came out of. The sweep cache is domain-independent, so re-ranking
E6 afterwards is a replay rather than a re-simulation.

**Where.** `experiments/scripts/lowload_sim_error.py`,
`profiles/calibration/a40.accuracy.yaml` (`request_count_note`),
`profiles/calibration/rngd_card_edf.yaml` (unchanged, flagged here),
`experiments/results/a40_lowload_envelope.md`.


## D70 — the accuracy-domain margin was `-e`, but the error's denominator is the measurement · Resolved 2026-09-14

*`WORK_ORDER_uq_stage_b_plus.md` STEP C0. First entry from that work order's
D70-D79 block.*

**The convention.** Every committed domain declares `e = (sim - measured) /
measured`, with negative meaning the simulator is optimistic. So
`measured = sim / (1 + e)`, and the multiplier that recovers the measurement from
a prediction is `1 / (1 + e)` -- a margin of **`-e / (1 + e)`**.

**What the code did.** `AccuracyDomain.margin_from_error` returned `max(0, -e)`,
applying a measurement-denominator error as a multiplier on a
prediction-denominator quantity. It under-corrects, and the shortfall grows with
the error:

| domain error | margin applied | margin needed | short by |
| ---: | ---: | ---: | ---: |
| -1.42 % (A40 @170.56) | 1.42 % | 1.44 % | 0.02 pp |
| -3.05 % (RNGD-CARD @16.5) | 3.05 % | 3.14 % | 0.10 pp |
| -11.68 % (RNGD-CARD @54.2) | 11.68 % | 13.22 % | 1.54 pp |
| **-18.03 % (RNGD-CARD @76)** | **18.03 %** | **21.99 %** | **3.96 pp** |
| -42.12 % (RNGD @139.4, extrapolated) | 42.12 % | 72.78 % | 30.66 pp |

The c76 point is the one that can be checked against a measurement:
`rngd_concurrency_envelope.md` records **52.7 ms measured against 43.2 simulated**.
`43.2 x 1.1803 = 50.99` -- still 1.7 ms short. `43.2 x 1.2199 = 52.70`, exactly.

**`48.41 x 1.18 = 57.1` is the same mistake, and it is in seven documents.**
57.1 ms is **not a measurement**: every occurrence of it is that one expression.
The only measurement behind the 18 % is the 52.7/43.2 pair above. Under the
corrected formula D22's winner is robust-**59.04 ms**, not 57.1. *The verdict is
unchanged* -- both breach the 50 ms SLO -- which is why the number propagated
unchallenged through `PROJECT_REPORT.md`, `rps_aware_planning_design.md`,
`uncertainty_planner.md`, `patent_future_ideas.md`, `e5_self_rejection.md`,
`rngd_card_edf.yaml` and this file.

**What the work order got wrong, and it is worth recording.**
`WORK_ORDER_uq_stage_b_plus.md` STEP C0 asks for the c76 point's
`tpot_err_pct: -18.0` to be **changed to -15.2**, on the reading that -18.0 is
D22's `(57.1 - 48.41)/48.41 = 17.95 %` with the sign flipped. It is not. -18.0 is
`(43.2 - 52.7)/52.7`, computed from the envelope's own measurement and correct
under the file's stated convention. The inconsistency the work order correctly
sensed is in the *multiplication*, not in the *stored datum*. **That instruction
was not carried out**; `CLAUDE.md` §*When spec and reality diverge* applies.

**Blast radius, measured rather than assumed.** All sixteen E6 cells were
re-scored under the corrected margin from their recorded operating points:
**zero verdicts change.** Margins rise (the largest, 42.12 -> 72.78, is on a cell
already rejected), and a margin that rises can only remove candidates from the
feasible set -- with every recommended plan surviving, the ranking cannot move.
The default `plan` path applies no automatic margin at all, so golden output is
untouched.

**Where.** `planner/predictor/calibration.py` (`margin_from_error`),
`tests/test_accuracy_domain.py` and eight other test modules whose pinned values
moved, `docs/deviations.md` D22.

## D40 — `EnvelopeCache` dedups candidates the simulator does not treat as equivalent · Open (measured, not fixed)

**What the key does on purpose.** `EnvelopeCache._path` hashes a placement
string, so two candidates with the same shape share one simulation. That is the
point: a homogeneous multi-island cluster should memoise rather than re-simulate
`rngd0-tp4-dp1` and `rngd1-tp4-dp1`, which are the same configuration on
interchangeable hardware.

**Where it is wrong.** The placement string records each island's
`hardware|arch|tp|pp|ep|dp` but not WHICH island got which share, so a pair that
differs only by mirroring the split collapses to one entry. Found by E-B3's
identity control — a x1.0 perturbation must reproduce the cached prediction, and
one A40 candidate deviated by 7.9e-2 while 120 RNGD runs deviated by 0.000. The
cause is not the perturbation: the x1.0 bundle copy is numerically exact, the
simulator is deterministic across two fresh runs to four decimals, and the two
mirrors genuinely differ —

```
mix(a40a-tp2-dp2 + a40b-tp2-dp1)-s128-t2048   p99 TTFT 518.95 ms / 108 750 J
mix(a40a-tp2-dp1 + a40b-tp2-dp2)-s128-t2048   p99 TTFT 563.28 ms / 108 930 J
```

— and the cache serves the second's value to both. Since `a40a` and `a40b` are
the same size, symmetry says these should be equal; they are 8.5 % apart on
TTFT, so **either the key is too coarse or the simulator is order-sensitive for
a logically identical configuration.** Which of the two has not been
established, and the difference matters: one is a cache bug, the other is a
simulator property the cache merely hides.

**How much is grouped**, reproduced on current `main`:

| fixture | candidates | distinct keys | in a shared group |
| --- | ---: | ---: | ---: |
| `pd-rngd-gpu`, no P/D | 324 | 180 | 264 |
| `pd-rngd-gpu`, P/D | 612 | 294 | — (234 shared groups, 114 with a P/D member) |
| `pd-rngd-gpu-card`, P/D | 528 | 246 | — (198 shared groups, 84 with a P/D member) |

Not every group is suspect. The 4-way cross-vendor groups
(`a40{a,b} P + rngd{0,1} D`) choose between genuinely interchangeable islands and
are sound dedup; so are the `rngd0`/`rngd1` aggregated pairs. The unsound shape
is the asymmetric split above.

**What still stands.** The E-A1 headline winner re-simulates to its cached entry
exactly (`cuda-a40-node_a40a-tp4-dp1-s256-t8192` → 2972.0628 ms / 49.3956 ms /
64 430 J) and its `a40b` twin is byte-identical, so dedup is sound for
single-island aggregated candidates. Every E6 winner in PRs #75 and #77 is
either single-island aggregated or a choice among interchangeable islands. What
is NOT established is how far the mix-candidate rows in those corpora move.

**Why it is not fixed here.** Adding the island assignment to the key
invalidates every committed corpus — the E-A1 cache, `outputs/e6*/`,
`outputs/.hp-*` — and therefore every result replayed from them, including D32's
card re-check and the surrogate regret tables. That is a decision about what to
re-run, not a patch. Recorded so the next person does not rediscover it from a
7.9e-2 residual.

**What it costs a ΔR: nothing, measured 2026-09-15** (STEP C4,
`experiments/uncertainty/results/eb3_f2.md`). `profile:cuda-a40-node_a40a` was
resimulated on F2 twice, with the mirrors excluded (223 candidates, 422 runs) and
with them present (460 candidates, 676 runs). The ΔR is **bit-identical** —
21,615.372406 J both times — so D40 contributes **0 pp** of that item's 52.9 %
magnitude error.

The defect is not absent from the second run; it is loud. The ×1.0 identity
control fails on **69** candidates there, worst at
`mix(cuda-a40-node_a40a-tp2-dp2+cuda-a40-node_a40b-tp2-dp1)-s128-t2048` on
`p99_ttft_ms` by **7.869e-02** — the same candidate and magnitude E-A1 reported,
so it reproduces on demand. It simply does not reach a ΔR, because ΔR is a regret
integral over the *recommendation* and the placements D40 corrupts are far from
the argmax: F2's winner is a P/D split across the two A40 nodes, and the mirrored
`mix(...)` candidates lose by a wide margin whichever prediction they read.

**This narrows the entry, it does not close it.** 0 pp is a statement about this
item on this fixture. A fixture whose recommendation *is* one of the mirrored
placements would be a different measurement, and the cache key is still what
should be fixed. What it does settle is PR #86's open worry that E-B3's magnitude
error was partly D40's: on the one fixture where both halves were run, it is not.

**Where.** `planner/envelope.py` (`_path`, `key_for`),
`experiments/uncertainty/eb3_closed_form_vs_resim.py` (the identity control that
found it, and `--include-mirrors` / `--expect-mirrors`, which make the
before-exclusion half runnable: with the mirrors back in the corpus the control
fails by design, so the expected set has to be declared rather than inferred).

## D33 — two accuracy-domain implementations, one kept; `refuse` becomes the default · Decided 2026-09-11

**What happened.** `WORK_ORDER_rps_aware.md` STEP 4 and
`WORK_ORDER_uncertainty_planner.md` STEP A2 both forked from `108e48a`
(2026-09-09) and each built "the simulator's error as a function of the
operating point" — 45 minutes apart on 2026-09-10 (uq-a2 `a650e33` 11:45, rps
PR #72 merged 12:30). By the time the uncertainty stack (`feat/uq-a1` … `b3`,
18 commits, never a PR) was noticed, `main` carried seven PRs on top of the rps
design and the two disagreed on ten files. The disagreement was structural, not
textual:

| | `main` (rps STEP 4, #72–#77) | uncertainty stack (A2–A5) |
| --- | --- | --- |
| where the curve lives | `HardwareCalibration.accuracy_domain`, one per hardware | `BucketError.operating_points`, one per (hardware, workload bucket) |
| class | `calibration.AccuracyDomain` / `AccuracyPoint` | `predictor.accuracy_domain.AccuracyDomain` wrapping `OperatingPoint` / `ConcurrencyDomain` |
| error convention | `(sim − measured) / measured × 100`, negative = optimistic | `(real − sim) / sim` fraction, positive = optimistic |
| outside the points | `widen_error_bars` (default): nearest error + \|slope\| × distance, uncapped | `None` → `UNMEASURED` rejection, always |
| consumer | `exhaustive._auto_margins`, `max(manual, auto)` | `optimizer.margin.MarginPolicy`, per candidate, `unmeasured` verdict |
| rejection stage | `OUTSIDE_CALIBRATION_DOMAIN` (slab3d), `OUTSIDE_MEASURED_ENVELOPE` | `UNMEASURED` |
| plan record | `operating_point: [OperatingPointRecord]`, `margin_source` | `margin_basis` |
| RNGD-CARD data | nine D32 points, served 1.02–76.0, matched-rate 300-request pairs | four E-A2 points, served 15.3–107.2, requested-concurrency pairs |

**Decision** (user, 2026-09-11): `main`'s accuracy domain is the one that lands;
the uncertainty stack is rebuilt on it; `refuse` is the default. Concretely:

1. **The curve is `calibration.AccuracyDomain`.** It gains `errors_at` (None
   when refused), `basis_at`, `has_metric`, an optional `workload_shape` scope
   (uq §2.4.2: a domain is matched on token mix, never on arrival rate) and
   `arrival_process` (D19: a closed-loop measurement carries no transferable
   TTFT). `predictor/accuracy_domain.py` keeps only the served-concurrency
   functions. One sign convention, `AccuracyPoint`'s.
2. **`outside_domain` defaults to `refuse`.** Uncertainty rule A2 and
   `docs/rps_aware_planning_design.md` §5 (line 103, "`refuse` is the default")
   both say so; only the committed yaml said `widen_error_bars`. The three
   committed domains (`rngd_card_edf.yaml`, `a40.accuracy.yaml`,
   `rngd_perpe.yaml`) keep their explicit `widen_error_bars`, so every E5/E6
   number is unchanged and its regression tests pass untouched. Flipping them
   to `refuse` is a science change — three of E6's sixteen cells rest on an
   extrapolated margin (HANDOVER §2.1) and would become
   `outside_calibration_domain` — and is left to the rps owners as a deliberate
   act, not done here.
3. **The margin layer is the uncertainty stack's `MarginPolicy`**, replacing
   `_auto_margins`. `AccuracyDomainMargin` reads `SimResult.operating_point`
   per hardware and P/D phase (prefill owns TTFT, decode owns TPOT, total both —
   rps STEP 4.3 unchanged), takes the worst, keeps the manual floors and the
   `margin_source` rule, and returns `unmeasured` for a refused point, a
   foreign token mix, or hardware with neither a domain nor a fitted bucket.
   Hardware with a fitted bucket but no domain falls back to the bucket's
   scalar error and says so (`status: scalar`).
4. **`UNMEASURED` is absorbed into `OUTSIDE_CALIBRATION_DOMAIN`.** A slab3d
   refusal (D28/D31) and an accuracy-domain refusal are the same kind of thing —
   a value nobody measured — and one epistemic bucket is easier to read than
   two. The reason string is prefixed `unmeasured:`; `SearchResult.unmeasured`
   and the "measure the envelope at c>=…" suggestion survive.
5. **Partial coverage is a caveat, not a rejection.** The uncertainty stack
   rejected a pass that rested on a metric with no measured margin. Every
   committed RNGD domain is TPOT-only by construction (D19), so that rule would
   have emptied every search on the committed data. A pass with an unmargined
   metric is kept and the search says, once, which check ran unmargined.
6. **`--accuracy-domain` takes zero or more files.** Bare, every domain under
   `profiles/calibration/` (rps); with files, exactly those (uq). Two files
   carrying a domain for one hardware are refused, never last-wins. A manual
   `--tpot-margin-percent` is a floor under it (rps), not an error (uq).
7. **E-A2's `rngd_card_edf.domain.yaml` is dropped.** It paired each real run
   with a simulation at the same *requested* concurrency; the sim side's own
   served concurrency, recorded in its notes, is 71–189 against real 15–107 —
   the D32 mis-pairing. `main`'s nine-point D32 domain supersedes it;
   `experiments/uncertainty/results/ea2_rngd_domain.md` says so at the top.
   E-A1 was re-run from its committed cache (2592 hits, 0 misses) against the
   committed domains, with a fourth condition showing `refuse`:
   `experiments/uncertainty/results/ea1_margin_modes.md`.

**Two things this leaves open, on purpose.**

- `AccuracyDomain.margin_from_error` charges `−err_pct` as the inflation, i.e.
  `sim × (1 + 0.18)` for an 18 % optimistic simulator. The exact correction is
  `sim / (1 − 0.18) = sim × 1.2195`; the uncertainty stack's convention gave
  that (its commit `a650e33` notes "+0.22 in this code's convention, not
  −0.18"). The linear form under-corrects by `e²/(1−e)` — 4 pp at 18 % — and
  E5's 56.97 ms depends on it. Not changed here; a change is a re-derivation of
  E5/E6 and belongs to the rps owners.
- `main`'s committed domains are *unscoped* (`workload_shape` empty) because
  they were measured on the fixture workload E5/E6 use. Scoping them is a
  measurement claim about which token mixes they transfer to, not a code
  change, and is not made here.

**Stage B (B1–B3), landed the same day on the same terms.** The perturbation
engine, sensitivity sweep and measurement plan (`planner/uncertainty/{perturb,
sensitivity,measurement_plan}.py`, `plan --measurement-plan`, `measure-apply`)
came over intact; four of the uncertainty stack's B-stage decisions were adapted:

8. **The "rev 2" domain key (`model|variant|in_*|out_*`) is not adopted as a
   key.** A domain is one per hardware (main's structure); the verification
   conditions it was measured under are the optional scope fields
   `AccuracyDomain.workload_shape` / `model` / `variant`, and the margin policy
   refuses a domain whose set fields differ from the service's. Scalar
   `BucketError` entries keep the canonical envelope key
   (`in_*-out_*-rps_*`) — `apply_robust_margins` on `main` looks them up by it,
   and re-keying would have made that lookup a silent `(0, 0)`.
9. **Served concurrency is per instance**, as the stack had it (`served_
   concurrency` = the busiest instance, `served_concurrency_per_island`,
   `instance_island_ids`). The margin still reads `SimResult.operating_point`,
   which `planner/util/operating_point.py` already computes per hardware and
   per P/D phase from the same per-instance attribution.
10. **The metrics-schema digest lives in the cache payload, not the file
    name.** The stack folded it into the entry name so a schema change misses;
    that would have renamed every entry of `main`'s committed replay caches
    (`outputs/perf/topk/cache_*`, E6). Stored inside the entry instead: a
    mismatch is a miss, a missing digest is served with a warning. E-A1's 162
    B-stage entries were re-keyed to the repository naming and stamped.
11. **`measure-apply --input sim_error:…` adds an `AccuracyPoint`** to the
    hardware's `accuracy_domain` (created under `refuse` if none), with
    `--value` the signed error in `AccuracyPoint`'s convention.
12. **`judge()` keeps two branches, not three.** The stack's second branch
    (pass on an unmeasured metric → unmeasured) is item 5 above; the verdict is
    kept and the search carries the gap as a caveat.

E-A1 was re-run a third time, off the per-island cache: 86 of the 114 RNGD
candidates now sit *inside* the RNGD-CARD domain and are rejected on a
measured 3.2–17.6 % margin — D22's two-card winner among them, at per-card
served 71.6–74.7 and 56.9 ms robust TPOT, which is E5's verdict reached from
inside the fixture. `experiments/uncertainty/results/ea1_margin_modes.md`.

**Where.** `planner/predictor/calibration.py`, `planner/optimizer/margin.py`,
`planner/optimizer/exhaustive.py`, `planner/__main__.py`,
`tests/test_margin_policy.py`, `tests/test_cli_accuracy_domain.py`,
`tests/test_accuracy_domain.py`, `planner/uncertainty/`, `tests/test_{perturb,sensitivity,measurement_plan,instance_attribution}.py`; PR stack `feat/uq-a1-registry` →
`feat/uq-a5-ea2-rngd-domain` → `feat/uq-b3-measurement-plan`.

---

## D34 — which perturbation rules are approximate, and the one whose clause was wrong · Resolved 2026-09-14

`WORK_ORDER_uncertainty_planner.md` STEP B5 asks for "근사 규칙이 근사인 kind
목록" — the register of which closed-form rules are exact and which are
first-order stand-ins. Writing it down turned one of them into a finding.

**The register.** `planner/uncertainty/perturb.py`, `PerturbResult.approximation`:

| kind | `approximation` | why |
| --- | --- | --- |
| `SIM_ERROR` | `False` | it moves the **margin**, not the prediction. No simulator input changes, so there is nothing to approximate |
| `LINK_BW` | `False` | the envelope cache stores the raw simulator output from *before* `apply_pd_transfer_cost`; the planner adds the transfer term itself, so re-pricing it with the same `kv_transfer.transfer_ms` is the same arithmetic |
| `LINK_LAT` | `False` | same path |
| `PROFILE` | `True` | latency is not linear in operator time once a queue forms |
| `POWER` | `True` | a constant factor on average power |

Items scored by an approximate rule are printed as "(approx)" wherever the
measurement plan quotes them, so an operator never spends a server-hour on a
figure without being told how it was obtained.

**The finding: `PROFILE`'s energy clause is wrong against this simulator.** Work
order §2.7 specifies, and `_scale_profile` implements, "에너지는 불변(전력 모델
별도)" — a profile perturbation scales latencies and rates and leaves energy
alone, on the reasoning that energy is `POWER`'s to move and counting it twice
would let one measurement appear to settle two.

The simulator disagrees. `serving/core/power_model.py:73`:

```python
def add_npu_active_energy_consumption(self, hardware, node_id, latency_ns, num_npus=1):
    latency_s = latency_ns * 1e-9
    energy_j = (active_power - idle_power) * latency_s
```

Active energy is **linear in operator latency**. Scaling a perf bundle's
`time_us` by `(1+δ)` therefore scales the active-energy term by `(1+δ)`, and the
closed form reports no change at all. Measured on the pre-D33 stack by copying
`profiler/perf/A40` with every `time_us` multiplied and re-simulating one
candidate: multiplier **1.38876**, total energy **162 260 J → 225 190 J**, ratio
**1.3878**. The small shortfall against the multiplier is the idle/standby, DRAM
and link terms, which do not scale with operator time.

**The rationale was wrong, not just the number.** §2.7 justified the clause by
double-counting: energy is POWER's, so moving it under PROFILE too would let one
measurement appear to settle two uncertainties. That is not what happens. Energy
is **watts × seconds**, and the two inputs move different factors of that
product — a PROFILE error is an error in per-operator **time**, a POWER error is
an error in **watts**. They are independent, and applying both composes:

```
PROFILE(δp) then POWER(δw)  →  energy × (1+δp)(1+δw)
```

exactly once, in either order. Double-counting would be one rule claiming the
other's factor, which is what *not* moving energy under PROFILE was protecting
against — by giving up the factor entirely rather than splitting it.

**What it cost, measured.** `ΔR` for a PROFILE item, before and after:

| case | before | after |
| --- | ---: | ---: |
| two candidates inside every SLO, differing only in energy (`tests/test_sensitivity.py`) | **0.0000**, `flip=False` | **375.0000**, `flip=True` |
| `profile:furiosa-rngd-card-node_rngd{0,1}` on the E-A1 corpus, Tier 0 multiplier 1.38876, `GlobalMargin`, 324 cache hits / 0 misses | 32 702.6183 | 35 880.9703 |
| `profile:cuda-a40-node_a40{a,b}`, same run | 0.0000 | 0.0000 |

The first row is the defect in isolation: two plans that both sit an order of
magnitude inside the SLOs and differ only in the ranked objective. Before the
fix, measuring the incumbent's profile was worth *exactly nothing* — the grid
could not move the ranking at all, because the only quantity being ranked was the
one the rule held fixed. After it, B's energy passes A's at ×2.5 and the item is
worth 375 J of avoided regret, 178.6 per server-hour.

The second row is the effect where an SLO boundary is *also* in play: the RNGD
items already earned regret by crossing one, and the energy channel adds 9.7 % on
top. The third row is 0 both ways and correctly so — the E-A1 incumbent is
RNGD-only, so perturbing an A40 profile can only make candidates that already
lose lose harder. On this fixture no PROFILE item goes 0 → positive through
energy alone, because the incumbent crosses its TPOT SLO (48.41 ms against 50,
×1.033) before it loses its energy lead (60 350 J against 64 430, ×1.068). The
defect is real regardless; this corpus just happens to hide it behind a tighter
constraint.

**The fix.** `_scale_profile` now scales `total_energy_j` by the same multiplier
as the latencies and recomputes `tokens_per_joule` from the new energy —
recomputed, not scaled, so it stays consistent with `completed_tokens`, which a
profile perturbation does not change. `average_power_w` and `peak_power_w` are
deliberately left alone: a slower engine draws the same power for longer, and
watts are POWER's axis. A candidate with no simulated energy (D2/D14) is left
alone rather than given an invented one.

Three tests hold it, and all three fail against the previous rule:
`test_a_slower_profile_burns_proportionally_more_energy`,
`test_profile_and_power_compose_without_double_counting` (which checks the
magnitude against the product, not merely order-independence — two rules that
both claimed the same factor would still commute), and
`test_a_profile_earns_regret_through_energy_alone`, the decision-level one.

**Still approximate.** The energy term inherits `_scale_profile`'s first-order
character and adds the idle/standby offset: the measured ×1.3878 against a
×1.38876 multiplier is the DRAM, link and standby terms not scaling with operator
time. `approximation` stays `True`.

**What is left to the work order's owner.** The one-line §2.7 amendment in
`WORK_ORDER_uncertainty_planner.md` — the table there still says 에너지는 불변.
The code, the tests and this entry are ahead of it deliberately, because STEP B4
must not produce numbers under the old rule.

**Corroboration on an unlanded branch.** The ×1.38876 → 162 260 / 225 190 J
resimulation was taken on `feat/uq-b4-wip`, which does not run against `main`
(D35, `docs/uncertainty_planner.md` §6). The `power_model.py` code path — the
load-bearing half — is on `main` and can be read directly, and the ΔR table above
is reproducible on `main`.

**Where.** `planner/uncertainty/perturb.py` (`_scale_profile`),
`tests/test_perturb.py`, `tests/test_sensitivity.py`,
`serving/core/power_model.py:73` (read, not modified),
`docs/uncertainty_planner.md` §2.4 and §5.

---

## D35 — the uncertainty planner is not documented in `README.md` or `CHANGELOG.md` · Recorded 2026-09-14

`WORK_ORDER_uncertainty_planner.md` STEP B5 lists four documentation
deliverables, two of which name files this fork does not own.

**What the work order asks for.** "`README.md`에 `--accuracy-domain`,
`--measurement-plan` 한 단락" and "`CHANGELOG.md`".

**What those files actually are.** Both are pure upstream LLMServingSim files
with no HeteroPilot content whatsoever:

* `README.md` is the simulator's front door — LLMServingSim branding, a
  `git clone https://github.com/casys-kaist/LLMServingSim.git` quickstart, and
  links to llmservingsim.ai. Upstream's own "README and docs split" policy (see
  its `[Unreleased]` changelog entry) makes it deliberately minimal and moves
  CLI flag tables to the docs site. Its last three commits are upstream's.
* `CHANGELOG.md` follows Keep a Changelog for *the simulator*, and its last
  three commits are upstream release commits. It contains no `planner/` entry.

Adding a HeteroPilot planner flag to either would put fork content into an
upstream file, contradict upstream's stated README policy, and guarantee a
conflict at the next re-pin. `CLAUDE.md` §*When spec and reality diverge* applies:
the real code wins, and the difference is recorded here.

**What was done instead.** The paragraph the work order asks for lives in
`CLAUDE.md`'s *Commands* section, beside the `python -m planner` block it already
documents, and the long form is `docs/uncertainty_planner.md` — which is where
this fork's own documents live (`deviations.md`, `CLAIMS.md`, `HANDOVER.md`,
`rps_aware_planning_design.md` are all top-level `docs/*.md`; `docs/docs/` and
`docs/README.md` are the upstream Docusaurus site).

**This generalises.** The same rule applies to any future work order that asks
for a `README.md` or `CHANGELOG.md` entry: while those files carry no fork
content, the entry goes to `CLAUDE.md` and to the relevant `docs/*.md`, and the
work order item is satisfied there. No further deviation entry is needed — cite
this one.

**When this should be revisited.** If the fork ever grows its own README, or
Phase 5 lifts the upstream-file restriction, the paragraph moves and this entry
is superseded. Until then a reader looking for the flags finds them in
`CLAUDE.md` and in `python -m planner plan --help`, both of which are accurate.

**Where.** `CLAUDE.md`, `docs/uncertainty_planner.md`; `README.md` and
`CHANGELOG.md` deliberately untouched.

## D71 — a TP=1 decode island reaches the simulator and raises; the planner cannot pre-reject it · Recorded 2026-09-14

STEP C1's F2 corpus (`WORK_ORDER_uq_stage_b_plus.md`) is the first one built with
`enable_pd=True` on this cluster, and 23 of its 246 candidates never produced
metrics. They all died the same way, in the simulator, at the moment the router
hands a finished prefill to a decode engine:

```
File "serving/core/router.py", line 348, in transfer_prefill_request
  self.decode_schedulers[instance_id].add_decode(req)
File "serving/core/scheduler.py", line 841, in add_decode
  self.memory.allocate(kv_size, Device.NPU)
RuntimeError: [MemoryModel] [node_id=1,inst=1] NPU: tried to load 92.00MB
              but only 25.49MB is available.
```

**The discriminator is TP=1 on the decode side, not TP=1.** All 23 have
`tp1-dp1` in both roles. Candidates that keep `tp1` for *prefill* and give decode
more — `pd(furiosa-rngd-card-node_rngd0-tp1-dp1 P + cuda-a40-node_a40a-tp2-dp1 D)`
— simulate normally, as do aggregated `mix(...tp1-dp1 + ...tp1-dp1)` placements
at the same total device count. A P/D decode engine holds the KV of every request
in flight, while an aggregated pair of the same two devices splits that working
set across both; one card's worth of KV is enough for the second arrangement and
not for the first.

**This is not a planner bug, and "fix the pruning" is the wrong fix.** Stage 2 is
`planner/util/memory.py::feasible` with its default `min_kv_tokens=1`, and the
generator calls it role-agnostically on purpose (`_parallelism_options`'s
docstring). Raising the threshold to a working-set estimate would make stage 2
reject candidates that `§5.6` declares nothing about — precisely the
*relaxation, never an extra condition* invariant in `CLAUDE.md`, and precisely
the mistake the throughput bound made before the oracle-agreement test caught it.
A stage may reject only when the most optimistic arithmetic already misses a
declared constraint, and "the decode engine will fill up at this concurrency" is
not one.

**What the simulator is missing is admission control**, not capacity.
`add_decode` allocates unconditionally instead of leaving the request queued
until KV frees, so an over-subscribed decode instance raises where a real engine
would simply run at a worse TPOT — which the planner would then have judged
against the SLO in the ordinary way. Changing that is an edit to upstream
`serving/core/scheduler.py`, which needs a work order that names the file
(absolute rule 1); D12 is the standing warning about guessing in this file.

**How it is handled meanwhile.** The runs are counted `sim_error` — no verdict,
not a rejection — and `experiments/uncertainty/results/f2_truth.md` lists them
under *Runs that did not finish* with the cause and the shared shape, so a reader
of the F2 truth is not left thinking the planner judged that sub-family. The cost
is 23 wasted simulator launches per cold corpus build; with the cache warm it is
paid once.

**Where.** `experiments/uncertainty/build_truth_cache.py`,
`experiments/uncertainty/results/f2_truth.md`; the D-number comes from the
`D70–D79` block `WORK_ORDER_uq_stage_b_plus.md` claims in STEP C0 (PR #87).

## D72 — a `sim_error` item is swept as a flat margin and restored as a domain swap, so E-B2's second criterion is not defined for it · Recorded 2026-09-15

`WORK_ORDER_uq_stage_b_plus.md` §2.4 asks E-B2 to report its precision and recall
two ways: (1) against the **realised** transition — restore an input to the one
value truth turned out to have, and see whether the recommendation moved — and
(2) against the **possible** transition, restoring it to every grid point of its
range instead. (2) is meant to be the detector's own accuracy, with (1) reading
partly as a statement about where this fixture's truth happened to sit.

`--truth-sweep` implemented (2) and then scored **identically to (1) on all 1,848
E-A1 cases**. That is not the two criteria agreeing. It is one criterion computed
twice, for the only kind of input the E-A1 corpus has false positives in.

**The two paths compute different functions of the same item.** For
`sim_error:domain`:

| | what a point of the range means |
| --- | --- |
| sweep (`analyze` → `perturb` → `_plans_at`) | a **flat one-sided margin** of `v × 100 %` on every candidate whose decision is not `unmeasured`. Its truth value, `v = 0`, is therefore *no margin at all* |
| restore (`_state`, and `realisable_flip` as first written) | swap `policy_truth` for `policy_scalar` — the **measured accuracy domain** against that domain collapsed to its fitted point. Truth is the domain, whose margin is +11.6 % at served concurrency 3.9 and −18 % at 76 (D29), and is nowhere zero |

The item's `range` is `[0, 0.4725]` in simulator-error fractions
(`WORST_MEASURED_ERROR`), but the restore has only two states to offer, so the
first implementation mapped the range onto them by a midpoint test. Every grid
point below the midpoint became `policy_truth` and every point above it
`policy_scalar` — which are exactly the two states criterion (1) already
compares. Criterion (2) could not return a different answer, and did not.

**This is also what the `delta` category is detecting**, and the two were found
together. On the E-A1 corpus all 19 false positives are `sim_error:domain` with
`approximation=False` — an *exact* rule — and the sweep grid puts the crossing at
value 0, which *is* the truth value, while restoring to that same value produces
no flip. Two computations of one point cannot disagree unless they are computing
different functions. They are: 0 means "no margin" to one and "the measured
domain" to the other. Read as `beta`, that would have accused the closed-form
rules of misplacing one crossing in six; separated, `beta` is 0, which is what
§2.4 predicted for it.

**Decision: scope, do not collapse.** `RESTORE_MATCHES_SWEEP` in
`experiments/uncertainty/eb2_flip_detection.py` lists the kinds whose restore
path and sweep path are both `perturb` — `profile`, `link_bw`, `link_lat`,
`power` — and `sim_error` is not among them. `realisable_flip` returns `None`
there rather than a number, `score(against="realisable")` drops those rows
instead of counting them as true negatives, and `realisable_by_kind` in the
payload shows per kind how many rows the figure actually covers. The consequence
has to travel with the number: **on the E-A1 corpus the realisable score
describes the link and profile rules and nothing else**, because every false
positive it has is scoped out.

**What was not done.** Making the restore walk the sweep's flat-margin model
would have produced a number for `sim_error`, but that number answers "is the
detector self-consistent with its own model of the input", not "what happens when
you measure the domain" — and measuring an accuracy domain yields a domain, not a
scalar error. Making the *sweep* walk the restore's model is the honest repair,
and it is out of C3's scope: `_state`'s SIM_ERROR degradation is what E-B1 was
built on, so changing it invalidates the results C2 committed
(`eb1_regret_vs_budget.md`, `eb1_f2.md`). Either repair belongs in its own work
order, and until one runs, **criterion (2) must not be quoted as a precision
figure for `sim_error`.**

**A smaller consequence worth recording.** `perturb` returns
`approximation=False` for a SIM_ERROR item, and D34 checked that claim against
the margin arithmetic, which is exact. It is not a claim that a flat scalar is a
faithful stand-in for refitting a concurrency-dependent domain — the same
distinction D29 draws when it rejects one error number for every candidate. The
flag is about the rule, not about the modelling choice above it.

**Where.** `experiments/uncertainty/eb2_flip_detection.py`
(`RESTORE_MATCHES_SWEEP`, `realisable_flip`, `score`, `classify_false_positive`'s
`delta`), `tests/test_uq_eb2_classification.py`,
`experiments/uncertainty/results/eb2_f2.md`; the D-number comes from the
`D70–D79` block `WORK_ORDER_uq_stage_b_plus.md` claims in STEP C0 (PR #87).

## D73 — a resimulated endpoint whose every run crashed is indistinguishable from one the input did not move · Recorded 2026-09-15

`planner/uncertainty/resimulate.py::_endpoint` evaluates the candidates an
uncertain input can touch and returns

```python
merged = dict(baseline)
merged.update(judged_metrics(evaluation))
...
return Endpoint(..., seconds=elapsed, simulated=len(touched))
```

Both halves of that are reasonable on their own and wrong together. A candidate
whose simulator run raised never reaches `judged_metrics`, so it is simply absent
from the update and **keeps the baseline's value**; and `simulated` reports the
number of runs *started*, not the number that came back. An endpoint at which
every run failed therefore returns the baseline unchanged, claims to have
simulated the whole touched set, and is byte-identical to an endpoint where the
input genuinely moved nothing.

**It fired on the first sharded run of E-B3** (STEP C4, 2026-09-15). Two helper
nodes were given a copy of the repository whose rsync excluded
`astra-sim/inputs` — 14 GB, believed to be generated per-run artifacts. It is
almost all generated, but it also holds the ASTRA-Sim config templates, so every
run on those nodes raised `FileNotFoundError: ASTRA-Sim system config template
'.../astra-sim/inputs/system/system.json' not found` before the simulator
started. The harness reported:

```
link_bw:fabric-rngd0-a40b: closed=0 resim=0 in 10s (122 runs)
link exactness link_bw:fabric-rngd0-a40b: ... rel=0.000e+00 -> agrees
```

A clean pass, an exact-rule agreement, and a `resim` that matched the closed form
to the digit — produced entirely by failure. **Link items are the worst case for
this**, because their true ΔR on this fixture really is 0: total failure and the
correct answer are the same number. The only visible symptom was that 122
simulations had taken ten seconds.

**Why `simulated` cannot be fixed into the answer.** Counting successes there
would be a `planner/` change that C4's A4 does not allow, and it would still not
distinguish the two cases for a caller that only reads ΔR. The discriminator
belongs where the comparison is made.

**What was done instead.** `endpoint_coverage()` in
`experiments/uncertainty/eb3_closed_form_vs_resim.py` counts, per endpoint, how
many touched candidates actually came back, using **object identity** against the
baseline: a candidate that was really re-simulated gets a fresh
`PredictedMetrics` out of `judged_metrics` even when its numbers are unchanged,
while a crashed one still holds the very object the baseline had. Equality would
be the opposite error — it would call a genuinely inert candidate a crash, and
inert candidates are exactly what the link items are there to confirm.

`coverage_gate()` then fails the run when any endpoint has **zero** successful
runs: the artifact is written, `STOP` names the endpoints, and the exit code is 4.
Partial failure is recorded and allowed through — D71 already puts 23 real holes
in this corpus, and a corpus with holes is not the same as no corpus at all. The
same gate runs over the union of shards in `--merge`, so a node with a broken
install cannot be averaged in by the nodes that worked.

**The near miss is the point.** Nothing in the result would have looked wrong.
Four link items would have entered `eb3_f2.md` as confirmed zeros with an exact-
rule agreement beside them, and the only evidence to the contrary was a timing
line no reader has reason to check.

**Where.** `experiments/uncertainty/eb3_closed_form_vs_resim.py`
(`endpoint_coverage`, `coverage_gate`, and the coverage union in
`merge_shards`), `tests/test_uq_eb3_gates.py`,
`experiments/uncertainty/results/eb3_f2.md`; the D-number comes from the
`D70–D79` block `WORK_ORDER_uq_stage_b_plus.md` claims in STEP C0 (PR #87).

## D74 — F2, the fixture built to make more than one input kind matter, and the three choices behind it · Recorded 2026-09-16

`WORK_ORDER_uq_stage_b_plus.md` exists because E-A1's corpus was degenerate for
the question Stage B asks. Its truth cache was built with `enable_pd=False`, so
it enumerates no P/D candidate; with no candidate whose prediction crosses a
fabric link, `perturb` reports `affected=0` for all six `link_bw` items and no
measurement strategy can be right or wrong about a link. Of the eleven registry
items, exactly one — `sim_error:domain` — carried essentially all the decision
regret, and any rule that ranks the cheapest item first ties the oracle. E-B1's
"`ours` = oracle at every budget" was therefore evidence about the fixture.

F2 is the replacement corpus. Three choices in it are not obvious and are
recorded here rather than inside a result document.

**1. The TTFT SLO is 8,000 ms, and the work order's 4,000 is wrong.** §2.1 asks
for 4,000 ms, citing a three-regime table. That table is in
`pd_slo_sweep.md`, which is **superseded**; `docs/d23_revalidation.md` is the
current one and its middle regime — where the KV transfer across a link can
actually change a P/D candidate's TTFT verdict — is at **8,000 ms**. The fixture
uses 8,000 and `experiments/uncertainty/fixtures/f2.json` records why in its
`note`. At E-A1's 25,000 ms the transfer never enters a verdict at all.

Everything else in the service spec is `examples/service_specs/llama31-8b.yaml`
unchanged, and the TTFT SLO enters neither the `EnvelopeKey` nor the trace
digest — which is what makes E-A1's 162 aggregated simulations reusable rather
than re-run.

**2. Degradation sets are stratified by kind, and the seed is recorded.** E-B1's
uniform sampling over an 11-item pool draws mostly link-only sets, because six of
the eleven are links. A sweep whose sets are mostly one kind cannot show a
ranking rule choosing *between* kinds, which is the whole point of F2. STEP C2's
`--stratified` requires at least two kinds in every k ∈ {2,3} set (k = 1 stays
exhaustive), giving 11 / 34 / 141 sets against 11 / 55 / 165 unstratified. The
flag and the seed are in the provenance of every run that uses it. **E-B2 and
E-B3 do not stratify** — they sweep the full combination space and the full pool
respectively, so their `n` is the unstratified one.

**3. D40's mirrors are excluded at build, and that is why `enable_pd` gates it.**
`_corpus(..., exclude_mirrors=enable_pd)`: a P/D corpus contains mirror-symmetric
splits that `EnvelopeCache` collapses onto one entry (D40), so keeping both would
count one simulation twice. 282 of F2's 528 generated candidates are excluded as
mirror members, leaving 246, of which 23 more have no cache entry (D71) — the
223 the experiments use. E-A1 keeps its mirrors because `enable_pd=False` there
and the committed result must reproduce; it has mirrored *aggregated* placements
all the same, which is where the ×1.0 identity control finds D40.

**Did it work? Partly, and the shortfall is the result.** F2 has two active
kinds rather than one (`sim_error` and `profile`), so §2.3's gate is met. But
`link_bw` is still inert, now for a third distinct reason: on E-A1 it could not
matter, on F2 it is priced into 54 candidates including the winner and the
decision is insensitive to it anyway (`eb1_f2.md`), and under flip detection the
sweep finds no crossing anywhere in its range (`eb2_f2.md`). Three experiments
agree that a measured 13 GB/s link does not decide anything on this cluster at
300 requests. A fixture where it does needs a tighter TTFT or a larger KV working
set, not merely P/D candidates — and the 23 candidates D71 removed are exactly
the `tp1-dp1` P/D family where a transfer is largest relative to the engine's own
work.

**Where.** `experiments/uncertainty/fixtures/f2.json`,
`experiments/uncertainty/fixtures/llama31-8b-ttft8000.yaml`,
`experiments/uncertainty/build_truth_cache.py`,
`experiments/uncertainty/eb1_regret_vs_budget.py` (`_corpus`, `stratified_subsets`),
`experiments/uncertainty/results/{f2_truth,eb1_f2,eb2_f2,eb3_f2}.md`; the
D-number comes from the `D70–D79` block `WORK_ORDER_uq_stage_b_plus.md` claims in
STEP C0 (PR #87).

## D100 — the c76 accuracy-domain point compares a simulated p50 against a measured mean · Open (measured, not changed)

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V0. First entry from that work
order's `D100–D109` block. D70 is its sibling: same number, different defect.*

**What D70 settled and what it left.** D70 established that 57.1 ms is not a
measurement — it is `48.41 × 1.18`, one expression repeated across seven
documents — and that the margin multiplier is `-e/(1+e)`, not `-e`. It put the
52.7/43.2 pair in 57.1's place as "the only measurement behind the 18 %". **That
pair is not measured like-for-like**, and D70 did not check it.

**The two sides are different aggregations.** Recomputed from the raw records by
`experiments/p2_evidence/v0_convert.py`:

- **52.7 is a MEAN.** `rngd_concurrency_envelope.md` interpolates its `TPOT avg`
  column between c64 and c128 to eff 76. From
  `outputs/rngd_envelope/edf/real_c{64,128}.json` (n = 256, 300; eff 59.185,
  107.192) the mean interpolates to **52.7149**, reproducing the committed 52.7.
- **43.2 is a p50.** It is the card fixture's winner `s256-t8192`
  (`pd_slo_sweep_margin.md:41`), whose cached metrics reproduce three figures the
  sweep quotes — p99 TTFT 480.09 ms, goodput 5.5519 rps, and 190 920 tokens /
  54.0 s makespan / 2 cards = 1767.78 output tok/s per card. That run's TPOT is
  **p50 43.1517**, p95 47.4974, **p99 48.4097** at served concurrency 74.750. The
  predictor emits no mean at all (`llmservingsim.py:600`), so mean-against-mean
  was never available.

**Every other committed point pairs p50 against p50.**
`experiments/scripts/lowload_sim_error.py` records `"compared_metric":
"tpot_p50"` and compares `pt.tpot_p50` against `m["tpot_p50_ms"]` (lines 198,
317-318); it produced six of the nine RNGD points and two of the three A40 ones.
The exceptions are c16.6 (a bucket mean error from the EDF fit), A40 c170.56 (p95
absolute error signed by the mean diff) and c76.

**Corrected by D101:** c76 is *not* the only point pairing a simulated p50 against
a measured mean. c14.832 and c25.181 do as well, because the performance
envelope's `tpot_p50` field holds a mean for the four points measured 2026-08-31.
Every number in this entry stands; the uniqueness claim does not.

| basis at c76 | measured ref @76 | file `e` % | disclosure `r` % | robust from p99 48.41 |
| --- | ---: | ---: | ---: | ---: |
| mean — **committed** | 52.7149 | −18.14 | +22.16 | 59.14 ms |
| p50 — consistent with the other eight | 53.9748 | **−20.05** | +25.08 | **60.55 ms** |
| p99 — what the feasibility check reads | 58.5442 | −26.29 | +35.67 | 65.68 ms |

And sim **p99** against measured **p99**: `e` = **−17.31 %**, `r` = +20.93 %, and
48.41 × 1.2093 = **58.54 ms**, which reconstructs the measured p99 by
construction.

**Why it stayed invisible.** Every row breaches the 50 ms SLO, so no verdict
moves — the same reason 57.1 survived seven documents and the same reason D70's
own blast radius was zero. Three arithmetic defects in one number have now each
been found by inspection rather than by a test, because the test that would catch
them would have to compare verdicts, and the verdicts agree.

**The A40 file declares its seam; this one does not.**
`profiles/calibration/a40.accuracy.yaml` carries a `METHODOLOGY SEAM` note saying
its 170.56 point is a p95-absolute-error-signed-by-the-mean while the others are
p50 errors, and quantifies the difference at 0.1 pp. `rngd_card_edf.yaml`'s c76
note says only "interpolated c64/c128 … Not re-measured here".

**Why it is not fixed here.** Changing the stored `-18.0` invalidates the E-A1
corpus, `outputs/e6*/`, and the pinned values in `tests/test_margin_policy.py`
and `tests/test_accuracy_domain.py` — a decision about what to re-run, not a
patch, exactly as D40 reasons about the envelope cache key. STEP V0's scope is to
determine the basis. **What a follow-on must decide is p50 or p99**, and the
argument for p99 is that `feasibility.py:67` applies the margin to `p99_tpot`, so
a margin fitted on any other statistic compares distributions at different
points.

**A further caveat for whoever changes it.** The envelope was measured
closed-loop (a request pool) while the simulated side is an open-loop Poisson
process at 9.9 rps. D19 establishes that closed-loop TTFT does not transfer;
for TPOT the transfer is untested, and that work order's STEP V2 measures it.

**Where.** `experiments/p2_evidence/{v0_convert.py,results/v0_source_reconciliation.md}`;
annotations added to `docs/deviations.md` D22, `docs/patent_future_ideas.md`,
`planner/predictor/calibration.py` and `experiments/results/e5_self_rejection.md`.
Nothing in `profiles/calibration/` was changed.

## D101 — the performance envelope's `tpot_p50` field holds a mean for four of its nine points · Open (measured, not changed)

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V1, percentile-pair audit. Second
entry from the `D100–D109` block. It corrects D100 and is the cause D100 named
but did not find.*

**What D100 saw and what it missed.** D100 established that the c76
accuracy-domain point pairs a simulated p50 against a measured mean, and claimed
c76 was the only such point. The audit that the V1 percentile item required went
to the raw records for all nine and found the claim wrong — and the cause one
layer down, in the envelope rather than in the domain.

**`profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml` has a
field named `tpot_p50` that does not always hold a p50.**

| envelope conc | stored `tpot_p50` | raw mean | raw p50 | what it is |
| ---: | ---: | ---: | ---: | --- |
| 1.00 | 15.71 | 15.6987 | **15.7124** | p50 (mean of two repeats' p50) |
| 1.99 | 18.01 | — | **18.0103** | p50 |
| 3.98 | 19.48 | — | **19.4805** | p50 |
| 7.88 | 21.86 | — | **21.8554** | p50 |
| 15.59 | 26.05 | — | **26.0489** | p50 |
| **15.3** | **25.71** | **25.7147** | 25.4357 | **mean** |
| **29.3** | **31.18** | **31.1754** | 31.8615 | **mean** |
| **59.2** | **44.54** | **44.5418** | 45.1225 | **mean** |
| **107.2** | **67.88** | **67.8762** | 70.3958 | **mean** |

The five low-load rows come from `outputs/rngd_envelope_lowload/point_c*.json`,
which record `tpot_ms: {p50, p95, p99}` explicitly, and every one reproduces to
four decimals as the average of the two repeats' p50. The four rows measured
2026-08-31 come from `outputs/rngd_envelope/edf/real_c{16,32,64,128}.json` and
reproduce as the **mean** — they are the `TPOT avg` column of
`experiments/results/rngd_concurrency_envelope.md`, copied into a field whose
name says p50. The `tpot_p99` column is a genuine p99 throughout.

**How it reaches the accuracy domain.** `experiments/scripts/lowload_sim_error.py`
declares `"compared_metric": "tpot_p50"` and compares a genuine simulated p50
against `pt.tpot_p50` — so every domain point whose measured side is one of those
four rows is a p50-against-mean comparison. That is **c14.832** (measured c15.3),
**c25.181** (measured c29.3) and **c76** (interpolated from c59.2 and c107.2).

**Who else reads the field.** `planner/perf_envelope.py:339`
(`tpot_p50_ms=env.metric_at(conc, "tpot_p50")`) and
`experiments/scripts/e6b_measured_curve.py:105` both take it at its name. Any
consumer asking this envelope for a p50 above concurrency 15 gets a mean.
Whether that changed an E6 result has **not** been established here.

**And none of the nine is on the basis the verdict uses.** Separately from the
mislabelling: `planner/optimizer/feasibility.py:67` judges
`p99_tpot × (1 + m)`, and **no committed RNGD-CARD point pairs a p99 against a
p99** — five are p50/p50, three are p50/mean, one is a five-sample bucket mean.
Recomputed on p99 from the same raw artifacts, two points change sign — c14.832
from +3.26 % to −6.90 % and c15.212 from +2.47 % to −3.11 % — and because the
margin is one-sided, that is the difference between charging nothing and
charging 7.41 % and 3.21 %. The domain's zero crossing moves from between 15.212
and 16.6 down to between 8.211 and 14.832.

**Why nothing is changed.** Rule A3 forbids rewriting a measured artifact, and
re-fitting the domain on p99 invalidates the E-A1 corpus, `outputs/e6*/` and the
pinned values in `tests/test_margin_policy.py` and `tests/test_accuracy_domain.py`
— D100 §2.5's blast radius, for the same reason. Renaming the envelope field is
also not free: it is read by the planner and by committed experiment scripts. Two
of the recomputed p99 rows rest on n = 128, so a re-fit should re-measure them
rather than adopt this audit's numbers.

**Where.** `experiments/p2_evidence/{v1_percentile_audit.py,results/v1_percentile_audit.md}`;
D100 and `experiments/p2_evidence/results/v0_source_reconciliation.md` annotated
with the correction. No profile, calibration file or envelope was modified.

**Addendum 2026-09-18 (domain-scoping S7.3, user decision).** New domains are
fitted on **p99 from here on**, and say so in `AccuracyDomain.compared_metric`,
so the basis the margin is charged on is the basis it was measured on. The two
committed domains are **not** touched — this entry's "why nothing is changed"
stands, blast radius included — so the repository now holds both bases, and
`compared_metric: ""` on the old files means NOT STATED rather than p50.
`AccuracyDomainMargin` warns when a domain explicitly declares p50, which is why
silence earns no warning: it would fire on every run and tell nobody anything.
`lowload_sim_error.py --compare-stat p50|p99|both` is the tool the migration will
use; its default is `p50`, so every committed invocation still writes what it
wrote. **Migrating the two committed domains onto p99 is left as a separate
step** and is not part of S7. See D115.

## D102 — the open-loop A40 domain cannot live where the work order put it · Resolved 2026-09-16

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V2. Third entry from the
`D100–D109` block. A small path change, recorded because the work order names a
path the code forbids.*

**What the work order asks for.** V2's side product is a new accuracy domain from
open-loop measurement, at `profiles/calibration/a40.accuracy.openloop.yaml` — a
new file rather than an extension of `a40.accuracy.yaml`, per rule A3.

**Why that path does not work.** `load_accuracy_domains` globs
`profiles/calibration/*.yaml` and raises on two domains for the same hardware
label:

```
two accuracy domains for A40: profiles/calibration/a40.accuracy.yaml and
profiles/calibration/a40.accuracy.openloop.yaml; refusing to choose between them
```

Placing the file there broke 2 tests and errored 25 more — every test that loads
the committed tree. **The guard is correct** (uncertainty work order §2.4.1: a
margin policy must never silently pick between two measurements of one device),
and the ambiguity it refuses is real: both files measure hardware `A40`, through
different harnesses, and one of them is not yet meant for planning.

**How we adapt.** The file lives at
`profiles/calibration/openloop/a40.accuracy.openloop.yaml`. The glob is not
recursive, so the subdirectory keeps the file where the work order wants it
without asking the planner to choose. A consumer that wants it passes it
explicitly through the loader's `paths=` argument.

**What this defers rather than settles.** Whether the two A40 domains should ever
be merged is open, and V2 measured the reason to be careful: the open-loop
points' TPOT agrees with the committed file at both anchors (−0.26 % against
−0.32 % at L 11.35, −1.44 % against −1.41 % at L 169.37) but its TTFT does not
(−35.06 % against −17.10 % at the low end), and only +20.26 ms of that is the
client-side transport V2 measured directly. Until the rest is explained the two
must not be interpolated on one axis.

**Where.** `profiles/calibration/openloop/a40.accuracy.openloop.yaml`,
`experiments/p2_evidence/results/v2_openloop_concurrency.md` §5-6.
---

## D90 — above c1 the RNGD runtime never runs a fused pipeline, and the compiled grid rules out mixed steps · Recorded 2026-09-16

**What the planner assumed, implicitly.** The RNGD perf bundle and its accuracy
domain were built from EDF traces without recording *which* compiled plan
produced them, and the simulator's cost model reproduces a vLLM-style scheduler:
a step may mix prefill and decode tokens, batches take any size, and attention is
looked up at the batch's real token count and real KV. Two facts about the vendor
artifact contradict that, and both are properties of the compiled artifact rather
than of the card.

**1. `composed` is a c1-only path.** `d6ae6a43` compiles 47 pipelines — one
`kernelwise` (a composable IR assembled per layer) and 46 `composed`, each a fused
graph for exactly one decode bucket. `furiosa-llm`'s own `Wire pipeline hit rate`
gauge, tabulated from the committed serve logs of both envelope campaigns, is
99.8 % at concurrency 1 and then 47.5 / 12.0 / 1.8 / 0.5 / 0.0 / 0.0 / 0.0 at
c2 / c4 / c8 / c16 / c32 / c64 / c128, with repeats inside 0.2 pp. So at every
load the planner cares about, the runtime is on the kernelwise path. This is
also why D17's per-layer EDF traces at c16–c32 have per-layer stages at all: a
`composed` step has none to trace.

*Why the hit rate collapses is open.* Either the running batch size rarely
matches a compiled `bs` (H-a), or a `composed` pipeline, being compiled for one
`(bs, attention_size)`, can serve a step only when every sequence shares that KV
bucket (H-b). 47.5 % at c2 and 12.0 % at c4 read like the probability that two,
then four, sequences share a bucket, which favours H-b; STEP C.1 of
`WORK_ORDER_npu_exec_model_spike.md` decides it with a fixed-prompt-length run.

**2. Mixed prefill+decode steps are not compiled, so the runtime cannot do them.**
Every prefill bucket (8) and every extend bucket (74) has `batch_size = 1`, and
every decode bucket (46) has `input_ids_size = 1`. No compiled plan holds both a
prefill chunk and a decode token, so one forward pass is either one request's
prefill/extend chunk or a decode batch. Meanwhile
`serving/core/scheduler.py:101-120` batches several prefills together when
`prioritize_prefill ∧ ¬chunked`, and mixes decode with prefill chunks under one
`max_num_batched_tokens` budget when chunked prefill is on. No knob makes the
simulator reproduce "prefill is bs=1 and exclusive of decode" exactly; it can only
be approximated.

**3. The decode grid is a KV budget, and it caps the batch below what E6 asked
for.** `bs × attention_size` is 131072 on the `bs=1` row and 262144–524288 on the
rest, so the widest context shrinks as the batch grows: 65536 at bs=4, 16384 at
bs=16, 4096 at bs=64 and 128, and 2048 at bs=256. On sharegpt, whose mean KV is
≈ 2200 (D17), the bs=256 row cannot be used at all and the largest executable
decode batch is **128**. `rps_aware`'s E6 gave RNGD candidates
`max_num_seqs 256`, a setting this artifact has no plan for. The measured
envelope is unaffected — it was taken from the server, which simply never used
that batch size — but a plan quoting 256 as the operating point is quoting one
the hardware cannot reach.

**Consequence for D17.** D17 recorded "whether the runtime genuinely never mixes
[prefill and decode] is **not decidable from these traces**: the EDF CSV carries
durations, not timestamps, so co-occurrence within one forward is unobservable."
That remains true of the traces. The compiled grid settles the question from the
other side, structurally, and the answer is that it never mixes them. D17's
sentence is annotated in place rather than rewritten.

**Not fixed here, on purpose.** This is a spike's STEP 0. Nothing in `serving/`
or `planner/` changes; the spike measures how much of the RNGD accuracy domain's
error these differences explain before anything is built. Two rules the spike's
own later steps are written in terms of were also found to be wrong and are
carried as open items, not adopted: the kernelwise attention menu is the union of
the 128 compiled buckets and **not** a uniform 1024-token grid (its decode edges
are powers of two from 1024), and the batch-padding ladder
`1, 2, …, 256, 384, 512, 1024` is **not** powers of two above 256.

**Where.** `experiments/scripts/furiosa_artifact_buckets.py`,
`experiments/scripts/serve_log_hit_rates.py`,
`experiments/results/rngd_artifact_buckets.md`,
`experiments/results/rngd_pipeline_hit_rate.md`,
`profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml`,
`docs/npu_exec_spike.md` §0. Artifact `d6ae6a43` (furiosa-llm `b62dbc1`,
compiler `d19a92a2f2`, tp=8); classification per
`furiosa_llm/metadata/config_types.py`. The D-number comes from the `D90–D99`
block `WORK_ORDER_npu_exec_model_spike.md` claims in `CLAUDE.md` (2026-09-15).

---

## D91 — the work order's A.2 knob set livelocks, and three of the four mechanisms it names are already absorbed or unmeasurable · Recorded 2026-09-17

`WORK_ORDER_npu_exec_model_spike.md` STEP A decomposes the RNGD accuracy domain's
error into mechanisms an AOT bucket executor obeys and a vLLM-style scheduler does
not. Four of its instructions did not survive contact with the code and the
bundle. None of this changes `serving/` or `planner/`; STEP A is counting.

**1. `--no-enable-chunked-prefill --max-num-batched-tokens 1024` cannot run
sharegpt.** With chunked prefill off, `max_num_batched_tokens` is not a step budget
— `serving/core/scheduler.py:164-180` drops requests from the batch until the total
fits and returns `None` when the batch empties, printing `[WARNNING] Cannot load the
request to batch due to max_num_batched_tokens limitation`. A prompt longer than the
budget is then never schedulable. Ten of the first twenty sharegpt requests are
longer than 1024 tokens, up to 3452, so the run livelocks: the probe emitted 80,821
of those warnings and died at the timeout (`outputs/npu_spike/a2_asspec_probe/`,
exit 124). The runtime chunks too — its 74 `extend` buckets are `chunk <= 1024`
against non-zero KV — so "no chunked prefill" was never a description of it. A.2 was
run instead as two one-sided approximations, `a2_exclusive` (prefill exclusive of
decode, chunking left on at 8192) and `a2_chunk1024` (a genuine 1024-token step
budget, exclusivity given up), each labelled with what it does not capture.

**2. Batch padding is already inside the measured bundle, so R-pad/P2 must not be
modelled.** Up to 38 % of the simulator's decode lanes at c7.88 are padding once the
batch is rounded to a compiled size, yet re-pricing every step at the padded batch
moves the decode-cost sum by less than 1 pp at every point. Two independent reasons.
The bundle's dense table is nearly flat from 1 to 256 tokens — `qkv_proj` is 45.2 µs
at one token and 51.2 µs at eight — because the card is latency-bound there. And the
table's rows *are* the bucket points: it was measured on the card, which was already
padding, so the padding is in the measurement. Charging it again double-counts.

**3. Decode-attention grouping is also already inside the bundle, once.**
`profiler/perf/RNGD-CARD/.../bf16/meta.yaml` states each decode row's time is "total
decode-attention device time over (forwards x 32)" measured on sharegpt at that
concurrency, so D17's 1.95-to-3.08 attention executions per layer are inside the
number the simulator looks up. The work order's R-attn — replace the lookup with a
sum over KV groups — multiplies the grouping in a second time; done that way it
reads +31.6 pp at c15, which is not a contribution but a double count. What the
simulator can be wrong about is the *residual* diversity, this run's batches against
the sharegpt batches the row was measured on.

**4. Neither candidate grouping grid is the rule, and a third one is worse.** The
work order asked STEP A.1 to count KV groups "both ways", the artifact's decode edges
against a uniform 1024-token grid, and to pick whichever tracks D17's measurement.
Neither does, and the two are indistinguishable on sharegpt (2.64 vs 2.73 groups at
mean batch 15, against D17's 3.03). A third grid has the better structural claim —
above c1 the runtime is on the kernelwise pipeline (D90), whose attention menu is the
union of all 128 compiled buckets, 15 sizes 128-spaced to 1024 and powers of two
above — and it is the worst fit of the three: it keeps splitting as the batch grows,
reaching 5.37 at batch 15 where D17's measurement has flattened at 3.03. D17's curve
*saturates*; no "one execution per distinct compiled bucket" rule does. So the
grouping rule is **unknown**, and the R-attn contribution is unquantified: the same
rule on the same steps reads -3.7 pp or +26.5 pp at c15 depending on the grid
assumed. STEP C.3 has to measure what caps the execution count near three; STEP 0's
prediction that the uniform grid would predict "many more" groups than the real edges
is retracted.

**What STEP A does establish.** The step-cost model rebuilt from the perf DB
reproduces the simulator's own per-step cycles to a median ratio of 0.999-1.000, and
the replay of 369,270 steps balanced exactly, so the counting is sound. Prefill steps
are already `bs = 1` in 300 of 300 cases at every point, so that constraint has zero
magnitude at c1-c16. Mixed steps are 0.08-2.3 % of steps and 0.4-7.6 % of time, and
their count is exactly 299 above c1 — one per request after the first. Prefill
128-padding is a flat +6.6 to +6.7 pp and lands on TTFT, not TPOT. The step policy is
worth +0.2 to +5.4 pp and has the wrong sign below c16. The +11 % pessimism at
c2-c4 is explained by none of them.

**Where.** `experiments/scripts/npu_exec_step_census.py`,
`experiments/scripts/npu_exec_recharge.py`,
`experiments/results/npu_exec_step_census.md`,
`experiments/results/npu_exec_recharge.md`, `docs/npu_exec_spike.md` §A,
`outputs/npu_spike/`. The D-number comes from the `D90-D99` block
`WORK_ORDER_npu_exec_model_spike.md` claims in `CLAUDE.md`.

---

## D92 — the bucketed execution model narrows the RNGD error by 7 %, not by two thirds · Recorded 2026-09-17

`WORK_ORDER_npu_exec_model_spike.md` STEP B built the opt-in prototype STEP A's
decomposition had cut down to one rule, and measured what it is worth. The
prototype lives on the spike branch and is **not merged**; `serving/` on `main`
is unchanged.

**What was built.** `execution_model: bucketed_aot` in the cluster JSON makes a
prefill/extend step carry exactly one request and at most 1024 tokens of it, and
makes it exclusive of decode -- the shape D90 says the compiled grid has plans
for. P2 (decode quantisation) and P3 (group attention) were **not** built,
because D91 found both are already inside the measured bundle. One guard rather
than a fallback: the rule is implemented in `schedule_base` only, so combining it
with prefix caching raises instead of silently running the vLLM policy under a
bucketed label.

**Equivalence.** The R1/R2 anchors reproduce all four committed SHA-256 sums from
`after_d28`, and the six low-load points re-run without the key are byte-identical
CSVs to the pre-edit runs. Absent `execution_model`, nothing moved.

**The prototype does what it claims to the step structure.** Mixed steps go 19 to
**0** at c3.98, prefill steps 20 to 35 as prompts chunk at 1024, prefill batch
size stays 1, prefill token total unchanged.

**And it is worth 7 %.** The nine-point TPOT error span goes from
`[-48.59, +11.00]` (width 59.59 pp) to `[-43.73, +11.58]` (width 55.31 pp). The
work order's threshold for "the execution model explains most of the error" was a
residual of one third of the current span. It is not close. The sign flip does not
disappear either: it moves from between served 15.21/25.18 to between 26.49/40.47.
Below c16 the prototype makes the error WORSE -- +10.79 to +11.58 at c3.98 --
which is the same wrong-signedness the knob approximation showed.

**`strict` and `alternate` cannot be told apart.** The largest difference across
nine points is 0.03 pp. The work order's contingency for STEP C.2 -- if the
runtime's step order cannot be observed, run both knob values and take whichever
lands closer to the measurement -- **does not work**, and C.2 has to be answered
directly or left open. This is a fact about the experiment, not the runtime.

**The TTFT hypothesis STEP A raised is falsified.** A.1 found the simulator mixes
prefill with decode on 299 of its 300 prefill steps and offered that as a
candidate for D17's -32.6 % TTFT gap. Removing the mixing entirely moves simulated
TTFT p50 by -5.7 % to +3.0 % and p95 by +2.0 % to +7.1 % (simulator against
simulator; the measured side is closed-loop, D19). Nothing of that size closes a
32.6 % gap, and p50 mostly moves down where it would have to move up. **A
20-request version of the same comparison read +14.0 % on p50** -- a small-sample
artifact of the same kind D32 recorded, and the number this spike would have
quoted had it stopped at its smoke test.

**Not established: anything off sharegpt.** B.3's hold-out needs a measured
counterpart from STEP C.4, and STEP C has not run -- `npu0`, the card every
committed RNGD measurement was taken on, is held by another tenant's pod. Recorded
as undecided rather than substituted.

**Where.** `serving/core/scheduler.py` and `serving/__main__.py` on
`spike/npu-exec-b-prototype` (do not merge),
`experiments/configs/clusters/rngd-card-llama31-8b-tp1-aot-{strict,alternate}.json`,
`experiments/scripts/npu_exec_ttft_compare.py`,
`experiments/results/npu_exec_ttft.md`, `docs/npu_exec_spike.md` §B,
`outputs/npu_spike/b_*`, `outputs/d23fix/anchor/after_npu_exec_b/`.

---

## D93 — the decode grouping grid is the powers of two from 1024, and an attention execution is mostly fixed cost · Recorded 2026-09-17

`WORK_ORDER_npu_exec_model_spike.md` STEP C.1 and C.3, on **`npu2` (PCI
`45:00.0`)** — `npu0`, the card every committed RNGD measurement was taken on, was
held throughout by another tenant's pod. Results that are properties of the
artifact carry; results that are properties of the card are labelled `npu2`'s.

**1. The `composed` path is single-KV-bucket, not batch-1.** Holding concurrency
fixed and removing only the KV diversity — 64 requests, every prompt exactly 512
tokens and every completion 128 — takes the wire hit rate from 12.0 % on sharegpt
to **97.5 %** at c4. H-a is excluded: a batch-size-matching rule cannot be moved by
the prompt-length distribution. And the batch is rounded up rather than matched,
because c3 holds 96-98 % although 3 is not a compiled decode batch size. D90's
description of `composed` as effectively c1-only is right about the traffic and
wrong about the rule.

**2. A decode attention execution costs ~40 us fixed plus a per-sequence term.**
From 29,728 raw EDF executions across two probe workloads and three
concurrencies: `45.1 us + 3.15 us x n` at `attention_size` 1024, `37.1 + 8.54 x n`
at 2048. Dividing the committed `npu0` bundle's per-layer totals by D17's measured
executions per layer gives `43.1 + 7.15` independently -- a different card, a
different workload, a different derivation, agreeing to 14 % on the fixed term and
19 % on the slope. The structure transfers; the constants are `npu2`'s.

At group size 1 the longer bucket is **cheaper** (38.8 us at 2048 against 48.3 at
1024), tightly so at both. Unexplained, recorded as observed.

**3. Decode groups on the decode buckets, and D91's open item is closed.** Every
decode attention execution in the `[900, 1000]` KV runs used `attention_size
= 1024` and every one in the `[1900, 2000]` runs used `2048`. Not one used 896,
768, 640 or any other rung of the kernelwise menu -- those are for prefill and
extend. Of D91's three candidate grids the decode-edge one is the runtime's, the
kernelwise menu is excluded, and the uniform-1024 grid was a coincidence of
sharegpt's range.

**4. Nothing caps the executions near three.** The decode ladder is geometric, so
`[2048, 4096)` spans a 2:1 range of KV and a workload with a bounded KV
distribution occupies two or three buckets however large the batch grows. The
STEP A census measured exactly that on this grid, 1.05 to 2.64 groups as the mean
batch went 1.07 to 14.97. D17's saturation needs no cap mechanism. **Testable and
unmeasured:** a wide-KV workload should show more groups and more executions.

**5. So R-attn is a number, and it is small.** Re-priced with the measured cost
model -- re-grouping changes only the fixed term, so scaling the whole lookup by a
ratio of group counts, which is what STEP A did, overstates it by about a third --
and on the settled grid, R-attn is **-2.0 to -4.1 pp and flat in load**, against
the `-3.7 to +26.5 pp` D91 had to leave it at. The simulator over-charges decode
attention slightly: it pays for the group structure of the sharegpt batches its
table was measured on while its own batches are marginally less ragged.

**Consequence for the spike's conclusion.** With all three mechanisms settled the
decomposition closes at conclusion **(ii)**: at served 2.19 the measured error is
+11.00 pp and the mechanisms account for -2.00, so the residual is the error; at
served 15.21 they account for +1.78 of +2.47. The execution model explains a
small, quantified part and the accuracy domain keeps the rest.

**Where.** `experiments/scripts/npu_exec_attention_groups.py`,
`experiments/results/npu_exec_attention_groups.md`,
`experiments/results/rngd_pipeline_hit_rate.md`,
`workloads/{fixedlen-512in-128out-64,bucket1024-900in-100out-96,bucket2048-1900in-100out-96}.jsonl`,
`outputs/npu_spike/{c1_fixedlen,c3_bucket1024,c3_bucket2048}`,
`docs/npu_exec_spike.md` §C and §D.

---

## D94 — STEP C ran without NUMA binding; the harnesses now bind by default and the cost model survives the check · Recorded 2026-09-17

**The debt.** Every measurement in `WORK_ORDER_npu_exec_model_spike.md` STEP C was
taken with placement left to the kernel scheduler. `measure_envelope.py`,
`rebuild_rngd_bundle_from_edf.py` and `bench_furiosa_endpoint.py` had no
`numactl`, no `taskset` and no affinity handling of any kind, and
`planner/util/provenance.py` recorded `cpu_count` but not where the process was
allowed to run -- so no artifact said whether it was bound.

This was **already known to matter in this repo** and the spike did not carry it
over: `WORK_ORDER_p2_regular_spec_evidence.md` Appendix A.3 measured that leaving
the server's binding to chance is worth **1.93x of throughput on this host**, and
that every A40 measurement before it ran unbound. That work landed on `main`
during this spike's session and was not read.

**The host.** Two NUMA nodes, 48 CPUs each, distances 10 local / 32 remote. All
three RNGD cards report `numa_node = 0` (PCI `03:00.0`, `04:00.0`, `45:00.0`), so
the spike's `npu0`-against-`npu2` cross-check was not confounded by different
nodes -- which was luck, not method.

**The fix.** `--numa-bind` on both harnesses: `auto` resolves the card's node from
its PCI address and pins the server **and** the bench client to it, a node number
pins explicitly, `off` reproduces the old behaviour. The choice is recorded in
each point's JSON, and `provenance.cpu_placement()` now records the affinity mask
and which NUMA nodes it lands on.

**The default is `auto`, against this repo's usual convention** that a new flag
defaults to the old behaviour so committed invocations stay byte-identical. That
argument does not transfer to a measurement whose old behaviour is
*unreproducible*: an unbound run's number depends on where the scheduler put it.

*Trap worth recording.* `/sys/class/rngd_mgmt/*` are **virtual** devices with no
PCI parent, so `device/numa_node` cannot be followed and the obvious sysfs walk
silently finds nothing. The resolver asks `furiosa-smi info` for the BDF and reads
`/sys/bus/pci/devices/<bdf>/numa_node`. The first implementation used the sysfs
guess, returned "unresolved" and bound nothing -- it failed safe by design, but a
reader would have believed binding was on.

**The check.** `bucket1024` re-collected on the same card with `--numa-bind auto`
(server mask `0-23,48-71`, against `0-95` unbound), same workload, same
concurrencies:

| group size | unbound us | bound us | delta |
| ---: | ---: | ---: | ---: |
| 1 | 48.3 | 48.3 | +0.02 % |
| 2 | 54.7 | 54.8 | +0.12 % |
| 4 | 50.3 | 50.2 | -0.24 % |
| 8 | 67.6 | 67.6 | -0.05 % |
| 16 | 96.5 | 96.6 | +0.05 % |

Least squares `43.7 + 3.20 x n` on both sides -- fixed term 0.05 %, slope 0.07 %.
**D93's cost model stands and needs no correction.** The reason it survives is
structural: these are EDF *device* cycles, and NUMA governs host memory and DMA
staging rather than on-device kernel time. The RNGD artifact's `tp=8` is two fused
quads inside one card, so from the host it is one engine -- the shape that showed
0.42 % in the A40 check, not the four-worker shape that showed 93 %.

**Not cleared by this.** The wall-clock half of `measure_envelope.py` -- served
concurrency, throughput, TPOT, TTFT -- was not re-measured. C.1's hit rates are
runtime counters and are unaffected. The committed RNGD envelope and accuracy
domain were taken unbound by earlier sessions and re-taking them is outside this
spike.

**Where.** `experiments/scripts/measure_envelope.py`,
`experiments/scripts/rebuild_rngd_bundle_from_edf.py`,
`planner/util/provenance.py`, `experiments/results/npu_exec_attention_groups.md`,
`outputs/npu_spike/c3_bucket1024_numa/`, `docs/nodes/npu.md`.

---

## D110 — an accuracy domain answers only for the configuration it was measured under, and a mismatch is its own refusal · Recorded 2026-09-17

*`WORK_ORDER_domain_scoping.md` STEP S1. First entry from the `D110–D119`
block. It splits one rejection stage into two and adds required application
conditions to `AccuracyDomain`; D33 is the parent, which built the domain and
its `refuse` default, and D102 is the sibling that placed the second A40 file.*

**What the code assumed, implicitly.** An accuracy domain was keyed by hardware
and narrowed, optionally, by `model` / `variant` / `workload_shape` /
`arrival_process`. Everything else about the deployment it was measured under —
the parallelism degrees, how many islands of that hardware ran, whether the
serving process was bound to the accelerators' NUMA node — was not in the
schema, so it could not be compared, so the domain answered for every candidate
on that hardware. The assumption underneath is that the simulator's error is a
property of the *device*. It is not: it is a property of the *deployment*.

**What V3 measured.** `experiments/p2_evidence/results/v3_verdict_accuracy.md`
took candidate P1 — one A40 island at **tp=4** — to hardware. Its margin came
from `a40.accuracy.yaml`, which is fitted at **tp=1**, and the operating point
was comfortably inside that domain's load axis, so every check the planner had
passed:

| | | |
| --- | ---: | --- |
| margin the domain charged at P1's operating point | **1.13 %** | correct arithmetic on the wrong measurement |
| simulator error actually measured at P1 | **−44.63 %** | p99 TPOT 36.5 ms predicted, 66.0 ms delivered |
| margin that would have been needed | **~80 %** | the per-point rule under-corrects by ~70× |
| the same domain's error at tp=1 (V2) | **−1.05 %** | the domain is not wrong, it is being asked the wrong question |

Two further conditions came out of the same measurement and are in the schema
for the same reason: NUMA binding was worth **1.93×** of throughput on one
otherwise unchanged deployment (A.3), and the effective all-reduce bandwidth of
the link that carries TP=4 measured **8.8 GB/s** against a `vendor_spec` 64.0
(D-entry for that is S3's, not this one).

**How we adapt.**

1. **`AccuracyDomain` gains required application conditions** — `hardware`,
   `parallelism {tp, pp, dp}`, `placement {islands, device_binding}` — and
   `check_conditions()`, which compares a candidate's `(hardware, model,
   variant, workload_shape, arrival_process, tp, pp, dp, islands,
   device_binding)` against them. **Exactly**, per assignment rather than
   aggregated over the plan: two islands of one hardware at different
   parallelism have no honest single `tp`.
2. **Unstated on either side is UNCHECKED, not matching.** That is D33's
   convention for `model`/`variant`/`workload_shape`, kept and extended: a
   domain that does not record what it was measured under cannot refuse
   anything on that ground, and refusing anyway would report "measured
   elsewhere" as "measured differently". Skipped fields come back as
   `condition_warnings` and land in the margin basis as applied on trust.
3. **New rejection stage `CALIBRATION_CONDITION_MISMATCH`**, split out of
   `OUTSIDE_CALIBRATION_DOMAIN`, which keeps the load-axis case. Both stay
   epistemic — unmeasured, never infeasible, never `closest_plan` — but they
   ask for different experiments, and that is the whole reason to separate
   them: this one means *measure at this candidate's configuration*, the other
   means *measure further along the load axis of the configuration we have*.
   The refusal carries `mismatch_fields` and `required_measurement
   {hardware, tp, pp, dp, islands, binding}`, which `search` prints as
   `measure at: {...}`.
4. **Policy key `--condition-mismatch {refuse,warn}`, default `refuse`**,
   symmetric with a domain's own `outside_domain: refuse` (D33). `warn` applies
   the domain anyway and records the mismatch; it exists to measure what the
   refusal costs, not to plan with, and a test pins its margin to the pre-S1
   number.
5. **`Node.device_binding`** (`numa_pinned` / `unpinned` / `unknown`, default
   `unknown`) is keyed through to the policy **by island**, not by hardware:
   one cluster can hold a bound and an unbound node of the same accelerator.
   Nothing in the compiler reads it — it is an application condition, not a
   performance input.
6. **`profiles/calibration/index.yaml`** lists every registered domain and its
   conditions, so a refusal can say what *does* exist. It is a copy, and
   `test_the_index_matches_the_domain_files` rebuilds it from the files so it
   cannot become a second truth.

**The four domain files, with no `points` value touched** (rule A3; each
parallelism value read off the file's own provenance, not inferred):
`a40.accuracy.yaml` tp=1 islands=1 `unpinned`; `openloop/a40.accuracy.openloop.yaml`
tp=1 islands=1 `unpinned` (PR #104 re-ran this ladder bound and it agreed to
0.42 % / 0.37 % / 0.46 % — recorded, not used to relabel the measurement);
`rngd_card_edf.yaml` tp=1 islands=1 `unknown` (nothing records the host binding
on the NPU node); `rngd_perpe.yaml` tp=8 islands=1 `unknown` — not in the work
order's list of three, added because it is the one file whose `tp` is not 1 and
leaving it unstated would make it the only domain the match test cannot check.

**What this does not do, deliberately.** `arrival_process` is in the match rule
and the CLI presents `open_loop` (`plan` always replays an arrival trace), but
no committed A40/RNGD-CARD file states the field, so it is skipped-and-flagged
today. Filling it in would make `rngd_card_edf.yaml` closed-loop (D19) and
refuse every RNGD candidate; `model` / `variant` are left unstated for the same
kind of reason (D33: scoping a domain is a measurement claim). Both move E-A1's
counts, so they belong to STEP S4's re-run, not to a side effect here.

**The risk is accepted, not mitigated.** With every domain at tp=1 but one
(`rngd_perpe.yaml`, tp=8), the exact match rule refuses most multi-GPU
candidates. That is the honest state of the measurements, and the refusal now
carries `required_measurement`, so the output
is not "unknown" but "measure here" — which is the point. Relaxing the rule
(ignoring `tp`, say) is explicitly not done.

**Where.** `planner/predictor/calibration.py`, `planner/optimizer/margin.py`,
`planner/optimizer/exhaustive.py`, `planner/plan.py`, `planner/inventory.py`,
`planner/render.py`, `planner/__main__.py`, `profiles/calibration/index.yaml`,
the four domain files, `tests/test_calibration_condition.py`,
`experiments/p2_evidence/results/v3_verdict_accuracy.md` §6.3 and A.3.

---

## D111 — an input with no sourced width left the measurement plan entirely; the grade now carries a default · Recorded 2026-09-17

*`WORK_ORDER_domain_scoping.md` STEP S2, the second entry from the same
`D110–D119` block D110 opens. It amends how absolute rule A1 of
`WORK_ORDER_uncertainty_planner.md` is enforced, so read that rule and D33
first.*

**What the code did.** `grades.yaml` maps a (kind, grade) to a rule that turns a
nominal value into a range. A combination it does not cover, or covers with
`rule: unbounded`, produced `Range(lo=None, hi=None)`, and everything downstream
followed honestly from there: `sensitivity.analyze` returns `delta_regret=None`
for an unbounded item, `measurement_plan.build` puts it in `undecidable`, and the
renderer prints "CANNOT BE DECIDED BEFORE MEASURING". The item never competed
for a server-hour, because comparing it to one that had a range would have meant
inventing its width.

**What that cost, measured.** V3 traced a −43.4 % TPOT error on an A40 TP=4
deployment to one input: `link_bw:pcie-a40a-02`, the PCIe path the all-reduce
crosses, `source: vendor_spec` at 64.0 GB/s against 8.8 GB/s measured. That
input was `vendor_spec`, `link_bw/vendor_spec` is an `unbounded` row (and the
row explains at length why no honest `r_min` could be formed for it), so the
planner's own measurement plan had listed it as undecidable and ranked nothing.
`costs.yaml` prices the measurement at **0.114 h**. The queue the operator was
handed could not recommend the six minutes that would have explained the whole
error.

**How we adapt.** `grades.yaml` gains a second layer, `defaults:`, reached only
when the (kind, grade) rule sources no width of its own:

1. **A default is per grade**, with an optional `kind` override. `vendor_spec`
   is `[1/8, 1] × nominal`, from V3's own measurement (8.8 / 64.0 = 0.1375,
   rounded down; a spec bandwidth is an upper bound). `link_lat/vendor_spec`
   overrides it with `[1, 8]`, because the grade-level row is the wrong shape
   for a latency — a spec latency is a *lower* bound, and applying the
   bandwidth row would have claimed every link beats its datasheet.
   `analytical`, `calibrated` and `placeholder`/`unknown` take E2's measured
   MAPEs (0.3888 / 0.2951 / 0.4288). `sim_error` takes **absolute** half-widths
   (0.02 for `measured`, from the two repeat measurements in the repo; 0.446 for
   unfitted hardware, from V3's own worst end-to-end error), because an error
   fraction is additive and may be zero — a relative width on a nominal of 0.0
   is exactly the silent zero the registry exists to refuse.
2. **`user_defined` has no default, deliberately.** A what-if is the user's
   number, not a population, so "cannot be decided before measuring" stays a
   real category rather than a formality.
3. **A default is labelled, everywhere.** `Range.range_source` is `sourced` or
   `default`, and it travels to `Sensitivity`, to `MeasurementItem`, to the
   registry table (`~` and its legend) and to the plan's rows
   (`[default range]`). A default never replaces a sourced width, and a range
   built by a caller — which is how E-B1/B2/B3 supply their degraded intervals —
   is `sourced` and untouched by this table.
4. **Absolute rule A1 still holds, in the form S2 sets it**: every default
   carries its own `source` and the loader refuses one that does not. What
   changed is not "a number may now be invented" but "a stated policy with a
   cited basis may stand in for a missing measurement, while saying that it is
   one".

**A fourth bucket, because the third stopped covering.** With ranges where there
were none, an input can now be *decidable and worthless*: swept over its range,
it moves the recommendation by exactly zero. §2.5 keeps it out of the queue, it
is not undecidable, and before S2 it had been hidden inside `undecidable` for
want of a range. `MeasurementPlan.inert` names it, so every registry input lands
in exactly one of `items` / `uncovered` / `undecidable` / `inert` — a property
`tests/test_cli_accuracy_domain.py` asserts.

**What the measurement showed, which is not what S2 predicted.** The work order's
hypothesis was that `link_bw:pcie-a40a-02` would rank **first**. On the E-A1
corpus it does not rank at all: it moves from `undecidable` to **`inert`**
(`experiments/uncertainty/results/s2_default_ranges.md`, both arms of the same
run). The reason is not the range. It is that a link bandwidth reaches a
predicted metric through exactly one path — `apply_pd_transfer_cost`'s
prefill→decode KV transfer (`perturb._reprice_transfer`) — and the E-A1 corpus
was generated with `enable_pd=False`, so it holds no P/D candidate for any link
to be on the path of. A link that carries a TP all-reduce *inside* an island has
no route into any predicted metric at all.

So S2 changes the diagnosis from "we cannot say what this link is worth" to "we
can say, and on this corpus it is worth nothing" — which is a real improvement
and is still not the V3 answer. **What is missing is the LINK_BW item's
meaning**, which is STEP S3's subject: the registry prices a link as a KV
transfer, while the error V3 measured came from the same link carrying a
collective inside one island.

**Where.** `profiles/uncertainty/grades.yaml` (`defaults:`),
`planner/uncertainty/grades.py` (`GradeDefault`, `DefaultRule`,
`GradesTable.default_for` / `without_defaults`),
`planner/uncertainty/registry.py` (`_default_range`, `Range.range_source`),
`planner/uncertainty/sensitivity.py`, `planner/uncertainty/measurement_plan.py`
(`inert`), `planner/render.py`, `planner/__main__.py` (`--grades`),
`tests/test_default_range.py`, `experiments/uncertainty/s2_default_ranges.py`.

## D112 — a link does not have one bandwidth; the LINK_BW item is keyed by the traffic that crosses it · Recorded 2026-09-18

*`WORK_ORDER_domain_scoping.md` STEP S3, the third entry from the `D110–D119`
block D110 opens. It finishes what D111 ends by naming, so read D111's closing
paragraph first, and A.4 of
`experiments/p2_evidence/results/v3_verdict_accuracy.md` for the measurement.*

**What the code did.** `Link.bandwidth_gbps` was a single number per link and
`link_bw:<link_id>` a single registry item per link, so both the simulator input
and the uncertainty accounting treated "the bandwidth of this wire" as a
quantity. The same PCIe bridge on the A40 node measures, in one run:

| what crosses it | measured | against a `vendor_spec` 64.0 |
| --- | ---: | ---: |
| a direct device-to-device copy | 25.0 GB/s | 0.39× |
| a two-rank all-reduce | 19.29 GB/s | 0.30× |
| the four-rank all-reduce a tp=4 island runs | 8.8 GB/s | **0.1375×** |

So the datasheet number is not merely optimistic — there is no single figure for
it to be optimistic *about*. Which of the three is right depends on the
collective, the group size, the message size and the binding, and V3 said so
outright: *"a static per-link field cannot hold a quantity that depends on how
many devices a candidate spans and which ones"*.

**What that cost, twice.** First, the simulator got 64.0 for a hop that
delivered 8.8, which is the whole of V3's **−43.44 %** p99 TPOT error on P1.
Re-simulated at 8.8 the same candidate predicts 60.09 ms against 64.62 measured,
an error of **−7.01 %**, and served concurrency lands within **0.73 %**
(`experiments/p2_evidence/results/v5_resim_measured_link.json`). The one input
was worth 23.5 ms of the 28.1 ms error.

Second — and this is the part S2 could not reach — the registry could not point
at it. `perturb._reprice_transfer` was the only LINK_BW rule, and it re-prices
`apply_pd_transfer_cost`'s prefill→decode handoff, so on a corpus built with
`enable_pd=False` it moved nothing, every LINK_BW item swept to a regret of
exactly zero, and the measurement plan reported them as **`inert`**: "measured,
they would change no plan". Both of E-A1's uncertain LINK_BW items are
intra-island, including `pcie-a40a-02` itself. The plan's own words for the
input that explained the error were that it was not worth 0.114 h. The module
docstring stated the premise that made this look correct — "the transfer term is
the only place a link bandwidth reaches a predicted metric today" — and it was
false: an intra-island link's bandwidth is the `min` that
`island_interconnect` reduces into the simulator's own scalar `link_bw`, which
prices every TP collective inside ASTRA-Sim.

**How we adapt.**

1. **The schema carries measurements beside the spec value, never over it.**
   `Link.measurements[]` holds a `LinkMeasurement` per
   `(collective, world_size, msg_size_class, binding)` with `bus_bw_gbps`,
   `method`, `msg_bytes`, `date` and `raw`. `bandwidth_gbps` and its `source`
   are untouched (absolute rule A3), so the comparison that makes a measurement
   worth having survives. `source: measured` without a `method` is refused: the
   torch probe this repo used and `nccl-tests all_reduce_perf` are not
   interchangeable evidence, and S6(ii) is open for exactly that reason.
2. **`world_size` is in the key, which the work order did not ask for.** §S3
   names `(link_id, collective, msg_size_class, device_binding)`. 8.8 and 19.29
   differ *only* by group size, so under the four-part key they collide and one
   silently answers for the other — the fixture carrying both would not load.
   This is the field V3 asked for in the sentence quoted above. Recorded here
   rather than left implicit; the real data wins.
3. **The simulator is asked for a stated traffic kind.** `island_interconnect`
   asks for `all_reduce` at `bulk` and the island's TP degree;
   `_inter_island` asks for `p2p` at `bulk`, world_size 2, because a P/D handoff
   is one sender and one receiver. A hit is used and recorded in
   `TopologyReduction.assumptions`; a miss falls back to the spec value and
   records *that*, naming the measurements the link does carry. Every caller
   that states no collective — which is every caller predating S3 — gets the
   nominal value unchanged, so no committed result moves.
4. **The band boundaries come from the curve, not from round numbers.** Measured
   all-reduce busbw on that path climbs from 0.23 GB/s at 8 KiB and is flat only
   from 4 MiB up (4–64 MiB within 4 %), so `bulk` starts at 4 MiB and is the
   only band that answers the simulator's asymptotic `link_bw` term. A
   `msg_size_class` its own `msg_bytes` contradicts is refused.
5. **An item no closed form can price is reported as such, not as inert.** An
   `all_reduce` LINK_BW item returns `requires_resimulation` from `perturb`,
   `delta_regret=None` from `sensitivity`, and lands in
   `MeasurementPlan.needs_resimulation` — a fifth bucket beside `items`,
   `uncovered`, `undecidable` and `inert`, and the partition stays total.
   `--resimulate-top` prices it exactly by rewriting the link and re-simulating,
   which is the only way to price it: reproducing ASTRA-Sim's collective cost
   model in arithmetic outside the simulator would be inventing physics.
   `resimulate.py`'s docstring claimed these items' closed form was "already
   exact, so any movement is a bug in the rule"; that was true of the handoff
   half only and is corrected.

   **Three things had to move before that escape hatch actually fired**, and
   each of them would have silently made the new bucket a dead end:

   * `sensitivity.refine` skipped every item whose `delta_regret` was None. The
     rule was written for an *unbounded* range, where there is nothing to
     simulate at; applied to an item that has an interval and only lacks a
     pricing rule it skipped exactly the inputs `--resimulate-top` exists for.
     It now takes them, and `tests/test_resimulate.py` pins that.
   * `Refinement.closed_form` was `float` and got `closed.delta_regret or 0.0`.
     For an item with no closed form that records a **zero**, which reads as
     "the closed form says this input does not matter" — the exact claim S3
     removed. It is `float | None` now.
   * E-B3's link identity control asserted that a LINK_BW item's closed form is
     exact and any disagreement with simulation is a bug in the rule. For an
     `all_reduce` item there is no rule to be exact, so it would have reported a
     bug in every one of them; `active_records` excludes them and
     `no_closed_form` counts them instead of dropping them.
6. **`measure-apply` files a measurement instead of overwriting a column.** A
   keyed `--input` appends to (or replaces by key) `measurements[]` on the copy;
   an unkeyed one still moves the spec value, so the pre-S3 invocation works.

**What this does NOT fix, and it is V3's item #2.** The simulator has no
representation of *which* devices inside an island a TP group occupies, so it
routes a tp=2 group over the cross-pair link although the hardware's tp=2 runs
inside the NVLink pair and never touches it. S3 fixes the group-**size** axis:
a tp=2 candidate now gets the 19.29 GB/s two-rank figure instead of inheriting
the four-rank 8.8. It does not fix the device-**identity** axis, and the +18.21 %
error V3 measured by giving tp=2 the tp=4 value is only partly removed by that.

**Two smaller consequences worth knowing.**

*A `vendor_spec` link with a measured collective now reports an uncertain
latency.* In the S3 fixture the two PCIe links go back to `source: vendor_spec`
with a measured `all_reduce`, so their bandwidth leaves the registry while their
latency correctly enters it — no link latency in this repository is measured.
The old hand-edited copy wrote `source: measured` and had been crediting those
latencies as measurements.

*On `pd-rngd-gpu.yaml` all 16 uncertain LINK_BW items are intra-island*, so
until `--resimulate-top` runs, `--measurement-plan` ranks no link on that
fixture at all. That is a truthful "cannot say without simulating" replacing a
false "not worth measuring", and quantifying what it does to E-A1's counts is
STEP S4's job, not this entry's.

**Where.** `planner/inventory.py` (`LinkMeasurement`, `Link.measurements`,
`Link.measurement_for`, `Collective`, `MsgSizeClass`, `MSG_SIZE_CLASS_BYTES`,
`msg_size_class_of`), `planner/topology.py` (`link_bandwidth_gbps`,
`effective_bandwidth_gbps`'s optional request, `island_interconnect`,
`_inter_island`, `_binding_of`, `world_sizes` on both reductions),
`planner/predictor/llmservingsim.py` (passes the per-assignment TP degree),
`planner/uncertainty/registry.py` (`LinkTraffic`, `LinkItemKey`,
`link_item_id`, `parse_link_item_id`, `_link_traffic`, `_link_items`),
`planner/uncertainty/perturb.py` (`_collective_needs_simulation`,
`requires_resimulation`, and the handoff rule now asking for `p2p` so it and
`apply_pd_transfer_cost` cannot disagree about one wire),
`planner/uncertainty/sensitivity.py` (`requires_resimulation` through
`_Swept`/`Sensitivity`, `refine` no longer skipping these items,
`Refinement.closed_form` nullable),
`planner/uncertainty/measurement_plan.py` (`needs_resimulation`),
`planner/uncertainty/resimulate.py`, `planner/render.py`,
`planner/optimizer/exhaustive.py` and `planner/util/kv_transfer.py` (the handoff
priced as a p2p bulk copy), `planner/candidate_generator.py` (the all-reduce
floor asks with the CANDIDATE's tp, not the island's size - a bound computed on
another group's bandwidth is not a relaxation),
`planner/__main__.py` (`measure-apply`),
`experiments/uncertainty/eb3_closed_form_vs_resim.py` (`no_closed_form`),
`experiments/uncertainty/s2_default_ranges.py` (reports the new bucket),
`tests/test_resimulate.py`,
`experiments/configs/clusters/pd-rngd-gpu-card-measured-pcie.yaml` (rewritten
onto the schema; the simulator still receives 8.8 for the tp=4 island, verified),
`tests/test_link_effective_bw.py`.

## D113 — a domain states the arrival process it was fitted under, and the committed E-A1 numbers need `warn` to reproduce · Recorded 2026-09-18

*`WORK_ORDER_domain_scoping.md` STEP S4, the fourth entry from the `D110–D119`
block. It fills the two fields D110 deliberately left unstated and records what
D110's own default does to a committed experiment. Read D110 first, and D19 for
the arrival-process evidence.*

**What the code did, and it is a reproduction failure rather than a bug.** S1
ships `condition_mismatch="refuse"` as the planner's default. Every committed
accuracy domain is fitted at one island with `dp=1`, and at `tp=1` except
`rngd_perpe.yaml`, which is tp=8. E-A1's fixture generates candidates at tp up
to 4, dp up to 2 and across two islands. (The tp=8 domain never enters here at
all: it is filed under hardware `RNGD`, the per-PE model, while this fixture's
NPU is `RNGD-CARD`. Hardware is matched before parallelism is.) So on
`main` after S1, `experiments/uncertainty/ea1_margin_modes.py` returns **0
feasible for conditions (c) and (d)** with 276 candidates held, and the
**50 / 244 / 30** that `ea1_margin_modes.md` publishes — and that D22's
revalidation is anchored to — cannot be obtained from the committed script at
all. Nothing is wrong with either; they are answers to different questions, and
there was no way to ask the older one.

**How we adapt.** Rules (a)–(d) are built with `condition_mismatch="warn"`,
which applies a domain measured elsewhere under protest and records the
mismatch. With it they reproduce exactly — 70 / 10 / 50 / 50, and (d)'s
244 + 30. `warn` exists to measure what the refusal costs, never to plan with;
the planner's own default is unchanged. A fifth rule **(e)** is the refusal:
per-operating-point margin, refusing an operating point outside the measured
range *and* a configuration no domain was measured under.

**What (e) costs on E-A1: every candidate.** 0 feasible, no recommendation,
**312 of 324 held** as `calibration_condition_mismatch`, 6 outside the domain,
6 `slo_violated`.

**And the work order's hypothesis was wrong about which field does it.** It
predicted "A40 `tp>=2` or multi-island candidates are all held". They are, but:

| fields that disagreed | held |
| --- | ---: |
| `dp` alone | 108 |
| `dp` + `islands` | 90 |
| `arrival_process` alone | 36 |
| `tp` alone | 24 |
| `dp` + `islands` + `tp` | 18 |
| `dp` + `tp` | 12 |
| `islands` + `tp` | 12 |
| `islands` alone | 6 |
| `arrival_process` + `islands` | 6 |

Per field, which is the tally the planner prints: **`dp` 228, `islands` 132,
`tp` 66, `arrival_process` 42.** So **`tp` accounts for 66 of 312**. The missing scoping axis
the disclosure describes as "the parallelism degree a domain was fitted at" is
wider than parallelism: **replication and placement are refused more often than
parallelism is**, and a measurement programme aimed only at tp would close less
than a quarter of these.

**The two fields S1 left unstated are now stated, each from its own file's
provenance and none inferred:**

| domain | `arrival_process` | where it is written down already |
| --- | --- | --- |
| `a40.accuracy.yaml` | `open_loop` | its header: "OPEN-LOOP, deliberately"; `measure_envelope_openloop.py`, and `python -m bench run` for the 170.56 point |
| `rngd_card_edf.yaml` | `closed_loop` | its `bucket_migration.why_unresolved`: "a burst against a closed-loop client… `bench_furiosa_endpoint.py` IGNORES `arrival_time_ns` and drives a fixed concurrency of 64 (D19)" |
| `rngd_perpe.yaml` | `closed_loop` | its measured half is the RNGD-CARD envelope, whose own file states `closed_loop: true` |

`variant: bf16` on the A40 and per-PE domains, from their `fitted_from` paths
and headers. **Not on `rngd_card_edf.yaml`**: its one `fitted_from` artifact is
a percentile table that records no model and no dtype, and the bucket label
`sharegpt-llama31-8b-20` names a family, not a precision.

**`model` is stated on NONE of them, and that is S4's decision.** This
repository holds two strings for one set of weights —
`meta-llama/Llama-3.1-8B` in the envelope paths and
`NousResearch/Meta-Llama-3.1-8B` in the open-loop A40 domain's own
`provenance.deployment`. `check_conditions` compares strings, so stating either
would refuse a run on the MIRROR NAME rather than on a model difference; a
false refusal is not a safer error than a missing check, it is a different
wrong answer. Stating it needs an alias policy that does not exist. Measured:
with `model` and `variant` stated on all three the counts are IDENTICAL
(312 / 6 / 6) and no candidate mismatches on either field, so guessing buys
nothing.

**`arrival_process` moved 36 candidates and exactly the right ones.** Held rose
276 → 312; all 42 candidates it touches are RNGD-touching (12 single-card, 30
A40+RNGD mixes), 36 of them disagreeing on that field alone. **No A40-only candidate moved**, because
the A40 domain is open-loop and so is `plan` — the control that says the field
discriminates rather than refuses.

**The 12 single-card RNGD candidates are the point.** They match every other
condition and had been held as `outside_calibration_domain` at served
concurrency 144.6 against a measured [1.02, 76]. They are now held as
`calibration_condition_mismatch`. The verdict is unchanged and **the measurement
it asks for is not**: an envelope point above c=76 no longer answers them, an
open-loop refit does.

**V3's P1 gets its ending.** One A40 island at tp=4 against a domain fitted at
tp=1: held at every TPOT SLO from 38 to 50 ms, identically, because a condition
mismatch is decided before any margin exists and no threshold can move it.
Rules (a)–(d) made 6 / 3 / 6 / 6 false passes there; (e) makes none, and none
of the correct rejections either. Beside it, S3 (D112) shows the prediction was
recoverable where the margin was not: at the measured 8.8 GB/s link the same
candidate predicts 60.09 ms p99 TPOT against 64.616 measured (−7.01 %) and
served concurrency within 0.73 %.

**A limit, recorded rather than fixed.** A condition mismatch is
**whole-domain**: `arrival_process` disagreeing withdraws the TTFT and the TPOT
error together. D19's evidence is narrower — a burst and a spread arrival
process differ and "the difference lands entirely in TTFT" — so a closed-loop
domain's TPOT error may survive into an open-loop deployment and refusing it is
conservative rather than correct. Per-metric condition scoping would express
that; it is not built and is not in this work order.

**A trap closed on the way past.** `v3_slo_sweep.py` wrote
`experiments/p2_evidence/results/v3_slo_sweep.json` — a **committed** artifact —
with no way to redirect it, the same shape as the `eb1_regret_vs_budget.py`
overwrite `docs/HANDOVER.md` §3 records. It now takes `--out-json`, and `--ea1`
(a record other than the committed one) **requires** it.

**Where.** `profiles/calibration/{a40.accuracy,rngd_card_edf,rngd_perpe}.yaml`
(`arrival_process`, `model`, `variant`; no `points` value touched, rule A3),
`experiments/uncertainty/ea1_margin_modes.py` (rule (e), `warn` for (a)–(d),
the scope built as `planner/__main__.py` builds it),
`experiments/p2_evidence/v3_slo_sweep.py` (the (e) column, `--ea1`,
`--out-json`), `experiments/p2_evidence/v3_select_candidates.py`
(`load(ea1_path)`, `row_for(..., rules)`),
`experiments/uncertainty/results/ea1_s4_condition_refuse.md`,
`experiments/p2_evidence/results/v3_addendum_s4.md`,
`tests/test_domain_conditions_stated.py`.

## D114 — the open-loop harness could not be ported to FuriosaAI, so the open-loop *protocol* was added to the server harness instead · Recorded 2026-09-18

*`WORK_ORDER_domain_scoping.md` STEP S7.2, the fifth entry from the `D110–D119`
block. Read D19 first (why the two load protocols are not interchangeable), then
D102 (why two harnesses for one hardware label need two domain files) and D113
(why the arrival process became a refusal, which is what created this task).*

**What the work order said to do.** S7.2, as written in the rev, said the job was
"the mirror image of the CUDA porting §2.2 records — server launch, sampler,
bench interpreter", i.e. add a `furiosa` backend to
`experiments/scripts/measure_envelope_openloop.py`, the way §2.2 added a `cuda`
backend to `measure_envelope.py`. `docs/HANDOVER.md` §2.10.1 item 1 said the same
thing and called it "a port, not a re-run".

**Why that is impossible, and it is not a matter of effort.**
`measure_envelope_openloop.py` does not launch a server. Its core is
`python -m bench run`, which is upstream's own harness, and `bench/core/runner.py`
drives **`vllm.v1.engine.async_llm.AsyncLLM`** in-process. There is no FuriosaAI
equivalent of AsyncLLM — `furiosa-llm` is a server, addressed over HTTP. So the
three seams §2.2 found in the closed-loop harness (server launch, power sampler,
bench interpreter) do not exist in this file: there is no launch line to swap,
because the engine is a Python object. Writing an in-process FuriosaAI driver
would mean adding a second engine path to `bench/`, and `bench/` is upstream and
frozen until Phase 5 (absolute rule 1). The work order named a file that cannot
host the change.

**What the repository already had that does work.** The A40 has *two* open-loop
domains, and D102 keeps them in separate files because they came from different
harnesses:

| domain file | harness |
| --- | --- |
| `profiles/calibration/a40.accuracy.yaml` | `python -m bench run`, in-process AsyncLLM replay |
| `profiles/calibration/openloop/a40.accuracy.openloop.yaml` | **a deployed server driven over HTTP** (`replay_to_endpoint.py --open-loop`), via `planner deploy` |

The second is the route RNGD needs, and its own header says why: it "is what a
Phase 4 deployment actually is". It also cannot be reused as-is, because it goes
through `planner deploy` and **there is no furiosa deploy backend** —
`planner/deploy/` holds `vllm_cuda`, an `vllm_ascend` stub and `kubernetes`.
Writing one is Phase 4 and out of S7's scope.

**How we adapt.** The open-loop *protocol* is added to `measure_envelope.py`,
which already owns everything the route needs and owns it per backend: the launch
line (`server_command`), NUMA binding of both server and client (`numa_prefix`,
default `auto`), the power sampler, the settle/idle windows and teardown by
process group. `--mode open` replaces one thing — the load generator —
substituting `replay_to_endpoint.py --open-loop --target-rps R` for
`bench_furiosa_endpoint.py --concurrency N`. `--mode closed` is the default, so
every committed invocation of the script runs unchanged, and the 18 pre-existing
tests pass untouched.

Three consequences, each of which is the point rather than a side effect:

* **`--numa-bind` came for free**, already implemented and already defaulting to
  `auto`. `docs/HANDOVER.md` §2.10.1 item 2 asked for the furiosa open-loop path
  to carry it "from the first commit" precisely so the card would not have to be
  measured twice; putting the protocol in the harness that already had it
  satisfies that by construction instead of by discipline.
* **Both backends gained the open-loop server route at once**, because the mode
  adds no vendor knowledge. A future A40 point on this route is the same command
  with `--backend cuda`.
* **The saturation test is a different quantity and has a different name.** A5(b)
  is a closed-loop rule; `measure_envelope_openloop.py` replaced its purpose with
  `d(scheduled_ts − queued_ts)/d(arrival)`, read from the engine's own
  timestamps. A server over HTTP does not expose those, so `ttft_drift_slope`
  reads client-side TTFT instead — queue wait **plus prefill plus transport**.
  The intercept is therefore not comparable with the bench-run route's; the slope
  is, and the threshold (`QUEUE_GROWTH_SLOPE = 0.05`) is deliberately shared.
  A large but flat TTFT must not read as saturation, and
  `test_a_growing_queue_is_caught_by_the_slope_not_by_the_ttft_value` pins that.

**What this means for the refit S7.3 will take.** The RNGD open-loop domain will
be comparable with `openloop/a40.accuracy.openloop.yaml`, not with
`a40.accuracy.yaml` — same protocol *and* same harness. It therefore inherits
that file's measured caveat: over HTTP, TPOT is transport-free (client 36.64 ms
against the engine's 36.62, +0.06 %) but **TTFT carries +20.26 ms of client-side
transport**, +9.5 % at that load, and the values are recorded raw with the offset
not subtracted. A consumer comparing an RNGD open-loop TTFT error against an
engine-side number must account for it. `launch_error_ms`, which
`replay_to_endpoint` already records, is what says whether the client kept up.

**A third route now exists and an artifact says which one produced it.**
`envelope.json`'s run block carries `protocol` and `harness`, and so does every
point. `closed_loop: true` used to be a hardcoded literal in that block — correct
while the script had one mode, and it would have mislabelled every open-loop
artifact.

**Where.** `experiments/scripts/measure_envelope.py` (`--mode`, `--rps`,
`--num-reqs`, `ttft_drift_slope`, `summarise_openloop_point`,
`openloop_client_command`, `run_point_open`, the run block's `protocol` /
`harness` / `closed_loop`), `tests/test_measure_envelope.py` (14 new tests),
`WORK_ORDER_domain_scoping.md` §S7.2 (amended to name this file),
`docs/HANDOVER.md` §2.10.1 items 1 and 2.

## D80 — the provenance block recorded how many accelerators, never which ones · Recorded 2026-09-21

*First entry from the `D80–D89` block (one-off work with no work order). Read the
*Which machine am I on?* section of `CLAUDE.md` first — this is the same failure
one level down.*

**What the block said, and what it could not say.** `provenance.accelerators()`
probed for hardware and recorded counts: `{"cuda": null, "rngd_cards": 3,
"atom_devices": 4}`. That was written to close a real gap — before it, an
artifact recorded only `hostname` and `cpu_count`, and every node of this project
reports `s8` (the NPU node now reports `etri-001`), so an A40-node artifact was
separated from an NPU-node one by an incidental 64-vs-96 core count.

It closed the gap it aimed at and left a second one: **counts do not identify a
machine.** Two RNGD nodes with three cards each produce a byte-identical
provenance block. `scripts/whichnode.sh` cannot separate them either — it sets
`NODE="npu"` for *any* box with an RNGD or ATOM device, by design, so a second
RNGD machine is detected as `npu` and pointed at `docs/nodes/npu.md`, whose
inventory, BDFs, NUMA placement and tenant list are assertions about **this**
machine.

**Why it stopped being harmless.** While one machine of each kind existed, a node
kind and a machine were the same thing. The moment a second RNGD node is used —
which is now planned — they are not. Concretely: S7.3's open-loop domain has six
points measured on this machine's `npu0`. A point measured on another machine's
card would append to the same file, under the same `hardware: RNGD-CARD` label,
with nothing in either artifact to tell them apart and nothing recoverable
afterwards. Different silicon, possibly different firmware and NUMA topology,
one interpolation axis — the error D22 was, arriving through the hardware door.

**How we adapt.** `accelerators()` gains `accelerator_set`, which records the
durable per-device identity the vendor tools already expose and hashes it:

| class | identity | source |
| --- | --- | --- |
| RNGD | `device_sn`, `device_uuid`, BDF, firmware | `furiosa-smi info --format json` |
| ATOM | `sid`, `uuid`, BDF | `rbln-smi -j` |
| CUDA | GPU UUID | `nvidia-smi -L`, already printed beside the model |

The **serial** is the identity, not the label or the BDF: `npuN` re-enumerates
(the card at `45:00.0` was `npu2` on 2026-09-17 and `npu3` on 2026-09-18) and a
BDF moves with the slot.

`fingerprint` is a 12-hex hash over the sorted serials. **It identifies the
accelerator SET, not the chassis**, and that is the useful thing rather than a
compromise: two artifacts agreeing on it were produced on the same physical
cards, which is the question a calibration actually asks. It therefore changes
when a card is added or removed — this node's RNGD count has gone 4 → 3 → 4 → 3
— and that is correct, because a different card set is a different measurement
configuration. This machine is **`6fe246ed1abf`**.

**Backward compatibility is deliberate.** `rngd_cards` and `atom_devices` keep
their integer shape and meaning; the identity is added beside them, never in
place of them, so the committed artifacts that carry `"rngd_cards": 3` stay
comparable. A missing vendor tool records `None` — "not detectable here" — and
never an absence of hardware, and a tool that prints something unparsable is a
failed probe rather than an exception while an artifact is being written.

**Where.** `planner/util/provenance.py` (`accelerator_set`, `_rngd_ids`,
`_atom_ids`, `_cuda_ids`), `scripts/whichnode.sh` (prints `accel serials` and
says the node kind does not identify a machine), `docs/nodes/npu.md` (states the
serials it describes and tells a reader on different hardware to stop),
`tests/test_provenance_machine_identity.py`.

**What this does not do.** It does not give the second machine a node doc — that
is written when someone works there. It does not retrofit identity onto existing
artifacts, which remain counts-only; they are all from this machine, but they say
so only by being older than this entry.
## D115 — the open-loop refit is fitted on p99, and that makes two bases live in `profiles/calibration/` at once · Recorded 2026-09-18

*`WORK_ORDER_domain_scoping.md` STEP S7.3, the sixth entry from the `D110–D119`
block. Read D113 first (why an open-loop refit became necessary at all), then
D101 (the p50/p99 mismatch this departs from) and D102 (why a second harness for
one hardware label needs a second file). D114 is the harness this uses.*

**Why a refit exists.** S4 stated `arrival_process: closed_loop` on
`rngd_card_edf.yaml` (D113). `python -m planner plan` always replays an arrival
trace, so under the default `condition_mismatch: refuse` **no RNGD candidate may
consult that domain** — 72 of 72 rows in S7.0's sweep were held on that one
field. A wider load range does not answer them; a measurement under the arrival
process the planner performs does. That is a new domain file, not more points on
the committed one: two protocols on one interpolation axis is the class of error
D22 was, and `a40.accuracy.yaml`'s own header is the precedent for splitting
instead of extending. No committed `points` value is touched (rule A3).

**The decision this entry records, and it is a departure.** Every domain
committed before today is fitted on **p50** — `lowload_sim_error.py` writes
`compared_metric: tpot_p50` — while the feasibility check applies the margin to a
**p99** (D101). D101 declined to re-fit, for a blast radius that has not
shrunk. S7.3's file is new, so it inherits nothing, and it is fitted on **p99
against p99**, because that is the basis the margin is charged on and the basis
S7.4's verdicts are read at. Fitting on p50 and applying to p99 would carry
D101's known wart into the one file the disclosure's §6 numbers depend on.

**So the repository now holds both bases, deliberately** (user decision,
2026-09-18). Three things keep that from becoming a silent trap:

* **`AccuracyDomain.compared_metric`** states the basis. `""` means **not
  stated**, which is how every older file reads — *not* p50-by-default, because
  stating it for them is a migration with its own measurement question.
* **A parallel `.p50.yaml` sibling** is emitted from the same measurement, so
  this refit can still be compared against `rngd_card_edf.yaml` on the basis that
  file uses. It is comparison material and is **not** on the loader's default
  glob path — D102's guard raises on two domains for one hardware label, so only
  the p99 file is loadable by default and the p50 one is loaded explicitly or not
  at all.
* **`AccuracyPoint.tpot_err_pct_p50`** carries the same point's p50 error beside
  the fitted p99 one, so a reader sees both bases without opening two files. It is
  recorded and never consulted: `_err_at` interpolates `tpot_err_pct`, whatever
  the domain declares.

**The margin policy warns rather than refusing.** `AccuracyDomainMargin` appends
to its basis when a domain explicitly declares `tpot_p50`, modelled on the
closed-loop warning beside it. A refusal would be wrong — a p50-fitted domain is
still a real measurement of that hardware at that operating point, and throwing
it away buys nothing. Silence earns no warning, for the reason above: it would
fire on every run against every committed domain and tell nobody anything new.

**Migration is out of scope and is left as a step.** `lowload_sim_error.py` grew
`--compare-stat p50|p99|both` so the migration has a tool; its default is `p50`,
so every committed invocation writes the artifact it wrote before, and
`tests/test_lowload_sim_error.py` pins that the default path is unchanged. The
envelope already carries `tpot_p99` on every point, so the p99 reference is a
measurement rather than an interpolation of a different statistic — a refusal
names which statistic was missing so a reader can tell "no p99 here" from
"nothing here".

**p99 costs sample size, and the protocol says so** (user instruction). A
300-request run puts roughly three observations above its p99, so
`openloop_sim_error.py --min-requests-p99` refuses to fit one on less. A5(b)'s
`pool >= 4x` does not transfer — an open loop has no pool — and its purpose is
served by the saturation slope D114's harness records. The empirical check on
stability is the run-to-run spread across repeats, which every point carries and
which S7.0 requires beside any verdict. A point whose repeats fall short is
reported with its p50 error and **excluded from the p99 file** rather than
dropped: making `tpot_err_pct` nullable was the alternative and was not taken,
because three committed domains depend on that field being present and the p50
sibling already carries the number.

**Pairing, and the guard that decides whether a point exists at all.** The same
offered rate goes to both sides, the committed `--match offered` convention.
Served concurrency is then an outcome on both, and a pair whose two
concurrencies differ by more than `--max-conc-gap` (default 20 %) is **flagged
and excluded** — the simulator's latency there is a different operating point's
latency, which is the failure `rngd_card_edf.yaml` records above c29.3 where the
sim settled at 37.67 and 44.47 against measured 59.2 and 107.2. The x axis stays
the **simulator's** served concurrency, because that is what the planner knows
when it consults the table.

**Where.** `experiments/scripts/openloop_sim_error.py` (new),
`experiments/scripts/lowload_sim_error.py` (`--compare-stat`, both percentiles on
the sim side), `planner/predictor/calibration.py`
(`AccuracyDomain.compared_metric`, `AccuracyPoint.tpot_err_pct_p50`),
`planner/optimizer/margin.py` (the p50 warning),
`profiles/calibration/openloop/rngd_card.accuracy.openloop{,.p50}.yaml` (new),
`profiles/calibration/index.yaml` (regenerated),
`tests/{test_calibration_condition,test_lowload_sim_error}.py`, D101's addendum.

## D116 — `nccl-tests` is not a second opinion on NCCL, so S6(ii) closes with a peer-copy control instead · Recorded 2026-09-21

*`WORK_ORDER_domain_scoping.md` STEP S6, the seventh entry from the `D110–D119`
block. Read D112 first — this measures the wire whose three numbers D112 keyed
by traffic. A40 node, accelerator set `83e3434d7696`. Full record:
`experiments/p2_evidence/results/s6_link_binding_and_p2p_control.md`.*

**What the work order asked for.** S6(ii) names `nccl-tests all_reduce_perf` and
`p2pBandwidthLatencyTest`, bound and unbound, filed in S3's schema. S3 filed its
figures from `experiments/p2_evidence/link_probe.py` (torch), so the vendor-tool
half stayed open and the handover carried it as such.

**It is not run, for two reasons, and the second is the one that matters.**

1. **It cannot run here, re-verified rather than inherited.** The only build on
   this host needs `GLIBC_2.34`; this host is `2.31`. It also fails to find
   `libnccl.so.2`. There is no `nvcc` and no system NCCL header, so rebuilding
   it means installing a CUDA toolchain. `p2pBandwidthLatencyTest` is absent
   entirely — no CUDA samples on this host.
2. **It would not have been a control.** `nccl-tests` links the same NCCL
   2.27.5 that `link_probe.py` reaches through torch. Two front ends onto one
   implementation agree by construction. What S6(ii) wanted was a second opinion
   on the number; what it would have got is a second opinion on the harness.

**What is done instead: the wire measured with NCCL out of the path.** A direct
`cudaMemcpyPeer` under `Tensor.copy_` (`experiments/p2_evidence/p2p_probe.py`),
at 64/256/512 MiB, one way and round trip, median of 20 with the spread. Across
the PCIe bridge it gives **25.15 GB/s**, against **19.3** for the two-rank
all-reduce and **8.8** for the four-rank one over the same wire, and a datasheet
**64.0**. Round trip agrees with one way to 0.04 %, so the link is symmetric per
direction and the legs serialise.

**That changes how D112's three numbers should be read.** They are not three
opinions about a link's speed: the wire is one number, 25.15, and the collective
gets **35 %** of it at four ranks and 77 % at two. 8.8 is a property of the ring
on that path, not of the path. The datasheet 64.0 is 2.5× the wire under any
traffic at all.

**And the bound half turned out to be the cheap half.** All ten link cases move
by at most **0.66 %** between `unpinned` and `numa_pinned`, in inconsistent
directions — while the same binding is worth **1.93×** of deployment throughput
on this node (`v3_verdict_accuracy.md` A.3). Both are true: NUMA buys the host
memory path — prefill and queueing — and not the GPU-to-GPU wire. A deployment
measured unbound is therefore not a link measured wrongly, which is what made
S3's `unpinned` figures safe to have filed.

**What is filed, and what is not.** Six `numa_pinned` entries on the two PCIe
links of `pd-rngd-gpu-card-measured-pcie.yaml`, appended after the unpinned ones
so the planner's current selection is unchanged (`measurement_for` prefers the
first stated-binding match when the node states `unknown`; both nodes state
`unpinned`, so the bound entries are filtered out for them). Datasheet values
untouched (A3). **The NVLink links were measured and deliberately not filed**:
39.2–39.3 against a datasheet 112.5 would move `island_interconnect`'s `min` for
every tp=2 candidate inside an NV4 pair, which is a planner behaviour change and
not what S6 was scoped to do. The figures are in the result document.

**Where.** `experiments/p2_evidence/{p2p_probe.py,run_link_probe_numa.sh}` (new),
`experiments/configs/clusters/pd-rngd-gpu-card-measured-pcie.yaml` (six entries
appended), `experiments/p2_evidence/results/s6_link_binding_and_p2p_control.md`
(new), `tests/test_calibration_condition.py` (§(vii), the committed A40 domain
against V3's real P1), `outputs/p2_evidence/{link_numa,p2p_control}/`.

---

## D121 — `PredictedMetrics.offered_requests` changes the metrics schema digest, so every envelope cache entry misses · Recorded 2026-09-22

*`WORK_ORDER_graph_search.md` STEP H1, the first entry from the `D120–D129`
block claimed by that work order, which lives in `swsok/heteropilot-graphsearch`.
When this entry was written a staged copy sat at `graphsearch/` in this repo; it
was removed on 2026-09-23 once G0 had committed the originals there, so the
three documents have one home. A5000 node, accelerator set `GPU-bd2a06dc`. This
entry records a cache consequence, not a disagreement with upstream.*

**What H1 added.** The service contract grew three optional SLO fields
(`min_goodput_rps`, `min_completion_ratio`, `observation_window_s`), a fourth
objective (`minimize_cost_per_hour`), a price on a plan (`cost_per_hour_usd`,
`cost_basis`), and one field on the metrics — `offered_requests`, the
denominator `completed_requests` is a ratio of. Everything defaults to None and
`_write_output` drops the three new keys while they are, so a flagless
`python -m planner plan` emits the same YAML it did before.

**The cache is the exception, and it is deliberate.** `envelope._METRICS_SCHEMA`
is a digest over `sorted(PredictedMetrics.model_fields)`, stored inside every
entry, and an entry whose stored digest differs from the running one is a miss.
Adding a field therefore invalidates every cached simulation at once:

    pre-H1   44b33e454ef8d016
    post-H1  aefae37de6a24695

That is the mechanism working. The digest exists because of what happened in
uncertainty STEP B2 — E-A1 re-ran to byte-identical numbers off entries written
before `served_concurrency_per_island` existed, which reads as "the change had
no effect" when it means "the change was never applied". An experiment run
against a warm pre-H1 cache must refill it; `outputs/uncertainty/ea1/cache`,
`outputs/perf/topk/cache_*` and `outputs/.hp-envelope` are all affected.

**Why a throughput floor had to come first.** `candidate_generator` once carried
a throughput lower bound and it was removed, because §5.6 declared no throughput
constraint for it to relax and the oracle-agreement test caught the
disagreement. The comment left at `planner/candidate_generator.py` says
restoring it requires feasibility to declare one. `feasibility.check_throughput`
is that declaration, so the graph-search bound in `graphsearch/bounds.py` and
the final feasibility test now read the same field. Until a spec sets
`min_goodput_rps` the check is a no-op and the declared constraint set is
unchanged — the bound stays off with it.

**`offered_requests` is None, not zero, when unknown.** A predictor with no
per-request records cannot say how many requests were offered, and
`check_throughput` then reports the completion-ratio constraint as unchecked
rather than satisfied. That is D2's rule for an unmeasurable constraint, applied
to a second one. `LLMServingSimPredictor` fills it with the CSV row count, which
equals `completed_requests` there — every admitted request finishes — so the
ratio is 1.0 and the field's value on that path is that the count is *known*.

**The work order put the block claim in H3; the guard puts it here.** STEP H3's
change list is where `WORK_ORDER_graph_search.md` says the `D120–D129` row lands
in `CLAUDE.md`. But `tests/test_deviations_numbering.py` fails on the *first*
entry taken from a block CLAUDE.md does not yet assign, and that entry is this
one — so the row had to be written in H1 or D121 could not be recorded at all.
The row is therefore already present when H3 runs; H3 still owns the experiment
tag `E-G*` and the document-table line. The real code wins, as ever.

**Where.** `planner/spec.py`, `planner/plan.py`,
`planner/optimizer/{feasibility,pareto}.py`, `planner/__main__.py`,
`planner/predictor/llmservingsim.py`, `tests/test_spec_contract.py` (new),
`CLAUDE.md` (the block row).

---

## D120 — `bandwidth_gbps` stays GB/s, and the v2 schema is documented in a fork file because upstream's page describes the other layer · Recorded 2026-09-22

*`WORK_ORDER_graph_search.md` STEP H2, the second entry from the `D120–D129`
block. Read D121 first for what H1 established. A5000 node, accelerator set
`GPU-bd2a06dc`.*

**What H2 added.** `ClusterSpecV2.schema_version`, defaulting to 1, and a set of
fields that exist only at 2: CPU sockets, PCIe switches and cluster-scoped
network switches as vertices; `shared_resources` with a capacity and an external
reservation; per-device and per-host prices; `runtime_capabilities` on a
profile; and `bandwidth_unit`, `direction`, `rdma`, `p2p`, `shared_resource` on
a link. Every committed cluster is v1 and reads exactly as it did.

**The unit does not move.** `Link.bandwidth_gbps` is named for bits and has
always held **GB/s** (`planner/topology.py`). v2 could have taken the
opportunity to fix the name; it does not. A v1 file setting `bandwidth_unit` to
anything but `GB/s` is refused outright, because the failure mode is silent: a
committed `64` reinterpreted as Gbit/s becomes `8`, every bound that reads it
moves by 8x, and nothing in the output says a unit changed. The name stays wrong
and the number stays right.

**A v2 field in a v1 file is an error, not an ignored key.** `_Strict` already
rejects unknown keys, but these keys are not unknown - they exist on the model
and pydantic would happily accept them, leaving a file that looks honoured and
is not. `ClusterSpecV2._consistent` compares each v2 field against its default
and names the first one set. This is the same class of failure as an entry
served from a stale cache (D121) and an unmeasurable constraint reading as
satisfied (D2): what makes it dangerous is that it is quiet.

**The work order's documentation target was the wrong file, and this is the
divergence.** STEP H2 says to add a v2 section to
`docs/docs/reference/cluster-config.md`. That page documents the **legacy JSON**
passed to `--cluster-config` - `num_nodes`, `link_bw`, `instances[]` - which is
the layer the planner *compiles down to*, not the layer `schema_version` belongs
to. It is also upstream's Docusaurus site (migrated in upstream `b55c52e`) with
no fork content, the situation D35 records for `README.md` and `CHANGELOG.md`.
Writing a `ClusterSpecV2` section there would document one schema on another
schema's reference page, in a file the fork does not own.

There was no fork-owned reference for the upper layer to extend either: the
`ClusterSpecV2` fields are documented by their docstrings, and `device_binding`
(D110) went into `docs/uncertainty_planner.md` because that is what consumed it.
So H2 writes **`docs/cluster_spec_v2.md`**, which says in its first paragraph
which of the two layers it describes and points at the other two pages for the
lower one. Upstream's page is untouched.

**A net_switch has no node, and that is load-bearing.** Every other endpoint is
`<node>/<device>`; a switch endpoint is a bare id. `Link.endpoints` reports it as
node `""`, and the intra-node callers (`_intra_node_adjacency`, `_dominant_link`)
compare that against a real node id, so a switch can never be read as a device
on some node. An empty node id is rejected to keep the sentinel sound. The v1
requirement of exactly one slash therefore moved out of `Link._endpoint_format`,
which cannot see its cluster's version, into `ClusterSpecV2._consistent`, which
can - a bare endpoint in a v1 file is still refused, with a message that says
`schema_version 2` is what it would need.

**A price with no source is refused** (`AcceleratorProfile.price_source`), the
rule-3 treatment a datasheet already gets. Unpriced stays None rather than zero,
so H1's `MINIMIZE_COST_PER_HOUR` declines to score the plan instead of ranking
an under-priced one cheapest.

**Where.** `planner/inventory.py`, `tests/test_inventory_v2.py` (new),
`tests/data/cluster_v2_min.yaml` (new), `docs/cluster_spec_v2.md` (new).

---

## D122 — the graph mode turns the generator's bounds OFF and applies its own; the default path keeps stages 4–5 · Recorded 2026-09-22

*`WORK_ORDER_graph_search.md` STEP H3, from the `D120–D129` block. Read D121 and
D120 first. A5000 node, accelerator set `GPU-bd2a06dc`.*

**The arrangement.** `swsok/heteropilot-graphsearch` calls
`CandidateGenerator(..., enable_bound_pruning=False)` and takes the surviving
candidates as *templates*, then applies its own cut-capacity bounds in
`graphsearch/bounds.py` after expanding each template into physical embeddings.
Stages 4 and 5 are not modified and not removed: `python -m planner plan` runs
them exactly as before, and `enable_bound_pruning=False` is the flag oracle mode
has always used.

**Why the graph mode cannot reuse them.** Stage 4 reasons about an island's
representative interconnect. Two embeddings of one template can place the same
TP group on two GPUs inside a node or across two nodes sharing a saturated
uplink, and the stage cannot tell them apart because a `CandidateConfig` names
islands, not devices. A bound that cannot see the difference is not wrong — it
is a bound on the template, and it stays sound for the path that uses templates.
It is simply not the bound the graph mode needs, which is over the *cut* between
the ranks of a specific placement, minus whatever an external reservation holds.

**Both must stay relaxations.** The rule is unchanged and it binds the new code
too: a stage may reject only when the most optimistic arithmetic already misses
a constraint §5.6 declares. The graph side's throughput bound is the one to
watch, because it exists only now that H1 added `slo.min_goodput_rps` — with the
field unset it must not run at all. `graphsearch`'s oracle-agreement harness is
what checks this, the same way `tests/test_search.py` checks it here.

---

## D123 — networkx is a graphsearch dependency; this repo keeps its hand-rolled BFS · Recorded 2026-09-22

*STEP H3, `D120–D129`.*

`planner/topology.py` walks the cluster with its own BFS and says so
(`path_aware=False`, `contention_modeled=False`). The graph search needs
max-flow over a cut, subgraph isomorphism (VF2) and Weisfeiler–Lehman hashing,
which are not worth hand-rolling and are exactly what networkx provides.

**It is not added here.** `pyproject.toml` is untouched, `planner/` imports
nothing new, and a checkout of this repo installs the same set it always did.
The dependency lives in `heteropilot-graphsearch`, whose `Signature.tool_version`
records the version it ran with, because a WL hash is only comparable against
itself: the same graph hashed by two networkx releases may differ, and an
equivalence class silently re-cut by a library upgrade would be invisible.

---

## D124 — the MVP adapter cannot hand shared resources to the simulator, so it reports what it dropped · Recorded 2026-09-22

*STEP H3, `D120–D129`. The extension of D3 the graph work runs into.*

D3 recorded that the legacy cluster config carries no topology graph. The
consequence for graph search is sharper than for the planner: two placements
that differ *only* in whether they cross a shared uplink compile to the **same**
simulator input, so the simulator's prediction cannot distinguish them, and a
difference the equivalence layer was careful to preserve is lost at the last
step.

The MVP does not fix this — a flow-level contention model is out of scope and
`graphsearch/contention.py` ships an interface with a null implementation. What
it does is refuse to lose the fact quietly: `compile_embedded` returns a
`TopologyLossReport` naming every shared resource the config could not express,
and a caveat travels with any plan whose report is non-empty.

**What this costs a reader.** A `graphsearch` result comparing two such
representatives is comparing their *bounds and their cost*, not their simulated
performance, because the simulator gave both the same answer. That is a real
limit on what the first paper can claim from simulation alone, and it is why the
contention experiment is listed as work that follows a `ContentionModel`
implementation rather than as something the MVP measures.

**Where (D122–D124).** `planner/plan.py` (three `RejectionStage` values),
`planner/optimizer/exhaustive.py` (`plan_id_base`, `_assemble_output`),
`planner/envelope.py` (`graph_signature`, `with_graph_signature`),
`planner/predictor/llmservingsim.py` (`set_compile_hook`),
`tests/test_search_hooks.py` (new), `CLAUDE.md` (the `E-G*` row and the document
table line). The consumers named above are in `swsok/heteropilot-graphsearch`
and none of them exists yet.
