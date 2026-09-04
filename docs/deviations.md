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
| D22 | Phase 4 | **Resolved** (retraction + measurement) — the c1–c32 curve's top point was a 24-request pool running at eff 21.2, not c32; envelope measured to eff 107.2. At eff 76 the simulator is 1.31× optimistic on throughput and 18 % on TPOT. The re-run is **done for the loose-TTFT regime**: with the measured 18 % margin every RNGD config is rejected on both fixtures and the winner becomes `agg[cuda:tp4]` at 2.595 tok/J — the committed winner is infeasible, not merely optimistic. Tight-TTFT rows still open (timeout artifact) |
| D24 | — | **Resolved** — the work order's layout lists `profiles/networks/`, but Level-1 interconnect-class values live inline in `planner/topology.py` and the YAMLs were an unread duplicate read only by ScenarioLab's cluster generator. Moved out with it (STEP 3.3); recoverable if Phase 5 ever wants them as data |

---

## D16 — `LinkType` has no on-package fabric, and cross-vendor P/D needs a shared TP degree · Resolved (one added type) + Open (the TP constraint)

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
it. **Any "heterogeneous P/D does not pay" statement from this repo is therefore
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

## D22 — The RNGD scaling curve's top point was request-pool-limited, and the envelope beyond it is now measured · Resolved (retraction + measurement)

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
