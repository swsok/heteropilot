# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **HeteroPilot** — a fork of `casys-kaist/LLMServingSim` that adds a control plane for LLM
serving on heterogeneous GPU/NPU clusters. Everything outside `planner/`, `profiles/`,
`experiments/`, `examples/`, `tests/`, `profiler/synth/` and `profiler/contract.py` is upstream
simulator code.

**ScenarioLab moved out** on 2026-09-03 (`WORK_ORDER_consolidation.md` STEP 3): the batch scenario
explorer and its planning workspace now live in `swsok/heteropilot-scenariolab`, which pins this
repo as a submodule at `e79ac4ab`. It imports from `planner/` and never the other way round, so
nothing here depends on it. `profiles/networks/` and `experiments/configs/lab/` went with it — see
`docs/deviations.md` D24 for why that does not contradict the work order's layout.

Authoritative documents, all of which outrank this file:

| Document | Covers | Note |
| --- | --- | --- |
| `AGENTS.md` | Upstream simulator internals, repo layout, code style, architecture patterns | Upstream file — do not edit |
| `WORK_ORDER_heteropilot.md` | HeteroPilot schemas, module contracts, phase gates (Korean, v1.0) | The HeteroPilot spec |
| `docs/phase0_formats.md` | Real CSV / stdout / cluster-config schemas, verified at the pin | Basis for the §5.5 compiler and parser |
| `docs/deviations.md` | Where the work order and upstream disagree, and how we adapt | Read before implementing any Phase 2+ module |
| `docs/rps_aware_planning_design.md` | Design for RPS-dependent GPU/NPU selection: performance envelopes, operating-point solving, accuracy domains | Proposal, not built; read before adding any RPS or concurrency axis |
| `docs/phase0_bench_plan.md` | What was measured vs simulated, and what a node can actually run | Provenance discipline |
| `WORK_ORDER_tiered_profiles.md` | Tier 0/1 synthetic profiles: datasheet schema, roofline generator, attention cost model, calibration | Executed; closed D4 |
| `docs/tier0_calibration.md` | E1–E4 — what a generated profile can and cannot be trusted to rank | Read before quoting any `profile_tier: analytical` plan |
| `WORK_ORDER_consolidation.md` | The 2026-09-03/04 sprint that merged the three tracks and split ScenarioLab out | Subordinate to the three above it |
| `WORK_ORDER_rps_aware.md` | RPS-aware planning: measured envelopes, accuracy domains, operating-point margins, `plan --rps`, E5/E6 | Executed (STEP 0–6, PRs #60–#77); supports patent 1 |
| `WORK_ORDER_uncertainty_planner.md` | Patent 2: uncertain-input registry, per-candidate margin policies, closed-form perturbation, sensitivity, measurement plan (Korean) | Stage A/B; read `docs/deviations.md` D33 first — its accuracy domain was reconciled onto the rps one |
| `docs/nodes/{a40,a5000,npu}.md` | Per-node inventory, topology and traps | Read the one `scripts/whichnode.sh` names — never all three |
| `docs/CLAIMS.md` | What can be claimed today, with each claim's label and artifact — Established / Not established / Retracted | Read before writing any result into a paper or deck |
| `docs/uncertainty_planner.md` | `--accuracy-domain` / `--measurement-plan`: per-candidate margins, the `unmeasured` verdict, closed-form perturbation, the measurement queue — design, CLI, results, limits | Read before touching `planner/uncertainty/` or any accuracy domain |
| `docs/HANDOVER.md` | Current state, next work by node, traps that cost a session | The live handover; the `HANDOVER_*.md` files are historical |
| `graphsearch/` | Graph-based placement search: research design, software design, work order | Staged here; the implementation repo is `swsok/heteropilot-graphsearch`, which pins this one as a submodule and imports `planner.*` one-way. Only the hook PRs H1–H3 land here — see `docs/cluster_spec_v2.md` and D120–D124 |

Upstream ships `CLAUDE.md` as a symlink to `AGENTS.md`. This fork replaces it with a real file;
read `AGENTS.md` directly for anything about the simulator itself.

**Upstream baseline is pinned** in `UPSTREAM_COMMIT` (`2c2042ce`, plus the `astra-sim` submodule at
`f82fb3d`). Do not silently rebase onto a newer upstream — re-pin deliberately and record it.

## What HeteroPilot does

Given a `ServiceSpec` (model + traffic distribution + TTFT/TPOT SLOs + power cap) and a
`ClusterSpecV2` (accelerator inventory + topology graph), the planner enumerates deployment
candidates, predicts each one's performance and energy via LLMServingSim, and emits a
`DeploymentPlan` that maximizes SLO-goodput/J under a power cap — plus Pareto alternatives.

Four planes: **Control** (`planner/`, the new work), **Simulation** (upstream `serving/` +
ASTRA-Sim), **Data** (real vLLM CUDA / vLLM-Ascend instances), **Profiling** (`profiler/`, `bench/`).

### Execution Island — the central abstraction

An *execution island* is a set of accelerators sharing one runtime backend (`cuda`, `ascend`),
mutually reachable by collectives, supporting the target model's kernels, and able to host one vLLM
engine. TP/PP are permitted **only within** an island. Heterogeneity is exploited at replica or
Prefill/Decode-role granularity, never inside a TP group.

Island id convention: `{backend}-{model_slug}-{node_id}` (e.g. `cuda-h100-node0`).

## Absolute rules

1. **Do not modify upstream code** (`serving/`, `profiler/`, `bench/`, `configs/`, `astra-sim/`,
   `AGENTS.md`) before Phase 5. Per-file exceptions unlock at specific phases — work order §7.
   An early fix to `serving/core/memory_model.py` / `scheduler.py` was authorized and attempted
   for D12, but **both attempts were wrong and have been reverted**.
   Read `docs/deviations.md` D12 before trying again; it records what was tried and why it failed.
   **`serving/` is no longer pristine.** Five edits are sanctioned and each is recorded with a
   byte-identical regression proof: **D15** (opt-in P/D KV-transfer cost, `router.py` +
   `__main__.py`), **D25** (`cwd=run_paths.inputs_root` on the ASTRA-Sim `Popen`), **D26**
   (`sys.executable` for the Chakra converter — this one is what D23 actually was), **D27**
   (that converter called in-process instead of spawned, 1.55–1.72× faster, D26 subsumed), and
   **D28** (`topology_mode: slab3d` in `config_builder.py` + the `_FMT` comm_type column in
   `utils.py`, for asymmetric TP per phase). Nothing else in `serving/` may change without a
   work order that names the file.
2. **Never mix backends in one TP group.** Candidate generation must exclude such configs automatically.
3. **Never invent hardware numbers.** Values with no measurement get `source: placeholder` in the
   profile file. Never label unmeasured data as measured, and never claim results from hardware
   that isn't present — run `scripts/whichnode.sh` to find out what is (see
   *Which machine am I on?* below).
4. **First optimizer is exhaustive enumeration + pruning, not RL.** RL, Kubernetes operators,
   cross-vendor TP, and live migration are out of scope.
5. **Never delete `planner/optimizer/exhaustive.py`.** It is the oracle that detects pruning bugs
   and separates surrogate error from search error.
6. Every result file records the work order §3.8 provenance metadata (git commits, versions, spec
   hashes, seed, full command line) via `planner/util/provenance.py`.
7. One feature = one branch = one PR (`feat/service-spec`, `feat/candidate-generator`, …).
   **Check `base == main` before merging.** A stacked PR whose base branch has already
   reached `main` merges into a branch nothing merges from again, so it reads MERGED on
   GitHub and is absent from `main`. This has happened three times — #64–#71, #110/#111
   (fixed by #112), #114 (fixed by #116). Verify with
   `git rev-list --count origin/main..<branch>` returning 0 after the merge, never the PR
   list, which says MERGED either way.
   *"Automatically delete head branches" is on and covers the common case*: deleting a
   merged head branch makes GitHub retarget any open PR based on it to the parent's base,
   so merging the parent first now rescues the child. It does not cover a child merged
   **before** its parent — that still lands in a branch which later disappears — so the
   check stays.
8. **Code comments, docstrings, and log messages in English only** — an upstream convention from
   `AGENTS.md`. The work order is Korean; the code it produces is not.

### Out of scope — stop and report to the user if a task seems to require it

GPU+NPU mixed TP · dynamic migration · multi-tenancy · RL · Kubernetes operator ·
cross-vendor P/D before Phase 5 · full switch-level congestion modeling.

## Which machine am I on? — detect it, never assume

**This repository moves between three nodes: an A40 node, an A5000 node and an NPU
node.** Any statement in a committed file about what hardware is present is true on
at most one of them. This section used to *be* such a statement, and it is why a
session once opened on a box with eight A40s while reading "no NVIDIA GPU at all".

**Hostname does not discriminate.** It has read `s8` on kernel 5.4.0-216-generic
and, as of 2026-09-07 on the NPU node, `etri-001` — so it changes *and* it does not
identify the node. Do not key off `hostname` in either direction: recognising a
familiar one is as misleading as recognising an unfamiliar one. The only incidental
difference in committed provenance is the CPU count (A40 64, NPU 96).

So the first command of a session on an unfamiliar checkout is:

```bash
bash scripts/whichnode.sh
```

It probes the accelerators — `nvidia-smi -L`, `/sys/class/rngd_mgmt`, `/dev/rbln*` —
and prints the detected node, **what can actually be run here** (real vLLM bench,
CUDA profiling, RNGD profiling), and which node profile to read:

| node | profile |
| --- | --- |
| A40 | `docs/nodes/a40.md` |
| A5000 | `docs/nodes/a5000.md` |
| NPU (RNGD / ATOM) | `docs/nodes/npu.md` |

Those files hold the machine-specific detail — inventory, topology, vendor runtime
splits, device-ownership traps — and each one says at the top that it applies only
when the detector names that node.

**Absolute rule 3 extends to hardware presence, not just hardware numbers.** Never
claim a result from hardware the detector does not list. Artifacts measured on
another node stay valid as measurements *of that node*: do not re-run, extend, or
relabel them here. Every result written through `planner/util/provenance.py` now
records the detected accelerator inventory, so an artifact says for itself which
node produced it.

**`npu`, `a40`, `a5000` are node KINDS, not machines.** The detector reports
`npu` for any box with an RNGD or ATOM card, so a second RNGD machine reads the
same and is pointed at the same node doc — whose inventory would be wrong for
it. Since **D80** the provenance block also records each accelerator's serial
and a 12-hex `accelerator_set.fingerprint`, and `whichnode.sh` prints the
serials as `accel serials`. **Compare them against the node doc's stated set
before trusting its inventory**, and never append a measurement taken on one
card set to a domain or envelope built on another.

## Architecture: the planning pipeline

```
ServiceSpec + ClusterSpecV2
  → detect_islands()                      planner/inventory.py
  → candidate enumeration + pruning       planner/candidate_generator.py
  → compile to configs/cluster/*.json     planner/predictor/llmservingsim.py
  → `python -m serving` subprocess → SimResult
  → feasibility (hard constraints)        planner/optimizer/feasibility.py
  → lexicographic ranking + Pareto        planner/optimizer/pareto.py
  → PlannerOutput (recommended + alternatives + rejected_summary)
```

Pruning stages run in **fixed order**, each recording rejection reasons for `rejected_summary`:
backend/model compatibility → memory feasibility → parallelism feasibility → topology lower bound →
analytical perf lower bound → (later) surrogate top-K → full simulation.

Key design decisions that span files:

- **Two schema layers.** The planner reasons over rich `ClusterSpecV2` YAML but *compiles* down to
  the untouched legacy `configs/cluster/*.json` format at simulation time. Read the real schema
  (`docs/docs/reference/cluster-config.md` + actual files in `configs/cluster/`) before writing the
  compiler — never guess it.
- **Optimization is lexicographic, never weighted-sum.** Feasibility → primary objective →
  tie-breaks in fixed order (fewest active accelerators → least fragmentation → lowest
  reconfiguration cost).
- **Infeasible is a diagnosis, not an error.** Emit `closest_plan`, `violated_constraints`, and
  rule-generated `suggestions`.
- **Predictor sits behind a `Predictor` ABC** so the simulator subprocess can be mocked in tests.
- **Percentiles come from one util** (`planner/util/percentile.py`, numpy `linear` interpolation)
  to prevent interpolation mismatches. P50/P95/P99 are the headline metrics, never means.
- **Two-level topology model.** Level 1 (interconnect-class representative values) for bulk
  candidate scoring; Level 2 (actual path + `contention_group` bandwidth sharing) for top-K only,
  from Phase 5.
- **Memory feasibility calls upstream `serving/core/memory_model.py`** rather than reimplementing
  it. If the subprocess boundary forbids importing, copy the formula into `planner/util/memory.py`
  with a source comment.

## Core metrics (use these definitions exactly)

```
energy_efficiency = completed_tokens / total_energy_joule        # tokens/J
SLO attainment    = fraction of requests meeting BOTH TTFT and TPOT SLOs
SLO goodput       = tokens (or requests) completed per second meeting BOTH SLOs
SLO-goodput/J     = tokens of SLO-satisfying requests / total joules
```

Always co-record `J/request`, `J/output-token`, average W, peak W, `SLO-goodput/W`.

## Environment

Built bare-metal in `.venv` (not Docker) so the planner can launch `python -m serving` as a plain
subprocess in Phase 2. Already set up; recreate with:

```bash
# System prerequisites (not pip-installable). `protoc` + the C++ headers are what
# scripts/compile.sh feeds to ASTRA-Sim's CMake; without them the build fails outright.
sudo apt install -y protobuf-compiler libprotobuf-dev
git submodule update --init --recursive      # astra-sim is a submodule; a plain clone leaves it empty

uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install pyyaml pyinstrument transformers datasets msgspec scikit-learn \
  xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 numpy==1.23.5 rich
uv pip install pydantic                                   # planner/ schemas; not optional
uv pip install pytest ruff mypy                            # the §Commands quality gates
bash scripts/compile.sh
uv pip install ./astra-sim/extern/graph_frontend/chakra   # compile.sh's bare pip3 misses the venv
uv pip install "protobuf>=7.35.1"                         # Chakra gencode 7.35.1 needs it
```

Both post-`compile.sh` lines are required, not optional — see `docs/phase0_formats.md` §1.
`transformers` warns that PyTorch is absent; that is expected — `.venv` uses it for tokenizers
only. cmake 4.x builds ASTRA-Sim fine (its `cmake_minimum_required` is 3.22).

## Commands

Upstream simulator (verified working at the pin):

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_power_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/my_run.csv' \
  --log-interval 1.0

scripts/compile.sh        # build ASTRA-Sim + Chakra
python -m profiler        # vLLM layerwise profiler (needs GPU)
python -m bench run       # real vLLM end-to-end (vLLM not installed here)
python -m bench validate  # bench vs sim comparison (works offline against committed artifacts)
```

Use `--run-id` / `--inputs-root` for concurrent candidate evaluation — each run gets an isolated
ASTRA-Sim input root. Since **D25** that isolation is sufficient again: the frontend gives the
ASTRA-Sim child `cwd=run_paths.inputs_root`, so the cwd-relative `tmp__mem/*.json` that no flag
reaches now lands inside the run's own tree. Before D25, concurrent runs raced on one shared copy
and 13 of 64 died at startup while the frontend spun instead of reporting it.
`experiments/scripts/astra_isolated.sh` is kept as a second line of defence, no longer required.

**Run the simulator through `.venv/bin/python`.** Before **D26** `graph_generator.py` invoked the
Chakra converter as bare `python`, and this node has a second `chakra` beside protobuf 6.33.1 —
below the `>=7.35.1` the *Environment* section requires. The same trace then converted to different
`.et` bytes and P/D runs hung with D23's signature: 6 of 6 hung that way against 18 of 18 completing
with the venv first. Since **D27** there is no second interpreter at all — the frontend converts
in-process — so PATH no longer decides anything, but *which venv you launch* now decides everything:
a venv without the right chakra raises at the first conversion instead of converting wrongly.
`tests/test_chakra_interpreter.py` guards the provisioning; `tests/test_chakra_inprocess.py` guards
the bytes and keeps the subprocess from coming back.

Wrap long or parallel runs in `experiments/scripts/livelock_watch.sh` regardless — it ends a
provably stuck run in seconds instead of at the timeout ceiling, and separates a tick stall
(exit 3) from a child that never reported or went quiet (exit 4).

Never write simulator output into `bench/examples/` or over `outputs/example_*` — those are
upstream tracked files, and `bench/examples/run.sh` overwrites them in place. Redirect to
`outputs/phase0_bench/` or a temp dir instead.

HeteroPilot CLI, `python -m planner`. `inspect-cluster`, `plan` and `validate-plan` work today;
`deploy` / `status` arrive in Phase 4.

```bash
# Phase 1 gate (done): islands, TP candidates, compatibility, memory fit
python -m planner inspect-cluster \
  --cluster examples/clusters/heterogeneous-lab.yaml \
  --service examples/service_specs/llama31-8b.yaml

# Phase 2 gate (done): enumerate, simulate, rank
python -m planner plan \
  --service examples/service_specs/llama31-8b.yaml \
  --cluster examples/clusters/heterogeneous-lab.yaml \
  --num-requests 300 --seed 42 \
  --cache-dir outputs/.hp-envelope \
  --output outputs/plans/plan.yaml
#   --oracle   disables bound-based pruning and simulates everything (slow, §5.6)

# Phase 2 gate (MVP): full plan
python -m planner plan \
  --service examples/service_specs/qwen3-32b.yaml \
  --cluster examples/clusters/heterogeneous-lab.yaml \
  --output outputs/plans/qwen3-plan.yaml

python -m planner validate-plan --plan outputs/plans/qwen3-plan.yaml --dataset workloads/...jsonl
python -m planner deploy --plan outputs/plans/qwen3-plan.yaml        # Phase 4
python -m planner status --deployment hp-00042                       # Phase 4
```

`plan` stdout must always include: feasible candidate count + top list, rejected counts by stage
with reasons, recommended plan, Pareto alternatives, predicted metrics.

**Uncertainty-aware planning is opt-in** (`WORK_ORDER_uncertainty_planner.md`;
`docs/uncertainty_planner.md` is the reference). `--accuracy-domain` sizes each candidate's SLO
margin from *its own* served concurrency using the measured domains in `profiles/calibration/`,
instead of one number for every candidate — the simulator's TPOT error on the RNGD card is +11.6 %
at served concurrency 3.9 and −18 % at 76, so one number is too loose somewhere and too tight
everywhere else (D29). A candidate whose operating point no domain covers is rejected as
`outside_calibration_domain`: **unmeasured, not infeasible**, and never offered as `closest_plan`.
`--measurement-plan` (which requires `--accuracy-domain`) then sweeps every uncertain input across
its range and ranks what to measure next by regret removed per hour, `ΔR_i / cost_i`. `measure-apply`
folds a measurement back into a **copy** of the spec and re-plans off the same cache.

**`WORK_ORDER_domain_scoping.md` (D110–D113) amended all of this and the amendments bite.** A domain
now answers only for the **configuration** it was measured under — hardware, `parallelism {tp,pp,dp}`,
`placement {islands, device_binding}`, `arrival_process` — and a candidate that differs is rejected as
`calibration_condition_mismatch`, a **different** refusal from `outside_calibration_domain` because
the two ask for different measurements. `--condition-mismatch {refuse,warn}` is the key and **`refuse`
is the default**: every committed domain is fitted at one island with `dp=1`, and at `tp=1` except
`rngd_perpe.yaml` (tp=8), so on a fixture whose candidates are not, the planner holds most of them
and recommends nothing. On E-A1 that is **312 of 324 held, 0 feasible** — the policy working, not
a broken run (D113), and it is why re-running an
experiment written before S1 needs `--condition-mismatch warn` to reproduce its numbers. An input
with no sourced range now takes its **grade's default** range rather than dropping out of the
ranking (D111), and a `link_bw` item is keyed by the traffic crossing the link — the same wire
measures 25.0 / 19.29 / 8.8 GB/s for a copy, a two-rank and a four-rank all-reduce (D112).

```bash
python -m planner plan ... --accuracy-domain --measurement-plan --budget-hours 8
python -m planner fit-accuracy-domain --real ...json --sim ...csv --hardware RNGD-CARD --out ...yaml
python -m planner measure-apply --plan out.yaml --input link_bw:fabric-rngd0-a40a --value 13.0 \
    --source measured --cluster experiments/configs/clusters/pd-rngd-gpu-card.yaml
python -m planner plan ... --accuracy-domain --condition-mismatch warn   # reproduce a pre-S1 result
```

Without `--accuracy-domain` no automatic margin is applied and the output is byte-identical to the
pre-uncertainty path — the golden-output tests guard it. The uncertainty flags are documented here
and not in `README.md`/`CHANGELOG.md`, which are upstream files with no fork content (D35).

Quality gates required before merging a PR:

```bash
pytest                     # single test: pytest tests/test_inventory.py::test_name
ruff check .
mypy planner/              # planner/ only — upstream code is not type-clean
```

### Never kill by pattern — `pkill -f` matches the shell that runs it

`pkill -f <pattern>` and `pkill -f <script>.sh` match **the invoking bash command
line itself**, because the pattern is in it. Every use so far has killed the
session's own shell (exit 144) and, on 2026-09-09, also SIGTERM'd an unrelated
long-running measurement that had to be restarted. It has additionally produced a
false "STILL ALIVE" report, because the survivor `ps` found was the grep.

**`ps | grep | kill` has the same hole.** The pattern text appears in the
pipeline's own command line, so `ps` lists the invoking shell and the loop kills
it — hit on 2026-09-09 with `ps -eo pid,cmd | awk '/[s]erving/ {print $1}' | kill`,
where the bracket trick does not help because the shell's command line contains
the literal `[s]erving`. Exclude the shell and its children by PID, not by pattern:

```bash
SELF=$$
PIDS=$(ps -eo pid,ppid,cmd | awk -v s="$SELF" '$1!=s && $2!=s' \
       | grep "<a distinctive prefix of the target command>" | awk '{print $1}')
for p in $PIDS; do kill "$p"; done
```

Verify with a second `ps` afterwards; a count that includes the checking pipeline
is not a count of survivors.

Prefer not killing at all: background work launched here is wrapped in
`experiments/scripts/livelock_watch.sh`, which ends a stuck run on its own and
distinguishes a tick stall (exit 3) from a dead child (exit 4).

## Testing requirements

Beyond ordinary unit tests, three test classes are mandatory and easy to overlook:

- **Oracle-agreement**: on a small synthetic cluster (e.g. 4 GPU + 2 NPU), the optimum found with
  pruning enabled must equal the pruning-disabled exhaustive optimum. If pruning removes the
  optimum, that is a bug.
- **Reproducibility**: same spec + same seed run twice ⇒ byte-identical plan output.
- **Golden output**: `PlannerOutput` for both `examples/` specs is frozen and regression-checked.

## Two invariants that are easy to break

**A pruning stage must be a relaxation of the feasibility test, never an extra condition.** The
generator's stages 4-5 are *lower bounds*: they may reject only when even the most optimistic
arithmetic misses a constraint that §5.6 actually declares. An early throughput bound violated this
— it rejected under-provisioned candidates although §5.6 declares no throughput constraint — and
the oracle-agreement test caught it as a pruned-vs-oracle disagreement. It was removed;
`planner/candidate_generator.py` records what restoring it would require.

**A mock predictor must respect the same physics as the bounds.** The first `MockPredictor`
invented throughput independently of the memory roofline, so bound-rejected candidates came back
"feasible" and the oracle test failed for reasons unrelated to the planner. It now derives latency
from the real weight/KV sizes and the profile bandwidth.

## Phase ordering (strict)

`Phase 0` baseline reproduction + format archaeology → `Phase 1` spec/inventory/islands →
`Phase 2` offline simulator-guided planner (**MVP / first paper result**) → `Phase 3` heterogeneous
profiles + NPU CSV importer → `Phase 4` real deployment + sim-vs-real calibration →
`Phase 5` topology-aware P/D → `Phase 6` online replanning (**requires explicit user approval**).

Do not begin topology graphs, P/D placement, or replanning before the Phase 0–2 static planner is done.

## When spec and reality diverge

If upstream's actual filenames, config schema, or output columns contradict the work order,
**the real code wins**. Record the difference in `docs/deviations.md` and continue.

### Claim a D-number from your work order's block, never "the next one"

**Every number up to the last entry in `docs/deviations.md` is taken.** When two work orders run in parallel they both reach for the
next free number from the same base and collide — which happened on 2026-09-14, when
`WORK_ORDER_uncertainty_planner.md` STEP B5 and the B4 experiments both wrote a `D34`.
**Git does not catch this.** The two entries land ~190 lines apart in a 2,500-line file,
so there is no textual overlap: the merge succeeds and the file simply contains two
`## D34` headings. A person caught it; that is not a repeatable defence.

Renumbering afterwards is not cheap either — there are **~1,100 D-references** across
`docs/`, `planner/`, `tests/` and `experiments/` (D22 alone has 193), so a number that
has been referenced for a while cannot move without a large, risky edit.

So each concurrent work order owns a block and takes numbers only from it:

| block | owner |
| --- | --- |
| D40–D49 | `WORK_ORDER_uncertainty_planner.md` |
| D50–D59 | `WORK_ORDER_pipeline_domain.md` |
| D60–D69 | `WORK_ORDER_cxl_kv_pool.md` |
| D70–D79 | `WORK_ORDER_uq_stage_b_plus.md` (claimed 2026-09-14, STEP C0) |
| D80–D89 | one-off work with no work order |
| D90–D99 | `WORK_ORDER_npu_exec_model_spike.md` (claimed 2026-09-15) |
| D100–D109 | `WORK_ORDER_p2_regular_spec_evidence.md` (claimed 2026-09-16, STEP V0) |
| D110–D119 | `WORK_ORDER_domain_scoping.md` (claimed 2026-09-17, STEP S1) |
| D120–D129 | `WORK_ORDER_graph_search.md` (claimed 2026-09-22, STEP H1) — the work order belongs to `swsok/heteropilot-graphsearch`; the staged copy is `graphsearch/WORK_ORDER_graph_search.md` |

D37–D39 are left free on purpose: they are the only numbers a stream may take
**without** a block, and only for an entry that must sit immediately after the last
one already recorded.
Gaps inside a block cost nothing — `deviations.md` already says its entries are in
the order they were written, not numerically, and `D29b` exists because one had to be
squeezed in.

`tests/test_deviations_numbering.py` enforces what a reader cannot: no duplicate
heading, and no reference to a `D<n>` that has no entry.

### Experiment IDs carry the work order's tag — the same problem, one letter cheaper

`E1` is not a name, it is a name *within a work order*, and nothing in the repo says
which one. **Four work orders already define the same numbers**: `E1`–`E4` in both
`WORK_ORDER_cxl_kv_pool.md` and `WORK_ORDER_tiered_profiles.md`, `E5`–`E6` in both
`cxl_kv_pool` and `WORK_ORDER_rps_aware.md`. Meanwhile `docs/CLAIMS.md`,
`docs/HANDOVER.md` and `docs/PROJECT_REPORT.md` all cite a bare `E5`/`E6` meaning
`rps_aware`'s. It reads unambiguously today only because `cxl_kv_pool` has not run —
the collision is already written down, exactly as D34's was before it fired.

So a **new** experiment gets a tagged id, `E-<tag><n>`, and the tag belongs to one
work order:

| tag | owner |
| --- | --- |
| `E-A*`, `E-B*` | `WORK_ORDER_uncertainty_planner.md` (Stage A, Stage B) |
| bare `E1`–`E7` | **history, do not extend** — `cxl_kv_pool`, `rps_aware`, `tiered_profiles`, read as scoped to the file that defines them |
| `E-N*` | `WORK_ORDER_npu_exec_model_spike.md` (claimed 2026-09-15) |
| `E-G*` | `WORK_ORDER_graph_search.md` (claimed 2026-09-22) — runs in `swsok/heteropilot-graphsearch`, not here |
| anything else | unclaimed — add the row here in the same commit that first uses it |

**The existing bare ids do not move.** There are ~460 `E<n>` references across `docs/`,
`WORK_ORDER_*.md` and `experiments/`, and renaming them would be the large risky edit
the D-block rule exists to avoid. They stay; only the next *new* experiment is tagged.
A work order still on bare numbers picks a tag when it next adds one.

**Citing another work order's experiment**, write it qualified at least once per
document — "`rps_aware`'s E6", not "E6" — for exactly as long as bare ids exist.

`tests/test_experiment_ids.py` holds the line: it fails when two work orders define an
id that is not on the known-legacy list, so the six collisions above can be worked
through without a seventh appearing behind them.

Thirty-six divergences are recorded there. D2 (power is stdout-only) and D3 (no topology graph in
the cluster config) are decided; **D4 is now closed** by the Tier 0 synthetic-bundle path (D21).
**D10 is the one to know before touching Phase 2**: the simulator's memory model applies no
utilization or activation reserve, so it over-estimates usable KV by +71% on a 24 GB card.
`planner/util/memory.py` derates explicitly. D11 quantifies what profile-grid density costs
(~2.2pp of end-to-end accuracy).

**Two open ones gate current work.** D12 (prefix-cache memory grows until the run dies) still
blocks Phase 2 — read it before retrying, both earlier fixes were wrong and were reverted. D20
blocks ATOM. **D30 forbids `--top-k` as a cost lever** on P/D or heterogeneous corpora: the
roofline surrogate's proxy is invariant to TP and DP. And note D22 — the "RNGD wins on energy by
1.67×" headline is **retracted**; do not quote it.

**D23 is resolved (2026-09-07) and its old description here was wrong.** It read "every
`pd_*`/`mix_*` candidate livelocks"; the candidates never livelocked, and the failure was never a
property of the candidates. The symptom was **D26** (the Chakra converter invoked as bare `python`)
and the crashes were **D25** (the ASTRA-Sim `tmp__mem` race). The discriminator is **instance
count**, not candidate kind. Both are fixed (D27 subsumes D26), D22's verdict holds on the fixed
harness with **0 timeouts**, and `pd_*`/`mix_*` candidates simulate normally.

**D33 is the one to know before touching accuracy domains or margins.** Two implementations were
built in parallel; `calibration.AccuracyDomain` is the one that remains, `refuse` is the default
for a new domain while the three committed domains opt into `widen_error_bars` explicitly, and the
per-candidate `MarginPolicy` in `planner/optimizer/margin.py` is the single consumer. A candidate
whose operating point no domain covers is `outside_calibration_domain` — unmeasured, not
infeasible. Stage B sits on top: `planner/uncertainty/{perturb,sensitivity,measurement_plan}.py`
re-judge perturbed metrics through the same `judge()`/`rank_plans()` the search uses, and
`plan --accuracy-domain --measurement-plan` emits what to measure next, ranked by regret per hour.
**Since D110 that is not the only refusal** — a candidate whose *configuration* no domain was
measured under is `calibration_condition_mismatch`, which asks for a measurement at that
configuration rather than a wider load range; see the *Uncertainty-aware planning* section above
before reading any count from an experiment written before it.

Derive schemas from real artifacts, with one trap: **`outputs/example_*_run.csv` are stale** and
must not be used as golden references — their `output` column counts `input + output` tokens while
current `main` counts decode tokens only. `bench/examples/` reproduces exactly at the pin and is
the safe regression anchor. Details in `docs/phase0_formats.md` §2.1.
