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
| `docs/nodes/{a40,a5000,npu}.md` | Per-node inventory, topology and traps | Read the one `scripts/whichnode.sh` names — never all three |
| `docs/CLAIMS.md` | What can be claimed today, with each claim's label and artifact — Established / Not established / Retracted | Read before writing any result into a paper or deck |
| `docs/HANDOVER.md` | Current state, next work by node, traps that cost a session | The live handover; the `HANDOVER_*.md` files are historical |

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
   for D12, but **both attempts were wrong and have been reverted** — `serving/` is pristine.
   Read `docs/deviations.md` D12 before trying again; it records what was tried and why it failed.
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

**Hostname does not discriminate.** Every node reports `s8` on kernel
5.4.0-216-generic. The only incidental difference in committed provenance is the
CPU count (A40 64, NPU 96). Do not key off `hostname`.

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
ASTRA-Sim input root, so parallel simulations need no extra locking.

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

Quality gates required before merging a PR:

```bash
pytest                     # single test: pytest tests/test_inventory.py::test_name
ruff check .
mypy planner/              # planner/ only — upstream code is not type-clean
```

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

Twenty-four divergences are recorded there. D2 (power is stdout-only) and D3 (no topology graph in
the cluster config) are decided; **D4 is now closed** by the Tier 0 synthetic-bundle path (D21).
**D10 is the one to know before touching Phase 2**: the simulator's memory model applies no
utilization or activation reserve, so it over-estimates usable KV by +71% on a 24 GB card.
`planner/util/memory.py` derates explicitly. D11 quantifies what profile-grid density costs
(~2.2pp of end-to-end accuracy).

**Three open ones gate current work.** D12 (prefix-cache memory grows until the run dies) still
blocks Phase 2 — read it before retrying, both earlier fixes were wrong and were reverted. D20
blocks ATOM. **D23 blocks the tight-TTFT regime**: every `pd_*`/`mix_*` candidate livelocks, which
is why no timeout settles that half of the three-regime table. And note D22 — the "RNGD wins on
energy by 1.67×" headline is **retracted**; do not quote it.

Derive schemas from real artifacts, with one trap: **`outputs/example_*_run.csv` are stale** and
must not be used as golden references — their `output` column counts `input + output` tokens while
current `main` counts decode tokens only. `bench/examples/` reproduces exactly at the pin and is
the safe regression anchor. Details in `docs/phase0_formats.md` §2.1.
