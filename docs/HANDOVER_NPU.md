# Handover — continuing HeteroPilot on the NPU server

*Written 2026-08-25 for a fresh Claude Code session that will `git pull` this repo
on the NPU server and continue. Read this first, then `docs/PROJECT_REPORT.md` for
the full state and `WORK_ORDER_heteropilot.md` for the spec.*

> **Why this file exists:** the previous machine's Claude Code memory
> (`~/.claude/.../memory/`) does **not** travel with `git pull`. This committed
> doc is the transferable handoff. Everything you need to resume is here or linked.

---

## 1. Where things stand (as of `git log` main = merge of PR #9)

- **Upstream pinned** at `2c2042ce` (`UPSTREAM_COMMIT`); do not silently rebase.
- **Phases 0–5 are done**, Phase 4 for **CUDA only**. Merged work: PR #1 (A40
  bring-up + Phase 4 deploy/calibration + Phase 5 P/D), #2 (sim-level P/D
  transfer), #3 (Level-2 topology), #4 (Exp 1 + A40 tp4 profile), #5 (baselines +
  ablation), #6 (figure packaging), #7 (router baselines), #8 (Exp 2/5 figures),
  #9 (surrogate top-K + parallel sim).
- **The one thing blocked on NPU hardware is Exp 4** (GPU vs NPU SLO-goodput/J) and
  turning the **Exp-5 NPU rows from SIM-PROXY into measured**. That is now your job.
- Quality bar to keep green before any merge: `pytest` (284 passing), `ruff check
  .`, `mypy planner/`.

---

## 2. Environment setup on the NPU server

Two virtualenvs are used (this is deliberate — see `docs/phase0_formats.md` §1):

```bash
# 2a. Planner + analytical simulator env (.venv). NO vLLM. Runs `plan`, experiments.
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install pyyaml pyinstrument transformers datasets msgspec scikit-learn \
  xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 numpy==1.23.5 rich
bash scripts/compile.sh                                   # ASTRA-Sim + Chakra
uv pip install ./astra-sim/extern/graph_frontend/chakra   # compile.sh's bare pip3 misses the venv
uv pip install "protobuf>=7.35.1"                          # Chakra gencode needs it

# 2b. Profiling / real-vLLM env (.venv-vllm). NEEDED to profile NPU hardware.
#     On the A40 box this held vLLM 0.19.0 + torch. On the NPU server it must hold
#     the NPU runtime + vLLM-Ascend / rbln / furiosa vLLM fork for the NPU vendor.
#     Set it up per the vendor's vLLM install guide; the profiler shells out to it.
```

- The **analytical simulator does not need a GPU/NPU** — it uses profiled traces.
  Only **profiling** (building a perf bundle) needs the real device + `.venv-vllm`.
- Experiment runners need `export PYTHONPATH=$REPO_ROOT` and `.venv/bin` on `PATH`
  (the sim shells out to `python -m chakra`). See any `experiments/scripts/run_*.sh`.

---

## 3. The NPU work path (do these in order)

The goal: replace every **SIM-PROXY / placeholder** NPU number with a **measured**
one, then run **Exp 4**. Concrete NPU targets: **Rebellions ATOM (×4, backend
`rbln`)** and **FuriosaAI RNGD (×4, backend `furiosa`)** — see
`docs/hardware_roadmap.md`. Profile stubs already exist:
`profiles/accelerators/rbln_atom.yaml`, `furiosa_rngd.yaml` (and `ascend_target.yaml`),
all `sim_hardware: null` so they **fail loud** until profiled.

1. **Inventory the NPU hardware into a ClusterSpecV2** (like
   `experiments/configs/clusters/a40x8.yaml`): real device count, memory, backend,
   topology links (measured bandwidth/latency), per-field provenance. Absolute rule
   3: label every number measured / vendor_spec / placeholder.
2. **Profile each NPU model** into `profiler/perf/<HW>/<model>/<variant>/tpN/`.
   - If the vendor's vLLM fork supports the layerwise profiler: run
     `.venv-vllm/bin/python -m profiler profile <model> --hardware <HW> --tp 1,2,...`
     (mirror the A40 run in `docs/hardware_roadmap.md` / the profiling memory).
   - If not: use the **`CsvProfileImporter`** (Phase 3 V1, `profiler/core/importer.py`)
     to import externally measured CSVs under the `profiler/CONTRACT.md` schema.
   - Set the profile's `sim_hardware`, `memory_gb`, `memory_bandwidth_gbps`,
     measured `power:` block, and `supported_models` from real kernel coverage.
3. **Measure NPU power** (idle/standby/active) the same way the A40 was measured
   (deviations D7 protocol) if the vendor exposes power telemetry; else mark
   placeholder and say so in every figure.
4. **Run Exp 4** — GPU (measured A40) vs NPU (now measured) island, same
   model/workload, compare SLO-goodput/J. Write a new driver
   `experiments/scripts/exp4_gpu_vs_npu.py` mirroring `exp2_selection.py` (reuse
   `planner.util.parallel.predict_all` for speed), emit committed JSON + a
   `make_figures.py` entry (`exp4_*.png`).
5. **Upgrade Exp 5** — re-run `pd_combo_compare.py` with the **real** NPU profile so
   the four combos are no longer byte-identical SIM-PROXY rows; relabel the results
   doc and figure.
6. **Sim-vs-real calibration for the NPU** (Phase 4) — deploy on real NPU via a new
   `planner/deploy/vllm_<backend>.py` (the CUDA one is `vllm_cuda.py`;
   `vllm_ascend.py` is a stub), collect real TTFT/TPOT/power, fit
   `profiles/calibration/<hw>.yaml` (like `a40.yaml`). This also unblocks the
   No-Calibration / No-Uncertainty ablations (currently N/A).

---

## 4. How to run the key commands

```bash
export PYTHONPATH=$PWD && export PATH="$PWD/.venv/bin:$PATH"

# Plan (now parallel by default; --workers N to tune, --top-k K for surrogate).
python -m planner plan --service <spec>.yaml --cluster <cluster>.yaml \
    --num-requests 300 --seed 42 --workers 32 --output outputs/plans/plan.yaml
python -m planner inspect-cluster --cluster <cluster>.yaml --service <spec>.yaml

# Experiments (one-command runners, all parallel-accelerated):
./experiments/scripts/run_exp1.sh                # TP sweep (needs the perf bundle)
./experiments/scripts/run_exp_baselines.sh       # optimizer/resource/arch/ablation
./experiments/scripts/run_exp_router.sh          # RR/RAND/LOAD
./experiments/scripts/run_exp_surrogate.sh       # surrogate top-K accuracy
./experiments/scripts/run_exp_pd.sh              # Exp 3 network sweep + Exp 5 combos
python experiments/scripts/make_figures.py       # regenerate all figures from JSON

# Profile a device (needs the real device + .venv-vllm):
CUDA_VISIBLE_DEVICES=<n> .venv-vllm/bin/python -m profiler profile \
    <model> --hardware <HW> --tp 1,2 --variant bf16
```

**Parallelism:** candidate simulations run concurrently (`planner/util/parallel.py`,
~half the CPUs, capped 32). Each candidate is an isolated subprocess (unique
`--run-id`) so there is no locking; results are byte-identical to sequential. Use
`--workers 1` to force sequential for debugging. Long runs: launch in the
background and poll, or use the run scripts.

---

## 5. Constraints & gotchas (do not relearn these the hard way)

- **Absolute rules** (CLAUDE.md): don't invent hardware numbers (label
  placeholder); never mix backends in a TP group; never delete
  `planner/optimizer/exhaustive.py`; code comments/docstrings/logs **English only**;
  one feature = one branch = one PR.
- **`serving/core/scheduler.py` must stay pristine** even in Phase 5 (work order
  §7). Two `serving/` edits are now sanctioned and shipped — D15 (sim-level P/D
  transfer, in `router.py` + `__main__.py`) and none in scheduler.py. Keep it that way.
- **Deviations to know before touching Phase 2+**: D10 (memory model over-estimates
  usable KV by +71 % → `planner/util/memory.py` derates), D3/D15 (no link graph in
  the sim config), D12 (prefix caching OFF — it crashes the sim). Full list:
  `docs/deviations.md`.
- **Two venvs**: `.venv` = analytical planner/sim (no GPU needed); `.venv-vllm` =
  profiling/real-vLLM (needs the device). Don't cross them.
- **PR workflow on these servers**: `gh` here is NOT the GitHub CLI. Create/merge
  PRs via the GitHub REST API using a token from `git credential fill` (never print
  the token). See how the prior session did it, or just push the branch and open the
  PR in the browser.
- **Don't write sim output into `bench/examples/` or over `outputs/example_*`** —
  those are upstream-tracked. Redirect to `outputs/` (gitignored-ish) or a temp dir.

---

## 6. Key documents

| Doc | What it gives you |
| --- | --- |
| `docs/PROJECT_REPORT.md` | Full integrated results + status (deck basis) |
| `WORK_ORDER_heteropilot.md` | The HeteroPilot spec (Korean, v1.0) — outranks CLAUDE.md |
| `docs/deviations.md` | D1–D15: where upstream diverges and how we adapt |
| `docs/hardware_roadmap.md` | NPU targets (ATOM/RNGD), bring-up plan, provenance rules |
| `docs/phase5_plan.md` | Phase 5 increments + P/D status |
| `docs/phase0_formats.md` | Real CSV / cluster-config schemas + env setup detail |
| `AGENTS.md` | Upstream simulator internals (do not edit) |

---

## 7. Suggested first move on the NPU server

1. Recreate `.venv` (step 2a), run `pytest` — expect 284 passing. This confirms the
   analytical stack works after the pull.
2. Set up `.venv-vllm` for the NPU vendor and confirm the profiler can boot the
   device (`python -m profiler profile ... --tp 1`).
3. Then start §3 step 1 (inventory) → step 2 (profile) → step 4 (Exp 4).

Save your own progress notes back into this repo (append to this file or a new
`docs/HANDOVER_NPU_progress.md`) so the next pull carries them forward.
