# Handover — continuing HeteroPilot on the NPU server

*Written 2026-08-25 for a fresh Claude Code session that will `git pull` this repo
on the NPU server and continue. Read this first, then `docs/PROJECT_REPORT.md` for
the full state and `WORK_ORDER_heteropilot.md` for the spec.*

> **Why this file exists:** the previous machine's Claude Code memory
> (`~/.claude/.../memory/`) does **not** travel with `git pull`. This committed
> doc is the transferable handoff. Everything you need to resume is here or linked.

---

## 1. Where things stand (as of `git log` main = merge of PR #12)

- **Upstream pinned** at `2c2042ce` (`UPSTREAM_COMMIT`); do not silently rebase.
- **Phases 0–5 are done**, Phase 4 for **CUDA only**. Merged work: PR #1 (A40
  bring-up + Phase 4 deploy/calibration + Phase 5 P/D), #2 (sim-level P/D
  transfer), #3 (Level-2 topology), #4 (Exp 1 + A40 tp4 profile), #5 (baselines +
  ablation), #6 (figure packaging), #7 (router baselines), #8 (Exp 2/5 figures),
  #9 (surrogate top-K + parallel sim), #10–#12 (project report, NPU handover,
  slide outline + HTML deck).
- **The one thing blocked on NPU hardware is Exp 4** (GPU vs NPU SLO-goodput/J) and
  turning the **Exp-5 NPU rows from SIM-PROXY into measured**. That is now your job.
- Quality bar to keep green before any merge: `pytest` (284 passing), `ruff check
  .`, `mypy planner/`.

---

## 2. Environment setup on the NPU server

**Done on 2026-08-25 — this section is now a verified log, not a plan.** The exact
sequence that worked, with the four gaps the original instructions had:

```bash
# 2.0 System prerequisites. Both were MISSING on a fresh NPU server.
sudo apt install -y protobuf-compiler libprotobuf-dev   # needs a password; ASTRA-Sim's
                                                        # CMake needs protoc + C++ headers
git submodule update --init --recursive                  # astra-sim was an empty dir
pip3 install --user uv                                   # uv was not installed

# 2a. Planner + analytical simulator env (.venv). NO vLLM. Runs `plan`, experiments.
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install pyyaml pyinstrument transformers datasets msgspec scikit-learn \
  xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 numpy==1.23.5 rich
uv pip install pydantic          # MISSING from the old list; planner/inventory.py imports it,
                                 # so without it conftest.py fails and NOTHING collects
uv pip install pytest ruff mypy  # the §1 quality gates were also absent from the list
bash scripts/compile.sh                                   # ASTRA-Sim + Chakra
uv pip install ./astra-sim/extern/graph_frontend/chakra   # compile.sh's bare pip3 misses the venv
uv pip install "protobuf>=7.35.1"                          # Chakra gencode needs it

# 2b. Profiling env — NO .venv-vllm IS NEEDED HERE.
#     The vendor runtimes are already installed in the SYSTEM python3:
#       vllm 0.13.0+cpu · vllm_rbln 0.10.2.post1 · optimum-rbln/rebel-compiler 0.10.2
#       furiosa-llm/furiosa-torch 2026.2.0 · torch 2.10.0+cu128
#     Invoke the profiler with `PYTHONPATH=$PWD python3 -m profiler ...` (system python3).
#     Importing vllm there is slow (tens of seconds) but works. Do NOT install vLLM into
#     .venv — the analytical planner must stay vLLM-free.
```

Verified on this machine after the above (all four green):

| check | result |
| --- | --- |
| `pytest` | **284 passed** (needs the fix in commit `071d282`, see below) |
| `ruff check .` | all checks passed |
| `mypy planner/` | no issues, 33 source files |
| `python -m serving` smoke run | power + TTFT/TPOT summary printed, CSV written |
| `python -m planner inspect-cluster` | 3 islands, TP candidates, D10 derating shown |

Notes and traps confirmed here:

- **cmake 4.3.2 builds ASTRA-Sim fine** (`cmake_minimum_required` is 3.22). Both
  `AstraSim_Analytical_Congestion_{Aware,Unaware}` link.
- `transformers` warns "PyTorch was not found" in `.venv`. Expected — tokenizers only.
- The four `tests/test_calibration.py` failures a fresh clone hits are **not** an
  environment problem: the test read `outputs/phase0_bench/A40/vllm/validation/summary.txt`,
  which was never committed (`outputs/phase0_bench/` is gitignored; only
  `validation-nominal/summary.txt` was force-added). Fixed on
  `fix/calibration-test-artifact-path`. `validation-nominal` is the canonical file —
  refitting from it reproduces `profiles/calibration/a40.yaml` bit-for-bit.
- The **analytical simulator does not need a GPU/NPU** — it uses profiled traces.
  Only **profiling** (building a perf bundle) needs the real device.
- Experiment runners need `export PYTHONPATH=$REPO_ROOT` and `.venv/bin` on `PATH`
  (the sim shells out to `python -m chakra`). See any `experiments/scripts/run_*.sh`.
- There is **no NVIDIA GPU on this machine**, so no CUDA `bench/` run can be
  reproduced or extended here. See CLAUDE.md *This machine*.

---

## 3. The NPU work path (do these in order)

The goal: replace every **SIM-PROXY / placeholder** NPU number with a **measured**
one, then run **Exp 4**. Concrete NPU targets: **Rebellions ATOM (×4, backend
`rbln`)** and **FuriosaAI RNGD (backend `furiosa`)** — 4 cards are on the PCI bus but
`furiosa-smi` enumerates only 3 (`44:00.0` reads PCI rev `ff`), so treat the count as
**3 usable until proven otherwise**. See `docs/hardware_roadmap.md`. Profile stubs already exist:
`profiles/accelerators/rbln_atom.yaml`, `furiosa_rngd.yaml` (and `ascend_target.yaml`),
all `sim_hardware: null` so they **fail loud** until profiled.

1. **Inventory the NPU hardware into a ClusterSpecV2** (like
   `experiments/configs/clusters/a40x8.yaml`): real device count, memory, backend,
   topology links (measured bandwidth/latency), per-field provenance. Absolute rule
   3: label every number measured / vendor_spec / placeholder.
2. **Profile each NPU model** into `profiler/perf/<HW>/<model>/<variant>/tpN/`.
   - If the vendor's vLLM fork supports the layerwise profiler: run
     `PYTHONPATH=$PWD python3 -m profiler profile <model> --hardware <HW> --tp 1,2,...`
     with the **system** python3 (§2b: the vendor runtimes live there, not in a venv)
     — mirror the A40 run in `docs/hardware_roadmap.md`.
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

# Profile a device (needs the real device; vendor runtimes are in the system python3):
PYTHONPATH=$PWD python3 -m profiler profile \
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
- **Two interpreters**: `.venv` = analytical planner/sim (no device needed); the
  **system** `python3` = profiling/real-vLLM (vendor runtimes, needs the device).
  Don't cross them — never install vLLM into `.venv`.
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

**§2 is already done on this server** (2026-08-25) — `.venv` exists, ASTRA-Sim is
built, and pytest/ruff/mypy are green. So:

1. Confirm the stack still works after your pull: `pytest` — expect 284 passing.
2. Confirm the profiler can reach a device with the **system** python3
   (`PYTHONPATH=$PWD python3 -m profiler profile ... --tp 1`). Resolve the
   FuriosaAI `44:00.0` card first, and note that idle `rbln-stat` shows leftover
   13–14 GiB contexts held by other users' `python3.10` processes — clear them or
   pick free devices before profiling.
3. Then start §3 step 1 (inventory) → step 2 (profile) → step 4 (Exp 4).

Save your own progress notes back into this repo (append to this file or a new
`docs/HANDOVER_NPU_progress.md`) so the next pull carries them forward.
