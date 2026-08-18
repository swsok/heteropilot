# A40 sim-vs-real validation plan (Phase 4 entry)

Roadmap step 4 (`docs/hardware_roadmap.md`) / `HANDOVER.md` §7 step 4. Mirrors
`docs/phase0_bench_plan.md` §2b — the A5000 loop — on the real **A40×8 migration
server**. This document is the protocol; the scaffolding it drives is authored
and committed but **not yet run** (GPUs busy, A40 perf bundle still being
profiled). It records what to run, in what order, and what is still missing.

> Provenance discipline (absolute rule 3): every A40 hardware number in the
> configs is labelled `measured` / `vendor_spec` / `to_be_measured` in
> `experiments/configs/clusters/a40-llama31-8b-tp1.provenance.yaml`. Nothing here
> is a fresh measurement yet.

## 1. Why this exists

The A5000 loop (`phase0_bench_plan.md` §2b) established the headline Phase 4
finding: on a memory-constrained card the simulator's memory accounting (D10) is
the single largest error term. Nominal `mem_size: 24` gave **22.54%** mean
absolute error over 15 metrics; a KV-matched `mem_size: 20.81` gave **9.26%**
(−13.28pp, 13/15 metrics improved). The A40 loop repeats the *nominal vs
KV-matched* comparison on a second real CUDA class, at a much larger memory
budget (45 GB usable vs 24 GB), to test whether that error shrinks when the card
is no longer the binding constraint.

## 2. Artifacts (authored, committed, not yet run)

| File | Role | Mirrors |
| --- | --- | --- |
| `experiments/configs/clusters/a40-llama31-8b-tp1.json` | sim cluster config, **nominal** `mem_size: 45` | `a5000-llama31-8b-tp1.json` |
| `experiments/configs/clusters/a40-llama31-8b-tp1-kvmatched.json` | sim cluster config, **KV-matched** `mem_size` (placeholder sentinel) | `a5000-llama31-8b-tp1-kvmatched.json` |
| `experiments/configs/clusters/a40-llama31-8b-tp1.provenance.yaml` | per-field sourcing for both configs | `a5000-llama31-8b-tp1.provenance.yaml` |
| `experiments/scripts/run_a40_sim_vs_real.sh` | one-command runner (bench → sim×2 → validate → compare) | `bench/examples/run.sh` + `validate.sh` |

Both JSON configs are byte-for-byte the A5000 config except: `hardware: A40`,
`npu_mem.mem_size` (45 nominal / KV-matched placeholder), and
`npu_mem.mem_bw: 696` (A40 vendor spec, `profiles/accelerators/a40.yaml`). The
single-A40 TP=1 config exercises no inter-device link, so `link_bw` /
`link_latency` are inherited and inert.

## 3. The 15 metrics compared

`bench validate` emits one `summary.txt` with 15 rows — 5 statistics × 3 latency
families (schema confirmed from `outputs/phase0_bench/A5000-np-kvmatched/vllm/validation/summary.txt`):

| Family | Statistics |
| --- | --- |
| **TTFT** (`first_token_ts − arrival_time`, incl. queueing) | Mean, Median, P90, P95, P99 |
| **TPOT** (`(last_token_ts − first_token_ts) / max(1, output_toks − 1)`) | Mean, Median, P90, P95, P99 |
| **Latency** (`last_token_ts − arrival_time`) | Mean, Median, P90, P95, P99 |

Each row is `vLLM  Sim  Diff%`, where `Diff% = (sim − vLLM) / vLLM × 100`
(`bench/README.md` "Latency definitions"). "Mean |error| over 15 metrics" is the
headline scalar, computed by `experiments/scripts/compare_validations.py`, which
also reports the change in |error| between the nominal and KV-matched runs.

## 4. Nominal vs KV-matched — the comparison

Both simulator runs are validated against the **same** real A40 bench artifacts,
so the vLLM column is shared and the two Diff% columns are directly comparable
(`compare_validations.py` warns loudly if the vLLM columns ever diverge).

- **Nominal** (`mem_size: 45`) — the raw usable memory. This is what the planner
  would use if D10 derating were switched off. On the A5000 this over-predicted
  KV capacity by +55% (bench settings) and drove the 22.54% error.
- **KV-matched** — `mem_size` set so the simulator's KV budget equals the real
  A40's. Derivation, mirroring the A5000 arithmetic `14.96 + 5.85 = 20.81`
  (`deviations.md` D10):

  ```
  kv_matched_mem_size = 14.96 (Llama-3.1-8B bf16 weights, GiB; model+dtype
                               dependent only, hardware-independent)
                      + observed_A40_KV_GiB
  ```

  where `observed_A40_KV_GiB` is the `Available KV cache memory: <X> GiB` line
  vLLM prints at engine startup on a **real A40**, captured under the **exact
  bench engine settings** (`max_model_len=8192`, CUDA graphs on — *not*
  `enforce_eager`). D10 warns the KV figure varies with engine settings, so it
  must come from the run that produces the bench artifacts, not a separate probe.

### Expected D10 behaviour (a prediction to record, not a shortcut)

45 GB usable minus ~15 GB weights leaves ~30 GB for KV, versus ~9 GB on the
A5000. The sharegpt-300 workload wants ~21.7 GiB of concurrent KV at
`max_num_seqs=128` (D10), so the A40 is **not** KV-bound the way the A5000 was.
The nominal-vs-KV-matched gap is therefore expected to be **much smaller** than
the A5000's 13.28pp — possibly negligible. That is the hypothesis under test:
D10's error scales inversely with card size, so on a large card the memory
accounting should stop being first-order and profile/regime error should
dominate the residual. Whatever the gap turns out to be is a finding; a small gap
does not mean the KV-matched run was unnecessary — it is the measurement that
confirms the scaling claim.

## 5. Execution order (run later, on the A40 server)

Prerequisites first (`HANDOVER.md` §7 steps 1–3):

1. **Inventory** done — `experiments/configs/clusters/a40x8.yaml` (measured
   2026-08-18, 45 GB usable, ECC on).
2. **A40 perf bundle** — `profiler/perf/A40/` must exist and
   `profiles/accelerators/a40.yaml` must say `sim_hardware: A40` (currently
   `null` by design). Produce with:
   ```bash
   CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python -m profiler profile \
     meta-llama/Llama-3.1-8B --hardware A40 --tp 1,2,4,8 \
     --max-num-batched-tokens 2048 --max-num-seqs 256 --measurement-iterations 3
   ```
3. **Power** (optional for this loop) — step 3; not required, this loop validates
   latency/throughput, not energy (D2: power is stdout-only; the configs carry no
   `power:` block).

Then the loop — one command (the runner does bench → sim×2 → validate → compare):

```bash
# On the A40 server, both venvs built (HANDOVER.md §6).
CUDA_VISIBLE_DEVICES=0 ./experiments/scripts/run_a40_sim_vs_real.sh
```

The runner will refuse to start and print the fixing command if the perf bundle,
`sim_hardware`, either venv, the dataset, or the KV-matched number is missing.

Two-pass reality (the KV number comes from the bench run itself):

- **Pass 1** — run with `SKIP_KVMATCHED=1` to do the real bench + nominal sim:
  ```bash
  CUDA_VISIBLE_DEVICES=0 SKIP_KVMATCHED=1 ./experiments/scripts/run_a40_sim_vs_real.sh
  ```
  Read `Available KV cache memory: <X> GiB` from the vLLM startup log in
  `outputs/phase0_bench/A40/vllm/` (or the bench stdout), compute
  `14.96 + X`, and:
  - replace `"TODO_MEASURE_A40_KV_BUDGET_GB"` in
    `experiments/configs/clusters/a40-llama31-8b-tp1-kvmatched.json` with the
    numeric GB value;
  - fill the same value into
    `experiments/configs/clusters/a40-llama31-8b-tp1.provenance.yaml` and flip
    that field's `source:` to `measured`, recording the raw vLLM line.
- **Pass 2** — re-run without `SKIP_KVMATCHED`. The bench is already recorded, so
  the runner re-uses it; the nominal sim re-runs harmlessly; the KV-matched sim
  and the side-by-side comparison now execute.

Manual equivalents (if not using the runner) — real bench with prefix caching
OFF (D12 / `phase0_bench_plan.md` §2b), then sim, then validate:

```bash
# real vLLM (prefix caching OFF, matches the simulator)
.venv-vllm/bin/python experiments/scripts/bench_run_no_prefix_cache.py \
  --model meta-llama/Llama-3.1-8B \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --output-dir outputs/phase0_bench/A40/vllm \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --max-num-seqs 128 --max-num-batched-tokens 2048 \
  --dtype bfloat16 --kv-cache-dtype auto --seed 42 --num-reqs 300

# simulator (nominal), repeat with the kvmatched config once filled in
.venv/bin/python -m serving \
  --cluster-config experiments/configs/clusters/a40-llama31-8b-tp1.json \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --output outputs/phase0_bench/A40-nominal/sim.csv \
  --num-reqs 300 --dtype bfloat16 --kv-cache-dtype auto --block-size 16 \
  --max-num-seqs 128 --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --log-level WARNING --network-backend analytical \
  > outputs/phase0_bench/A40-nominal/sim.log 2>&1
# --no-enable-prefix-caching is REQUIRED: serving defaults it ON, but the vLLM
# side runs prefix-off and D12 forbids prefix caching on a saturated sim run.

# compare (repeat for kvmatched)
.venv/bin/python -m bench validate \
  --bench-dir outputs/phase0_bench/A40/vllm \
  --sim-csv outputs/phase0_bench/A40-nominal/sim.csv \
  --sim-log outputs/phase0_bench/A40-nominal/sim.log \
  --output-subdir validation --title "vLLM vs LLMServingSim - A40 nominal"

.venv/bin/python experiments/scripts/compare_validations.py \
  nominal=outputs/phase0_bench/A40-nominal/validation/summary.txt \
  kvmatched=outputs/phase0_bench/A40-kvmatched/validation/summary.txt
```

## 6. Protocol invariants (do not silently change)

- **Prefix caching OFF on both sides.** The simulator cannot complete a saturated
  workload with it on (D12); comparing prefix-off sim to prefix-on vLLM compares
  two systems. The A5000 loop used the same `bench_run_no_prefix_cache.py`
  wrapper rather than editing upstream `bench/`.
- **Matched engine settings.** `bench run` and `python -m serving` share
  `dtype`, `kv_cache_dtype`, `max_num_seqs`, `max_num_batched_tokens`,
  `block_size`, `num_reqs`, `seed`, and dataset — the runner uses one set of
  variables for both (defaults from `outputs/phase0_bench/A5000/vllm/meta.json`).
- **Repo-root-relative paths only.** `serving/__main__.py` chdir's into astra-sim
  and prepends `../`; the runner cd's to the repo root and passes relative paths
  (HANDOVER.md §6 trap 4).
- **Same vLLM ground truth for both sim runs**, so the two Diff% columns are
  comparable (`compare_validations.py` enforces this).

## 7. Still MISSING (blockers to running this)

1. **Real A40 bench data** — `outputs/phase0_bench/A40/vllm/` does not exist. GPUs
   0/1 are busy (live profiling + power measurement); no bench has run.
2. **A40 perf bundle** — `profiler/perf/A40/` is being profiled now;
   `profiles/accelerators/a40.yaml` still has `sim_hardware: null`. Without it the
   simulator rejects `hardware: A40`.
3. **KV-matched `mem_size` number** — the placeholder
   `"TODO_MEASURE_A40_KV_BUDGET_GB"` in `a40-llama31-8b-tp1-kvmatched.json` must
   be replaced with `14.96 + observed_A40_KV_GiB`, where the KV figure is read
   from the real A40 vLLM startup under bench settings (§4). Not yet measured →
   not invented (absolute rule 3).
4. **A40 power** (only for a later energy loop, not these 15 metrics) —
   unmeasured; the configs deliberately carry no `power:` block (D2/D7).

## 8. What "done" looks like

`outputs/phase0_bench/A40-nominal/validation/summary.txt` and
`.../A40-kvmatched/validation/summary.txt` exist, `compare_validations.py` prints
the mean |error| for both and the per-metric improvement, and the result plus the
expected-vs-observed D10 gap (§4) is written up in `phase0_bench_plan.md` as a
new section (mirroring §2b), with the KV-matched provenance flipped to
`measured`. Only then does the A40 half of the Phase 4 calibration table
(`HANDOVER.md` §4) get its two rows.
