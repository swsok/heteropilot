# Phase 0 — bench / vLLM validation baseline

Work order Phase 0 item 3. Records what was actually reproduced on this machine, what was not, and
the exact path to close the gap.

- Upstream pin: `2c2042ce`, `astra-sim` `f82fb3d`
- Run on: 2026-08-07
- Machine: 2 × NVIDIA RTX A5000 24 GB, 20 cores, 93 GB RAM, Python 3.10.12, no NPU

## 1. What was reproduced — simulator side, against real vLLM measurements

`bench/examples/` ships **real vLLM measurements** (`vllm/meta.json`, `requests.jsonl`,
`timeseries.csv`) alongside the simulator output and the validation report. The measurements are
recorded data, so the sim-vs-real comparison can be regenerated without a GPU: rerun the simulator
side and re-run `bench validate` against the committed vLLM artifacts.

Upstream's `bench/examples/run.sh` overwrites the committed `sim.csv` / `sim.log` in place, which
absolute rule 1 forbids, so the equivalent commands were run by hand with outputs redirected to
`outputs/phase0_bench/<model>/`.

All three bundled examples reproduce **byte-identically** against the committed validation
summaries (`diff` empty in every case):

| Example | TP | vs. committed validation |
| --- | --- | --- |
| `meta-llama/Llama-3.1-8B` | 1 | byte-identical |
| `Qwen/Qwen3-32B` | 2 | byte-identical |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 2 (MoE) | byte-identical |

```bash
source .venv/bin/activate
python -m serving \
  --cluster-config bench/examples/configs/Llama-3.1-8B.json \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --output outputs/phase0_bench/Llama-3.1-8B/sim.csv \
  --num-reqs 300 --dtype bfloat16 --kv-cache-dtype auto \
  --block-size 16 --max-num-seqs 128 --max-num-batched-tokens 2048 \
  --log-level WARNING --network-backend analytical \
  > outputs/phase0_bench/Llama-3.1-8B/sim.log 2>&1

cp -r bench/examples/Llama-3.1-8B/vllm outputs/phase0_bench/Llama-3.1-8B/vllm
python -m bench validate \
  --bench-dir outputs/phase0_bench/Llama-3.1-8B/vllm \
  --sim-csv  outputs/phase0_bench/Llama-3.1-8B/sim.csv \
  --sim-log  outputs/phase0_bench/Llama-3.1-8B/sim.log \
  --output-subdir validation
```

All simulator arguments are read out of `vllm/meta.json` so the two sides stay matched — dataset,
`num_requests`, `dtype`, `kv_cache_dtype`, `max_num_seqs`, `max_num_batched_tokens`.

**Result — `meta-llama/Llama-3.1-8B`, byte-identical to the committed validation summary:**

| Metric | vLLM | Sim | Diff |
| --- | ---: | ---: | ---: |
| TTFT Mean | 7097.2 | 7072.4 | −0.3% |
| TTFT Median | 9442.7 | 9031.7 | −4.4% |
| TTFT P99 | 19755.3 | 19957.2 | +1.0% |
| TPOT Mean | 32.5 | 32.7 | +0.7% |
| TPOT P99 | 37.3 | 38.1 | +2.1% |
| Latency P99 | 37638.8 | 37956.9 | +0.8% |

Tail error across all three (sim vs real):

| Model | TTFT P99 | TPOT Mean | TPOT P99 | Latency P99 |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | +1.0% | +0.7% | +2.1% | +0.8% |
| Qwen3-32B | +2.0% | +1.7% | +2.2% | +2.3% |
| Qwen3-30B-A3B | +4.7% | +1.1% | +2.7% | +0.8% |

Note the contrast with `outputs/example_*_run.csv`, which **are** stale (`deviations.md` D6): the
bench examples reproduce exactly at the pinned commit and are safe to treat as a regression anchor.

Three properties worth carrying into Phase 4 calibration (§5.8):

- **The bias is systematic and one-directional.** Every tail figure above is positive — the
  simulator predicts *slower* than reality on all three models. For SLO feasibility that is the
  safe direction: planning on uncorrected sim output is conservative, not optimistic. The §5.8
  robust margin (`predicted * (1 + p95_error)`) therefore compounds an already-conservative
  estimate, so calibration should correct the bias rather than only pad it.
- **The tail fits better than the middle.** On Llama the largest error is TTFT Median at −4.4%
  while P90/P95/P99 stay within ±2.1%. A single scalar per hardware will fit the tail better than
  the median, which matters because the SLOs are defined on P99.
- **`meta.json` is the provenance precedent.** It already records `engine_kwargs.seed`,
  `dataset_hash`, `vllm_version`, and `num_requests` — exactly the shape §3.8 asks for.

These are three models on **one** hardware class, so they do not yet tell us whether the bias is a
property of the simulator or of the RTXPRO6000 profile. Distinguishing the two needs a second
hardware class (§3).

## 2. What was not reproduced — a fresh real-vLLM run

No new vLLM measurement was taken. vLLM is not installed, no model weights are cached, and this
machine cannot serve two of the three bundled models at all.

Per absolute rule 3 and §11: **every number in §1 above is either upstream's recorded measurement
or our simulation of it. Nothing in this repository is a fresh measurement taken on this machine.**

### Feasibility on 2 × RTX A5000 24 GB

| Model | bf16 weights | Verdict |
| --- | --- | --- |
| `meta-llama/Llama-3.1-8B` | ~16 GB | **Feasible**, TP=1 on one A5000, ~7 GB left for KV cache |
| `Qwen/Qwen3-32B` | ~64 GB | Not feasible — needs TP≥4 at 24 GB/GPU |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ~60 GB | Not feasible — same reason |

The work order's headline model (`Qwen/Qwen3-32B`, `examples/service_specs/qwen3-32b.yaml`) cannot
be served on this hardware. It remains fine as a *planning* target — the planner reasons over a
described cluster, not the local one — but it can never be the Phase 4 calibration model here.
Llama-3.1-8B is the only locally serviceable option.

### Commands to run it when the environment is ready

```bash
# 1. vLLM (bare metal, separate venv — do not mix with the simulator venv)
bash scripts/install-vllm.sh          # VLLM_USE_PRECOMPILED=1 uv pip install vllm==0.19.0
# or containerised:
bash scripts/docker-vllm.sh           # pulls vllm/vllm-openai:v0.19.0

# 2. Real measurement
export HF_TOKEN=...                   # Llama-3.1 is gated
python -m bench run \
    --model meta-llama/Llama-3.1-8B \
    --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
    --output-dir bench/results/<run_id> \
    --tensor-parallel-size 1 --data-parallel-size 1 \
    --max-num-seqs 128 --max-num-batched-tokens 2048 \
    --dtype bfloat16 --kv-cache-dtype auto

# 3. Simulator side + comparison (as in §1, with the new bench dir)
```

`bench run` pins each request with `SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`, so
the vLLM run replays the exact token counts the simulator sees.

Requirements not currently met: ~10 GB disk for the vLLM image plus HF model cache, an `HF_TOKEN`
for gated Llama-3.1 weights, and NVIDIA Container Toolkit if the container path is used (Docker
itself is available and working).

## 2b. A5000 sim-vs-real result (2026-08-07)

A5000 was profiled locally (TP=1, `profiler/perf/A5000/`, x2 attention grid) and benchmarked on
real hardware, giving the first fully local sim-vs-real loop. Both sides ran with **prefix caching
disabled** — the simulator cannot complete this workload on 24 GB with it enabled
(`deviations.md` D12), and comparing a prefix-cache-off simulation against a prefix-cache-on
deployment would not be a comparison. The real side used
`experiments/scripts/bench_run_no_prefix_cache.py`.

| mean \|error\| over 15 metrics | |
| --- | ---: |
| RTXPRO6000, full grid (upstream reference) | 1.23% |
| RTXPRO6000, x2 grid (density control, D11) | 3.05% |
| **A5000, nominal `mem_size: 24`** | **22.54%** |
| **A5000, KV-matched `mem_size: 20.81`** | **9.26%** |

| Metric | vLLM | nominal | KV-matched |
| --- | ---: | ---: | ---: |
| TTFT Mean | 102,063.9 | −32.0% | −7.7% |
| TTFT P99 | 224,846.1 | −30.7% | −8.8% |
| TPOT Mean | 56.0 | +18.5% | −7.2% |
| TPOT P99 | 96.5 | +22.6% | −8.7% |
| Latency P99 | 246,860.3 | −26.1% | −8.6% |

Correcting only the memory accounting moved mean \|error\| by **−13.28pp**, improving 13 of 15
metrics. This is the empirical confirmation of D10: it is not a bookkeeping nicety, it is the
single largest error term on a memory-constrained device.

Three things to carry forward:

- **The regime is saturated.** TTFT averages 102 s here versus 7 s on the RTXPRO6000 — the A5000
  cannot keep up with this workload, so queueing dominates and small throughput errors amplify
  into large latency errors. These percentages are not comparable to upstream's sub-3% claim,
  which was measured in an unsaturated regime. A lighter workload is needed before attributing
  the residual to the profile.
- **After KV matching the residual is uniformly negative** (−5.8% to −19.4%): the simulator
  consistently under-predicts on A5000. On RTXPRO6000 the residual was uniformly *positive*. The
  sign flip means these are different error sources, not one miscalibrated constant, so a single
  per-hardware linear correction (§5.8) will not cover both.
- **Attribution of the remaining 9.26%**: about 1.8pp is grid density (D11, measured
  independently), leaving ~6pp that is A5000 profile quality, hardware/toolchain difference
  (CUDA 12.8/Ampere vs 13.0/Blackwell), or the saturated regime. These are not yet separated.

The two metrics that got *worse* under KV matching (TPOT P90 +4.4pp, TPOT P95 +14.1pp) were near
zero error in the nominal run only because the bias was crossing zero there; both are firmly
negative afterwards, consistent with the rest.

## 3. The A5000 profiling path — closes `deviations.md` D4

A fresh bench run on this machine would compare real A5000 against a simulator configured with the
**RTXPRO6000** profile, which is meaningless. A meaningful local sim-vs-real loop needs an A5000
profile bundle first. That is worth doing on its own merits: `profiler/perf/` currently holds one
hardware class (D4), and this is the only second class we can measure honestly rather than import.

```bash
# profiler/profile.sh, then run it
MODEL="meta-llama/Llama-3.1-8B"
HARDWARE="A5000"
TP_DEGREES="1,2"
MEASUREMENT_ITERATIONS=3
```

The profiler boots vLLM at `tensor_parallel_size=1` regardless of the TP degree being profiled and
emulates per-rank shapes by dividing `SHARD_FIELDS` via `hf_overrides`, leaving collectives to
ASTRA-Sim. So TP=2 data is obtainable even though only TP=1 fits comfortably in memory.

Output lands at `profiler/perf/A5000/meta-llama/Llama-3.1-8B/<variant>/tp<N>/` and is consumed by
setting `"hardware": "A5000"` in a cluster config.

**Unverified assumption**: `docs/docs/profiler/adding-hardware.md` lists A100 and RTX 6000 Ada as
vLLM 0.19.0-supported but does not name the A5000. A5000 is Ampere (sm_86) and bf16-capable, so
support is expected — confirm with a trivial vLLM boot before committing to the profiling run.

Sequencing note: this produces a *measured* second hardware class, but it does not produce H100 or
Ascend data. D4's open question — where H100 / Ascend performance data comes from — stands
regardless.

## 4. Status against the work order

| Phase 0 item 3 requirement | Status |
| --- | --- |
| Reproduce `bench/` validation | **Done** — all three examples (Llama-3.1-8B, Qwen3-32B, Qwen3-30B-A3B), simulator side vs committed real-vLLM data, byte-identical to upstream |
| If no GPU: document commands + environment | Documented in §2 — GPUs exist but are too small for 2 of 3 models and vLLM is not installed |
| Never label unmeasured hardware as measured | Held — §2 states the provenance of every number explicitly |
