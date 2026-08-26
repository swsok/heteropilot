# Handover — continuing on the A40 server

*Written 2026-08-26 for a Claude Code session that will `git pull` this repo on
the **A40 server** and continue. Read this first, then
`experiments/results/rngd_sim_vs_real_summary.md` and
`docs/hardware_roadmap.md` "First access".*

> The NPU server work is done and pushed. What is left needs a **GPU**, and there
> is no NVIDIA GPU on the NPU server (`nvidia-smi` cannot reach a driver). This
> file says exactly what to measure there and why it matters.

---

## 1. The one measurement that is blocked here

**GPU → host transfer bandwidth**, so the cross-vendor Prefill/Decode KV handoff
path can be priced end to end.

A GPU-prefill / NPU-decode split has no device-to-device route between vendors.
The KV cache goes **GPU → host → NPU**: two copies. Only the NPU leg is
measurable on the NPU server, and it was measured:

| streams | aggregate host → RNGD PE |
| ---: | ---: |
| 1 | 5.06 GB/s |
| 2 | 10.39 GB/s |
| 4 | 19.10 GB/s |
| 8 | **35.47 GB/s** (median 31.36) |

A single stream is only 5 GB/s, which is *below* the ~10 GB/s P/D adoption
crossing Exp 3 found — but a TP=8 decode island shards its KV across 8 PEs, so
the handoff uses the parallel path and sustains ~35 GB/s at 88 % of ideal
scaling. Raw data: `outputs/rngd_profile/host_bandwidth.json`.

So the NPU leg is **not** the bottleneck. Whether the whole path clears the
crossing depends on the GPU leg, which is what the A40 server has to answer.

### What to run there

```bash
# 1. GPU -> host, single stream and parallel across GPUs, mirroring the NPU method:
#    contiguous bfloat16 buffer, host-side perf_counter, best and median of N reps
#    after one warm copy, sizes 1/4/16/64/256 MB.
#    The NPU-side implementation to copy is measure_host_bandwidth() in
#    experiments/scripts/rngd_device_facts.py -- keep the method identical or the
#    two legs are not comparable.
# 2. Pinned vs pageable host memory BOTH ways. cudaHostAlloc changes this by
#    several x on CUDA and the NPU figures above are pageable, so report both and
#    say which the comparison uses.
# 3. Parallel across 2 and 4 GPUs, because the NPU leg scaled almost linearly and
#    the question is whether the GPU leg does too.
```

Then set `bandwidth_gbps` on the `fabric-*` links in
`experiments/configs/clusters/pd-rngd-gpu.yaml` to
`1 / (1/gpu_leg + 1/npu_leg)` for a serialised handoff, or `min(gpu_leg,
npu_leg)` if the implementation pipelines the two copies — and say in the file
which one was assumed. Right now those links carry **35 GB/s labelled
`placeholder`**, i.e. the NPU leg used as an upper bound on the whole path.

With a real number, re-run:

```bash
./experiments/scripts/run_exp_pd.sh          # Exp 3 network sweep + Exp 5 combos
PYTHONPATH=$PWD .venv/bin/python experiments/scripts/pd_slo_sweep.py \
    --service examples/service_specs/llama31-8b.yaml \
    --cluster experiments/configs/clusters/pd-rngd-gpu.yaml
```

---

## 2. State of the answer to "can mixed GPU+NPU P/D ever win?"

Established on the NPU server, from measurements:

**Per watt, RNGD wins both phases, so mixing cannot win on efficiency alone.**
Derived the same way from every committed bundle (tp1 `dense.csv`, 1 token =
bandwidth-bound, 2048 tokens = compute-bound):

| | achieved GB/s | TFLOP/s | W | GB/s per W | GFLOP/s per W |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNGD card (8 PE) | 1750 | 240 | 299.7 | **5.84** | **801** |
| A40 | 569 | 127 | 297.8 | 1.91 | 427 |
| A5000 | 694 | 104 | 227.6 | 3.05 | 457 |
| RTXPRO6000 | 1473 | 417 | 600 | 2.45 | 696 |

If one device dominates both axes, the efficiency optimum is homogeneous and
adding a GPU strictly lowers tokens/J. **Mixing can only pay when a constraint
binds**, and the candidates are:

1. **TTFT SLO.** Single-request prefill latency is capped by one island's
   compute, and RNGD is far behind per device: 2048-token `down_proj` takes
   8513 µs on one PE against 2010 µs on an A40. Below some TTFT there is no
   feasible all-RNGD plan and prefill must move to the GPU. Finding that
   threshold is `experiments/scripts/pd_slo_sweep.py`.
2. **Bucket quantisation, which taxes RNGD prefill only.** The vendor artifact is
   compiled for fixed `tokenwise_buckets` (1, 2, 4, …, 256, 384, 512, 1024), so a
   prompt pays for the buckets covering it: 399 tokens runs the 512 bucket.
   Measured over the 20-request trace: 27,408 charged tokens against 24,725
   actual, **+10.9 % aggregate, +20.7 % mean per request**. Decode always runs
   `input_size=1`, so it pays nothing. The vendor's own compilation model
   penalises exactly the phase a mixed split would move to the GPU.
3. **A hard power cap**, where the optimum is "buy the minimum GPU prefill
   capacity that meets TTFT, spend every remaining watt on RNGD decode".

Note the honest direction of the errors: the RNGD figures come from a harness
that measures **unfused single layers**, which the sim-vs-real check showed is
~26 % pessimistic on decode. So RNGD's per-watt advantage above is a **lower
bound**. Absolute latencies, however, carry ±30 % error, so the TTFT threshold in
(1) must come from simulation with calibration applied, not from these numbers.

---

## 3. A structural blocker found while building the experiment — do not lose this

**Cross-vendor P/D is unrepresentable whenever the two backends' feasible TP
degrees do not overlap**, and for this hardware pair they naturally do not.

`planner/candidate_generator.py` requires `tp_p == tp_d` for a P/D pair
(deviations D14: the simulator infers its topology as
`[npus_per_group, num_instances]` by integer division over the total device
count, so unequal instance sizes are unrepresentable). And:

- RNGD reaches only **tp ∈ {4, 8}** for Llama-3.1-8B, because a PE holds 6.25 GB
  against 14 GB of weights. tp=1 and tp=2 are correctly rejected on memory.
- An **NVLink-pair** A40 island offers only **tp ∈ {1, 2}**.

Those sets are disjoint, so the first run of the experiment emitted **no mixed
P/D candidate at all** — 3 representatives instead of 5, with both mixed combos
silently absent. Nothing errored; the combos just did not exist.

The fix in `experiments/configs/clusters/pd-rngd-gpu.yaml` is to bridge the
NVLink pairs over PCIe into **size-4** A40 islands, exactly as
`experiments/configs/clusters/exp1-a40-tp-sweep.yaml` already does, giving
tp ∈ {1, 2, 4} and making **tp=4 the shared degree** where GPU-P + NPU-D exists.

**Consequence for the paper, and for any future hardware pair:** a heterogeneous
P/D deployment is only expressible if the island sizes can be chosen to share a
TP degree. That is a real planner-level constraint on heterogeneity, not a
fixture detail, and it deserves to be stated wherever cross-vendor P/D is
claimed. If it becomes limiting, lifting it means addressing D14 — teaching the
compiler to emit non-uniform instance sizes — which is a simulator-side change.

---

## 4. What is already measured and needs no repeat

Do not re-measure these on the A40 server; they are committed.

| thing | where | note |
| --- | --- | --- |
| RNGD perf bundle, tp1/2/4/8 | `profiler/perf/RNGD/` | measured via `furiosa.torch`; tp8 is the weakest (40 CPU fallbacks vs 1 elsewhere) |
| RNGD per-PE power | `profiles/accelerators/furiosa_rngd.yaml` | `board = 38.01 + 32.71 × PEs`, R² 0.996, from a 0..8 loaded-PE sweep |
| RNGD sim-vs-real | `experiments/results/rngd_sim_vs_real_summary.md` | TTFT −32.6 %, TPOT +25.5 %; **opposite signs** |
| RNGD calibration | `profiles/calibration/rngd.yaml` | scoped to one workload bucket; do not extrapolate |
| host → RNGD bandwidth | `outputs/rngd_profile/host_bandwidth.json` | the NPU leg of the KV path |
| A40 profile + calibration | `profiles/accelerators/a40.yaml`, `profiles/calibration/a40.yaml` | measured on the A40 server previously; α 1.017 / 1.011 |

**The A40 numbers already in the repo are valid measurements of that server.**
Re-running the A40 profiler is not required for the P/D question and would only
be worth it if you want to refresh the bundle for a newer vLLM.

---

## 5. Environment on the A40 server

Follow `docs/HANDOVER_NPU.md` §2 for `.venv` (planner + analytical simulator);
it is machine-independent except for the NPU-specific §2b. Two traps that cost
time on the NPU server and will repeat:

- **`pydantic`, `pytest`, `ruff`, `mypy` are missing from the documented install
  list.** Without pydantic, `tests/conftest.py` fails at import and nothing
  collects. `CLAUDE.md` §Environment now includes them.
- **`astra-sim` is a submodule**; a plain clone leaves it empty and
  `scripts/compile.sh` fails. `git submodule update --init --recursive`, and
  `protoc` + `libprotobuf-dev` are system prerequisites.

For the profiler on CUDA you also need the `.venv-vllm` described in
`docs/HANDOVER_NPU.md` §2b — that part *is* CUDA-specific and applies on the A40
server, unlike on the NPU server where the vendor stacks live in the system
interpreter.

Expect `pytest` to report **284 passed**. If it reports 280 passed / 4 failed in
`tests/test_calibration.py`, the branch is missing
`fix/calibration-test-artifact-path` (a stale artifact path, not a real failure).

---

## 6. Branches pushed from the NPU server

| branch | what |
| --- | --- |
| `fix/calibration-test-artifact-path` | stale test path; restores 284 passing |
| `docs/npu-server-env-setup` | env bring-up log, NPU inventory, who holds the NPUs |
| `feat/rngd-profiling` | everything RNGD: harness, bundle, power, sim-vs-real, P/D fixture |

`feat/rngd-profiling` is the one to continue from; it is the longest chain and
contains the P/D fixture and both sweep drivers. Merge order does not matter —
they touch disjoint files apart from the docs.
