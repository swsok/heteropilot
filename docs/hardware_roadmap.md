# Hardware roadmap — incoming cluster access (recorded 2026-08-14)

The user has confirmed access, expected from the week of 2026-08-17 onward, to:

| Resource | Count | Backend | Status |
| --- | --- | --- | --- |
| A40 x8 GPU nodes | up to 8 nodes (up to 64 GPUs) | cuda | not yet reachable |
| Rebellions ATOM server | 4 devices | rbln (new) | not yet reachable |
| FuriosaAI RNGD server | 4 devices | furiosa (new) | not yet reachable |

This changes the project's constraint structure. Until now every non-RTXPRO6000
number was either locally measured on 2xA5000 or a placeholder; the plan below
records what the new hardware unblocks and in what order to bring it up.

## What it changes, by open item

**D4 (only one shipped hardware profile; NPU data source undecided) — resolution
path changes.** The work order assumed Ascend as the NPU target and CSV import
of externally measured data as V1. The concrete NPU targets are now ATOM and
RNGD, both physically accessible: profiles can be *measured*, not imported.
The `backend` field was always declared extensible (`cuda | ascend | <향후 추가>`);
`rbln` and `furiosa` are added as identifiers. The `ascend_target.yaml` stub
stays as a schema example but is no longer the expected Phase 3 vehicle.

**Phase 4 calibration model problem — solved.** Qwen3-32B (the work order's
headline model, ~64 GB bf16) does not fit local hardware, which made Phase 4
calibration impossible here. On A40 (48 GB) it serves at TP=2. Real-vLLM
calibration on the headline model becomes possible.

**Exp 3 (network sensitivity) gains ground truth.** Up to 8 multi-GPU nodes
means real inter-node collectives to validate the Level-1/Level-2 topology
models against, instead of simulator-only sweeps.

**Exp 4 (GPU vs NPU SLO-goodput/J) becomes a measured experiment.** This was
the largest "simulator-only, placeholder inputs" caveat in the paper plan.
Both NPU vendors position their parts on efficiency, which is exactly the
tokens/J axis this project optimizes.

## Bring-up order (work order §11 V1→V2 discipline)

1. **Inventory before anything** (each system, day one):
   - GPU nodes: `nvidia-smi -L`, `nvidia-smi topo -m` (NVLink bridge vs PCIe),
     NIC inventory (`ibstat` / `ethtool`), inter-node fabric and speed.
   - NPU servers: device count/memory via vendor tools (`rbln-stat`,
     `furiosactl` or equivalents — verify actual tool names on the machine),
     host topology.
   - Write each as a `ClusterSpecV2` YAML under `examples/clusters/` with
     per-field `source:` provenance. No number enters a profile that was not
     read off the machine or a vendor document (cite which).
2. **A40 profiling** (existing profiler works as-is; CUDA sm_86 like A5000):
   `python -m profiler profile meta-llama/Llama-3.1-8B --hardware A40 --tp 1,2,4,8`
   plus Qwen/Qwen3-32B. Use the x2 grid initially; densify later if the D11
   penalty (~2.2pp) matters for the experiment at hand.
3. **A40 real bench + calibration** (Phase 4 entry): repeat the A5000
   sim-vs-real protocol (docs/phase0_bench_plan.md) on one A40, then at TP=2/4,
   then across nodes. Re-measure the D10 KV budget on 48 GB.
4. **NPU V1 — serving-stack survey and CSV import.** Verify what actually runs:
   vLLM-compatible serving stacks exist for both vendors (vllm-rbln for ATOM,
   furiosa-llm for RNGD) but versions, supported models and metrics endpoints
   must be confirmed on the machines, not assumed. Measured latency data enters
   through `CsvProfileImporter` against `profiler/CONTRACT.md` (write the
   contract first — it is the §3.7 prerequisite and needs no hardware).
5. **NPU V2 — native profiling adapters** only after V1 produces a working
   planner loop with measured NPU envelopes.

## What stays true regardless

- TP/PP never crosses a backend boundary (absolute rule 2): A40 islands,
  ATOM islands and RNGD islands are separate; heterogeneity is replica- or
  role-level.
- Anything not yet measured on these machines is `source: placeholder` and the
  profile stubs added alongside this document say so field by field.
- The simulator predicts; the new hardware measures. Results must keep the
  label they were born with.

## Open questions to resolve at first access

- A40 intra-node interconnect (NVLink bridge pairs? PCIe only?) and inter-node
  fabric (IB? 100/200G? Ethernet?) — determines the ClusterSpecV2 link graph
  and whether Exp 3 has interesting bandwidth points.
- ATOM / RNGD: device memory and TDP as reported by the machine's own tools;
  serving stack versions; which of our target models they support at which
  dtypes (fills `supported_models`, which is deliberately empty in the stubs —
  empty means excluded from candidates until verified).
- Scheduler/queue system on the cluster (SLURM? bare SSH?) — affects the
  Phase 4 `deploy/` launcher design (§5.7 allows local/SSH only).

---

## First access — what the NPU server actually looks like (recorded 2026-08-25)

Access happened. The inventory below is what the machine's own vendor tools
report; nothing here is a profile yet, so every accelerator profile stub keeps
`source: placeholder` (absolute rule 3).

- **No NVIDIA GPU.** `nvidia-smi` cannot reach a driver. The A40 and A5000
  measurements already committed remain valid artifacts of *other* machines and
  must not be re-run, extended or relabelled here. Exp 4's "GPU" arm therefore
  has to reuse the committed measured A40 profile, not a fresh run.
- **4 × Rebellions ATOM** — `RBLN-CA22`, `/dev/rbln0..3`, PCI `83/84/c3/c4:00.0`,
  15.7 GiB each, KMD 3.0.0, ~19 W idle. `rbln-stat` also shows leftover 13–14 GiB
  contexts held by other users' `python3.10` processes; clear or avoid those
  devices before profiling.
- **4 × FuriosaAI RNGD** on PCI `03/04/44/45:00.0`, **47.5 GiB and 8 PEs each**,
  firmware `2026.3.0`, ~40 W idle, all four `alive`. Torch sees them as
  PrivateUse1 devices `rngd:0..31` (card × 8 PEs), and `/dev/rngd/npu<N>pe<a>-<b>`
  nodes allow fusing 2 or 4 PEs (no 8-PE fusion node; `furiosa-llm build`'s
  `-tp` default of 8 must therefore use the mesh path, `MeshKind` is
  `Mesh|Single` and currently `Single`).
  **Availability is partial: only npu3 (`rngd:24..31`) is allocatable.** Every PE
  on npu0/1/2 returns `EBUSY: Device or resource busy`.
  Re-check before each run — the driver re-enumerated at 06:26 on 2026-08-25 and
  changed what was visible (an earlier read of `44:00.0` showed PCI rev `ff` and
  `furiosa-smi` listed only three cards; both cleared after re-enumeration).

### Who holds the NPUs — this is a shared, actively used machine

`furiosa-smi ps` and `rbln-stat`'s context table under-report: the holders are
Kubernetes pods, so they are invisible to a per-user view. Identified from `ps`
+ `/proc/<pid>/cgroup` (all `kubepods-burstable-*` / `cri-containerd-*`, running
as root):

| PID | command | holds |
| --- | --- | --- |
| 17783 | `rngd_pd.serving.cluster --role prefill --backend rngd-full --C 256 --group 2 --chip 0` | RNGD npu0 |
| 17792 | `rngd_pd.serving.cluster --role decode --backend rngd --chip 1` | RNGD npu1 |
| 17773 | `rngd_pd.serving.cluster --role decode --backend rngd --B 4 --group 8 --batch-wait 8000 --sched static --admit-min 2 --chip 2` | RNGD npu2 |
| 10054 | `rngd_pd.serving.cluster --role prefill --backend atom --C 256` | one ATOM |
| 10137 / 10198 / 10233 | `rngd_pd.serving.cluster --role decode --backend atom --B 1 / 4 / 2` | three ATOMs |
| 6945 | `rngd_pd.serving.cluster --role gateway --targets-file /cfg/targets.json` | none (router) |

So someone is running a **P/D-disaggregated serving experiment** across both NPU
vendors on this box, one pod per chip. `--chip 0/1/2` maps 1:1 onto the three
`EBUSY` RNGD cards, and npu3 has no claimant — which is exactly the card that is
allocatable. Consequences:

- **Do not free these devices.** They belong to a live workload that is not ours;
  `furiosa-smi drain` and killing pods are both off-limits without the owner's
  agreement. npu3 alone is enough for the profiling route above.
- **All four ATOM cards are occupied.** So route 2 below is blocked on *device
  availability* as well as on the broken vendor install — fixing the wheels does
  not by itself give you an ATOM to profile.
- A RNGD k8s device plugin is registered (`/var/lib/kubelet/device-plugins/rngd.sock`),
  so allocation is scheduler-driven; the authoritative per-pod device map is
  `kubelet_internal_checkpoint`, readable only as root.
- **Unprivileged availability check** (no root, no vendor tool): a PE is claimed
  iff `/sys/class/rngd_mgmt/rngd!npu<N>pe<M>/alloc_status` returns a non-empty
  allocation table. npu0/1/2 print a (zeroed) table; npu3 prints nothing. This
  matched the allocation sweep exactly on two separate runs and is the cheapest
  pre-flight check before starting a profiling job.

### The serving stacks are installed, split across two site-packages, and conflict

| stack | location | version |
| --- | --- | --- |
| `vllm`, `vllm_rbln`, `rebel`/`rebel-compiler`, `tvm`, `transformers` | system `/usr/local/lib/python3.10/dist-packages` | 0.13.0+cpu, 0.10.2.post1, 0.10.2, 0.20.dev0, 4.57.6 |
| `furiosa_llm`, `transformers`, `torch` | user `~/.local/lib/python3.10/site-packages` | 2026.2.0, **5.1.0**, 2.10.0+cu128 |

Three defects follow from that split, all confirmed by running the profiler:

1. **user-site `transformers 5.1.0` shadows system 4.57.6 and breaks system
   vLLM 0.13.0** — `ImportError: cannot import name 'ALLOWED_LAYER_TYPES' from
   'transformers.configuration_utils'` (removed in transformers 5.x).
   Workaround: `PYTHONNOUSERSITE=1`.
2. **`rebel-compiler 0.10.2` does not match the installed `tvm 0.20.dev0`** —
   `ImportError: cannot import name 'set_data_ptr_name_overrides' from
   'tvm.relay.frontend.pytorch'` (the symbol is absent from that file). Because
   vLLM auto-activates the `rbln` platform plugin, this crashes *every* vLLM
   import, not just ATOM work. Workaround to get vLLM up at all:
   `VLLM_PLUGINS=""`. Real fix: install the patched TVM that Rebellions pairs
   with this `rebel-compiler` (vendor index), or match the two versions.
3. **RNGD has no vLLM platform plugin.** The only entry point in
   `vllm.platform_plugins` is `rbln`; `furiosa_llm` is a separate API. So
   `profiler/core/engine.py`'s `from vllm import LLM` cannot drive an RNGD at
   all, whatever the env vars.

`PYTHONNOUSERSITE=1 VLLM_PLUGINS="" PYTHONPATH=$PWD python3 -m profiler --help`
does work today — the CLI is intact; only the device paths are blocked.

### RNGD can be layerwise-profiled without vLLM — verified on hardware

RNGD does not use vLLM at all; `furiosa-llm` is its own stack (`build` compiles a
bucketed AOT artifact, `serve` serves it, `--prefill-buckets` /
`--decode-buckets` take `batch_size,context_length`). But underneath it,
`furiosa.torch` exposes a **torch.compile backend, a PrivateUse1 device, and a
torch-profiler-compatible `RNGDProfiler`** — and that is enough for the §3.7 CSV
contract. What was actually run on `rngd:24` (npu3):

- **Eager mode works but is useless for timing.** A module and input move to the
  device and forward correctly, but `furiosa.torch.backend.eager.run_aten_op`
  JIT-compiles *every aten op per call* and calls `run_by_rngd(...)` **without**
  a `profiler=`, so no device spans are recorded and the wall time is the
  compiler's. Measured: 45.7 s of CPU for 3 forwards of a small attention block,
  entirely dynamo/AOT-dispatch. Do not time anything this way.
- **`torch.compile(m, backend=furiosa.torch.backend)` inside
  `furiosa.torch.config.profiler_context(RNGDProfiler())` does record device
  spans** — `backend/torch_compile.py` passes `profiler=config.get_profiler()`
  into `run_by_rngd`, which calls `generate_profiles` → execute →
  `load_spans_from_profiles`. Measured: 85 device spans over 5 forwards, and the
  per-iteration CPU cost collapses from 45.7 s to ~2 ms (compile is cached).
- **The spans are hardware-unit-level, not op-named**: `Renegade::TuExec`
  (tensor-unit execution, ~12 per forward), `DMA (n)`, `Task`. They carry
  durations but no aten/nn.Module identity, so you cannot decompose one big graph
  into canonical layers from the trace. `key_averages()` shows CPU columns only —
  the device spans live in `prof.tuc_profiles` and are merged on
  `export_chrome_trace`, so read the exported JSON (or that list), not the table.
- **Therefore: compile one canonical layer per graph and sum its spans.** That
  sidesteps the naming problem entirely and produces exactly
  `layer, tokens, time_us`. Verified with a single `nn.Linear(512, 2048)` bf16 on
  `rngd:24`, summing `Renegade::TuExec` over 5 reps:

  | tokens | TuExec µs/forward | DMA µs/forward |
  | ---: | ---: | ---: |
  | 64 | 4.80 | 12.18 |
  | 128 | 18.84 | 14.35 |
  | 256 | 38.67 | 17.09 |
  | 512 | 59.14 | 22.24 |

  Compute scales with token count as the contract's `dense.csv` assumes; DMA
  grows far more slowly. **These four numbers are a mechanism proof on a
  synthetic layer, not a profile** — they must never be written into a profile
  file or a figure (rule 3). What they establish is that the measurement path
  exists and returns sane, monotonic device time.

### Consequence for the §3 NPU work path in `docs/HANDOVER_NPU.md`

The vLLM layerwise profiler cannot profile either device as installed, but the
cheapest route is no longer the ATOM one. Ordered by how much they unblock:

1. **RNGD via `CsvProfileImporter`** (Phase 3 V1, `profiler/core/importer.py`) —
   a standalone script in a furiosa-only venv walks the canonical layers of
   `profiler/models/<model_type>.yaml`, compiles each through
   `furiosa.torch.backend` with an `RNGDProfiler`, sweeps the contract's token /
   sequence / attention grids, and writes the CSV bundle for import. **The
   mechanism is verified working on hardware today**, needs no vendor fix, and
   leaves `profiler/` untouched. This is the shortest path to a *measured* NPU
   profile and to Exp 4.
2. **ATOM via vLLM** — blocked on a **broken vendor install**, not a design gap:
   `importlib.metadata` resolves `rebel-compiler` to **0.11.0** while
   `vllm_rbln 0.10.2.post1` and `optimum-rbln 0.10.2` expect 0.10.2, and *both*
   `rebel_compiler-0.10.2.dist-info` and `-0.11.0.dist-info` are present with
   both RECORDs claiming `tvm/relay/frontend/pytorch.py` — a partial upgrade left
   `tvm/` mixed. Fix by installing one consistent set of the three packages into
   a clean rbln-only venv. Cheap once the right wheels are at hand, and it is the
   only device with a vLLM plugin, so it reuses `profiler/` as-is.
3. **RNGD via a native adapter** — a `furiosa.torch` engine backend behind
   `profiler/core/engine.py`, replacing the `from vllm import LLM` dependency.
   This is what route 1 would graduate into. V2 per the bring-up order above; do
   not start it before V1 produces a working planner loop.

Whichever route: **one venv per vendor**. The two stacks cannot share an
interpreter — that is what defect 1 is. `.venv` stays vLLM-free regardless.
