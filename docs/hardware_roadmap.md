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

### UPDATE 2026-08-27 — the tenants left, a card vanished, and the ATOM stack is repaired

Everything in the two sections above describes 2026-08-25. Re-checked on
2026-08-27, three of its load-bearing facts are no longer true. Re-verify before
planning any run; this machine's device state has now changed twice in three days.

**The other tenant's workload is gone.** Of the pod table above, only PID 6945
(`--role gateway`) survives, and it holds no device. Every per-chip pod
(`--chip 0/1/2`, and all four `--backend atom` roles) has exited. Consequently:

- **Every RNGD PE is free.** All 8 PEs on npu0, npu1 and npu3 return an empty
  `alloc_status`, and `furiosa-smi ps` is empty. The "only npu3 is allocatable"
  constraint is lifted.
- **All four ATOMs are free.** `rbln-stat` shows `0.0B / 15.7GiB` used, 0.0 %
  util and an empty context table on every card; `rbln-smi -j` reports all four
  `status: normal` with `contexts: []`, and `/dev/rbln0..3` all accept `O_RDWR`.
  The leftover 13–14 GiB contexts recorded on 2026-08-25 are gone. So route 2 is
  no longer blocked on device availability — only on the runtime (below).

**RNGD npu2 (PCI `44:00.0`) has left the PCI bus entirely.** Not busy, not
`rev ff` as on 2026-08-25 — absent. No `/sys/class/rngd_mgmt/rngd!npu2*` node, no
`lspci` entry, no `furiosa-smi` row. Three RNGD cards remain: npu0, npu1, npu3.

**Torch numbers RNGD devices densely over the cards that are present.** This is
the trap the missing card creates: with npu2 gone, `rngd:16..23` is **npu3**, and
`rngd:24` — the device id every earlier document names — **does not exist**,
failing with `Expected allocator != nullptr to be true`. Confirmed by pinning a
load to `rngd:16` and reading `furiosa-smi ps`, which reported `npu3:0`. Anything
computing a card as `device_index // 8` is now wrong; `card_of()` in
`experiments/scripts/rngd_device_facts.py` was fixed to resolve through sysfs
enumeration instead. **Never assume the arithmetic — re-check the mapping.**

**NUMA, newly recorded.** All three RNGD cards are on **NUMA node 0**; all four
ATOMs are on **NUMA node 1** (`03/04/45:00.0` vs `83/84/c3/c4:00.0`). Host buffer
placement is not bound by default, so a cross-vendor RNGD↔ATOM KV path crosses
sockets. This is a variable in any host-bandwidth measurement and was not
controlled in the ones committed so far.

#### The ATOM (Rebellions) stack — root cause found and repaired

The `rebel-compiler` / `tvm` conflict recorded below was **not** a version
mismatch between two separately installed packages. `rebel-compiler` vendors its
own TVM and declares no `tvm` dependency at all. What actually happened is a
**broken partial upgrade**: `/usr/local/lib/python3.10/dist-packages` carries
*two* dist-info directories, `rebel_compiler-0.10.2` and `rebel_compiler-0.11.0`,
and the tree is a mixture of both.

Verified by hashing every file against the 0.11.0 `RECORD` manifest already on
disk:

| top-level | ok | bad | missing |
| --- | ---: | ---: | ---: |
| `rebel/` | 289 | 0 | 0 |
| `rebel_compiler.libs/` | 2 | 0 | 0 |
| `tvm/` | 602 | **24** | **1** |

So `rebel/` is a clean 0.11.0 and the vendored `tvm/` is 0.11.0 with **24 stale
0.10.2 files** left behind, all under `tvm/`. 0.11.0's `rebel.core.torch_compile`
imports `set_data_ptr_name_overrides` from `tvm.relay.frontend.pytorch`, which
exists only in the 0.11.0 copy — hence the `ImportError` that killed `import
rebel`, and with it `optimum.rbln` and `vllm_rbln`.

**Repair, without touching the shared system install.** There is no outbound
network on this box (`pypi.org` resolves but times out; `pypi.rebellions.ai` does
not resolve) and no wheel in the pip cache, so the fix reuses a
cryptographically complete 0.11.0 tree found elsewhere on the machine, validated
file-by-file against the 0.11.0 `RECORD` we already own — 923/923 files matching,
0 bad, 0 missing. `.venv-rbln` is created with `--system-site-packages` and the
verified `tvm/` and `rebel_compiler.libs/` copied into its site-packages, where
they shadow the broken system copies. **System `dist-packages` is unmodified**,
which matters on a shared node.

The venv also **excludes the user site** (`~/.local`) via a `.pth` file, because
that is where `furiosa-torch` keeps `torch 2.10.0+cu128` and `transformers 5.1.0`
— which shadow the `torch 2.9.1+cpu` / `transformers 4.57.6` that `optimum-rbln`
pins. This is the "one venv per vendor" rule made mechanical rather than
remembered.

Result — the whole chain imports, where previously all three died at `import rebel`:

```
$ .venv-rbln/bin/python -c "import rebel, optimum.rbln, vllm_rbln"
rebel          OK  0.11.0
optimum.rbln   OK  0.11.0.post1
vllm_rbln      OK
```

#### Rebuilding `.venv-rbln`

`.venv-rbln/` is gitignored (923 files, ~440 MB). To recreate it:

```bash
# 1. Verify the system tree really is 0.11.0-with-stale-tvm before repairing it.
python3 - <<'PY'
import base64, hashlib, csv, os
D = "/usr/local/lib/python3.10/dist-packages"
h = lambda p: "sha256=" + base64.urlsafe_b64encode(
    hashlib.sha256(open(p, "rb").read()).digest()).rstrip(b"=").decode()
rec = {r[0]: r[1] for r in csv.reader(
    open(f"{D}/rebel_compiler-0.11.0.dist-info/RECORD")) if len(r) >= 2 and r[1]}
bad = [f for f, e in rec.items()
       if not os.path.exists(os.path.join(D, f)) or h(os.path.join(D, f)) != e]
print(len(bad), "files not matching 0.11.0:", bad[:5])
PY

# 2. Build the venv on top of the system packages, then shadow the broken bits.
python3 -m venv --system-site-packages .venv-rbln
DST=.venv-rbln/lib/python3.10/site-packages
SRC=<a directory whose tvm/ and rebel_compiler.libs/ verify against that RECORD>
cp -a "$SRC/tvm" "$SRC/rebel_compiler.libs" "$DST/"

# 3. Drop the user site, which holds furiosa's torch 2.10 / transformers 5.1 and
#    shadows the torch 2.9.1+cpu / transformers 4.57.6 optimum-rbln pins.
printf "%s\n" "import sys; sys.path[:] = [p for p in sys.path if '/.local/lib/python' not in p]" \
    > "$DST/_zz_no_user_site.pth"

# 4. Verify.
.venv-rbln/bin/python -c "import rebel, optimum.rbln, vllm_rbln; print(rebel.__version__)"
```

Step 2's `SRC` must be checked against the `RECORD` **before** copying — that
manifest is the only thing making a copy from elsewhere on the machine
trustworthy. Do not repair system `dist-packages` in place; it is shared.

#### Still blocked: the runtime enumerates zero ATOMs

**`rebel.device_count()` returns 0 and `rebel.npu_is_available()` is False**, on
a machine where all four cards are healthy and idle. This is *not* caused by the
repair, and not a packaging problem at all:

- It reproduces in a **separate, self-consistent 0.10.1 venv** that predates this
  work, so it affects at least 0.10.1 and 0.11.0 alike.
- The driver side is healthy: `status 0`, `dram_used 0`, `qstat 4/4/(4/4)/4`,
  `fw_ver`/`smc_ver`/`kernel_version` all 3.0.0, distinct `group_id` 1–4.
- The runtime *does* reach the driver — `get_kmd_version()` returns 3.0.0, inside
  the compiled compat range `[3.0.0, 4.0.0)`.
- It *does* enumerate: `strace` shows it exec'ing `/usr/bin/rbln-smi -j`
  successfully, opening all four `/dev/rbln*`, and reading every sysfs attribute.
  `rbln-smi -j` itself returns 4 devices, `status: normal`, `contexts: []`.
- **No permission error anywhere** in the trace — the device nodes are `0666`.

One anomaly worth chasing: every device reports **`"npu": 0`** in `rbln-smi -j`
(and NPU column `0` in `rbln-stat`, above a stray `N/A` row), while `group_id` is
correctly 1–4. If the runtime keys its device map on that field, four devices all
claiming index 0 would explain the collapse. Whether that is a driver quirk, an
`rbln-smi` schema drift, or a red herring is **not decidable without root or the
vendor** — the debug channels (`RBLN_DEBUG_LEVEL`, `RBLN_COMPILER_LOG_LEVEL`) are
refused by this deploy build, and `RBLN_DEVICES` / `RBLN_DUMMY_DEVICE` do not
change the count.

**So ATOM profiling remains blocked, but the blocker has moved** — from a broken
Python install (fixed) to device enumeration in the vendor runtime. Next step is
root access or a vendor question, not more packaging work. `rbln_atom.yaml` keeps
`sim_hardware: null` and empty `supported_models`, and every ATOM number in the
repo stays placeholder (absolute rule 3) — the hardware being present, idle and
importable changes none of that until it is actually profiled.

### UPDATE 2026-08-28 — a reboot cleared both blockers, and the vanished card returned

The machine was rebooted at 04:48 on 2026-08-28. It changed two things that the
2026-08-27 section above records as blockers, and both changed for the better.

**ATOM is usable.** `rebel.device_count()` now returns **4**, `npu_is_available()`
is `True`, `get_npu_name()` is `RBLN-CA22`, and a model compiled with
`rebel.compile_from_torch` runs on **all four devices** (checked individually,
max abs error 3.75e-03 against the CPU reference — ordinary reduced-precision
agreement). The `.venv-rbln` repair from 2026-08-27 was necessary but not
sufficient; the reboot supplied the rest.

**The zero-device blocker was stale driver state, and the `npu: 0` anomaly was
the tell.** Comparing the driver's own view either side of the reboot:

| | before reboot | after reboot |
| --- | --- | --- |
| `npu` index (`rbln-smi -j`) | `0, 0, 0, 0` | `0, 1, 2, 3` |
| `group_id` (sysfs) | `1, 2, 3, 4` | `0, 0, 0, 0` (one group) |
| `topology` (sysfs) | `rbln0 0` — one column | `rbln0 0 4 4 4` — full 4×4 matrix |

So the driver had never completed device-group and topology initialisation: each
card sat in its own group with no inter-device distances and a collapsed npu
index, which is exactly the shape that would make a runtime keying on `npu`
count one device or none. The most likely cause is the abrupt exit of the
root-owned `rngd_pd.serving.cluster` pods that held the cards — they left on
2026-08-27, and the driver never recovered until it was reloaded by the reboot.

*Diagnostic worth keeping:* if `device_count()` returns 0 again on healthy cards,
read `rbln-smi -j` and the sysfs `topology` **first**. A collapsed `npu` index or
a one-column topology means driver state, not packaging, and the fix is a
reload/reboot — which on this shared node is the owner's call, not ours.

The post-reboot `topology` matrix is also the first real inter-ATOM distance data
this repo has seen: diagonal 0, every off-diagonal 4, i.e. uniform. Relevant to
the Level-2 topology model, and still not a measurement — it is the driver's
nominal distance metric, not a bandwidth.

**RNGD npu2 came back.** `44:00.0` is on the bus again, so there are **4 RNGD
cards** and all **32 PEs are free**. This is the third distinct device set in four
days (4 cards on 08-25 → 3 on 08-27 → 4 on 08-28).

The consequence is the numbering trap in reverse: with all four cards present,
`rngd:16..23` is **npu2** again and `rngd:24..31` is npu3 — confirmed by pinning a
load to `rngd:16` and reading `furiosa-smi ps`, which reported `npu2:0`. While
npu2 was missing, that same index was npu3. **The index→card mapping is not
stable across reboots**, which is precisely why `card_of()` resolves it through
live sysfs enumeration rather than `index // 8`; that function returns the right
answer in both states without modification.

*Provenance note.* `outputs/rngd_profile/parallel_bandwidth.json` records
`device: rngd:16`, `card: npu3`, measured 2026-08-27. That was correct when
measured — npu2 was off the bus. After this reboot `rngd:16` is npu2, so the
artifact's device id no longer points at the card it was measured on. The card
label is the durable fact; the index is not. Re-running that measurement today
would land on a different physical card unless `--device rngd:24` is used.

**Only the gateway pod restarted.** `rngd_pd.serving.cluster --role gateway` is
running again and holds no device; no per-chip pod came back. Every RNGD PE and
every ATOM is unclaimed. As always this can change without warning — re-check
before planning a run.

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

1. **RNGD via `CsvProfileImporter`** — ✅ **DONE 2026-08-25.**
   `experiments/scripts/profile_rngd.py` walks the canonical layers of
   `profiler/models/llama.yaml`, compiles each through `furiosa.torch.backend`
   with an `RNGDProfiler`, and `run_rngd_profile.py` shards the grid one worker
   per PE (24 workers finish a TP degree in 105–315 s, against 40–60 min
   serially). Imported to `profiler/perf/RNGD/meta-llama/Llama-3.1-8B/bf16/` at
   tp1/tp2/tp4/tp8, and a 20-request simulation runs on it. `profiler/` was not
   touched. **This is the project's first measured NPU number** — everything
   before it was SIM-PROXY or placeholder.

   What it does *not* settle: the power block's per-PE split (below), end-to-end
   `furiosa-llm` serving, and whether these layer implementations track vLLM's
   fused kernels closely enough for a like-for-like GPU-vs-NPU comparison.
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

### What the first measured RNGD profile settled, and what it did not

Measured facts now in `profiles/accelerators/furiosa_rngd.yaml`:

| field | value | how |
| --- | --- | --- |
| `memory_gb` | 6.25 | largest bf16 block one PE accepts, bisected |
| `memory_bandwidth_gbps` | 200 | achieved HBM→PE read, from DMA span scaling |
| card power | idle 39.0 W / active 285.53 W / standby 265.0 W | all 8 PEs loaded |

**One accelerator is one PE, not one card.** `furiosa-llm build -tp` counts PEs
per TP group (default 8), the bundle's `tp<N>` holds one PE's shapes, and the
simulator picks `tp<N>` by the accelerator count in the TP group. So
ClusterSpecV2 must declare 8 accelerators per card. The simulator confirms the
consequence: TP=1 is refused because Llama-3.1-8B bf16 weights (14 GB) exceed
one PE's 6.25 GB, exactly why the vendor defaults to `-tp 8`.

**The open blocker for Exp 4 is power attribution, not latency.** Board power is
only measurable per card, but the simulator applies the profile's power block per
NPU instance. The profile divides the card totals by 8 and keeps
`source: placeholder`, because an even split was not measured and idle draw in
particular is a board cost that does not scale with PE count. Loading a single
PE reads 68 W against 285 W for the whole card — a 4× gap, so this is not a
rounding concern. **Resolve it by sweeping 1..8 loaded PEs and fitting the
marginal per-PE cost**; until then no tokens/J figure for RNGD should be quoted,
even though the latency bundle is sound.

**Two measurement traps worth not rediscovering.** Timing must use the *union* of
device spans, not their sum: on down_proj@256 the TuExec sum alone is 1603 µs
and DMA adds 964 µs, but the union is 1250 µs and equals the full timeline, so
tensor units run concurrently and DMA overlaps compute. And one process walking a
grid must free device tensors *and* call `torch._dynamo.reset()` between shots —
dynamo caches the compiled graph, which keeps the loaded EDF's device buffers
alive, so a large shot that succeeds leaves the PE too full for smaller ones
after it.
