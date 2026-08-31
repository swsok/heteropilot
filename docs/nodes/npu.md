# Node profile — the NPU node

*Read this only after `bash scripts/whichnode.sh` says `detected node : npu`. If it
says anything else, this file describes a machine you are not on.*

Moved here from `CLAUDE.md` on 2026-08-28, because a committed file that asserts
"this machine has X" is true on at most one of the three nodes this project runs
on, and it had already misled a session into believing an 8×A40 box had no GPU.

**Everything below was true when written. Re-check with the detector — the RNGD
driver has re-enumerated three times now, and the card count has gone 4 → 3 → 4.**

## Inventory (as reported by vendor tools, 2026-08-25)

96 cores, 1.5 TB RAM, Python 3.10.12, `cmake` 4.3.2 / `g++` 11.4.0 / `protoc` 3.12.4.
**No NVIDIA GPU** — `nvidia-smi` cannot reach a driver.

- **4 × Rebellions ATOM** — `RBLN-CA22`, `/dev/rbln0..3`, PCI `83/84/c3/c4:00.0`,
  15.7 GiB each, KMD 3.0.0, ~19 W idle.
- **4 × FuriosaAI RNGD** on PCI `03/04/44/45:00.0`, 47.5 GiB and 8 PEs each,
  firmware `2026.3.0`, ~40 W idle. Torch addresses them as PrivateUse1 devices
  `rngd:0..31` (card × 8 PEs); `/dev/rngd/npu<N>pe<a>-<b>` nodes allow fusing 2 or
  4 PEs.

> **Device numbering is not stable, in both directions.** On 2026-08-27 npu2 (PCI
> `44:00.0`) was gone from the PCI bus and `furiosa-smi` entirely, and torch
> renumbers densely over the cards that remain — so `rngd:16` was npu3 and
> `rngd:24` did not exist. **The 2026-08-28 reboot brought npu2 back**: four cards
> again, so `rngd:16..23` is npu2 once more and `rngd:24` exists — confirmed by
> pinning a load to `rngd:16` and reading `furiosa-smi ps`, which said `npu2:0`.
> `card_of()` in `experiments/scripts/rngd_device_facts.py` resolves through live
> sysfs and returns the right answer in every one of these states without
> modification. Never compute a card from `index // 8`.
>
> Consequence for artifacts: `outputs/rngd_profile/parallel_bandwidth.json` records
> `device: rngd:16, card: npu3`, correct when measured. Today that index is npu2.
> **The card label is the durable fact; the index is not.**

## Who holds the devices

**This is a shared Kubernetes node and another tenant's workload owns most of it.**
`rngd_pd.serving.cluster` pods (P/D-disaggregated serving, one pod per chip, root,
under `kubepods-burstable-*`) have held RNGD npu0/1/2 via `--chip 0/1/2` and all
four ATOMs. Do not drain devices or kill those pods.

`furiosa-smi ps` and `rbln-stat` **under-report**, because the holders are pods.
The reliable check:

```bash
cat /sys/class/rngd_mgmt/rngd\!npu<N>pe<M>/alloc_status   # non-empty == claimed
```

Availability has changed three times: 2026-08-25 only npu3 was allocatable;
2026-08-27 every PE on all three surviving cards was free and `furiosa-smi ps` was
empty; **2026-08-28, after the reboot, all 32 PEs across four cards and all four
ATOMs are free**. Only the `--role gateway` pod restarted, and it holds no device.
This can revert without warning — check, do not assume. For ATOM the equivalent
check is `rbln-smi -j` and its `contexts` list.
Full holder table: `docs/hardware_roadmap.md` "Who holds the NPUs".

**NUMA, worth knowing before measuring any host↔device path:** all RNGD cards are
on **node 0**, all four ATOMs on **node 1**. A cross-vendor RNGD↔ATOM path
therefore crosses sockets, and host buffers are not bound by default — an
uncontrolled variable in every host-bandwidth number committed so far.

## Vendor runtimes — split and mutually incompatible

System `dist-packages` holds `vllm 0.13.0+cpu`, `vllm_rbln 0.10.2.post1`,
`rebel-compiler` 0.10.2, `tvm 0.20.dev0`, `transformers 4.57.6`. User `~/.local`
holds `furiosa-llm` / `furiosa-torch` 2026.2.0, `torch 2.10.0+cu128`,
`transformers 5.1.0`.

The user-site `transformers` breaks system vLLM. **One venv per vendor.** Never
install vLLM into `.venv` — that is the planner/analytical-sim environment and it
needs no device. Details and routes forward: `docs/hardware_roadmap.md`
"First access" and its 2026-08-28 update.

**The `rebel-compiler`/`tvm` breakage was not a version conflict, and it is fixed.**
`rebel-compiler` vendors its own TVM and declares no `tvm` dependency at all. What
actually happened was a **broken partial upgrade**: `dist-packages` carries two
dist-info directories (0.10.2 and 0.11.0) and the tree is a mixture. Hashing every
file against the 0.11.0 `RECORD` already on disk: `rebel/` 289 ok, 0 bad;
`rebel_compiler.libs/` 2 ok, 0 bad; `tvm/` 602 ok, **24 bad, 1 missing** — all 24
stale 0.10.2 files, all under `tvm/`. 0.11.0's `rebel.core.torch_compile` imports
`set_data_ptr_name_overrides` from `tvm.relay.frontend.pytorch`, which only the
0.11.0 copy has. `optimum_rbln` has the same two-dist-info problem.

`.venv-rbln` repairs it **without touching the shared system install** (there is no
outbound network here, so the fix reuses a hash-verified 0.11.0 tree from elsewhere
on the machine) and drops `~/.local` from `sys.path` via a `.pth`, making the
one-venv-per-vendor rule mechanical rather than remembered. `.venv-rbln-vllm`
additionally carries the vLLM stack (torch 2.11.0+cpu, transformers 5.8.1,
vllm 0.22.0, vllm_rbln 0.11.0), which is what `RblnPlatform` needs. Both are
gitignored; rebuild recipe in `docs/hardware_roadmap.md`.

ATOM-specific traps, each of which cost a session:

- **`rebel` must be imported after `torch`**, same reason and same `# isort: off`
  guard as `furiosa.torch`.
- **`rebel.device_count()` returning 0 on healthy cards means stale driver state,
  not packaging.** Before the 2026-08-28 reboot every card reported `npu: 0` with
  `group_id` 1–4 and a one-column `topology`; afterwards `npu` 0–3, `group_id` 0
  and a full 4×4 matrix. Read `rbln-smi -j` and sysfs `topology` first — a
  collapsed `npu` index is the tell, and the fix is a reload, which on this shared
  node is the owner's call.
- **Never subtract a constant host round-trip floor.** Per-call cost scales with
  bytes moved: 6.4 µs at 16 B, 56.3 at 2 KB, 999.7 at 8 MB (deviations D20).

RNGD-specific traps that have each cost a session:

- `furiosa.torch` **must** be imported after `torch`, behind the `# isort: off`
  guard in `rngd_device_facts.py`. `ruff check --fix` reordering it breaks every
  run with a circular-import error (fixed in `46f0c70`).
- Multi-PE work uses **one subprocess per PE**, not threads — nothing establishes
  that `furiosa.torch` allows several PE contexts in one interpreter, and
  `start_load()` / `measure_parallel_bandwidth()` both prove the subprocess pattern
  on this hardware.

## What is measured here, and what is not

Measured and committed (do not re-measure): the RNGD perf bundles
(`profiler/perf/RNGD/` layerwise, `profiler/perf/RNGD-CARD/` from the vendor's EDF
profiler), per-PE power `board = 38.01 + 32.71 × PEs`, the on-package all-reduce
(115 µs/layer at TP=8), host↔PE bandwidth, and both calibrations.

**ATOM is now usable, partly measured, and still not profiled.** As of 2026-08-28
the stack works (a compiled Linear runs on all four cards, matching CPU to
3.75e-03) and `memory_gb` (15.047 GiB largest single allocation, against a 15.719
GiB card) plus the whole `power:` block (idle 19.44 W, active 68.73 W at 95.1 %
utilisation) are **measured** — `experiments/results/atom_device_facts.md`.

But there is **no `profiler/perf/ATOM/` bundle**, and `rbln_atom.yaml` therefore
keeps `sim_hardware: null` and empty `supported_models`, so ATOM fails loud and
stays out of candidate generation and out of Exp 4. Layerwise profiling is blocked
on the instrument, not the hardware: this card's per-call host I/O exceeds the
kernels being measured, and the device tracer's protobuf schema is undocumented.
Deviations **D20** and `experiments/results/atom_layerwise_blocked.md` have the
evidence and the three subtraction schemes that failed.

`memory_bandwidth_gbps` is still a **placeholder** — do not cite it.

**Do not run CUDA `bench/` here** — there is no NVIDIA driver. Committed A40 and
A5000 artifacts remain valid measurements of *those* machines; do not re-run,
extend, or relabel them here.

## Open work that needs this node

- `docs/npu_concurrency_envelope_work_order.md` — c64/c128 concurrency run. c32 is
  the highest ever measured, and the card-fixture sweep assumes ~76 concurrent
  sequences per card.
