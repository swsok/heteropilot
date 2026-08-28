# Node profile — the NPU node

*Read this only after `bash scripts/whichnode.sh` says `detected node : npu`. If it
says anything else, this file describes a machine you are not on.*

Moved here from `CLAUDE.md` on 2026-08-28, because a committed file that asserts
"this machine has X" is true on at most one of the three nodes this project runs
on, and it had already misled a session into believing an 8×A40 box had no GPU.

**Everything below was true when written. Re-check with the detector — the RNGD
driver has re-enumerated at least twice, and npu2 has since left the PCI bus
entirely.**

## Inventory (as reported by vendor tools, 2026-08-25)

96 cores, 1.5 TB RAM, Python 3.10.12, `cmake` 4.3.2 / `g++` 11.4.0 / `protoc` 3.12.4.
**No NVIDIA GPU** — `nvidia-smi` cannot reach a driver.

- **4 × Rebellions ATOM** — `RBLN-CA22`, `/dev/rbln0..3`, PCI `83/84/c3/c4:00.0`,
  15.7 GiB each, KMD 3.0.0, ~19 W idle.
- **4 × FuriosaAI RNGD** on PCI `03/04/44/45:00.0`, 47.5 GiB and 8 PEs each,
  firmware `2026.3.0`, ~40 W idle. Torch addresses them as PrivateUse1 devices
  `rngd:0..31` (card × 8 PEs); `/dev/rngd/npu<N>pe<a>-<b>` nodes allow fusing 2 or
  4 PEs.

> **Device numbering is not stable.** On 2026-08-27 npu2 (PCI `44:00.0`) was gone
> from the PCI bus and `furiosa-smi` entirely, and torch renumbers densely over the
> cards that remain — so `rngd:16` was npu3 and `rngd:24` did not exist.
> `card_of()` in `experiments/scripts/rngd_device_facts.py` now resolves through
> live sysfs for exactly this reason. Never compute a card from `index // 8`.

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

Availability has changed twice: 2026-08-25 only npu3 was allocatable; 2026-08-27
every PE on all three surviving cards was free and `furiosa-smi ps` was empty.
Full holder table: `docs/hardware_roadmap.md` "Who holds the NPUs".

## Vendor runtimes — split and mutually incompatible

System `dist-packages` holds `vllm 0.13.0+cpu`, `vllm_rbln 0.10.2.post1`,
`rebel-compiler` 0.10.2, `tvm 0.20.dev0`, `transformers 4.57.6`. User `~/.local`
holds `furiosa-llm` / `furiosa-torch` 2026.2.0, `torch 2.10.0+cu128`,
`transformers 5.1.0`.

The user-site `transformers` breaks system vLLM; the auto-loaded `rbln` vLLM plugin
breaks on the `rebel-compiler`/`tvm` mismatch. **One venv per vendor.** Never
install vLLM into `.venv` — that is the planner/analytical-sim environment and it
needs no device. Details and routes forward: `docs/hardware_roadmap.md`
"First access".

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

**ATOM has never been profiled.** `rebel-compiler` resolves to 0.11.0 while
`vllm_rbln`/`optimum-rbln` expect 0.10.2. `profiles/accelerators/rbln_atom.yaml`
keeps `sim_hardware: null` and empty `supported_models` so it fails loud and stays
out of candidate generation.

**Do not run CUDA `bench/` here** — there is no NVIDIA driver. Committed A40 and
A5000 artifacts remain valid measurements of *those* machines; do not re-run,
extend, or relabel them here.

## Open work that needs this node

- `docs/npu_concurrency_envelope_work_order.md` — c64/c128 concurrency run. c32 is
  the highest ever measured, and the card-fixture sweep assumes ~76 concurrent
  sequences per card.
