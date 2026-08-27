# Work order — commit the NPU leg's multi-stream bandwidth

*Written 2026-08-27 on the A40 server, to be executed on the **NPU server**.
Branch: `feat/rngd-parallel-bandwidth`. Everything you need is in this file.*

---

## 1. The gap, stated precisely

Four numbers are load-bearing across this repo:

> host → RNGD PE aggregate: **5.06 / 10.39 / 19.10 / 35.47 GB/s** at 1 / 2 / 4 / 8
> streams — 88 % of ideal scaling at 8.

They appear in `docs/HANDOVER_A40.md` §1, `docs/PROJECT_REPORT.md` §4.8.2, and the
link comments of **both** P/D fixtures (`experiments/configs/clusters/pd-rngd-gpu.yaml`,
`pd-rngd-gpu-card.yaml`).

**Only the single-stream one is committed.** `outputs/rngd_profile/host_bandwidth.json`
holds one run (`rngd:16`, npu2, peak H2D 5.03 GB/s) produced by
`rngd_device_facts.py --host-bandwidth`. The 2 / 4 / 8-stream figures exist **only as
prose**. No committed code produces them, and no committed artifact records them.

This violates absolute rule 3's intent as much as an unlabelled placeholder would:
the numbers are almost certainly real measurements someone took, but nothing in the
repo can reproduce or audit them, and they are labelled as measured.

### Why it now matters more than it did

The GPU leg of the cross-vendor KV path was measured on the A40 server on
2026-08-27 (branch `feat/gpu-host-bandwidth`), which closed the last item in
`docs/HANDOVER_A40.md` §1. Both P/D fixtures' `fabric-*` links moved from
`source: placeholder` to `source: measured`, carrying values composed as a
serialised handoff `1/(1/gpu + 1/npu)`:

| fixture | link | composition | GB/s |
| --- | --- | --- | ---: |
| `pd-rngd-gpu-card.yaml` (tp1) | rngd ↔ a40 | 1/(1/26.03 + 1/**35.47**) | 15.0 |
| | rngd ↔ rngd | 1/(1/**35.47** + 1/**35.47**) | 17.7 |
| `pd-rngd-gpu.yaml` (tp4) | rngd ↔ a40 | 1/(1/62.42 + 1/**19.10**) | 14.6 |
| | rngd ↔ rngd | 1/(1/**19.10** + 1/**19.10**) | 9.6 |

**Every bolded term is one of the uncommitted numbers.** Four of the six fabric
links in the two fixtures now say `measured` while resting partly on a
measurement that exists only in prose. Closing this gap is what makes that label
true.

For reference, the GPU leg as measured (pinned host memory, D2H, 256 MB):
26.03 GB/s at 1 stream, 62.42 at 4, 79.98 at 8 — only **38 % of ideal at 8
streams**, because the host path saturates around 80 GB/s. The NPU leg's claimed
88 % is the thing that makes the composed numbers as good as they are, so it is
worth being sure of.

---

## 2. What to run

`experiments/scripts/rngd_device_facts.py` has a new `--parallel-bandwidth` mode
on this branch. Pre-flight first — device availability on that box changes, and
another tenant's pods hold most of it:

```bash
# 2.1 Which PEs are actually free. Non-empty alloc_status == claimed by someone.
for n in 0 1 2 3; do for m in 0 1 2 3 4 5 6 7; do
  f="/sys/class/rngd_mgmt/rngd!npu${n}pe${m}/alloc_status"
  [ -e "$f" ] && echo "npu${n}pe${m}: $(cat "$f" 2>/dev/null || echo FREE)"
done; done
```

You need **all 8 PEs of one card** free, because the headline is the 8-stream
figure. Last known state: only npu3 (`rngd:24..31`) was allocatable; npu0/1/2
returned `EBUSY` on every PE. Re-check — the driver re-enumerated once already.

```bash
# 2.2 The measurement. Runs in the vendor stack, NOT .venv (see §4).
cd <repo> && git fetch && git checkout feat/rngd-parallel-bandwidth

PYTHONPATH=$PWD python3 experiments/scripts/rngd_device_facts.py \
    --device rngd:24 \
    --parallel-bandwidth \
    --streams 1,2,4,8 \
    --parallel-size-mb 256 \
    --parallel-duration-s 5 \
    --parallel-trials 3 \
    --out outputs/rngd_profile/parallel_bandwidth.json
```

Adjust `--device` to the first PE of whichever card is free (`rngd:0`, `:8`, `:16`,
`:24`); the script derives the card and the PE range from it.

Expect roughly 4 stream counts × 3 trials × 2 directions × 5 s ≈ **2 minutes**
plus process startup.

---

## 3. What the mode does, and the three design decisions behind it

Read these before judging the output — two of them are departures you would
otherwise flag as bugs.

**One process per PE, not threads.** Nothing in this repo establishes that
`furiosa.torch` lets one interpreter hold several PE contexts at once, and
`start_load()` — the existing multi-PE driver in the same file — already uses
subprocess-per-PE. The new code reuses that pattern and its `stop_load()`
cleanup. The parent synchronises workers by sending an absolute `time.time()`
instant on stdin after every worker reports `READY`; `time.time()` is comparable
across processes and `perf_counter` is not, which is why the single-stream path
uses `perf_counter` and this one cannot.

**Sustained throughput over a fixed window, not best-of-N transfers.** The GPU
leg used a per-transfer barrier and best-of-N; coordinating that across processes
is not worth the failure modes. A KV handoff is a sustained bulk transfer, so
sustained is the more honest statistic anyway. The **aggregate definition is
identical** to the GPU script — total bytes across all workers over the wall time
of the concurrent region, first start to last finish — which is what lets the two
legs compose.

**Repeat over independent trials, and report the median plus the spread.** On the
GPU side a single trial was *not reproducible*: host buffers are not NUMA-bound,
their placement is fixed at allocation, and two runs disagreed by 38 % on the
4-stream figure with the same-node/cross-node ordering reversing between them.
Workers are respawned per trial so each trial re-allocates.

### Read H2D, not D2H

`h2d` (host → PE) is the direction the KV path needs on the decode side, and its
host buffer is allocated once and reused, so it measures the link.

`d2h` is **indicative only**. Its loop calls `.cpu()`, which allocates a fresh
pageable destination every iteration; above ~16 MB PyTorch's CPU allocator stops
reusing a cached block and mmaps instead, so every copy pays page faults and the
number becomes allocator-bound rather than link-bound. This was measured and
documented on the GPU side, where pageable D2H peaked at 16 MB and fell 5× at
64 MB and above. The committed single-stream RNGD run has the same property, so
d2h is kept for continuity — do not quote it as a link figure.

---

## 4. Environment — do not cross the interpreters

- **System `python3`** runs this: it has the vendor runtimes. `.venv` is the
  planner/analytical-sim environment and has no `furiosa.torch`. Never install
  vLLM into `.venv`.
- The `furiosa.torch` import in this file is pinned **after** `torch` with an
  `# isort: off` guard. If `ruff check --fix` ever reorders it, every run breaks
  with a circular-import error. Do not let that happen (commit `46f0c70` is the
  fix that established this).
- Do not drain devices or kill the `rngd_pd.serving.cluster` pods. `furiosa-smi ps`
  and `rbln-stat` under-report because the holders are pods; the sysfs check in
  §2.1 is the reliable one. Full holder table: `docs/hardware_roadmap.md`
  "Who holds the NPUs".

---

## 5. Acceptance — what "done" looks like

1. `outputs/rngd_profile/parallel_bandwidth.json` exists and is **committed**
   (`git add -f` if a gitignore rule catches it; the surrounding
   `outputs/rngd_profile/` artifacts are tracked, so it should not).
2. **The `streams: 1` row cross-checks against the committed single-stream table.**
   This is the method's self-validation: `outputs/rngd_profile/host_bandwidth.json`
   says peak H2D 5.03 GB/s, and the docs quote 5.06. If the new 1-stream figure
   lands near that, the sustained method agrees with the best-of-N method and the
   multi-stream rows can be trusted. **If it does not, stop and report it** — that
   is a finding about the method, not a number to paper over.
3. The 2 / 4 / 8-stream aggregates and their `h2d_scaling_vs_ideal` are recorded.
4. **Reconcile against the prose.** Compare the new figures with 10.39 / 19.10 /
   35.47 and 88 %:
   - *If they agree* — replace the prose with a citation to the committed JSON in
     all four places (`docs/HANDOVER_A40.md` §1, `docs/PROJECT_REPORT.md` §4.8.2,
     both fixture link comments) and say the numbers are now reproducible.
   - *If they disagree* — **the new measurement wins, and the fixtures must be
     recomputed.** Redo the compositions in §1's table with the new NPU leg and
     update `bandwidth_gbps` on the four affected `fabric-*` links in both
     fixtures. Then re-run both SLO sweeps
     (`experiments/scripts/pd_slo_sweep.py`, invocation in `docs/HANDOVER_A40.md`
     §1) and diff against `experiments/results/pd_slo_sweep.md`. Record the
     retraction in `docs/deviations.md` the way `8ca075d` and `d3cd9b6` did — this
     project retracts numbers in public, it does not quietly overwrite them.
5. Write `experiments/results/rngd_parallel_bandwidth.md` — table, method, the
   1-stream cross-check, and the reconciliation verdict.
6. Gates green before the PR: `pytest` (284 passing), `ruff check .`,
   `mypy planner/`.

---

## 6. Status of the code you are about to run

**The `--parallel-bandwidth` implementation has never run on RNGD hardware.** It
was written on the A40 server, which has no NPU. What *was* verified there:

- `ruff check` and `py_compile` clean.
- The full parent/worker protocol — spawn, `READY` handshake, `GO` dispatch for
  both directions, `RESULT` parsing, aggregate arithmetic, `scaling_vs_ideal`,
  trial repetition, and `stop_load` cleanup — exercised end to end against a CPU
  stub (`device` forced to `"cpu"`, furiosa import stubbed). The plumbing works;
  only the physics is untested.

So the likely failure modes are device-specific, and there are three worth
knowing in advance:

- **Concurrent PE access from separate processes may be refused.** If workers
  fail at `READY`, that is the finding — it would mean the 8-stream number cannot
  be produced this way at all, and would cast doubt on however the prose figures
  were originally obtained. Report it rather than working around it.
- **`.to(device)` may need a compile step** the single-stream path does not hit.
  The `READY` handshake exists to keep compilation out of the timed window; if a
  worker stalls there, raise the startup budget rather than trimming the warm-up.
- **A 256 MB buffer per PE × 8 PEs** is 2 GB of host allocation plus device-side
  residency. The card has 47.5 GB and a PE addresses ~6.25 GB, so 256 MB is
  comfortable — but if allocation fails, drop `--parallel-size-mb` to 64 and say
  so in the results file, because the GPU leg was composed at 256 MB and a
  different size is not like-for-like.

---

## 7. Context you may want

| Doc | What it gives you |
| --- | --- |
| `docs/HANDOVER_NPU.md` | NPU-server environment, the two-interpreter split, the RNGD work path |
| `docs/HANDOVER_A40.md` | The cross-vendor KV path, both fixtures, what each is for |
| `docs/hardware_roadmap.md` "Who holds the NPUs" | Which pods hold which devices |
| `experiments/results/gpu_host_bandwidth.md` | The GPU leg: method this mirrors, and the two traps it documents. **Lands with branch `feat/gpu-host-bandwidth`** — not on `main` yet |
| `experiments/results/rngd_sim_vs_real_summary.md` | RNGD sim-vs-real, for calibration context |
