# D23 fix — baseline before the sanctioned edits

*`WORK_ORDER_d23_fix_revalidation.md` STEP 0. Recorded 2026-09-04 on `main` =
`60c6d22` (PR #56 merged). Simulation and code only — no hardware measurement
(absolute rule A1).*

## Node

`scripts/whichnode.sh`, run first as CLAUDE.md requires:

```
detected node : npu   (read docs/nodes/npu.md)
hostname      : etri-001
cores / RAM   : 96 / 1511 GiB
NVIDIA        : none reachable
RNGD cards    : 3
ATOM devices  : 4
```

Everything this work order needs — planner plus the analytical simulator — runs
here without a device. Nothing in it claims a hardware measurement.

**One committed statement is now wrong.** `CLAUDE.md` says "Every node reports `s8`
on kernel 5.4.0-216-generic" and warns not to key off the hostname. The hostname is
now `etri-001`. The warning still stands and the detector is still the right tool —
but the specific claim about `s8` is stale, and a session that recognised `s8` as
"the usual hostname" would now be misled in the opposite direction. Noted for the
STEP 1 `CLAUDE.md` edit, which touches that file anyway.

## Gates, before any edit

| gate | result |
| --- | --- |
| `pytest -q` | **460 passed** in 134.59 s |
| `ruff check .` | All checks passed |
| `mypy planner/` | Success: no issues found in 34 source files |

## `serving/` is at its sanctioned state

`git log --oneline -3 -- serving/` → `8f17612 M5(partial): Phase 5 sim-level P/D
KV-transfer model (deviations D15)`. D15 is the only sanctioned edit so far; D25-a
and D25-b will be the second and third.

## The two regression anchors

`experiments/scripts/d23fix_anchors.sh <label>` runs both and writes
`outputs/d23fix/anchor/<label>/SHA256SUMS`. It is deliberately **sequential** —
these are the reference, so they must not race each other.

**R1 — upstream anchor.** The three `bench/examples/` runs that
`docs/phase0_formats.md` §2.1 guarantees reproduce exactly at the pin.
`bench/examples/run.sh` hardcodes its output into `bench/examples/<model>/outputs/`,
which CLAUDE.md forbids overwriting, so the script reconstructs the same command
with `--output` redirected. The committed references to match:

| model | committed `sim.csv` sha256 | reproduced |
| --- | --- | --- |
| Llama-3.1-8B | `dd0eca3fda903e54…` | yes, before and after the edits |
| Qwen3-30B-A3B-Instruct-2507 | `7d0ff3cee6a973ba…` | **only under D26** — see the retraction below |
| Qwen3-32B | `a0563bc75d0b1367…` | yes, before and after the edits |

**R2 — P/D anchor.** `outputs/.hp-pd-slo/work/pd_cuda-a40…tp4_P___cuda-a40…tp4_D_-s256-t8192`,
the candidate `docs/d23_spike.md` timed at 343 s solo. Committed `sim1.csv` sha256
`fff63c22d69dd8c1…`. Flags read back from its recorded banner and cross-checked
against `planner/predictor/llmservingsim.py:393-415`.

## Pre-flight for STEP 1 — every claim in §1.2 verified against the code

The edit is `cwd=run_paths.inputs_root` on the `Popen` at `serving/__main__.py:548`.
That is safe only if the child's cwd is used for nothing else, so each argument was
checked:

| argument | source | absolute? |
| --- | --- | --- |
| `binary` | `os.path.join(astra_sim, "build/…/AnalyticalAstra")` | yes |
| `--workload-configuration` | `input_path()`, which `abspath`s `inputs_root` (`run_paths.py:49-50`) | yes |
| `--network/system/memory-configuration` | `build_run_paths`, `os.path.abspath` (`run_paths.py:36-46`) | yes |
| `--logical-topology-configuration` (ns3 only) | `astra_sim + "/inputs/…"` | yes |

`astra_sim = os.path.join(cwd, "astra-sim")` is computed at `__main__.py:198`
**before** the `os.chdir` on the next line, so it is absolute. **The child's cwd is
therefore used for exactly one thing: ASTRA-Sim's `tmp__mem/`.**

Two further checks the work order did not ask for:

- **The mount point exists at `Popen` time.** `_prepare_input_config_paths` creates
  `inputs_root/{network,system,memory}` before the child starts, so `inputs_root`
  is already a directory.
- **Nothing leaks.** `_cleanup_inputs_root` (`__main__.py:86-96`) removes the tree
  only when `inputs_root` is under `astra-sim/inputs/runs/`, and warns-and-skips
  otherwise — so a custom `--inputs-root` would accumulate `tmp__mem/`. The only
  caller that passes one is `experiments/scripts/pd_sim_network_sweep.py:128`, and
  it uses a `tempfile.TemporaryDirectory`, which deletes the tree itself. No leak
  in either path.

## Retracted: "the committed MoE baseline is stale"

**This section originally reported a finding that was wrong, and the correction is
the most useful thing in this document.**

The first R1 run of `Qwen3-30B-A3B-Instruct-2507` came out as `0b548376…` against
the committed `7d0ff3ce…`, with all 300 rows differing. Two more runs agreed with
the first, so the run looked deterministic and the committed file looked stale. A
plausible cause was even available and was written down: upstream `c4edd0a`
("fix non-DP multi-instance collective scoping in ASTRA-Sim traces", 2026-06-09)
landed a month *after* the baseline refresh `3723a94` (2026-05-11), and the MoE
example is the only one of the three with two non-DP instances, which is exactly
what that commit touched.

**All of that was wrong.** Under D26 the same example reproduces `7d0ff3ce…` —
the committed value — four times out of four. The three agreeing runs agreed
because they shared a *common cause*: none of them had `.venv/bin` first on
`PATH`, so all three converted their workload with the wrong `chakra`. "The
reruns agree with each other, so the committed file is stale" does not follow
when the reruns share an environment defect, and that is the reasoning error to
carry away from this.

So:

- `docs/phase0_formats.md` §2.1's guarantee holds. All three examples reproduce.
- CLAUDE.md's "`bench/examples/` is the safe regression anchor" holds.
- Upstream `c4edd0a` is exonerated; nothing here implicates it.
- The MoE example is in fact the *most* useful of the three anchors, because it is
  the only single-node example that goes through the multi-instance path and so is
  sensitive to exactly what D26 fixes. Its hash flips with the interpreter.

## One upstream detail confirmed

`WORK_ORDER_d23_fix_revalidation.md` §1.1 claims the cleanup in
`congestion_unaware/main.cc` has a typo. It does:

```
main.cc:28    const char* dir = "tmp__mem";     // two underscores
main.cc:148   ::rmdir("tmp_mem");               // one
```

so the directory is never removed even on the success path. Not ours to fix
(`astra-sim/` is off-limits); it is in
`docs/upstream_issues/astra-sim-tmp-mem-race.md`. After D25-a it stops mattering:
the directory lands inside the run's own tree and goes with it.

## STEP 3.1 input is already on disk

`outputs/.hp-slo-margin18-tight-pd-rngd-gpu/work` holds **253** candidate
directories with their `cluster.json`, so the memory-json census needs no
recompilation.

**And the work order's one open question there is already answered.** §3.1 and §5
say to check whether `build_cluster_config` imports in `.venv` without torch, and
to fall back to duplicating the JSON rules with lowered confidence if not. It
imports and runs: `WORK_ORDER_spikes.md` STEP B called it directly, repeatedly, to
produce `network.yml` and `memory_expansion.json` (`docs/d14_spike.md`). The census
can use the real builder, and that risk row does not fire.
