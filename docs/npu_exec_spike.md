# Spike — how much of the simulator's RNGD error is the execution model?

**Work order:** `WORK_ORDER_npu_exec_model_spike.md` (2026-09-15). **D-block:** D90–D99.
**Experiment tag:** `E-N*`. This document is the spike's running record; STEP D turns it into the
decomposition table and the conclusion.

**The question.** The RNGD card's accuracy domain shows a TPOT error that changes sign with load —
+11 % at served concurrency 2, ~0 at 16, −18 % at 76 (`profiles/calibration/rngd_card_edf.yaml`),
with TTFT at −32.6 % (D17, `rngd_sim_vs_real_summary.md`). How much of that is the structural
difference between what LLMServingSim simulates (a vLLM-style continuous-batching scheduler) and
what Furiosa-LLM is (a static, ahead-of-time-compiled bucket executor)? "Most" means a follow-up
work order for an opt-in execution model; "some" means formalising only the mechanisms that carry
weight and leaving the rest in the accuracy domain.

This spike **decomposes**. Finishing an execution model is a later work order's job.

---

## §0 — Setup (STEP 0)

### Baseline

| item | value |
| --- | --- |
| branch point | `main` at `f6ee919` ("update WORK_ORDER_p2_regular_spec_evidence.md: V0 rewritten on D70") |
| work branch | `spike/npu-exec-0-setup` |
| date | 2026-09-16 |
| node | **NPU** (`scripts/whichnode.sh`: 3 RNGD cards, 4 ATOM devices, no NVIDIA reachable, 96 cores / 1511 GiB) |

`whichnode.sh` on this node reports: planner + analytical simulation YES; real vLLM bench NO (no
NVIDIA driver); CUDA layerwise profiler NO; RNGD profiling YES via `furiosa.torch` under the system
python3, not `.venv`; ATOM present but the vendor install is broken (`docs/nodes/npu.md`).

STEP C of this work order is therefore runnable on this node. STEP 0 and STEP A need no device.

### Gates

At the branch point (`f6ee919`, before anything in this table existed):

```
pytest   904 passed, 1 skipped in 185.00s
ruff     All checks passed!
mypy     Success: no issues found in 47 source files
```

With STEP 0 applied:

```
pytest   912 passed, 1 skipped in 184.51s
ruff     All checks passed!
mypy     Success: no issues found in 47 source files
```

The eight new tests are the seven in `tests/test_artifact_buckets.py` and the `D90` case added to
`test_claude_md_still_documents_the_blocks`. Nothing in `serving/` or `planner/` changed, so no
golden output moved.

### What STEP 0 committed

| path | what |
| --- | --- |
| `experiments/scripts/furiosa_artifact_buckets.py` | reads `artifact.json`, classifies buckets, `--json` / `--yaml` |
| `experiments/scripts/serve_log_hit_rates.py` | tabulates `Wire pipeline hit rate` from committed serve logs |
| `experiments/results/rngd_artifact_buckets.md` | the grid of all three artifacts, and what follows from it |
| `experiments/results/rngd_pipeline_hit_rate.md` | the composed-vs-kernelwise measurement, H-a/H-b left open |
| `profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml` | the machine-readable grid for `d6ae6a43` |
| `docs/deviations.md` D90 | the kernelwise-path finding, and the D17 annotation it resolves |
| `tests/test_artifact_buckets.py` | the grid's invariants, including D90's mixed-step claim |
| `tests/test_deviations_numbering.py` | block range now read from CLAUDE.md instead of hardcoded |

Both scripts were written during the 2026-09-15 investigation and were living in a session
scratchpad, not in the repo; STEP 0 committed them and re-ran both from scratch on this checkout.
Every number in §1.1 and §1.2 of the work order reproduced exactly (see the two results files).

`CLAUDE.md` already carried the `D90–D99` block row and the `E-N*` tag row from 2026-09-15; both
were confirmed present and **not** re-added.

**A note on absolute rule 1.** `artifact_buckets.yaml` lands under `profiler/perf/`, which is an
upstream *directory* (the pinned upstream ships an `RTXPRO6000` bundle there). The `RNGD-CARD`
subtree under it is entirely fork-added (`4aac7a5`), and the new file is data beside the fork's own
bundle, not a change to upstream code. Work order A7 requires this location: the grid is a property
of the served artifact, not of the card, so it cannot live in `profiles/accelerators/*.yaml`.

### One stale guard, fixed

`tests/test_deviations_numbering.py` rejected D90 with *"claim a block in CLAUDE.md
in the same commit"* — the one instruction that had already been followed on
2026-09-15. Its allowed range was the literal `BLOCK_RANGE = (37, 89)`, written
when D80–D89 was the last row, so claiming a block in CLAUDE.md could never be
enough on its own. The range is now parsed from CLAUDE.md's table (six blocks,
D40–D99, plus the block-free D37–D39), which is where the rule lives; a stray
`D100` still fails. The parametrised presence check gained a `D90` case so that
deleting a row is a failure rather than a silently smaller range.

### Two corrections to the work order's §1.1

Both were found by re-running the extraction, and both change rules that STEP A and STEP B are
written in terms of. Neither is settled by measurement yet, so both are carried forward as open
items rather than adopted.

**1. The kernelwise attention grid is not a uniform 1024 tokens.** §1.1 derives a 1024-token bucket
width from `131072 / 128`. The 128 is not a division result: pipeline 0's menu holds exactly
`8 prefill + 74 extend + 46 decode = 128` buckets and is the **union** of the compiled buckets
(verified as a set; the 46 composed pipelines are a strict subset). Its spacing is 128 tokens up to
1024 and then powers of two, and the decode edges are `{1024, 2048, …, 131072}`.

*Consequence.* The "1024-unit KV bucket" of R-attn (STEP A.3) and P3 (STEP B.1) is not the grid the
artifact actually has. Grouping on the real edges puts a ~29-sequence sharegpt batch into about 2–4
groups, which is what D17 measured (1.95 → 3.08 attention executions per layer); a uniform 1024-token
grid would predict many more. **STEP A.1 counts distinct groups both ways** and compares each with
D17's 1.95 / 2.40 / 2.87 / 3.03 / 3.08. Nothing here measures the runtime's grouping — only what it
was compiled for.

**2. The batch-padding ladder is not powers of two throughout.** The kernelwise `tokenwise_buckets`
ladder is 1, 2, 4, 8, 16, 32, 64, 128, 256, **384**, 512, 1024. "Round the batch up to the next
power of two" (R-pad in A.3, P2 in B.1) is correct at or below 256 only; above it the target is the
next entry of this ladder.

### Open items carried out of STEP 0

| item | where it is decided |
| --- | --- |
| H-a vs H-b — why the composed hit rate collapses | STEP C.1 |
| KV-group unit: real decode edges vs uniform 1024 | STEP A.1 census, both ways, against D17 |
| batch padding above 256 (the 384 rung) | STEP A.1 census / A.3 R-pad |
| KV usage at 100 % even at c1 — bucket-granular reservation or gauge definition | STEP B.1 P5 (an open investigation item there) |
| does the `INFO` batch log carry per-request ids and token counts | STEP A.1 (fallback: `--no-cleanup-inputs` trace `input_size`) |
| does `lowload_sim_error.py` accept scheduler knobs | STEP A.2 (fallback: a cluster-json copy) |
| is there attention-stage shape in the raw EDF CSV | STEP A.3 R-attn (fallback: D17 median, marked provisional, replaced by C.3) |
