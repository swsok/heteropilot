# HeteroPilot — current state and what to do next

> **This is the live handover.** Last updated **2026-09-21**, at the end of
> `WORK_ORDER_domain_scoping.md` **S7 — the work order is now complete except
> S6**, which is A40 work. Eleven PRs landed that day: #116–#126.
>
> **The next session is on the A40 node. Read §2.12 first** — it is the A40
> list and it is short. §2.10 is where the work order stands; §2.10.1 is the
> RNGD list and is now almost entirely closed.
>
> **One thing on `main` is not finished and is not S7's: S6** (§2.12). The
> `test_livelock_watch.py` flake that stood beside it is **fixed** (§2.11) —
> the read loop reopened the FIFO by name, so a command that exited promptly
> left its `open(2)` with no writer to wait for.
>
> The body below was rewritten 2026-09-10 at the end of
> `WORK_ORDER_rps_aware.md` rev 2 (STEP 0–6, PRs #60–#72). **The whole stack is on
> `main` as of `3aadc2b`** and every `feat/rps-step*` branch, plus
> `spike/d14-asym-tp`, has been deleted — `origin` holds `main` and nothing else.
>
> That took a second merge. PRs #64–#71 each merged into their *parent feature
> branch* rather than into `main`, and the parent had already reached `main`
> sixteen seconds earlier, so thirteen commits — STEP 2 through STEP 6 — read as
> MERGED on GitHub while being absent from `main`. PR #72 landed them. If a future
> stack is merged bottom-up again, check
> `git rev-list --count origin/main..<tip>` before believing the PR list.
>
> **It happened again on 2026-09-18, in the same shape, eight seconds apart.**
> #109 (S3) merged into `main` at 04:47:14Z; #110 (S4) merged into
> `feat/ds-s3-link-effective-bw` at 04:47:22Z and #111 (S5) into
> `feat/ds-s4-ea1-condition-refuse` at 04:47:32Z — both parents already on
> `main`, so both children went into branches nothing merges from again. All
> three read MERGED while `main` had no `D113` and no
> `tests/test_domain_conditions_stated.py`. **PR #112 landed S4, S5 and this
> handover revision together**, which is also why they arrive in one commit
> range rather than three.
>
> Twice now the warning above was in this file and did not prevent it, because
> it is read after the merge rather than before. **The rule that would have:
> merge a stack top-down — retarget the tip PR at `main` first — or merge each
> PR into `main` one at a time, rebasing as you go.** Bottom-up is what fails,
> and `git rev-list --count origin/main..<tip>` is the only check that notices.
>
> It is **node-agnostic**:
> every open item says which machine it needs. Earlier handovers are historical
> and must not be read as status:
> `docs/HANDOVER_2026-08-31.md` (→ NPU, the previous live one),
> `docs/HANDOVER_NPU.md` (→ NPU, 2026-08-25),
> `docs/HANDOVER_A40.md` (→ A40, 2026-08-26).
>
> Authority order is unchanged: `WORK_ORDER_heteropilot.md` → `docs/deviations.md`
> → `CLAUDE.md` → this file.

**Gates at this commit**, on the NPU node in `.venv`:

```
pytest -q     681 passed in 183.17s         # NPU node, 2026-09-11
ruff check .  All checks passed!
mypy          Success: no issues found in 38 source files
```

**Gates on `main` at `3cbc8a0`**, A40 node, after the domain-scoping S1+S2 merges
(§2.10):

```
pytest -q     976 passed, 1 skipped in 107.80s   # A40 node, 2026-09-18
ruff check .  All checks passed!
mypy          Success: no issues found in 47 source files
```

**Gates at S5's tip**, the commit PR #112 lands, **A5000 node**:

```
pytest -q     1014 passed in 127.33s             # A5000 node, 2026-09-18
ruff check .  All checks passed!
mypy          Success: no issues found in 47 source files
```

**Gates for the `livelock_watch` fix (§2.11)**, A40 node, accelerator set
`83e3434d7696`:

```
pytest -q     1076 passed, 2 skipped in 142.00s   # A40 node, 2026-09-21
ruff check .  All checks passed!
mypy          Success: no issues found in 47 source files
```

The A5000 node is the third machine and this is the first gate run recorded on
it. Nothing in S1–S5 needed an accelerator: the whole CPU half of the
domain-scoping work order replays a warm cache.

**The three counts differ by node and that is expected**, so do not read a
smaller number on the RNGD node as a regression. 681 is the NPU node at
`main`'s state on 2026-09-11, before the ~330 tests the last two sprints added;
a fresh run there should now land near the A5000's 1014, minus whatever skips
on that hardware. Compare like for like: re-run on the node you are on and
compare against the previous run **on that node**, not across this table.

622 of those are the count this handover was written at, on the NPU node. One of
them **failed on the A40 node** before any new work:
`test_there_is_no_user_site_fallback` asserted `any('.local' in p for p in
sys.path)`, which matches uv's *interpreter* prefix
(`~/.local/share/uv/python/.../lib/python3.10`) — the standard library, not a
user-site directory a stray `pip install --user` could poison. The property it
cared about held on the node where it failed; it passed on the NPU node only
because that interpreter lives elsewhere. It now asserts the two load-bearing
facts instead: the user site directory is not importable, and the chakra this
interpreter resolves lives under its own prefix.

622, up from 440 at the last handover: the rps-aware sprint added ~180, most of
them holding a refusal in place (an envelope that will not be read past its ends,
a calibration domain that returns None rather than 1.0, a surrogate that is not
silently swapped). `mypy` covers `planner`, `profiler/synth`, `profiler/contract.py`.

---

## 0. First command on any machine

```bash
bash scripts/whichnode.sh
```

**This repository moves between an A40 node, an A5000 node and an NPU node, and
every one of them reports hostname `s8`/`etri-001`.** Any committed sentence about
what hardware is present is true on at most one of them. The detector probes
`nvidia-smi -L`, `/sys/class/rngd_mgmt` and `/dev/rbln*`, prints what can actually
be run here, and names the node profile to read (`docs/nodes/{a40,a5000,npu}.md` —
read only the one it names).

Absolute rule 3 covers hardware **presence**, not just hardware numbers: never
claim a result from hardware the detector does not list.

---

## 1. Status

| Phase | Status |
| --- | --- |
| 0 Baseline · 1 Inventory/islands · 2 Offline planner (MVP) | ✅ done |
| 3 Heterogeneous profiles | ✅ done (`CsvProfileImporter`) |
| **Tiered profiles (Tier 0/1)** | ✅ **done** — D4 closed without external measurements; `docs/tier0_calibration.md` |
| 4 Real deploy + calibration | ✅ CUDA. NPU launcher still a stub |
| 5 Topology-aware P/D | ✅ core, and **asymmetric TP per phase now representable** (D28). Network-aware routing deferred. **D23 resolved** — it was D26, a PATH-resolved Chakra interpreter |
| **RPS-aware selection** | ✅ **done 2026-09-09** — envelope, accuracy domain, operating-point margins, `plan --rps`. §2.1 |
| **Domain scoping (`WORK_ORDER_domain_scoping.md`)** | ✅ **S1–S5, S7 done**; only **S6** left (A40, §2.12). D110–D115 |
| 6 Online replanning | ⛔ not started — **requires explicit user approval** |

**Accuracy domains on `main` as of 2026-09-21** — **five** files, and which one
the planner reaches matters:

| file | hardware | protocol | basis | on the default glob path? |
| --- | --- | --- | --- | --- |
| `a40.accuracy.yaml` | A40 | open loop | not stated | yes |
| `openloop/a40.accuracy.openloop.yaml` | A40 | open loop | not stated | no — D102 |
| `rngd_card_edf.yaml` | RNGD-CARD | **closed** loop | not stated | yes |
| `openloop/rngd_card.accuracy.openloop.yaml` | RNGD-CARD | **open** loop | **`tpot_p99`** | **no** — load explicitly |
| `rngd_perpe.yaml` | RNGD (per-PE) | closed loop | not stated | yes |

(`rngd_perpe.yaml` is filed under hardware `RNGD`, not `RNGD-CARD`, so it never
competes with the two above — hardware is matched before anything else.)

`load_accuracy_domains` globs **non-recursively**, so `RNGD-CARD` still resolves
to the closed-loop file by default and the open-loop one is opt-in
(`--accuracy-domain <path>`). Since **D115** a domain states the percentile it
was fitted on; `""` means **not stated**, not p50, and `AccuracyDomainMargin`
warns only when a file explicitly declares `tpot_p50`. Migrating the three older
files onto p99 is a separate step and is **not** done.

**ScenarioLab moved out** on 2026-09-03 to `swsok/heteropilot-scenariolab`
(private), which pins this repo as a submodule at `e79ac4ab`. It imports from
`planner/` and never the reverse, so nothing here depends on it. `profiles/networks/`
and `experiments/configs/lab/` went with it — D24 records why that does not
contradict the work order's layout.

**Accelerator profiles** (`profiles/accelerators/`) **and what they may claim:**

| profile | `sim_hardware` | `source` | usable in candidate generation |
| --- | --- | --- | --- |
| `a40.yaml` | A40 | measured | yes |
| `a5000.yaml` | A5000 | measured | yes |
| `furiosa_rngd.yaml` (PE-as-device) | RNGD | measured | yes |
| `furiosa_rngd_card.yaml` (card-as-device) | RNGD-CARD | measured | yes |
| `rtxpro6000.yaml` | RTXPRO6000 | vendor_spec | yes, labelled |
| `ascend_target.yaml` | **ASCEND_TARGET-t0** | vendor_spec | **yes now** — Tier 0 synthetic bundle, plans carry `profile_tier: analytical` |
| `rbln_atom.yaml` | **null** | placeholder | **no — fails loud** (D20) |

`ascend_target` is the change: it was `placeholder` / `sim_hardware: null` and is
now backed by a datasheet-derived bundle that `scripts/gen-tier0-bundles.sh`
regenerates. Synthetic bundles are **gitignored on purpose** (`profiler/perf/*-t0/`,
`*-t1/`) so measured and synthetic data never mix in the tree. Every plan that
touches one carries the weakest tier in `PlannerOutput.profile_tier` plus a
mandatory caveat — D21.

Calibrations (`profiles/calibration/`): `a40.yaml`, `rngd.yaml`, `rngd_card_edf.yaml`,
and since this sprint `a40.accuracy.yaml` (opt-in, D29) and `slab3d_latency.yaml`
(a topology correction, not a hardware calibration — `load_accuracy_domains` skips
it deliberately). Tier 1 efficiency fits: `a40.efficiency.yaml`,
`rtxpro6000.efficiency.yaml`. All bucket-scoped — **do not extrapolate outside the
bucket named in the file.**

Since 2026-09-17 every accuracy domain also **states the configuration it was
measured under** — hardware, `parallelism {tp,pp,dp}`, `placement {islands,
device_binding}`, and since S4 `arrival_process` and `variant` — and
`profiles/calibration/index.yaml` lists them all. A candidate that differs on one
of those is refused as `calibration_condition_mismatch`: unmeasured **at its own
configuration**, which is a different gap from an operating point past the end of
the load axis, and it asks for a different measurement. D110, D113, §2.10.

**What the four domains state, because this decides what gets refused:**

| domain | tp | islands | binding | arrival | variant |
| --- | ---: | ---: | --- | --- | --- |
| `a40.accuracy.yaml` | 1 | 1 | `unpinned` | `open_loop` | `bf16` |
| `openloop/a40.accuracy.openloop.yaml` | 1 | 1 | `unpinned` | `open_loop` | — |
| `rngd_card_edf.yaml` | 1 | 1 | `unknown` | **`closed_loop`** | — |
| `rngd_perpe.yaml` | **8** | 1 | `unknown` | **`closed_loop`** | `bf16` |

The two `closed_loop` rows are why **every RNGD candidate is now held** —
§2.10.1 item 1. `model` is stated on none of them on purpose: this repo holds
two strings for one set of weights (`meta-llama/Llama-3.1-8B` in the envelope
paths, `NousResearch/Meta-Llama-3.1-8B` in the open-loop A40 domain's own
provenance), so a string comparison would refuse on the mirror name (D113).

**Two new artifact kinds, and the rule attached to each:**

| artifact | what it is | the refusal it carries |
| --- | --- | --- |
| `profiles/envelopes/<HW>/…/tp<N>.yaml` | a measured performance curve | `concurrency_metric: served` and `validity` are mandatory; a file missing either **does not load** |
| `accuracy_domain:` inside a calibration | the predictor's own error vs operating point | outside the measured points the margin **widens with distance and never caps**; hardware without one gets margin 0 and a note |
| `measurements:` on a cluster link (S3) | the effective bandwidth for one kind of traffic | keyed by `(collective, world_size, msg_size_class, binding)`; the datasheet `bandwidth_gbps` is never edited (A3), and `source: measured` without a `method` is refused |

**What can be claimed right now, with the label each claim earns, is one page:
`docs/CLAIMS.md`.** It is the input to the paper outline — Established / Not
established / Retracted — and every figure in it points at a committed artifact.

### What the last cycle established

**The headline finding is negative, and it should be said first: no experiment in
this repository currently shows a heterogeneous configuration winning.** The one
that did — RNGD beating the A40 on energy by 1.67× at loose TTFT — was retracted
by this project's own margin discipline (D22). The tight-TTFT half, where P/D
disaggregation was supposed to buy the sub-second end, is undetermined and blocked
by a simulator livelock (D23). That is the starting point for the next work order,
not a gap to paper over.

- **Tier 0/1 works well enough to plan on hardware we do not own.** Against the
  measured-bundle ranking, tier0+sim gives **Kendall τ 0.90–0.91**. Exact top-1
  agreement is 0/2 — but the *cost* of the top-1 disagreement is **0.4 %** of the
  true objective on `llama31-8b` and **11.3 %** on `llama31-8b-light`, i.e. the
  candidates it confuses are near-equivalent. Ranking is preserved where it
  matters.
- **Attention is where the error lives, and anchoring it is what pays.** Tier 0
  with no anchors is **38.9 % MAPE**; 200 anchors spent entirely on attention take
  it to **29.5 %**. GEMM transfers on a single scalar efficiency (9.5–11.3 %);
  attention does not.
- **D4 is closed** — not by external measurements arriving, but by generating a
  datasheet-derived Ascend bundle and labelling it honestly. Measured data still
  supersedes it whenever it arrives.
- **D22 — the RNGD envelope, retracted and remeasured.** "c32 is the highest
  concurrency ever run on RNGD" was a 24-request pool at **effective concurrency
  21.2**, and the exponent derived from it read a pool-capped ×1.74 interval as a
  doubling. Measured to **eff 107.2 at 1473 output tok/s**, zero failures. At the
  eff 76 the card fixture's winner runs at, the simulator is **1.31× optimistic on
  throughput and 18 % on TPOT**.
- **And that 18 % kills the winner.** Re-run as a feasibility margin, **every RNGD
  configuration is rejected on both fixtures**; the loose-TTFT winner becomes
  `agg[cuda:tp4]` at 2.595 tok/J. The committed winner cleared the 50 ms TPOT SLO
  by 1.59 ms, so **any margin above 3.3 %** rejects it. It was infeasible, not
  merely optimistic.
- **D23 — the tight-TTFT candidates livelock.** Every `pd_*`/`mix_*` candidate the
  tight regime needs fails to terminate: 52,903 progress ticks with prefill pinned
  at 1 running request, decode never fed, memory flat at 9 %. Not D12 (no memory
  growth, prefix caching off) and not a timeout (3600 s fails too). The same
  candidate completed in **280.6 s** in an earlier committed run, so it is a
  regression, and the cause is open.

---

## 2. Next work, in priority order

### 2.12 The A40 node list — **read this one first if you are on the A40**

Two items, and they are unrelated to each other.

**1. S6 — the last step of `WORK_ORDER_domain_scoping.md`.** Optional in the
work order, and it is what would close the one thing S7 left open on the A40
side. Three parts, and PR #104 may already contain some of them — check before
measuring:

* **(i) NUMA-bound P1 (TP=4) open-loop, 3 runs**, against the simulator with
  S3's measured link value. `v3_addendum_s4.md` §3 has the pairing: with the
  link priced from measurement the simulator lands within **−7.01 %** on p99
  TPOT and **−0.73 %** on concurrency. The residual −7.01 % is **not margined by
  anything** — no A40 domain is fitted at tp=4 — which is exactly what rule (e)
  says, and S6(i) is the measurement that would close it.
* **(ii) `nccl-tests all_reduce_perf` on 4 GPUs** over the TP message-size
  range, plus `p2pBandwidthLatencyTest`, **bound and unbound**, loaded into the
  S3 (D112) link-measurement schema.
* **(iii) the same candidate at TP=2 inside an NV4 pair** (GPU0↔1), to check the
  simulator on an NVLink-only path and rule out a TP-arithmetic explanation.

Workload: lengthen it so a steady-state interval exists and exclude warm-up
(V3's transient lesson). `binding: unknown` is what the fixture's nodes say, so
**record the binding** — it is the axis worth 1.93× of throughput here.

**2. ~~The `test_livelock_watch.py` flake~~ — DONE 2026-09-21**, §2.11
immediately below. One line: the read loop ended `done <"$FIFO"` instead of
`done <&3`. No `retry` and no `xfail` were needed — the test was reporting a
real hang, exactly as the constraint assumed.

**What is NOT on the A40 list.** S7 is finished and needs no A40 work. The
RNGD-side items in §2.10.1 are closed except where that section says otherwise.
Do not re-run anything under `experiments/results/s73_npu_6fe246ed1abf/` here:
those are measurements of **accelerator set `6fe246ed1abf`** (D80) and an A40 box
is a different machine — `whichnode.sh` prints `accel serials` so the two cannot
be confused.

### 2.11 `tests/test_livelock_watch.py` was intermittently red on `main` — **FIXED 2026-09-21**

*Found 2026-09-21 on the NPU node (accelerator set `6fe246ed1abf`) while gating
#121; fixed the same day on the A40 node (accelerator set
`83e3434d7696`, 8 x A40). Filed here rather than as a GitHub issue: this token
cannot create issues (`createIssue` → 403). `livelock_watch.sh` and its tests
came from **#115**, and this was that work's to finish, not S7.3's.*

**The cause is one line, and it is #115's own fix left half-applied.** #115 added
`exec 3<"$FIFO"` and converted the drain to `cat <&3` so that nothing would open
the FIFO by name a second time. It did not convert the read loop, which kept
ending `done <"$FIFO"` — a second `open(2)`. Opening a FIFO read-only blocks
until a **writer** appears, and previous writers do not count, so a command that
exits before the shell reaches that line has already taken the only writer with
it. The parent then parks in `pipe_wait` forever, holding fd 3, with no child
left to rendezvous with.

The comment #115 wrote above `exec 3<` said the loop already read from fd 3
("exactly as the old `done <&3` did"). It never did: that commit introduced the
comment and `done <"$FIFO"` together.

**Why a different test failed each run, and why each passed alone.** Only a
command that exits promptly can lose the race, and the watcher's own start-up is
the window. The four failures recorded on the NPU node are exactly the four cases
whose command has no trailing `sleep`:

| run | failed | its command |
| --- | --- | --- |
| 1 | `test_healthy_run_is_never_flagged` | `cat <log>` |
| 2 | `test_command_exit_code_passes_through` | `exit 7` |
| 3 | `test_a_growing_queue_alone_does_not_trigger` | `cat <log>` |
| 4 | `test_the_streak_must_be_consecutive` **and** `test_command_exit_code_passes_through` | `cat <log>`, `exit 7` |

Every case carrying `; sleep 30` or `sleep 60` still has a writer when the loop
opens, and none of them ever failed. It is a race, not inter-test interference —
"passes alone" is load, not state carried between tests.

**Measured, on the A40 node:**

| | pre-fix | post-fix |
| --- | ---: | ---: |
| hangs in 40 runs of `livelock_watch.sh -q -- bash -c 'exit 7'` | **5 (12.5 %)** | — |
| hangs in 200 runs | — | **0** |
| leftover `/tmp/livelock_watch.*` | one per hang | 0 |
| `pytest tests/test_livelock_watch.py` | failed on every run | **9 consecutive runs, 17 passed each** |

A hung watcher's `/proc/<pid>` is the direct evidence and is what identified it:
`wchan: pipe_wait`, fd 3 open on the FIFO, **no fd 0 on it**, and no children —
i.e. blocked inside the loop's own `open()`, not in a read.

**The FIFO-file leak was the same bug, not a second one.** §3 recorded `cat=0`
but the `/tmp/livelock_watch.*` *files* accumulating, and guessed at a second
leak in the drain path. There is none: a watcher hung in `open()` is SIGKILLed
by whatever was waiting on it, so it never reaches its EXIT trap and the
`rm -f "$FIFO"` never fires. From a cleaned `/tmp`, three consecutive runs of the
file leave **0**, as do 200 runs of the stress case. The one file that did appear
during this work came from the deliberately-reverted script used to check the new
tests — i.e. from a hang, which is the mechanism, not a counterexample to it.

**Two regression tests**, and they were checked against the pre-fix script —
both fail on it, the structural one deterministically and the behavioural one
with the same `TimeoutExpired` signature recorded above:

* `test_the_watcher_never_reopens_the_fifo_to_read_it` — the invariant is
  structural, so it is asserted on the source: after fd 3 is open, `$FIFO` is a
  name to be unlinked, never one to be opened.
* `test_a_command_that_exits_immediately_never_hangs` — 60 runs of `exit 7`,
  which fails under the bug with probability > 0.999 and cannot flake once fixed,
  because there is no `open()` left to block in.

No `retry` and no `xfail`: the test was reporting something real, which is what
the S7.3 constraint assumed.

**It never affected S7.3 (#121).** Both files are byte-identical to `main`'s
there and that branch never touched them; S7.3's own suites are green, and its
measurements did not use `livelock_watch.sh` — the open-loop harness bounds its
children with the predictor's own `timeout_s`.

<details>
<summary>Superseded diagnosis, kept because it records what was reasonable to believe</summary>

The first pass read this as inter-test interference and looked for state carried
between tests, on the strength of `test_a_growing_queue_alone_does_not_trigger`
passing alone in 0.07 s and failing inside the file. That observation is real and
the inference from it was wrong: a race whose window is the watcher's start-up
widens under the load of the other tests in the file, so "passes alone" is
exactly what a race looks like too. **Before concluding interference, check
whether the isolated run is also the unloaded one.**

It also cleaned up **20 orphaned `cat /tmp/livelock_watch.*` processes**, aged
2d21h–3d, all `ppid=1` with their FIFO already deleted — genuinely the leak #115
fixed, so genuinely predating it, and each verified to belong to no live watcher
and killed by PID. That cleanup was correct and changed nothing, which was the
first evidence that the remaining fault was not in the drain:

| | before cleanup | after cleanup |
| --- | --- | --- |
| watcher invoked directly, 8× | 1 hang, then 0 on a second sample | 0 hang |
| `pytest tests/test_livelock_watch.py` | 1 failed | **1 failed, then 2 failed** |

> 8 runs cannot resolve a ~12 % rate, and the direct-invocation samples disagree
> with each other, so that row was inconclusive at the time. 40 runs settled it.

The `/tmp/livelock_watch.*` files growing 3 → 5 → 6 across runs were read as "a
second leak in the same drain path". They were the same bug: the hung parent
never reaches its EXIT trap.

</details>

### 2.10 `WORK_ORDER_domain_scoping.md` — **S1–S5 and S7 done. Only S6 remains, and it is A40 work**

It exists because STEP V3 of `WORK_ORDER_p2_regular_spec_evidence.md` (PR #98)
found two defects that the disclosure's own claims rest on. **S1–S5 (CPU) and
S7 (RNGD node, 2026-09-21) are done.** The only step left is **S6, on the A40
node** — see §2.12.

**S7's result in three lines, because it is not what the work order expected:**

* **§6 is still not established.** The one rankable verdict case is *circular*
  — the open-loop domain's robust value returns the measurement it was fitted
  from. Leave-one-out gives ±0.63 ms on held-out interior points, which is
  predictive accuracy and not a verdict count, and the **boundary points cannot
  be leave-one-out validated at all**.
* **What S7 did measure is a counterfactual, and it belongs to §5.2/§8(3):**
  consulted outside its arrival-process condition, the closed-loop domain
  **false-passes a violating candidate by 4.911 ms = 42× the run-to-run
  spread**. D113 asserted that as policy; this is the number.
* **A new open-loop domain is installed** at
  `profiles/calibration/openloop/rngd_card.accuracy.openloop.yaml` — six points,
  sim L 11.730–37.965, `compared_metric: tpot_p99`. It is **not** on the default
  glob path; `load_accuracy_domains` still resolves `RNGD-CARD` to the
  closed-loop file (the D102 pattern). Load it explicitly.

Read `experiments/p2_evidence/results/v3r_verdict_accuracy.md` and
`experiments/results/s73_openloop_refit.md` before quoting any of it.

**S1 — an accuracy domain answers only for the configuration it was measured
under (D110, PR #106).** V3 deployed P1, one A40 island at **tp=4**, whose margin
came from `a40.accuracy.yaml`, a domain fitted at **tp=1**. The operating point
was inside that domain's load axis and the arithmetic was right, so every check
passed: **1.13 %** charged where the measured error was **−44.63 %**, about a 70×
under-correction, while the same domain is right to −1.05 % at the tp it was
fitted at. `AccuracyDomain` now carries `hardware` / `parallelism` / `placement`,
`check_conditions()` compares them per island assignment, and a mismatch is its
own rejection stage carrying `mismatch_fields` and `required_measurement` — the
disclosure's "additional measurement condition" output. Policy key
`--condition-mismatch {refuse,warn}`, default `refuse`. **No `points` value was
touched** (rule A3); the four domain files gained only the conditions their own
provenance records.

**S2 — a grade carries a default range (D111, PR #107).** An input whose own
(kind, grade) sourced no width had no range, so no regret, so no place in the
measurement plan at all. That is what happened to `link_bw:pcie-a40a-02` — the
`vendor_spec` link that explained the whole of V3's −43.4 % TPOT error, for
**0.114 h** of measurement. `grades.yaml` gains `defaults:`, every range built
from one is labelled `range_source: default` end to end, `user_defined` still has
none on purpose, and `MeasurementPlan.inert` is new because with ranges where
there were none an input can be *decidable and worth zero*.

**What S2 measured is not what it predicted, and it is why S3 existed.**
The hypothesis was that the V3 link would rank first. It did not rank at all: it
moved from `undecidable` to **`inert`**. A link bandwidth reached a predicted
metric through exactly one path — the prefill→decode KV transfer — and the E-A1
corpus was built with `enable_pd=False`, so nothing crossed any link.
(`experiments/uncertainty/results/s2_default_ranges.md`.)

**S3 — a link does not have one bandwidth (D112, PR #109).** The same A40 PCIe
bridge measures **25.0** GB/s for a direct copy, **19.29** for a two-rank
all-reduce and **8.8** for the four-rank all-reduce a tp=4 island runs, against
a `vendor_spec` **64.0**. So the LINK_BW item is keyed by the traffic —
`(collective, world_size, msg_size_class, binding)` — and `Link.measurements[]`
files the figures *beside* the datasheet value, never over it (A3).
`world_size` is in the key although the work order named only four parts: 8.8
and 19.29 differ by nothing else and the fixture carrying both would not load.
That is the field V3 asked for in as many words.

S3 also found the reason S2 saw `inert`. An intra-island link's bandwidth is the
`min` that `island_interconnect` reduces into the simulator's own `link_bw`, so
it prices **every TP collective**, and no closed form can reach it. Such an item
now lands in `MeasurementPlan.needs_resimulation` — a fifth bucket — instead of
scoring a confident zero. (`s3_link_effective_bw.md`.)

**S4 — what the condition refusal costs, and V3's P1 ending (D113, PR #110).**
Rule (e) on E-A1: **312 of 324 held, 0 feasible, no recommendation.** Two things
came out of it that were not predicted:

* **The refused axis is wider than parallelism.** Per field: `dp` **228**,
  `islands` 132, `tp` **66**, `arrival_process` 42. A measurement programme
  aimed only at `tp` would close less than a quarter of the holds. The
  disclosure must not call the axis "the parallelism degree".
* **The committed E-A1 script no longer reproduces the committed E-A1 numbers.**
  Under the planner's own `refuse` default, (c) and (d) return 0 feasible and
  50/244/30 cannot be got back. `--condition-mismatch warn` is what asks the
  older question, and (a)–(d) then reproduce exactly. Any experiment written
  before S1 needs it. (`ea1_s4_condition_refuse.md`, `v3_addendum_s4.md`.)

**S5 — documentation (PR #111).** `docs/uncertainty_planner.md` had no mention
of S1 at all; `patent2_evidence_map.md` still claimed `AccuracyDomain` has no
parallelism axis. Both fixed, plus `CLAUDE.md` and the registry docstring. One
finding is worth carrying out of it: **the patent-3 embodiment's first step is
not established.** "The registry names the input" is what S2 predicted and
measured as `inert`, and S3 as `needs_resimulation` — a human investigating V3
named it. The mechanism is fixed; the demonstration has not been done.

**What is left, and both need hardware:**

- **S6 (A40 node, half a day, optional)** — NUMA-pinned P1 re-measurement,
  `nccl-tests all_reduce_perf` on 4 GPUs and a TP=2 run inside one NV4 pair.
  S3's measurements come from `experiments/p2_evidence/link_probe.py` (torch),
  **not** `nccl-tests`, so S6(ii) is still open if the work order's tool is
  required. S6(i) is also what would margin P1's residual −7.01 %: no A40
  domain is fitted at tp=4, which is exactly what rule (e) says.
- **S7 (RNGD node, optional)** — see §2.10.1, which is where it now lives.

### 2.10.1 The RNGD node list — **mostly closed as of 2026-09-21; three items remain**

> **Items 1, 2 and 3 are DONE** (S7.2/S7.3/S7.4, PRs #116–#126). The open-loop
> refit exists and is installed, it was taken **NUMA-bound** on npu0, and V3-R
> ran. What survives from the original five:
>
> * **item 4** — the NPU exec-model spike's leftovers (wide-KV workload, B.3's
>   hold-out, C.2), unchanged;
> * **item 5** — ATOM (D20), still blocked on the vendor, unchanged;
> * **one thing S7 added**: §6 of the disclosure is still not established, and
>   closing it needs a verdict case whose domain point was **not fitted from the
>   measurement that judges it** — a hold-out design, which the domain's
>   boundary points cannot support, or a measurement at an operating point the
>   domain already covers from other data. That is RNGD work and it is not
>   scheduled.
>
> The detail below is kept as written because it records **why** each item
> existed and what was measured; read it for the reasoning, not for the queue.

Five things wanted the RNGD card. They are ordered by what they unblock, and the
first two were new as of the 2026-09-18 sprint.

**1. An OPEN-LOOP refit of the RNGD-CARD accuracy domain.** *This did not exist
as a task before S4.* `rngd_card_edf.yaml` now states
`arrival_process: closed_loop`, sourced from its own provenance — the fit is a
burst against a closed-loop client and `bench_furiosa_endpoint.py` ignores
`arrival_time_ns` (D19). `python -m planner plan` always replays an arrival
trace, so **every RNGD candidate is now held as
`calibration_condition_mismatch`**, and an envelope point above c=76 no longer
answers them. What does is an open-loop refit, and **it is a port, not a
re-run**: the two harnesses split by protocol, not by vendor.

| harness | protocol | backends |
| --- | --- | --- |
| `experiments/scripts/measure_envelope.py` | **closed loop** — a request pool of fixed size | `furiosa` and `cuda` |
| `experiments/scripts/measure_envelope_openloop.py` | **open loop** — an arrival trace replayed at a rate | `cuda` only (it writes `"backend": "cuda"`) |

So the RNGD card has a closed-loop harness and the open-loop one has no furiosa
path. **This entry used to say the job was the mirror image of §2.2's CUDA port.
It is not, and that file cannot host it** — `measure_envelope_openloop.py` runs
`python -m bench run`, i.e. `vllm.v1.engine.async_llm.AsyncLLM` **in process**,
and furiosa-llm is a server with no AsyncLLM equivalent. There is no launch line
to swap. Full reasoning in `docs/deviations.md` **D114**.

**S7.2 is DONE (2026-09-18) and it took the other route**: the open-loop
*protocol* was added to `measure_envelope.py`, which already owned server launch
per backend, NUMA binding of both halves, the power sampler and the settle/idle
windows. `--mode open` swaps one thing, the load generator:

```bash
experiments/scripts/measure_envelope.py --backend furiosa --mode open \
    --rps 0.5,1,2 --num-reqs 300 --artifact "$ART" --dataset <sharegpt.jsonl> \
    --card 0 --port 8020 --out outputs/rngd_openloop_envelope
```

`--mode closed` is the default, so every committed invocation runs unchanged.
**`--numa-bind` came for free** — already implemented there and already
defaulting to `auto` — which is item 2 below satisfied by construction rather
than by discipline, so the card is not measured twice.

**Do not add an open-loop point to the existing closed-loop domain.** Two
protocols on one interpolation axis is the class of error D22 was, and
`a40.accuracy.yaml`'s header is the precedent: it went open-loop *because* its
existing 170.56 point came from an arrival replay, and consistency with the
point being joined beat consistency with the other device's curve. A refit is a
new domain file, not three more points in this one.

**2. Re-take the RNGD envelope and accuracy domain NUMA-BOUND (D94).** Both
were taken unbound. STEP C's cost model re-measured bound agreed to 0.24 %
because it reads device cycles, but the wall-clock half of
`measure_envelope.py` did not get that check, and binding is worth up to 1.93×
on the A40 node. Both harnesses now take `--numa-bind`, **default `auto`**, and
record the affinity mask. **Trap:** `/sys/class/rngd_mgmt/*` are virtual devices
with no PCI parent, so resolve the node through `furiosa-smi info` and the BDF,
and verify with `taskset -cp` rather than trusting the flag.

**Items 1 and 2 were the same trip and S7.2 made them one.** The furiosa
open-loop path lives in the harness that already had `--numa-bind` at default
`auto`, so it cannot produce an unbound curve by accident and item 2 does not
have to retake anything S7.3 measures. What item 2 still covers is the
**existing committed** RNGD envelope and card domain, both taken unbound: those
are closed-loop artifacts and re-taking them bound is a separate decision from
the open-loop refit.

**S7.3 IS DONE (2026-09-21).** The open-loop domain is measured, fitted and
installed: `profiles/calibration/openloop/rngd_card.accuracy.openloop.yaml`, six
points over sim L 11.730–37.965, `compared_metric: tpot_p99`, `outside_domain:
refuse`, `device_binding: numa_pinned`. 15 runs on npu0 of accelerator set
`6fe246ed1abf` (D80). Record and raw data:
`experiments/results/s73_npu_6fe246ed1abf/`; analysis:
`experiments/results/s73_openloop_refit.md`.

Three things it establishes that **S7.4 must absorb before selecting anything**:

* **The error changes sign.** +2.02 % at sim L 11.73 to −17.63 % at 37.97,
  crossing zero between L 11.73 and 16.55. No scalar expresses it.
* **The domain stops at sim L 37.965, measured not inferred.** 2.25 rps was run
  twice for that purpose and lands at a −27.85 % concurrency gap, outside the
  ±20 % guard. Four unpaired rates are kept as evidence in the domain's
  `provenance.unpaired_points`; only 3.5 rps is saturated.
* **S7.0's selection is invalidated and must be redone.** Its P2 candidate
  (sim L 74.498) is **unmeasurable open-loop** — twice the highest pairable
  concurrency. Its P3 candidate (sim L 37.965) **inverts**: the margin there
  moves from the closed-loop 7.51 % to **21.40 %**, above the global 18 %, so
  the per-point rule becomes the stricter of the two and the candidate becomes
  a P2-shape.

Measured run-to-run p99 TPOT spread, which S7.4 must report beside every
verdict: **0.116 ms (0.27 %)** at sim L 37.965 over three runs, and 0.776 ms
(1.97 %) at L 32.475 — it is not uniform, and it is larger at the lower load.

**S7.4 AND S7.5 ARE DONE (2026-09-21). S7 is complete.** V3-R ran, and its
result is not the one the work order expected:

* **§6 is still not established, and now for a stated reason.** The one rankable
  case is **circular** — the open-loop domain's robust 42.894 returns the
  measured 42.895 because that point was fitted from it. Leave-one-out gives the
  domain ±0.63 ms on held-out interior points, which is predictive accuracy and
  not a verdict count, and the **boundary points cannot be leave-one-out
  validated at all**.
* **What V3-R did establish is a counterfactual, and it belongs to §5.2/§8(3).**
  Consulted outside its arrival-process condition, the **closed-loop** domain
  false-passes a violating candidate by **4.911 ms = 42× the run-to-run
  spread**. D113 asserted that as policy; this is the number.
* **P2 is a true positive of the hold**: `sim feasible / planner refuse /
  measured saturated`.

`experiments/p2_evidence/results/v3r_verdict_accuracy.md`;
`patent2_evidence_map.md` §0(ii) and its §6 rows.

**What §6 would still need**: a verdict case whose domain point was **not**
fitted from the measurement that judges it — a hold-out design (which the
boundary points cannot support) or a measurement at an operating point the
domain already covers from other data.

**~~4. S7 / V3-R~~** — done; kept here only so the numbering below is stable. The gate this entry used to leave open — `--condition-mismatch warn`
now, or wait for item 1 — **is decided**: `WORK_ORDER_domain_scoping.md` §S7.1
picks the refit first, because a §6 number obtained with the condition check
switched off would support the margin claim and contradict the §8(3) scoping
claim in one table. `warn` survives as a labelled control arm only.

**S7.0 is done (2026-09-18) and it chose the candidates, on CPU, before any card
was touched** — `experiments/p2_evidence/results/v3r_candidate_selection.md`.
Three things from it change what this trip is for:

* **Both candidates are single-card `tp1-dp1` on `node_rngd0`**, not P2/P3's
  original shapes: V3's two are `mix(cuda-a40-…)` and cannot run here at all.
  They are re-read as verdict shapes — catch and rescue.
* **P2 case: 3.5 rps, `s128-t2048`, L = 74.498**, at the fixture's own 50 ms —
  (a) passes by 3.309 ms, (c) rejects by 6.640 ms. **P3 case: 2.0 rps,
  `s128-t8192`, L = 37.975, at a post-hoc 40 ms SLO** — (b) rejects by 1.748 ms,
  (c) passes by 1.963 ms. Two deployments, two thresholds, three repeats each.
* **The ≈21 % margin and the measurable P3 case do not coexist.** 21 % belongs to
  L ≈ 74.5, which is the P2 case; the P3 case sits at 7.51 %. Do not quote 21 %
  as the margin that rescued a candidate.

**And S7.0 hands item 1 a requirement:** all three operating points fall in
`L ∈ (25.181, 76.0)`, where the committed domain **has no point** — its lower
anchor is measured and its upper one (76.0) is itself interpolated from c64/c128.
So the refit must place measured points inside that interval, at or above 70 for
the P2 case and near 38 for the P3 one, or S7.4 will be testing a straight line
rather than a margin.

**4. The NPU exec-model spike's leftovers** (§2.0): a **wide-KV workload** to
test C.3's bucket-ladder explanation, **B.3's hold-out** (needs C.4, needs a
card), and **C.2** — whether the runtime alternates prefill and decode steps or
drains the prefill queue. C.2 cannot be settled by running both knob values
(they differ by 0.03 pp); it needs a step-timestamp logging option from Furiosa.

**5. ATOM (D20)** — still blocked on the vendor. Host I/O exceeds the kernels
and the device tracer's `.pb` schema is undocumented. Unchanged; listed so it is
not rediscovered.

**Two things the RNGD node cannot fix, so do not try.** The 10 rps E6 cell's
per-PE operating point is 139.4 against an envelope ceiling of 107.2, and the
envelope says that ceiling is pool-bound — no simulated point up there can be
given a measured reference (§2.3). And the residual on V3's P1 is an **A40**
measurement (S6(i)), not an RNGD one.

**Before running anything here**, `bash scripts/whichnode.sh`, and read
`docs/nodes/npu.md` — in particular the trap that cost a device slot: `npu0` was
held throughout by another tenant's pod while `alloc_status`, `furiosa-smi ps`
and the power reading all said it was free. Read the process list for `--chip`
flags instead. Today's `npu2` is the card the four-card inventory called `npu3`.

### 2.0 `WORK_ORDER_npu_exec_model_spike.md` — **STEP 0/A/B/C done 2026-09-17, conclusion (ii)**

The spike asked how much of the RNGD accuracy domain's sign-changing error is the
vLLM-vs-bucketed-AOT structural difference. Answer: **7 %** of the error span, and
at low load none of it. Full record in `docs/npu_exec_spike.md`; D90–D93.

**Do not redo these.** Three mechanisms the work order names are settled and two
of them must *not* be built: batch padding (P2) and group attention (P3) are
already inside the measured bundle, and modelling them double-counts. The
prototype on `spike/npu-exec-b-prototype` is **do-not-merge** and its default path
is proven byte-identical (R1/R2 anchors plus six identical CSVs).

**Still open, in the order they are worth doing:**

- **A wide-KV workload.** C.3 explains D17's saturation near three attention
  executions as the geometry of a power-of-two bucket ladder against a workload
  with a bounded KV spread — no runtime cap involved. That predicts more groups on
  a wide-KV workload and is currently unmeasured. Any node for the census half;
  the NPU node for the measurement.
- **B.3's hold-out.** Needs C.4, which needs a card. Never run.
- **C.2.** Whether the runtime alternates prefill and decode steps or drains the
  prefill queue is still unobserved, and B.2 showed the two knob values differ by
  0.03 pp, so it **cannot** be settled by running both and comparing. Ask Furiosa
  for a step-timestamp logging option, or find one.
- **C.5's staircase**, not run.

**NUMA: bind, and check that it took.** STEP C originally ran unbound, which the
repo already knew was worth up to 1.93x of throughput here (`p2ev` Appendix A.3).
Both harnesses now take `--numa-bind`, **default `auto`**, and `provenance`
records the affinity mask. Re-measuring C.3 bound agreed to 0.24 % — the cost
model was safe because it reads device cycles — but the wall-clock half of
`measure_envelope.py` and the committed RNGD envelope and accuracy domain were
all taken unbound and have **not** been re-taken. D94. Note the trap:
`/sys/class/rngd_mgmt/*` are virtual devices with no PCI parent, so resolve the
node through `furiosa-smi info` and the BDF, and verify with `taskset -cp` rather
than trusting the flag.

**Trap, and it cost a device slot.** `npu0` — the card every committed RNGD
measurement was taken on — was held throughout by another tenant's pod, and all
three vendor-level checks said it was free. `alloc_status` read all zeros,
`furiosa-smi ps` was empty, and power sat at idle. `docs/nodes/npu.md` now says to
read the process list for `--chip` flags instead. Everything in §C was measured on
`npu2` (PCI `45:00.0`) and is labelled as such; the card labels have shifted again
and today's `npu2` is the card the four-card inventory called `npu3`.


### 2.1 `WORK_ORDER_rps_aware.md` — **complete, STEP 0–6, PRs #60–#70**

Delivered, in the order it binds:

| step | what landed |
| --- | --- |
| 0 | D22 §4.4's "never terminates at 3.3 rps" was a **D26 artifact**; Exp 5 survives the retrospective check with zero differing fields |
| 1 | The RPS axis costs **21.4×**, not 6× (1 rps = 10.3× a 10 rps point) → E6a was 218 h, not 60. **D27** returned 1.55–1.72× byte-identically |
| 1.5 | §4.7's "regret 0 at every K" **partly retracted** (D30); top-K is not a cost lever on P/D corpora |
| 2 | **D28** — `slab3d` for `tp_d = 2·tp_p`, plus a 24-point measured latency table with `extrapolation: refuse` |
| 3 | The RNGD envelope from **concurrency 1**: tok/J spans **9.4×** while power spans 1.085×. "tokens/J is unimodal" is **not observed** |
| 4 | Envelope, accuracy domain, operating-point margins, `plan --rps`. **D29**, **D31** |
| 5 | **E5** the planner self-rejects D22's winner; **E6** there is a crossover; **E7** it survives losing any one calibration point but not the domain |
| 6 | This rewrite, plus `docs/PAPER_OUTLINE.md` |

**The regression ladder is now committed** (`outputs/d23fix/anchor/`, PR #72).
Absolute rule 1 requires a byte-identical proof for every sanctioned `serving/`
edit; before that PR only D25 and D26 had a committed artifact, and D27's and
D28's runs existed as untracked files on the NPU node alone. Read
`outputs/d23fix/anchor/README.md` before reading the hashes — a `SHA256SUMS` with
three lines instead of four is a *failed* anchor, not an agreeing one, and
`anchors.log` beside it is what says which.

**The one-sentence state of the science:** the planner can now price its own
predictor and finds a crossover on the RPS axis — and since 2026-09-11 **thirteen
of sixteen E6 cells rest on a measured accuracy domain**, up from one, with every
winner unchanged except the one the new domain rejects (§2.2, §2.3). Of the three
that do not, two rest on an A40 prefill leg at served concurrency 0.499 — costly
to measure, not impossible — and one runs at an RNGD concurrency of 139.4, above
anything the hardware envelope reaches.

Not delivered, deliberately: a third E6 fixture. The work order names
`pd-rngd-gpu-card`, `pd-rngd-gpu` and "STEP 2's asymmetric fixture", but
`pd-rngd-gpu.yaml` already yields the `A40 tp4 P + RNGD tp8 D` shape once
`--enable-pd` is on, so a separate ClusterSpecV2 would have duplicated it. Recorded
here rather than silently dropped.

### 2.2 A second A40 accuracy-domain point — **DONE 2026-09-10/11 on the A40 node**

**Closed.** The domain has three points, the E6 sweep was re-run on them, and
**seven of sixteen switchover cells moved from `extrapolated` to `measured`**
with every winner and every tok/J unchanged. Full write-ups:
`experiments/results/a40_lowload_envelope.md` and the re-run section appended to
`experiments/results/e6_rps_sweep.md`.

| | before | after |
| --- | ---: | ---: |
| domain points | 1 (conc 170.56) | **3** (4.043, 10.800, 170.56) |
| `measured` cells | 2 | **9** |
| `extrapolated` cells | 9 | **2** |
| `unknown` cells | 5 | 5 |

**The finding is TTFT, and it was not the one this section predicted.** Held
flat, the one point declared the simulator 1.97 % optimistic on TTFT everywhere.
Measured at served concurrency 4 and 11 it is optimistic by **~18 %** — a factor
of nine. Both are right about their own regime: at saturation TTFT is 40 s of
queueing, which the simulator models well; at low load it is ~150 ms of nearly
pure prefill, which it does not. It changes no E6 outcome, because no
recommended plan comes within 25 % of its TTFT SLO in this regime — but it would
in the tight-TTFT regime §2.5 still records as not quotable. TPOT, by contrast,
is nearly exact at low load and **changes sign** across the range (+0.59 % to
−1.42 %), so the one-sided margin correctly charges nothing at the bottom.

**What remains.** The two cells still `extrapolated` are the card fixture at
3.3 rps, whose winner's A40 leg is a *prefill role at served concurrency 0.499*,
below the domain's new floor of 4.043. **This is closable and an earlier draft of
this section said it was not** — see the correction in
`experiments/results/e6_rps_sweep.md`. An open-loop point at 0.0254 rps reaches
it: ~1.1 h per repeat at 100 requests, ~3.3 h at 300, on an otherwise idle A40.
Expensive, not impossible. ~~The five `unknown` cells are the ones that genuinely
cannot be reached from an NVIDIA node — they need a per-PE RNGD accuracy domain,
which needs the NPU node.~~ **Wrong on the second half, corrected 2026-09-11:**
they could not be reached from an NVIDIA node, but the per-PE domain needed no
node at all — §2.3.

**A method correction came out of it — `docs/deviations.md` D32.** The simulator
side must run the same number of requests as the hardware. At the script's
20-request default the sim's served concurrency lands −8.2 % and −31.7 % from the
measured points purely because the drain tail dominates a short run's wall; at
300 it is −0.8 % and −4.2 %. The four RNGD low-load domain points were taken at
the default, and `rngd_card_edf.yaml` discards two further points at a "40 %
below the hardware" gap that has the same signature. Unverified on that side and
testable with no NPU hardware, since the RNGD envelope is committed.

**Also delivered:** the A40 half of E6b now exists
(`e6b_measured_curve.py --hardware A40`), so a crossover sentence no longer has
to read "RNGD measured against A40 simulated" — but it must now say which
*protocol* each side was measured under, because they differ (below).

<details>
<summary>Superseded planning text, kept because it records what was expected</summary>

Two things in it turned out wrong, and both are worth knowing:

**"Every plan E6 recommends runs far below [170.56]."** They do not. The A40
operating points in the E6 winners are 90.8, 121.8, 125.3, 138.1, 157.1 and
169.9 — just *below* the fitted point, not far below it. Only the 3.3 rps prefill
leg at 0.499 is far below. The prescribed fix was still the right one, but the
mechanism is bracketing those points from underneath, not reaching down to them,
and the reachable ceiling was **seven** cells rather than fifteen. That ceiling
was computed from the committed operating points before the measurement and the
run hit it exactly.

**The prescribed route was closed-loop, and taking it would have been a mistake.**
The table below asks for `vllm serve` plus the closed-loop bench client. But the
existing 170.56 point came from `python -m bench run`, an *arrival-trace replay*,
and adding a closed-loop point to that domain puts two definitions of "served
concurrency 8" on one interpolation axis — the class of error D22 was. The A40
points were therefore measured **open-loop**, by
`experiments/scripts/measure_envelope_openloop.py`, whose reproduction of
170.5619 from the committed artifact is the check that all three points share an
axis. That choice is also what made the TTFT finding visible at all: both sides
replay the same arrival process, so **D19 does not apply** and TTFT is
comparable, which on the closed-loop route it would not have been.

The CUDA execution half described below was built anyway and is committed —
`measure_envelope.py --backend cuda`, `power_sampler_nvidia.sh` — because the
closed-loop curve is what makes the two protocols comparable to each other. Its
original text:

> **Only half of `measure_envelope.py` is vendor-agnostic, and an earlier draft of
> this section said otherwise.** The *analysis* half is: `summarise_point` already
> enforces A5 (served concurrency by Little's law, the pool floor, power recorded
> with utilisation from the same samples), and `read_sampler_csv` cares about a CSV
> schema, not a vendor. The *execution* half is hardcoded to FuriosaAI in three
> places, and all three need a CUDA path before any A40 point can be taken:
>
> | what | today | needed |
> | --- | --- | --- |
> | server launch | `furiosa-llm serve --devices npu:N:*` | `vllm serve` with `CUDA_VISIBLE_DEVICES` pinned to one card |
> | sampler | `power_sampler.sh` | a twin emitting the same columns from `nvidia-smi` at 1 Hz |
> | `--bench-python` | `/usr/bin/python3` | `.venv-vllm/bin/python`, which has `openai` |
>
> `experiments/scripts/lowload_sim_error.py` is the other half of the point and is
> pinned to RNGD by three module constants — `ENV`, `CLUSTER`, `DATASET`. Lifting
> them into arguments is the whole change.

All three were done. One correction to the table itself: `vllm` is **not on
PATH** — it lives in `.venv-vllm/bin/vllm`, and a bare `vllm` killed the first
closed-loop run at its first point.

</details>

### 2.3 The per-PE RNGD accuracy domain — **DONE 2026-09-11, and it needed no device**

**Closed, and the closure is a result.** `profiles/calibration/rngd_perpe.yaml`
is nine points from served concurrency 1.832 to 79.028. Write-up:
`experiments/results/rngd_perpe_accuracy_domain.md`.

**The premise this section used to rest on was wrong.** §2.2 and `docs/CLAIMS.md`
said the five `unknown` cells "need a per-PE RNGD domain, which needs the NPU
node". Per-PE and card are not two devices — they are two *simulator models* of
one physical card at TP=8, which is what
`experiments/results/rngd_card_vs_pe_model.md` does when it fits both to a single
real furiosa-llm run. So the measured curve was already committed (the RNGD-CARD
envelope, nine points, 1.00 to 107.2) and only the **simulator** side under the
per-PE fixture was missing. That is `python -m serving`, and it runs on any node.

The old recipe could not be used. RNGD-CARD and A40 offered each envelope point's
arrival rate to both sides and checked they landed at the same served
concurrency; here the model is slow enough to back up — at the rate that puts the
hardware at 1.00 it sits at 1.83 — so every point fails the ±20 % guard.
`lowload_sim_error.py --match served` pairs against the measured curve
interpolated at the *simulator's own* served concurrency, which is also the axis
`planner/util/operating_point.py` indexes the domain by.

| | |
| --- | --- |
| error sign | **pessimistic at every load** — so the one-sided margin charges it nothing |
| magnitude | +63.16 % at served 1.832 → +26.34 % at 19.885 → +32.70 % at 79.028 |
| cells `unknown` → `measured` | **4** (at conc 19.634 and 73.679), plans unchanged |
| cells whose winner changed | **1** |

**The one that changed is the point.** E6's per-PE winner at 10 rps is
`agg[furiosa:tp8]` at **4.956 tok/J** — D22's retracted headline reproduced
exactly. Its leg runs at served concurrency 139.4, above the 79.03 the domain
reaches, so `widen_error_bars` charges 42.12 %: 48.355 × 1.4212 = 68.7 ms against
a 50 ms p99 TPOT SLO. Re-ranked, the winner is `agg[cuda:tp4]` at 2.595 tok/J,
validity `measured`. The planner now rejects the headline on its own calibration.

The other four cells were **not** re-simulated and do not need to be: their
margin is 0, and a margin can only inflate a predicted TPOT, so it removes
candidates and never promotes one — with the recommended plan untouched the
ranking cannot move. That argument is load-bearing because of the next trap.

**Trap: the replay cache does not cost seconds.** `outputs/e6_rerun_cache_backup/
README.md`, D32 and PR #74 all say re-ranking under a changed domain is "a replay
costing seconds". The cache stores only the simulations that **succeeded**; every
candidate that dies in the KV allocator is re-run every time, and there are 114 of
those per point at 10 rps and **291** at 1 rps. Measured: the 10 rps row replayed
in 2.7 min (243 of 357 from cache); the 1 rps row was still going after 10 min
against 150 min originally. Read it as *minutes to hours*. All three documents
now carry the correction.

**What remains, and it needs the NPU node or nothing.** The 10 rps cell's 139.4
is above the hardware envelope's own ceiling (107.2, `pool_binding_above`), so no
simulated point up there can be given a measured reference at all. Extending the
domain would need the card measured above 107.2, which the envelope says is
pool-bound. This is a real ceiling, not pending work.

### 2.4 D32 on the card fixture — **DONE 2026-09-11**, any node

**D32 is closed.** `profiles/calibration/rngd_card_edf.yaml` is rebuilt at 300
requests: nine points from served concurrency 1.020 to 76.0, up from six from
1.09. Write-up: `experiments/results/d32_card_recheck.md`.

**The two points the file discarded were discarded for the wrong reason.** It
dropped them at a "40 % below the hardware" gap it attributed to throughput
error. Changing only `--num-reqs` from 20 to 300 moves them to **−3.1 %** and
**−2.4 %**; the gap was the drain tail. Their error changes sign too, −1.22 /
−1.97 % to +3.26 / +2.47 %, so readmitting them **removes** a margin near c15
rather than adding one.

**The split D32 asked for is measured.** Above 29.3 the artifact stops explaining
everything: at the rates matching c59.2 and c107.2 the simulator is still −36.4 %
and −58.5 % low at 300 requests, having moved 2.9× and 3.3×. Those stay refused.
The residual is the card model's real throughput ceiling — it saturates near
served 44 and 35 ms TPOT where the card reaches 107.2 and 67.88 ms.

**E6 is unchanged: 0 of 8 card rows moved**, every winner and tok/J identical,
margins 3.05 → 2.88 % and 12.52 → 11.68 %.

**The lesson worth carrying is why that had to be re-run.** A margin that RISES
can only remove candidates, so if the recommended plan survives the ranking
cannot move — that is how §2.3 avoided re-ranking four cells. A margin that
FALLS readmits candidates, and three did flip to feasible here. Check the
direction before reusing the argument.

**And the "replay costs seconds" claim now has both ends measured.** With the
corpus covering every candidate (3.3 rps, 318/318) a row really does replay in
**seconds**; the cost is entirely the candidates that die in the simulator and are
never cached — 36 of them at 1 rps cost **44 minutes**, because a failure at a low
arrival rate is a long simulated span before it fails. Whole eight-row re-rank:
1.26 h.

### 2.5 ATOM layerwise bundle (D20) — **needs the NPU node**, and probably the vendor

Unchanged. Host I/O exceeds the kernels and the device tracer's `.pb` schema is
undocumented, so no bundle ships and ATOM stays out of candidate generation and
Exp 4. Memory and power *are* measured. Resolution paths, in order of expected
effort: the trace schema from Rebellions; a torch backend registering device
`rbln`; a llama entry in vllm-rbln's native model registry.

### 2.6 A surrogate that can rank P/D — **MOSTLY DONE 2026-09-11**, any node

**D30 is resolved for K≥20.** `plan --surrogate` now defaults to
`BinnedRooflineRanker`. Write-up: `docs/surrogate_topk_regret.md`.

The fix was inside D30's own evidence. `tpj_then_floor` — an explicit tie-break
on the parallelism-sensitive term — had measured **byte-identical to
`roofline`**, because the proxy's TP/DP cancellation is algebraic while the
arithmetic is floating point: about one part in ten thousand survives, nothing is
ever exactly tied, and `sorted()` reads the dust as a preference. The ranker does
not ignore the parallelism axis, it lets rounding error pick for it. So make the
tie explicit — bin proxy tok/J by a relative tolerance, order inside a bin by
`roofline_tpot_ms` — and leave the coarse order (accelerator, `max_num_seqs`,
factor-scale gaps) alone, which is why it does not break the corpus `floor`
broke.

| | |
| --- | --- |
| corpora measured | **16**, up from 3 — including E6's two sweeps read one arrival rate at a time |
| dominance over the shipped ranker | 96 cells: **11 better, 0 worse**, 85 identical |
| false-infeasible at K=20 | **8 corpora → 2** |
| tolerance sensitivity | 0.001 / 0.01 / 0.05 give identical curves and identical top-K membership |

**What is still open, and it is a real limit.** At **20 rps** every
efficiency-ordered ranker is false-infeasible to K=50 on both fixtures while
`floor` finds a plan, so feasibility at the top of the load axis is decided by
the TPOT floor alone. `--top-k` is **not** yet a general cost lever. K=5 and K=10
are unchanged. And sixteen corpora are still only **two cluster fixtures** — a
third needs a corpus that does not exist.

**Two things not to repeat.** Ordering by the floor alone is worse than D30
recorded: the rate-keyed corpora catch it at regret **2.734** (1 rps) and
**1.980** (3.3 rps), not merely false-infeasible. And **rank fusion** — round-robin
merging the two orderings, no fitted weight — is false-infeasible at K=20 on 7
corpora against 2; the depth it gives up costs more than the complementarity
buys. Both stay in `exp_surrogate.py --rankers`, neither ships.

**Trap: a multi-point sweep's cache cannot be replayed by candidate id.** It
holds one entry per arrival rate under each id (`outputs/e6/pd-rngd-gpu/cache` is
660 files for 289 candidates), and the old loader would have kept whichever
sorted last — four rates silently mixed into one ranking. That is now refused,
and `--cache-rps` resolves such a corpus properly through the planner's own cache
key.

### 2.7 The rest of the tight-TTFT and asymmetric-P/D story — **any node**

D23 and D14 are **closed** (D26 and D28 respectively); what remains is narrower:

- The `slab3d` dim-1 calibration is fitted at **two** dims (a single colocated
  instance) and applied at **three** (a P/D pair), taking dim 1 to be the same
  axis in both. Hop counting supports it; nothing measures it. A P/D-side
  measurement would close the one assumption in that path.
- Only **TPOT p50** was fitted. TTFT and throughput under `slab3d` are unmeasured,
  so an asymmetric plan's absolute TTFT is not quotable.
- 68–114 candidates per E6 point fail in the simulator's KV allocator
  (`tried to load 92.00MB but only 25.49MB is available`) after passing the
  generator's memory bound. Deterministic, reproducible, and it means every "best"
  in E6 is the best of what evaluated. Reconciling the two memory models is
  D10-adjacent and unstarted.

### 2.8 Smaller, any node

- **PR #13** (`docs/slide-deck-ko`) — dispositioned by the consolidation sprint's
  STEP 4.3; see that PR for what was decided.
- **Remote branches are clean.** `origin` holds `main` and nothing else. The 17
  already-merged branches turned out to have been deleted already; the D22 chain
  and the ScenarioLab workspace branches went during the consolidation sprint, the
  latter after verifying their content reached the split repo; the twelve
  `feat/rps-step*` branches and `spike/d14-asym-tp` went after PR #72, once
  `git rev-list --count origin/main..<tip>` was 0 for each. Nothing on the spike
  branch was unique except its throwaway `serving/` edits, which A3 forbids
  merging — its findings live in `docs/d14_spike.md`,
  `docs/upstream_issues/llmservingsim-trace-column-overflow.md`,
  `outputs/d14/evidence/` and D28.
- **`docs/nodes/a5000.md` is thin and says so** — written from committed artifacts,
  not from the node. Fill it in from `scripts/whichnode.sh` when next on that
  machine.

---

### 2.9 The uncertainty-planner stack (patent 2) — **landed 2026-09-11 as three stacked PRs**

`WORK_ORDER_uncertainty_planner.md` A1–B3 was built on `feat/uq-*` from
`108e48a` and never opened as PRs while `main` took #71–#77 on top of the rps
design; the two had each built an accuracy domain. **D33** records the
reconciliation (main's `calibration.AccuracyDomain` stays, `refuse` default, the
per-candidate `MarginPolicy` in `planner/optimizer/margin.py` is the single
consumer, `UNMEASURED` → `outside_calibration_domain`). PRs: #78 A1 registry →
#79 A2–A5 margins → B1–B3 perturbation/sensitivity/measurement plan on top.
Left for the stack's owner: **STEP B4** (truth-degradation experiments E-B1–E-B3,
`feat/uq-b4-wip` holds the harness, stopped before a valid run) and **STEP B5**
(`docs/uncertainty_planner.md`, README, CHANGELOG). Left for the rps owners:
whether to flip the committed domains to `refuse` (three E6 cells become
undecidable) and the linear `margin_from_error` (under-corrects by 4 pp at 18 %).

## 3. Traps that have each cost a session

Recorded because they are not discoverable from the code.

**Planning a sweep's cost**

- **A low-RPS point is not one point.** Wall time is strongly superlinear as the
  arrival rate falls: at 300 requests on the R2 candidate, 3.3 rps costs **2.63×**
  10 rps and 1 rps costs **10.32×**. A six-point RPS axis weighs 21.4×, not 6×, and
  half of that is the bottom two rows (`docs/sim_cost_profile.md`). Budget the axis
  by summing multipliers, never by multiplying a 10 rps figure by the point count.
- **top-k pruning drops the optimum on P/D and heterogeneous fixtures.** 8.1× faster
  and it turned a FEASIBLE plan (`hp-00077`, 2.963 tok/J) INFEASIBLE. The planner
  warns when this may have happened — read that line. `SLIDE_OUTLINE.md`'s "regret
  0.000 at every K" was measured on an aggregated-heavy fixture and does not
  transfer. Fixing the surrogate is `WORK_ORDER_rps_aware.md` STEP 4.

**Reading a sweep**

- **A sweep's INFEASIBLE is not a result until you have counted the timeouts.**
  `pd_slo_sweep.py` prints INFEASIBLE identically whether candidates were rejected
  or never evaluated, and it **does not persist `PlannerOutput.rejected_summary`**.
  Recover the count by counting work directories with no `sim*.csv`. On the
  2026-09-03 tight re-run that was 71 of 222 and 126 of 252 — including the
  committed winner. This is D23's first symptom and it was misread twice.
- **Record effective concurrency, never requested.** The c1–c32 RNGD curve was
  labelled by the concurrency asked for; a 24-request pool meant the c32 point
  actually ran at **eff 21.2** (Little's law), and an exponent fitted across it
  read a ×1.74 interval as a doubling. That is the whole of D22.
- **Timeouts are not cached and are retried once per SLO point.** A sweep whose
  candidates hang costs `candidates × timeout ÷ workers` *per point*, and the
  envelope cache is written only when a point joins — kill a run mid-point and its
  completed simulations are lost, not resumed.

**Re-running an experiment after a policy change**

- **S1's `refuse` empties a measurement plan, and it looks like the plan is
  broken.** Every committed accuracy domain is fitted at `tp=1` (only
  `rngd_perpe.yaml` is `tp=8`), so on any fixture whose candidates are not tp=1 —
  including the E-A1 corpus, whose winner is a tp=4 A40 island —
  `--condition-mismatch refuse` holds every candidate, leaves no recommendation
  to flip, and reports **every** uncertain input as undecidable. That is D110
  working, not a regression, and it is what S4 has to quantify. An experiment
  that asks a different question (S2 asked what the default ranges change) must
  pass `--condition-mismatch warn` and say so in its output header; the
  measurement-plan numbers are otherwise about S1 and nothing else.
  **S4 quantified it: 312 of 324 held, 0 feasible, and the refused axis is
  `dp` (228) far more than `tp` (66).**
- **A committed experiment script can stop reproducing its own committed
  numbers, with nothing broken.** `ea1_margin_modes.py` on `main` returns 0
  feasible for conditions (c) and (d), where `ea1_margin_modes.md` publishes
  50/244/30 — because the planner's default changed under it (S1). The fix is
  `--condition-mismatch warn`, which asks the question the script was written
  to ask. Before concluding a script is broken, check whether a *default* moved;
  before re-publishing its numbers, check which question the flags now ask.
- **`--out-dir` does not redirect `--out-json`.** `eb1_regret_vs_budget.py`
  defaults its JSON to the fixed `outputs/uncertainty/eb1/eb1_regret_vs_budget.json`
  — a **committed** artifact — whatever `--out-dir` says. A re-run overwrote it on
  2026-09-17 and it was restored from git. Pass `--out-json` explicitly, and
  `git status outputs/` after any experiment re-run.
- **A script whose default output path is a committed artifact will overwrite
  it, and `v3_slo_sweep.py` was the second one found.** It wrote
  `experiments/p2_evidence/results/v3_slo_sweep.json` with no way to redirect,
  and fired once during S4 before `git status` caught it. It now takes
  `--out-json`, and `--ea1` requires it. **Two of these have now been found by
  tripping over them**; if you add an experiment script, give it an output flag
  before you give it a default.
- **Reproduce an E-B result with the invocation the result md records, not the
  script's defaults.** E-B1's committed numbers are `--exhaustive --k 1 2 3`, 231
  degraded sets; without `--exhaustive` it samples ten sets per k, which is a
  different experiment whose numbers read like a regression.
- **E-B3 is the one E-B experiment that re-simulates.** Its committed result
  records 34.8 minutes for that side; a re-run on a shared A40 node was cut off at
  90 minutes with ~306 simulations done. Give it an idle machine and
  `experiments/scripts/livelock_watch.sh`.

**Measurement**

- **A single parallel-transfer trial is not reproducible.** Host buffers are not
  NUMA-bound and placement is fixed at allocation; two runs disagreed by 38 % on
  the 4-GPU figure and the same-node/cross-node ordering reversed. Repeat over
  independent allocations, report median and spread.
- **Pageable D2H is allocator-bound, not link-bound.** It peaks at 16 MB and drops
  ~4.5× above, because PyTorch's CPU allocator stops caching there. Read pinned.
- **Peak is not sustained.** On RNGD best-of-N overstates a bulk copy by 25 %; on
  the A40 it does not (0.998). Compose a fabric bandwidth from the *same* statistic
  on both legs.
- **Judge a power reading by its utilisation.** ATOM's first pass read 30 % low at
  36 % util; a single-PE RNGD reading understated its card 4×. `active_util_pct` is
  recorded beside the power for exactly this reason.
- **Never subtract a constant host round-trip floor on ATOM** — per-call cost
  scales with bytes moved.

**Harness**

- **`livelock_watch.sh` leaked one orphaned `cat` per stopped run, for eight
  days.** 250 of them were found idle on 2026-09-18, oldest 8 days, all
  `ppid=1` and all blocked in `wait_for_partner` — a FIFO `open()` waiting for
  a writer that was already dead. The drain at the end reopened the FIFO
  *after* the child had been TERMed, so nothing was left to open against, and
  the EXIT trap then unlinked the name. One leak per run ending in verdict 3 or
  4; ~28 MB PSS and 250 PIDs by the time it was noticed. **Fixed** — the read
  end is opened once on fd 3 and the drain inherits it (`cat <&3`), so no
  `open()` can block and the drain is reaped. Measured before and after on the
  same verdict-3 case: old +1, new +0.
- **That fix was half-applied, and the other half hung the parent for three more
  days.** #115 converted the drain to `cat <&3` but left the read loop ending
  `done <"$FIFO"` — the same `open(2)` on the same FIFO, so the same block, just
  in the parent instead of an orphan. A child that exits before the loop attaches
  leaves no writer for it: 5 hangs in 40 runs of `-- bash -c 'exit 7'`, and the
  intermittently red `tests/test_livelock_watch.py` (§2.11). **The rule is the
  general one, not the line:** once a FIFO's read end is open on a descriptor,
  every reader must inherit it (`<&3`), because opening a FIFO by name waits for
  a *future* writer and a dead child is not one. The `/tmp/livelock_watch.*`
  files §3 reported accumulating were this — a process hung in `open()` never
  reaches its EXIT trap — and not a second leak.
- **Three different self-match filters in a row found "survivors" that were the
  checking command itself.** `CLAUDE.md` warns about `pkill -f` and
  `ps | grep`; the same trap also defeats a *structural* argv match written in
  Python, because the literal you are matching on is in your own script's text
  and `/proc` shows your own `bash -c`. **The only filter that held was
  excluding self AND every ancestor by PID** — walk `PPid:` up to init and skip
  that set. Do that before believing any "still alive" count.
- **`experiments/scripts/bench_furiosa_endpoint.py` ignores `arrival_time_ns`** and
  fires everything at once, while `python -m serving` replays it. Feeding both the
  same file compares a burst against a spread arrival process; the difference lands
  entirely in TTFT. That is D19. Use `outputs/envcheck/rngd20_burst.jsonl` on the
  simulator side.
- **The Chakra converter no longer cares about `PATH` — it cares which venv you
  launch.** Until D27 `python -m serving` shelled out to a bare `python -m chakra`,
  so the venv had to be *on* `PATH` and not merely invoked by full path; that is
  what D23 turned out to be. Since D27 the frontend converts in-process, so there
  is no second interpreter to resolve and no `PATH` export to forget — but a venv
  without a chakra at `protobuf>=7.35.1` now raises at the first conversion instead
  of silently converting wrongly.
- **Run a multi-hour sweep detached** (`setsid`/`nohup`/`tmux`), not as a job owned
  by an interactive session. One was killed 30 minutes into its second fixture and
  lost 84 completed simulations.
- **The SLO sweeps do not price the fabric.** The simulator charges the P/D handoff
  at zero unless `--pd-transfer-model bandwidth` is passed, and only
  `experiments/scripts/pd_sim_network_sweep.py` passes it.
  `experiments/scripts/run_exp_pd.sh` is planner-side on both drivers.
- **`link_bw` is an overloaded scalar in multi-node configs** — the `none`-mode
  control moves too, which the driver's pass criteria do not anticipate at tp1.

**Environment**

- **One venv per vendor.** `.venv` = planner + analytical sim, no device, never
  install vLLM into it. `.venv-vllm` = CUDA. `.venv-rbln` / `.venv-rbln-vllm` =
  ATOM. System `python3` = FuriosaAI.
- **`.venv` needs `fastapi`/`uvicorn` only for ScenarioLab**, which has left. If a
  checkout predating the split fails `pytest` at collection, that is why.
- **ASTRA-Sim is built `RelWithDebInfo` (`-O2 -g`), not `-O3`.** `scripts/compile.sh`
  passes no `-DCMAKE_BUILD_TYPE` and the analytical `CMakeLists.txt` defaults to
  `RelWithDebInfo` (its `# Default: Release` comment is wrong). **Do not rebuild it
  to "optimise"** — build flags are part of a result's provenance, like the
  `UPSTREAM_COMMIT` pin, and `-O2`→`-O3` is worth single-digit percent here.
- **`furiosa.torch` and `rebel` must import after `torch`**, behind the
  `# isort: off` guards. `ruff check --fix` reordering them breaks every run.
- **Multi-device work on NPUs uses one subprocess per device**, not threads.

---

## 4. Invariants that are easy to break

Beyond `CLAUDE.md`'s absolute rules, four that recent cycles exercised:

1. **Absolute rule 3 covers hardware *presence*, not just hardware numbers.** Never
   claim a result from hardware `scripts/whichnode.sh` does not list. Artifacts
   measured on another node stay valid as measurements *of that node* — do not
   re-run, extend, or relabel them.
2. **Retract in public.** D18, D19, D20 and D22 each state what was claimed, what
   was measured, and what it changes. When a committed number turns out wrong,
   correct every site that carried it and say so — do not quietly overwrite. The
   superseded text stays, marked, because it records what was reasonable to believe.
3. **A pruning stage must be a relaxation of the feasibility test**, and **a mock
   predictor must respect the same physics as the bounds.** Both have been violated
   before and both were caught by the oracle-agreement test.
4. **A synthetic bundle must never be able to shadow a measured one.** Tier 0/1
   bundles carry a `-t0`/`-t1` hardware-label suffix, are gitignored, and propagate
   `profile_tier` into every plan built on them. A profile whose `sim_hardware` ends
   in `-t0`/`-t1` without a `datasheet:` block is rejected at load time.
