# The uncertainty-aware planner — design, CLI, results, limits

`WORK_ORDER_uncertainty_planner.md` STEP B5. This is the reference for
`--accuracy-domain`, `--measurement-plan`, `fit-accuracy-domain` and
`measure-apply`: what they compute, what they are allowed to claim, and where
the numbers behind each claim live.

**Read `docs/deviations.md` D33 first if you are touching the accuracy domain
itself.** Two implementations of "the simulator's error as a function of the
operating point" were built in parallel; one was kept, and D33 records which and
why. This document describes what landed.

**A second work order has since amended it.** `WORK_ORDER_domain_scoping.md`
S1–S4 (**D110–D113**) added the application conditions a domain answers under,
a default range per grade, the traffic key on a link item, and the measurement
of what the first of those costs. Where this document says "domain" without
qualification it now means "domain **for this configuration**"; §2.2.1 is the
part to read first.

**Everything here is opt-in.** Without `--accuracy-domain` the planner applies no
automatic margin and its output is byte-identical to the pre-uncertainty path
(work order rule A4, guarded by the golden-output regression tests).

---

## 1. The problem, in one paragraph

A planner that ranks deployment candidates on simulated metrics is ranking
*predictions*, and a prediction carries an error that is not constant. On the
RNGD card the simulator's TPOT error is **+11.6 %** at served concurrency 3.9 and
**−18 %** at 76 — not merely different in size but opposite in sign (D29). One
global margin is therefore too loose somewhere and too tight everywhere else, and
D22 is what "too loose at the top" looks like: a winner passed a 50 ms TPOT SLO
at a predicted 48.41 ms, and 48.41 × 1.18 = 57.1 ms (that product is the pre-D70 arithmetic; the corrected margin gives 59.04 — same verdict). The configuration was
infeasible and nothing in the pipeline could see it.

Two things follow, and they are the two halves of this work:

* **Stage A** — size each candidate's margin from *its own* operating point, and
  refuse to give a verdict at all where nobody has measured the error.
* **Stage B** — when the plan is uncertain, say *which measurement would change
  it* and what that measurement costs, rather than asking for everything.

---

## 2. Design summary (work order §2)

### 2.1 The uncertain-input registry

Every planner input that is not a measurement becomes an `UncertainInput`
(`planner/uncertainty/registry.py`) carrying an id, a **kind**, a **source
grade**, the current **nominal** value, a sourced **range**, a **measurement
cost**, and the set of candidates it affects.

Five kinds: `sim_error`, `profile`, `link_bw`, `link_lat`, `power`.

Two rules govern the whole table, and they are the reason it can be trusted:

* **A1 — no unsourced range.** A range with no citation is `unbounded`, never a
  guessed number. `profiles/uncertainty/grades.yaml` carries 15 rows and every
  numeric one names its source file; a `GradeRule` validator rejects a number
  without one.
* **A2 — no data is `null`, not zero.** An input with no measurement gets no
  range, and an input with no range gets no regret — it is listed under "cannot
  be decided before measuring", which is a stronger statement than any ranking.

What that yields today (`profiles/uncertainty/grades.yaml`):

| kind | grade | range rule |
| --- | --- | --- |
| `sim_error` | `measured` | `calibration_derived` — from the domain's own points |
| `profile` | `analytical` / `calibrated` | `symmetric_fraction` — Tier 0/1 operator error |
| `profile` | `vendor_spec` / `placeholder` / `user_defined` | `unbounded` |
| `link_bw`, `link_lat`, `power` | every grade present | `unbounded` |

**Since S3 (D112) a `link_bw` item is one per link PER TRAFFIC KIND**, not one
per link: `link_bw:pcie-a40a-02@all_reduce/w4/bulk/unpinned`. A link does not
have one bandwidth — the A40 node's PCIe bridge measures 25.0 GB/s for a direct
copy, 19.29 for a two-rank all-reduce and 8.8 for the four-rank all-reduce a
tp=4 island runs, against a `vendor_spec` 64.0 — so the item is keyed by
`(collective, world_size, msg_size_class, binding)` and the same wire carrying
two kinds is two items with two measurement costs. `link_lat` stays one item per
link: `path_latency_ns` applies it to all traffic alike and no link latency here
is measured at all. Use `parse_link_item_id` rather than splitting on `:`.

Twelve of the fifteen rows are `unbounded`. That is the honest state of this
repository's inputs, not a placeholder to be filled in later: `link_bw`'s
vendor-spec ratio was retracted (the numerator was a host↔device measurement and
the denominator a GPU↔GPU vendor spec whose own comment says it is not a measured
p2p bandwidth), and `power`'s was retracted because the CSV it was to be derived
from holds no power data.

### 2.2 The accuracy domain

An **accuracy domain** is the simulator's error as a piecewise-linear function of
the **served concurrency** `L` the candidate actually runs at, per hardware. It
lives in `planner/predictor/calibration.py` as `AccuracyDomain` / `AccuracyPoint`
and is fitted into `profiles/calibration/*.yaml`.

`L` is **served**, never the client's requested concurrency — a request pool too
small to keep the server busy makes the two differ by 30 % or more, which is
precisely how D22's retracted top point came about. Two ways to obtain it, both
in `planner/predictor/accuracy_domain.py` (which holds nothing else since D33):

* `served_concurrency_from_sim(latencies, wall)` — Little's law as a time
  average, `sum(latency) / wall`, off a trace. This is what D22 published.
* `served_concurrency_little(...)` — solved forward from predicted latencies
  where no trace exists (the surrogate stage).

**Outside the measured range the default is `refuse`.** The candidate comes back
`unmeasured` and is rejected as `outside_calibration_domain` — *not* as
infeasible. The alternative, `widen_error_bars`, extrapolates the nearest error
by |slope| × distance and is what the three domains on the default load path
still carry explicitly, so every E5/E6 number predating D33 is unchanged. (The
fourth, D102's open-loop A40 domain in `profiles/calibration/openloop/`, carries
`refuse`; `load_accuracy_domains` globs non-recursively and never sees it unless
it is named, which is why it can exist beside the other A40 one at all.)
Flipping those three to `refuse` is a science change and is deliberately not
done here (D33 §2).

#### 2.2.1 The conditions a domain answers under (S1, D110)

A domain is not only a curve over `L`. It also states **the configuration its
real side was measured under**, and it may not be consulted for a candidate
that differs on one:

| field | example | what it is |
| --- | --- | --- |
| `hardware` | `A40` | which accelerator; refused if the file is mis-filed |
| `parallelism` | `{tp: 1, pp: 1, dp: 1}` | the degrees the REAL side ran at |
| `placement` | `{islands: 1, device_binding: unpinned}` | how many islands of this hardware, bound how |
| `arrival_process` | `open_loop` / `closed_loop` | how load was offered (D19, S4) |
| `model`, `variant`, `workload_shape` | `bf16` | what was served |

`AccuracyDomain.check_conditions()` compares them **per island assignment**,
because a candidate may put two islands of one hardware at different
parallelism and there is no honest single `tp` for that pair.

**Not stated on either side means not compared** — the same convention
`model`/`variant`/`workload_shape` have had since D33. An unstated condition is
carried as a `skipped` warning, never as a match: a plan resting on an unchecked
condition should say so. This is why `binding: unknown` is not the same claim as
`unpinned`.

**A mismatch is its own refusal**, `CALIBRATION_CONDITION_MISMATCH`, carrying
`mismatch_fields` and `required_measurement` — the configuration a measurement
would have to be taken at. It is a **different** refusal from
`OUTSIDE_CALIBRATION_DOMAIN`, and the two ask for different measurements: a
wider load range against a measurement at the candidate's own configuration.

The policy key is `--condition-mismatch {refuse,warn}`, **default `refuse`**,
symmetric with `outside_domain: refuse`. `warn` applies the domain anyway and
records the mismatch; it exists to measure what the refusal costs, not to plan
with.

`profiles/calibration/index.yaml` lists every registered domain and its
conditions, so a refusal can say what *does* exist ("measured at tp=1; you asked
for tp=4") instead of only that nothing matched. It is a copy, and
`tests/test_calibration_condition.py` rebuilds it from the files and compares —
regenerate it rather than hand-editing.

**What the committed domains state**, after S4 (D113):

| domain | tp | islands | binding | arrival | variant |
| --- | ---: | ---: | --- | --- | --- |
| `a40.accuracy.yaml` | 1 | 1 | `unpinned` | `open_loop` | `bf16` |
| `openloop/a40.accuracy.openloop.yaml` | 1 | 1 | `unpinned` | `open_loop` | — |
| `rngd_card_edf.yaml` | 1 | 1 | `unknown` | `closed_loop` | — |
| `rngd_perpe.yaml` | 8 | 1 | `unknown` | `closed_loop` | `bf16` |

**`model` is stated on none of them, deliberately.** This repository holds two
strings for one set of weights — `meta-llama/Llama-3.1-8B` in the envelope paths
and `NousResearch/Meta-Llama-3.1-8B` in the open-loop A40 domain's own
provenance — and the check compares strings, so stating either would refuse on
the mirror name rather than on a model difference. It needs an alias policy that
does not exist. Measured: stating `model` and `variant` on all three changes no
count (D113).

### 2.3 Per-candidate margins and the three-way verdict

`planner/optimizer/margin.py` turns (candidate, prediction) into a
`MarginDecision`. `GlobalMargin` is the old one-number behaviour, kept as the
default. `AccuracyDomainMargin` interpolates each hardware's measured error at
*that candidate's* operating point, takes the worst across the candidate's
islands, and reports `unmeasured` when any of them falls outside its domain.

A margin only ever **inflates** a prediction (`max(0, error)`): a negative error
means the simulator was already pessimistic there, and taking credit for that
would shrink the prediction rather than harden it.

The verdict is three-way (`exhaustive.judge`), and the middle case is the point
of the whole exercise:

1. nothing could be margined at all → `UNMEASURED`, decided *before* the SLO
   check, and never entered as an infeasibility — an undecidable candidate must
   not surface as `closest_plan`, which is a claim about how narrowly something
   missed. Since S1 this case has **two distinct grounds**, and the rejection
   says which: `OUTSIDE_CALIBRATION_DOMAIN` (a domain applies here but the
   operating point is past the end of its load axis) and
   `CALIBRATION_CONDITION_MISMATCH` (no domain was measured under this
   configuration at all — §2.2.1). Collapsing them would hide that they ask for
   different measurements;
2. it **passed**, but on a metric whose margin is unknown → `UNMEASURED`. A
   violation is real whatever the coverage, because a margin only inflates; a
   *pass* that rests on an unmeasured metric is a gap, not a verdict;
3. otherwise the feasibility report decides.

Coverage can be partial. A domain fitted by comparing a burst simulation against
a closed-loop bench has TPOT points and no TTFT ones — queued requests inflate
sim TTFT by two orders of magnitude there, so there is nothing honest to compare
(D19). Such a decision margins TPOT and reports TTFT in `unmeasured_metrics`.

Since S4 that same fact is also a **stated condition**: `rngd_card_edf.yaml`
says `arrival_process: closed_loop`, so under the default `refuse` an open-loop
run does not reach the partial-coverage path at all — it is held first. The two
mechanisms overlap and the condition is the blunter one; see §5.

### 2.4 Closed-form perturbation

The central Stage B decision: **a perturbation is post-processing of cached
predictions, not a re-simulation** (`planner/uncertainty/perturb.py`). That is
what makes `inputs × grid points × candidates` finish in seconds instead of
weeks.

| kind | what moves | rule | exact? |
| --- | --- | --- | --- |
| `SIM_ERROR` | the margin `m_c` | the domain's interpolated value is moved inside its range; **the prediction does not move** | exact by construction |
| `PROFILE` | TTFT, TPOT, throughput, **energy** | `×(1+δ)` on latencies and on energy, `/(1+δ)` on rates; watts untouched | first-order |
| `POWER` | energy, tokens/J | `×(1+δ)` on average power; latency unchanged | first-order |
| `LINK_BW` (`p2p`), `LINK_LAT` | TTFT of P/D candidates | the KV transfer is re-priced with `kv_transfer.transfer_ms`, the same helper the planner itself uses | **exact** |
| `LINK_BW` (`all_reduce`) | nothing — **it declines to answer** | the value enters through the simulator's own `link_bw`; no closed form exists, so the item returns `requires_resimulation` | **not priceable** |

**`PROFILE` moves energy, and that is a correction** (D34). The rule originally
held energy fixed, reasoning that energy was `POWER`'s and moving it here would
double-count. Energy is watts × seconds; a PROFILE error is an error in the
seconds and a POWER error in the watts, so the two compose — apply both in either
order and energy comes out `×(1+δp)(1+δw)` exactly once. The simulator settled
it: `serving/core/power_model.py` accumulates active energy as
`(active_power − idle_power) × latency_s`. Holding energy fixed made every
`PROFILE` item score `ΔR = 0` under a `minimize_energy` objective unless it
happened to cross an SLO boundary, so the plan systematically under-valued the
most expensive measurement in `costs.yaml`.

**An intra-island `LINK_BW` item cannot be priced in closed form, and saying so
is the correction S3 made** (D112). The rule used to be the P/D one for every
link, and the module claimed the handoff was "the only place a link bandwidth
reaches a predicted metric today". It is not: an intra-island link's bandwidth is
the `min` that `island_interconnect` reduces into the simulator's scalar
`link_bw`, which prices every TP collective inside ASTRA-Sim. On a corpus built
with `enable_pd=False` the closed form therefore moved nothing, and every such
item swept to a regret of exactly zero and was reported `inert` — "measured, it
would change no plan" — including `pcie-a40a-02`, the input that explained V3's
−43.44 % TPOT error for 0.114 h of measurement. These items now go in their own
`needs_resimulation` bucket and `--resimulate-top` prices them for real: at 8.8
GB/s instead of 64.0 the same candidate predicts 60.09 ms p99 TPOT against 64.62
measured, an error of −7.01 %.

`LINK_BW`'s `p2p` rule is exact for a reason worth knowing: the envelope cache stores the raw
simulator output from *before* `apply_pd_transfer_cost`, and the planner adds the
transfer term itself afterwards. Applying the rule to post-evaluation metrics is
therefore the same arithmetic, not an approximation. Applying it to raw cache
metrics would add a difference to a term that is not there, so `perturb` refuses
a plain `dict` and demands a `JudgedMetrics` wrapper — a type, not a docstring.

### 2.5 Flip detection and decision regret

`planner/uncertainty/sensitivity.py` sweeps each input across its range on a grid
(§2.5's default five points: `lo`, `lo+w/4`, `nominal`, `hi−w/4`, `hi`, with
nominal included deliberately because a `ratio_floor` range puts it at `hi`).

* a **flip** is the recommended plan's id changing at some grid point;
* `ΔR_i`, the **decision regret reduction**, is the grid-point mean of
  `V_g(π*(g)) − V_g(π̂)` on the objective's primary metric.

**The infeasible case has a definition that took a bug to settle.** When the
current recommendation `π̂` is infeasible at a grid point it cannot be deployed,
so what the operator actually gets is that point's best *feasible* value, and the
penalty is charged on top of it as SLO overshoot:

```
V_g(π̂) = V_g(π*(g)) − penalty × overshoot_ratio(π̂, g)
```

so the contribution is `penalty × overshoot_ratio` and **ΔR ≥ 0 by
construction**. The draft's `value − penalty × overshoot` had no comparison
baseline and let an infeasible recommendation outscore a feasible one — B2's
first test produced a *negative* regret and that is how it was found. An
`UNMEASURED` incumbent is charged a whole `penalty`: a passing report's
`worst_overshoot` is 0.0, and charging that would price an unverifiable
recommendation at no regret at all.

`penalty` defaults to the largest objective magnitude observed across the grid.
It dominates the ranking, so it is recorded in
`provenance["uncertainty"]["slo_penalty"]`.

### 2.6 The measurement plan

`planner/uncertainty/measurement_plan.py` turns `ΔR_i` and
`profiles/uncertainty/costs.yaml` into an ordered, budgeted queue. Three rules:

* only `ΔR_i > 0` entries are candidates — an input the sweep cannot move is not
  worth a server-hour however cheap;
* the order is **`ΔR_i / cost_i`** — value for money, not raw value;
* an input with an unbounded range is neither ranked nor dropped. It has no
  regret to compare because nothing bounds it, so it goes in its own
  `undecidable` list.

Every input lands in exactly one of five buckets, and the last two are the ones
that are easy to conflate:

| bucket | what it means |
| --- | --- |
| `items` | ranked and inside the budget |
| `uncovered` | worth doing, the budget ran out |
| `inert` | swept, and moving across its whole range changed no decision |
| `undecidable` | no width of its own **and** no default for its grade (D111) |
| `needs_resimulation` | width known, **no closed form prices it** (D112) |

`inert` and `needs_resimulation` are opposite claims and were one bucket until
S3. An `inert` input is one a measurement would not repay; a
`needs_resimulation` input may be the most valuable on the list and the sweep
cannot say. Run `--resimulate-top` to turn it into a rank.

An item whose cost is unknown is still planned and sorts last; pretending to know
its cost would be the invention the work order forbids. Ranks number the
*queue*, so an item that did not fit the budget keeps the rank it would have had
— "number 3 did not fit" is more useful than a renumbered list.

Current costs (`profiles/uncertainty/costs.yaml`):

| kind | hours | exclusive | method |
| --- | ---: | --- | --- |
| `sim_error` (one operating point) | 0.041 | yes | serve the model, one bench run |
| `link_bw` | 0.114 | yes | `gpu_host_bandwidth.py` / `rngd_collective_probe.py` |
| `power` | 0.172 | no | `nvidia-smi power.draw` alongside a real run |
| `profile` | 2.1 | yes | `python -m profiler profile` |
| `link_lat` | — | — | not implemented; sorts last |

The cheapest measurement in the table is the accuracy domain, at **1/51 of a
profiling run**. That ratio is the practical content of the whole Stage B claim.

---

## 3. CLI

```bash
# Stage A — per-candidate margins, unmeasured rejection, the registry
python -m planner plan --service ... --cluster ... --accuracy-domain

# ... against exactly these calibration files
python -m planner plan ... --accuracy-domain profiles/calibration/rngd_card_edf.yaml

# Stage B — also rank what to measure next
python -m planner plan ... --accuracy-domain --measurement-plan \
    [--budget-hours 8] [--grid 5] [--slo-penalty X]

# fit a domain from paired real/sim runs
python -m planner fit-accuracy-domain \
    --real outputs/.../real_c16.json --sim outputs/.../sim_c16.csv \
    --hardware RNGD-CARD --service examples/service_specs/llama31-8b.yaml \
    --arrival-process closed_loop --metric tpot --out profiles/calibration/x.yaml

# fold a measurement back in and re-plan off the same cache
python -m planner measure-apply --plan outputs/plans/plan.yaml \
    --input link_bw:fabric-rngd0-a40a --value 13.0 --source measured \
    --cluster experiments/configs/clusters/pd-rngd-gpu-card.yaml

# A keyed link_bw measurement is FILED beside the spec value, not written over
# it: the group size comes from the key and --method is required (S3, D112).
python -m planner measure-apply --plan outputs/plans/plan.yaml \
    --input 'link_bw:pcie-a40a-02@all_reduce/w4/bulk/unpinned' --value 8.8 \
    --method 'nccl-tests all_reduce_perf 2.27.5' \
    --raw outputs/p2_evidence/link/tp4.json \
    --cluster experiments/configs/clusters/pd-rngd-gpu-card.yaml
```

Notes that are easy to get wrong:

* `--condition-mismatch {refuse,warn}` decides what happens when no domain was
  measured under a candidate's configuration (§2.2.1). **`refuse` is the
  default**, and on a fixture whose candidates are not `tp=1, dp=1, islands=1`
  it holds a great many of them — on E-A1 it holds 312 of 324 and leaves no
  recommendation (only `rngd_perpe.yaml` is fitted at tp=8; the rest are tp=1,
  and all four are one island at dp=1). **That is the policy working, not a broken run**, and the
  suggestions name the measurements. `warn` measures what the refusal costs;
  an experiment that passes it must say so in its output header, because the
  numbers are then about something else.
* `--measurement-plan` **requires** `--accuracy-domain`. Without the domain there
  is no margin policy to compute regret against, and the CLI says so rather than
  falling back.
* `--accuracy-domain` and `--ttft/--tpot-margin-percent` are **mutually
  exclusive**: one derives a margin per candidate, the other applies one to all.
* `--calibration-bucket` must be a **canonical key** (`in_*-out_*-rps_*`); a
  human label is never accepted as a lookup key. It asserts that a different
  workload's error applies here, so it is recorded in provenance and warned about
  in the output.
* `measure-apply` never overwrites a measured file. It writes a **copy** of the
  cluster or calibration yaml and prints the re-plan command (absolute rule A3).
* A **keyed** `link_bw` input appends to the link's `measurements[]` on that copy
  and leaves `bandwidth_gbps` and its `source` alone, so the datasheet figure and
  the measured one can still be compared. An **unkeyed** one names no traffic and
  so moves the spec value itself, as it did before S3. `--method` is mandatory for
  a keyed `measured` figure: this repo's torch probe and `nccl-tests` are not
  interchangeable evidence, which is why S6(ii) is still open.

---

## 4. Results

### E-A1 — one margin for everything vs a margin per candidate

`experiments/uncertainty/results/ea1_margin_modes.md`. The same 162 cached
simulations judged four ways, so the comparison is of *decision rules*, not of
predictions: 2592 cache reads, **0 misses**, nothing simulated.

| condition | feasible | recommended | energy | tokens/J | rejected |
| --- | ---: | --- | ---: | ---: | --- |
| (a) no margin | 70 | `mix(rngd0+rngd1)-s256-t8192` | 60 350 J | 3.1635 | slo_violated 254 |
| (b) global 18 % (D22) | 10 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 73 560 J | 2.5954 | slo_violated 314 |
| (c) accuracy domain, `widen_error_bars` | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 73 560 J | 2.5954 | slo_violated 274 |
| (d) accuracy domain, `refuse` | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 73 560 J | 2.5954 | slo_violated 244, **outside_calibration_domain 30** |

Read the result document for what this does and does not establish. Two things
worth carrying: (b) reproduces D22's winner and tok/J to the last digit, so the
chain from fixture to verdict is anchored; and the difference between (c) and (d)
is **30 candidates that stop being called infeasible and start being called
unmeasured** — the same evidence, a different and more honest verdict.

**Reproducing that table now needs `--condition-mismatch warn`**, and this is
the first thing to know before re-running anything here. The committed script
under the planner's own default returns **0 feasible for (c) and (d)** with 276
held, because every committed domain is fitted at one island with `dp=1` — and
at `tp=1` except `rngd_perpe.yaml`, which is tp=8 — while this fixture's
candidates are spread over tp 1–4, dp 1–2 and two islands. Nothing regressed; the default changed
(S1, D110), and `warn` is what asks the older question. See §4's E-A1 (e).

### E-A1 (e) — what the condition refusal costs (S4, D113)

`experiments/uncertainty/results/ea1_s4_condition_refuse.md`. The same cache
and the same four rules, plus **(e)**: per-operating-point margin refusing both
an operating point outside the measured range and a configuration no domain was
measured under.

| | (d) | **(e)** |
| --- | ---: | ---: |
| feasible | 50 | **0** |
| slo_violated | 244 | 6 |
| outside_calibration_domain | 30 | 6 |
| calibration_condition_mismatch | — | **312** |
| recommended | `…a40a-tp4-dp1-s128-t8192` | **none** |

**The field that does it is not the one the work order expected.** Per field:
`dp` **228**, `islands` 132, `tp` 66, `arrival_process` 42. `tp` explains only
66 of 312 holds and `dp` alone is the largest single group at 108 — replication
and placement are refused more often than parallelism is, so a measurement
programme aimed only at `tp` would close less than a quarter of them.

**Stating `arrival_process` moved 36 candidates and exactly the right ones**:
all 42 it touches are RNGD-touching, and no A40-only candidate moved, because
the A40 domain is open-loop and so is `plan`. The 12 single-card RNGD candidates
change from `outside_calibration_domain` to `calibration_condition_mismatch` —
**same hold, different measurement requested**: an envelope point above c=76 no
longer answers them, an open-loop refit does.

**V3's P1 gets its ending** (`experiments/p2_evidence/results/v3_addendum_s4.md`):
held at every TPOT SLO from 38 to 50 ms, identically, because a condition
mismatch is decided before any margin exists and no threshold can move it.
Rules (a)–(d) made 6 / 3 / 6 / 6 false passes on that grid; (e) makes none, and
none of the correct rejections either.

### E-A2 — the RNGD accuracy domain

`experiments/uncertainty/results/ea2_rngd_domain.md`, **superseded**. The domain
it produced is not on `main` and the file is removed: its sim side ran at served
71–189 against a real 15–107, which is the D32 mis-pairing. `main`'s nine-point
D32 domain for RNGD-CARD supersedes it. The document is kept with the retraction
at the top, because a retracted measurement is evidence about method.

### E-B1 – E-B3 — truth degradation

**Run 2026-09-14 on the reconciled architecture** (PRs #82, #86). All three
re-ran after D33, so the 2026-09-11 numbers do not carry over — the margin
policy, the `refuse` default and the partial-coverage rule all changed, and so
did the truth itself (D33 dropped the E-A2 domain as D32-mispaired; truth is now
the nine-point D32 RNGD-CARD domain, 50 of 324 candidates feasible).

**E-B1** — `experiments/uncertainty/results/eb1_regret_vs_budget.md`. Over the
37 of 231 degraded sets that actually move the plan, `ours` reaches zero regret
in **0.041 h against `random`'s 0.452 and `round_robin`'s 1.097**, matching the
oracle at every budget. The write-up says plainly why matching an oracle is less
impressive than it sounds: the pool is 11 items and exactly one of them, the
cheapest, carries essentially all the decision regret, so any rule that ranks it
first ties the oracle.

**E-B2** — `eb2_flip_detection.md`. Recall **0.990**, precision **0.843**,
identical at m = 3, 5 and 9 — the grid density does not matter, because a flip is
decided by whether the interval *contains* the crossing. The error is not where
the work order looks for it: the **approximate** rules are perfect over 224
cases, while every false positive and the single false negative come from an
**exact** rule, all of them on `sim_error`. (Re-run 2026-09-15 on the D70 margin;
the earlier 0.983 / 0.773 are superseded and kept only for comparison in
`eb2_f2.md`.) STEP C3 then classified those false positives: **none** is α, β or
γ — all are one modelling mismatch in the `sim_error` item, which **D72** records,
and β = 0 means no closed-form rule misplaced a crossing. The same detector on the
F2 corpus keeps precision but loses recall, 0.271, for a reason C5 takes up:
`eb2_f2.md`.

**E-B3** — `eb3_closed_form_vs_resim.md`. Closed form **0.287 s** against
**2,090.8 s** of resimulation over 1,104 runs, a **7,285× speed-up**. The
ordering survives — same item first, three inert inputs at exactly zero — and the
magnitude does not: **30.5 % low** on the one item where there was anything to
get wrong. Its Spearman of 1.000 is over four items three of which are tied at
zero and is close to vacuous; the `--top 1` run's 1.000 is tautological and is
committed as evidence of nothing.

**What they do not establish.** The E-A1 truth cache was built with
`enable_pd=False`, so there are no P/D candidates and `link_bw` flips nothing —
one input kind carries the whole decision. `WORK_ORDER_uq_stage_b_plus.md` exists
to re-run all three on a fixture where two or more kinds are active. That is F2,
and it is below.

### E-B1 – E-B3 on F2 — P/D on, TTFT 8,000 ms, D40's mirrors excluded

**Run 2026-09-15/16** (PRs #88–#91). F2 is defined in **D74**: the same cluster,
trace and seed as E-A1 with `enable_pd=True` and the TTFT SLO at 8,000 ms, which
is `d23_revalidation.md`'s middle regime — the first budget at which a KV
transfer can change a P/D candidate's verdict. Corpus 223 candidates (61 P/D)
after excluding 282 D40 mirrors and 23 D71 crashes. Truth winner
`pd(cuda-a40-node_a40a-tp2-dp1 P + cuda-a40-node_a40b-tp4-dp1 D)-s256-t8192`.

**§2.3's gate is met, barely.** Two kinds carry regret rather than one —
`sim_error:domain` (52 of 186 sets) and the two RNGD profiles (7 each). Three of
eleven inputs are ever active, and one of them carries 52 of the 66 activations.

**E-B1 on F2** — `eb1_f2.md`. `ours` wins the metric the work order names, mean
budget to zero regret **1.952 h against 1.988–2.136** for the baselines, with
`oracle` strictly better at 1.857 — so the fixture is not degenerate in §2.3's
sense. But at every *intermediate* budget `ours` is worse than `random`, and the
cause is not noise: on **78 of 186 sets (42 %) nothing is feasible while
degraded**, `analyze` short-circuits and returns ΔR = None for every input, and
`_rank` falls through to alphabetical order — which puts the six inert
`link_bw:fabric-*` items first. See §5.

**E-B2 on F2** — `eb2_f2.md`. Precision **0.706**, recall **0.271** against
E-A1's 0.990. All 15 false positives are `sim_error` and all are the `delta`
category (D72); α, β and γ are **zero on both corpora**, so no closed-form rule
misplaced a crossing. The recall collapse is the same defect E-B1 hit, measured
from the other side: **82 of the 97 false negatives** sit in the 87 sets with no
feasible plan, and in all 82 restoring that one input brings a real winner back.
`link_bw` produces 336 true negatives and not one predicted flip.

**E-B3 on F2** — `eb3_f2.md`. Both A40 profiles: closed form **33,041.1** against
a resimulated **21,615.4**, **+52.86 %** — the closed form *over*estimates here,
where on E-A1 it underestimated by 30.5 %. **D40's share of that is 0 pp**,
measured rather than argued: the same item resimulated with the mirrors present
(460 candidates, 676 runs) gives a bit-identical ΔR, while the ×1.0 identity
control fails on 69 candidates at 7.869e-02 — so the defect reproduces and simply
does not reach a regret integral over the recommendation. Spearman is **not
computed**: two active items, below §2.5's floor of three. Closed form 0.71 s
against 25,962 s of resimulation, **36,566×**.

---

## 5. Limits

Six, in the order they are likely to bite. The first two are the domain-scoping
work order's and are the newest.

**The condition refusal is whole-domain.** A single disagreeing field withdraws
the TTFT *and* the TPOT error together. For `arrival_process` that is demonstrably
blunter than the evidence: D19's finding is that a burst and a spread arrival
process differ and *"the difference lands entirely in TTFT"*, so a closed-loop
domain's TPOT error may well survive into an open-loop deployment. Refusing it
is conservative, not correct, and on E-A1 it is what holds all 42 RNGD-touching
candidates (D113). Per-metric condition scoping would express it; it is not
built.

**A run that judges nothing still suggests relaxing the SLO.** When
`condition_mismatch: refuse` holds every candidate there is no verdict to
explain, but the infeasible-run reporting path still emits "relax the TPOT SLO
from 50 ms to at least 116 ms" alongside the `measure at: {...}` lines that are
the real answer. The measurement requests are right and complete; the SLO advice
is about the wrong thing. Cosmetic, and listed so a reader does not act on it.

Then the four that predate S1:

**The closed form is first-order for two of five kinds.** `PROFILE` and `POWER`
are scalings, not re-simulations. `PROFILE`'s energy term carries an extra
offset on top of that: the measured ×1.3878 against a ×1.38876 multiplier is the
idle/standby, DRAM and link terms, which do not scale with operator time (D34).
Items scored by an approximate rule are marked `approximation=True` and printed
as "(approx)" wherever they are quoted. `--resimulate-top N` — the escape hatch
§2.7 leaves for checking one by really simulating both ends of its range — **is
implemented** (`planner/uncertainty/resimulate.py`) and E-B3 is what it measures.

**The order survives; the magnitude is not usable and is not even reliably
signed.** E-B3 puts the same items first on both fixtures and correctly calls the
inert ones inert. On magnitude it was **30.5 % low on E-A1** and is **52.9 % high
on F2**. The F2 figure is clean — D40's contribution to it was measured at
**0 pp** by resimulating the same item with the mirrors present and getting a
bit-identical ΔR — so the two numbers are a difference between fixtures, not one
correcting the other. **Do not quote ΔR as an amount of energy saved.** Use it to
order what to measure next, which is what it is for.

**The plan is silent exactly where measuring is worth most.** When the degraded
state has no feasible candidate, `analyze` short-circuits — *"no recommendation
to flip"* — and returns ΔR = None for every input, so the ranking degenerates to
alphabetical. On F2 that is **42 % of degradation sets**, and it is not a
misconfiguration: the verdicts are real `slo_violated`, not `unmeasured`. Two
experiments measure the cost from opposite sides — it is why `ours` loses to
`random` at intermediate budgets (`eb1_f2.md`), and it accounts for **82 of
E-B2's 97 false negatives**, in every one of which restoring the single input
brings a real winner back (`eb2_f2.md`). *"Which measurement would make something
feasible again?"* is not a question the closed-form sensitivity asks, and it is
the question an operator with nothing feasible actually has. Stage D.

**The grid is uniformly weighted.** `ΔR_i` is the unweighted mean over grid
points, so it assumes every point in an input's range is equally likely. Nothing
in the registry carries a distribution, and inventing one would be exactly the
unsourced number rule A1 forbids. The weighting is recorded in provenance as
`uniform` so a later reader knows it was a choice.

**The domain is one-dimensional.** Error is a function of served concurrency
only. Prompt-length mix enters as a *scope* (a domain is matched on token mix and
never on arrival rate) but not as an axis, so a workload whose shape matches and
whose length distribution does not is treated as covered. Two-dimensional domains
are Stage D.

**The penalty dominates the ranking.** `--slo-penalty` decides how much an
infeasible or unmeasured recommendation costs, and therefore how a `sim_error`
item ranks against a `profile` one. The default is a defensible convention, not a
measurement. On F2 it is at least not the knob that decides the *order*: at the
default and at a tenth of it the ΔR values scale by exactly 10 and the ranking is
identical in all 186 sets, Spearman 1.0, zero inversions.

**One kind has never been shown to matter.** `link_bw` is inert on both fixtures,
and on F2 for a reason stronger than E-A1's: it is priced into 54 candidates
including the winner, and the decision is insensitive to it at *both* ends of its
range. So the registry's evidence covers `profile` and `sim_error`; `link_lat`
and `power` have no result at all. D74 records what a fixture would need for a
link to decide something — a tighter TTFT or a larger KV working set — and notes
that D71 removes precisely the `tp1-dp1` P/D family where a transfer weighs most.

**Criterion (2) does not cover `sim_error`.** E-B2 scores flip detection two
ways, and the second — is the flip anywhere in the range, not only at truth — is
**not defined** for the accuracy domain: the sweep spends its range as a flat
margin while the harness restores it as a policy swap, which are different
functions (D72). Since every false positive on both corpora is `sim_error`, the
possible-transition precision of 1.000 describes the link and profile rules and
nothing else. Quote it with that sentence attached or not at all.

---

## 6. What is not built

* ~~**STEP B4**~~ and ~~**`--resimulate-top N`**~~ — **both done**, PRs #82 and
  #86; see §4. The `feat/uq-b4-wip` branch this entry warned against was ported
  rather than resumed: its two B4 commits were cherry-picked onto the reconciled
  `main` and the eighteen below them, the pre-D33 A1–B3, were dropped as
  superseded. Two of its changes were deliberately not carried over because D33
  had reversed them — `--accuracy-domain` mutually exclusive with the manual
  margins (D33 §3 keeps those as floors), and a pass resting on one unmargined
  metric rejected as `UNMEASURED` (D33 §5 makes partial coverage a caveat, and the
  old rule empties every search on the committed TPOT-only domains).
* **A fixture where more than one input kind is active.** E-B1–E-B3 all ran on a
  truth cache built with `enable_pd=False`, so `link_bw` flips nothing and one
  input carries the whole decision. That makes `ours = oracle` a fact about the
  fixture rather than about the ranking rule.
  `WORK_ORDER_uq_stage_b_plus.md` is the follow-up: a P/D fixture with a 4 s TTFT
  SLO, stratified degradation sets, and E-B3 re-run with the D40 mirror pairs
  excluded so the magnitude error can be separated from the cache defect.
* **Stage C** (closed-loop replanning against a live A40) and **Stage D**
  (conformal surrogate bounds, multi-dimensional domains) — separate work orders.

---

## 7. Where the pieces are

| what | where |
| --- | --- |
| registry, grades, costs | `planner/uncertainty/registry.py`, `grades.py`, `profiles/uncertainty/*.yaml` |
| the domain curve | `planner/predictor/calibration.py` (`AccuracyDomain`) |
| served concurrency | `planner/predictor/accuracy_domain.py` |
| per-candidate margin | `planner/optimizer/margin.py` |
| the three-way verdict | `planner/optimizer/exhaustive.py` (`judge`) |
| perturbation | `planner/uncertainty/perturb.py` |
| flips and `ΔR` | `planner/uncertainty/sensitivity.py` |
| the measurement queue | `planner/uncertainty/measurement_plan.py` |
| CLI | `planner/__main__.py` |
| application conditions | `planner/predictor/calibration.py` (`DomainParallelism`, `DomainPlacement`, `CandidateConditions`, `check_conditions`), `profiles/calibration/index.yaml` |
| link measurements by traffic | `planner/inventory.py` (`LinkMeasurement`), `planner/topology.py` (`link_bandwidth_gbps`) |
| decisions | `docs/deviations.md` D19, D22, D29, D32, **D33**, **D34**, **D35**, and the domain-scoping block **D110** (conditions) **D111** (default ranges) **D112** (link traffic key) **D113** (arrival process, and what the refusal costs) |
| what the refusal costs, measured | `experiments/uncertainty/results/ea1_s4_condition_refuse.md`, `experiments/p2_evidence/results/v3_addendum_s4.md` |
