# The uncertainty-aware planner — design, CLI, results, limits

`WORK_ORDER_uncertainty_planner.md` STEP B5. This is the reference for
`--accuracy-domain`, `--measurement-plan`, `fit-accuracy-domain` and
`measure-apply`: what they compute, what they are allowed to claim, and where
the numbers behind each claim live.

**Read `docs/deviations.md` D33 first if you are touching the accuracy domain
itself.** Two implementations of "the simulator's error as a function of the
operating point" were built in parallel; one was kept, and D33 records which and
why. This document describes what landed.

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
at a predicted 48.41 ms, and 48.41 × 1.18 = 57.1 ms. The configuration was
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
by |slope| × distance and is what the three committed domains still carry
explicitly, so every E5/E6 number predating D33 is unchanged. Flipping those to
`refuse` is a science change and is deliberately not done here (D33 §2).

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
   missed;
2. it **passed**, but on a metric whose margin is unknown → `UNMEASURED`. A
   violation is real whatever the coverage, because a margin only inflates; a
   *pass* that rests on an unmeasured metric is a gap, not a verdict;
3. otherwise the feasibility report decides.

Coverage can be partial. A domain fitted by comparing a burst simulation against
a closed-loop bench has TPOT points and no TTFT ones — queued requests inflate
sim TTFT by two orders of magnitude there, so there is nothing honest to compare
(D19). Such a decision margins TPOT and reports TTFT in `unmeasured_metrics`.

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
| `LINK_BW`, `LINK_LAT` | TTFT of P/D candidates | the KV transfer is re-priced with `kv_transfer.transfer_ms`, the same helper the planner itself uses | **exact** |

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

`LINK_BW` is exact for a reason worth knowing: the envelope cache stores the raw
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
```

Notes that are easy to get wrong:

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

### E-A2 — the RNGD accuracy domain

`experiments/uncertainty/results/ea2_rngd_domain.md`, **superseded**. The domain
it produced is not on `main` and the file is removed: its sim side ran at served
71–189 against a real 15–107, which is the D32 mis-pairing. `main`'s nine-point
D32 domain for RNGD-CARD supersedes it. The document is kept with the retraction
at the top, because a retracted measurement is evidence about method.

### E-B1 – E-B3 — truth degradation

**Not run.** STEP B4 is outstanding; see §6.

---

## 5. Limits

Four, in the order they are likely to bite.

**The closed form is first-order for two of five kinds.** `PROFILE` and `POWER`
are scalings, not re-simulations. `PROFILE`'s energy term carries an extra
offset on top of that: the measured ×1.3878 against a ×1.38876 multiplier is the
idle/standby, DRAM and link terms, which do not scale with operator time (D34).
Items scored by an approximate rule are marked `approximation=True` and printed
as "(approx)" wherever they are quoted. `--resimulate-top N` (the escape hatch
§2.7 leaves for checking one by really simulating both ends of its range) is
**not implemented on `main`**.

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
measurement.

---

## 6. What is not built

* **STEP B4** — the truth-degradation experiments E-B1 – E-B3, which are the
  quantitative basis for the patent's §6 effect claim. Prior work exists on the
  `feat/uq-b4-wip` branch and **does not run against `main`**: it was built on the
  deleted `predictor/accuracy_domain.py`, on the dropped E-A2 domain yaml (one end
  of its scalar↔domain degradation), and on a cache whose key scheme has since
  changed — `main`'s 162-entry E-A1 cache and that branch's share zero entry
  names. Treat it as a design reference, not as code to resume.
* **`--resimulate-top N`** — §2.7's escape hatch for checking an approximate rule
  by really simulating both ends of its range.
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
| decisions | `docs/deviations.md` D19, D22, D29, D32, **D33**, **D34**, **D35** |
