# Paper outline — a planner that knows what it does not know

*`WORK_ORDER_rps_aware.md` rev 2 STEP 6.5. Written 2026-09-10 at `c78ff16`.
**This file is the input to the next conversation.** Every section names the
committed artifact it would cite; nothing here may be written up from memory.*

Read `docs/CLAIMS.md` first and treat it as binding. It says, per claim, whether
the label is **measured**, **sim-on-measured** or **analytical**, and §3 lists
what has been retracted. A sentence in the paper that outruns its row in CLAIMS is
a defect, not a simplification.

---

## The one-sentence claim

> Heterogeneous serving decisions depend on the arrival rate, and a planner cannot
> make them safely from a scalar performance profile — because the simulator's own
> error is a function of the operating point, so a scalar profile recommends
> configurations that are not merely optimistic but **infeasible**.

The evidence that this is a real failure and not a hypothetical is that **we made
it ourselves and had to retract it** (D22). The paper's spine is that retraction
and the machinery built so the pipeline would catch it unaided.

---

## 1. Problem — the error is a function of the operating point

**Artifacts:** `docs/deviations.md` D22 · `experiments/results/rngd_concurrency_envelope.md`
· `docs/CLAIMS.md` §3

The concrete case. A sweep recommended two RNGD cards at a predicted p99 TPOT of
**48.41 ms** against a 50 ms SLO. On silicon the simulator is **18 % optimistic**
at the load that plan implies, so the real figure is ~57 ms. The plan was
**infeasible**, and nothing in the pipeline could see it, because the 18 % was a
fact about the simulator that the simulator did not carry.

Two further facts make this structural rather than a one-off:

- the error is not one number — **+11.6 % at served concurrency 3.9, −18 % at 76**
  on the same device (`profiles/calibration/rngd_card_edf.yaml`), so it changes
  sign, not just size;
- the original curve that motivated the whole comparison was itself
  pool-limited — "c32" was a 24-request pool running at **effective concurrency
  21.2** — so an exponent fitted across it read a ×1.74 interval as a doubling.

**The framing to keep:** the problem is not accuracy, it is *shape*. A profile
stating one throughput and one TPOT is a claim about one operating point
pretending to be a claim about a device.

## 2. Method — four artifacts, each defined by what it refuses

**Artifacts:** `docs/rps_aware_planning_design.md` · `planner/perf_envelope.py` ·
`planner/predictor/calibration.py` · `planner/util/operating_point.py` ·
`planner/sweep.py` · `docs/rps_sweep_example.md`

| piece | what it is | what it refuses |
| --- | --- | --- |
| performance envelope | measured curve, `concurrency_metric: served` mandatory | reading past `validity`; `Saturated` separates **genuine** from **unmeasured** |
| accuracy domain | the predictor's error, indexed by operating point | a margin where nothing was measured — 0 and a note, never a borrowed number |
| operating point | served concurrency per hardware, from the run's own CSV | for P/D the phase split gives prefill and decode their own loads |
| RPS axis | one plan per rate; switchover and crossovers | a crossover presented as measured — it is interpolated and labelled `estimated` |

**The paper's methodological contribution is the refusals**, not the curves. Each
one is a place where the obvious engineering choice — extrapolate, default to 1.0,
reuse the nearby value — is the choice that produced the retraction. Three worked
examples, all measured:

1. `factor_for()` returns `None` outside the measured domain rather than 1.0.
   **1.0 is the uncorrected case**, so the "harmless" default would reinstate a
   13–24 % error precisely where nobody had checked (D28, `slab3d_calibration.md`).
2. `widen_error_bars` grows the margin with distance and **never caps**, so a
   candidate far outside is rejected by its own uncertainty.
3. When the low end of the RNGD domain was finally measured, the extrapolation it
   replaced had the **wrong sign** — the simulator is *pessimistic* below
   concurrency ~7 (`outputs/lowload_sim_error/`).

**Asymmetric TP per phase (D28)** belongs here as a prerequisite, not a headline:
`tp_d = 2·tp_p` had been recorded as a simulator limitation and was not one — it
was `_compute_network_dims` plus a shared `local_dim`. Its correction is a
**24-point measured table, not a law**: the factor varies with the split, so an
unmeasured split is refused rather than predicted, even though all three measured
values equal `tp/2` and hop counting derives that.

## 3. Generalisation — planning on hardware you do not own

**Artifacts:** `docs/tier0_calibration.md` · `WORK_ORDER_tiered_profiles.md` ·
`docs/CLAIMS.md` §1.1

Tier 0/1 synthetic profiles, and the honest statement of what they buy: Kendall
**τ 0.914 / 0.902** against a measured ranking, with exact top-1 agreement **not**
achieved but the disagreement costing 0.4 % / 11.3 % of the objective. A Tier 0
plan is a **shortlist, not an oracle**, and the label travels with it
(`PlannerOutput.profile_tier`).

This section is what makes the method more than a report about two devices.

## 4. Evaluation

**Artifacts:** `experiments/results/e5_self_rejection.md` · `e6_rps_sweep.md` ·
`rngd_lowload_envelope.md` · `docs/surrogate_topk_regret.md` ·
`docs/sim_cost_profile.md` · `experiments/figures/e6_rps_sweep.png`

**E5 — the pipeline reproduces its own retraction.** With no hand-set margin the
planner rejects the committed winner: operating point **74.75**, domain
**−17.69 %**, robust TPOT **56.97 ms > 50**, `SLO_VIOLATED`, falling back to
`agg[cuda:tp4]` at **2.595 tok/J**. Held in CI by replaying 199 committed
simulation records through a mock predictor — no simulator needed
(`tests/test_e5_self_rejection.py`).

**E6 — there is a crossover, and almost none of it is measured.** RNGD →
cross-vendor P/D → A40 between 1 and 10 rps, at both TTFT SLOs. **One of sixteen
switchover cells is labelled `measured`.** Both halves of that sentence are the
result; a paper that prints the first without the second is doing what D22 did.

**E7 — the dependency is on having a domain, not on any one point.** Leave-one-out
across the accuracy domain never changes the winner (0 of 24). Removing the domain
entirely flips 3.3 rps to an RNGD-only plan that looks **66 % more efficient**.

**The low-load envelope** is the measured backbone: tok/J spans **9.4×** from
concurrency 1 to 15.6 while power spans **1.085×**, so energy per token is set
almost entirely by throughput. It also **refutes the design's own hypothesis** —
"tokens/J is unimodal in concurrency" is not observed anywhere in [1, 107.2], and
the bound holds even at the highest wattage ever seen on the card.

**Negative results to report as results**, not omit:
- top-K surrogate pruning is unusable on P/D corpora (**D30**), and the obvious
  repair breaks a third fixture — the same one-fixture error being corrected;
- utilisation does not explain this card's power (**D31**), so the design's
  `piecewise_linear_in_util` was re-keyed on served concurrency.

**Retraction discipline as a contribution.** Four retractions (D18, D19, D22,
D30), every one caught by internal discipline — provenance labels,
sustained-vs-peak hygiene, applying a measured model error as a feasibility
margin, and re-measuring an accuracy claim on fixtures it had not been measured
on. Retracted text is marked in place and never overwritten. This is worth a
subsection; most systems papers cannot show the negative half of their own record.

## 5. Limitations — state these before a reviewer does

**Artifacts:** `docs/CLAIMS.md` §2 · `docs/HANDOVER.md` §2

1. ~~**The A40 accuracy domain has one point**~~ — **resolved 2026-09-11**
   (`experiments/results/a40_lowload_envelope.md`). It has three, measured on the
   A40 node, and nine of sixteen cells now read `measured`. Two residual limits
   replace it, and both are structural rather than pending: five cells need a
   **per-PE RNGD** accuracy domain that does not exist, and two rest on an A40
   **prefill leg at served concurrency 0.499** — an almost-idle server, which is
   not an operating point a bench can hold.
   The correction that came with it belongs in this section: the simulator is
   **~18 % optimistic on TTFT at low load**, against the 1.97 % the single
   saturated point declared. It changes no E6 outcome because TPOT binds first
   there, but any tight-TTFT claim inherits it.
2. ~~**The A40 side of E6b is simulation only.**~~ — **resolved**, but replaced by
   a narrower caveat: the A40 curve is **open-loop** and the RNGD curve is
   **closed-loop**, so a sentence putting both on one axis must say which
   protocol each side was measured under. `closed_loop` is carried in every row
   of the artifact for that reason.
3. **One workload.** The envelope is conditional on a token distribution
   (`measured_on_workload`), and everything here uses one sharegpt trace at
   input p50 732 / output mean 652.5.
4. **ATOM is out** (D20): host I/O exceeds the kernels and the device tracer's
   schema is undocumented, so no bundle reaches contract fidelity.
5. **The per-PE RNGD profile has no accuracy domain**, so its rows carry margin
   0.00 % and validity `unknown` — including a 4.956 tok/J that **is** the
   retracted headline. It appears in E6's tables and is not rehabilitated by
   appearing there.
6. **68–114 candidates per E6 point fail in the simulator's KV allocator** after
   passing the generator's memory bound, so every "best" is the best of what
   evaluated. The count is printed for that reason.
7. **`slab3d`'s calibration is fitted at two dims and applied at three**, taking
   dim 1 to be the same axis in both. Derivable, not measured. TTFT and throughput
   under `slab3d` are unfitted, so an asymmetric plan's absolute TTFT is not
   quotable.

---

## What to decide before writing

1. **Venue and framing.** The methodology (refusal-shaped planning, accuracy
   domains) is the durable contribution; the crossover is one instance of it. A
   systems venue may want the reverse emphasis — decide once, then make §1 and §4
   agree.
2. **Whether to wait for the A40 point.** With it, the crossover claim moves from
   one measured cell to several and §5.1 and §5.2 both shrink. Without it the
   paper is a methods paper with an illustrated example, which is defensible but a
   different paper.
3. **How much of the retraction record to foreground.** §4's last subsection could
   be a paragraph or a section; it is unusual material and reviewers respond to it
   in both directions.

**Do not** write any number into a draft without opening its row in
`docs/CLAIMS.md` first. That file exists because this project has already
published four numbers it had to take back.
