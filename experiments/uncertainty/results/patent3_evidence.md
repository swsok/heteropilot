# Patent 3 (measurement planning) — evidence map

`WORK_ORDER_uq_stage_b_plus.md` §5, filled at the end of STEP C5. One row per
item the filing's §6 is expected to make, the artifact that supports it, and the
number as that artifact reports it.

> **Citation rule (work order §5).** Figures in §6 are quoted **only** from the
> files named here. If a number is not in this table it has not been checked
> against a result document, and several plausible-sounding ones are wrong — see
> *Retracted and superseded* at the bottom.

Two fixtures appear throughout and they are not interchangeable:

| | **E-A1** | **F2** |
| --- | --- | --- |
| candidates | 324 aggregated, `enable_pd=False` | 223 (61 P/D), `enable_pd=True` |
| TTFT SLO | 25,000 ms | 8,000 ms (`d23_revalidation.md` middle regime) |
| D40 mirrors | present | 282 excluded at build |
| active input kinds | **1** (`sim_error`) | **2** (`sim_error`, `profile`) |
| defined in | `eb1_regret_vs_budget.md` | **D74** |

---

## 1. Background — some inputs carry large error and change no decision

| | |
| --- | --- |
| **Artifact** | `eb2_flip_detection.md`, `eb2_f2.md`, `eb1_f2.md` |
| **Number** | `link_bw`: **336 of 336 cases true-negative on both fixtures**, not one predicted flip. On F2 the same link is priced into **54 candidates including the winner** and the decision is insensitive to it at both ends of its range. |
| **Also** | Per-kind activity on F2: 3 of 11 inputs are ever active; `sim_error` carries 52 of 66 activations, the two RNGD profiles 7 each, the two A40 profiles and all six links zero (`eb1_f2.md`). |

**Strength.** This is the premise the invention rests on and it is the best-
supported row in the table: three independent experiments agree, and on F2 the
statement is the strong form — the link is *charged* to the recommendation and
still decides nothing.

**Does not support** "measurement planning is unnecessary for links in general".
One cluster, 300 requests, a measured 13 GB/s fabric. D74 records what a fixture
where a link decides something would need.

## 2. Construction — perturbation as closed-form post-processing of cached predictions

| | |
| --- | --- |
| **Artifact** | `eb3_closed_form_vs_resim.md` (E-A1), `eb3_f2.md` (F2) |
| **Number** | E-A1: **0.287 s** against **2,090.8 s** over 1,104 simulator runs — **7,285×**. F2: **0.71 s** against **25,962 s** over 1,528 runs — **36,566×**. |

**Does not support** a general speed-up figure. Both ratios are "one sweep versus
resimulating every endpoint of every item"; an operator who resimulates nothing
saves nothing, and one who resimulates the top item only saves proportionally
less.

## 3. Limit — the closed form preserves order; its magnitude does not

| | |
| --- | --- |
| **Artifact** | `eb3_f2.md`, `eb3_closed_form_vs_resim.md` |
| **Number** | Order: **preserved on both fixtures** — same item first, every inert input at exactly zero. Magnitude: **30.5 % low on E-A1**, **52.9 % high on F2**. |
| **D40 separated** | **0 pp.** The same F2 item resimulated with the mirrors present (460 candidates, 676 runs) returns a **bit-identical** ΔR of 21,615.372406 J, while the ×1.0 identity control fails on 69 candidates at 7.869e-02 — the defect reproduces and does not reach the regret integral. |

**This is the row the filing must not overstate.** The honest sentence is: *the
closed form decides **what** to measure, and its ΔR must not be quoted as an
amount of energy saved.* It was wrong by a third in one direction on one fixture
and by half in the other direction on the other.

**Rank correlation is deliberately absent.** §2.5 forbids a Spearman below three
active items; F2 has two, so it is `None` with the reason recorded. Two earlier
1.000s (PR #86) are **evidence of nothing** — one over four items with three tied
at zero, one over a single refined item — and are retracted below.

## 4. Construction — the definition of ΔR_i

| | |
| --- | --- |
| **Artifact** | `docs/uncertainty_planner.md` §2.5; `planner/uncertainty/sensitivity.py`; `tests/test_sensitivity.py` |
| **Content** | Regret of an infeasible or unmeasured recommendation = best feasible objective − `slo_penalty` × worst overshoot. ΔR_i is the unweighted mean over the item's grid. |

**Does not support** a claim that the weighting is principled: it is uniform
because no registry entry carries a distribution, and inventing one would be the
unsourced number rule A1 forbids. Provenance records `uniform` as a choice.

## 5. Construction — ΔR/cost ordering, a budget, and `undecidable`

| | |
| --- | --- |
| **Artifact** | `planner/uncertainty/measurement_plan.py`; `profiles/uncertainty/costs.yaml`; `docs/uncertainty_planner.md` §2.6 |
| **Content** | Items ranked by regret removed per hour; an input with no sourced range is listed **undecidable** rather than scored zero; `--budget-hours` truncates. |

## 6. Effect — regret at a given budget, and budget to zero regret

| | |
| --- | --- |
| **Artifact** | `eb1_f2.md` (two active kinds), `eb1_regret_vs_budget.md` (single active item) |
| **Number, F2** | Mean budget to zero regret: `ours` **1.952 h** against `round_robin` 1.988, `widest` 1.989, `random` 2.136; `oracle` 1.857. 115 of 186 sets moved the recommendation. |
| **Number, E-A1** | `ours` **0.041 h** against `random` 0.452 and `round_robin` 1.097, matching the oracle at every budget — over the 37 of 231 sets that moved. |

**Both must be quoted together, and the E-A1 one needs its caveat.** Its pool has
**one** item carrying essentially all the regret and that item is also the
cheapest, so any rule ranking it first ties the oracle. It is a *single-active-
item case*, not a demonstration that the ranking rule beats baselines.

**F2 is the honest comparison and it is mixed.** `ours` wins the headline metric
and `oracle` is strictly better than it, so the fixture is not degenerate — but at
every intermediate budget `ours` is **worse than `random`**, for the reason in
row 8.

## 7. Effect — precision and recall of the flip prediction, both criteria

| | |
| --- | --- |
| **Artifact** | `eb2_f2.md`, `eb2_flip_detection.md` |
| **Realised transition** | E-A1 precision **0.843**, recall **0.990**. F2 precision **0.706**, recall **0.271**. Identical at m = 3, 5, 9 on both. |
| **Possible transition** | E-A1 precision **1.000**, recall **1.000** over 560 cases. F2 precision **1.000**, recall 0.146 over 560. |
| **False-positive cause** | **0 α, 0 β, 0 γ** on both corpora. All 19 (E-A1) and all 15 (F2) are `delta`, one modelling mismatch in the `sim_error` item (**D72**). |

**β = 0 is the load-bearing number here**: no closed-form rule ever placed a
crossing in the wrong place. Read undifferentiated, E-B2 appears to say the rules
cry wolf about one flip in six.

**The possible-transition figures exclude `sim_error` entirely** (D72), and
`sim_error` is the only kind with false positives on either corpus. So "precision
1.000" is a statement about the link and profile rules. **Quote it with that
sentence or not at all.**

**F2's recall of 0.271 is not a detector defect** — see row 8.

## 8. Limits to state explicitly

| limit | evidence |
| --- | --- |
| **Silent where measuring is worth most.** With nothing feasible while degraded, `analyze` returns ΔR = None for every input and the ranking falls through to alphabetical. **42 % of F2's sets.** Costs `ours` its intermediate-budget curve, and accounts for **82 of E-B2's 97 false negatives** — in every one of which restoring the one input brings a real winner back. | `eb1_f2.md`, `eb2_f2.md` |
| **ΔR is not a saving.** −30.5 % / +52.9 %, opposite signs on two fixtures. | `eb3_f2.md` |
| **One kind has never been shown to matter**; `link_lat` and `power` have no result at all. | `eb1_f2.md`, `eb2_f2.md` |
| **Uniform grid weighting**, recorded as a choice in provenance. | §2.5 |
| **One-dimensional domain** — error as a function of served concurrency only. | §2.2, D29 |
| **Penalty is a convention.** On F2 it does not decide the order: default and one tenth give Spearman 1.0, zero inversions across 186 sets. | `eb1_f2.md` |
| **Grid density does not matter**, m = 3/5/9 agree case for case across 5,544 cases — favourable to the description, and measured rather than assumed. | `eb2_f2.md` |

---

## Retracted and superseded — do not cite

| number | status |
| --- | --- |
| E-B2 precision **0.773**, recall **0.983** | Superseded. Pre-D70 margin; regenerated as 0.843 / 0.990. |
| E-B2 "**17** false positives" | Superseded; 19 on the corrected margin. |
| E-B3 Spearman **1.000** (either run) | **Retracted.** One is four items with three tied at zero; the other is a single refined item and tautological. §2.5 now refuses both. |
| "RNGD wins on energy by **1.67×**" (D22) | **Retracted** — `CLAUDE.md`, D22. |
| E-A2's four-point RNGD domain | Superseded by the nine-point D32 domain (D33). |
| TTFT SLO **4,000 ms** as F2's middle regime | Wrong source (`pd_slo_sweep.md`, superseded). The value is **8,000 ms** — D74. |

## Provenance of the underlying runs

Every figure above is a cache replay or a resimulation against the committed F2
envelope cache and the committed RNGD-CARD and A40 accuracy domains. **No figure
here is a hardware measurement**, and the domains were measured on the nodes that
own them, not on the machines that ran these experiments. E-B3's resimulation was
split across three CPU machines, two of which `scripts/whichnode.sh` does not
classify; the closed form returned identical values to the last digit on all of
them, which is the evidence the split changed nothing (`eb3_f2.md`).
