# S2 — what the grade default ranges change, and what they do not

**What this is.** `WORK_ORDER_domain_scoping.md` STEP S2, the measured half.
Before S2, an uncertain input whose own (kind, grade) sourced no width had no
range, therefore no regret, therefore no place in the measurement plan's
ranking: it was listed as "cannot be decided before measuring" and the operator
was never told what it would be worth. S2 gives each *grade* a default range for
that case. This is what happened when the change was run against the fixture V3
measured.

The generated tables are in `s2_default_ranges_table.md`; the raw record is
`outputs/uncertainty/s2/s2_default_ranges.json`. Both arms come from **one** run
of the E-A1 corpus (300 requests, seed 42, `pd-rngd-gpu-card.yaml`,
`examples/service_specs/llama31-8b.yaml`, per-point accuracy-domain margin,
warm cache) so nothing re-simulates and the recommendation — V3's P1,
`cuda-a40-node_a40a-tp4-dp1-s128-t8192` — is the same on both sides by
construction. Only the registry's ranges differ.

## The answer

| | without defaults (pre-S2) | with defaults |
| --- | --- | --- |
| ranked | 2 | 2 |
| inert (swept, moves nothing) | 0 | **4** |
| undecidable | **4** | 0 |
| `link_bw:pcie-a40a-02` | undecidable | **inert** |

**The hypothesis was rank 1. The measured answer is `inert`.** The work order
predicted that the V3 link would take first place once it had a range; it takes
no place at all, and the reason has nothing to do with the range it was given.

## Why the link is still not ranked

A link bandwidth reaches a predicted metric through exactly one path:
`apply_pd_transfer_cost`'s prefill→decode KV transfer, which
`perturb._reprice_transfer` re-prices exactly. The E-A1 corpus was generated
with `enable_pd=False` and holds no P/D candidate, so no candidate's transfer
crosses any link, and every `link_bw` item sweeps its whole range with
`affected=0`. E-B1 recorded the same inertness on the same corpus and called it
structural; S2 does not change it and was never going to.

Put plainly: **the registry prices a link as a KV transfer, and the error V3
measured came from the same link carrying a TP all-reduce inside one island.**
Those are different quantities on the same wire. That gap is STEP S3's subject —
LINK_BW is to be redefined as the *effective collective bandwidth of the
deployment*, keyed by `(link_id, collective, msg_size_class, device_binding)` —
and until it is, no default range can make this link rank, because nothing in
the prediction depends on it.

## What the change did do

1. **The undecidable list is now a real category rather than a catch-all.** Four
   inputs left it. What they left it *for* is the new `inert` bucket: swept over
   a known range, they move the decision by exactly zero. "We cannot say" and
   "we can say, and it is worth nothing" were indistinguishable before S2, and
   they call for opposite responses — the first is a gap in the evidence, the
   second is a closed question.
2. **Every default-ranged row says so.** `[8, 64] gbps · default` in the
   registry table, `[default range]` on a ranked row, and the same flag in the
   YAML (`range_source`). A reader can see at a glance which numbers rest on a
   measurement of that input and which on a policy about its grade.
3. **A latency does not inherit a bandwidth's direction.** The grade-level
   `vendor_spec` default is `[1/8, 1] ×` nominal — a spec bandwidth is an upper
   bound. `link_lat` overrides it with `[1, 8] ×`, so the plan says a 2,000 ns
   spec link may cost up to 16,000 ns and never that it beats its datasheet.
4. **Nothing sourced moved.** Both `sim_error` items keep the ranges the
   calibration store gives them, the same dR, the same dR/h, the same order.

## Regression: E-B1, E-B2, E-B3

Re-run on the same cache after the change. **The numbers are unchanged, and
that is structural rather than lucky**: the truth-degradation experiments hand
`analyze` the interval they degraded themselves — `[min(truth, degraded),
max(truth, degraded)]` — so no E-B range is ever read from `grades.yaml` and the
defaults layer cannot reach one. `tests/test_default_range.py::
test_an_experiment_supplied_range_is_sourced` pins the reason rather than the
result.

| experiment | re-run | result |
| --- | --- | --- |
| E-B1 regret vs budget | `--exhaustive --k 1 2 3`, 231 degraded sets | **byte-identical** to `outputs/uncertainty/eb1/eb1_regret_vs_budget.json` once `provenance` is excluded |
| E-B2 flip detection | `--truth-sweep`, 616 cases per grid, 1,848 total | **byte-identical**; precision 0.843, recall 0.990, FPR 0.037 at m = 3, 5 and 9, unchanged |
| E-B3 closed form vs resimulation | `--top 4 --workers 32` | **not completed**: the re-run was cut off at 90 minutes of wall clock (`timeout` exit 124) with ~306 simulations done. See below |

The re-run LOGS are under `outputs/uncertainty/s2/eb/`, with the headline
numbers in them. The two JSONs they produced are **not** committed: they are
identical to `outputs/uncertainty/eb1/eb1_regret_vs_budget.json` and
`outputs/uncertainty/eb2/eb2_flip_detection.json` bar the provenance block, and
a 2.6 MB copy of an unchanged artifact is not evidence. The committed ones are
untouched, as rule 5 of the work order requires.

**E-B3 was attempted and is not reported.** It is the one E-B experiment that
re-simulates, and the committed result records 34.8 minutes for that side; this
re-run was still going at 90 minutes on the A40 node, shared with the rest of
this session's work, and was cut off. It is not chased further, because the
property under test here does not depend on it: E-B3 takes its pool from E-B1's
construction, so its intervals are the experiment's own degraded endpoints, the
same ones E-B1 and E-B2 use and the same ones `grades.yaml` never sees. What
would be evidence against that is E-B1 or E-B2 moving, and neither did. A future
re-run should give it an idle machine and `experiments/scripts/livelock_watch.sh`.

**A note on the first attempt, because it is the kind of mistake that survives
into a table.** E-B1 was first re-run without `--exhaustive`, which samples ten
degradation sets per k instead of enumerating all 231 — a different experiment
whose numbers would have looked like a regression. It also wrote to the
committed `outputs/uncertainty/eb1/eb1_regret_vs_budget.json`, whose `--out-json`
default is a fixed path rather than the `--out-dir` given on the command line.
The file was restored from git and the run repeated with the committed
invocation.

## What this does not settle

* The default magnitudes are settings, not findings. Each is sourced to the
  widest relevant observation the repository holds (V3's 8.8/64.0 for
  `vendor_spec`, E2's MAPEs for the profile grades, the two repeat measurements
  for `sim_error/measured`), and every one of those is `n = 1` or `n = 2`.
  Tightening them needs measurements, not argument.
* `user_defined` still has no default and still reports as undecidable. That is
  deliberate: a deliberate what-if is the user's number to bound.
* Whether the V3 link is worth measuring **on a corpus that contains P/D
  candidates** is not answered here, because no such corpus is cached. S3 is
  what makes the question askable on this one.
