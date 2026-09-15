# E-B1 on F2 — the link is priced in, and `ours` loses where nothing is feasible

`WORK_ORDER_uq_stage_b_plus.md` STEP C2. E-B1 re-run on the F2 corpus that STEP C1
built: P/D on, TTFT SLO at 8 000 ms, D40's mirrors collapsed, judged by the
committed domains under `refuse`.

Generated tables: `eb1_f2_table.md`. Figure: `figures/eb1_f2_regret_vs_budget.png`.
Raw: `outputs/uncertainty/eb1_f2/eb1_f2_regret_vs_budget.json`.

```
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/eb1_regret_vs_budget.py \
    --fixture experiments/uncertainty/fixtures/f2.json \
    --cache-dir outputs/uncertainty/f2/cache \
    --work-dir outputs/uncertainty/eb1_f2/work \
    --exhaustive --stratified --workers 8 \
    --out-json outputs/uncertainty/eb1_f2/eb1_f2_regret_vs_budget.json
```

Corpus 223 candidates (61 of them P/D), truth winner
`pd(cuda-a40-node_a40a-tp2-dp1 P + cuda-a40-node_a40b-tp4-dp1 D)-s256-t8192` at
V = −79 550 J — the same winner and value STEP C1's independent evaluation found.
282 candidates excluded as D40 mirrors, 23 with no cache entry (D71). Degradation
sets are stratified by kind: 11 at k=1, 34 at k=2 (of 55), 141 at k=3 (of 165).

## The completion criterion: `n_active` = 2 kinds

| input | kind | sets with ΔR > 0 (of 186) | max ΔR (J) |
| --- | --- | ---: | ---: |
| `sim_error:domain` | sim_error | 52 | 42 616 |
| `profile:furiosa-rngd-card-node_rngd0` | profile | 7 | 4 800 |
| `profile:furiosa-rngd-card-node_rngd1` | profile | 7 | 4 800 |
| `profile:cuda-a40-node_a40a` | profile | 0 | — |
| `profile:cuda-a40-node_a40b` | profile | 0 | — |
| all six `link_bw:fabric-*` | link_bw | 0 | — |

Two kinds carry regret, so §2.3's gate is met and the superiority table below is
the one to read. It is worth saying what "met" is worth here: **three of eleven
inputs are ever active**, and one of them carries 52 of the 66 activations.

## What F2 actually changed, and what it did not

F2 was built because E-A1's corpus had no P/D candidate, which made all six
fabric links **structurally inert** — `perturb` reported `affected=0` for every
one of them, so no measurement strategy could be right or wrong about a link.
That part worked:

| link | candidates whose metrics move | winner among them |
| --- | ---: | --- |
| `fabric-a40a-a40b` | 54 | yes |
| `fabric-rngd0-a40a` | 7 | no |
| `fabric-rngd0-a40b` | 0 | — |
| `fabric-rngd0-rngd1` | 0 | — |
| `fabric-rngd1-a40a` | 0 | — |
| `fabric-rngd1-a40b` | 0 | — |

`fabric-a40a-a40b` is now priced into 54 candidates including the recommended
one. **And it still never changes the decision**: dropping it from 13 GB/s to the
placeholder 35 leaves the same winner standing, so ΔR is 0 — not None. That is a
different and stronger statement than E-A1's. On E-A1 the link could not matter;
on F2 it is charged, to the winner, and the decision is insensitive to it anyway.

The reason is visible in the corpus: the winner splits P and D across the two A40
nodes, and at 300 requests the KV transfer over that link is small beside a
TTFT budget of 8 000 ms. A fixture where `link_bw` carries decision regret needs
either a tighter TTFT or a larger KV working set, not merely P/D candidates.

**The 23 candidates D71 removed were the ones most likely to disagree.** They are
every `tp1-dp1` P/D split — the smallest decode engines, where a KV transfer is
largest relative to the engine's own work. They raised in the simulator rather
than producing metrics, so they are absent from this sweep, and the link result
above is a statement about the 223 that survived.

## Superiority: `ours` reaches zero fastest and is worst in between

From `eb1_f2_table.md`, default penalty, the 115 sets whose recommendation moved:

| strategy | R(0 h) | R(0.041 h) | R(2.1 h) | mean h to regret 0 |
| --- | ---: | ---: | ---: | ---: |
| `oracle` | 152 818 | 145 834 | 15 614 | 1.857 |
| `ours` | 152 818 | 168 201 | 157 349 | **1.952** |
| `round_robin` | 152 818 | 152 500 | 134 982 | 1.988 |
| `widest` | 152 818 | 156 974 | 164 583 | 1.989 |
| `random` | 152 818 | 156 781 | 103 504 | 2.136 |

`ours` wins the metric the work order names — mean budget to zero regret, 1.952 h
against 1.988–2.136 for the baselines — and `oracle` is strictly better than it at
1.857, so the fixture is not degenerate in §2.3's sense. But at every intermediate
budget `ours` is **worse than `random`**, by a wide margin at 2.1 h (157 349
against 103 504 J).

That is not noise and not a harness fault. It has one cause.

### Where it comes from: no incumbent, no ranking

`analyze` short-circuits when the degraded search finds nothing feasible —
`"no recommendation to flip: the search found nothing feasible"` — and returns
ΔR = None for **every** input. `_rank` then sorts the all-None list by
`(2, 0.0, 0.0, input_id)`, which is alphabetical order, and alphabetical order
puts the six `link_bw:fabric-*` items ahead of every `profile:*` item.

On F2 that state is not rare. Degrading either A40 profile by the Tier 0
multiplier SLO-rejects the whole corpus at this TTFT:

```
profile:cuda-a40-node_a40a    {'rejected': 223}
profile:cuda-a40-node_a40b    {'rejected': 223}
sim_error:domain              {'rejected': 189, 'feasible': 34}
link_bw:fabric-rngd0-a40a     {'rejected': 192, 'feasible': 31}
```

**78 of 186 degradation sets (42 %) have no feasible plan while degraded** — 2 at
k=1, 12 at k=2, 64 at k=3 — and 72 of those 78 contain at least one link. In every
one of them `ours` buys a link first, verified against the curves: the first
purchase on all 78 is one of the six links, seven sets per link.

`random` shuffles the same set, so it picks a link with probability equal to the
link share of that set — 102 of the 218 items across these sets, 47 %. `ours`
picks one with probability 1. That is the whole difference: on 42 % of the sweep
`ours` is not ranking at all, and its tie-break is alphabetical, which happens to
name the one kind this fixture has just been shown never to decide anything.

The per-set histogram says the same thing from the other side: `n_active` is 0 on
134 sets, 1 on 38, 2 on 14. It is never 3.

This is the same degeneracy E-B1's write-up recorded when the first run passed no
domains at all — but that was a harness misconfiguration, and this is not. The
state is legitimate, the verdicts are real (`rejected`, not `unmeasured`), and the
measurement planner has nothing to say about it. **The gap is that "which
measurement could make something feasible again?" is not a question the
closed-form sensitivity asks.** It ranks by how much a measurement moves an
existing recommendation, and when there is no recommendation it has no signal —
where, in fact, the value of measuring is at its highest.

`require_judged_degraded` now separates the two cases so this cannot be confused
with the misconfiguration again: a state that judged nothing is fatal, a state
that judged everything infeasible runs and is recorded in `degraded_verdicts`.

## Penalty sensitivity

§2.5 asks for the default penalty (207 990 J) and a tenth of it. The regret
values scale by exactly 10, and the **ranking is identical in all 186 sets** —
Spearman ρ = 1.0, zero inversions. On this fixture the penalty is not the knob
that decides the order, which is the opposite of what §2.5 anticipated and worth
carrying into C3.

## What this means for C3

The E-B2 false-positive classification runs on a corpus where `link_bw` is
active-but-never-decisive and two A40 profiles are decisive-but-unrankable. §2.3
requires `link_bw` and `profile` to be non-zero for flip detection; the numbers
above say `link_bw` will produce structural (α) false positives by construction —
it moves metrics, so a crossing exists in its range, and the decision never
follows. That is the α category §2.4 defines, measured rather than assumed.
