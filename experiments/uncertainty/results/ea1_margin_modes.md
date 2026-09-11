# E-A1 — scalar margin vs per-candidate margin (uncertainty work order STEP A5)

**What this is.** The same 162 simulations judged four ways, so the comparison is
of decision rules and not of predictions. Generated tables and the full
per-candidate record are in `ea1_margin_modes_table.md` and
`outputs/uncertainty/ea1/ea1_margin_modes.json`.

**Provenance.** Re-run 2026-09-11 at commit `f0b6b3d` (the A2–B3 reconciliation,
deviations D33) on a node `scripts/whichnode.sh` reports as **unknown — no
accelerator** (4 cores). Simulation only, and not even that: every one of the
2592 cache reads hit, 0 missed, so nothing was simulated here. The cache itself
was filled 2026-09-10/11 on the **a5000** node (`528c6f5`, 24 m 26 s at
`--workers 8`, 0 timeouts, then re-filled at `3c8ad7a` once served concurrency
became per-instance; see the notes at the end). Fixture read from
`outputs/pd_slo_sweep_margin18/pd-rngd-gpu-card.json`:
`experiments/configs/clusters/pd-rngd-gpu-card.yaml` +
`examples/service_specs/llama31-8b.yaml`, 300 requests, seed 42, SLO TTFT
25 000 ms p99 / TPOT 50 ms p99 (the loose-TTFT regime, the one D22's verdict was
re-run in). Canonical bucket `in_lt1024-out_ge512-rps_lt20`, shape
`in_lt1024-out_ge512`.

**What changed since the first two runs.** The first E-A1 (2026-09-10) judged
condition (c) against `profiles/calibration/rngd_card_edf.domain.yaml`, a
four-point domain fitted in E-A2 over served concurrency [15.32, 107.19]; the
second (2026-09-11, `3c8ad7a`) re-ran it with per-instance served concurrency.
That domain file is gone: D33 reconciled the uncertainty stack onto `main`'s
`calibration.AccuracyDomain`, and E-A2's domain turned out to be the D32
mis-pairing (its sim side ran at served 71–189 against real 15–107), so
`main`'s nine-point D32 domain for RNGD-CARD supersedes it — see
`ea2_rngd_domain.md`. Condition (c) therefore now uses the three committed
domains as they are committed, with their explicit `outside_domain:
widen_error_bars`, and a fourth condition (d) applies the same domains under
`refuse`, which is what a *new* domain gets by default. The findings below are
re-derived from this run; the earlier texts are in git history (`1e293a9`,
`3c8ad7a`).

**How the operating point is read.** The cached records carry the served
concurrency **per island** (busiest instance of each island, §2.4.1 rev 2) but
predate the per-hardware, per-phase decomposition the margin policy reads. The
harness places each hardware kind at the busiest of its islands' figures, in
phase "total" — the aggregation `planner/util/operating_point.py` applies to an
aggregated deployment, so for these 324 aggregated and mixed candidates (no P/D)
it is the operating point the predictor would have recorded. All 324 were filled
from the per-island figure, none from the run-level one; the JSON says so.

## Headline

The spec's primary objective is **`minimize_energy`** (secondary
`minimize_active_accelerators`), so plans are ranked on total joules; tokens/J is
co-recorded, and is the figure D22 quoted.

| condition | feasible | recommended | total energy | tokens/J | rejected |
| --- | ---: | --- | ---: | ---: | --- |
| (a) no margin | 70 | `mix(rngd0-tp1-dp1 + rngd1-tp1-dp1)-s256-t8192` | 60 350 J | **3.1635** | slo_violated 254 |
| (b) global 18 % (D22) | 10 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 73 560 J | **2.5954** | slo_violated 314 |
| (c) accuracy domain, committed policy (`widen_error_bars`) | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 73 560 J | **2.5954** | slo_violated 274 |
| (d) accuracy domain, `outside_domain: refuse` | 50 | `cuda-a40-node_a40a-tp4-dp1-s128-t8192` | 73 560 J | **2.5954** | slo_violated 244, **outside_calibration_domain 30** |

**(b) reproduces D22 exactly**, and so do (c) and (d). D22's revalidation
records the winner as `agg[cuda:tp4]` at 2.5954323001631323 tok/J; all three
return that candidate at that value to the last digit. The chain from fixture to
verdict is therefore anchored, and the conditions differ only by the margin rule.

## What the conditions actually disagree about

| candidate kind | n | (a) feasible | (b) feasible | (c) feasible | (d) feasible | (d) unmeasured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A40 only (agg + mix) | 210 | 58 | 10 | 50 | 50 | 10 |
| RNGD-touching (agg, mix, mix RNGD+A40) | 114 | 12 | 0 | 0 | 0 | **20** |

Three findings.

### 1. D22's winner is now rejected on a MEASURED margin, inside the domain

Read per card rather than per deployment, **86 of the 114 RNGD-touching
candidates sit inside the RNGD-CARD domain** — served concurrency 26.6 to 74.7
against a domain measured over [1.02, 76.0] — and every one of them is
`slo_violated` on an interpolated, in-domain TPOT margin of **3.2 to 17.6 %**.
The twelve RNGD candidates (a) admits are among them: the eight A40+RNGD mixes
at per-card L≈26.6 sit at 50.2–50.8 ms p99 TPOT and a 3.2–3.7 % margin tips them
over 50 ms; the four two-card RNGD mixes — (a)'s winner and its knob twins — run
each card at **71.6–74.7** and the domain reads **16.7–17.6 %** there, so 48–50 ms
becomes 55.7–58.3 ms. That is E5's verdict (`agg[furiosa:tp1]` n=2, operating
point ~75, margin ~17.7 %, robust 57 ms) reproduced from inside this fixture by
a different code path, and it is the first E-A1 in which the RNGD arm is decided
by a measurement rather than by an extrapolation or a refusal.

The other 28 put RNGD-CARD at **86 to 178** — above the measured 76 — and here
the policy decision D33 records shows:

* under (c), the committed `widen_error_bars`, the margin is EXTRAPOLATED from
  the -18 % point's slope and grows with distance — 9 to 47 % TPOT — and every
  one is `slo_violated`. The planner says so once, not 28 times: *"20
  candidate(s) put RNGD-CARD at served concurrency 85.96-177.56, outside its
  measured accuracy domain; those margins are EXTRAPOLATED, not measured"* (20,
  not 28: eight of them are also outside the A40 domain and are counted there);
* under (d), `refuse`, no margin exists there and 20 are
  `outside_calibration_domain`: not judged infeasible, not judged at all. The
  rejection names the measurement: *"served concurrency 143.9 on RNGD-CARD
  outside its accuracy domain [1.02, 76] (policy refuse); measure the envelope
  at c>=144 or set outside_domain: widen_error_bars to accept an extrapolation"*.

The first two runs found the RNGD arm entirely outside the domain (118–178
against [15.32, 107.19], then still above it per card). What moved is not the
hardware or the fixture but the coordinate: the per-deployment figure charged a
two-card candidate both cards' load, and the E-A2 domain was fitted at the wrong
operating points (D32).

### 2. The scalar margin's damage was to A40, not to RNGD — and the hypothesis had it backwards

The work order's stated expectation was that **(c) would rescue some
low-concurrency RNGD candidates** from (b)'s blanket rejection. **It did not, and
still cannot: every RNGD candidate inside the domain fails on its own measured
margin**, most by a few tenths of a millisecond at the 50 ms line. Recorded as
the work order requires rather than adjusted after the fact.

What (c) rescued was **40 A40 candidates** (10 feasible → 50). The mechanism is
not per-operating-point precision; it is that **D22's 18 % was measured on RNGD
and condition (b) charges it to A40 hardware**, which nothing justifies. Under
(c) an A40 island is margined from `a40.accuracy.yaml`'s own three-point domain
— 1.1 to 1.3 % TPOT and 3.6 to 7 % TTFT at these loads — and 40 candidates that
were never in doubt come back.

That reframes the argument for the whole stage. A single scalar is not merely
*too harsh at low load and too generous at high load* on one device; on a
heterogeneous cluster it is a **cross-hardware extrapolation**, and that is the
larger error. The A40 arm of this fixture was rejected by a number measured on a
different vendor's accelerator.

The margin is not toothless where it is real, either: (a) passes 58 A40
candidates and (c) passes 50. Two of the eight it removes are the
`agg[cuda:tp4]` placements at `max_num_seqs=256`, which (a) admits at exactly
50.0 ms p99 TPOT and a 1.3 % margin tips over the 50 ms SLO; the other six are
mixed A40 placements whose p99 TTFT sits within 4 % of the 25 s SLO, where the
A40 domain's 6–7 % TTFT margin — measured, on this hardware — binds.

### 3. `refuse` costs nothing here that `widen` would have kept

(c) and (d) admit the same 50 candidates; they disagree on how to describe 30
rejections, not on any verdict. The 10 A40 candidates (d) declines to judge sit
at served 172–198, above the A40 domain's 170.56, and the 20 RNGD ones at
86–178; every one of them (c) also rejected on its extrapolated margin. On this
fixture, in other words, the candidates outside the measured range are all
candidates that fail anyway — which is the benign case for `widen_error_bars`.
The malign case, an extrapolated margin that *admits* a candidate the hardware
would not, is precisely what `refuse` exists to make impossible, and it cannot
be exhibited on a fixture where nothing outside the domain passes.

### Consequence for the recommendation

All three margin conditions recommend `agg[cuda:tp4]` at `max_num_seqs=128`,
73 560 J, 2.5954 tok/J — D22's revalidated winner, at an A40 operating point of
125 inside the A40 domain. The first E-A1 had (c) recommending the `s256` twin
at 12.4 % less energy; that rested on the scalar `a40.yaml` fit's 1.13 % TPOT
margin, under which 50.0 ms p99 passed. The A40's own *domain* charges 1.3 % at
served 154 and the same plan fails by the width of that margin. Neither answer
was dishonest; the second rests on a measurement taken at the plan's operating
point, the first on one taken at 10 rps.

(a)'s winner — the two-card RNGD mix at 3.1635 tok/J — is the configuration D22
retracted. Under (c) and (d) alike it is rejected at 56.9 ms robust TPOT on a
margin the domain measured at that load. That is the same conclusion D22 reached
by hand and E5 reached from the record; here the planner reaches it from the
card's own operating point.

## Suggestions and notes the planner emitted

(b) emits none: with 10 candidates feasible it simply reports them.

(c) emits two notes and no suggestion — every candidate was judged:

> 10 candidate(s) put A40 at served concurrency 172.36-197.88, outside its
> measured accuracy domain; those margins are EXTRAPOLATED, not measured
>
> 20 candidate(s) put RNGD-CARD at served concurrency 85.96-177.56, outside its
> measured accuracy domain; those margins are EXTRAPOLATED, not measured

(d) emits two suggestions, naming the strongest undecidable candidate and where
to measure it:

> 30 candidate(s) could not be judged: their operating point lies outside every
> measured accuracy domain (or their hardware has none), so no margin applies.
> They are undecidable, not infeasible (outside_calibration_domain).
>
> the strongest of them by minimize_energy is
> `furiosa-rngd-card-node_rngd0-tp1-dp1-s128-t8192` (5.005e+04) at served
> concurrency 143.9 - measure the envelope there to decide it, or set
> outside_domain: widen_error_bars to accept an extrapolation.

That names one measurement that would settle the best undecidable plan. STEP B3
turns it into a ranked, costed plan over every uncertain input
(`plan --accuracy-domain --measurement-plan`).

## Reproduce

```bash
# 1. fill the envelope (24 m 26 s at --workers 8 on 20 cores; 0 timeouts)
python -m planner plan \
  --service examples/service_specs/llama31-8b.yaml \
  --cluster experiments/configs/clusters/pd-rngd-gpu-card.yaml \
  --num-requests 300 --seed 42 \
  --cache-dir outputs/uncertainty/ea1/cache \
  --work-dir outputs/uncertainty/ea1/work \
  --output outputs/uncertainty/ea1/a_margin0.yaml --workers 8 --timeout 900

# 2. the four conditions, off that cache, no simulation (~3 min on 4 cores)
PYTHONPATH=$PWD .venv/bin/python experiments/uncertainty/ea1_margin_modes.py \
  --cache-dir outputs/uncertainty/ea1/cache \
  --out experiments/uncertainty/results/ea1_margin_modes_table.md \
  --json-out outputs/uncertainty/ea1/ea1_margin_modes.json
```

## Notes

**The cache had to be rebuilt once, and that is a finding.** Re-running the
conditions after served concurrency became per-instance reproduced the previous
numbers exactly — which read as "the change had no effect" and was in fact "the
change was never applied": `PredictedMetrics.model_validate` accepts an entry
written before a field existed, so the envelope cache served stale-schema metrics
silently. `EnvelopeCache` now stores a digest of the metrics field set in every
entry it writes and treats a mismatch as a miss (D33 keeps it in the payload
rather than the file name, so `main`'s committed replay caches keep their names);
an entry with no digest at all is served with a warning saying so. The 162
entries here were re-keyed to the repository's cache naming and stamped.

**D23 did not bite when the cache was filled, and the project instructions were
wrong about it.** 228 of the 324 candidates are `mix_*`, which `CLAUDE.md` said
would every one livelock. D23 has been resolved since 2026-09-07 — the symptom
was D26 and the crashes D25, and the discriminator is instance count, not
candidate kind. All 162 simulations completed with 0 timeouts, which is the
third independent confirmation of that after D23's own revalidation.

**Why (c) does not use `--top-k`.** D30 forbids it as a cost lever on P/D or
heterogeneous corpora: the roofline surrogate's proxy is invariant to TP and DP,
and `--top-k 20` has turned a feasible plan infeasible on `pd-rngd-gpu`. The full
162 were simulated instead.

**P/D is off**, matching `plan`'s default, so the 468-candidate `--enable-pd`
space is not covered here. The 324 are 96 aggregated + 228 mixed.

**TTFT on RNGD carries no margin.** The RNGD-CARD domain is TPOT-only (its real
side is a closed-loop bench, D19), so the RNGD candidates' TTFT check ran
unmargined in every condition; the planner records it as a caveat rather than a
rejection (D33). It changes nothing here — 84 of the 86 in-domain RNGD
candidates fail on TPOT, and the 26 that also fail on TTFT do so by 3–4× the
SLO, far beyond any plausible margin.
