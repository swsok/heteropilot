# V3 — verdict accuracy: what the hardware says about the four rules

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V3. **Deployment has not run.**
This file exists now because one of its results does not need a deployment, and
because the candidates and their caveats are fixed before the GPU is touched —
the work order requires the selection to be confirmed first, and it was
(2026-09-16). The measured sections are marked PENDING and are filled by the
deployment run.*

## Status

| section | state |
| --- | --- |
| §1 side result — what holding costs | **complete**, needs no hardware |
| §2 the candidates and their predictions | **complete** (selection confirmed, predictions re-verified) |
| §3 measured verdicts vs rules | **PENDING** — needs A40 GPUs |
| §4 what V3 cannot settle | **complete** |

---

## 1. Side result: on this fixture, holding costs nothing

E-A1's `refuse` policy holds **30** candidates as `outside_calibration_domain`
rather than judging them. The disclosure's §6 asks what that costs — whether the
policy withholds candidates that would have been recommended.

**All 30 are already rejected under rule (a), with no margin applied at all.**

So on this fixture the policy never withholds a candidate the unmargined
simulator would have accepted: every held candidate is one that was going to be
rejected anyway, and holding changes the *reason* recorded, not the outcome.
Their composition is 10 A40-only, 12 RNGD-only and 8 mixed.

The ten A40-only ones sit at served concurrency **172.4 to 197.9**, all above the
A40 domain's top measured point at **170.56**, with predicted p99 TPOT of **90 to
229 ms** against a 50 ms SLO.

**What this does and does not establish.** It establishes that `refuse` is cheap
here — a reader asking "how many good candidates does the refusal throw away?"
gets the answer **zero, on this fixture**. It does not establish that refusing is
*right*, because no rule wanted these candidates; a fixture in which a held
candidate is feasible unmargined would be needed for that, and none exists in the
committed corpora. It is also fixture-specific: the 30 are held because their
operating points exceed the domains' top measured points, which is a property of
where the sweep lands, not of the policy.

**This is why P4 was dropped from the deployment** (user decision, 2026-09-16):
measuring one held candidate would record what the hardware does at an unmeasured
operating point — useful for the domain, and V2's extra `L > 170` stage now picks
that up more cheaply — but it could not have shown the refusal to be wrong.

## 2. The candidates, and the predictions V3 will compare against

Selection and its reasoning: `v3_candidate_selection.md`. SLOs are the fixture's
own: **TPOT 50.0 ms, TTFT 25 000 ms**.

| group | candidate | GPUs | predicted p99 TPOT / TTFT | L | deciding metric | separation |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| P1 | `cuda-a40-node_a40a-tp4-dp1-s128-t2048` | 4 | 36.79 / 16 159 ms | 127.3 | — (all four agree) | — |
| P2 | `mix(a40a-tp2-dp1+a40b-tp2-dp2)-s32-t8192` | 6 | 30.57 / 24 064 ms | 39.8 | TTFT | +3 111.6 ms over |
| P3 | `mix(a40a-tp1-dp1+a40b-tp1-dp4)-s32-t2048` | 5 | 49.16 / 14 755 ms | 28.1 | TPOT | (b) +8.009 over, (c) 0.623 under |

**D40 was checked, not assumed.** Both mix candidates were re-simulated on their
own with the cache disabled, so a key collision was impossible rather than
unlikely — `v3_resimulation.json`, produced by `v3_resimulate.py`. Result below
in §2.1.

**P3's result must be recorded beside V2's run-to-run p99 spread** (user
instruction). Rule (b) is wrong by 8.009 ms if the hardware comes in under 50 ms;
rule (c) is right by only **0.623 ms**. The first claim survives any plausible
repeatability; the second does not survive a spread above ~0.6 ms, and V3 must
say which case it is rather than reporting a bare pass.

**P2 must be driven open loop** at the arrival rate matching its predicted
operating point, not by a closed-loop burst. D19: a closed-loop TTFT is inflated
by queueing and does not transfer, and P2 is decided *on TTFT*.

### 2.1 Re-simulation result — D40 did not fire

Both candidates re-simulated to **byte-identical** metrics, on every compared
field (`p99_tpot_ms`, `p99_ttft_ms`, `p50`/`p95` TPOT, served concurrency,
throughput, energy, tokens/J): worst |Δ| = **0.000000 %** for each.

| group | p99 TPOT cached → resim | p99 TTFT cached → resim | served L |
| --- | --- | --- | ---: |
| P3 | 49.16014071 → 49.16014071 | 14 754.56842926 → 14 754.56842926 | 28.0938 |
| P2 | 30.57248391 → 30.57248391 | 24 063.81209391 → 24 063.81209391 | 39.7532 |

**What this establishes, exactly.** The cached entry each candidate was served is
the one its own configuration produces, so **the predictions V3 compares against
are these candidates' own** — which is all V3 needs. It does **not** establish
that D40 is harmless in general: the mirrors may still share the key, in which
case the entry is this candidate's and it is the *mirror's* cached value that is
wrong. D40 stays open.

Method: `v3_resimulate.py` calls `predict()` directly with **the cache disabled**,
so a key collision is impossible rather than unlikely. Same cluster, same service
spec, same 300-request trace at seed 42 as E-A1. Raw: `v3_resimulation.json`;
run log `experiments/p2_evidence/results/v3_resim.log` (the work dir itself is
regenerable and untracked).

## 3. Measured verdicts against the four rules

**PENDING — needs A40 GPUs.** The table will carry, per candidate: measured p99
TTFT and TPOT, the verdict each of the four rules gave, and whether the hardware
agreed. Metrics come from the server's `/metrics`, aggregated on the basis V0
fixed.

Per the work order, **the rates are cases, not statistics** — three candidates
cannot estimate a pass rate, and the specification must present them as a case
table without generalising.

## 4. What V3 cannot settle

V3 is **A40-only**; there is no RNGD card on this node. The A40 per-point TPOT
margin is 0.37–1.63 %, against ~22 % for RNGD-CARD at the sweep's operating
point, so the verdict-changing cases live on RNGD — including the disclosure's
own §5.6 candidate B (predicted 48.41, robust 59.04). The §6 numbers for that
regime are therefore built from **V0 and V1**, not from V3, and
`patent2_evidence_map.md` says so. **V3-R** — one RNGD candidate each of the P2
and P3 shapes — is left as a selection step for when an RNGD node is available.
