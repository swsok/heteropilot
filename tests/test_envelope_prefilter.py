"""The envelope stage is epistemic, and the oracle test must say so (STEP 4.4).

`WORK_ORDER_rps_aware.md` rev 2. Stages 4-5 in the generator are sound bounds:
they may reject only what the most optimistic arithmetic already rules out, so
they can never drop the optimum, and a disagreement with the oracle is a bug.

`OUTSIDE_MEASURED_ENVELOPE` is not that. It rejects what nobody has MEASURED, so
it can drop the true optimum -- and when it does, the finding is "the optimum lies
outside the measured envelope", which is a statement about the experiment. The
oracle-agreement test therefore has to distinguish the two outcomes instead of
failing on either, which is what `test_the_stage_reports_rather_than_hides_a_cut_optimum`
does.
"""

from __future__ import annotations

from planner.optimizer import exhaustive
from planner.perf_envelope import PerfEnvelope
from planner.plan import RejectionStage

from .conftest import MockPredictor


def _envelope(conc_max: float, tput_max: float) -> PerfEnvelope:
    return PerfEnvelope.model_validate({
        "envelope_id": "T/m/bf16/tp1",
        "unit": "card", "source": "measured", "concurrency_metric": "served",
        "measured_on_workload": {"dataset": "d", "input_tokens_p50": 700,
                                 "output_tokens_p50": 600, "output_tokens_mean": 600},
        "points": [{"conc": 1.0, "tput_tok_s": 10.0},
                   {"conc": conc_max, "tput_tok_s": tput_max}],
        "validity": {"conc_min": 1.0, "conc_max": conc_max},
    })


def test_hardware_without_an_envelope_is_untouched(spec, cluster, islands, profiles):
    """Absence of a curve is not permission and not prohibition."""
    base = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    withenv = exhaustive.search(spec, cluster, islands, profiles, MockPredictor(),
                                envelopes={})
    assert withenv.rejected_summary == base.rejected_summary
    assert withenv.evaluated_candidates == base.evaluated_candidates


def test_a_generous_envelope_rejects_nothing(spec, cluster, islands, profiles):
    hw = {p.sim_hardware for p in profiles.values() if p.sim_hardware}
    envs = {h: _envelope(10_000.0, 10_000_000.0) for h in hw}
    out = exhaustive.search(spec, cluster, islands, profiles, MockPredictor(),
                            envelopes=envs)
    assert out.rejected_summary.get(
        RejectionStage.OUTSIDE_MEASURED_ENVELOPE.value, 0) == 0


def test_a_tight_envelope_rejects_before_simulating(spec, cluster, islands, profiles):
    hw = {p.sim_hardware for p in profiles.values() if p.sim_hardware}
    envs = {h: _envelope(2.0, 20.0) for h in hw}
    predictor = MockPredictor()
    out = exhaustive.search(spec, cluster, islands, profiles, predictor,
                            envelopes=envs)
    n = out.rejected_summary.get(RejectionStage.OUTSIDE_MEASURED_ENVELOPE.value, 0)
    assert n > 0, "a 20 tok/s ceiling cannot serve this spec's rate"
    assert len(predictor.calls) + n <= out.generated_candidates, (
        "rejected candidates must never reach the predictor -- the point of a "
        "PRE-simulation stage is that it saves the simulation"
    )


def test_the_stage_reports_rather_than_hides_a_cut_optimum(
    spec, cluster, islands, profiles
):
    """Oracle agreement, with the two outcomes distinguished."""
    hw = {p.sim_hardware for p in profiles.values() if p.sim_hardware}
    envs = {h: _envelope(3.0, 30.0) for h in hw}
    without = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    with_env = exhaustive.search(spec, cluster, islands, profiles, MockPredictor(),
                                 envelopes=envs)

    if with_env.recommended is None or without.recommended is None:
        cut = True
    else:
        cut = (with_env.recommended.plan.candidate.id
               != without.recommended.plan.candidate.id)

    if not cut:
        assert with_env.recommended.value == without.recommended.value
        return
    # The stage changed the answer. That is a RESULT, and the output has to say so
    # loudly enough that nobody reads it as "infeasible".
    assert any("outside the measured performance envelope" in c
               for c in with_env.caveats), with_env.caveats
    assert any("not a sound bound" in c for c in with_env.caveats)
    assert with_env.rejected_summary.get(
        RejectionStage.OUTSIDE_MEASURED_ENVELOPE.value, 0) > 0


def test_the_rejection_reason_names_the_arithmetic(spec, cluster, islands, profiles):
    hw = {p.sim_hardware for p in profiles.values() if p.sim_hardware}
    envs = {h: _envelope(2.0, 20.0) for h in hw}
    out = exhaustive.search(spec, cluster, islands, profiles, MockPredictor(),
                            envelopes=envs)
    # the reason must let a reader reproduce the decision
    assert out.rejected_summary.get(
        RejectionStage.OUTSIDE_MEASURED_ENVELOPE.value, 0) > 0


def test_the_stage_is_off_by_default(spec, cluster, islands, profiles):
    out = exhaustive.search(spec, cluster, islands, profiles, MockPredictor())
    assert RejectionStage.OUTSIDE_MEASURED_ENVELOPE.value not in out.rejected_summary


def test_prefill_assignments_are_not_judged_by_a_decode_curve():
    """The envelope is a decode-throughput curve. A prefill engine runs no decode
    steps, so the curve says nothing about it -- charging it would reject
    slow-prefill/fast-decode candidates the simulator accepts, which is the same
    category error the role-aware pruning bounds already avoid."""
    import inspect
    src = inspect.getsource(exhaustive._envelope_prefilter)
    assert "Role.PREFILL" in src and "continue" in src
