"""E5: the planner rejects D22's winner on its own, with no manual margin.

`WORK_ORDER_rps_aware.md` rev 2 STEP 5. D22's retraction needed a human to notice
that the simulator was 18 % optimistic at the load the winner implied, and to
re-run the sweep with that as a hand-set margin. The question E5 asks is whether
the pipeline now finds it without being told.

It does, and this test holds it there. No simulator is involved: the 199 committed
simulation records from the 2026-09-07 re-validation sweep are replayed through a
mock predictor, so CI can keep the verdict permanently.

What must hold:

  * the committed winner -- two RNGD cards, tp1 each, predicted p99 TPOT 48.41 ms
    against a 50 ms SLO -- ran at served concurrency ~75;
  * the accuracy domain prices the simulator at ~-17.7 % there, so the robust TPOT
    is ~57.0 ms and the candidate is SLO_VIOLATED;
  * the recommendation becomes `agg[cuda:tp4]` at 2.595 tok/J.

The margin is asserted with a tolerance because it is INTERPOLATED between the
domain's 16.6 and 76 points and the operating point is 74.75, not 76. Tightening
the tolerance by moving a domain point would be fitting the calibration to the
answer, which the work order forbids in as many words.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planner.inventory import detect_islands, load_cluster_spec, load_profiles_for
from planner.optimizer import exhaustive
from planner.plan import PredictedMetrics, RejectionStage
from planner.predictor import Predictor, SimOutcome, SimResult
from planner.predictor.calibration import load_accuracy_domains
from planner.spec import load_service_spec

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "tests/data/e5_sim_records.json"

#: The committed winner: `agg[furiosa:tp1]` n=2, hp-00323 in the 2026-08-26 sweep.
WINNER = ("mix(furiosa-rngd-card-node_rngd0-tp1-dp1+"
          "furiosa-rngd-card-node_rngd1-tp1-dp1)-s256-t8192")
TPOT_SLO_MS = 50.0


class _Replay(Predictor):
    """Returns the committed record for a candidate, or a crash for one that was
    never simulated -- which is what the original sweep saw too."""

    def __init__(self, records: dict):
        self.records = records
        self.calls: list[str] = []

    def predict(self, candidate, spec, cluster, islands, profiles) -> SimResult:
        self.calls.append(candidate.id)
        rec = self.records.get(candidate.id)
        if rec is None:
            return SimResult(candidate.id, SimOutcome.CRASHED,
                             detail="not in the committed record set")
        return SimResult(
            candidate.id, SimOutcome.OK,
            metrics=PredictedMetrics.model_validate(rec["metrics"]),
            operating_point=rec.get("operating_point", {}),
        )


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(RECORDS.read_text())


@pytest.fixture(scope="module")
def domains() -> dict:
    return load_accuracy_domains(ROOT)


@pytest.fixture(scope="module")
def searched(data, domains):
    """`evaluate_candidates` rather than `search`: the per-candidate rejection
    stage and applied margin are what E5 is about, and `PlannerOutput` carries
    only the counts."""
    from planner.candidate_generator import CandidateGenerator

    spec = load_service_spec(ROOT / data["service"]).model_copy(deep=True)
    spec.slo.ttft.max_ms = 64000.0
    cluster = load_cluster_spec(ROOT / data["fixture"])
    profiles = load_profiles_for(cluster, ROOT)
    islands = detect_islands(cluster, profiles)
    by_id = {i.id: i for i in islands}
    generated = CandidateGenerator(
        spec, cluster, islands, profiles, enable_prefix_caching=False,
        enable_bound_pruning=True, enable_pd=True,
    ).generate()
    return exhaustive.evaluate_candidates(
        generated.candidates, spec, cluster, by_id, profiles,
        _Replay(data["records"]),
        accuracy_domains=domains,
        # No manual margin. That is the experiment.
        ttft_margin_percent=0.0, tpot_margin_percent=0.0,
    )


def test_the_committed_winner_ran_at_the_load_d22_identified(data):
    op = data["records"][WINNER]["operating_point"]["RNGD-CARD"]
    assert op["concurrency"] == pytest.approx(74.75, abs=1.0), (
        "D22 interpolated the winner's load to ~76; if this has moved, the whole "
        "premise of the -18 % correction moved with it"
    )
    assert data["records"][WINNER]["metrics"]["p99_tpot_ms"] == pytest.approx(48.41, abs=0.01)


def test_the_accuracy_domain_prices_that_load_at_about_eighteen_percent(domains, data):
    conc = data["records"][WINNER]["operating_point"]["RNGD-CARD"]["concurrency"]
    margin = domains["RNGD-CARD"].tpot_margin_pct(conc)
    assert margin == pytest.approx(21.41, abs=0.5)
    robust = 48.41 * (1 + margin / 100.0)
    assert robust > TPOT_SLO_MS, f"robust TPOT {robust:.2f} should breach {TPOT_SLO_MS}"
    assert robust == pytest.approx(58.78, abs=0.3)


def test_the_winner_is_rejected_for_the_slo_and_not_for_something_else(searched):
    """It must fail on TPOT, not fall out for an unrelated reason."""
    rejected = {r.candidate_id: r for r in searched.rejections}
    assert WINNER in rejected, "the committed winner should not survive E5"
    assert rejected[WINNER].stage is RejectionStage.SLO_VIOLATED, rejected[WINNER]


def test_the_margin_that_rejected_it_is_the_one_the_domain_gave(searched):
    plan = next(p for p, _ in searched.infeasible_plans if p.candidate.id == WINNER)
    assert plan.margin_source == "accuracy_domain"
    assert plan.robust_margin_tpot_percent == pytest.approx(21.41, abs=0.5)
    op = {r.hardware: r for r in plan.operating_point}
    assert op["RNGD-CARD"].in_calibration_domain is True, (
        "74.75 sits between the domain's 16.6 and 76 points, so the margin is "
        "interpolated rather than extrapolated"
    )


def test_the_recommendation_becomes_the_a40_plan(searched):
    from planner.optimizer import pareto
    from planner.spec import Objective

    assert searched.feasible_plans, "something must remain feasible"
    best = pareto.rank(searched.feasible_plans, Objective.MINIMIZE_ENERGY)[0].plan
    assert best.candidate.id.startswith("cuda-a40"), best.candidate.id
    assert best.candidate.assignments[0].tp_size == 4
    assert best.predicted.tokens_per_joule == pytest.approx(2.595, abs=0.001)


def test_no_manual_margin_was_used(searched):
    """The point of E5 is that nobody had to know the number in advance."""
    for plan in searched.feasible_plans:
        assert plan.margin_source in ("accuracy_domain", ""), plan.margin_source
