"""The four hooks H3 opens for the graph-search driver, and their no-op defaults.

Every one of them is off unless a caller reaches for it. That is the whole
design: `swsok/heteropilot-graphsearch` drives the search itself - it enumerates
physical embeddings, folds them by exact equivalence and evaluates
representatives in batches - and it has to do that through this code rather than
around it, or the golden outputs stop guarding anything. So H3 adds no
behaviour, only seams:

* `evaluate_candidates(plan_id_base=)` so batched evaluation keeps plan ids unique
* `_assemble_output()` so a driver can build the same `PlannerOutput`
* `EnvelopeCache(graph_signature=)` so placements the envelope key cannot see
  do not collide
* `LLMServingSimPredictor.set_compile_hook()` so a driver can compile a
  placement the planner has no type for

The last test in each section is the one that matters: with the seam unused,
nothing moved.
"""

from __future__ import annotations

import pytest

from planner.envelope import EnvelopeCache
from planner.plan import (
    CandidateConfig,
    IslandAssignment,
    PredictedMetrics,
    Rejection,
    RejectionStage,
    VllmKnobs,
    summarize_rejections,
)
from planner.predictor import SimOutcome, SimResult
from planner.util.workload import WorkloadTrace


def _trace(tmp_path) -> WorkloadTrace:
    p = tmp_path / "wl.jsonl"
    p.write_text('{"input_toks": 8, "output_toks": 4, "arrival_time_ns": 0}\n')
    return WorkloadTrace(path=p, num_requests=1, seed=42,
                         total_input_tokens=8, total_output_tokens=4, horizon_s=1.0)


def candidate(cid: str) -> CandidateConfig:
    return CandidateConfig(
        id=cid, model="meta-llama/Llama-3.1-8B", dtype="bfloat16",
        assignments=[IslandAssignment(island_id="isl", tp_size=1)],
    )


# --- (i) plan_id_base -----------------------------------------------------

def test_plan_ids_continue_from_the_base(
    spec, cluster, islands, profiles, mock_predictor
) -> None:
    from planner.optimizer.exhaustive import evaluate_candidates

    by_id = {i.id: i for i in islands}
    cands = [
        CandidateConfig(
            id=f"c{n}", model=spec.model, dtype="bfloat16",
            assignments=[IslandAssignment(island_id=islands[0].id, tp_size=1)],
            knobs=VllmKnobs(max_num_seqs=cap),
        )
        for n, cap in enumerate((32, 64, 128))
    ]
    result = evaluate_candidates(
        cands, spec, cluster, by_id, profiles, mock_predictor, plan_id_base=5
    )
    ids = [p.plan_id for p in result.feasible_plans] + [
        p.plan_id for p, _ in result.infeasible_plans
    ]
    assert sorted(ids) == ["hp-00005", "hp-00006", "hp-00007"]


def test_the_default_base_is_zero(
    spec, cluster, islands, profiles, mock_predictor
) -> None:
    from planner.optimizer.exhaustive import evaluate_candidates

    by_id = {i.id: i for i in islands}
    cands = [
        CandidateConfig(
            id="c0", model=spec.model, dtype="bfloat16",
            assignments=[IslandAssignment(island_id=islands[0].id, tp_size=1)],
        )
    ]
    result = evaluate_candidates(
        cands, spec, cluster, by_id, profiles, mock_predictor
    )
    ids = [p.plan_id for p in result.feasible_plans] + [
        p.plan_id for p, _ in result.infeasible_plans
    ]
    assert ids == ["hp-00000"]


# --- (ii) _assemble_output ------------------------------------------------

def test_assemble_output_is_what_search_returns(
    spec, cluster, islands, profiles, mock_predictor
) -> None:
    """The extraction is a refactor, so `search` must still be its only caller
    in this repo and must still produce what it always did. `tests/test_search.py`
    and `tests/test_render.py` carry the byte-level half of that; this checks the
    seam exists and is callable on its own."""
    from planner.optimizer import exhaustive

    out = exhaustive.search(
        spec, cluster, islands, profiles, mock_predictor
    )
    direct = exhaustive._assemble_output(
        spec=spec,
        cluster=cluster,
        generated=out.generated_candidates,
        evaluation=exhaustive.SearchResult(),
        all_rejections=[],
        caveats=[],
        prov={},
        island_tiers={},
        island_hw={},
    )
    # An empty evaluation is infeasible with the "nothing was simulated" reason,
    # which is exactly what `search` reports for an empty candidate set.
    assert direct.feasible is False
    assert "nothing was simulated" in direct.reason
    assert direct.service_model == spec.model
    assert direct.cluster_id == cluster.cluster_id


# --- (iii) the cache's graph signature ------------------------------------

def _cache(tmp_path, spec, **kw) -> EnvelopeCache:
    return EnvelopeCache(
        tmp_path, spec, accelerator_of={"isl": "A5000"}, link_bw_gbps=64.0, **kw
    )


def test_no_signature_leaves_the_file_name_alone(tmp_path, spec) -> None:
    """The committed replay caches keep their names, which is the point."""
    plain = _cache(tmp_path, spec)
    assert plain.graph_signature is None
    assert plain.cache_key(candidate("c")) == _cache(tmp_path, spec).cache_key(candidate("c"))


def test_two_signatures_give_three_distinct_keys(tmp_path, spec) -> None:
    c = candidate("c")
    keys = {
        _cache(tmp_path, spec).cache_key(c),
        _cache(tmp_path, spec, graph_signature="x").cache_key(c),
        _cache(tmp_path, spec, graph_signature="y").cache_key(c),
    }
    assert len(keys) == 3


def test_with_graph_signature_shares_the_counters(tmp_path, spec) -> None:
    """A driver makes one sibling per representative and still wants one tally."""
    base = _cache(tmp_path, spec)
    sibling = base.with_graph_signature("rep-1")
    assert sibling.graph_signature == "rep-1"
    assert sibling.root == base.root

    assert sibling.get(candidate("c")) is None          # a miss on the sibling
    assert base.stats() == {"hits": 0, "misses": 1}     # counted on the original
    assert sibling.stats() == base.stats()


def test_siblings_do_not_share_a_key(tmp_path, spec) -> None:
    base = _cache(tmp_path, spec)
    assert base.cache_key(candidate("c")) != base.with_graph_signature("s").cache_key(
        candidate("c")
    )


# --- (iv) the predictor's compile hook ------------------------------------

def _real_candidate(spec, islands) -> CandidateConfig:
    return CandidateConfig(
        id="c", model=spec.model, dtype="bfloat16",
        assignments=[IslandAssignment(island_id=islands[0].id, tp_size=1)],
    )


def _stub_predictor(monkeypatch, tmp_path, hook):
    """A predictor whose real compile always fails loudly and whose simulator
    never runs, so `predict` reports which of the two paths it took."""
    from planner.predictor import llmservingsim

    def boom(*_a, **_k):
        raise llmservingsim.CompileError("the real compile ran")

    monkeypatch.setattr(llmservingsim, "compile_to_sim_config", boom)
    predictor = llmservingsim.LLMServingSimPredictor(
        _trace(tmp_path), work_dir=tmp_path / "w"
    )
    monkeypatch.setattr(
        predictor, "_run_once",
        lambda *a, **k: SimResult("c", SimOutcome.OK, detail="hook config was used"),
    )
    predictor.set_compile_hook(hook)
    return predictor


def test_no_hook_takes_the_normal_compile(monkeypatch, tmp_path, spec, cluster,
                                          islands, profiles) -> None:
    predictor = _stub_predictor(monkeypatch, tmp_path, None)
    result = predictor.predict(_real_candidate(spec, islands), spec, cluster,
                               {i.id: i for i in islands}, profiles)
    assert result.outcome is SimOutcome.CRASHED
    assert "the real compile ran" in result.detail


def test_a_hook_returning_none_declines_and_falls_through(monkeypatch, tmp_path, spec,
                                                          cluster, islands, profiles) -> None:
    """None means "not mine", so a driver binds only what it drives."""
    seen: list[str] = []
    predictor = _stub_predictor(
        monkeypatch, tmp_path, lambda cand, *a: seen.append(cand.id) or None
    )
    result = predictor.predict(_real_candidate(spec, islands), spec, cluster,
                               {i.id: i for i in islands}, profiles)
    assert seen == ["c"]
    assert "the real compile ran" in result.detail


def test_a_hook_returning_a_config_replaces_the_compile(monkeypatch, tmp_path, spec,
                                                        cluster, islands, profiles) -> None:
    from planner.topology import TopologyReduction

    reduction = TopologyReduction(link_bw_gbps=1.0, link_latency_ns=2.0, basis="hook")
    predictor = _stub_predictor(
        monkeypatch, tmp_path,
        lambda *a: ({"num_nodes": 1, "link_bw": 1.0, "link_latency": 2.0,
                     "nodes": [{"power": {}}]}, reduction),
    )
    result = predictor.predict(_real_candidate(spec, islands), spec, cluster,
                               {i.id: i for i in islands}, profiles)
    assert result.outcome is SimOutcome.OK
    assert result.detail == "hook config was used"
    assert predictor.last_reduction is reduction


def test_clearing_the_hook_restores_the_normal_path(monkeypatch, tmp_path, spec, cluster,
                                                    islands, profiles) -> None:
    predictor = _stub_predictor(monkeypatch, tmp_path, lambda *a: None)
    predictor.set_compile_hook(None)
    result = predictor.predict(_real_candidate(spec, islands), spec, cluster,
                               {i.id: i for i in islands}, profiles)
    assert "the real compile ran" in result.detail


# --- (v) the three new rejection stages -----------------------------------

@pytest.mark.parametrize(
    "stage, value",
    [
        (RejectionStage.THROUGHPUT_UPPER_BOUND, "throughput_upper_bound"),
        (RejectionStage.EXCLUDED_BY_SCOPE, "excluded_by_scope"),
        (RejectionStage.NOT_EVALUATED_BUDGET, "not_evaluated_budget"),
    ],
)
def test_new_stages_summarise_under_their_own_key(stage, value) -> None:
    summary = summarize_rejections(
        [Rejection(candidate_id="c", stage=stage, reason="r")]
    )
    assert summary[value] == 1


def test_the_two_non_verdicts_do_not_collide_with_a_verdict() -> None:
    """`excluded_by_scope` and `not_evaluated_budget` say something about the
    SEARCH, not the candidate. Counting either as infeasible would report a
    budget as a property of the hardware, which is the same mistake
    `outside_calibration_domain` exists to avoid."""
    summary = summarize_rejections([
        Rejection(candidate_id="a", stage=RejectionStage.SLO_VIOLATED, reason="r"),
        Rejection(candidate_id="b", stage=RejectionStage.EXCLUDED_BY_SCOPE, reason="r"),
        Rejection(candidate_id="c", stage=RejectionStage.NOT_EVALUATED_BUDGET, reason="r"),
    ])
    assert summary["slo_violated"] == 1
    assert summary["excluded_by_scope"] == 1
    assert summary["not_evaluated_budget"] == 1


def test_existing_stage_strings_are_unchanged() -> None:
    """These strings are `rejected_summary` keys and appear in frozen output."""
    assert RejectionStage.SLO_VIOLATED.value == "slo_violated"
    assert RejectionStage.SURROGATE_PRUNED.value == "surrogate_pruned"
    assert RejectionStage.OUTSIDE_CALIBRATION_DOMAIN.value == "outside_calibration_domain"
    assert RejectionStage.SIM_ERROR.value == "sim_error"


# --- H4: the three hooks the graph driver needs to price and key correctly ---

def _pd_candidate(spec, islands) -> CandidateConfig:
    """A P/D candidate across the first two islands of different nodes."""
    from planner.plan import Role, ServingArch

    by_node: dict[str, str] = {}
    for island in islands:
        by_node.setdefault(island.node_id, island.id)
    nodes = sorted(by_node)[:2]
    return CandidateConfig(
        id="pd", model=spec.model, dtype="bfloat16",
        serving_arch=ServingArch.PD_SPLIT,
        assignments=[
            IslandAssignment(island_id=by_node[nodes[0]], role=Role.PREFILL, tp_size=1),
            IslandAssignment(island_id=by_node[nodes[1]], role=Role.DECODE, tp_size=1),
        ],
    )


def test_pd_transfer_can_be_left_to_the_caller(
    spec, cluster, islands, profiles, mock_predictor
) -> None:
    """D125: a caller that knows the physical path prices the handoff itself.

    The class-default figure has to be ABSENT rather than subtracted afterwards:
    the feasibility verdict is taken inside `evaluate_candidates`, so a metric
    corrected after the fact would disagree with the verdict already reached.
    """
    from planner.optimizer.exhaustive import evaluate_candidates

    by_id = {i.id: i for i in islands}
    candidate = _pd_candidate(spec, islands)

    charged = evaluate_candidates(
        [candidate], spec, cluster, by_id, profiles, mock_predictor
    )
    assert charged.pd_transfers, "the fixture produced no P/D handoff to price"

    left = evaluate_candidates(
        [candidate], spec, cluster, by_id, profiles, mock_predictor,
        pd_transfer=False,
    )
    assert left.pd_transfers == []

    def ttft(result) -> float:
        plans = result.feasible_plans + [p for p, _ in result.infeasible_plans]
        return plans[0].predicted.p99_ttft_ms

    added = float(charged.pd_transfers[0]["xfer_ms_p99"])
    assert ttft(charged) == pytest.approx(ttft(left) + added)


def test_the_default_still_charges_it(
    spec, cluster, islands, profiles, mock_predictor
) -> None:
    from planner.optimizer.exhaustive import evaluate_candidates

    by_id = {i.id: i for i in islands}
    result = evaluate_candidates(
        [_pd_candidate(spec, islands)], spec, cluster, by_id, profiles, mock_predictor
    )
    assert result.pd_transfers


def test_a_result_hook_is_applied_before_the_verdict_and_the_cache(
    monkeypatch, tmp_path, spec, cluster, islands, profiles
) -> None:
    """D125. A correction made after `predict` would leave the verdict taken on
    different numbers, and would be lost entirely on a cache hit."""
    from dataclasses import replace

    from planner.predictor import llmservingsim

    predictor = llmservingsim.LLMServingSimPredictor(
        _trace(tmp_path), work_dir=tmp_path / "w"
    )
    baseline = SimResult(
        "c", SimOutcome.OK,
        metrics=PredictedMetrics(
            p50_ttft_ms=1.0, p95_ttft_ms=1.0, p99_ttft_ms=1.0,
            p50_tpot_ms=1.0, p95_tpot_ms=1.0, p99_tpot_ms=1.0,
            throughput_tps=1.0, slo_goodput_rps=1.0, slo_attainment=1.0,
            completed_requests=1, completed_tokens=1,
        ),
    )
    monkeypatch.setattr(
        llmservingsim, "compile_to_sim_config",
        lambda *a, **k: ({"num_nodes": 1, "nodes": [{}]}, None),
    )
    monkeypatch.setattr(predictor, "_run_once", lambda *a, **k: baseline)

    def add_100(candidate, result):
        metrics = result.metrics
        return replace(
            result,
            metrics=metrics.model_copy(
                update={"p99_ttft_ms": metrics.p99_ttft_ms + 100.0}
            ),
        )

    predictor.set_result_hook(add_100)
    candidate = CandidateConfig(
        id="c", model=spec.model, dtype="bfloat16",
        assignments=[IslandAssignment(island_id=islands[0].id, tp_size=1)],
    )
    out = predictor.predict(candidate, spec, cluster, {i.id: i for i in islands}, profiles)
    assert out.metrics is not None
    assert out.metrics.p99_ttft_ms == pytest.approx(101.0)

    predictor.set_result_hook(None)
    plain = predictor.predict(
        candidate, spec, cluster, {i.id: i for i in islands}, profiles
    )
    assert plain.metrics is not None
    assert plain.metrics.p99_ttft_ms == pytest.approx(1.0)


def test_a_per_candidate_signature_splits_two_identical_candidates(
    tmp_path, spec
) -> None:
    """D126: `EnvelopeKey` cannot see a candidate's physical boundary, so two
    placements alike in parallelism and hardware collide on one key."""
    first, second = candidate("a"), candidate("b")
    signatures = {"a": "crosses-a-busy-uplink", "b": "crosses-a-free-one"}

    base = _cache(tmp_path, spec)
    keyed = base.with_signature_of(lambda c: signatures.get(c.id))
    assert keyed.cache_key(first) != keyed.cache_key(second)

    # Without it they are the same entry, which is the collision D126 records.
    assert base.cache_key(first) == base.cache_key(second)


def test_a_signature_of_returning_none_falls_back(tmp_path, spec) -> None:
    base = _cache(tmp_path, spec)
    keyed = base.with_signature_of(lambda c: None)
    assert keyed.cache_key(candidate("a")) == base.cache_key(candidate("a"))


def test_signature_of_takes_precedence_over_graph_signature(tmp_path, spec) -> None:
    both = _cache(tmp_path, spec, graph_signature="whole-cache")
    keyed = both.with_signature_of(lambda c: "per-candidate")
    assert keyed.cache_key(candidate("a")) != both.cache_key(candidate("a"))


def test_a_keyed_sibling_shares_the_counters(tmp_path, spec) -> None:
    base = _cache(tmp_path, spec)
    sibling = base.with_signature_of(lambda c: "x")
    assert sibling.get(candidate("a")) is None
    assert base.stats() == {"hits": 0, "misses": 1}
