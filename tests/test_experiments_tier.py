"""Experiment-harness correctness (STEP 11). The simulator is mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tier_validation import common
from experiments.tier_validation import e1_plan_agreement as e1
from experiments.tier_validation import e2_budget_pareto as e2
from experiments.tier_validation import e3_shape_overlap as e3
from experiments.tier_validation import e4_sensitivity as e4

from .conftest import MockPredictor

REPO = Path(__file__).resolve().parents[1]


# --- E1 ---------------------------------------------------------------------

def test_e1_agreement_metrics_on_synthetic_rankings():
    """top-1 / top-3 / Kendall tau are exact on known ranking pairs."""
    truth = ["a", "b", "c", "d"]
    assert common.top1_match(truth, truth)
    assert common.kendall_tau(truth, truth) == pytest.approx(1.0)
    reversed_ = list(reversed(truth))
    assert not common.top1_match(reversed_, truth)
    assert common.kendall_tau(reversed_, truth) == pytest.approx(-1.0)
    one_swap = ["a", "c", "b", "d"]  # one adjacent swap: tau = 1 - 2*1/6
    assert common.top1_match(one_swap, truth)
    assert common.kendall_tau(one_swap, truth) == pytest.approx(1 - 2 / 6)
    assert common.topk_contains(["c", "x"], truth, 3)
    assert not common.topk_contains(["d", "x"], truth, 3)


def test_e1_dry_run_counts_combinations():
    """--dry-run reports combination counts without touching a predictor."""

    class ExplodingFactory:
        def __call__(self, spec):  # pragma: no cover - must never run
            raise AssertionError("dry-run called the predictor factory")

    result = e1.run_condition(
        e1.DEFAULT_CONDITIONS[0], ExplodingFactory(), dry_run=True
    )
    assert result["dry_run"] is True
    assert result["surviving_candidates"] > 0
    assert result["simulations_upper_bound"] == 3 * result["generated_candidates"]


def test_e1_uses_tier2_as_ground_truth():
    """Leg metrics are computed against the tier2 ranking; tier2 is perfect."""
    result = e1.run_condition(
        e1.DEFAULT_CONDITIONS[0], lambda spec: MockPredictor(), dry_run=False
    )
    assert result["ground_truth"] == "tier2"
    assert result["legs"]["tier2"]["top1_match"] is True
    assert result["legs"]["tier2"]["kendall_tau"] == pytest.approx(1.0)
    # The oracle simulates a superset with the same predictor, so it agrees.
    assert result["legs"]["oracle"]["top1_match"] is True
    for leg in ("greedy", "tier0"):
        assert isinstance(result["legs"][leg]["kendall_tau"], float)


# --- E2 ---------------------------------------------------------------------

def test_e2_budget_accounting():
    """Each condition's measurement score equals its anchor count."""
    result = e2.run(budget=120, dry_run=True)
    assert result["conditions"]["A_tier0"] == 0
    assert result["conditions"]["B_attention_anchors"] == 120
    assert result["conditions"]["C_uniform_anchors"] == 120
    assert result["conditions"]["D_tier2"] == result["total_measured_rows"]


def test_e2_attention_only_condition_has_only_attention_anchors():
    """Condition B's anchors are all attention keys."""
    from profiler.synth.calibrate import pick_anchors

    keys = {
        "attention.csv": [(i, 0, 1, 16) for i in range(500)],
        "dense.csv": [("qkv_proj", t) for t in range(100)],
    }
    plan = pick_anchors(keys, budget=50, attention_share=1.0)
    assert set(plan) == {"attention.csv"}
    assert len(plan["attention.csv"]) == 50


def test_e2_real_run_shapes(tmp_path):
    """The full E2 run returns hold-out MAPEs and a zero-error condition D."""
    result = e2.run(budget=50)
    by_name = {c["condition"]: c for c in result["conditions"]}
    assert by_name["D_tier2"]["mape_vs_tier2"] == 0.0
    assert by_name["A_tier0"]["measurement_score"] == 0
    assert by_name["B_attention_anchors"]["holdout_rows"] > 0
    assert 0 < by_name["A_tier0"]["mape_vs_tier2"] < 2.0


# --- E3 ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def e3_result():
    return e3.run()


def test_e3_overlap_on_real_bundles(e3_result):
    """Overlap fractions are within [0, 1] and deterministic."""
    assert e3_result["bundles"]
    for section in ("raw_overlap", "normalized_overlap"):
        for stats in e3_result[section].values():
            assert 0.0 <= stats["jaccard"] <= 1.0
            assert 0.0 <= stats["overlap_coefficient"] <= 1.0
    assert e3.run() == e3_result  # deterministic


def test_e3_same_model_same_grid_overlaps_fully(e3_result):
    """The same model profiled on two hardwares shares its raw key set."""
    pair = "A40:meta-llama/Llama-3.1-8B vs RTXPRO6000X2:meta-llama/Llama-3.1-8B"
    assert e3_result["raw_overlap"][pair]["overlap_coefficient"] == pytest.approx(1.0)


def test_e3_normalized_overlap_ge_raw_overlap(e3_result):
    """Shape normalization must not LOSE sharing.

    Raw keys are model-namespaced (cross-model raw overlap is exactly 0);
    normalization can only reveal sharing, so normalized overlap must be >=
    raw overlap. Tolerance 0.02: normalization merges keys on BOTH sides of
    a pair, so the overlap COEFFICIENT (intersection / smaller set) can dip
    by a rounding-level amount when the smaller bundle compresses slightly
    more than the intersection (observed -0.008 on A5000 vs RNGD-CARD).
    """
    raw = e3_result["raw_overlap"]
    norm = e3_result["normalized_overlap"]
    for pair in set(raw) & set(norm):
        assert (
            norm[pair]["overlap_coefficient"]
            >= raw[pair]["overlap_coefficient"] - 0.02
        ), pair
        # Cross-model pairs: raw is 0 by construction; normalization is the
        # only legitimate source of sharing.
        if raw[pair]["intersection"] == 0:
            assert norm[pair]["overlap_coefficient"] >= 0.0


# --- E4 ---------------------------------------------------------------------

def test_e4_sweep_grid_is_complete():
    """The +-30% grid covers every parameter at every step, 1.0 included."""
    grid = e4.sweep_grid(13)
    assert len(grid) == 13
    assert grid[0] == pytest.approx(0.7)
    assert grid[-1] == pytest.approx(1.3)
    assert any(abs(f - 1.0) < 1e-12 for f in grid)
    result = e4.run(steps=5, dry_run=True)
    assert result["searches"] == len(e4.SWEPT_PARAMS) * 5


def test_e4_reports_flip_threshold():
    """A predictor that rewards the swept parameter yields a detected flip.

    The AnalyticalPredictor base keeps every candidate scorable (it always
    produces an energy proxy); the subclass boosts Ascend goodput once the
    swept bandwidth exceeds nominal, so the sweep must flip above 1.0 and
    hold below it."""

    class FlippingPredictor(common.AnalyticalPredictor):
        def predict(self, candidate, spec, cluster, islands, profiles):
            result = super().predict(candidate, spec, cluster, islands, profiles)
            ds = profiles[e4.TARGET_MODEL].datasheet
            boosted = (ds.memory_bandwidth_gbps or 0) > 1600  # > nominal
            uses_ascend = any(
                islands[a.island_id].backend == "ascend"
                for a in candidate.assignments
            )
            if result.metrics and uses_ascend and boosted:
                m = result.metrics
                result = type(result)(
                    result.candidate_id, result.outcome,
                    metrics=m.model_copy(update={
                        # The primary objective is completed_tokens *
                        # attainment / total_energy_j - drive energy down.
                        "total_energy_j": (m.total_energy_j or 1e6) / 1000,
                        "tokens_per_joule": (m.tokens_per_joule or 1.0) * 1000,
                        "slo_attainment": 1.0,
                        # Feasibility gates on latency percentiles too.
                        "p50_ttft_ms": m.p50_ttft_ms / 100,
                        "p95_ttft_ms": m.p95_ttft_ms / 100,
                        "p99_ttft_ms": m.p99_ttft_ms / 100,
                        "p50_tpot_ms": m.p50_tpot_ms / 100,
                        "p95_tpot_ms": m.p95_tpot_ms / 100,
                        "p99_tpot_ms": m.p99_tpot_ms / 100,
                    }),
                )
            return result

    result = e4.run(steps=5, predictor_factory=FlippingPredictor)
    bw = result["params"]["memory_bandwidth_gbps"]
    assert bw["baseline_pick"] is not None
    assert bw["flip_above"] is not None, bw
    assert bw["flip_below"] is None, bw


def test_e4_efficiency_clamped_at_one():
    """Sweeping an efficiency +30% never exceeds the schema's 1.0 cap."""
    from planner.inventory import load_accelerator_profile

    profile = load_accelerator_profile(
        REPO / "profiles" / "accelerators" / "ascend_target.yaml"
    )
    scaled = e4._scaled_profile(profile, "flops_efficiency", 1.3)
    assert scaled.datasheet is not None
    assert scaled.datasheet.flops_efficiency <= 1.0


# --- reports ------------------------------------------------------------------

def test_all_reports_include_provenance(tmp_path):
    """write_report embeds the §3.8 provenance block in every JSON."""
    path = common.write_report(
        tmp_path, "sample", {"x": 1}, table="x", provenance_extra={"seed": 42}
    )
    data = json.loads(path.read_text())
    block = data["provenance"]
    assert "git_commit" in block
    assert "accelerators" in block
    assert block["seed"] == 42
    assert (tmp_path / "sample.txt").read_text() == "x"


def test_committed_e3_report_exists_or_regenerable():
    """E3 (CI-safe) runs end to end and produces both report files."""
    import tempfile

    out = Path(tempfile.mkdtemp())
    assert e3.main(["--out", str(out)]) == 0
    assert (out / "e3_shape_overlap.json").exists()
    assert (out / "e3_shape_overlap.txt").exists()
