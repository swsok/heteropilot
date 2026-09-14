"""The per-PE RNGD profile finally has an accuracy domain, and no device made it.

`docs/HANDOVER.md` §2.2 left five E6 cells labelled `unknown` because
`profiles/accelerators/furiosa_rngd.yaml` (sim_hardware: RNGD) had no accuracy
domain at all -- they carried margin 0.00 % with nothing behind the zero. The
handover, `docs/CLAIMS.md` §2 and `docs/PAPER_OUTLINE.md` all said closing them
needed the NPU node.

They were wrong about that, and this test file is where the correction is pinned.
Per-PE and card are not two devices; they are two simulator models of ONE
physical card at TP=8 (`experiments/results/rngd_card_vs_pe_model.md` fits both
to a single real furiosa-llm run). So the measured half already existed as the
RNGD-CARD envelope, and the missing half -- the simulator run under the per-PE
fixture -- is pure simulation that runs on any node.

The load-bearing facts:

  - the error is PESSIMISTIC at every measured load, so the one-sided margin
    charges this hardware nothing and the cells it labels keep margin 0.00 %.
    That zero is now a measurement rather than the absence of one;
  - it falls with load, 63 % to 26 %, which is what a per-step overhead charged
    too often looks like as more tokens per forward amortise it;
  - the range brackets the operating points of four of the five `unknown` cells
    and NOT the fifth, and the fifth is the one that matters -- see
    `test_the_cell_carrying_d22s_retracted_headline_is_now_rejected`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planner.predictor.calibration import load_accuracy_domains

ROOT = Path(__file__).resolve().parents[1]

#: The RNGD operating points E6 recommends on the per-PE fixture, read off
#: outputs/e6_rerun/pd-rngd-gpu/pd_slo_sweep.json. Two cells share each of the
#: first two; the third is its own cell.
E6_PERPE_POINTS = (19.633850282087344, 73.67892887919689, 139.36708458689458)


@pytest.fixture(scope="module")
def rngd():
    domains = load_accuracy_domains(ROOT)
    assert "RNGD" in domains, "the per-PE domain must stay opt-in but present"
    return domains["RNGD"]


def test_it_has_nine_points_spanning_the_simulators_whole_usable_range(rngd):
    assert len(rngd.points) == 9
    assert rngd.conc_min == 1.832
    assert rngd.conc_max == 79.028


def test_the_simulator_is_pessimistic_everywhere_so_it_is_charged_nothing(rngd):
    """The margin is one-sided by construction. A model that predicts a LONGER
    TPOT than the card delivers needs no inflating, and inflating it anyway would
    reject configurations the hardware can serve."""
    assert all(p.tpot_err_pct > 0 for p in rngd.points)
    for point in rngd.points:
        assert rngd.tpot_margin_pct(point.conc) == 0.0


def test_the_error_falls_with_load(rngd):
    """63 % at served 1.8 against 26 % at 19.9. Not smoothed into one number,
    because the fall is the physical claim: a per-step cost charged too often is
    amortised by more tokens per forward."""
    assert rngd.tpot_error_at(1.832) == pytest.approx(63.16, abs=0.01)
    assert rngd.tpot_error_at(19.885) == pytest.approx(26.34, abs=0.01)
    assert rngd.tpot_error_at(1.832) > 2 * rngd.tpot_error_at(19.885)


def test_the_saturation_knee_is_kept_rather_than_smoothed(rngd):
    """Between 19.885 and 52.440 the simulated TPOT nearly doubles while the
    hardware curve rises by half. The interval is interpolated across a change of
    regime with nothing measured inside it, and flattening it would hide that."""
    assert rngd.tpot_error_at(19.885) == pytest.approx(26.34, abs=0.01)
    assert rngd.tpot_error_at(52.440) == pytest.approx(71.58, abs=0.01)


@pytest.mark.parametrize("conc", E6_PERPE_POINTS[:2])
def test_it_brackets_the_four_cells_that_only_needed_a_label(rngd, conc):
    """Two cells at 19.63 and two at 73.68. In domain, margin 0, so their plans
    do not move -- only their validity does, from `unknown` to `measured`."""
    assert rngd.in_domain(conc)
    assert rngd.tpot_margin_pct(conc) == 0.0


def test_the_cell_carrying_d22s_retracted_headline_is_now_rejected(rngd):
    """The fifth cell is the result.

    E6's per-PE winner at 10 rps is `agg[furiosa:tp8]` at 4.956 tok/J -- which
    `experiments/results/e6_rps_sweep.md` identifies as D22's retracted headline
    reproduced exactly, and which stood only because nothing could size its
    margin. Its RNGD leg runs at served concurrency 139.4, far above the 79.03
    where the simulator was last measured, so `widen_error_bars` extrapolates
    down a negative slope and charges it 42 %. 48.355 ms x 1.4212 = 68.7 ms
    against a 50 ms p99 TPOT SLO: the plan is infeasible.

    The planner now rejects it on its own calibration, which is E5's pattern
    applied to the half of E6 that E5 could not reach.
    """
    conc = E6_PERPE_POINTS[2]
    assert not rngd.in_domain(conc), "139.4 must stay outside; it is not measured"
    margin = rngd.tpot_margin_pct(conc)
    assert margin == pytest.approx(72.7858, abs=0.01)

    sweep = json.loads(
        (ROOT / "outputs/e6_rerun/pd-rngd-gpu/pd_slo_sweep.json").read_text())
    cell = next(s for s in sweep["sweep"]
                if s["rps"] == 10.0 and s["ttft_slo_max_ms"] == 64000.0)
    assert cell["recommended"]["backend_mix"] == "agg[furiosa:tp8]"
    assert cell["recommended"]["tokens_per_joule"] == pytest.approx(4.9564, abs=1e-4)
    p99 = cell["recommended"]["p99_tpot_ms"]
    assert p99 * (1 + margin / 100) > 50.0, "the margin has to bind, or this is a label"


def test_the_domain_is_still_opt_in(rngd):
    """It must not have leaked into `rngd.yaml`, which is on the default planning
    path and carries the linear fit for this same hardware key."""
    import yaml
    default = yaml.safe_load((ROOT / "profiles/calibration/rngd.yaml").read_text())
    assert "accuracy_domain" not in str(default), (
        "the domain belongs in rngd_perpe.yaml, loaded only under "
        "`plan --accuracy-domain`")


def test_it_did_not_disturb_the_card_domain(rngd):
    """Both files describe the same silicon under different abstractions, and
    their errors differ by a factor of twenty. Merging them would be the D22
    error in a new place."""
    domains = load_accuracy_domains(ROOT)
    card = domains["RNGD-CARD"]
    assert card.conc_min == 1.020 and card.conc_max == 76.0
    assert card.tpot_error_at(16.6) == pytest.approx(-3.1, abs=0.01)
    # Pessimistic per-PE against optimistic card at a concurrency both cover.
    assert rngd.tpot_error_at(16.6) > 0 > card.tpot_error_at(16.6)
