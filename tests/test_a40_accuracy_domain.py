"""The A40 accuracy domain has three points, and they are not interchangeable.

`docs/HANDOVER.md` §2.2. The domain had ONE point, at served concurrency 170.56,
so `widen_error_bars` had no slope to widen along and every margin was that
point's value held flat. Two points measured on the A40 node on 2026-09-10 gave
it a slope; what they also gave it is a shape nobody could have guessed from the
single point, and these tests pin that shape because it is the result.

The load-bearing facts:

  - **TTFT error at low load is ~9x the saturated value.** Flat, the domain
    declared -1.97 % everywhere. Measured at served concurrency 4 and 11 the
    simulator is optimistic by ~18 %. At saturation TTFT is 40 s of queueing,
    which the simulator gets right; at low load it is ~150 ms of prefill, which
    it does not.
  - **TPOT error changes SIGN inside the range**, so no single number summarises
    it and the one-sided margin correctly charges nothing at the low end.
  - the range now brackets every A40 operating point E6 recommends, which is
    what turns `extrapolated` cells into `measured` ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.predictor.calibration import load_accuracy_domains

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def a40():
    domains = load_accuracy_domains(ROOT)
    assert "A40" in domains, "the A40 domain must stay opt-in but present"
    return domains["A40"]


def test_it_has_three_points_and_therefore_a_slope(a40):
    assert len(a40.points) == 3
    assert [p.conc for p in a40.points] == [4.043, 10.800, 170.56]
    # The whole complaint in §2.2: one point cannot be widened along anything.
    assert a40.conc_min == 4.043 and a40.conc_max == 170.56


def test_the_simulator_is_far_more_optimistic_on_ttft_at_low_load(a40):
    """The finding. A single point at saturation said -1.97 % and that was the
    number being reused at concurrency 4, where the truth is -18.66 %."""
    assert a40.ttft_error_at(4.043) == pytest.approx(-18.66, abs=0.01)
    assert a40.ttft_error_at(170.56) == pytest.approx(-1.97, abs=0.01)
    # Monotone in between, so the margin falls as load rises rather than jumping.
    errs = [a40.ttft_error_at(c) for c in (4.043, 10.8, 50, 100, 170.56)]
    assert errs == sorted(errs), errs
    # An order of magnitude apart at the two ends.
    assert abs(errs[0]) > 9 * abs(errs[-1])


def test_tpot_error_changes_sign_so_the_margin_is_zero_at_the_low_end(a40):
    """`margin_from_error` is one-sided: a pessimistic predictor is left alone.
    At concurrency 4 the simulator is SLOWER than the card, so it earns no TPOT
    margin -- inflating there would make plans look worse than the hardware."""
    assert a40.tpot_error_at(4.043) > 0        # pessimistic
    assert a40.tpot_error_at(170.56) < 0       # optimistic
    assert a40.tpot_margin_pct(4.043) == 0.0
    assert a40.tpot_margin_pct(170.56) == pytest.approx(1.42, abs=0.01)


@pytest.mark.parametrize("conc", [90.79, 121.81, 125.26, 138.10, 157.14, 169.86])
def test_it_brackets_every_a40_operating_point_e6_recommends(a40, conc):
    """These are the A40 served concurrencies in the committed E6 sweep. Each was
    outside a one-point domain and is inside a three-point one; that is the
    mechanism by which a switchover cell stops reading `extrapolated`."""
    assert a40.in_domain(conc)


def test_the_prefill_leg_at_0499_is_still_outside_and_must_stay_flagged(a40):
    """E6's 3.3 rps card cells rest on an A40 PREFILL leg at served concurrency
    0.499. No hardware measurement reaches an almost-idle server, so this cell
    cannot be rescued by measuring harder -- and it must keep saying so rather
    than borrowing the nearest point's number."""
    assert not a40.in_domain(0.499)
    # widen_error_bars extrapolates DOWNWARD from the low end and never caps, so
    # the margin there exceeds the one at the nearest measured point.
    assert a40.ttft_margin_pct(0.499) > a40.ttft_margin_pct(4.043)


def test_the_domain_is_still_opt_in(a40):
    """It must not have leaked into `a40.yaml`, which is on the default planning
    path: a margin applied by default changes the frozen plan output for both
    `examples/` specs."""
    import yaml
    default = yaml.safe_load((ROOT / "profiles/calibration/a40.yaml").read_text())
    blob = str(default)
    assert "accuracy_domain" not in blob, (
        "the domain belongs in a40.accuracy.yaml, loaded only under "
        "`plan --accuracy-domain`")
