"""The measured power curve must carry its load and its validity (STEP 3.3).

`docs/rps_aware_planning_design.md` §3. Two rules are structural rather than
advisory, so the schema enforces them:

  A5(c) -- a power figure without the utilisation it was taken at is not a
           measurement, so every point carries `util_pct`;
  A6    -- outside the measured range is not a number, so `validity` is required
           and `refuse` is the default.

The third property is that `kind` is free text. The explanatory variable belongs
to the device and the measured range, not to the schema: ATOM's power is monotone
over a 59 pp utilisation span, while the RNGD card sits at 84.7-92.1 % throughout
and is U-shaped in concurrency (deviations D31). Forcing one form would mean
fitting a curve the data does not support.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from planner.inventory import AcceleratorProfile, PowerModel, load_accelerator_profile

ROOT = Path(__file__).resolve().parents[1]
RNGD_CARD = ROOT / "profiles/accelerators/furiosa_rngd_card.yaml"


def _model(**over) -> dict:
    d = {
        "kind": "piecewise_linear_in_served_conc",
        "idle_w": 40.0,
        "points": [{"power_w": 151.1, "util_pct": 92.1, "served_conc": 1.0}],
        "validity": {"served_conc_min": 1.0, "served_conc_max": 15.59},
    }
    d.update(over)
    return d


def test_a_point_without_utilisation_is_rejected():
    """A5(c), as a schema error rather than a review comment."""
    with pytest.raises(ValidationError):
        PowerModel.model_validate(_model(points=[{"power_w": 151.1, "served_conc": 1.0}]))


def test_validity_is_required():
    bad = _model()
    del bad["validity"]
    with pytest.raises(ValidationError):
        PowerModel.model_validate(bad)


def test_extrapolation_defaults_to_refuse():
    m = PowerModel.model_validate(_model())
    assert m.validity.extrapolation == "refuse"


def test_an_empty_point_list_is_rejected():
    with pytest.raises(ValidationError):
        PowerModel.model_validate(_model(points=[]))


def test_utilisation_outside_0_100_is_rejected():
    with pytest.raises(ValidationError):
        PowerModel.model_validate(
            _model(points=[{"power_w": 151.1, "util_pct": 140.0, "served_conc": 1.0}])
        )


def test_kind_is_free_text_because_the_device_decides_it():
    for kind in ("piecewise_linear_in_util", "piecewise_linear_in_served_conc"):
        assert PowerModel.model_validate(_model(kind=kind)).kind == kind


def test_a_profile_without_a_power_model_is_still_valid():
    """Back-compatibility: only one profile has a measured curve so far."""
    p = AcceleratorProfile.model_validate({
        "profile_id": "x", "vendor": "v", "model": "m", "backend": "cuda",
        "memory_gb": 40.0, "memory_bandwidth_gbps": 696.0,
    })
    assert p.power_model is None


def test_the_rngd_card_profile_carries_its_measured_curve():
    """The STEP 3.2 measurement, round-tripped through the loader."""
    prof = load_accelerator_profile(RNGD_CARD)
    pm = prof.power_model
    assert pm is not None, "furiosa_rngd_card.yaml should carry the measured curve"
    assert pm.kind == "piecewise_linear_in_served_conc"
    assert len(pm.points) == 5
    assert pm.validity.served_conc_max == pytest.approx(15.59)
    assert pm.validity.extrapolation == "refuse"
    assert pm.quantisation_w == pytest.approx(1.0), "furiosa-smi reports whole watts"
    # the scalar block is untouched -- the compiler still emits it
    assert prof.power is not None
    assert prof.power.active_power == pytest.approx(290.93)

    # the U: power falls then rises while utilisation only falls (D31)
    powers = [p.power_w for p in pm.points]
    utils = [p.util_pct for p in pm.points]
    assert utils == sorted(utils, reverse=True), "utilisation should fall monotonically"
    assert powers[0] > min(powers) < powers[-1], "power should be U-shaped, not monotone"
    assert max(powers) / min(powers) < 1.10, (
        "the whole point is that power barely moves across the range"
    )
