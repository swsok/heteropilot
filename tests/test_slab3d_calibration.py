"""The slab3d dim-1 latency table is a table, not a law (STEP 2.2).

`WORK_ORDER_rps_aware.md` rev 2: *if the factor varies with (split, bw), write a
table, not a law*. It does vary with the split -- 4 for `[4,2]`, 2 for `[2,2]` --
so what is asserted here is that the artifact stays a bounded lookup:

  - every measured combination is present, including the spike's regression point;
  - `validity` is present and refuses everything else;
  - the corrected residual is small enough for the licence the doc claims, and the
    uncorrected error is large enough to justify correcting at all.

No simulation: this reads the committed fit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "profiles/calibration/slab3d_latency.yaml"


@pytest.fixture(scope="module")
def table() -> dict:
    return yaml.safe_load(TABLE.read_text())


MEASURED_BW = (7.7, 12.6, 16.0, 35.2, 100.0)
MEASURED_SPLITS = ("[4,2]", "[2,2]", "[1,2]")


def test_all_measured_points_are_present(table):
    got = {(p["split"], p["link_bw"]) for p in table["points"]}
    want = {(s, bw) for s in MEASURED_SPLITS for bw in MEASURED_BW}
    assert got == want


def test_the_fixture_link_bandwidths_are_covered(table):
    """Every inter-island bandwidth in `pd-rngd-gpu.yaml` that an asymmetric
    candidate can run over. Each was added because an end-to-end
    `plan --enable-pd` refused candidates for want of it: 12.6 is A40<->RNGD,
    7.7 is `fabric-rngd0-rngd1`, 35.2 is A40<->A40."""
    got = {(p["split"], p["link_bw"]) for p in table["points"]}
    for bw in (7.7, 12.6, 35.2):
        assert ("[4,2]", bw) in got and ("[2,2]", bw) in got, bw


def test_the_spike_regression_point_is_reproduced(table):
    """The spike reported factor 4 and a 24.4 % accuracy cost at [4,2] / bw 16."""
    p = next(p for p in table["points"]
             if p["split"] == "[4,2]" and p["link_bw"] == 16.0)
    assert p["factor"] == 4
    assert p["uncorrected_pct"] == pytest.approx(-24.4, abs=0.1)


def test_the_factor_is_bandwidth_independent_but_split_dependent(table):
    by_split: dict[str, set[int]] = {}
    for p in table["points"]:
        by_split.setdefault(p["split"], set()).add(p["factor"])
    assert by_split == {"[4,2]": {4}, "[2,2]": {2}, "[1,2]": {1}}, (
        "one factor per split across all bandwidths is what makes this a table "
        "indexed by split alone"
    )


def test_correcting_is_worth_it_and_works(table):
    for p in table["points"]:
        if p["factor"] == 1:
            # [1,2]: flat tp2 is 2 hops and the split is 0 + 2, so there is
            # nothing to correct and the uncorrected error is already zero.
            assert abs(p["uncorrected_pct"]) < 0.01, p
            continue
        assert abs(p["uncorrected_pct"]) > 10.0, (
            f"{p['split']} @ {p['link_bw']}: uncorrected error is only "
            f"{p['uncorrected_pct']} %, which would not justify a correction"
        )
        assert abs(p["residual_pct"]) < 0.01, (
            f"{p['split']} @ {p['link_bw']}: residual {p['residual_pct']} % is "
            f"above the 0.008 % the calibration document licenses"
        )


def test_validity_refuses_everything_not_measured(table):
    v = table["validity"]
    assert v["extrapolation"] == "refuse"
    assert set(v["splits"]) == set(MEASURED_SPLITS)
    assert set(v["link_bw_gbps"]) == set(MEASURED_BW)


def test_every_point_records_what_it_was_fitted_on(table):
    for p in table["points"]:
        assert p["fitted_on"], f"{p} has no commit"
        assert len(str(p["fitted_on"])) >= 7


def test_the_table_does_not_claim_a_general_law(table):
    """`tp/2` fits both entries and is derivable, but must not be shipped as a
    rule -- a tp16 split would need measuring, not predicting."""
    src = TABLE.read_text()
    assert "NOT A LAW" in src
    assert "refuse" in src
    for p in table["points"]:
        assert p["factor"] == p["tp"] // 2, "the two measured values do equal tp/2"
    assert "tp16" not in src and "[8,2]" not in src, (
        "an unmeasured split must not appear in the table"
    )
