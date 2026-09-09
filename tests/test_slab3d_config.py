"""`topology_mode: slab3d` encodes asymmetric TP per phase (D28).

`WORK_ORDER_rps_aware.md` rev 2 STEP 2.1, formalising the `spike/d14-asym-tp`
prototype. D14/D16(b) recorded that "the simulator requires uniform instance
sizes"; the spike established that it does not -- ASTRA-Sim reads a per-collective
`involved_dim`, the Chakra converter passes an arbitrary-length one through, and
`tp_dim` is already a per-instance field. The uniformity came from exactly one
place: `_compute_network_dims` folding everything into
`[npus_per_group, num_instances]` and handing every instance the same `local_dim`.

So the properties worth pinning are:

  1. `auto` -- the default and what an absent key means -- is untouched. The
     byte-identity anchors (R1, R2, R3) are the end-to-end proof; these are the
     unit-level snapshot.
  2. `slab3d` produces `[g, 2, n_slabs]` and the per-instance `tp_dim` table from
     `WORK_ORDER_spikes.md` §1.2.
  3. **`tp_dim` is keyed on the COLLECTIVE, not the footprint.** A prefill
     instance occupies a full slab (compute + sender) but its TP group is still
     only `g` wide. Keying on width declares a 2g-rank allreduce for a g-rank TP
     group, and the prefill batch then waits forever for ranks that never join --
     which during the spike presented exactly as D23's livelock and was reported
     as "D23 reproduced" before it was found to be self-inflicted. That is the
     mistake this file exists to make impossible to repeat.
  4. Configurations the grid cannot express are refused, not silently reshaped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_config_builder():
    """Import `serving.core.config_builder` without importing `serving.core`,
    whose package import pulls in the whole simulator."""
    path = ROOT / "serving" / "core" / "config_builder.py"
    if "serving.core.config_builder" in sys.modules:
        return sys.modules["serving.core.config_builder"]
    spec = importlib.util.spec_from_file_location("serving.core.config_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = _load_config_builder()

T, F = True, False


def inst(tp: int, *, pd_type: str | None = None, ep: int = 1,
         iid: int = 0, mode: str | None = "slab3d") -> dict:
    """The subset of a resolved instance that the dim helpers read."""
    d = {
        "instance_id": iid,
        "tp_size": tp,
        "num_npus": tp,
        "pp_size": 1,
        "ep_size": ep,
        "pd_type": pd_type,
        "dp_group": None,
    }
    if mode:
        d[cb.TOPOLOGY_MODE_KEY] = mode
    return d


# --- 1. the auto path is untouched -----------------------------------------

def test_auto_is_the_default_when_the_key_is_absent():
    assert cb._topology_mode([inst(4, mode=None)]) == "auto"


def test_auto_dims_are_unchanged_for_a_uniform_pd_pair():
    """Uniform P/D stays on `auto`: prefill (full slab) + one tp4 decode (a single
    half slab) is an odd number of half slabs and does not fit the grid. §1.2 says
    so explicitly, and it is why the default path never reaches the new code."""
    pair = [inst(4, pd_type="prefill", iid=0, mode=None),
            inst(4, pd_type="decode", iid=1, mode=None)]
    assert cb._compute_network_dims(pair) == [4, 3]


def test_auto_dims_are_unchanged_for_independent_instances():
    two = [inst(4, iid=0, mode=None), inst(4, iid=1, mode=None)]
    assert cb._compute_network_dims(two) == [4, 2]


# --- 2. the slab3d dims and tp_dim table (§1.2) -----------------------------

def test_the_asymmetric_pd_case_from_the_work_order():
    """`A40 tp4 prefill + RNGD tp8 decode` = [4, 2, 2], 16 ranks, no idle rank."""
    pair = [inst(4, pd_type="prefill", iid=0), inst(8, pd_type="decode", iid=1)]
    dims, tp_dims = cb._slab3d_dims_and_tp_dims(pair)
    assert dims == [4, 2, 2]
    assert dims[0] * dims[1] * dims[2] == 16
    assert tp_dims == [[T, F, F], [T, T, F]]


def test_prefill_spans_one_dim_because_its_tp_group_is_g_wide():
    """The footprint-vs-collective distinction, asserted on its own.

    The prefill instance and the decode instance occupy the SAME number of ranks
    (2g). If tp_dim were keyed on that, both would get [T, T, F].
    """
    pair = [inst(4, pd_type="prefill", iid=0), inst(8, pd_type="decode", iid=1)]
    _, tp_dims = cb._slab3d_dims_and_tp_dims(pair)
    prefill_width = 4 * 2      # compute + sender
    decode_width = 8
    assert prefill_width == decode_width, "the premise of this test"
    assert tp_dims[0] != tp_dims[1], (
        "prefill and decode occupy equal rank counts but have different TP group "
        "sizes; keying tp_dim on the footprint is the spike's self-inflicted D23"
    )
    assert tp_dims[0] == [T, F, F]


def test_two_colocated_half_slabs_share_a_slab():
    pair = [inst(4, iid=0), inst(4, iid=1)]
    dims, tp_dims = cb._slab3d_dims_and_tp_dims(pair)
    assert dims == [4, 2]          # trailing n_slabs == 1 is dropped
    assert tp_dims == [[T, F], [T, F]]


def test_four_colocated_half_slabs_make_two_slabs():
    four = [inst(4, iid=i) for i in range(4)]
    dims, tp_dims = cb._slab3d_dims_and_tp_dims(four)
    assert dims == [4, 2, 2]
    assert all(td == [T, F, F] for td in tp_dims)


def test_a_full_slab_decode_beside_two_half_slabs():
    mixed = [inst(4, iid=0), inst(4, iid=1), inst(8, pd_type="decode", iid=2)]
    dims, tp_dims = cb._slab3d_dims_and_tp_dims(mixed)
    assert dims == [4, 2, 2]
    assert tp_dims == [[T, F, F], [T, F, F], [T, T, F]]


# --- 3. what the grid cannot express is refused -----------------------------

def test_an_odd_number_of_half_slabs_is_refused():
    with pytest.raises(ValueError, match="odd number of half-slab"):
        cb._slab3d_dims_and_tp_dims([inst(4, iid=0), inst(4, iid=1), inst(4, iid=2)])


def test_moe_is_refused_rather_than_mapped_onto_the_grid():
    with pytest.raises(ValueError, match="MoE/EP"):
        cb._slab3d_dims_and_tp_dims([inst(4, ep=2, iid=0), inst(4, ep=2, iid=1)])


def test_a_width_that_is_neither_half_nor_full_slab_is_refused():
    """tp = 4g would need idle-rank padding, which §1.2 puts out of scope."""
    with pytest.raises(ValueError, match=r"only 4 .* or 8 .* fit"):
        cb._slab3d_dims_and_tp_dims([inst(4, iid=0), inst(16, pd_type="decode", iid=1)])


def test_unaligned_half_slabs_are_refused_rather_than_reordered():
    """Ranks are handed out as consecutive blocks, so a half slab sandwiched
    between full ones straddles a slab boundary. Refusing keeps the fixture
    honest instead of silently permuting what the caller asked for."""
    bad = [inst(4, iid=0), inst(8, pd_type="decode", iid=1), inst(4, iid=2)]
    with pytest.raises(ValueError, match="not slab-aligned"):
        cb._slab3d_dims_and_tp_dims(bad)


def test_split2_is_valid_but_not_deployable():
    """`split2` came back in STEP 2.2 as a calibration instrument.

    2.1 was right to leave it out: it describes no real placement. 2.2's
    calibration is defined as "single instance, flat vs split", and splitting one
    TP group across two dims is the only way to isolate the allreduce latency term
    that slab3d changes. So it is valid for the simulator and excluded from what a
    plan may use.
    """
    assert set(cb.VALID_TOPOLOGY_MODES) == {"auto", "slab3d", "split2"}
    assert set(cb.DEPLOYABLE_TOPOLOGY_MODES) == {"auto", "slab3d"}
    assert "split2" not in cb.DEPLOYABLE_TOPOLOGY_MODES


def test_the_planner_never_emits_split2():
    """The instrument must not leak into a deployment path."""
    planner = ROOT / "planner"
    offenders = [
        f for f in planner.rglob("*.py")
        if "split2" in f.read_text()
    ]
    assert offenders == [], f"planner/ mentions split2: {offenders}"


def test_split2_splits_one_group_and_spans_both_dims():
    dims, tp_dims = cb._split2_dims_and_tp_dims([inst(8, mode="split2")])
    assert dims == [4, 2]
    assert tp_dims == [[T, T]], "the allreduce must span both dims, or it is not a split"
    dims4, tp4 = cb._split2_dims_and_tp_dims([inst(4, mode="split2")])
    assert dims4 == [2, 2] and tp4 == [[T, T]]


def test_split2_refuses_anything_but_one_colocated_instance():
    with pytest.raises(ValueError, match="ONE TP group"):
        cb._split2_dims_and_tp_dims([inst(8, mode="split2"), inst(8, iid=1, mode="split2")])
    with pytest.raises(ValueError, match="colocated instances only"):
        cb._split2_dims_and_tp_dims([inst(8, pd_type="prefill", mode="split2")])
    with pytest.raises(ValueError, match="MoE/EP"):
        cb._split2_dims_and_tp_dims([inst(8, ep=2, mode="split2")])
    with pytest.raises(ValueError, match="odd"):
        cb._split2_dims_and_tp_dims([inst(3, mode="split2")])


# --- 4. _resolve_dp_groups gives each instance its own local_dim -------------

def test_resolve_dp_groups_assigns_per_instance_tp_dim_under_slab3d():
    """The other half of the constraint. On `auto` every instance gets the same
    `local_dim`, which is what makes a mixed layout impossible."""
    pair = [inst(4, pd_type="prefill", iid=0), inst(8, pd_type="decode", iid=1)]
    cb._resolve_dp_groups(pair)
    assert pair[0]["tp_dim"] == [T, F, F]
    assert pair[1]["tp_dim"] == [T, T, F]
    assert pair[0]["dp_group_size"] == 1 and pair[1]["dp_group_size"] == 1


def test_resolve_dp_groups_is_unchanged_on_auto():
    two = [inst(4, iid=0, mode=None), inst(4, iid=1, mode=None)]
    cb._resolve_dp_groups(two)
    assert two[0]["tp_dim"] == [T, F]
    assert two[1]["tp_dim"] == [T, F]


# --- 5. the trace column must hold a 3-D tag --------------------------------

def test_comm_type_column_is_wide_enough_for_a_three_dim_tag():
    """`_FMT` pads but does not truncate, so an overlong comm_type runs into the
    next column and the whitespace-splitting reader mis-assigns every field after
    it. A 3-D tag is what overflows the original 15."""
    from serving.core.utils import formatter

    tag = "ALLREDUCE:1,1,0"          # 15 characters -- exactly the old column width
    row = formatter("Layer0", 100, "LOCAL", 1, "LOCAL", 2, "LOCAL", 3,
                    tag, 4, "NONE")
    fields = row.split()
    assert len(fields) == 11, (
        f"a {len(tag)}-character comm_type ran into the next column: {fields}"
    )
    assert fields[8] == tag
    assert fields[9] == "4", "comm_size must still be its own field"
