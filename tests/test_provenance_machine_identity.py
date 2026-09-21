"""An artifact must be able to say WHICH machine produced it, not just what kind.

`docs/deviations.md` D80. `accelerators()` recorded counts, and counts do not
identify a machine: two RNGD nodes with three cards each produce a byte-identical
provenance block. `hostname` does not separate them either — every node of this
project has reported `s8`, and the NPU node reports `etri-001` — so until this
existed the only thing distinguishing an A40-node artifact from an NPU-node one
was an incidental core count.

That was harmless while one machine of each kind existed. It stops being harmless
the moment a second RNGD node does: a calibration point measured there would
append to the same domain file with nothing to tell it apart, and nothing to
recover afterwards.

These tests use recorded vendor output rather than the live tools, so they pass
on a machine with none of the hardware.
"""

from __future__ import annotations

import json

import pytest

from planner.util import provenance as prov

# Real output from `furiosa-smi info --format json` on the NPU node, trimmed.
FURIOSA_JSON = json.dumps([
    {"arch": "rngd", "dev_name": "npu0", "device_uuid": "C0413086-0708-4606",
     "device_sn": "RNG26040100181Q", "firmware": "2026.3.0, 2d3f72a",
     "pci_bdf": "0000:03:00.0"},
    {"arch": "rngd", "dev_name": "npu3", "device_uuid": "89413086-0709-4707",
     "device_sn": "RNG26040100105Q", "firmware": "2026.3.0, 2d3f72a",
     "pci_bdf": "0000:45:00.0"},
])

RBLN_JSON = json.dumps({"KMD_version": "3.0.0", "devices": [
    {"npu": 0, "sid": "0000000022502229", "uuid": "d86d3e84-f99d",
     "pci": {"bus_id": "0000:83:00.0", "numa_node": "1"}},
]})

NVIDIA_L = ("GPU 0: NVIDIA A40 (UUID: GPU-aaaa1111-2222-3333-4444-555566667777)\n"
            "GPU 1: NVIDIA A40 (UUID: GPU-bbbb1111-2222-3333-4444-555566667777)\n")


def _fake_run(mapping):
    def run(cmd, cwd=None):
        return mapping.get(cmd[0])
    return run


def test_rngd_cards_are_identified_by_serial_not_by_label(monkeypatch):
    """The `npuN` label re-enumerates — the same card at `45:00.0` was `npu2` on
    2026-09-17 and `npu3` on 2026-09-18 — and a BDF moves with the slot. The
    serial does neither."""
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": FURIOSA_JSON}))
    ids = prov._rngd_ids()
    assert [d["sn"] for d in ids] == ["RNG26040100105Q", "RNG26040100181Q"]
    assert ids[0]["bdf"] == "0000:45:00.0"
    assert ids[0]["firmware"] == "2026.3.0, 2d3f72a"


def test_two_machines_with_the_same_card_count_get_different_fingerprints(monkeypatch):
    """The whole point: a count cannot tell them apart and this must."""
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": FURIOSA_JSON}))
    here = prov.accelerator_set()["fingerprint"]

    other = json.dumps([
        {"device_uuid": "X", "device_sn": "RNG99999999999Z",
         "firmware": "2026.3.0, 2d3f72a", "pci_bdf": "0000:03:00.0"},
        {"device_uuid": "Y", "device_sn": "RNG88888888888Z",
         "firmware": "2026.3.0, 2d3f72a", "pci_bdf": "0000:45:00.0"},
    ])
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": other}))
    there = prov.accelerator_set()["fingerprint"]

    assert here and there and here != there


def test_the_fingerprint_is_stable_across_reordering(monkeypatch):
    """Device order is not identity: the driver may enumerate either way."""
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": FURIOSA_JSON}))
    first = prov.accelerator_set()["fingerprint"]
    reversed_json = json.dumps(list(reversed(json.loads(FURIOSA_JSON))))
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": reversed_json}))
    assert prov.accelerator_set()["fingerprint"] == first


def test_removing_a_card_changes_the_fingerprint(monkeypatch):
    """Deliberate. It identifies the ACCELERATOR SET, and a different set is a
    different measurement configuration — the RNGD count here has gone
    4 -> 3 -> 4 -> 3."""
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": FURIOSA_JSON}))
    both = prov.accelerator_set()["fingerprint"]
    one = json.dumps(json.loads(FURIOSA_JSON)[:1])
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": one}))
    assert prov.accelerator_set()["fingerprint"] != both


def test_cuda_is_identified_by_gpu_uuid(monkeypatch):
    monkeypatch.setattr(prov, "_run", _fake_run({"nvidia-smi": NVIDIA_L}))
    assert prov._cuda_ids() == [
        "GPU-aaaa1111-2222-3333-4444-555566667777",
        "GPU-bbbb1111-2222-3333-4444-555566667777",
    ]
    assert prov.accelerator_set()["fingerprint"] is not None


def test_atom_is_identified_by_sid(monkeypatch):
    monkeypatch.setattr(prov, "_run", _fake_run({"rbln-smi": RBLN_JSON}))
    ids = prov._atom_ids()
    assert ids == [{"sid": "0000000022502229", "uuid": "d86d3e84-f99d",
                    "bdf": "0000:83:00.0"}]


def test_a_missing_vendor_tool_is_not_an_absence_of_hardware(monkeypatch):
    """Absolute rule 3 from the other side: `None` means "not detectable here",
    and must not read as "no cards"."""
    monkeypatch.setattr(prov, "_run", _fake_run({}))
    got = prov.accelerator_set()
    assert got["rngd"] is None and got["atom"] is None and got["cuda"] is None
    assert got["fingerprint"] is None, "no evidence must not hash to something"


def test_unparsable_vendor_output_does_not_raise(monkeypatch):
    """A tool that exists but prints something else is a failed probe, not a
    crash in the middle of writing an artifact."""
    monkeypatch.setattr(prov, "_run", _fake_run({"furiosa-smi": "not json",
                                                 "rbln-smi": "<html>"}))
    got = prov.accelerator_set()
    assert got["rngd"] is None and got["atom"] is None


def test_the_count_keys_are_unchanged_so_old_artifacts_stay_comparable(monkeypatch):
    """`rngd_cards: 3` appears in committed artifacts as an int. The identity is
    added beside it, never in place of it."""
    monkeypatch.setattr(prov, "_run", _fake_run({}))
    block = prov.accelerators()
    assert set(block) >= {"cuda", "rngd_cards", "atom_devices", "accelerator_set"}
    assert block["rngd_cards"] is None or isinstance(block["rngd_cards"], int)


@pytest.mark.skipif(not prov._run(["furiosa-smi", "--help"]),
                    reason="furiosa-smi not installed here")
def test_the_live_probe_agrees_with_the_vendor_tool():
    """On a machine that has the tool, what we record must be what it says."""
    ids = prov._rngd_ids()
    assert ids, "furiosa-smi is present, so this must not be empty"
    raw = json.loads(prov._run(["furiosa-smi", "info", "--format", "json"]))
    assert {d["sn"] for d in ids} == {d["device_sn"] for d in raw}
