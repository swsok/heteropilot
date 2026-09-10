"""The low-load sim-error driver must not be pinned to one device (§2.2).

`docs/HANDOVER.md` §2.2. `experiments/scripts/lowload_sim_error.py` measures the
simulator's error against a measured envelope at the operating points the
envelope covers, and it held its envelope, cluster config and dataset as module
constants -- which pinned it to RNGD-CARD for no reason except that it was
written there. Lifting them into arguments is what let the A40 accuracy domain get
a second point.

Two things are asserted, and the second is the one that matters:

  - the RNGD defaults did not move, so every committed invocation still runs with
    no arguments and produces the same comparison;
  - a dataset that is not the one the envelope was measured on is REFUSED. The
    offered arrival rate is computed from the envelope's own
    `output_tokens_mean`, so a different token distribution puts the two sides at
    different loads while every printed number still looks plausible. That is
    D19's failure mode in a new place: a comparison that is not apples to apples
    and does not say so.

No simulator runs here. The guard fires before the loop, which is the point of
putting it there.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/lowload_sim_error.py"


def _load():
    spec = importlib.util.spec_from_file_location("lowload_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ll = _load()


def test_the_rngd_defaults_did_not_move():
    """STEP 4.2 ran this with no arguments; that invocation must still mean the
    same thing, or the committed RNGD error figures stop being reproducible."""
    assert ll.DEFAULT_ENV == (
        ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml")
    assert ll.DEFAULT_CLUSTER == (
        ROOT / "experiments/configs/clusters/rngd-card-llama31-8b-tp1.json")
    assert ll.DEFAULT_DATASET == (
        ROOT / "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl")
    assert ll.DEFAULT_ENV.exists() and ll.DEFAULT_CLUSTER.exists()
    assert ll.DEFAULT_DATASET.exists()


def test_write_trace_respaces_the_dataset_it_is_given(tmp_path):
    """The dataset is an argument now, so the trace must come from that file and
    the spacing must be exactly 1/rps -- the sim side's offered load is nothing
    but this column."""
    src = tmp_path / "src.jsonl"
    src.write_text("".join(
        json.dumps({"input_toks": 10, "output_toks": 20,
                    "arrival_time_ns": 999, "index": i}) + "\n"
        for i in range(5)
    ))
    out = ll.write_trace(tmp_path / "trace.jsonl", src, rps=0.25, n=3)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3, "n caps the trace"
    assert [r["index"] for r in rows] == [0, 1, 2]
    # 0.25 rps = one every 4 s, and the original 999 must be overwritten, not kept.
    assert [r["arrival_time_ns"] for r in rows] == [0, 4_000_000_000, 8_000_000_000]


def test_a_dataset_the_envelope_was_not_measured_on_is_refused(tmp_path, capsys):
    """Not warned about -- refused. A wrong dataset changes the offered rate
    silently, and the resulting error figure would be a comparison of two
    different loads reported as one operating point."""
    other = ROOT / "workloads/sharegpt-qwen3-32b-300-sps10.jsonl"
    assert other.exists(), "fixture workload went missing"
    argv = ["lowload_sim_error.py",
            "--envelope", str(ll.DEFAULT_ENV),
            "--cluster", str(ll.DEFAULT_CLUSTER),
            "--dataset", str(other),
            "--out", str(tmp_path / "out")]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(SystemExit) as exc:
            ll.main()
    finally:
        sys.argv = old
    message = str(exc.value)
    assert "not the workload this envelope was measured on" in message, message
    # It must name both sides; "mismatch" alone leaves the operator guessing which
    # of the two to change.
    assert other.name in message
    assert "sharegpt-llama-3.1-8b-300-sps10.jsonl" in message


def test_the_matching_dataset_is_accepted(tmp_path, monkeypatch):
    """The guard must not be so strict that the real invocation trips it. Only
    the basename is compared, because the same workload reached by a different
    path is the same workload."""
    calls = []

    class _Done(Exception):
        pass

    def _boom(*args, **kwargs):
        calls.append(args)
        raise _Done

    # Stop at the first simulator launch: everything before it is what is under
    # test, and running `python -m serving` here would need ASTRA-Sim.
    monkeypatch.setattr(ll.subprocess, "run", _boom)
    argv = ["lowload_sim_error.py",
            "--dataset", str(ll.DEFAULT_DATASET),
            "--out", str(tmp_path / "out")]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(_Done):
            ll.main()
    finally:
        sys.argv = old
    assert calls, "the guard rejected the dataset the envelope was measured on"


def test_the_run_prefix_keeps_two_devices_from_colliding(tmp_path, monkeypatch):
    """`--run-id` names the ASTRA-Sim input root (D25). Two devices' runs at the
    same concurrency would otherwise share one, and a concurrent pair would race
    on the same tmp__mem tree."""
    seen = []

    class _Done(Exception):
        pass

    def _capture(cmd, *args, **kwargs):
        seen.append(cmd)
        raise _Done

    monkeypatch.setattr(ll.subprocess, "run", _capture)
    argv = ["lowload_sim_error.py", "--run-prefix", "a40lowload",
            "--out", str(tmp_path / "out")]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(_Done):
            ll.main()
    finally:
        sys.argv = old
    run_id = seen[0][seen[0].index("--run-id") + 1]
    assert run_id.startswith("a40lowload-"), run_id
