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


def test_a_relative_repo_path_stays_relative(monkeypatch, tmp_path):
    """The simulator is launched with `cwd=ROOT` and prepends `../` to what it
    is given, so an absolute path becomes `..//home/...` and the run dies at
    startup before simulating anything.

    This is a regression test for a real failure: `_arg_path` compared the
    argument to ROOT without resolving it first, and a relative argument is not
    `relative_to` anything -- so every repo-local path passed on the command
    line took the absolute branch. The two A40 runs died in
    `config_builder.build_cluster_config` with
    `'..//home/swsok/heteropilot/experiments/...' not found`.
    """
    seen = []

    class _Done(Exception):
        pass

    def _capture(cmd, *args, **kwargs):
        seen.append(cmd)
        raise _Done

    monkeypatch.setattr(ll.subprocess, "run", _capture)
    # Relative, exactly as it is typed on the command line.
    argv = ["lowload_sim_error.py",
            "--cluster", "experiments/configs/clusters/rngd-card-llama31-8b-tp1.json",
            "--out", str(tmp_path / "out")]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(_Done):
            ll.main()
    finally:
        sys.argv = old
    cluster = seen[0][seen[0].index("--cluster-config") + 1]
    assert cluster == "experiments/configs/clusters/rngd-card-llama31-8b-tp1.json", cluster
    assert not cluster.startswith("/"), "an absolute path here becomes '..//...'"


def test_a_path_outside_the_repo_is_absolute_not_a_crash(tmp_path):
    """The other half: `relative_to` RAISES rather than returning something, so
    an out-dir outside the tree used to abort the run."""
    outside = tmp_path / "trace.jsonl"
    assert ll._arg_path(outside) == str(outside.resolve())
    assert ll._arg_path(ll.DEFAULT_CLUSTER) == (
        "experiments/configs/clusters/rngd-card-llama31-8b-tp1.json")


# --- served-concurrency matching -------------------------------------------
#
# The RNGD-CARD and A40 domains were built by offering the envelope point's own
# arrival rate and checking that both sides then landed at the same SERVED
# concurrency. That check is not a formality: on the per-PE RNGD fixture the
# simulator sits at served 1.86 where the hardware sits at 1.00, so every point
# fails the guard and the recipe measures nothing. `--match served` pairs against
# the measured curve interpolated at the simulator's own operating point instead,
# which is also the axis `planner/util/operating_point.py` indexes the domain by.


@pytest.fixture
def env():
    return ll.load_envelope(
        ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml")


def _raw(conc_sim, tpot_sim, conc_measured=1.00, tpot_measured=15.71):
    return {"conc_measured": conc_measured, "conc_sim": conc_sim,
            "offered_rps": 0.0967, "tpot_measured_ms": tpot_measured,
            "tpot_sim_ms": tpot_sim, "requests": 300, "rc": 0}


def test_offered_matching_is_unchanged_and_still_refuses_a_moved_operating_point(env):
    """The default must keep scoring exactly as it did, guard included."""
    near = ll.score(env, _raw(1.05, 16.5), "offered")
    assert near["comparable"] is True
    assert near["tpot_err_pct"] == pytest.approx((16.5 - 15.71) / 15.71 * 100)
    # No extra keys: a committed artifact must re-run byte-identical.
    assert "tpot_ref_ms" not in near and "match" not in near

    moved = ll.score(env, _raw(1.86, 29.04), "offered")
    assert moved["comparable"] is False, "a +86 % concurrency gap is not a pair"


def test_served_matching_compares_against_the_curve_at_the_sims_own_point(env):
    """The reference is the hardware's TPOT where the SIMULATOR ran, not where
    the offered rate was supposed to put it."""
    rec = ll.score(env, _raw(1.86, 29.04), "served")
    assert rec["comparable"] is True
    # Between the measured (1.00, 15.71) and (1.99, 18.01), log-log.
    assert 15.71 < rec["tpot_ref_ms"] < 18.01
    assert rec["tpot_err_pct"] == pytest.approx(
        (29.04 - rec["tpot_ref_ms"]) / rec["tpot_ref_ms"] * 100)
    # Smaller than the offered-matched error, because it charges the model for
    # its TPOT error alone instead of adding its throughput error to it.
    offered = ll.score(env, _raw(1.86, 29.04), "offered")
    assert rec["tpot_err_pct"] < offered["tpot_err_pct"]
    assert rec["match"] == "served"


def test_a_sim_point_outside_the_measured_range_is_refused_not_extrapolated(env):
    """`validity.extrapolation: refuse` has to reach up through the pairing. An
    invented reference TPOT is exactly what the policy exists to prevent."""
    rec = ll.score(env, _raw(0.4, 40.0), "served")
    assert rec["comparable"] is False
    assert rec["tpot_err_pct"] is None
    assert rec["tpot_ref_ms"] is None
    assert "outside the measured range" in rec["refused"]


def test_rescoring_from_raw_needs_no_simulator(tmp_path, monkeypatch, env):
    """Changing the pairing is arithmetic on facts the artifact already records.
    Re-simulating to get it would be hours of compute for the same answer, so
    --from-raw must not launch `python -m serving` at all."""
    def _boom(*args, **kwargs):
        raise AssertionError("--from-raw must not run the simulator")

    monkeypatch.setattr(ll.subprocess, "run", _boom)
    prior = tmp_path / "prior.json"
    # A failed record alongside a good one: it carries no facts to re-score and
    # dropping it would silently shorten the sweep.
    prior.write_text(json.dumps([_raw(1.86, 29.04), {"conc": 7.88, "rc": 1,
                                                     "error": "run failed"}]))
    out = tmp_path / "out"
    argv = ["lowload_sim_error.py", "--match", "served",
            "--from-raw", str(prior), "--out", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        assert ll.main() == 0
    finally:
        sys.argv = old
    got = json.loads((out / "lowload_sim_error.json").read_text())
    assert len(got) == 2
    assert got[0]["match"] == "served" and got[0]["comparable"] is True
    assert got[1] == {"conc": 7.88, "rc": 1, "error": "run failed"}
    run = json.loads((out / "run.json").read_text())
    assert run["match"] == "served" and run["rescored_from"].endswith("prior.json")


@pytest.mark.parametrize("raw,envelope", [
    ("outputs/lowload_sim_error/lowload_sim_error.json",
     "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"),
    ("outputs/a40_lowload_sim_error/lowload_sim_error.json",
     "profiles/envelopes/A40/meta-llama/Llama-3.1-8B/bf16/tp1.yaml"),
])
def test_the_committed_artifacts_rescore_byte_identically(tmp_path, raw, envelope):
    """Adding a second pairing must not move the first one's numbers.

    Both committed domains were scored by the code path `--match offered` still
    takes, so re-scoring their raw artifacts has to return the file unchanged --
    the cheap half of the proof that the refactor was a refactor. The expensive
    half, that the simulator still produces those raw records, is not re-run here;
    it is what the artifacts themselves are.
    """
    src = ROOT / raw
    out = tmp_path / "out"
    argv = ["lowload_sim_error.py", "--envelope", str(ROOT / envelope),
            "--from-raw", str(src), "--out", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        assert ll.main() == 0
    finally:
        sys.argv = old
    assert (out / "lowload_sim_error.json").read_text() == src.read_text()


# --- --compare-stat (S7.3) -------------------------------------------------
#
# `WORK_ORDER_domain_scoping.md` S7.3. Every committed domain point is fitted on
# p50 while the feasibility check applies the margin to a p99 (D101). S7.3's new
# domain fits on p99; this flag is what a later step will use to migrate the two
# committed ones, and its default must leave them alone until then.


def test_the_default_compare_stat_is_p50_so_committed_artifacts_do_not_move(env):
    """`score` keeps its old behaviour when the stat is not named."""
    explicit = ll.score(env, _raw(1.86, 29.04), "served", "p50")
    implicit = ll.score(env, _raw(1.86, 29.04), "served")
    assert explicit == implicit


def test_served_matching_can_reference_the_p99_curve(env):
    """The envelope carries `tpot_p99` on every point, so a p99 reference is a
    measurement and not an interpolation of a different statistic."""
    p50 = ll.score(env, _raw(1.86, 29.04), "served", "p50")
    p99 = ll.score(env, _raw(1.86, 29.04), "served", "p99")
    assert p99["tpot_ref_ms"] is not None
    # The p99 curve sits above the p50 curve at the same concurrency, so the same
    # simulated value is LESS optimistic against it.
    assert p99["tpot_ref_ms"] > p50["tpot_ref_ms"]
    assert p99["tpot_err_pct"] < p50["tpot_err_pct"]


def test_a_stat_the_envelope_does_not_carry_is_refused_by_name(tmp_path):
    """The refusal has to say WHICH statistic was missing.

    Not the same refusal as an out-of-range concurrency, which names the range
    instead -- a reader has to be able to tell "I have no p99 here" from "I have
    nothing here at all", because they ask for different measurements. The
    committed RNGD envelope carries `tpot_p99` on every point, so this builds one
    that does not rather than asserting on a branch the fixture cannot reach.
    """
    src = (ROOT / "profiles/envelopes/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/tp1.yaml")
    text = src.read_text()
    # Drop tpot_p99 from the two points that bracket the concurrency probed below.
    for conc in ("  1.00", "  1.99"):
        line_start = text.index(f"{{conc: {conc},")
        line_end = text.index("}", line_start)
        line = text[line_start:line_end]
        stripped = ", ".join(part for part in line.split(", ")
                             if not part.strip().startswith("tpot_p99"))
        text = text[:line_start] + stripped + text[line_end:]
    variant = tmp_path / "tp1.yaml"
    variant.write_text(text)
    env_no_p99 = ll.load_envelope(variant)

    rec = ll.score(env_no_p99, _raw(1.5, 20.0), "served", "p99")
    assert rec["tpot_err_pct"] is None
    assert "tpot_p99" in (rec.get("refused") or ""), rec.get("refused")
    # The p50 curve is still there, so the same point scores fine on p50 -- which
    # is exactly the case S7.3 handles by leaving the p99 field empty.
    assert ll.score(env_no_p99, _raw(1.5, 20.0), "served", "p50")["tpot_err_pct"] is not None


def test_the_sim_side_computes_both_percentiles(tmp_path):
    """Both are recorded whichever one is compared, so the artifact says what was
    available rather than only what was used."""
    csv = tmp_path / "sim.csv"
    csv.write_text(
        "arrival,end_time,latency,TPOT\n"
        + "".join(f"{i},{i + 10},10,{(20 + i) * 1e6}\n" for i in range(100))
    )
    m = ll.sim_metrics(csv)
    assert m["tpot_p50_ms"] is not None
    assert m["tpot_p99_ms"] is not None
    assert m["tpot_p99_ms"] > m["tpot_p50_ms"]
