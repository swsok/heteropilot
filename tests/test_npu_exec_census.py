"""The step replay the NPU execution-model spike's STEP A is built on.

`experiments/scripts/npu_exec_step_census.py` reconstructs every scheduled step of
a finished `python -m serving` run from its INFO log, and every number in STEP A
is a count over that reconstruction. If the replay mis-classifies a step, the
whole decomposition is wrong in a way that still produces a plausible table --
so the classification, the chunk arithmetic and the two ladders are pinned here.

These run on any node: the fixtures are synthetic log text, not a device and not
a real run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/npu_exec_step_census.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("npu_exec_step_census", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution: `Step` is a dataclass and `dataclasses`
    # resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _log(*lines: str) -> str:
    return "".join(f"[00:00:00.000] {line}\n" for line in lines)


def _batch(bid: int, n: int, total_len: int, ids: list[int]) -> str:
    return (
        f"[TraceGenerator] [node=0,inst=0] INFO     Batch #{bid}: "
        f"model=meta-llama/Llama-3.1-8B num_reqs={n} total_len={total_len} "
        f"req_ids={ids}"
    )


def _iter(it: int, cycles: int) -> str:
    return (
        f"[Controller] INFO     NPU[0] iteration {it} finished, {cycles} cycles, "
        f"exposed communication 0 cycles."
    )


def _write_run(tmp_path: Path, inputs: list[int], log_text: str) -> tuple[Path, Path]:
    trace = tmp_path / "trace_t.jsonl"
    trace.write_text(
        "".join(
            json.dumps({"input_toks": n, "output_toks": 4, "arrival_time_ns": 0}) + "\n"
            for n in inputs
        )
    )
    log = tmp_path / "sim_t.log"
    log.write_text(log_text)
    return log, trace


def test_a_step_is_prefill_until_the_input_is_consumed(mod, tmp_path):
    """Two chunks of a 300-token prompt, then decode. The boundary is the whole
    classification: one step early and a decode is counted as a prefill."""
    log, trace = _write_run(
        tmp_path,
        [300],
        _log(
            _batch(0, 1, 256, [0]),
            _iter(1, 100),
            _batch(1, 1, 44, [0]),
            _iter(2, 200),
            _batch(2, 1, 1, [0]),
            _iter(3, 250),
        ),
    )
    steps, meta = mod.parse_run(log, trace, strict=True)
    assert [(s.n_prefill, s.n_decode) for s in steps] == [(1, 0), (1, 0), (0, 1)]
    assert [s.prefill_chunks for s in steps] == [[256], [44], []]
    # The decode attends over everything computed before it, not over input+1.
    assert steps[2].decode_kv == [300]
    # kv_prefill is what a chunked prefill already has resident.
    assert [s.kv_prefill for s in steps] == [0, 256, 0]
    assert meta["unbalanced_steps"] == 0


def test_cycles_are_a_first_difference(mod, tmp_path):
    """The Controller's count is cumulative; a step costs the difference."""
    log, trace = _write_run(
        tmp_path,
        [4],
        _log(
            _batch(0, 1, 4, [0]),
            _iter(1, 1000),
            _batch(1, 1, 1, [0]),
            _iter(2, 1700),
            _batch(2, 1, 1, [0]),
            _iter(3, 2300),
        ),
    )
    steps, _ = mod.parse_run(log, trace, strict=True)
    assert [s.cycles for s in steps] == [1000, 700, 600]


def test_a_mixed_step_is_recognised_and_its_chunks_balance(mod, tmp_path):
    """One request decoding while another prefills -- the shape D90 says the
    compiled grid has no bucket for."""
    log, trace = _write_run(
        tmp_path,
        [8, 200],
        _log(
            _batch(0, 1, 8, [0]),
            _iter(1, 100),
            # req 0 decodes (1 token), req 1 prefills its whole 200.
            _batch(1, 2, 201, [0, 1]),
            _iter(2, 400),
        ),
    )
    steps, meta = mod.parse_run(log, trace, strict=True)
    assert steps[0].mixed is False
    assert steps[1].mixed is True
    assert steps[1].n_prefill == 1 and steps[1].n_decode == 1
    assert steps[1].prefill_chunks == [200]
    assert steps[1].decode_kv == [8]
    assert meta["unbalanced_steps"] == 0


def test_an_unbalanced_step_is_reported_not_silently_absorbed(mod, tmp_path):
    """A total_len that the chunk reconstruction cannot spend means the replay
    has diverged from the scheduler, which must not be averaged into a result."""
    log, trace = _write_run(
        tmp_path,
        [10],
        _log(_batch(0, 1, 999, [0]), _iter(1, 100)),
    )
    steps, meta = mod.parse_run(log, trace)
    assert meta["unbalanced_steps"] == 1
    assert steps[0].balanced is False
    with pytest.raises(SystemExit, match="does not match the scheduler"):
        mod.parse_run(log, trace, strict=True)


def test_a_batch_logged_twice_is_an_error(mod, tmp_path):
    """Double-counting a batch would inflate every token total in the census."""
    log, trace = _write_run(
        tmp_path,
        [4],
        _log(_batch(0, 1, 4, [0]), _iter(1, 100), _batch(0, 1, 4, [0]), _iter(2, 200)),
    )
    with pytest.raises(SystemExit, match="logged twice"):
        mod.parse_run(log, trace)


def test_next_up_rungs_and_overflow(mod):
    ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024]
    assert mod._next_up(ladder, 1) == 1
    assert mod._next_up(ladder, 3) == 4
    # The 384 rung: the work order's "next power of two" would say 512 here.
    assert mod._next_up(ladder, 300) == 384
    assert mod._next_up(mod.POW2_LADDER, 300) == 512
    # Above the top rung the step is priced at the top rung rather than crashing.
    assert mod._next_up(ladder, 99999) == 1024


def test_the_two_kv_grids_are_counted_separately(mod, tmp_path):
    """A batch whose KVs span one artifact bucket but several 1024-token slices
    is exactly the case that tells the two candidate grids apart."""
    log, trace = _write_run(
        tmp_path,
        [1100, 1500, 1900],
        _log(
            _batch(0, 1, 1100, [0]),
            _iter(1, 10),
            _batch(1, 1, 1500, [1]),
            _iter(2, 20),
            _batch(2, 1, 1900, [2]),
            _iter(3, 30),
            _batch(3, 3, 3, [0, 1, 2]),
            _iter(4, 40),
        ),
    )
    steps, _ = mod.parse_run(log, trace, strict=True)
    decode = [s for s in steps if s.n_decode and not s.n_prefill]
    assert decode[0].decode_kv == [1100, 1500, 1900]

    ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024]
    edges = [1024, 2048, 4096, 8192]
    c = mod.census(decode, ladder, edges)
    # All three sit under the 2048 edge: one group on the artifact grid.
    assert c["kv_groups_artifact_edges"]["mean"] == 1.0
    # Floor-divided by 1024 they are 1, 1 and 1 -- also one, so raise one of them
    # past the slice boundary to show the grids genuinely disagree.
    decode[0].decode_kv = [1100, 1500, 2100]
    c2 = mod.census(decode, ladder, edges)
    assert c2["kv_groups_uniform1024"]["mean"] == 2.0
    assert c2["kv_groups_artifact_edges"]["mean"] == 2.0


def test_prefill_128_padding_is_measured_against_the_real_chunks(mod, tmp_path):
    """ceil128 of 200 is 256, so a single 200-token prefill pads by 28 %."""
    log, trace = _write_run(
        tmp_path, [200], _log(_batch(0, 1, 200, [0]), _iter(1, 100))
    )
    steps, _ = mod.parse_run(log, trace, strict=True)
    c = mod.census(steps, [1, 2, 4], [1024, 2048])
    assert c["prefill_tokens"] == 200
    assert c["prefill_pad128_frac"] == pytest.approx(256 / 200 - 1)
    assert c["prefill_bs1_frac"] == 1.0


def test_the_committed_grid_supplies_both_ladders(mod):
    """`--grid` must read the artifact's own ladders, not the fallbacks -- the
    fallbacks exist so the script runs, not so a result can quote them."""
    grid = ROOT / "profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml"
    ladder, edges, prov = mod.load_grid(grid)
    assert 384 in ladder, "the 384 rung is the whole reason the ladder is read"
    assert edges[0] == 1024 and edges[-1] == 131072
    assert "d6ae6a43" in prov

    fb_ladder, fb_edges, fb_prov = mod.load_grid(None)
    assert "fallback" in fb_prov
    assert fb_ladder == mod.FALLBACK_BATCH_LADDER and fb_edges == mod.FALLBACK_DECODE_EDGES
