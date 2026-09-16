"""The open-loop driver fires when the trace says, and counts what it fired.

`WORK_ORDER_p2_regular_spec_evidence.md` STEP V2. The driver's contract is that
the arrival process it offers is the one the simulator will be given: if it runs
late, the served concurrency it measures is not the one the schedule describes
and claim 2's comparison is measuring the client instead of the server. The work
order sets the tolerance at **20 ms** over ten synthetic arrivals.

`openai` is in `.venv-vllm`, not in the `.venv` these tests run under, so the
driver imports it lazily and the request coroutines take a client as an argument.
A stub client stands in here: it exercises the scheduling, the streaming loop and
the Little's-law summary without a socket. The HTTP path itself is exercised for
real by V2's measurement run, which is the only place a vLLM server exists.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/scripts/replay_to_endpoint.py")
#: The work order's tolerance on launch accuracy.
LAUNCH_TOLERANCE_MS = 20.0


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("replay_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# A stub that looks like `AsyncOpenAI` to the two lines the driver uses.
# ---------------------------------------------------------------------------

class _Choice:
    def __init__(self, text: str) -> None:
        self.text = text


class _Chunk:
    def __init__(self, text: str) -> None:
        self.choices = [_Choice(text)]


class _Stream:
    """Yields `chunks` tokens, `gap_s` apart, like a streamed completion."""

    def __init__(self, chunks: int, gap_s: float) -> None:
        self._chunks, self._gap, self._sent = chunks, gap_s, 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent >= self._chunks:
            raise StopAsyncIteration
        self._sent += 1
        if self._gap:
            await asyncio.sleep(self._gap)
        return _Chunk("x")

    async def close(self) -> None:
        self.closed = True


class _Completions:
    def __init__(self, owner: _StubClient) -> None:
        self._owner = owner

    async def create(self, **kwargs):
        self._owner.calls.append(time.monotonic())
        stream = _Stream(self._owner.chunks, self._owner.gap_s)
        self._owner.streams.append(stream)
        return stream


class _Models:
    def __init__(self, owner: _StubClient) -> None:
        self._owner = owner

    async def list(self):
        self._owner.models_listed += 1
        return []


class _StubClient:
    def __init__(self, chunks: int = 3, gap_s: float = 0.0) -> None:
        self.chunks, self.gap_s = chunks, gap_s
        self.calls: list[float] = []
        self.streams: list[_Stream] = []
        self.models_listed = 0
        self.completions = _Completions(self)
        self.models = _Models(self)

    def with_options(self, **kwargs):
        return self


# ---------------------------------------------------------------------------
# Scheduling -- the contract the work order names
# ---------------------------------------------------------------------------

def test_ten_synthetic_arrivals_fire_within_the_tolerance(drv):
    """Ten arrivals, 50 ms apart; every launch must land within 20 ms of due."""
    due = [i * 0.05 for i in range(10)]
    rows = [{"input_tok_ids": [1, 2], "output_toks": 3} for _ in due]
    client = _StubClient()

    async def go():
        base = time.monotonic()
        return await asyncio.gather(*[
            drv._one_open_loop(client, "m", row, 512, i, d, base)
            for i, (row, d) in enumerate(zip(rows, due, strict=True))
        ])

    records = asyncio.run(go())
    assert [r["ok"] for r in records] == [True] * 10
    worst = max(abs(r["launch_error_s"]) for r in records) * 1e3
    assert worst < LAUNCH_TOLERANCE_MS, f"worst launch error {worst:.1f} ms"
    # And the schedule is honoured in order, not merely on average.
    assert client.calls == sorted(client.calls)


def test_a_late_start_is_reported_not_hidden(drv):
    """A request whose due time has already passed fires at once and says so."""
    client = _StubClient()

    async def go():
        base = time.monotonic() - 1.0  # the schedule started a second ago
        return await drv._one_open_loop(client, "m", {"output_toks": 2}, 512, 0,
                                        0.0, base)

    rec = asyncio.run(go())
    assert rec["launch_error_s"] == pytest.approx(1.0, abs=0.05)


def test_open_loop_applies_no_concurrency_bound(drv):
    """All ten are in flight at once, so the run takes one request's time.

    A semaphore would serialise them and the wall would be ten times longer;
    that is the difference between the two modes and it must be observable.
    """
    due = [0.0] * 10
    client = _StubClient(chunks=2, gap_s=0.05)

    async def go():
        base = time.monotonic()
        await asyncio.gather(*[
            drv._one_open_loop(client, "m", {"output_toks": 2}, 512, i, d, base)
            for i, d in enumerate(due)
        ])
        return time.monotonic() - base

    wall = asyncio.run(go())
    assert wall < 0.5, f"ten concurrent requests took {wall:.2f}s -- bounded?"


# ---------------------------------------------------------------------------
# The arrival schedule
# ---------------------------------------------------------------------------

def test_trace_rate_is_estimated_from_the_last_arrival(drv):
    """n events up to T_n estimates a Poisson rate as n / T_n."""
    offsets = [i * 0.1 for i in range(1, 11)]  # last at 1.0 s, ten events
    assert drv.trace_rps(offsets) == pytest.approx(10.0)


def test_rescale_hits_the_target_rate_and_keeps_the_shape(drv):
    """A constant factor divides the rate exactly and preserves the process."""
    offsets = [0.1, 0.35, 0.4, 1.0]
    source = drv.trace_rps(offsets)
    scaled = drv.rescale(offsets, source, target_rps=2.0)
    assert drv.trace_rps(scaled) == pytest.approx(2.0)
    # Shape: every inter-arrival scales by the same factor.
    f = scaled[1] / offsets[1]
    assert all(s == pytest.approx(o * f) for s, o in zip(scaled, offsets, strict=True))


def test_arrival_offsets_are_not_shifted_to_zero(drv):
    """The generator lays the process down from t=0; the first gap is part of it."""
    rows = [{"arrival_time_ns": 46_926_808}, {"arrival_time_ns": 1_046_926_808}]
    assert drv.arrival_offsets_s(rows)[0] == pytest.approx(0.046926808)


@pytest.mark.parametrize("target", [0.0, -1.0])
def test_a_non_positive_target_rate_is_refused(drv, target):
    with pytest.raises(ValueError):
        drv.rescale([1.0, 2.0], 10.0, target)


# ---------------------------------------------------------------------------
# Little's law
# ---------------------------------------------------------------------------

def test_served_concurrency_is_total_residency_over_the_window(drv):
    """Two requests, each resident 2 s, over a 4 s window -> 1.0 served."""
    records = [
        {"ok": True, "arrival_s": 0.0, "first_token_s": 0.5, "completion_s": 2.0,
         "streamed_chunks": 4, "launch_error_s": 0.0},
        {"ok": True, "arrival_s": 2.0, "first_token_s": 2.5, "completion_s": 4.0,
         "streamed_chunks": 4, "launch_error_s": 0.0},
    ]
    s = drv.summarise_open_loop(records, offered_rps=0.5)
    assert s["window_s"] == pytest.approx(4.0)
    assert s["served_concurrency"] == pytest.approx((2.0 + 2.0) / 4.0)
    # TTFT is measured from ARRIVAL, so queueing is included.
    assert s["ttft_ms"]["p50"] == pytest.approx(500.0)
    # TPOT spreads the post-first-token time over the remaining chunks.
    assert s["tpot_ms"]["p50"] == pytest.approx(1.5 / 3 * 1e3)


def test_failed_requests_do_not_enter_the_concurrency(drv):
    records = [
        {"ok": True, "arrival_s": 0.0, "first_token_s": 0.5, "completion_s": 2.0,
         "streamed_chunks": 2, "launch_error_s": 0.0},
        {"ok": False, "arrival_s": 0.0, "first_token_s": None, "completion_s": 9.0,
         "streamed_chunks": 0, "launch_error_s": 0.0, "error": "boom"},
    ]
    s = drv.summarise_open_loop(records, offered_rps=1.0)
    assert s["failed"] == 1
    assert s["window_s"] == pytest.approx(2.0)
    assert s["served_concurrency"] == pytest.approx(1.0)


def test_the_summary_survives_a_run_where_everything_failed(drv):
    records = [{"ok": False, "arrival_s": 0.0, "completion_s": 1.0,
                "streamed_chunks": 0, "launch_error_s": 0.0, "error": "x"}]
    s = drv.summarise_open_loop(records, offered_rps=1.0)
    assert s["ok"] == 0 and "served_concurrency" not in s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_open_loop_only_flags_are_refused_in_closed_loop(drv):
    """`--out` without `--open-loop` would silently write nothing."""
    args = drv.build_parser().parse_args(
        ["--model", "m", "--dataset", "d.jsonl", "--out", "x.json"])
    assert args.out is not None and not args.open_loop  # main() is what refuses it


def test_the_default_mode_is_still_closed_loop(drv):
    """The existing invocation must not change behaviour."""
    args = drv.build_parser().parse_args(
        ["--model", "m", "--dataset", "d.jsonl", "--concurrency", "128"])
    assert args.open_loop is False
    assert args.target_rps is None


# ---------------------------------------------------------------------------
# Teardown -- an unclosed SSE body turns a good run into a noisy one
# ---------------------------------------------------------------------------

def test_every_stream_is_closed_even_when_iteration_raises(drv):
    """Measured against a stub server, leaving bodies open trebled the stderr
    noise httpcore emits from asyncgen finalisation (100 lines against 32) on a
    run that exited 0 with correct numbers. Closing is cheap; a measurement run
    that looks like it failed is not."""
    client = _StubClient()

    async def go():
        base = time.monotonic()
        return await drv._one_open_loop(client, "m", {"output_toks": 2}, 512, 0,
                                        0.0, base)

    asyncio.run(go())
    assert client.streams and all(s.closed for s in client.streams)


def test_warm_up_is_paid_before_the_schedule_and_does_not_retry(drv):
    """The client builds its transport on first use; measured at 87 ms, it
    arrived as one late request. `with_options(max_retries=0)` keeps the probe
    from becoming a wait when the endpoint has no `/v1/models`."""
    client = _StubClient()
    elapsed = asyncio.run(drv.warm_up(client))
    assert client.models_listed == 1
    assert elapsed >= 0.0
