"""Simulator-level P/D KV-transfer model, router side (Phase 5, deviations D15).

`serving/core/scheduler.py` is left pristine per work order section 7, so the
bandwidth-sensitive prefill->decode handoff lives in `serving/core/router.py`
(a deferred-transfer queue) and the main loop (which drains it). These tests pin
the router half without a full simulation:

- default (bw None) hands off immediately and enqueues nothing -> byte-identical;
- bandwidth mode withholds the request and enqueues it at
  current + link_latency + KV_bytes / link_bw;
- pop_ready_transfers / get_next_transfer_ready_time respect the clock;
- lower bandwidth => later ready time (the monotonicity the sweep observes).

The end-to-end monotonicity + byte-identical defaults are verified separately by
experiments/scripts/pd_sim_network_sweep.py against the real simulator.
"""

from __future__ import annotations

from serving.core.router import Router


class _FakeMemory:
    def __init__(self, kv_bytes):
        self._kv = kv_bytes

    def get_total_kv(self, req):
        return self._kv


class _FakeScheduler:
    def __init__(self, pd_type, kv_bytes=0):
        self.pd_type = pd_type
        self.memory = _FakeMemory(kv_bytes)
        self.enable_prefix_caching = False
        self.added = []  # requests handed off via add_decode

    def add_decode(self, req):
        self.added.append(req)


class _Req:
    def __init__(self, rid):
        self.id = rid


def _router(kv_bytes, bw, latency_ns=0):
    prefill = _FakeScheduler("prefill")
    decode = _FakeScheduler("decode", kv_bytes=kv_bytes)
    r = Router(2, [prefill, decode], req_num=0, routing_policy="RR",
               pd_transfer_bw_gbps=bw, pd_transfer_latency_ns=latency_ns)
    return r, decode


def test_default_hands_off_immediately_and_enqueues_nothing():
    r, decode = _router(kv_bytes=2_000_000, bw=None)
    req = _Req(0)
    r.transfer_prefill_request([req], current=1000)
    assert decode.added == [req]           # immediate add_decode
    assert not r.has_pending_transfers()   # nothing deferred
    assert r.pop_ready_transfers(10**18) == []


def test_bandwidth_mode_withholds_and_enqueues_at_correct_time():
    # 2 MB over 16 GB/s = 2e6 / 16 = 125000 ns, plus 20000 ns link latency.
    r, decode = _router(kv_bytes=2_000_000, bw=16.0, latency_ns=20_000)
    req = _Req(0)
    r.transfer_prefill_request([req], current=1_000)
    assert decode.added == []              # NOT handed off yet
    assert r.has_pending_transfers()
    expected_ready = 1_000 + 20_000 + 2_000_000 / 16.0
    assert r.get_next_transfer_ready_time(0) == expected_ready


def test_lower_bandwidth_gives_later_ready_time():
    ready = {}
    for bw in (64.0, 16.0, 4.0):
        r, _ = _router(kv_bytes=2_000_000, bw=bw)
        r.transfer_prefill_request([_Req(0)], current=0)
        ready[bw] = r.get_next_transfer_ready_time(-1)
    assert ready[64.0] < ready[16.0] < ready[4.0]


def test_pop_ready_transfers_respects_clock():
    r, _decode = _router(kv_bytes=1_600, bw=16.0)  # 1600/16 = 100 ns transfer
    r.transfer_prefill_request([_Req(0)], current=0)      # ready at 100
    r.transfer_prefill_request([_Req(1)], current=1_000)  # ready at 1100
    # Before the first completes: nothing pops, clock-advance points at 100.
    assert r.pop_ready_transfers(50) == []
    assert r.get_next_transfer_ready_time(50) == 100
    # At 100 the first is admitted, the second is still in flight.
    popped = r.pop_ready_transfers(100)
    assert [req.id for req, _ in popped] == [0]
    assert r.has_pending_transfers()
    assert r.get_next_transfer_ready_time(100) == 1_100
    # Past the second's ready time it drains and the queue empties.
    popped = r.pop_ready_transfers(2_000)
    assert [req.id for req, _ in popped] == [1]
    assert not r.has_pending_transfers()
    assert r.get_next_transfer_ready_time(2_000) is None


def test_pop_returns_decode_index_for_add_decode():
    r, decode = _router(kv_bytes=1_600, bw=16.0)
    r.transfer_prefill_request([_Req(7)], current=0)
    (req, idx), = r.pop_ready_transfers(10**9)
    # The index must address the decode scheduler the main loop will call.
    assert r.decode_schedulers[idx] is decode
    assert req.id == 7
