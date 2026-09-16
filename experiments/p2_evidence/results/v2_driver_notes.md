# V2 driver — the investigation the work order asked for, and what the driver now does

*`WORK_ORDER_p2_regular_spec_evidence.md` STEP V2, the CPU half. Run 2026-09-16
on the **A40 node**; the GPUs were not used and no vLLM server was started. This
is the "작은 변경, 별도 PR" the work order names.*

## 1. 조사 필요: is there already an open-loop driver for a deployed server?

**No.** Two candidates exist and each is missing one half:

| harness | honours `arrival_time_ns`? | drives a deployed HTTP endpoint? |
| --- | --- | --- |
| `experiments/scripts/measure_envelope_openloop.py` | **yes** | **no** — it runs `python -m bench run`, an in-process `AsyncLLM` |
| `experiments/scripts/replay_to_endpoint.py` (before this PR) | **no** — "arrival timing intentionally ignored" | **yes** |
| `experiments/scripts/bench_furiosa_endpoint.py` | **no** — its docstring says it IGNORES it, fixed concurrency 64 (D19) | yes |

So the work order's fallback applies and `--open-loop` was added to
`replay_to_endpoint.py`. `bench/` was not touched — it is upstream, and absolute
rule 1 holds until Phase 5.

## 2. What the mode does

Every row fires at its own `arrival_time_ns` with **no concurrency bound**, and
the response is streamed so a first-token time exists. `--target-rps` rescales
the whole schedule by a constant factor, which preserves the shape of the
arrival process — a rescaled Poisson process is still Poisson — and divides the
rate exactly. The trace's own rate is estimated as `n / T_n`, the rate estimator
for the n-th event of a Poisson process, or stated with `--trace-rps`.

`--out` writes per-request `(due, arrival, first token, completion)` plus a
summary carrying

    L_meas = sum(residency) / window

with residency running from **arrival**, so a request waiting in the server's
queue counts. That is the disclosure's definition and it is what makes the
figure comparable with the simulator's `served_concurrency_from_sim`.

**The default is unchanged.** Without `--open-loop` the closed-loop path is the
same code it was; the diff there is a docstring and one `--help` string.

## 3. Measured against a local stub

A stub SSE server (`TTFT 50 ms`, `ITL 10 ms`) with ten synthetic arrivals at
5 rps. The client recovers the stub's own numbers, which is the check that the
streaming loop and the Little's-law summary are right:

| | stub setting | driver measured |
| --- | ---: | ---: |
| TTFT p50 | 50 ms | **51.65 ms** |
| TPOT p50 | 10 ms | **10.15 ms** |

### 3.1 The warm-up, which the tolerance forced

The work order's contract is a launch error under **20 ms**. The first run
measured **86.69 ms** — and on exactly one request:

```
idx  due_s   arrival_s  launch_err_ms
  0   0.200     0.200       0.40
  1   0.400     0.487      86.69     <-- while request 0 was connecting
  2   0.600     0.600       0.14
  ...            (everything else < 1.1 ms)
```

The client builds its httpx transport and connection pool on first use, and that
work is synchronous: it blocked the event loop and the request due during it woke
late. An open-loop driver whose first inter-arrival is wrong is offering a
different arrival process from the one it reports, so the connection is now
opened **before** `t_base` (`warm_up()`, `--no-warmup` to opt out). After it the
worst launch error is **1.30 ms**.

The warm-up probes `/v1/models` rather than sending a throwaway completion, so it
does not put a generation on the server under test. It also passes
`max_retries=0`: against a stub with no `/v1/models` the SDK's default backoff
spent **1684 ms** where one attempt costs ~237 ms.

### 3.2 One thing that is NOT fixed, and must be re-checked in V2's real run

Concurrent streamed responses make `httpcore2` print a `RuntimeError: generator
didn't stop after athrow()` traceback out of asyncgen finalisation. It is
**cosmetic here** — exit status 0, all ten requests complete, every number above
is correct — but on a measurement run it would look like a failure.

It is **not this driver's bug**: a fifteen-line script using the `openai` SDK
directly reproduces it against the same stub. Closing each stream cuts it from
~100 stderr lines to ~32, so the driver does that; closing the client as well
does not remove the rest. Switching the stub from chunked framing to HTTP/1.0
with `Connection: close` changes nothing (102 lines), so the framing is not the
cause either — it is the `httpcore2` in `.venv-vllm` (note the name: not the
usual `httpcore`).

**Whether a real vLLM server triggers it at all is unknown and must be checked
in V2's first measurement run.** If it does, the run is still valid — but the log
must be read knowing this, and it is worth checking whether `.venv-vllm`'s
`httpcore2` can be replaced.

## 4. Tests

`tests/test_replay_open_loop.py`, 15 cases. They run under `.venv`, which has no
`openai` — so the driver's import of it is now lazy and the two request
coroutines take the client as an argument, which lets a stub client stand in.
That covers scheduling, the streaming loop, the summary and the CLI without a
socket.

The HTTP path itself is **not** covered by a committed test, because a socket
test would need `openai` and would therefore always skip under `.venv`. It is
covered by §3 above, run by hand, and by V2's measurement run.

Cases worth naming: ten arrivals must each land within 20 ms of due and in
order; a request whose due time has passed fires at once and reports the lateness
rather than hiding it; ten simultaneous arrivals must complete in one request's
time, which is what proves the mode is unbounded (a semaphore would serialise
them); and the Little's-law summary is checked against a hand-computed case —
two requests resident 2 s each over a 4 s window is 1.0 served.

## 5. What is still needed before V2 can measure

1. **A40 GPUs.** All eight were busy when this was written; the measurement needs
   a deployed vLLM server.
2. The rate ladder: 2, 5, 10, 15 rps × 300 requests, staying below saturation.
3. The simulator side at the same arrival times, for `L_pred`.
4. `profiles/calibration/a40.accuracy.openloop.yaml` — a **new file**, per rule
   A3; the committed `a40.accuracy.yaml` is not extended in place.
