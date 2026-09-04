# D23 spike — the tight-TTFT candidates do not livelock alone; ASTRA-Sim races on a shared temp file

*`WORK_ORDER_spikes.md` STEP A. Run 2026-09-04 on the NPU node, `main` = `8bb2f6f`.
Simulation only — no hardware measurement (absolute rule A1). `serving/` unchanged
(A3). Evidence: `outputs/d23/evidence/`.*

## The short answer

**D23's premise is wrong.** The candidate does not livelock. It completes alone at
every request count from 20 to 300 — **343 s at N=300**, the exact point where the
sweep burned 1800 s and then 3600 s.

What the sweep actually hit is a **race in ASTRA-Sim on a fixed, cwd-relative temp
path**, plus a frontend that cannot notice its child has died:

```
astra-sim/.../congestion_unaware/main.cc:27   const char* dir = "tmp__mem";
                                              write → read → std::remove, no pid, no run id
serving/core/controller.py:14                 read_wait() loops forever on EOF
serving/__main__.py:548                       stderr=PIPE, never read — the message is lost
```

Under concurrency one process removes the file another is about to open. The child
exits 1 (or aborts). The frontend never finds out and spins at 100 % CPU, growing
its `out` list without bound, until the timeout fires. **That is what a 4 h 37 m
sweep was made of.**

## Hypotheses, as the work order framed them

| # | hypothesis | verdict | how |
| --- | --- | --- | --- |
| H1 | environment differs from the completed run | **refuted** | same node, same `.venv`: N=300 completes in 343 s |
| H2 | `link_bw` 35.0 → 35.2 threshold bug | **refuted** | the sweep's own `cluster.json`, `link_bw` 35.2, completes |
| H3 | frontend zero-progress scheduling | **not reached** | never reproduced alone, so there was nothing to inspect |
| H4 | ASTRA-Sim COMM_SEND/RECV deadlock | **not reached** | same |
| H5 | workload generation changed | **refuted** | both traces are `e9dd5a84…`, byte-identical |
| **H6** | **concurrency** (added here; not in the work order's list) | **confirmed, with a different mechanism than D23 described** | 13 of 64 bare ASTRA-Sim processes fail when launched together |

## What was actually run

### The ladder — D23 does not reproduce alone

Same `cluster.json` copied from the sweep's own work directory (`link_bw` 35.2),
trace regenerated at seed 42 and confirmed `sha256`-identical to the sweep's, flags
compared token-by-token against the `command:` line the sweep recorded. The only
difference was `--no-cleanup-inputs`, which I added to keep the inputs.

| N | verdict | elapsed | progress ticks | CSV rows |
| ---: | --- | ---: | ---: | ---: |
| 20 | COMPLETED | — | — | 21 |
| 100 | COMPLETED | 151 s | 52 | 101 |
| 150 | COMPLETED | 254 s | 68 | 151 |
| 200 | COMPLETED | 278 s | 80 | 201 |
| 300 | COMPLETED | **343 s** | 102 | 301 |

`outputs/d23/evidence/ladder.log`.

### H6 — the same candidate, 64 at once

Sweep-faithful flags, one `--run-id` each (the isolation `CLAUDE.md` claims makes
locking unnecessary), run twice.

| | run a | run b |
| --- | --- | --- |
| COMPLETED | 60 | 60 |
| LIVELOCK | 0 | 0 |
| stuck | 4 (`TIMEOUT`, 1801 s) | 4 (`NOPROGRESS`, 903 s) |
| which instances | 2, 19, 21, 52 | 1, 5, 33, 46 |

**Different instances each time, the same count.** Not an input, not an index — a
race. And the stuck ones emit **zero** progress ticks, where D23's own evidence has
**52,903**. Whatever the sweep hit, this is not the same shape.

### The frontend is not the cause

Interrogating a stuck run through `/proc`:

```
python -m serving        stat=Rl  %cpu=100  wchan=0   RSS 6.51 → 6.59 → 6.68 GB (15 s apart)
  └─ AnalyticalAstra     stat=Z   cpu=00:00:00  exit status 1, signal 0
inputs root              15 files, all mtime = the launch second (a completed run writes 11,380)
```

Running at 100 % CPU with `wchan=0` is a **busy loop, not a wait**; the unbounded
RSS is `read_wait`'s `out` list. The child is a **zombie that exited 1 immediately**.

### The bare binary — the frontend removed entirely

64 copies of one input root, 64 `AnalyticalAstra` launched together, stderr
captured. No frontend, no planner, no Chakra:

| exit | count | stderr |
| ---: | ---: | --- |
| 124 | 51 | — (my 45 s cap: started fine, waiting on stdin) |
| **1** | **5** | `Unable to open file: tmp__mem/remote_mem.json` |
| **134** | **8** | `terminate called without an active exception` (SIGABRT) |

**13 of 64.** `outputs/d23/evidence/bare64_{exit_codes,stderr}.txt`.

## The mechanism

`astra-sim/astra-sim/network_frontend/analytical/congestion_unaware/main.cc:27`:

```cpp
static std::string save_json_to_tmp(const json& j, const std::string& name) {
  const char* dir = "tmp__mem";                        // fixed, relative to cwd
  ...
  std::string path = std::string(dir) + "/" + name + ".json";
  std::ofstream ofs(path);
  ...
}
...
auto path = save_json_to_tmp(j, "remote_mem");
memory_levels.push_back(std::make_unique<AnalyticalMemory>(path));
std::remove(path.c_str());
```

No pid, no run id, no `mkdtemp`. Every concurrent process shares one cwd
(`astra-sim/`, which the frontend chdirs into) and therefore one
`tmp__mem/remote_mem.json`. The window is between one process's write and its read;
another's `std::remove` lands inside it. Three names go through the same helper —
`local_mem`, `remote_mem`, `cxl_mem` — so there are three windows per start.

`--run-id` / `--inputs-root` isolate the *input* tree. `tmp__mem/` is outside that
tree, and no flag reaches it.

## Then the frontend hides it

```python
# serving/core/controller.py:14
out = [""]
while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
    line = p.stdout.readline()     # dead child → "" forever
    out.append(line)               # list grows without bound
```

`""` matches neither exit condition. There is no `poll()`, no `returncode` check,
no timeout — and `stderr=subprocess.PIPE` at `serving/__main__.py:548` is never
read, so the one line that would have explained everything
(`Unable to open file: tmp__mem/remote_mem.json`) dies with the pipe.

**This is why the diagnosis took a sprint.** The failure presents as "the simulator
is slow", the only visible remedy is a longer timeout, and every longer timeout
confirms the wrong theory.

## Upstream status — both unfixed

| bug | file | upstream head | fixed? |
| --- | --- | --- | --- |
| shared `tmp__mem` path | `casys-kaist/astra-sim` `congestion_unaware/main.cc:27` | `d346994` (2026-08-23), 10 commits past our pin | **no** — byte-identical |
| `read_wait` EOF spin | `casys-kaist/LLMServingSim` `controller.py:14` | `a4053bc` (2026-08-28), 69 commits past our pin | **no** — docstring added, loop unchanged |
| child stderr never read | same, `__main__.py:630` | same | **no** |
| any child liveness check | same, `__main__.py` | same | **none exists** |

astra-sim has **0 open issues**. Drafts to report both:
`docs/upstream_issues/`.

One upstream change helps us by accident: `fa6fbde` *"Run the Chakra converter
in-process instead of per-batch subprocess"* removes the other shared-cwd call
(`graph_generator.py`'s `subprocess.run(cmd, cwd=chakra)`). It does not touch
`tmp__mem`.

Two open upstream issues are adjacent and worth watching: **#23** (different NPU
counts for prefill and decode — STEP B's subject) and **#47** (P/D handoff eagerly
allocates KV on the decode worker), which touches D23's original symptom of a
prefill request holding with no KV block allocated.

## The tooling this produced

`experiments/scripts/livelock_watch.sh` + `tests/test_livelock_watch.py` (13 tests).
It ends a provably-stuck run in seconds instead of at the timeout ceiling.

Calibrated against both sides of the real evidence — the longest stalled streak is
**0 ticks** in a healthy run and **1999** in D23's log, so the threshold has no
false-positive risk on anything observed.

It also earned its own correction. The first version only judged runs that were
*talking*, so the four stuck H6 instances — which never printed a line — sailed
past it into the 1800 s ceiling, exactly the outcome it existed to prevent. It now
separates **`-g`** (no first progress line, the dead-child shape) from **`-s`**
(went quiet after starting), both reported as **exit 4**, distinct from the tick
stall's exit 3. Run b caught the same four at 903 s instead of 1801 s.

## What is still open

**D23's original symptom has not been reproduced** — 52,903 progress ticks with
prefill pinned at one running request and memory flat. Nothing here produced it:
alone the candidate completes, concurrently it dies before emitting a single tick.
The remaining untested difference is that the sweep ran 64 **different** candidates,
not 64 copies of one. That is the next experiment, and it is cheap now that the
watcher is honest.

So the tight-TTFT regime stays undetermined, and its INFEASIBLE verdicts stay
unreadable — but the reason has moved from "the candidates livelock" to "the
harness cannot tell a dead simulator from a slow one".
