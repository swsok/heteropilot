# Draft issue for `casys-kaist/astra-sim`

*Not filed yet. Written 2026-09-04 from `docs/d23_spike.md`. Verified against
upstream head `d346994` (2026-08-23) — the code is byte-identical to our pin
`f82fb3d`, so this is current, not something already fixed.*

---

**Title:** Analytical backend races on a fixed `tmp__mem/*.json` path, so concurrent runs kill each other

**Labels:** bug

---

## Summary

`save_json_to_tmp()` in the analytical network frontend writes its memory
configuration to a **fixed, cwd-relative** path with no pid, run id or `mkdtemp`,
reads it back, and then deletes it. Any two `AnalyticalAstra` processes sharing a
working directory therefore share that file, and one deletes it while the other is
opening it.

Running 64 processes at once on identical inputs, **13 fail**.

## Where

`astra-sim/network_frontend/analytical/congestion_unaware/main.cc:27`
(and the same helper in `congestion_aware/main.cc`, and in the ns3 frontend):

```cpp
static std::string save_json_to_tmp(const json& j, const std::string& name) {
  const char* dir = "tmp__mem";                          // fixed, relative to cwd
  if (::mkdir(dir, 0755) == -1) { ... }
  std::string path = std::string(dir) + "/" + name + ".json";
  std::ofstream ofs(path);
  if (!ofs) {
    std::cerr << "Unable to write tmp file: " << path << "\n";
    std::exit(1);
  }
  ofs << j.dump(2);
  return path;
}
```

used as:

```cpp
auto path = save_json_to_tmp(j, "remote_mem");
memory_levels.push_back(std::make_unique<AnalyticalMemory>(path));   // reads it back
std::remove(path.c_str());                                           // then deletes it
```

The race window is between the write and the read. It is entered three times per
start — `local_mem`, `remote_mem`, `cxl_mem` — so a start has three chances to lose.

## Reproduction

Two identical memory-configuration inputs are enough; we used 64 copies of one
input tree from an LLMServingSim P/D run (12 ranks, analytical backend), each with
its own `--workload-configuration` / `--system-configuration` /
`--network-configuration` / `--memory-configuration`, launched simultaneously from
one working directory:

```bash
for i in $(seq 1 64); do
  ./AnalyticalAstra \
    --workload-configuration=inputs/runs/bare-$i/workload/event_handler/llm \
    --system-configuration=inputs/runs/bare-$i/system/system.json \
    --network-configuration=inputs/runs/bare-$i/network/network.yml \
    --memory-configuration=inputs/runs/bare-$i/memory/memory_expansion.json \
    >out_$i.txt 2>err_$i.txt </dev/null &
done; wait
```

Result:

| exit | count | stderr |
| ---: | ---: | --- |
| (still running) | 51 | — started normally, waiting on stdin |
| **1** | **5** | `Unable to open file: tmp__mem/remote_mem.json` |
| **134** | **8** | `terminate called without an active exception` (SIGABRT) |

The inputs are valid: run any one of them alone and it starts normally. Failures
land on different instances on every repeat, and the count is stable at roughly
6–20 %.

Note that the `exit 1` message is **not** the one in `save_json_to_tmp` — that
path guards the *write*. This is the *read* in `AnalyticalMemory`, i.e. the file
was gone by the time it was opened.

## Why it matters downstream

Any harness that evaluates candidates in parallel hits this. LLMServingSim isolates
each run's ASTRA-Sim input tree with `--run-id` / `--inputs-root`, and that
isolation does not reach `tmp__mem/`, because the path is relative to the process's
cwd rather than to the inputs root. In our case a 252-simulation sweep at 64-way
concurrency lost candidates this way and the failures were misread as simulator
slowness for several days — see the note below.

## Suggested fix

Any of these removes the sharing; the first is the smallest:

1. Put the pid in the name — `tmp__mem/<name>.<pid>.json`.
2. Use `mkdtemp()` per process and remove the directory at exit.
3. Better, skip the file entirely: `AnalyticalMemory` is constructed from a path,
   so an overload taking the already-parsed `json` would make the temp file
   unnecessary. The value is serialized and immediately re-read by the same
   process.
4. Failing all that, derive the directory from the memory-configuration path
   (which is already per-run) rather than from the cwd.

## One more thing, if you want it

`exit(1)` on a failed open is reasonable, but the eight SIGABRTs suggest the
failure path is not always clean —`terminate called without an active exception`
points at a thread being destroyed while joinable, after the error. Worth a look
while the file is open, though it is secondary to the race.
