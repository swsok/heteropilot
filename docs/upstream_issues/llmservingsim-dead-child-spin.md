# Draft issue for `casys-kaist/LLMServingSim`

*Not filed yet. Written 2026-09-04 from `docs/d23_spike.md`. Verified against
upstream head `a4053bc` (2026-08-28), 69 commits past our pin `2c2042ce` — the
loop is unchanged there, so this is current.*

---

**Title:** `read_wait()` spins forever when the ASTRA-Sim child dies, and the child's stderr is discarded

**Labels:** bug

---

## Summary

If the ASTRA-Sim subprocess exits — for any reason — `python -m serving` does not
notice. It spins at 100 % CPU, grows a list without bound, and only stops when an
external timeout kills it. The child's stderr, which says what went wrong, is
captured and never read, so the failure is invisible.

The result is that a crashed simulator is indistinguishable from a slow one. In our
case that cost several days: the symptom looked like "this candidate is slow", the
only apparent remedy was a longer `--timeout`, and each longer timeout appeared to
confirm the theory. It was a crash the whole time.

## Where

**1. The spin** — `serving/core/controller.py`, `read_wait()` (line 14 at our pin;
unchanged at `a4053bc`):

```python
out = [""]
while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
    line = p.stdout.readline()
    out.append(line)
return out
```

When the child is gone, `readline()` returns `""` immediately and forever. `""`
contains neither sentinel, so the loop never exits. It is a tight busy loop, and
every iteration appends to `out`.

**2. The discarded diagnosis** — `serving/__main__.py:548` (`:630` at `a4053bc`):

```python
p = subprocess.Popen(astra_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, universal_newlines=True)
```

`stderr=PIPE` is never read anywhere in the file. Whatever ASTRA-Sim printed dies
with the pipe.

**3. No liveness check anywhere** — `__main__.py` contains no `poll()`, no
`returncode`, no `wait()`, no `terminate`.

## Observed

A run in this state, via `/proc`:

```
python -m serving      stat=Rl  %cpu=100  wchan=0
                       RSS 6.51 → 6.59 → 6.68 GB over 30 s   (the `out` list)
  └─ AnalyticalAstra   stat=Z   cpu=00:00:00   exit status 1, signal 0
ASTRA-Sim inputs root  15 files, all mtime = the launch second
                       (a run that completes writes 11,380)
```

`stat=Rl` with `wchan=0` is a busy loop, not a blocked wait, which is what makes it
look like ongoing work. The child is a zombie: it exited immediately and was never
reaped.

## Reproduction

Anything that kills the child at startup will do. Ours was concurrency: with 64
simulations running at once, ASTRA-Sim races on a fixed `tmp__mem/*.json` path and
some processes exit 1 (filed separately against `casys-kaist/astra-sim`). Four of
64 runs then hung until their 1800 s ceiling, twice, on different instances each
time.

A direct reproduction without waiting for the race: make the binary fail at
startup (e.g. point `--memory-configuration` at a file that is removed between the
frontend's write and the child's read), and watch the frontend spin.

## Suggested fix

The minimum is a liveness check in the loop:

```python
line = p.stdout.readline()
if line == "":                      # EOF: the child is gone
    rc = p.poll()
    err = p.stderr.read() if p.stderr else ""
    raise RuntimeError(
        f"ASTRA-Sim exited with code {rc} before finishing the run.\n{err}"
    )
```

Two things make this worth more than the three lines suggest:

- **Read stderr on the failure path.** The single line
  `Unable to open file: tmp__mem/remote_mem.json` would have replaced days of
  bisecting. Draining it only on EOF avoids the deadlock risk of reading it inline.
- **Raise rather than return.** Callers that run the simulator as a subprocess
  currently see a timeout and record "too slow"; a non-zero exit is a different
  outcome and should be reportable as one.

`check_end()` has the same shape and the same problem.

## Context

Found while diagnosing a P/D sweep that appeared to livelock. Full write-up,
including the ladder that showed the candidate completes in 343 s when run alone,
is in our fork at `docs/d23_spike.md`. Happy to open a PR if the shape above looks
right to you.
