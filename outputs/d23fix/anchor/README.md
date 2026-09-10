# The regression anchor ladder

`docs/d23fix_baseline.md` §"The two regression anchors" describes the harness that
writes these directories; this file says which label proves which claim, and what
is and is not committed.

Every sanctioned edit to `serving/` (CLAUDE.md absolute rule 1) has to be shown
byte-identical against a fixed set of runs. The runs are:

| anchor | what it is |
| --- | --- |
| `r1_Llama-3.1-8B` | single-instance, the baseline every deviation is measured against |
| `r1_Qwen3-32B` | a second dense model |
| `r1_Qwen3-30B-A3B-Instruct-2507` | MoE, which exercises a different graph path |
| `r2_pd_a40tp4` | a P/D pair, the only anchor that crosses the router |

## The ladder

| label | commit | state it captures |
| --- | --- | --- |
| `before` | `60c6d22` | pre-D25 reference |
| `after_step1` | `60c6d22` | D25 (`cwd=run_paths.inputs_root`) applied |
| `after_step1_v2` | `60c6d22` | the same, re-run after a harness fix |
| `after_d25_d26` | `60c6d22` | D26 (`sys.executable` for the Chakra converter) added |
| `after_d27` | `2c373b9` | **D27** — that converter called in-process, 1.55-1.72x faster |
| `after_d28` | `a75e93e` | **D28** — `topology_mode: slab3d` + the `_FMT` comm_type column |

Read the ladder by comparing `SHA256SUMS` between consecutive labels. Two things
change along it, and both are the point:

**`r2_pd_a40tp4` is missing from the first three labels because it FAILED, not
because it was skipped.** `anchors.log` is where that is recorded and it is why the
logs are committed alongside the hashes: `before` ended `exit=143` after 6836 s
(killed at the ceiling), and `after_step1` and `after_step1_v2` each ended `exit=3`
twice in about 6 s — `livelock_watch.sh`'s tick stall, D23's signature. The anchor
first completed under `after_d25_d26`, at 329 s. A label whose `SHA256SUMS` has
three lines instead of four is reporting a failure; do not read it as agreement.

**`r1_Qwen3-30B-A3B-Instruct-2507` changes once**, `0b548376` -> `7d0ff3ce`,
between `after_step1_v2` and `after_d25_d26`. That change is the whole of D26 and
the post-D26 hash is the correct one — it is the committed reference in
`docs/d23fix_baseline.md`. Before D26 the MoE trace was converted by a second
`chakra` beside protobuf 6.33.1 and produced different `.et` bytes.

Everything else holds: `after_d25_d26` -> `after_d27` -> `after_d28` are
**byte-identical on all four anchors**. That is the proof D27 and D28 are
behaviour-preserving.

`outputs/d14/anchor/after_d28/` is a *different* proof and is committed separately:
R3, a colocated `tp4x2` run under `auto` against the same run under `slab3d`, where
the two CSVs must equal each other rather than a previous label.

## What is committed here

`SHA256SUMS` and `anchors.log` only. The CSVs they hash are ~2 MB each and
regenerable; a hash that fails to reproduce is the finding, not the file's absence.
The logs carry exit code, elapsed time, row count and retry count per anchor, which
is what distinguishes "identical" from "never ran".
