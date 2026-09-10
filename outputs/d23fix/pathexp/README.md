# D23's controlled pair — the `PATH` experiment

`docs/deviations.md` D26 cites this directory. Holding the input, the cluster
config, the binary and `serving/` constant and varying **only** `PATH`, two runs
each way:

| run | chakra on `PATH` | outcome |
| --- | --- | --- |
| `venv_a`, `venv_b` | `.venv/bin` (protobuf 7.36.0) | exit 0, 301 rows, 327 s |
| `local_a`, `local_b` | `~/.local/bin` (protobuf 6.33.1) | exit 3, 0 rows, ~5 s |

`SHA256SUMS` records that both venv runs produced
`fff63c22d69dd8c1491c9c6b40ebf4e9c51bdc7aff74718f220406eb69981e98`, which is the
same hash the anchor ladder carries for `r2_pd_a40tp4.csv` — the controlled pair
reproduces the committed anchor exactly.

The two 1.9 MB CSVs are not committed; their hashes above are. The `.log` files
are, because the failing side has no CSV and the log is the only record of *how*
it failed: exit 3 is `livelock_watch.sh`'s tick stall, five seconds in.

Since D27 the converter runs in-process and there is no second interpreter to
resolve, so this experiment cannot be repeated as written. It is kept as the
evidence for a closed deviation.
