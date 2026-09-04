# LLMServingSim — a 3-D collective tag overflows the trace column and breaks the round-trip

*Draft, not filed. Repo: `casys-kaist/LLMServingSim`. Found 2026-09-04 against our
pin `2c2042ce`; the relevant code is unchanged at head `a4053bc`.*

## Summary

`generate_trace` writes each trace row with a fixed-width formatter and then
**re-reads its own file with whitespace splitting**. A collective tag for a
3-dimensional topology is exactly as wide as its column, so it abuts the next
field, the row comes back with 10 values instead of 11, and the run dies with

```
TypeError: formatter() missing 1 required positional argument: 'misc'
  serving/core/trace_generator.py:1557
```

## Where

`serving/core/utils.py` gives `comm_type` a 15-character left-aligned column with
no separator:

```python
ileft_list = [30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
#                                             ^^ comm_type
```

`serving/core/trace_generator.py:978` encodes the involved dimensions into the tag:

```python
def _with_dim(comm_type, involved_dim):
    """Encode involved_dim into comm_type string: 'ALLREDUCE' + [T,F] -> 'ALLREDUCE:1,0'."""
    dim_str = ','.join('1' if d else '0' for d in involved_dim)
    return f"{comm_type}:{dim_str}"
```

| dims | tag | chars | 15-wide column |
| ---: | --- | ---: | --- |
| 2 | `ALLREDUCE:1,0` | 13 | fits (2 spaces of padding) |
| **3** | `ALLREDUCE:1,1,0` | **15** | **zero padding — abuts `comm_size`** |
| 4 | `ALLREDUCE:1,1,0,0` | 17 | overflows |

The row is then re-parsed at `trace_generator.py:1514`:

```python
with open(output_path, 'r') as f:
    dic = []
    for line in f.readlines():
        split = re.findall(r'\S+', line)
        dic.append(split)
```

so `ALLREDUCE:1,1,0` + `7503872` becomes the single token
`ALLREDUCE:1,1,07503872`, and `formatter(new_string, *result[i][1:])` at line 1557
is called with 10 arguments.

## Reproducing

Any topology with three or more dimensions and TP > 1. ASTRA-Sim accepts these —
`astra-sim/inputs/network/Ring_FullyConnected_Switch.yml` ships `npus_count:
[2,8,4]` — so the trace writer is the only thing that does not. Observed with a
2-node cluster whose network config is `npus_count: [4, 2, 2]`; the first
`o_proj` row of the first batch raises.

Diagnostic (no source change), which prints the offending row instead of raising:

```python
import serving.core.trace_generator as tg
_real = tg.formatter
def probe(*args):
    if len(args) != 11:
        print("SHORT ROW:", args)
        args = tuple(args) + ("NONE",) * (11 - len(args))
    return _real(*args)
tg.formatter = probe
```

```
SHORT ROW (10 fields): ('o_proj_5', '92076', 'LOCAL', '1875968', 'LOCAL', '8388608',
                        'LOCAL', '7503872', 'ALLREDUCE:1,1,07503872', 'NONE')
```

## Suggested fix

Widening the `comm_type` column removes this instance and is a one-line change,
but the class of bug is the round-trip: a fixed-width writer paired with a
whitespace-splitting reader will break again for any long field (a longer
collective name, a deeper topology, a location string with a wide index).

The narrower fix is to keep the rows in memory rather than re-parsing the file
`_synthesize_trace` just wrote — the values are already structured at that point,
and the parse exists only to bolt on the `kv_load` / `kv_evict` rows and the layer
index. Failing that, a delimiter (or `str.split('\t')`) would make the column
width irrelevant.

## Note

We hit this while prototyping non-uniform instance sizes, which need a third
topology dimension. It is not specific to that: any 3-D topology with tensor
parallelism reaches it.
