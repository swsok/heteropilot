r"""Run the simulator with the trace-row `comm_type` column widened at runtime.

WORK_ORDER_spikes.md STEP B.3. A 3-dimensional topology makes the TP collective
tag `ALLREDUCE:1,1,0` -- exactly 15 characters, which is the full width of the
`comm_type` column in `serving/core/utils.py::_FMT`. With no padding left it abuts
`comm_size`, and `generate_trace` re-reads its own file with `re.findall(r'\S+')`,
so the row comes back with 10 fields instead of 11 and `formatter()` raises
`TypeError: missing 1 required positional argument: 'misc'`
(`trace_generator.py:1557`).

The bug is upstream's and pre-dates this spike; it is simply unreachable at <= 2
dims, where the tag is 13 characters. Per WORK_ORDER_spikes.md section 7 a
`serving/` bug found mid-spike is recorded, not fixed, so the widening is applied
**here at runtime** rather than committed. `serving/` on disk is untouched.

Usage:  b3_widecol_run.py -- <the usual `python -m serving` arguments>
"""
from __future__ import annotations

import runpy
import sys

WIDTHS = [30, 15, 15, 15, 15, 15, 15, 15, 24, 15, 15]   # comm_type 15 -> 24


def main() -> int:
    if "--" not in sys.argv:
        print(__doc__, file=sys.stderr)
        return 2
    args = sys.argv[sys.argv.index("--") + 1:]
    if not args:
        print("no simulator arguments given", file=sys.stderr)
        return 2

    import serving.core.utils as utils

    wide = "".join("{" + str(i) + ":<" + str(w) + "}" for i, w in enumerate(WIDTHS)) + "\n"
    utils._FMT = wide

    def header() -> str:
        names = ["Layername", "comp_time", "input_loc", "input_size", "weight_loc",
                 "weight_size", "output_loc", "output_size", "comm_type", "comm_size", "misc"]
        return "".join(f"{n:<{w}}" for n, w in zip(names, WIDTHS, strict=True)) + "\n"

    utils.header = header

    import serving.core.trace_generator as tg
    tg.formatter = utils.formatter
    tg.header = header

    print(f"[b3_widecol_run] comm_type column widened 15 -> {WIDTHS[8]} at runtime; "
          f"serving/ on disk is unmodified", file=sys.stderr, flush=True)

    sys.argv = ["serving", *args]
    runpy.run_module("serving", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
