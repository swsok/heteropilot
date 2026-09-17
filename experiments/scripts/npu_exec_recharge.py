"""Re-price a simulated run's steps under the rules an AOT bucket executor obeys.

`WORK_ORDER_npu_exec_model_spike.md` STEP A.3 (E-N3). Post-processing only: the
steps come from a run that already happened (STEP A.1's census replay), and the
prices come from the same perf DB the simulator used, imported **read-only** from
`serving.core.trace_generator`. Nothing in `serving/` or `planner/` is touched and
the simulator is not run again.

**The cost model, and why it can be checked.** At `tp_size = 1` a step's simulated
cost is a sum of table lookups with no collectives in it, so it can be rebuilt
from the architecture's own layer sequence:

    embedding(T)
    + L x [ layernorm(T) + qkv_proj(T) + rotary_emb(T) + attention(...)
            + o_proj(T) + layernorm(T)
            + gate_up_proj(T) + act_fn(T) + down_proj(T) ]
    + final_layernorm(T) + lm_head(S) + sampler(S)

with `T` the step's token count and `S` its sequence count. The run's own log
records ASTRA-Sim's cumulative cycle count, whose first difference is what that
step actually cost, so `--check` prints the model against the simulator step by
step. A rule's contribution is only worth reading if that agreement is tight;
the number is printed rather than assumed.

**The rules.** Each is applied alone, against the same baseline, because the work
order asks for a per-mechanism contribution and the rules do not commute.

* **R-pad** -- a decode step runs at the next compiled batch size, so every dense
  and per-sequence lookup is re-priced at the padded count instead of the real
  one. Reported on the artifact's own ladder (which has a 384 rung) and on powers
  of two, because the work order assumed the latter.
* **R-attn** -- the *residual* KV diversity, and not the grouping itself.

  The grouping is already in the table. `meta.yaml` says each decode row's time
  is "total decode-attention device time over (forwards x 32)", measured on
  sharegpt at that concurrency, so the 1.95-to-3.08 attention executions per layer
  D17 measured are baked into the number the simulator looks up. Replacing that
  lookup with a sum over groups -- the obvious reading of the work order's R-attn
  -- multiplies the grouping in a second time. That miscomputation is what this
  script did first, and it reported +17.3 pp at c15 where the correct answer is
  far smaller; the number is recorded in `docs/npu_exec_spike.md` rather than
  deleted, because the size of it is what makes the double count easy to make.

  What the simulator can actually be wrong about is the *diversity of this run's
  batches versus the diversity of the sharegpt batches the row was measured on*.
  So the step's attention is scaled by

      groups(this step, from the census) / groups(D17's curve at this batch size)

  which is 1.0 exactly when the run's batches are as ragged as the measured ones,
  and which is the only part of the mechanism that survives off-workload. It is
  **provisional** until STEP C.3 measures a clean per-group cost: D17's curve is
  five points against mean batch size, and it is being read as if diversity were
  a function of batch size alone.
* **R-128** -- a prefill chunk runs on the next multiple of 128, so prefill steps
  are re-priced at the padded chunk, in the dense lookups and in attention.
* **R-c1** -- the control. Batch-1 decode is the one place the runtime runs a
  fused `composed` plan (D90), and the prototype leaves it alone (STEP B.1 P4),
  so this rule changes nothing and must come back at 0.0 pp.

**What this cannot see.** Scheduling feedback. Re-pricing a step does not change
which requests the scheduler would have put in the next one, so a rule that makes
steps longer does not get the larger batches that would follow. Every number here
is a first-order contribution; STEP B's prototype is what closes that loop.

Usage::

    python experiments/scripts/npu_exec_recharge.py outputs/npu_spike/a1_current \\
        --grid profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml \\
        --hardware RNGD-CARD --model meta-llama/Llama-3.1-8B --variant bf16 --tp 1 \\
        --check --markdown experiments/results/npu_exec_recharge.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serving.core import trace_generator as tg  # noqa: E402

_CENSUS_PATH = Path(__file__).resolve().parent / "npu_exec_step_census.py"


def _load_census_module():
    """Import the census replay by path, so both scripts stay runnable directly.

    The module must be registered in `sys.modules` BEFORE it executes: its `Step`
    is a dataclass, and `dataclasses` resolves annotations through
    `sys.modules[cls.__module__]`, which is None for a module that is only half
    imported. The failure is an `AttributeError` on `NoneType.__dict__`, which
    says nothing about the cause.
    """
    spec = importlib.util.spec_from_file_location("npu_exec_step_census", _CENSUS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


census_mod = _load_census_module()
Step = census_mod.Step
parse_run = census_mod.parse_run
load_grid = census_mod.load_grid
_next_up = census_mod._next_up
POW2 = census_mod.POW2_LADDER


def _ceil128(x: int) -> int:
    return -(-x // 128) * 128


def _load_db_from_repo_root(hardware: str, model: str, variant: str, tp: int, model_type: str):
    """Load the perf DB the way the simulator does, from where it expects to be.

    `trace_generator._variant_root` builds `../profiler/perf/...`, a path relative
    to the CWD, because `python -m serving` runs from inside the tree. Called from
    anywhere else it resolves to a directory that does not exist, and the error
    names a missing profile rather than a wrong CWD. Borrowing the simulator's own
    CWD for the duration of the read keeps the lookup byte-for-byte the one the
    run used; the tables are cached afterwards, so nothing later needs it.
    """
    here = os.getcwd()
    try:
        os.chdir(ROOT / "serving")
        return tg._load_perf_db(hardware, model, variant, {tp}, model_type)
    finally:
        os.chdir(here)


def _tag_order(path: Path) -> tuple:
    """Sort runs by the concurrency in their tag, not by its spelling.

    `sorted(glob(...))` puts c15p3 before c1p0, which makes a table that reads as
    if the trend ran backwards.
    """
    tag = path.stem.split("_", 1)[-1]
    try:
        return (0, float(tag.lstrip("c").replace("p", ".")))
    except ValueError:
        return (1, 0.0)


class Pricer:
    """The simulator's own lookups, wrapped so a step can be priced twice."""

    def __init__(self, hardware: str, model: str, variant: str, tp: int, model_type: str):
        self.db = _load_db_from_repo_root(hardware, model, variant, tp, model_type)
        self.tp = tp
        seq = self.db["architecture"]["sequence"]
        self.prologue = [n for n in seq.get("prologue", []) if n != "attention"]
        self.block = [
            n
            for n in (
                list(seq.get("pre_attn", []))
                + list(seq.get("post_attn", []))
                + list(seq.get("mlp_dense", []))
            )
            if n != "attention"
        ]
        self.head_dense = [
            n for n in seq.get("head", []) if tg._layer_category(self.db, n) == "dense"
        ]
        self.head_per_seq = [
            n for n in seq.get("head", []) if tg._layer_category(self.db, n) == "per_sequence"
        ]

    def dense(self, name: str, tokens: int) -> float:
        return tg._lookup_dense(self.db, name, self.tp, max(1, tokens))

    def per_seq(self, name: str, sequences: int) -> float:
        return tg._lookup_per_sequence(self.db, name, self.tp, max(1, sequences))

    def attention(self, pc: int, kvp: int, nd: int,
                  kv_mean: int, kv_max: int, kv_min: int) -> float:
        if pc == 0 and nd == 0:
            return 0.0
        return tg._lookup_attention_with_skew(
            self.db, self.tp, pc, kvp, nd, kv_mean, kv_max, kv_min
        )

    def step_cost(self, layers: int, tokens: int, sequences: int, attn_ns: float) -> float:
        per_block = sum(self.dense(n, tokens) for n in self.block)
        return (
            sum(self.dense(n, tokens) for n in self.prologue)
            + layers * (per_block + attn_ns)
            + sum(self.dense(n, tokens) for n in self.head_dense)
            + sum(self.per_seq(n, sequences) for n in self.head_per_seq)
        )


def _attn_args(step: Step) -> tuple[int, int, int, int, int, int]:
    """(prefill_chunk, kv_prefill, n_decode, kv_mean, kv_max, kv_min) as the
    simulator builds them in `_build_batch_ctx`."""
    pc = sum(step.prefill_chunks)
    # kv_prefill is the KV already resident for a chunked prefill; the replay
    # tracks it as the tokens computed before this step's chunk.
    kvp = step.kv_prefill
    nd = step.n_decode
    if nd:
        kv_mean = sum(step.decode_kv) // nd
        return pc, kvp, nd, kv_mean, max(step.decode_kv), min(step.decode_kv)
    return pc, kvp, 0, 0, 0, 0


def price_baseline(p: Pricer, layers: int, step: Step) -> float:
    pc, kvp, nd, kvm, kvx, kvn = _attn_args(step)
    tokens = pc + nd
    return p.step_cost(layers, tokens, step.sequences, p.attention(pc, kvp, nd, kvm, kvx, kvn))


def price_rule(p: Pricer, layers: int, step: Step, rule: str, ladder, edges,
               decode_edges=None) -> float:
    pc, kvp, nd, kvm, kvx, kvn = _attn_args(step)
    tokens = pc + nd
    sequences = step.sequences

    if rule in ("R-pad", "R-pad-pow2") and nd and not pc:
        lad = ladder if rule == "R-pad" else POW2
        padded = _next_up(lad, nd)
        tokens, sequences = padded, padded
        # The padded lanes are real decodes as far as the kernel is concerned,
        # but they carry no KV, so attention is left at the served count.
    elif rule.startswith("R-attn") and nd:
        grid = decode_edges if rule == "R-attn-coarse" else edges
        buckets = {_next_up(grid, max(kv, 1)) for kv in step.decode_kv}
        if rule == "R-attn-naive":
            # The double count, kept so its size can be quoted: it re-groups a
            # table entry that is already a per-layer total over the groups.
            groups: dict[int, list[int]] = {}
            for kv in step.decode_kv:
                groups.setdefault(_next_up(grid, max(kv, 1)), []).append(kv)
            attn = sum(
                p.attention(0, 0, len(kvs), bucket, bucket, bucket)
                for bucket, kvs in groups.items()
            )
            if pc:
                attn += p.attention(pc, kvp, 0, 0, 0, 0)
            return p.step_cost(layers, tokens, sequences, attn)
        # Since C.3 the correction is a measured cost model rather than a ratio
        # of group counts. One decode attention execution costs a large FIXED
        # amount plus a modest amount per sequence it covers, so re-grouping a
        # batch changes only the fixed term -- the per-sequence work is done
        # either way. Scaling the whole lookup by g_actual/g_table, which is what
        # this rule did before C.3, wrongly re-scales the per-sequence term too
        # and overstates the correction by about a third.
        g_table = _d17_groups(nd)
        base = ATTN_FIXED_US * g_table + ATTN_SLOPE_US * nd
        actual = ATTN_FIXED_US * len(buckets) + ATTN_SLOPE_US * nd
        attn = p.attention(pc, kvp, nd, kvm, kvx, kvn) * (actual / base)
        return p.step_cost(layers, tokens, sequences, attn)
    elif rule == "R-128" and pc:
        pc = _ceil128(pc)
        tokens = pc + nd
    elif rule == "R-c1":
        pass

    return p.step_cost(layers, tokens, sequences, p.attention(pc, kvp, nd, kvm, kvx, kvn))


#: The per-execution decode attention cost, C.3 (E-N8). Two independent routes:
#: the committed npu0 bundle's per-layer totals divided by D17's executions per
#: layer give 43.1 us fixed + 7.15 us per sequence, and 19,392 raw EDF executions
#: on npu2 at attention_size 2048 give 37.1 + 8.54. The npu0 pair is used here,
#: because the bundle being re-priced is npu0's; the npu2 measurement is the
#: independent confirmation that the STRUCTURE -- a large fixed cost and a modest
#: per-sequence slope -- is real rather than an artefact of dividing by D17.
ATTN_FIXED_US = 43.1
ATTN_SLOPE_US = 7.15

#: What D17 measured on the card: attention executions per layer against the mean
#: number of sequences in the batch. This is the diversity the bundle's decode rows
#: were measured under, so it is the denominator R-attn compares this run against.
D17_GROUPS = [(1.95, 1.95), (3.91, 2.40), (8.91, 2.87), (15.16, 3.03), (29.09, 3.08)]


def _d17_groups(batch: int) -> float:
    """D17's executions-per-layer at this batch size, linear between its points.

    Clamped at both ends rather than extrapolated: below batch 1.95 and above
    29.09 there is no measurement, and a straight line through five points has no
    business predicting a sixth. STEP C.3 replaces the whole curve.
    """
    xs = [x for x, _ in D17_GROUPS]
    ys = [y for _, y in D17_GROUPS]
    if batch <= xs[0]:
        return ys[0]
    if batch >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= batch <= xs[i + 1]:
            t = (batch - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


RULES = ["R-pad", "R-pad-pow2", "R-attn-menu", "R-attn-coarse", "R-attn-naive",
         "R-128", "R-c1"]


def recharge(p: Pricer, layers: int, steps: list[Step], ladder, edges,
             decode_edges=None) -> dict:
    base = [price_baseline(p, layers, s) for s in steps]
    is_decode = [s.n_decode > 0 and s.n_prefill == 0 for s in steps]
    is_prefill = [s.n_prefill > 0 for s in steps]

    def _share(vals, mask):
        return sum(v for v, m in zip(vals, mask, strict=True) if m)

    out: dict = {
        "steps": len(steps),
        "baseline_total_ns": sum(base),
        "baseline_decode_ns": _share(base, is_decode),
        "baseline_prefill_ns": _share(base, is_prefill),
        "rules": {},
    }
    for rule in RULES:
        new = [price_rule(p, layers, s, rule, ladder, edges, decode_edges) for s in steps]
        tot, dec, pre = sum(new), _share(new, is_decode), _share(new, is_prefill)
        out["rules"][rule] = {
            "total_pp": (tot / out["baseline_total_ns"] - 1) * 100,
            "decode_pp": (
                (dec / out["baseline_decode_ns"] - 1) * 100
                if out["baseline_decode_ns"]
                else None
            ),
            "prefill_pp": (
                (pre / out["baseline_prefill_ns"] - 1) * 100
                if out["baseline_prefill_ns"]
                else None
            ),
        }
    return out


def check_against_simulator(p: Pricer, layers: int, steps: list[Step]) -> dict:
    """How closely the rebuilt cost model reproduces the run it is re-pricing."""
    ratios = []
    for s in steps:
        if s.cycles <= 0:
            continue
        ratios.append(price_baseline(p, layers, s) / s.cycles)
    if not ratios:
        return {"compared_steps": 0}
    return {
        "compared_steps": len(ratios),
        "model_over_simulator_median": statistics.median(ratios),
        "p05": sorted(ratios)[int(0.05 * len(ratios))],
        "p95": sorted(ratios)[int(0.95 * len(ratios))],
    }


def _fmt(x) -> str:
    return "--" if x is None else f"{x:+.2f}"


def to_markdown(rows: list[dict], prov: str) -> str:
    out = [
        "# E-N3 — what each bucket rule costs, priced offline",
        "",
        "`WORK_ORDER_npu_exec_model_spike.md` STEP A.3. Post-processing of STEP A.1's runs:",
        "no simulator re-run, no hardware. Prices come from the same perf DB the run used,",
        f"grid from `{prov}`. **First-order only** — re-pricing a step does not change which",
        "requests the scheduler would have put in the next one.",
        "",
        "Percentage points on the sum of DECODE step cost (the TPOT proxy); the prefill column",
        "is the TTFT proxy. `R-c1` is the control and must read 0.00.",
        "",
        "| point | steps | model/sim (median) | R-pad (artifact) | R-pad (pow2) "
        "| R-attn (menu grid) | R-attn (decode-edge grid) "
        "| R-attn naive (double count) | R-128 (prefill) | R-c1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        c, chk = r["recharge"], r.get("check", {})
        rl = c["rules"]
        med = chk.get("model_over_simulator_median")
        out.append(
            f"| {r['tag']} | {c['steps']} | "
            f"{'--' if med is None else f'{med:.3f}'} | "
            f"{_fmt(rl['R-pad']['decode_pp'])} | "
            f"{_fmt(rl['R-pad-pow2']['decode_pp'])} | "
            f"{_fmt(rl['R-attn-menu']['decode_pp'])} | "
            f"{_fmt(rl['R-attn-coarse']['decode_pp'])} | "
            f"{_fmt(rl['R-attn-naive']['decode_pp'])} | "
            f"{_fmt(rl['R-128']['prefill_pp'])} | "
            f"{_fmt(rl['R-c1']['decode_pp'])} |"
        )
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--grid", type=Path, default=None)
    ap.add_argument("--hardware", default="RNGD-CARD")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--variant", default="bf16")
    ap.add_argument("--model-type", default="llama")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--check", action="store_true",
                    help="also report how closely the rebuilt cost model "
                         "reproduces the simulator's own per-step cycles")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    ladder, edges, decode_edges, prov = load_grid(args.grid)
    p = Pricer(args.hardware, args.model, args.variant, args.tp, args.model_type)

    rows = []
    for log in sorted(args.run_dir.glob("sim_*.log"), key=_tag_order):
        tag = log.stem[len("sim_"):]
        trace = args.run_dir / f"trace_{tag}.jsonl"
        if not trace.exists():
            continue
        steps, meta = parse_run(log, trace)
        if not steps:
            print(f"  {tag}: no batch lines -- run needs --log-level INFO", file=sys.stderr)
            continue
        row = {"tag": tag, "meta": meta,
               "recharge": recharge(p, args.layers, steps, ladder, edges, decode_edges)}
        if args.check:
            row["check"] = check_against_simulator(p, args.layers, steps)
        rows.append(row)
        rl = row["recharge"]["rules"]
        chk = row.get("check", {}).get("model_over_simulator_median")
        print(f"  {tag}: R-pad {_fmt(rl['R-pad']['decode_pp'])} pp  "
              f"R-attn {_fmt(rl['R-attn-coarse']['decode_pp'])}..."
              f"{_fmt(rl['R-attn-menu']['decode_pp'])} pp  "
              f"R-128 {_fmt(rl['R-128']['prefill_pp'])} pp  "
              f"R-c1 {_fmt(rl['R-c1']['decode_pp'])} pp  "
              f"model/sim {'--' if chk is None else f'{chk:.3f}'}",
              file=sys.stderr)

    if not rows:
        print("no runs priced", file=sys.stderr)
        return 1

    payload = {"grid": prov, "runs": rows}
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(to_markdown(rows, prov))
    if not args.json and not args.markdown:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
