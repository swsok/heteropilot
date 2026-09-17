"""Count how the simulator's steps differ from the shape the RNGD runtime can run.

`WORK_ORDER_npu_exec_model_spike.md` STEP A.1 (E-N1). This measures the
*simulator*, not the card: it reconstructs every scheduled step of a completed
`python -m serving` run and counts the things a static AOT bucket executor cannot
do. No device, no hardware claim -- the comparison against the vendor grid is
arithmetic on a committed YAML.

**Where the step structure comes from.** The work order expected to have to add a
debug print, because `scheduler.py`'s batch log carries only a batch id. It does
not: `trace_generator.py:1308` already logs, at INFO,

    Batch #7: model=meta-llama/Llama-3.1-8B num_reqs=4 total_len=4 req_ids=[0, 1, 2, 3]

which is one line per step with the requests in it, and `Controller` logs a
running cycle count per iteration whose first difference is that step's duration.
Together with the trace the run was given -- which holds each request's input
length -- that is enough to replay every step exactly, so `serving/` is untouched
(work order A3).

**How a step is classified.** Replaying in log order and tracking how many tokens
each request has had computed:

* a request with `computed < input_toks` is in *prefill* this step, and consumes
  some chunk of the step's token budget;
* any other request is in *decode*, consumes exactly one token, and attends over
  `computed` tokens of KV.

So `sum(prefill chunks) == total_len - n_decode`, and the chunks themselves are
recovered greedily in `req_ids` order (each prefilling request takes as much of
the remaining budget as it still needs). That reproduces the scheduler's own
allocation and is self-checking: the budget must land exactly on zero, and
`--strict` makes a step that does not balance an error instead of a counter.

**What is counted, and why each one is a way the grid is violated.**

* *mixed steps* -- prefill tokens and decode tokens in one step. The artifact
  compiles no such bucket (D90), so every one of these is a step the runtime
  would have had to split.
* *prefill batch size* -- every compiled prefill and extend bucket is `bs=1`.
  Steps with more than one prefilling request are likewise not executable.
* *decode batch padding* -- a decode batch runs on the next compiled batch size
  up. Reported against both ladders, because the work order assumed powers of two
  and the artifact has a 384 rung between 256 and 512 (STEP 0).
* *KV groups* -- the kernelwise path issues attention per group of sequences
  sharing an attention bucket. Reported against both candidate grids, the uniform
  1024-token one the work order assumed and the artifact's real decode edges, so
  that D17's measured 1.95 / 2.40 / 2.87 / 3.03 / 3.08 executions per layer can
  pick one.
* *prefill 128-padding* -- a prefill chunk runs on the next multiple of 128, to be
  read against D17's +10.9 %.

Usage::

    python experiments/scripts/npu_exec_step_census.py outputs/npu_spike/a1 \\
        --grid profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml \\
        --markdown experiments/results/npu_exec_step_census.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BATCH_RE = re.compile(
    r"\[TraceGenerator\].*?Batch #(?P<bid>\d+): model=(?P<model>\S+) "
    r"num_reqs=(?P<n>\d+) total_len=(?P<tl>\d+) req_ids=\[(?P<ids>[^\]]*)\]"
)
#: The cycle count is cumulative simulated time, so a step costs the FIRST
#: DIFFERENCE. Reading it as a duration makes every step look monotonically
#: longer, which is the kind of error that still produces a plausible table.
ITER_RE = re.compile(
    r"\[Controller\].*?NPU\[(?P<npu>\d+)\] iteration (?P<it>\d+) finished, "
    r"(?P<cycles>\d+) cycles"
)

#: Fallback ladders, used only when --grid is not given. The real ones are read
#: off the artifact grid, because they are properties of the compiled artifact.
FALLBACK_BATCH_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024]
FALLBACK_DECODE_EDGES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
POW2_LADDER = [1 << i for i in range(0, 21)]


@dataclass
class Step:
    batch_id: int
    n_prefill: int = 0
    n_decode: int = 0
    prefill_chunks: list[int] = field(default_factory=list)
    decode_kv: list[int] = field(default_factory=list)
    #: KV already resident for the requests still prefilling, i.e. what a chunked
    #: prefill attends over. `_build_batch_ctx` calls it `kv_prefill`.
    kv_prefill: int = 0
    #: `lm_head_len`: the sequence count the per-sequence layers are priced at.
    sequences: int = 0
    cycles: int = 0
    balanced: bool = True

    @property
    def mixed(self) -> bool:
        return self.n_prefill > 0 and self.n_decode > 0


def _next_up(ladder: list[int], value: int) -> int:
    """The smallest rung at or above `value`; the top rung if it overflows."""
    for rung in ladder:
        if rung >= value:
            return rung
    return ladder[-1]


def load_grid(path: Path | None) -> tuple[list[int], list[int], str]:
    """(batch ladder, decode attention edges, provenance) from an artifact grid."""
    if path is None:
        return FALLBACK_BATCH_LADDER, FALLBACK_DECODE_EDGES, "fallback (no --grid given)"
    import yaml

    doc = yaml.safe_load(path.read_text())
    ladder = doc.get("kernelwise", {}).get("tokenwise_buckets")
    if not ladder:
        ladder = sorted({row["batch_size"] for row in doc["buckets"]["decode"]})
    edges = sorted({row["attention_size"] for row in doc["buckets"]["decode"]})
    src = doc.get("source", {})
    # A relative --grid is not `relative_to` anything, and an absolute one outside
    # the tree RAISES rather than falling back -- the same trap lowload_sim_error.py
    # records in _arg_path. Resolve first, then degrade to the plain path.
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path.resolve()
    prov = f"{shown} (artifact {src.get('artifact_id', '?')})"
    return sorted(ladder), edges, prov


def parse_run(log: Path, trace: Path, strict: bool = False) -> tuple[list[Step], dict]:
    """Replay one run's steps from its INFO log and the trace it was given."""
    inputs: dict[int, int] = {}
    with trace.open() as fh:
        for i, line in enumerate(fh):
            inputs[i] = int(json.loads(line)["input_toks"])

    steps: list[Step] = []
    computed: dict[int, int] = {}
    seen_batches: set[int] = set()
    # Cumulative cycles, per NPU, so a multi-instance run does not mix clocks.
    last_cycles: dict[str, int] = {}
    pending: Step | None = None
    unbalanced = 0

    for line in log.open(errors="replace"):
        m = BATCH_RE.search(line)
        if m:
            if pending is not None:
                steps.append(pending)
            bid = int(m.group("bid"))
            if bid in seen_batches:
                # A batch re-logged would double-count every token in it.
                raise SystemExit(f"{log}: batch #{bid} logged twice")
            seen_batches.add(bid)
            ids = [int(x) for x in m.group("ids").split(",") if x.strip()]
            step = Step(batch_id=bid)

            step.sequences = len(ids)
            for rid in ids:
                if computed.get(rid, 0) < inputs.get(rid, 0):
                    step.n_prefill += 1
                    step.kv_prefill += computed.get(rid, 0)
                else:
                    step.n_decode += 1
                    step.decode_kv.append(computed.get(rid, 0))

            budget = int(m.group("tl")) - step.n_decode
            for rid in ids:
                done = computed.get(rid, 0)
                need = inputs.get(rid, 0) - done
                if need > 0:
                    chunk = min(need, budget)
                    step.prefill_chunks.append(chunk)
                    computed[rid] = done + chunk
                    budget -= chunk
                else:
                    computed[rid] = done + 1
            if budget != 0:
                step.balanced = False
                unbalanced += 1
                if strict:
                    raise SystemExit(
                        f"{log}: batch #{bid} leaves {budget} tokens unassigned; "
                        f"the chunk reconstruction does not match the scheduler"
                    )
            pending = step
            continue

        m = ITER_RE.search(line)
        if m and pending is not None:
            npu, cycles = m.group("npu"), int(m.group("cycles"))
            pending.cycles = max(0, cycles - last_cycles.get(npu, 0))
            last_cycles[npu] = cycles

    if pending is not None:
        steps.append(pending)

    return steps, {"requests": len(inputs), "unbalanced_steps": unbalanced}


def census(steps: list[Step], ladder: list[int], edges: list[int]) -> dict:
    total = len(steps)
    total_cycles = sum(s.cycles for s in steps)
    mixed = [s for s in steps if s.mixed]
    with_prefill = [s for s in steps if s.n_prefill > 0]
    pure_decode = [s for s in steps if s.n_decode > 0 and s.n_prefill == 0]

    def _pad_ratio(rows: list[Step], lad: list[int]) -> float | None:
        served = sum(s.n_decode for s in rows)
        if not served:
            return None
        padded = sum(_next_up(lad, s.n_decode) for s in rows)
        return padded / served - 1.0

    def _groups(rows: list[Step], mode: str) -> dict:
        counts: Counter[int] = Counter()
        for s in rows:
            if not s.decode_kv:
                continue
            if mode == "uniform1024":
                keys = {kv // 1024 for kv in s.decode_kv}
            else:
                keys = {_next_up(edges, max(kv, 1)) for kv in s.decode_kv}
            counts[len(keys)] += 1
        n = sum(counts.values())
        mean = sum(k * v for k, v in counts.items()) / n if n else None
        return {"mean": mean, "histogram": dict(sorted(counts.items()))}

    prefill_tokens = sum(sum(s.prefill_chunks) for s in steps)
    prefill_padded = sum(-(-c // 128) * 128 for s in steps for c in s.prefill_chunks)

    return {
        "steps": total,
        "steps_with_decode": sum(1 for s in steps if s.n_decode),
        "mixed_steps": len(mixed),
        "mixed_step_frac": len(mixed) / total if total else None,
        "mixed_step_cycle_frac": (
            sum(s.cycles for s in mixed) / total_cycles if total_cycles else None
        ),
        "prefill_steps": len(with_prefill),
        "prefill_bs1_frac": (
            sum(1 for s in with_prefill if s.n_prefill == 1) / len(with_prefill)
            if with_prefill
            else None
        ),
        "prefill_bs_histogram": dict(
            sorted(Counter(s.n_prefill for s in with_prefill).items())
        ),
        "decode_bs_histogram": dict(
            sorted(Counter(s.n_decode for s in pure_decode).items())
        ),
        "decode_pad_frac_artifact_ladder": _pad_ratio(pure_decode, ladder),
        "decode_pad_frac_pow2": _pad_ratio(pure_decode, POW2_LADDER),
        "kv_groups_artifact_edges": _groups(pure_decode, "artifact"),
        "kv_groups_uniform1024": _groups(pure_decode, "uniform1024"),
        "prefill_tokens": prefill_tokens,
        "prefill_pad128_frac": (
            prefill_padded / prefill_tokens - 1.0 if prefill_tokens else None
        ),
        "total_cycles": total_cycles,
    }


#: What D17 measured on the card: attention executions per layer, against the
#: mean number of sequences in the batch. Carried here so the census prints the
#: comparison rather than leaving it to a reader with two documents open.
D17_ATTENTION_EXECUTIONS = {1.95: 1.95, 3.91: 2.40, 8.91: 2.87, 15.16: 3.03, 29.09: 3.08}


def _fmt_pct(x: float | None) -> str:
    return "--" if x is None else f"{x * 100:.1f} %"


def _fmt_num(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}"


def to_markdown(rows: list[dict], grid_prov: str) -> str:
    out = [
        "# E-N1 — what the simulator's steps look like, counted",
        "",
        "`WORK_ORDER_npu_exec_model_spike.md` STEP A.1. Every number here is a property of a",
        "**simulated** run; nothing was measured on hardware. The grid the padding and grouping",
        f"ladders come from is `{grid_prov}`.",
        "",
        "| point | steps | mixed steps | mixed cycle share | prefill steps bs=1 "
        "| decode pad (artifact ladder) | decode pad (pow2) "
        "| KV groups (artifact edges) | KV groups (uniform 1024) | prefill 128-pad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        c = r["census"]
        out.append(
            f"| {r['tag']} | {c['steps']} | "
            f"{c['mixed_steps']} ({_fmt_pct(c['mixed_step_frac'])}) | "
            f"{_fmt_pct(c['mixed_step_cycle_frac'])} | "
            f"{_fmt_pct(c['prefill_bs1_frac'])} | "
            f"{_fmt_pct(c['decode_pad_frac_artifact_ladder'])} | "
            f"{_fmt_pct(c['decode_pad_frac_pow2'])} | "
            f"{_fmt_num(c['kv_groups_artifact_edges']['mean'])} | "
            f"{_fmt_num(c['kv_groups_uniform1024']['mean'])} | "
            f"{_fmt_pct(c['prefill_pad128_frac'])} |"
        )
    out += [
        "",
        "D17 measured, on the card, attention executions per layer of",
        "1.95 / 2.40 / 2.87 / 3.03 / 3.08 at mean batch 1.95 / 3.91 / 8.91 / 15.16 / 29.09.",
        "The two KV-group columns are the two candidate grids; the one that tracks those",
        "numbers is the grid the runtime actually groups on.",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path,
                    help="a lowload_sim_error.py --out directory, holding "
                         "sim_<tag>.log and trace_<tag>.jsonl pairs")
    ap.add_argument("--grid", type=Path, default=None,
                    help="artifact_buckets.yaml; without it the ladders are the "
                         "hardcoded fallbacks and the run is labelled as such")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="fail on a step whose chunk reconstruction does not "
                         "balance, instead of counting it")
    args = ap.parse_args()

    ladder, edges, prov = load_grid(args.grid)

    rows = []
    for log in sorted(args.run_dir.glob("sim_*.log")):
        tag = log.stem[len("sim_"):]
        trace = args.run_dir / f"trace_{tag}.jsonl"
        if not trace.exists():
            print(f"  {tag}: no trace file, skipped", file=sys.stderr)
            continue
        steps, meta = parse_run(log, trace, strict=args.strict)
        if not steps:
            print(f"  {tag}: no batch lines -- was the run given --log-level INFO?",
                  file=sys.stderr)
            continue
        c = census(steps, ladder, edges)
        rows.append({"tag": tag, "log": str(log), "meta": meta, "census": c})
        print(f"  {tag}: {c['steps']} steps, "
              f"{c['mixed_steps']} mixed ({_fmt_pct(c['mixed_step_frac'])}), "
              f"prefill bs=1 {_fmt_pct(c['prefill_bs1_frac'])}, "
              f"KV groups {_fmt_num(c['kv_groups_artifact_edges']['mean'])}"
              f"/{_fmt_num(c['kv_groups_uniform1024']['mean'])}, "
              f"unbalanced {meta['unbalanced_steps']}",
              file=sys.stderr)

    if not rows:
        print("no runs parsed", file=sys.stderr)
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
