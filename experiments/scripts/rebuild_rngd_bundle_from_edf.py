"""Rebuild the RNGD perf bundle from FuriosaAI's EDF profiler traces.

Replaces synthetic-layer measurements with the stage times the vendor compiler
actually emits, for the graph the server actually runs. Why this matters:

* Our harness accounts for only 58% of real per-layer decode cost (507 us
  measured against 290-307 us) -- `rngd_vendor_profiler_vs_layerwise.md`.
* It measures ONE PE at a time, but the artifact's `tensor_parallel_size: 8` is
  realised as two fused 4-PE quads (`leader_device` is `npu0pe0-3`), so the
  harness models a rank granularity the hardware does not use.
* Most importantly, an EDF stage time is the card's REAL per-layer latency with
  the intra-card reduction already inside it. That rehabilitates the
  card-as-device abstraction: a `tp1` instance needs no ASTRA-Sim collective
  because the measurement already paid for it. Card-as-device failed earlier only
  because it was fed per-PE synthetic numbers (`rngd_card_vs_pe_model.md`).

So the bundle produced here is **card-as-device, tp1**, hardware `RNGD-CARD`.

Mapping EDF stages onto the §3.7 contract
-----------------------------------------
`attention.csv` maps directly and faithfully. EDF attention buckets are
`(batch_size, attention_size, kv_cache_size)`:
  kv_cache_size > 0 -> a decode step: n_decode=batch_size, kv_decode=kv_cache_size
  kv_cache_size == 0 -> a prefill chunk: prefill_chunk=attention_size

`dense.csv` cannot map directly, and this is the one modelling decision here. A
single `Tokenwise` execution covers ALL the dense work of one decoder layer
(qkv, rotary, o_proj, gate_up, act, down_proj, both norms) fused together; the
compiler does not expose them separately. But the simulator only ever SUMS the
per-layer lookups for one decoder iteration, so what has to be right is the sum.
This script therefore takes the MAGNITUDE from the vendor (the measured Tokenwise
stage time per layer) and the DISTRIBUTION from the harness bundle (each canonical
layer's share of the harness total at the same token count), scaling the harness
values so they sum to the vendor figure. Stated plainly: absolute per-layer
latency is measured on the real graph; the split between canonical layers within a
decoder layer is inherited and is not a vendor measurement.

`per_sequence.csv` (lm_head, sampler) has no separately visible EDF stage, so it
is copied from the harness bundle unchanged and flagged in meta.yaml.

Usage::

    # 1. collect (starts and stops furiosa-llm serve once per concurrency)
    PYTHONPATH=$PWD python3 experiments/scripts/rebuild_rngd_bundle_from_edf.py collect \
        --artifact <path> --card 0 --concurrency 1,2,4,8,16,32
    # 2. build the bundle from everything collected
    PYTHONPATH=$PWD python3 experiments/scripts/rebuild_rngd_bundle_from_edf.py build
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Derived in rngd_vendor_profiler_vs_layerwise.md: total device cycles over wall
#: time lands on 1,599.9 MHz, a round 1.6 GHz to 0.006%, which is only consistent
#: with a saturated card - and 5 concurrent requests on one card is saturated.
CLOCK_HZ = 1.6e9

#: Canonical dense layers of one decoder layer, in the order llama.yaml walks
#: them. `embedding` and `final_layernorm` are NOT here: they run once per
#: iteration, not once per layer, so they must not take a share of a per-layer
#: stage time.
DECODER_DENSE = (
    "layernorm", "qkv_proj", "rotary_emb", "o_proj",
    "gate_up_proj", "act_fn", "down_proj",
)
#: Per-iteration dense layers, carried over from the harness unchanged.
ITERATION_DENSE = ("embedding", "final_layernorm")

TOKENWISE_RE = re.compile(r"Tokenwise\(TokenwiseBucket \{ input_size: (\d+) \}\)")
ATTENTION_RE = re.compile(
    r"Attention\(AttentionBucket \{ batch_size: (\d+), attention_size: (\d+), "
    r"kv_cache_size: (\d+) \}\)"
)


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def wait_for_server(port: int, timeout: float) -> str | None:
    """Poll /v1/models until the server answers; return the served model id."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.load(response)["data"][0]["id"]
        except Exception:
            time.sleep(3)
    return None


def collect(args) -> int:
    out_root = args.out / "edf"
    out_root.mkdir(parents=True, exist_ok=True)
    passes = [int(v) for v in args.concurrency.split(",") if v]

    for concurrency in passes:
        edf = (out_root / f"edf_c{concurrency}.csv").resolve()
        if edf.exists() and not args.refresh:
            print(f"concurrency {concurrency}: {edf.name} exists, skipping")
            continue
        env = dict(os.environ)
        env["EDF_PROFILER_OUTPUT_PATH"] = str(edf)
        env["TUC_PROFILE_LEVEL"] = "info"
        env["RUST_LOG"] = "span::tuc=info"
        log_path = out_root / f"serve_c{concurrency}.log"
        with log_path.open("w") as log:
            server = subprocess.Popen(
                ["furiosa-llm", "serve", str(args.artifact),
                 "--host", "127.0.0.1", "--port", str(args.port),
                 "--devices", f"npu:{args.card}:*"],
                env=env, stdout=log, stderr=log, start_new_session=True,
            )
        try:
            model = wait_for_server(args.port, args.startup_timeout)
            if model is None:
                print(f"concurrency {concurrency}: server never came up; see "
                      f"{log_path}")
                continue
            print(f"concurrency {concurrency}: server up, driving "
                  f"{args.num_reqs} request(s)")
            # sys.executable may be a venv without `openai`; the bench client
            # needs the system interpreter where the vendor stack lives.
            subprocess.run(
                [args.bench_python, "-u",
                 str(REPO_ROOT / "experiments/scripts/bench_furiosa_endpoint.py"),
                 "--base-url", f"http://127.0.0.1:{args.port}/v1",
                 "--model", model, "--dataset", str(args.dataset),
                 "--num-reqs", str(args.num_reqs),
                 "--concurrency", str(concurrency),
                 "--out", str(out_root / f"real_c{concurrency}.json")],
                cwd=REPO_ROOT, check=False, timeout=args.bench_timeout,
            )
        finally:
            # The runtime flushes the EDF file on shutdown, so it must be a clean
            # signal to the whole process group, not a kill.
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        rows = len(edf.read_text().splitlines()) - 1 if edf.exists() else 0
        print(f"concurrency {concurrency}: {rows} stage executions -> {edf.name}")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
# What the traces actually contain, established by census over all six runs
# (`experiments/results/rngd_edf_bundle_notes.md` carries the tables):
#
# THREE stage kinds, not two. Besides `Tokenwise` and `Attention` there is
# `Composed(a, b)`, which the first draft of this script ignored entirely and
# which is 98.8% of device cycles at concurrency 1. Nine variants partition
# [0, 64] in steps of 8 plus a terminal `Composed(64, 64)`, each executing
# EXACTLY once per forward (n identical across all nine, 16473 at c=1 against
# 16495 generated tokens). So the runtime has two compiled plans:
#
#   batch 1        -> one fully-fused Composed graph, no Tokenwise, no Attention
#   batch >= 2     -> per-layer Tokenwise + Attention, 32 Tokenwise per forward
#
# That is why there is no `input_size: 1` Tokenwise bucket anywhere in 1.2 M
# stage executions: at batch 1 the bucketed path is not used at all. tokens=1 is
# the row the simulator needs most for decode, so it is built from Composed.
#
# STAGE TIMES OVERLAP ONLY ON THE COMPOSED PATH. Summed device cycles against
# wall time: c4 99.5%, c8 99.3%, c16 100.5%, c32 98.7% - the bucketed path sums
# to wall time, so its stage times can be used as-is. c1 is 114.7%, so the fused
# graph pipelines internally and a sum of its nine medians over-counts a forward.
# `_composed_dense` divides that sum back down to the measured wall-clock forward.
#
# ATTENTION HAS TWO REGIMES, distinguished by `attention_size - kv_cache_size`:
#   == 1  -> a decode step: batch_size sequences, each attending over kv_cache_size
#   >  1  -> a prefill chunk of (attention_size - kv_cache_size) new tokens on top
#            of kv_cache_size already cached (kv > 0 is chunked-prefill continuation)
# The first draft mapped every kv > 0 bucket to decode, which put 32 prefill
# chunks on the decode axis and made it non-monotonic in kv (154 us at kv=128
# against 49 us at kv=1023). Fixed here.
#
# DECODE ATTENTION IS ATTRIBUTED PER TRACE, NOT PER BUCKET, and this is a
# modelling decision worth stating plainly. The runtime groups a decode batch by
# kv bucket, so one forward pays 1.95 attention executions per layer at batch 2
# and 3.08 at batch 29 - the count tracks the batch's kv DIVERSITY, which the
# §3.7 contract cannot express (it asks for one number given n_decode and a mean
# kv). Charging a single bucket median would therefore under-count per-layer
# attention by ~3x at large batch. Instead each trace contributes one row whose
# time is total decode-attention device time / (forwards x 32 layers), so the
# total the simulator accumulates closes on the measured total. The consequence:
# the decode attention axis is calibrated to THIS traffic mix (sharegpt, mean kv
# ~2200, stable to +-1% across all five batched traces).

COMPOSED_RE = re.compile(r"Composed\((\d+), (\d+)\)")

#: Llama-3.1-8B decoder layers. The Composed partition of [0, 64] is 2 units per
#: decoder layer; what matters here is only that its 8 body segments cover the
#: whole stack, so body_total / 32 is one decoder layer.
N_DECODER_LAYERS = 32


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


class Trace:
    """One concurrency's EDF trace, parsed into the three stage kinds."""

    def __init__(self, path: Path, bench: Path):
        self.path = path
        self.concurrency = int(re.search(r"edf_c(\d+)", path.name).group(1))
        self.tokenwise: dict[int, list[float]] = collections.defaultdict(list)
        self.composed: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
        self.decode_attn: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
        self.prefill_attn: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
        self.decode_attn_us = 0.0     # total, all executions
        self.decode_seq_layers = 0    # sum of batch_size over decode executions
        self.decode_kv_weighted = 0.0
        self.decode_attn_execs = 0
        self.other_us = 0.0
        for row in csv.DictReader(path.open(newline="")):
            us = int(row["cycle"]) / CLOCK_HZ * 1e6
            name = row["name"]
            m = TOKENWISE_RE.match(name)
            if m:
                self.tokenwise[int(m.group(1))].append(us)
                self.other_us += us
                continue
            m = ATTENTION_RE.match(name)
            if m:
                batch, size, kv = (int(g) for g in m.groups())
                self.other_us += us
                if size - kv == 1:
                    self.decode_attn[(batch, kv)].append(us)
                    self.decode_attn_us += us
                    self.decode_seq_layers += batch
                    self.decode_kv_weighted += kv * batch
                    self.decode_attn_execs += 1
                else:
                    self.prefill_attn[(size - kv, kv)].append(us)
                continue
            m = COMPOSED_RE.match(name)
            if m:
                self.composed[(int(m.group(1)), int(m.group(2)))].append(us)
        self.bench = json.loads(bench.read_text()) if bench.exists() else {}

    #: A Tokenwise bucket at or below 64 is a decode forward; 128 and above is a
    #: prefill chunk. The gap in the artifact's bucket ladder makes this crisp.
    def decode_forwards(self) -> float:
        execs = sum(len(v) for size, v in self.tokenwise.items() if size <= 64)
        return execs / N_DECODER_LAYERS

    def composed_forwards(self) -> float:
        counts = {len(v) for v in self.composed.values()}
        return float(next(iter(counts))) if len(counts) == 1 else 0.0


def read_traces(edf_dir: Path) -> list[Trace]:
    def cc(path: Path) -> int:
        return int(re.search(r"edf_c(\d+)", path.name).group(1))

    paths = sorted(edf_dir.glob("edf_c*.csv"), key=cc)
    return [Trace(p, p.parent / f"real_c{cc(p)}.json") for p in paths]


def harness_table(path: Path, key: str) -> dict[str, dict[int, float]]:
    table: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for row in csv.DictReader(path.open(newline="")):
        table[row["layer"]][int(row[key])] = float(row["time_us"])
    return table


def nearest(table: dict[int, float], want: int) -> float | None:
    if not table:
        return None
    return table[min(table, key=lambda k: (abs(k - want), k))]


def distribute(stage_us: float, harness: dict[str, dict[int, float]],
               tokens: int) -> tuple[list[dict], float]:
    """Split one per-layer stage time across the canonical dense layers.

    MAGNITUDE from the vendor (`stage_us`), DISTRIBUTION from the harness. The
    compiler fuses a whole decoder layer into one stage and does not expose the
    pieces, but the simulator only ever sums the per-layer lookups, so the sum is
    what has to be right. The split within a decoder layer is inherited from our
    own harness and is NOT a vendor measurement.
    """
    shares = {name: nearest(harness[name], tokens) or 0.0 for name in DECODER_DENSE}
    total = sum(shares.values())
    if total <= 0:
        return [], 0.0
    rows = [{"layer": name, "tokens": tokens, "time_us": stage_us * value / total}
            for name, value in shares.items()]
    return rows, total


def composed_dense(trace: Trace, harness: dict[str, dict[int, float]],
                   attn_1seq_us: float) -> tuple[list[dict], dict]:
    """Build the tokens=1 dense row from the fully-fused batch-1 graph.

    Three corrections, in order:

    1. UNION. The nine Composed segments overlap in time (device cycles are
       114.7% of wall at c=1), so their summed medians over-count one forward.
       Scale them so the sum equals the measured wall-clock forward time. Wall
       time also carries client and scheduler overhead, so this correction is if
       anything too large and the result too small - it errs conservative.
    2. HEAD. The terminal `Composed(64, 64)` segment is everything after the
       decoder stack (final norm, lm_head, sampling). It belongs in
       per_sequence.csv, not in a per-layer dense row.
    3. ATTENTION. At batch 1 attention is fused inside Composed, but the
       simulator will separately charge an attention row for the same step. So
       the measured b=1 per-layer decode attention is subtracted here to avoid
       counting it twice.
    """
    body = {k: v for k, v in trace.composed.items() if k[0] < k[1]}
    tail = [k for k in trace.composed if k[0] == k[1]]
    if not body or not tail:
        return [], {}
    body_us = sum(median(v) for v in body.values())
    tail_us = median(trace.composed[tail[0]])
    forwards = trace.composed_forwards()
    wall_s = trace.bench.get("wall_s")
    if not (forwards and wall_s):
        return [], {}
    # Wall time minus the device time of the prefill stages that ran alongside.
    decode_wall_us = (wall_s - trace.other_us / 1e6) * 1e6
    union = decode_wall_us / forwards / (body_us + tail_us)
    per_layer = body_us * union / N_DECODER_LAYERS
    dense_us = per_layer - attn_1seq_us
    rows, harness_sum = distribute(dense_us, harness, 1)
    audit = {
        "segments": len(trace.composed), "forwards": int(forwards),
        "body_sum_us": round(body_us, 1), "tail_us": round(tail_us, 1),
        "union_factor": round(union, 4),
        "per_layer_us": round(per_layer, 2),
        "attn_1seq_us": round(attn_1seq_us, 2),
        "dense_per_layer_us": round(dense_us, 2),
        "harness_sum_us": round(harness_sum, 1),
        "head_us": round(tail_us * union, 1),
    }
    return rows, audit


def build(args) -> int:
    traces = read_traces(args.out / "edf")
    if not traces:
        raise SystemExit(f"no EDF traces under {args.out / 'edf'}; run `collect` first")
    print(f"read {len(traces)} trace(s): "
          + ", ".join(f"c{t.concurrency}" for t in traces))

    harness = harness_table(args.harness_bundle / "dense.csv", "tokens")
    missing = [name for name in DECODER_DENSE if not harness.get(name)]
    if missing:
        raise SystemExit(f"harness bundle lacks {missing}; cannot distribute the "
                         f"vendor stage time across canonical layers")

    # --- attention, decode: one row per trace, total-preserving ---------------
    attn_rows: list[dict] = []
    decode_audit: list[dict] = []
    for trace in traces:
        forwards = trace.decode_forwards()
        if forwards < 1 or not trace.decode_attn_execs:
            continue
        layers = forwards * N_DECODER_LAYERS
        per_layer = trace.decode_attn_us / layers
        seqs = trace.decode_seq_layers / layers
        kv = trace.decode_kv_weighted / trace.decode_seq_layers
        attn_rows.append({"prefill_chunk": 0, "kv_prefill": 0,
                          "n_decode": max(1, round(seqs)), "kv_decode": round(kv),
                          "time_us": per_layer})
        decode_audit.append({
            "concurrency": trace.concurrency, "decode_forwards": round(forwards, 1),
            "sequences_per_forward": round(seqs, 2),
            "attn_execs_per_layer": round(trace.decode_attn_execs / layers, 2),
            "mean_kv": round(kv), "per_layer_us": round(per_layer, 2),
        })
        print(f"  decode attn c{trace.concurrency:<2}: nd={round(seqs):>3} "
              f"kv={round(kv):>5}  {per_layer:7.2f} us/layer  "
              f"({trace.decode_attn_execs / layers:.2f} exec/layer)")

    # nd=1 has no batched trace of its own (batch 1 runs the fused graph), but
    # b=1 decode buckets appear in every batched trace, and at batch 1 there is
    # exactly one kv group, so one bucket median IS the per-layer cost.
    b1 = [us for t in traces for (b, _kv), v in t.decode_attn.items() if b == 1
          for us in v]
    attn_1seq_us = median(b1) if b1 else 0.0
    if b1:
        kv1 = median([kv for t in traces for (b, kv), v in t.decode_attn.items()
                      if b == 1 for _ in v])
        attn_rows.insert(0, {"prefill_chunk": 0, "kv_prefill": 0, "n_decode": 1,
                             "kv_decode": int(kv1), "time_us": attn_1seq_us})
        print(f"  decode attn nd=1  : kv={int(kv1):>5}  {attn_1seq_us:7.2f} us/layer "
              f"(pooled b=1 buckets, n={len(b1)})")

    # --- attention, prefill: per-bucket medians ------------------------------
    prefill: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for trace in traces:
        for key, values in trace.prefill_attn.items():
            prefill[key].extend(values)
    for (chunk, kv), values in sorted(prefill.items()):
        attn_rows.append({"prefill_chunk": chunk, "kv_prefill": kv,
                          "n_decode": 0, "kv_decode": 0, "time_us": median(values)})
    print(f"  prefill attn      : {len(prefill)} bucket(s), "
          f"chunk {min(k[0] for k in prefill)}..{max(k[0] for k in prefill)}")

    # --- dense --------------------------------------------------------------
    dense_rows: list[dict] = []
    breakdown: list[dict] = []
    c1 = next((t for t in traces if t.composed_forwards() and not t.decode_forwards()),
              None)
    composed_audit: dict = {}
    if c1 is not None:
        rows, composed_audit = composed_dense(c1, harness, attn_1seq_us)
        dense_rows.extend(rows)
        if composed_audit:
            print(f"  tokenwise     1: fused Composed graph -> "
                  f"{composed_audit['per_layer_us']:.1f} us/layer "
                  f"(union x{composed_audit['union_factor']}), minus "
                  f"{attn_1seq_us:.1f} us attention -> "
                  f"{composed_audit['dense_per_layer_us']:.1f} us dense")
            breakdown.append({
                "tokens": 1, "executions": composed_audit["forwards"],
                "vendor_stage_us": composed_audit["dense_per_layer_us"],
                "harness_sum_us": composed_audit["harness_sum_us"],
                "vendor_over_harness": round(
                    composed_audit["dense_per_layer_us"]
                    / composed_audit["harness_sum_us"], 4),
                "source": "Composed (fused batch-1 graph)",
            })

    pooled: dict[int, list[float]] = collections.defaultdict(list)
    for trace in traces:
        for size, values in trace.tokenwise.items():
            pooled[size].extend(values)
    for tokens in sorted(pooled):
        stage_us = median(pooled[tokens])
        rows, harness_sum = distribute(stage_us, harness, tokens)
        if not rows:
            continue
        dense_rows.extend(rows)
        breakdown.append({
            "tokens": tokens, "executions": len(pooled[tokens]),
            "vendor_stage_us": round(stage_us, 3),
            "harness_sum_us": round(harness_sum, 3),
            "vendor_over_harness": round(stage_us / harness_sum, 4),
            "source": "Tokenwise",
        })
        print(f"  tokenwise {tokens:>5}: vendor {stage_us:9.1f} us "
              f"(n={len(pooled[tokens]):>6}) vs harness sum {harness_sum:8.1f} us "
              f"-> x{stage_us / harness_sum:5.2f}")

    # Per-iteration layers keep their harness values: they are not part of a
    # per-layer stage, so scaling them by one would be wrong.
    for name in ITERATION_DENSE:
        for tokens, value in sorted(harness.get(name, {}).items()):
            dense_rows.append({"layer": name, "tokens": tokens, "time_us": value})

    # --- per_sequence -------------------------------------------------------
    # The head (final norm + lm_head + sampling) is a visible stage only on the
    # fused batch-1 path, as the terminal Composed segment. So take its MAGNITUDE
    # at 1 sequence from that measurement and the SHAPE over sequence count from
    # the harness, which is the only source for how the head scales with batch.
    seq_harness = harness_table(args.harness_bundle / "per_sequence.csv", "sequences")
    seq_rows: list[dict] = []
    head_us = composed_audit.get("head_us")
    anchor = sum(nearest(seq_harness[n], 1) or 0.0 for n in seq_harness)
    seq_scale = (head_us / anchor) if (head_us and anchor > 0) else 1.0
    for name, table in seq_harness.items():
        for seqs, value in sorted(table.items()):
            seq_rows.append({"layer": name, "sequences": seqs,
                             "time_us": value * seq_scale})
    print(f"  per_sequence      : harness shape x{seq_scale:.3f} "
          f"(vendor head {head_us} us vs harness {anchor:.1f} us at 1 seq)")

    stage = args.out / "bundle" / "tp1"
    stage.mkdir(parents=True, exist_ok=True)
    keys = ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode")
    # Contract: attention keys must be unique.
    merged: dict[tuple, list[float]] = collections.defaultdict(list)
    for row in attn_rows:
        merged[tuple(row[k] for k in keys)].append(row["time_us"])
    attn_rows = [dict(zip(keys, key, strict=True), time_us=median(v))
                 for key, v in sorted(merged.items())]

    _write(stage / "dense.csv", ("layer", "tokens", "time_us"), dense_rows)
    _write(stage / "attention.csv", (*keys, "time_us"), attn_rows)
    _write(stage / "per_sequence.csv", ("layer", "sequences", "time_us"), seq_rows)
    _write(args.out / "edf_vs_harness_dense.csv",
           ("tokens", "executions", "vendor_stage_us", "harness_sum_us",
            "vendor_over_harness", "source"), breakdown)
    _write(args.out / "edf_decode_attention.csv",
           ("concurrency", "decode_forwards", "sequences_per_forward",
            "attn_execs_per_layer", "mean_kv", "per_layer_us"), decode_audit)
    (args.out / "edf_composed.json").write_text(
        json.dumps(composed_audit, indent=2) + "\n")

    print(f"\nwrote {stage}")
    print(f"  dense.csv        {len(dense_rows):4d} rows")
    print(f"  attention.csv    {len(attn_rows):4d} rows")
    print(f"  per_sequence.csv {len(seq_rows):4d} rows")
    print("  edf_vs_harness_dense.csv / edf_decode_attention.csv / "
          "edf_composed.json: the audit trail")
    return 0


def _write(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="drive the server with EDF profiling on")
    c.add_argument("--artifact", required=True, type=Path)
    c.add_argument("--card", type=int, default=0)
    c.add_argument("--port", type=int, default=8020)
    c.add_argument("--concurrency", default="1,2,4,8,16,32")
    c.add_argument("--num-reqs", type=int, default=24)
    c.add_argument("--dataset", type=Path,
                   default=Path("workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl"))
    c.add_argument("--out", type=Path, default=Path("outputs/rngd_edf_bundle"))
    c.add_argument("--startup-timeout", type=float, default=300.0)
    c.add_argument("--bench-timeout", type=float, default=1200.0)
    c.add_argument("--refresh", action="store_true")
    c.add_argument("--bench-python", default="/usr/bin/python3",
                   help="interpreter for the bench client (needs openai)")
    c.set_defaults(func=collect)

    b = sub.add_parser("build", help="emit a §3.7 bundle from the collected traces")
    b.add_argument("--out", type=Path, default=Path("outputs/rngd_edf_bundle"))
    b.add_argument("--harness-bundle", type=Path,
                   default=Path("profiler/perf/RNGD/meta-llama/Llama-3.1-8B/bf16/tp8"),
                   help="source of the within-decoder-layer distribution only")
    b.set_defaults(func=build)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
