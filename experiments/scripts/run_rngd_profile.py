"""Run the RNGD layerwise profile in parallel across PEs, then merge the shards.

Each ``profile_rngd.py`` worker holds exactly one PE and takes every
``num_shards``-th task, so the work list is partitioned with no locking and no
shared state -- the same discipline ``planner/util/parallel.py`` uses for
candidate simulations.

Two placement rules matter:

* **One worker per PE.** A PE accepts a single context; a second process on it
  fails with ``EBUSY``.
* **Spread workers across cards.** Each worker's attention shots may hold up to
  ``--kv-budget-gb`` of K/V, and the 47.5 GiB is per *card*, shared by its 8
  PEs. Assigning round-robin over cards keeps the per-card worst case at
  ``ceil(workers / cards) * kv_budget``, which this script checks before
  launching rather than discovering as an OOM mid-run.

Availability is re-checked immediately before launch: PEs come and go on this
shared node (``docs/hardware_roadmap.md``, "Who holds the NPUs"), and a PE is
claimed iff its ``alloc_status`` sysfs file is non-empty.

Usage::

    PYTHONPATH=$PWD python3 experiments/scripts/run_rngd_profile.py --tp 1
    PYTHONPATH=$PWD python3 experiments/scripts/run_rngd_profile.py \
        --tp 1 --workers 16 --out outputs/rngd_profile
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILER = REPO_ROOT / "experiments" / "scripts" / "profile_rngd.py"

PES_PER_CARD = 8
CARDS = 4
CARD_MEMORY_GB = 47.5

#: Files each shard writes, with the columns to merge on.
CONTRACT_FILES = {
    "dense.csv": ("layer", "tokens"),
    "per_sequence.csv": ("layer", "sequences"),
    "attention.csv": ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode"),
}
BREAKDOWN_FILES = {
    "breakdown_dense_tp{tp}.csv": ("layer", "tokens"),
    "breakdown_per_sequence_tp{tp}.csv": ("layer", "sequences"),
    "breakdown_attention_tp{tp}.csv": ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode"),
}


def pe_is_free(pe: int) -> bool:
    """A PE is claimed iff its ``alloc_status`` holds an allocation table."""
    card, index = divmod(pe, PES_PER_CARD)
    path = Path(f"/sys/class/rngd_mgmt/rngd!npu{card}pe{index}/alloc_status")
    try:
        return not path.read_text().strip()
    except OSError:
        return False


def free_pes() -> list[int]:
    return [pe for pe in range(CARDS * PES_PER_CARD) if pe_is_free(pe)]


def pick_devices(available: list[int], workers: int) -> list[int]:
    """Round-robin over cards, so no card hosts many more workers than others."""
    by_card: dict[int, list[int]] = {}
    for pe in available:
        by_card.setdefault(pe // PES_PER_CARD, []).append(pe)
    picked: list[int] = []
    while len(picked) < workers:
        progressed = False
        for card in sorted(by_card):
            if by_card[card] and len(picked) < workers:
                picked.append(by_card[card].pop(0))
                progressed = True
        if not progressed:
            break
    return picked


def merge_csv(sources: list[Path], dest: Path, key_columns: tuple[str, ...]) -> int:
    """Concatenate shard CSVs, dropping duplicate keys, sorted for determinism."""
    header: list[str] | None = None
    rows: dict[tuple, dict] = {}
    for source in sources:
        if not source.is_file():
            continue
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = list(reader.fieldnames)
            for row in reader:
                rows[tuple(row[c] for c in key_columns)] = row
    if header is None:
        return 0

    def sort_key(item: tuple) -> tuple:
        out = []
        for value in item:
            try:
                out.append((0, float(value), ""))
            except ValueError:
                out.append((1, 0.0, value))
        return tuple(out)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for key in sorted(rows, key=sort_key):
            writer.writerow(rows[key])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--out", type=Path, default=Path("outputs/rngd_profile"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--max-kv", type=int, default=8192)
    parser.add_argument("--kv-budget-gb", type=float, default=4.0)
    parser.add_argument("--poll-s", type=float, default=15.0)
    args = parser.parse_args()

    available = free_pes()
    print(f"free PEs ({len(available)}): {available}")
    if not available:
        raise SystemExit("no free PE: every PE is claimed, nothing to run on")
    workers = min(args.workers, len(available))
    if workers < args.workers:
        print(f"only {workers} of the requested {args.workers} workers fit the free PEs")
    devices = pick_devices(available, workers)

    per_card: dict[int, int] = {}
    for pe in devices:
        per_card[pe // PES_PER_CARD] = per_card.get(pe // PES_PER_CARD, 0) + 1
    worst = max(per_card.values()) * args.kv_budget_gb
    print("worker placement: " + ", ".join(
        f"npu{card}x{count}" for card, count in sorted(per_card.items())))
    print(f"worst-case KV per card: {worst:.1f} GB of {CARD_MEMORY_GB} GB")
    if worst > CARD_MEMORY_GB * 0.8:
        raise SystemExit(
            f"placement would risk OOM ({worst:.1f} GB on one card). Lower "
            f"--workers or --kv-budget-gb."
        )

    shard_root = args.out / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    procs = []
    for shard, pe in enumerate(devices):
        shard_out = shard_root / f"shard{shard:02d}"
        cmd = [
            sys.executable, "-u", str(PROFILER),
            "--model", args.model, "--tp", str(args.tp),
            "--device", f"rngd:{pe}", "--reps", str(args.reps),
            "--out", str(shard_out),
            "--max-tokens", str(args.max_tokens),
            "--max-seqs", str(args.max_seqs),
            "--max-kv", str(args.max_kv),
            "--kv-budget-gb", str(args.kv_budget_gb),
            "--shard", str(shard), "--num-shards", str(len(devices)),
        ]
        log_path = shard_root / f"shard{shard:02d}.out"
        handle = log_path.open("w")
        procs.append((shard, pe, subprocess.Popen(cmd, stdout=handle, stderr=handle,
                                                 cwd=REPO_ROOT), handle, shard_out))
        print(f"  shard {shard:02d} -> rngd:{pe}")

    print(f"launched {len(procs)} workers; polling every {args.poll_s:.0f}s")
    remaining = {shard for shard, *_ in procs}
    while remaining:
        time.sleep(args.poll_s)
        for shard, pe, proc, _handle, _shard_out in procs:
            if shard not in remaining or proc.poll() is None:
                continue
            remaining.discard(shard)
            status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
            print(f"  shard {shard:02d} (rngd:{pe}) finished: {status}")
        done = len(procs) - len(remaining)
        if remaining:
            measured = sum(
                1 for shard, *_ in procs
                for line in (shard_root / f"shard{shard:02d}.out").read_text().splitlines()
                if "time=" in line
            )
            print(f"  [{time.time() - started:6.0f}s] {done}/{len(procs)} workers done, "
                  f"{measured} shapes measured")

    for _shard, _pe, _proc, handle, _out in procs:
        handle.close()

    failures = [shard for shard, _pe, proc, *_ in procs if proc.returncode != 0]
    shard_outs = [out for *_rest, out in procs]

    print("\nmerging shards")
    merged_rows = {}
    for filename, keys in CONTRACT_FILES.items():
        sources = [out / f"tp{args.tp}" / filename for out in shard_outs]
        dest = args.out / f"tp{args.tp}" / filename
        merged_rows[filename] = merge_csv(sources, dest, keys)
        print(f"  {filename:20s} {merged_rows[filename]:5d} rows -> {dest}")
    for template, keys in BREAKDOWN_FILES.items():
        filename = template.format(tp=args.tp)
        sources = [out / filename for out in shard_outs]
        merge_csv(sources, args.out / filename, keys)

    skipped: list[dict] = []
    powers: list[dict] = []
    for out in shard_outs:
        summary_path = out / f"summary_tp{args.tp}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        skipped += summary.get("attention_shots_skipped_over_kv_budget") or []
        powers.append({"device": summary.get("device"), **(summary.get("power_w") or {})})

    elapsed = time.time() - started
    merged_summary = {
        "model": args.model, "tp": args.tp,
        "workers": len(procs), "devices": [f"rngd:{pe}" for pe in devices],
        "elapsed_s": round(elapsed, 1),
        "rows": merged_rows,
        "failed_shards": failures,
        "attention_shots_skipped_over_kv_budget": skipped,
        "per_worker_power_w": powers,
        "reps": args.reps,
    }
    (args.out / f"summary_tp{args.tp}.json").write_text(
        json.dumps(merged_summary, indent=2) + "\n")
    print(f"\ndone in {elapsed:.0f}s")
    if failures:
        print(f"WARNING: shards {failures} exited non-zero; the merged bundle may "
              f"be missing their tasks. Check {shard_root}/shardNN.out")
    if skipped:
        print(f"{len(skipped)} attention shot(s) skipped over the KV budget -- the "
              f"grid is not complete at the largest corners")


if __name__ == "__main__":
    main()
