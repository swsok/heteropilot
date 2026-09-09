"""Where a simulated candidate actually ran, per hardware kind.

`WORK_ORDER_rps_aware.md` rev 2 STEP 4.3. The safety margin a candidate deserves
depends on how loaded it was: the simulator is 3.1 % optimistic on TPOT at served
concurrency 16.6 and 18 % at 76 (`profiles/calibration/rngd_card_edf.yaml`). A
hand-set `--tpot-margin-percent` cannot know which, so it is either too loose
somewhere or too tight everywhere. This reads the operating point off the run.

Concurrency is always SERVED -- Little's law, `sum(residence) / wall` -- because
requested concurrency is what D22 retracted.

**The phase decomposition is what makes P/D work.** The simulator's CSV records one
row per request and attributes it to the instance that completed it, so a
disaggregated run books every request against the decode engine and the prefill
engine's load is invisible to a naive grouping. But a request's residence splits
exactly:

    latency = TTFT + (latency - TTFT)

and Little's law applies to each phase on its own. `sum(TTFT)/wall` is the mean
number of requests in the pre-first-token phase -- the prefill engine's occupancy --
and `sum(latency - TTFT)/wall` is the decode engine's. For an aggregated
deployment both phases run on the same hardware and the operating point is the
whole `sum(latency)/wall`.
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path

#: When several instances of one hardware kind run at different loads, the
#: hardware's operating point is the BUSIEST of them. The accuracy domain's error
#: grows with concurrency, so the max is the conservative choice -- a margin sized
#: for the least-loaded replica would under-protect the others.
_AGGREGATE = max


@dataclass(frozen=True)
class HardwareOperatingPoint:
    """One hardware kind's served concurrency over a simulated run."""

    hardware: str
    concurrency: float
    #: "total" for an aggregated deployment, "prefill" / "decode" for P/D.
    phase: str
    requests: int
    wall_s: float


def _instances(config: dict) -> list[dict]:
    """The compiled instances, flattened in the order the simulator numbers them.

    Read from the compiled config rather than re-derived from the candidate's
    assignments: `compile_to_sim_config` groups by node and emits nodes in cluster
    order, so the flattened index does NOT follow assignment order. On the
    `pd-rngd-gpu` fixture, index 0 is the RNGD decode instance and index 1 the A40
    prefill one, which is the reverse of how the candidate names them.
    """
    return [inst for node in config.get("nodes", []) for inst in node.get("instances", [])]


def operating_points(
    csv_path: str | Path, config: dict
) -> dict[str, HardwareOperatingPoint]:
    """Per-hardware operating points for one simulated candidate.

    Returns {} when the CSV is missing or has no usable rows -- callers treat that
    as "no operating point known", which means margin 0 and a caveat, never a
    guessed margin.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    with path.open() as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return {}

    parsed = []
    for r in rows:
        try:
            parsed.append((
                int(r["instance id"]),
                float(r["arrival"]),
                float(r["end_time"]),
                float(r["latency"]),
                float(r["TTFT"]),
            ))
        except (KeyError, ValueError):
            continue
    if not parsed:
        return {}

    wall_ns = max(p[2] for p in parsed) - min(p[1] for p in parsed)
    if wall_ns <= 0:
        return {}
    wall_s = wall_ns / 1e9

    instances = _instances(config)
    prefill_hw = sorted({i["hardware"] for i in instances if i.get("pd_type") == "prefill"})
    decode_hw = sorted({i["hardware"] for i in instances if i.get("pd_type") == "decode"})

    if prefill_hw or decode_hw:
        return _pd_points(parsed, instances, prefill_hw, decode_hw, wall_ns, wall_s)
    return _aggregated_points(parsed, instances, wall_ns, wall_s)


def _aggregated_points(
    parsed: list[tuple[int, float, float, float, float]],
    instances: list[dict],
    wall_ns: float,
    wall_s: float,
) -> dict[str, HardwareOperatingPoint]:
    per_instance: dict[int, list[float]] = {}
    for idx, _arr, _end, lat, _ttft in parsed:
        per_instance.setdefault(idx, []).append(lat)

    by_hw: dict[str, list[tuple[float, int]]] = {}
    for idx, lats in per_instance.items():
        if idx >= len(instances):
            continue
        hw = instances[idx]["hardware"]
        by_hw.setdefault(hw, []).append((sum(lats) / wall_ns, len(lats)))

    return {
        hw: HardwareOperatingPoint(
            hardware=hw,
            concurrency=_AGGREGATE(c for c, _ in vals),
            phase="total",
            requests=sum(n for _, n in vals),
            wall_s=wall_s,
        )
        for hw, vals in by_hw.items()
    }


def _pd_points(
    parsed: list[tuple[int, float, float, float, float]],
    instances: list[dict],
    prefill_hw: list[str],
    decode_hw: list[str],
    wall_ns: float,
    wall_s: float,
) -> dict[str, HardwareOperatingPoint]:
    """Split the residence time by phase and give each phase to its hardware.

    Replica counts divide the load: `dp_replicas` engines of one role share the
    phase's occupancy, so a single engine sits at the phase total over the count.
    """
    prefill_occ = sum(ttft for *_rest, ttft in parsed) / wall_ns
    decode_occ = sum(lat - ttft for *_r, lat, ttft in parsed) / wall_ns
    n = len(parsed)

    def count(role: str, hw: str) -> int:
        return sum(1 for i in instances
                   if i.get("pd_type") == role and i["hardware"] == hw)

    out: dict[str, HardwareOperatingPoint] = {}
    for hw in prefill_hw:
        k = max(1, count("prefill", hw))
        out[hw] = HardwareOperatingPoint(hw, prefill_occ / k, "prefill", n, wall_s)
    for hw in decode_hw:
        k = max(1, count("decode", hw))
        existing = out.get(hw)
        point = HardwareOperatingPoint(hw, decode_occ / k, "decode", n, wall_s)
        # Same hardware on both sides (homogeneous P/D): it carries both phases,
        # so its operating point is the sum, not one of them.
        if existing is not None:
            point = HardwareOperatingPoint(
                hw, existing.concurrency + point.concurrency, "total", n, wall_s
            )
        out[hw] = point
    return out
