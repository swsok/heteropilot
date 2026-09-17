"""Extract the prefill / decode bucket set of a furiosa-llm artifact.

Pure JSON: reads ``artifact.json`` only, so it runs under ``.venv`` with no
``furiosa_llm`` import and touches no device. Two artifact schemas are handled,
because the two vendor artifacts on this node were built by different
furiosa-llm versions:

* ``b62dbc1`` (Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct) -- buckets live in
  ``model.pipeline_metadata_list[i].attention_buckets``; the ``kernelwise``
  entry carries the full menu, the ``composed`` entries one bucket each.
* ``f285611`` (Llama-3.1-8B-Instruct-FP8) -- no metadata list; buckets are
  encoded in the pipeline names as ``-kv{K}-b{B}-attn{A}``.

Bucket classification follows furiosa_llm/metadata/config_types.py verbatim:
    input_ids_size = attention_size - kv_cache_size
    prefill  : kv_cache_size == 0
    decode   : kv_cache_size > 0 and input_ids_size == 1
    extend   : kv_cache_size > 0 and input_ids_size > 1   (chunked prefill)
``furiosa_llm.utils.compute_bucket_lengths`` counts extend with prefill.

``--yaml`` emits the grid in the machine-readable form the spike keeps beside the
perf bundle it was measured with (``WORK_ORDER_npu_exec_model_spike.md`` A7: the
grid is a property of the artifact, not of the hardware, so it does not belong in
``profiles/accelerators/*.yaml``).

Usage::

    python3 furiosa_artifact_buckets.py <artifact_dir_or_artifact.json> [--json]
    python3 furiosa_artifact_buckets.py <artifact_dir> --yaml > artifact_buckets.yaml
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"kv(\d+)-b(\d+)-attn(\d+)")


def kind(kv: int, input_ids: int) -> str:
    if kv == 0:
        return "prefill"
    return "decode" if input_ids == 1 else "extend"


def buckets_from_artifact(doc: dict) -> list[dict]:
    model = doc["model"]
    metas = model.get("pipeline_metadata_list") or []
    rows: list[dict] = []
    if metas and "attention_buckets" in metas[0]:
        for i, meta in enumerate(metas):
            for bucket in meta["attention_buckets"]:
                rows.append(
                    {
                        "pipeline": i,
                        "composition_type": meta.get("composition_type"),
                        "batch_size": bucket["batch_size"],
                        "attention_size": bucket["attention_size"],
                        "kv_cache_size": bucket["kv_cache_size"],
                    }
                )
    else:
        for i, pipeline in enumerate(model["pipelines"]):
            match = NAME_RE.search(pipeline["name"])
            if match is None:
                continue
            rows.append(
                {
                    "pipeline": i,
                    "composition_type": None,
                    "batch_size": int(match.group(2)),
                    "attention_size": int(match.group(3)),
                    "kv_cache_size": int(match.group(1)),
                }
            )
    for row in rows:
        row["input_ids_size"] = row["attention_size"] - row["kv_cache_size"]
        row["kind"] = kind(row["kv_cache_size"], row["input_ids_size"])
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (row["kind"], row["batch_size"], row["attention_size"], row["kv_cache_size"])
        seen.setdefault(key, row)
    return [seen[k] for k in sorted(seen)]


def kernelwise_menu(doc: dict) -> dict:
    """The composable pipeline's own menu: what a non-composed step can run on.

    The `kernelwise` pipeline is the one the runtime actually uses above
    concurrency 1 (D90), and its `tokenwise_buckets` ladder is the rung a decode
    batch is padded up to. That ladder is NOT the powers of two -- it has a 384
    between 256 and 512 -- so a consumer that assumes powers of two is wrong above
    256, which is why it is emitted rather than left to be re-derived.
    """
    for meta in doc["model"].get("pipeline_metadata_list") or []:
        if meta.get("composition_type") != "kernelwise":
            continue
        ladder = sorted(b["input_size"] for b in (meta.get("tokenwise_buckets") or []))
        if not ladder:
            # An emitted `tokenwise_buckets: []` reads as "nothing to pad to",
            # which is not what an artifact without the field is saying. Omit the
            # section so a consumer's own fallback fires and is visible.
            continue
        return {"attention_buckets": len(meta.get("attention_buckets") or []),
                "tokenwise_buckets": ladder}
    return {}


def emit_yaml(path: Path, doc: dict, rows: list[dict], out) -> None:
    """Write the grid as YAML, with the provenance a later reader needs.

    Hand-rolled rather than via ``yaml.safe_dump`` so the comment header and the
    row ordering survive; the payload is plain scalars and flow maps, which
    ``yaml.safe_load`` reads back unchanged (asserted by the test).
    """
    meta = doc["metadata"]
    parallel = doc["model"]["parallel_config"]
    decode = [r for r in rows if r["kind"] == "decode"]
    by_bs: dict[int, list[int]] = {}
    for row in decode:
        by_bs.setdefault(row["batch_size"], []).append(row["attention_size"])

    print("# Bucket grid of a furiosa-llm AOT artifact, extracted from artifact.json.", file=out)
    print("# Generated by experiments/scripts/furiosa_artifact_buckets.py --yaml"
          " -- do not hand-edit.", file=out)
    print("#", file=out)
    print("# This file sits beside the perf bundle measured while serving THIS artifact.", file=out)
    print("# Serving a different artifact changes the grid and therefore the execution", file=out)
    print("# model's input; it does not change the hardware."
          " See WORK_ORDER_npu_exec_model_spike.md A7.", file=out)
    print("schema_version: 1", file=out)
    print("source:", file=out)
    print(f"  artifact_name: {meta['name']}", file=out)
    print(f"  artifact_id: {meta['artifact_id']}", file=out)
    print(f"  artifact_path: {path}", file=out)
    print(f"  furiosa_llm_version: {meta['furiosa_llm_version']}", file=out)
    print(f"  furiosa_compiler_version: {meta['furiosa_compiler_version']}", file=out)
    print("  classification_rule: furiosa_llm/metadata/config_types.py", file=out)
    print("  extracted_by: experiments/scripts/furiosa_artifact_buckets.py", file=out)
    print(f"  extracted_on: {datetime.date.today().isoformat()}", file=out)
    print("parallel_config:", file=out)
    print(f"  tensor_parallel_size: {parallel['tensor_parallel_size']}", file=out)
    print(f"  pipeline_parallel_size: {parallel['pipeline_parallel_size']}", file=out)
    print("derived:", file=out)
    print("  # Every decode bucket is one compiled plan; bs x attention_size is the KV", file=out)
    print("  # budget it was compiled for. A batch whose largest KV exceeds the widest", file=out)
    print("  # attention bucket available at that batch size cannot run at that size.", file=out)
    print("  decode_max_attention_by_batch_size:", file=out)
    for bs in sorted(by_bs):
        print(f"    {bs}: {max(by_bs[bs])}", file=out)
    print("  decode_kv_budget_by_batch_size:", file=out)
    for bs in sorted(by_bs):
        print(f"    {bs}: {bs * max(by_bs[bs])}", file=out)
    menu = kernelwise_menu(doc)
    if menu:
        print("kernelwise:", file=out)
        print("  # The composable pipeline, which is the one the runtime uses above", file=out)
        print("  # concurrency 1 (D90). `tokenwise_buckets` is the ladder a decode batch", file=out)
        print("  # is padded up to, and it is NOT the powers of two -- note the 384.", file=out)
        print(f"  attention_buckets: {menu['attention_buckets']}", file=out)
        print(f"  tokenwise_buckets: [{', '.join(str(x) for x in menu['tokenwise_buckets'])}]",
              file=out)
    print("buckets:", file=out)
    for want in ("prefill", "extend", "decode"):
        group = [r for r in rows if r["kind"] == want]
        print(f"  {want}:", file=out)
        for row in group:
            print(
                f"    - {{batch_size: {row['batch_size']}, attention_size: {row['attention_size']},"
                f" kv_cache_size: {row['kv_cache_size']},"
                f" input_ids_size: {row['input_ids_size']}}}",
                file=out,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json", action="store_true", help="emit rows as JSON")
    parser.add_argument("--yaml", action="store_true",
                        help="emit the grid as YAML with a provenance header (A7 form)")
    args = parser.parse_args()

    path = args.artifact
    if path.is_dir():
        path = path / "artifact.json"
    doc = json.loads(path.read_text())
    meta = doc["metadata"]
    parallel = doc["model"]["parallel_config"]
    rows = dedup(buckets_from_artifact(doc))

    if args.yaml:
        emit_yaml(path, doc, rows, sys.stdout)
        return 0

    if args.json:
        json.dump(
            {"artifact": str(path), "metadata": meta, "parallel_config": parallel, "buckets": rows},
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print(f"artifact   : {path}")
    print(f"name       : {meta['name']}  (artifact_id {meta['artifact_id']})")
    print(f"built by   : furiosa-llm {meta['furiosa_llm_version']}, "
          f"compiler {meta['furiosa_compiler_version']}")
    print(f"parallelism: tp={parallel['tensor_parallel_size']} "
          f"pp={parallel['pipeline_parallel_size']}")
    for want in ("prefill", "extend", "decode"):
        group = [r for r in rows if r["kind"] == want]
        label = "extend (chunked prefill)" if want == "extend" else want
        print(f"\n## {label}: {len(group)} distinct buckets")
        for row in group:
            print(
                f"   bs={row['batch_size']:<4d} attn={row['attention_size']:<7d}"
                f" kv={row['kv_cache_size']:<7d} input_ids={row['input_ids_size']}"
            )
    prefill_max = max((r["attention_size"] for r in rows
                       if r["kind"] in ("prefill", "extend")), default=0)
    decode_max = max((r["attention_size"] for r in rows if r["kind"] == "decode"), default=0)
    print(f"\nmax_prefill_bucket_len = {prefill_max}   max_decode_bucket_len = {decode_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
