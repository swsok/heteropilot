"""The committed RNGD bucket grid, and the emitter that produced it.

`profiler/perf/RNGD-CARD/.../artifact_buckets.yaml` is the machine-readable form
of the vendor artifact's compiled bucket set (work order A7: the grid belongs to
the artifact, not to the card, so it sits beside the perf bundle measured under
it). D90 reads three facts off that file, and two of them are load-bearing for
any later execution model:

* no compiled plan holds a prefill chunk and a decode token at once, so the
  runtime structurally cannot run a mixed step;
* `bs x attention_size` is a KV budget, which caps the executable decode batch
  below the `max_num_seqs 256` `rps_aware`'s E6 handed to RNGD candidates.

These run on any node: they read the committed YAML, never a device and never the
HuggingFace cache. The emitter itself is exercised on a synthetic artifact
document for the same reason.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "profiler/perf/RNGD-CARD/meta-llama/Llama-3.1-8B/bf16/artifact_buckets.yaml"
SCRIPT = ROOT / "experiments/scripts/furiosa_artifact_buckets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("furiosa_artifact_buckets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grid() -> dict:
    return yaml.safe_load(GRID.read_text())


def test_grid_records_the_artifact_it_came_from(grid: dict) -> None:
    """A grid with no artifact id cannot be told apart from another build's."""
    source = grid["source"]
    assert source["artifact_id"].startswith("d6ae6a43")
    assert source["furiosa_llm_version"] == "b62dbc1"
    assert grid["parallel_config"]["tensor_parallel_size"] == 8


def test_bucket_counts(grid: dict) -> None:
    buckets = grid["buckets"]
    assert (len(buckets["prefill"]), len(buckets["extend"]), len(buckets["decode"])) == (8, 74, 46)


def test_classification_is_self_consistent(grid: dict) -> None:
    """`input_ids = attention - kv` and the three kinds partition on it."""
    for kind, rows in grid["buckets"].items():
        for row in rows:
            assert row["input_ids_size"] == row["attention_size"] - row["kv_cache_size"]
            if kind == "prefill":
                assert row["kv_cache_size"] == 0
            elif kind == "decode":
                assert row["kv_cache_size"] > 0 and row["input_ids_size"] == 1
            else:
                assert row["kv_cache_size"] > 0 and row["input_ids_size"] > 1


def test_no_compiled_plan_mixes_prefill_and_decode(grid: dict) -> None:
    """D90's structural claim. If this fails the mixed-step finding is void."""
    buckets = grid["buckets"]
    assert all(row["batch_size"] == 1 for row in buckets["prefill"] + buckets["extend"])
    assert all(row["input_ids_size"] == 1 for row in buckets["decode"])


def test_decode_grid_is_a_kv_budget(grid: dict) -> None:
    """The derived table must agree with the buckets it summarises."""
    widest: dict[int, int] = {}
    for row in grid["buckets"]["decode"]:
        bs = row["batch_size"]
        widest[bs] = max(widest.get(bs, 0), row["attention_size"])
    assert grid["derived"]["decode_max_attention_by_batch_size"] == widest
    assert grid["derived"]["decode_kv_budget_by_batch_size"] == {
        bs: bs * attn for bs, attn in widest.items()
    }
    assert set(widest.values()) != {max(widest.values())}, "a flat ladder is not a budget"


def test_sharegpt_cannot_reach_batch_256(grid: dict) -> None:
    """D90(3): mean KV on sharegpt is ~2200, and bs=256 tops out at attn=2048.

    So the largest decode batch that workload can execute is 128, and E6's
    `max_num_seqs 256` named an operating point this artifact has no plan for.
    """
    widest = grid["derived"]["decode_max_attention_by_batch_size"]
    sharegpt_mean_kv = 2200
    usable = [bs for bs, attn in widest.items() if attn > sharegpt_mean_kv]
    assert max(usable) == 128
    assert widest[256] == 2048


def test_emitter_round_trips(tmp_path: Path) -> None:
    """The emitter is hand-rolled to keep its header; YAML must still read it."""
    module = _load_script()
    doc = {
        "metadata": {
            "name": "vendor/Model",
            "artifact_id": "0000beef-0000-0000-0000-000000000000",
            "furiosa_llm_version": "deadbee",
            "furiosa_compiler_version": "cafe1234",
        },
        "model": {
            "parallel_config": {"tensor_parallel_size": 4, "pipeline_parallel_size": 2},
            "pipeline_metadata_list": [
                {
                    "composition_type": "kernelwise",
                    "attention_buckets": [
                        {"batch_size": 1, "attention_size": 128, "kv_cache_size": 0},
                        {"batch_size": 2, "attention_size": 1024, "kv_cache_size": 1023},
                        {"batch_size": 1, "attention_size": 256, "kv_cache_size": 128},
                    ],
                }
            ],
        },
    }
    rows = module.dedup(module.buckets_from_artifact(doc))
    out = io.StringIO()
    module.emit_yaml(Path("/nowhere/artifact.json"), doc, rows, out)
    parsed = yaml.safe_load(out.getvalue())

    assert parsed["source"]["artifact_id"] == doc["metadata"]["artifact_id"]
    assert parsed["parallel_config"]["pipeline_parallel_size"] == 2
    assert len(parsed["buckets"]["prefill"]) == 1
    assert len(parsed["buckets"]["extend"]) == 1
    assert parsed["buckets"]["decode"] == [
        {"batch_size": 2, "attention_size": 1024, "kv_cache_size": 1023, "input_ids_size": 1}
    ]
    assert parsed["derived"]["decode_kv_budget_by_batch_size"] == {2: 2048}
