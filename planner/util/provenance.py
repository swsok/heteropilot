"""Experiment metadata collection (work order §3.8).

Every result file must carry enough to reproduce it. Absolute rule 3 means the
absence of a value is recorded as `null` with a reason, never guessed at - a
provenance block that quietly invents a version is worse than one that admits
it could not find it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_commit() -> str | None:
    return _run(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool | None:
    status = _run(["git", "status", "--porcelain"])
    return None if status is None else bool(status.strip())


def upstream_commit() -> str | None:
    """The pinned upstream baseline, from the UPSTREAM_COMMIT file."""
    path = REPO_ROOT / "UPSTREAM_COMMIT"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def hash_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: list[str | Path]) -> str | None:
    """One digest over several files, order-independent.

    Used for `hardware_profile_hash`: which profiles were in play matters, the
    order they were listed in does not.
    """
    digests = sorted(d for d in (hash_file(p) for p in paths) if d is not None)
    if not digests:
        return None
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def hash_object(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def collect(
    *,
    service_spec_path: str | Path | None = None,
    cluster_spec_path: str | Path | None = None,
    profile_paths: list[str | Path] | None = None,
    dataset_path: str | Path | None = None,
    random_seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the §3.8 metadata block."""
    block: dict[str, Any] = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "llmservingsim_commit": upstream_commit(),
        "vllm_version": package_version("vllm"),
        "backend_version": package_version("torch"),
        "model_revision": None,  # set by the caller when a pinned revision is used
        "hardware_profile_hash": hash_paths(profile_paths or []),
        "cluster_spec_hash": hash_file(cluster_spec_path) if cluster_spec_path else None,
        "service_spec_hash": hash_file(service_spec_path) if service_spec_path else None,
        "dataset_hash": hash_file(dataset_path) if dataset_path else None,
        "random_seed": random_seed,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    if extra:
        block.update(extra)
    return block


def note_missing(block: dict[str, Any]) -> list[str]:
    """Which provenance fields could not be determined.

    Callers should surface this rather than let a null slip by unnoticed: a
    result whose upstream commit is unknown is not reproducible, and that should
    be visible at the time it is produced.
    """
    return sorted(k for k, v in block.items() if v is None)
