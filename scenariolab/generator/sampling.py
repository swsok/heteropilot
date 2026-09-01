"""Seed derivation and shared distribution samplers (DESIGN §3.2, FR-S2).

Every random draw in ScenarioLab flows through this module so that:
  * one master seed determines the whole batch (byte-identical reruns),
  * scenario i is unchanged when the batch grows (independent derived seeds),
  * a single scenario can be re-run in isolation from its own seed.

The derivation is SHA-256 over a canonical string, truncated to the low
8 bytes and masked to 63 bits so a derived seed fits both numpy's uint64 and
SQLite's signed 64-bit INTEGER. It is fixed here as a constant contract:
changing it invalidates every recorded seed, so treat it like a file format.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SEED_DERIVATION = "sha256-low8-63bit"
_SEED_MASK = (1 << 63) - 1


def derive_seed(master_seed: int, *parts: str | int) -> int:
    """Derive a 63-bit child seed from the master seed and a label path.

    derive_seed(s, "cluster", i) / (s, "slo", j) / (s, "scenario", i, j) /
    (s, "verify") per DESIGN §3.2.
    """
    text = ":".join([str(int(master_seed)), *[str(p) for p in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[-8:], "big") & _SEED_MASK


def rng_for(seed: int) -> np.random.Generator:
    """The one RNG constructor ScenarioLab uses (PCG64 via default_rng)."""
    return np.random.default_rng(seed)


class DistSpec(BaseModel):
    """One sampled field in LabConfig: uniform / loguniform / choice / fixed."""

    model_config = ConfigDict(extra="forbid")

    dist: Literal["uniform", "loguniform", "choice", "fixed"]
    min: float | None = None
    max: float | None = None
    values: list[float] | None = Field(default=None, min_length=1)
    value: float | None = None

    @model_validator(mode="after")
    def _fields_for_dist(self) -> DistSpec:
        if self.dist in ("uniform", "loguniform"):
            if self.min is None or self.max is None:
                raise ValueError(f"dist={self.dist} requires min and max")
            if self.min > self.max:
                raise ValueError(f"dist={self.dist}: min={self.min} > max={self.max}")
            if self.dist == "loguniform" and self.min <= 0:
                raise ValueError("loguniform requires min > 0")
        elif self.dist == "choice" and self.values is None:
            raise ValueError("dist=choice requires values")
        elif self.dist == "fixed" and self.value is None:
            raise ValueError("dist=fixed requires value")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        if self.dist == "uniform":
            assert self.min is not None and self.max is not None
            return float(rng.uniform(self.min, self.max))
        if self.dist == "loguniform":
            assert self.min is not None and self.max is not None
            return float(math.exp(rng.uniform(math.log(self.min), math.log(self.max))))
        if self.dist == "choice":
            assert self.values is not None
            return float(self.values[int(rng.integers(len(self.values)))])
        assert self.value is not None
        return float(self.value)


class IntRange(BaseModel):
    """Inclusive integer range, sampled uniformly."""

    model_config = ConfigDict(extra="forbid")

    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> IntRange:
        if self.min > self.max:
            raise ValueError(f"min={self.min} > max={self.max}")
        return self

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.min, self.max + 1))


class FloatRange(BaseModel):
    """Inclusive float range, sampled uniformly."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> FloatRange:
        if self.min > self.max:
            raise ValueError(f"min={self.min} > max={self.max}")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.min, self.max))
