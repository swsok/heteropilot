"""Seed derivation and distribution samplers (DESIGN §3.2, FR-S2)."""

from __future__ import annotations

import math

import pytest

from scenariolab.generator.sampling import DistSpec, FloatRange, IntRange, derive_seed, rng_for


def test_derive_seed_deterministic() -> None:
    assert derive_seed(20260901, "cluster", 0) == derive_seed(20260901, "cluster", 0)


def test_derive_seed_distinct_paths() -> None:
    seeds = {
        derive_seed(1, "cluster", 0),
        derive_seed(1, "cluster", 1),
        derive_seed(1, "slo", 0),
        derive_seed(1, "scenario", 0, 0),
        derive_seed(1, "verify"),
        derive_seed(2, "cluster", 0),
    }
    assert len(seeds) == 6


def test_derive_seed_fits_sqlite_and_numpy() -> None:
    for i in range(200):
        seed = derive_seed(20260901, "scenario", i, i + 1)
        assert 0 <= seed < 2**63
        rng_for(seed)  # must not raise


def test_uniform_bounds() -> None:
    spec = DistSpec(dist="uniform", min=3.0, max=7.0)
    rng = rng_for(42)
    samples = [spec.sample(rng) for _ in range(500)]
    assert all(3.0 <= s <= 7.0 for s in samples)
    assert min(samples) < 4.0 and max(samples) > 6.0  # actually spreads


def test_loguniform_bounds_and_log_spread() -> None:
    spec = DistSpec(dist="loguniform", min=1.0, max=1000.0)
    rng = rng_for(42)
    samples = [spec.sample(rng) for _ in range(2000)]
    assert all(1.0 <= s <= 1000.0 for s in samples)
    # Log-scale median should sit near sqrt(min*max), far below the linear mean.
    samples.sort()
    median = samples[len(samples) // 2]
    assert 10.0 < median < 100.0, f"median {median} is not log-uniform-like"


def test_choice_membership() -> None:
    spec = DistSpec(dist="choice", values=[256, 512, 1024])
    rng = rng_for(7)
    drawn = {spec.sample(rng) for _ in range(200)}
    assert drawn == {256.0, 512.0, 1024.0}


def test_fixed() -> None:
    spec = DistSpec(dist="fixed", value=0.0)
    assert spec.sample(rng_for(1)) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dist": "uniform", "min": 5.0},                     # missing max
        {"dist": "uniform", "min": 7.0, "max": 3.0},         # inverted
        {"dist": "loguniform", "min": 0.0, "max": 10.0},     # log of zero
        {"dist": "choice"},                                  # missing values
        {"dist": "fixed"},                                   # missing value
    ],
)
def test_distspec_validation_errors(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DistSpec(**kwargs)


def test_ranges_sample_inclusive() -> None:
    rng = rng_for(3)
    ints = {IntRange(min=1, max=3).sample(rng) for _ in range(100)}
    assert ints == {1, 2, 3}
    floats = [FloatRange(min=0.5, max=1.0).sample(rng) for _ in range(100)]
    assert all(0.5 <= f <= 1.0 for f in floats)
    assert math.isclose(FloatRange(min=0.7, max=0.7).sample(rng), 0.7)
