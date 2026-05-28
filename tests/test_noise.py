"""Tests for the deterministic Perlin noise module."""

from __future__ import annotations

import random

import pytest

from worldgen.noise import PerlinNoise2D, fbm, ridged_fbm


@pytest.fixture
def noise() -> PerlinNoise2D:
    return PerlinNoise2D.from_rng(random.Random(0))


def test_noise_is_bounded(noise: PerlinNoise2D) -> None:
    """Perlin output should be approximately in [-1, 1]."""
    rng = random.Random(1)
    samples = [noise.sample(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(2000)]
    assert max(samples) <= 1.01
    assert min(samples) >= -1.01


def test_noise_zero_at_integer_lattice(noise: PerlinNoise2D) -> None:
    """At integer coordinates the gradient dot products vanish → noise = 0."""
    for ix in range(-5, 6):
        for iy in range(-5, 6):
            assert abs(noise.sample(float(ix), float(iy))) < 1e-9


def test_noise_continuous(noise: PerlinNoise2D) -> None:
    """Noise should be C0-continuous: nearby points have nearby values."""
    rng = random.Random(2)
    for _ in range(100):
        x = rng.uniform(-20, 20)
        y = rng.uniform(-20, 20)
        v0 = noise.sample(x, y)
        v1 = noise.sample(x + 0.001, y)
        # Gradient is bounded for Perlin; small step → small change.
        assert abs(v0 - v1) < 0.05


def test_noise_seed_reproducible() -> None:
    """Same seed → same noise field, byte-for-byte."""
    n1 = PerlinNoise2D.from_rng(random.Random(123))
    n2 = PerlinNoise2D.from_rng(random.Random(123))
    for x in (-3.7, 0.0, 4.4, 11.1):
        for y in (-2.2, 1.5, 8.8):
            assert n1.sample(x, y) == n2.sample(x, y)


def test_noise_seed_diverges() -> None:
    """Different seeds → different noise fields.

    Sample at non-integer points, since Perlin vanishes at integer lattice
    coordinates regardless of seed.
    """
    n1 = PerlinNoise2D.from_rng(random.Random(1))
    n2 = PerlinNoise2D.from_rng(random.Random(2))
    diff = sum(abs(n1.sample(x + 0.37, x * 0.5 + 0.21) - n2.sample(x + 0.37, x * 0.5 + 0.21))
               for x in range(40))
    assert diff > 1.0


def test_fbm_bounded(noise: PerlinNoise2D) -> None:
    rng = random.Random(3)
    samples = [
        fbm(noise, rng.uniform(-50, 50), rng.uniform(-50, 50),
            octaves=6, base_frequency=0.05)
        for _ in range(500)
    ]
    assert max(samples) <= 1.01
    assert min(samples) >= -1.01


def test_fbm_one_octave_matches_single_noise(noise: PerlinNoise2D) -> None:
    """fBm with 1 octave at frequency f equals a single noise(x*f, y*f) call."""
    for x, y in [(0.3, 0.4), (1.7, -2.1), (5.5, 5.5)]:
        direct = noise.sample(x * 1.5, y * 1.5)
        via_fbm = fbm(noise, x, y, octaves=1, base_frequency=1.5)
        assert abs(direct - via_fbm) < 1e-12


def test_fbm_octaves_yield_different_output(noise: PerlinNoise2D) -> None:
    """6-octave fBm should differ measurably from 1-octave fBm at the same point."""
    diff = 0.0
    for x in range(0, 200):
        a = fbm(noise, x * 0.1 + 0.13, 0.7, octaves=1, base_frequency=0.5)
        b = fbm(noise, x * 0.1 + 0.13, 0.7, octaves=6, base_frequency=0.5)
        diff += abs(a - b)
    assert diff > 0.5


def test_ridged_fbm_non_negative(noise: PerlinNoise2D) -> None:
    """Ridged multifractal outputs are ``1 - |noise|`` style — non-negative."""
    rng = random.Random(5)
    for _ in range(500):
        x = rng.uniform(-50, 50)
        y = rng.uniform(-50, 50)
        v = ridged_fbm(noise, x, y, octaves=4, base_frequency=0.05)
        assert v >= 0.0
