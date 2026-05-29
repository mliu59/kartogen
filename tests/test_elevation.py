"""Tests for the elevation layer.

Elevation is always driven by the tectonics simulation (plates mode is the
only mode). These tests go through ``pipeline.generate`` since the elevation
layer requires a ``LithosphereState`` as input.
"""

from __future__ import annotations

from worldgen import generate
from worldgen.types import WorldgenConfig


def test_elevation_deterministic(small_world_config: WorldgenConfig) -> None:
    """Same seed + config produces byte-identical elevation."""
    a = generate(config=small_world_config, seed=42)
    b = generate(config=small_world_config, seed=42)
    for h in a.hexes:
        assert a.elevation.elevation[h] == b.elevation.elevation[h]
    assert a.elevation.sea_level == b.elevation.sea_level


def test_elevation_different_seeds_diverge(small_world_config: WorldgenConfig) -> None:
    a = generate(config=small_world_config, seed=42)
    b = generate(config=small_world_config, seed=7)
    diffs = sum(
        1 for h in a.hexes
        if a.elevation.elevation[h] != b.elevation.elevation[h]
    )
    assert diffs > len(a.hexes) * 0.95
