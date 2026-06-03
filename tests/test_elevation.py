"""Tests for the elevation layer.

Elevation is always driven by the tectonics simulation (plates mode is the
only mode). These tests go through ``pipeline.generate`` since the elevation
layer requires a ``LithosphereState`` as input.
"""

from __future__ import annotations

import pytest
from kartogen import generate
from kartogen.types import KartogenConfig

pytestmark = pytest.mark.slow  # full generate()/sim per test
def test_elevation_deterministic(small_world_config: KartogenConfig) -> None:
    """Same seed + config produces byte-identical elevation."""
    a = generate(config=small_world_config, seed=42)
    b = generate(config=small_world_config, seed=42)
    for h in a.hexes:
        assert a.elevation.elevation[h] == b.elevation.elevation[h]
    assert a.elevation.sea_level == b.elevation.sea_level


