"""End-to-end tests for the world generation pipeline."""

from __future__ import annotations

from dataclasses import replace

import pytest
from kartogen import generate
from kartogen.types import KartogenConfig, WorldShape
from kartogen.world import rect_world_hexes

pytestmark = pytest.mark.slow  # full generate()/sim per test
def _shape(side_km: float) -> WorldShape:
    return WorldShape(width_km=side_km, height_km=side_km)


def test_pipeline_deterministic(small_world_config: KartogenConfig) -> None:
    """Same (seed, config) produces byte-identical output."""
    a = generate(config=small_world_config, seed=42)
    b = generate(config=small_world_config, seed=42)
    for h in a.hexes:
        da, db = a.hexes[h], b.hexes[h]
        assert da.elevation == db.elevation
        assert da.is_ocean == db.is_ocean
        assert da.is_river == db.is_river
        assert da.is_lake == db.is_lake
        assert da.temperature_c == db.temperature_c
        assert da.precipitation_mm == db.precipitation_mm
        assert da.flow_accumulation == db.flow_accumulation
        assert da.biome == db.biome


def test_pipeline_hex_count_matches_world_shape(
    default_kartogen_config: KartogenConfig,
) -> None:
    """Pipeline assembles HexData for every hex in the rectangular world's
    footprint — and only those — so ``len(gen.hexes)`` equals
    ``len(rect_world_hexes(shape, hex_size_km))``."""
    for side_km in (50.0, 80.0, 130.0):
        cfg = replace(default_kartogen_config, world=_shape(side_km))
        gen = generate(config=cfg, seed=42)
        expected = len(rect_world_hexes(cfg.world, cfg.hex_size_km))
        assert len(gen.hexes) == expected


def test_pipeline_runs_at_tiny_world(default_kartogen_config: KartogenConfig) -> None:
    """A tiny world (a handful of hexes) shouldn't crash — useful smoke test."""
    cfg = replace(default_kartogen_config, world=_shape(20.0))
    gen = generate(config=cfg, seed=42)
    assert len(gen.hexes) > 0
