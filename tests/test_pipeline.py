"""End-to-end tests for the world generation pipeline."""

from __future__ import annotations

from dataclasses import replace

from worldgen import generate
from worldgen.pipeline import GeneratedWorld
from worldgen.types import WorldgenConfig, WorldShape
from worldgen.world import rect_world_hexes


def _shape(side_km: float) -> WorldShape:
    return WorldShape(width_km=side_km, height_km=side_km)


def test_pipeline_deterministic(small_world_config: WorldgenConfig) -> None:
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


def test_pipeline_different_seeds(small_world_config: WorldgenConfig) -> None:
    """Different seeds → different worlds."""
    a = generate(config=small_world_config, seed=42)
    b = generate(config=small_world_config, seed=7)
    diffs = sum(1 for h in a.hexes if a.hexes[h].biome != b.hexes[h].biome)
    assert diffs > 50  # most hexes differ


def test_pipeline_produces_all_biome_categories(medium_world: GeneratedWorld) -> None:
    """At default settings, a moderate-sized world should produce hexes in all
    major biome categories: water, lowland, forest, elevated."""
    biomes_seen = {d.biome for d in medium_world.hexes.values()}
    # At least one water type:
    assert biomes_seen & {"ocean", "deep_ocean", "coast"}
    # At least one lowland biome:
    assert biomes_seen & {"plains", "grassland", "savanna", "desert", "tundra"}
    # At least one elevated:
    assert biomes_seen & {"hills", "mountain", "snow_peak"}


def test_pipeline_produces_rivers(medium_world: GeneratedWorld) -> None:
    """At default settings the pipeline should always produce *some* rivers
    (the rain-shadow + sink-fill combo should never leave a continent dry)."""
    river_count = sum(1 for d in medium_world.hexes.values() if d.is_river)
    assert river_count > 0


def test_pipeline_hex_count_matches_world_shape(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Pipeline assembles HexData for every hex in the rectangular world's
    footprint — and only those — so ``len(gen.hexes)`` equals
    ``len(rect_world_hexes(shape, hex_size_km))``."""
    for side_km in (50.0, 80.0, 130.0):
        cfg = replace(default_worldgen_config, world=_shape(side_km))
        gen = generate(config=cfg, seed=42)
        expected = len(rect_world_hexes(cfg.world, cfg.hex_size_km))
        assert len(gen.hexes) == expected


def test_pipeline_runs_at_tiny_world(default_worldgen_config: WorldgenConfig) -> None:
    """A tiny world (a handful of hexes) shouldn't crash — useful smoke test."""
    cfg = replace(default_worldgen_config, world=_shape(20.0))
    gen = generate(config=cfg, seed=42)
    assert len(gen.hexes) > 0
