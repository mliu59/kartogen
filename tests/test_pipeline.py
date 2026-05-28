"""End-to-end tests for the world generation pipeline."""

from __future__ import annotations

from worldgen import generate
from worldgen.pipeline import GeneratedWorld
from worldgen.types import WorldgenConfig


def test_pipeline_deterministic(default_worldgen_config: WorldgenConfig) -> None:
    """Same (seed, config, radius) produces byte-identical output."""
    a = generate(radius=12, config=default_worldgen_config, seed=42)
    b = generate(radius=12, config=default_worldgen_config, seed=42)
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


def test_pipeline_different_seeds(default_worldgen_config: WorldgenConfig) -> None:
    """Different seeds → different worlds."""
    a = generate(radius=12, config=default_worldgen_config, seed=42)
    b = generate(radius=12, config=default_worldgen_config, seed=7)
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


def test_pipeline_hex_count_matches_radius(default_worldgen_config: WorldgenConfig) -> None:
    """A hex grid of radius R contains 3R² + 3R + 1 hexes."""
    for r in (3, 5, 8):
        gen = generate(radius=r, config=default_worldgen_config, seed=42)
        expected = 3 * r * r + 3 * r + 1
        assert len(gen.hexes) == expected


def test_pipeline_runs_at_radius_one(default_worldgen_config: WorldgenConfig) -> None:
    """A radius-1 world (7 hexes) shouldn't crash — useful as a smoke test."""
    gen = generate(radius=1, config=default_worldgen_config, seed=42)
    assert len(gen.hexes) == 7
