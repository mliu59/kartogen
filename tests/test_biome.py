"""Tests for biome assignment and Whittaker classification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from worldgen import generate
from worldgen.terrain import TERRAIN_NAMES
from worldgen import biome as biome_layer
from worldgen.pipeline import GeneratedWorld
from worldgen.types import WorldgenConfig


def test_every_biome_name_is_known(small_world: GeneratedWorld) -> None:
    """Every assigned biome must appear in TERRAIN_NAMES — otherwise the
    engine's terrain_types config lookup will fail."""
    known = set(TERRAIN_NAMES)
    for d in small_world.hexes.values():
        assert d.biome in known


@pytest.fixture(scope="module")
def pole_to_pole_world(medium_world_config: WorldgenConfig) -> GeneratedWorld:
    """A medium world with a true pole-to-pole latitude window so polar
    hexes actually sit at near-90° latitude. (The bundled config uses an
    asymmetric mid-latitude slice.)"""
    cfg = replace(medium_world_config, map_lat_min=-90.0, map_lat_max=90.0)
    return generate(config=cfg, seed=42)


def test_polar_hexes_are_tundra_or_snow_or_water(
    pole_to_pole_world: GeneratedWorld,
) -> None:
    """At |latitude| > 75°, land hexes must be tundra / snow / cold-band biomes
    (no jungle / savanna / plains). Uses a pole-to-pole world so this test
    isn't fooled by an asymmetric latitude window in the bundled config."""
    from worldgen.climate import hex_latitude_deg
    from worldgen.world import map_half_extents_km
    cfg = pole_to_pole_world.config
    _, half_h = map_half_extents_km(pole_to_pole_world.hexes.keys(), cfg.hex_size_km)
    for h, d in pole_to_pole_world.hexes.items():
        if abs(hex_latitude_deg(h, half_h, cfg)) <= 75.0:
            continue
        if d.is_ocean:
            continue
        assert d.biome in {
            "tundra", "snow_peak", "mountain", "hills",
            "taiga", "lake", "river", "coast",
        }, f"polar hex at {h} got biome={d.biome}"


def test_high_elevation_overrides_lowland_biome(medium_world: GeneratedWorld) -> None:
    """The top elevation quantile should be mountain or snow_peak — never plains/desert."""
    config = medium_world.config
    elev_layer = medium_world.elevation
    sea_layer = medium_world.sea

    land_elevs = sorted(
        elev_layer.elevation[h] - elev_layer.sea_level
        for h in elev_layer.elevation
        if not sea_layer.is_ocean[h]
    )
    if not land_elevs:
        return
    n = len(land_elevs)
    mtn_threshold = land_elevs[int(config.elevation_mountain_threshold * (n - 1))]

    forbidden = {"plains", "grassland", "savanna", "desert", "jungle",
                 "temperate_forest", "taiga"}
    for h, d in medium_world.hexes.items():
        if sea_layer.is_ocean[h] or d.is_river or d.is_lake:
            continue
        elev_above_sea = elev_layer.elevation[h] - elev_layer.sea_level
        if elev_above_sea >= mtn_threshold:
            assert d.biome not in forbidden, (
                f"hex at {h} elev={elev_above_sea:.3f} classed as {d.biome}"
            )


def test_whittaker_lookup_basic(default_worldgen_config: WorldgenConfig) -> None:
    """Spot-check Whittaker for known (T, P) combinations."""
    cfg = default_worldgen_config
    # Tropical wet → jungle
    assert biome_layer._whittaker(28.0, 2000.0, cfg) == "jungle"
    # Tropical dry → desert
    assert biome_layer._whittaker(28.0, 50.0, cfg) == "desert"
    # Polar → tundra regardless of moisture
    assert biome_layer._whittaker(-15.0, 500.0, cfg) == "tundra"
    # Temperate wet → temperate_forest
    assert biome_layer._whittaker(15.0, 1500.0, cfg) == "temperate_forest"
    # Cold wet → taiga
    assert biome_layer._whittaker(2.0, 400.0, cfg) == "taiga"


def test_no_river_outside_river_hex(small_world: GeneratedWorld) -> None:
    """A hex is biome=='river' iff its is_river flag is True."""
    for h, d in small_world.hexes.items():
        if d.biome == "river":
            assert d.is_river
        if d.is_river:
            assert d.biome == "river"
