"""Biome assignment: Whittaker (temperature × precipitation) lookup with
elevation, coast and water overrides."""

from __future__ import annotations

from sim.world.hex import Hex
from sim.world.worldgen.types import (
    ClimateLayer,
    ElevationLayer,
    HydrologyLayer,
    SeaLayer,
    WorldgenConfig,
)

DEEP_OCEAN_DEPTH = 0.15  # elevation below sea_level by more than this → deep_ocean


def _whittaker(
    temp_c: float,
    precip_mm: float,
    config: WorldgenConfig,
) -> str:
    """Return the lowland biome name for a (temperature, precipitation) pair.

    Cold zones override regardless of precipitation.
    """
    if temp_c < config.tundra_max_temp_c:
        # Polar: always tundra regardless of precipitation.
        return "tundra"
    if temp_c < config.taiga_max_temp_c:
        # Cool / subarctic: tundra when dry, taiga when wet.
        if precip_mm < config.cool_band_dry_threshold:
            return "tundra"
        return "taiga"
    if temp_c < config.temperate_max_temp_c:
        # Temperate band: desert → plains → grassland → temperate_forest.
        if precip_mm < config.desert_max_precip:
            return "desert"
        if precip_mm < config.grassland_max_precip:
            return "plains"
        if precip_mm < config.forest_max_precip:
            return "grassland" if precip_mm < (config.forest_max_precip * 0.75) else "temperate_forest"
        return "temperate_forest"
    # Warm / hot band: desert → savanna → grassland → jungle.
    if precip_mm < config.desert_max_precip:
        return "desert"
    if precip_mm < config.grassland_max_precip:
        return "savanna"
    if precip_mm < config.forest_max_precip:
        return "grassland"
    return "jungle"


def assign(
    elevation: ElevationLayer,
    sea: SeaLayer,
    climate: ClimateLayer,
    hydrology: HydrologyLayer,
    config: WorldgenConfig,
) -> dict[Hex, str]:
    """Assign a terrain/biome name per hex with priority:

        1. ocean  → deep_ocean / ocean / coast
        2. river  → river (overrides biome lookup)
        3. lake   → lake
        4. high-elevation → snow_peak / mountain / hills
        5. Whittaker(T, P) → lowland biome
    """
    biomes: dict[Hex, str] = {}

    # Pre-compute land-only elevation quantile thresholds so hill/mountain/snow
    # thresholds adapt to the actual elevation distribution of land in this world.
    land_elevations = sorted(
        elevation.elevation[h] - elevation.sea_level
        for h in elevation.elevation
        if not sea.is_ocean[h]
    )
    if land_elevations:
        n_land = len(land_elevations)
        hill_t = land_elevations[int(config.elevation_hills_threshold * (n_land - 1))]
        mtn_t = land_elevations[int(config.elevation_mountain_threshold * (n_land - 1))]
        snow_t = land_elevations[int(config.elevation_snow_threshold * (n_land - 1))]
    else:
        hill_t = mtn_t = snow_t = 1.0

    for h, elev in elevation.elevation.items():
        if sea.is_ocean[h]:
            depth = elevation.sea_level - elev
            if sea.is_coast.get(h, False):
                # Should not happen for ocean hexes, but guard anyway.
                biomes[h] = "coast"
            elif depth > DEEP_OCEAN_DEPTH:
                biomes[h] = "deep_ocean"
            else:
                biomes[h] = "ocean"
            continue

        if sea.is_coast.get(h, False):
            # Coast trumps biome but yields to river-on-coast (river mouth).
            if hydrology.is_river[h]:
                biomes[h] = "river"
            else:
                biomes[h] = "coast"
            continue

        if hydrology.is_river[h]:
            biomes[h] = "river"
            continue
        if hydrology.is_lake[h]:
            biomes[h] = "lake"
            continue

        # Elevation overrides for high terrain.
        elev_above_sea = elev - elevation.sea_level
        # Snow peak: above snow threshold AND cold enough.
        if elev_above_sea >= snow_t and climate.temperature_c[h] < 0.0:
            biomes[h] = "snow_peak"
            continue
        if elev_above_sea >= mtn_t:
            biomes[h] = "mountain"
            continue
        if elev_above_sea >= hill_t:
            biomes[h] = "hills"
            continue

        # Lowland Whittaker assignment.
        biomes[h] = _whittaker(climate.temperature_c[h], climate.precipitation_mm[h], config)

    return biomes
