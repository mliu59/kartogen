"""Orchestrator: runs all generation layers in order to produce a ``GeneratedWorld``."""

from __future__ import annotations

from dataclasses import dataclass

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.worldgen import biome as biome_layer
from sim.world.worldgen import climate as climate_layer
from sim.world.worldgen import elevation as elevation_layer
from sim.world.worldgen import hydrology as hydrology_layer
from sim.world.worldgen import resources as resources_layer
from sim.world.worldgen import sea as sea_layer
from sim.world.worldgen.types import (
    ClimateLayer,
    ElevationLayer,
    HexData,
    HydrologyLayer,
    SeaLayer,
    WorldgenConfig,
)


@dataclass(frozen=True)
class GeneratedWorld:
    """Final output of the generation pipeline.

    Exposes both per-hex final data (``hexes``) and the intermediate layer
    outputs (useful for testing, debugging, and rendering map overlays).
    """

    radius: int
    config: WorldgenConfig
    hexes: dict[Hex, HexData]
    elevation: ElevationLayer
    sea: SeaLayer
    climate: ClimateLayer
    hydrology: HydrologyLayer


def generate(
    radius: int,
    config: WorldgenConfig,
    seed: int,
) -> GeneratedWorld:
    """Run the full generation pipeline.

    Returns the assembled GeneratedWorld. Pure function of (radius, config, seed).
    """
    rng = RngHierarchy(seed)
    center = Hex(0, 0)
    hexes = center.spiral(radius)

    elev = elevation_layer.compute(hexes, radius, config, rng)
    sea = sea_layer.compute(elev)
    clim = climate_layer.compute(elev, sea, radius, config, rng)
    hydro = hydrology_layer.compute(elev, sea, clim.precipitation_mm, config)
    biomes = biome_layer.assign(elev, sea, clim, hydro, config)
    crop_scores = resources_layer.compute_crop_suitability(
        hexes, elev, sea, clim, hydro, biomes, config.crops,
    )
    deposits = resources_layer.compute_resource_deposits(
        hexes, elev, sea, clim, hydro, biomes, config.resources, config, rng,
    )

    hex_data: dict[Hex, HexData] = {}
    for h in hexes:
        hex_data[h] = HexData(
            elevation=elev.elevation[h] - elev.sea_level,
            is_ocean=sea.is_ocean[h],
            is_coast=sea.is_coast.get(h, False),
            is_lake=hydro.is_lake[h],
            is_river=hydro.is_river[h],
            temperature_c=clim.temperature_c[h],
            precipitation_mm=clim.precipitation_mm[h],
            flow_accumulation=hydro.flow_accumulation[h],
            biome=biomes[h],
            crop_suitability=crop_scores.get(h, {}),
            deposits=deposits.get(h, {}),
        )

    return GeneratedWorld(
        radius=radius,
        config=config,
        hexes=hex_data,
        elevation=elev,
        sea=sea,
        climate=clim,
        hydrology=hydro,
    )
