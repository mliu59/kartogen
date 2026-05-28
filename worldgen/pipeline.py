"""Orchestrator: runs all generation layers in order to produce a ``GeneratedWorld``."""

from __future__ import annotations

from dataclasses import dataclass

from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from worldgen import biome as biome_layer
from worldgen import climate as climate_layer
from worldgen import elevation as elevation_layer
from worldgen import hydrology as hydrology_layer
from worldgen import plates as plates_layer
from worldgen import resources as resources_layer
from worldgen import sea as sea_layer
from worldgen.plates import PlateField
from worldgen.types import (
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
    plates: PlateField | None


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

    # L0 — plates: run only when the elevation layer asks for them. Cheap to
    # skip entirely so analytic continent masks pay nothing for the plate code.
    plate_field: PlateField | None = None
    if config.mask_mode == "plates":
        if config.plates is None:
            raise ValueError(
                "mask_mode='plates' but no [worldgen.elevation.plates] config "
                "was provided."
            )
        plate_field = plates_layer.generate_plates(
            hexes, radius, config.plates, config.hex_size_km, rng,
        )

    elev = elevation_layer.compute(hexes, radius, config, rng, plate_field)
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
        if plate_field is not None:
            pid = plate_field.hex_to_plate[h]
            ptype: str | None = plate_field.plates[pid].type
            btype: str | None = plate_field.boundary_type[h]
            bdist: float | None = plate_field.distance_to_boundary_km[h]
            pid_out: int | None = pid
        else:
            pid_out = None
            ptype = None
            btype = None
            bdist = None
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
            plate_id=pid_out,
            plate_type=ptype,
            nearest_boundary_type=btype,
            distance_to_boundary_km=bdist,
        )

    return GeneratedWorld(
        radius=radius,
        config=config,
        hexes=hex_data,
        elevation=elev,
        sea=sea,
        climate=clim,
        hydrology=hydro,
        plates=plate_field,
    )
