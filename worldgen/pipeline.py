"""Orchestrator: runs all generation layers in order to produce a ``GeneratedWorld``."""

from __future__ import annotations

from dataclasses import dataclass

from worldgen._log import get_logger, timed_layer
from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from worldgen import biome as biome_layer
from worldgen import climate as climate_layer
from worldgen import elevation as elevation_layer
from worldgen import hydrology as hydrology_layer
from worldgen import ocean as ocean_layer
from worldgen import plates as plates_layer
from worldgen import sea as sea_layer
from worldgen import tectonics as tectonics_layer
from worldgen.ocean import OceanLayer
from worldgen.plates import PlateField
from worldgen.tectonics import LithosphereState
from worldgen.world import rect_world_hexes

_log = get_logger("pipeline")
from worldgen.types import (
    ClimateLayer,
    ElevationLayer,
    HexData,
    HydrologyLayer,
    SeaLayer,
    WorldgenConfig,
)


# Ordered names of the generation steps. ``stop_after`` in ``generate`` accepts
# any of these; the last entry (``"biome"``) is equivalent to running the full
# pipeline.
PIPELINE_STEPS: tuple[str, ...] = (
    "plates",
    "tectonics",
    "elevation",
    "sea",
    "ocean",
    "climate",
    "hydrology",
    "biome",
)


def _step_index(name: str) -> int:
    try:
        return PIPELINE_STEPS.index(name)
    except ValueError as e:
        raise ValueError(
            f"unknown pipeline step {name!r}; expected one of {PIPELINE_STEPS}"
        ) from e


@dataclass(frozen=True)
class GeneratedWorld:
    """Final output of the generation pipeline.

    Exposes both per-hex final data (``hexes``) and the intermediate layer
    outputs (useful for testing, debugging, and rendering map overlays).

    When ``generate`` is called with ``stop_after`` set to an intermediate
    step, all layer fields downstream of that step are ``None`` and
    ``hexes`` is ``None`` (the per-hex assembly happens only when the full
    pipeline runs). ``stop_after`` records the step the pipeline stopped at
    so consumers can branch without inspecting every field.
    """

    config: WorldgenConfig
    plates: PlateField
    lithosphere: LithosphereState | None
    elevation: ElevationLayer | None
    sea: SeaLayer | None
    ocean: OceanLayer | None
    climate: ClimateLayer | None
    hydrology: HydrologyLayer | None
    biomes: dict[Hex, str] | None
    hexes: dict[Hex, HexData] | None
    stop_after: str


def generate(
    config: WorldgenConfig,
    seed: int,
    stop_after: str | None = None,
) -> GeneratedWorld:
    """Run the generation pipeline up to (and including) ``stop_after``.

    ``stop_after=None`` (default) runs the full pipeline. Otherwise the
    string must be one of ``PIPELINE_STEPS``; downstream layers and the
    per-hex assembly are skipped. Pure function of (config, seed,
    stop_after) — the world footprint comes from ``config.world``.
    """
    stop = stop_after if stop_after is not None else PIPELINE_STEPS[-1]
    stop_ix = _step_index(stop)

    rng = RngHierarchy(seed)
    hexes = rect_world_hexes(config.world, config.hex_size_km)
    _log.info(
        "world: %g×%g km hexes=%d hex_size=%.1f km seed=%d stop_after=%s",
        config.world.width_km, config.world.height_km,
        len(hexes), config.hex_size_km, seed, stop,
    )

    # L0a — plates: t=0 plate seeds + initial Voronoi assignment.
    with timed_layer("plates"):
        plate_field = plates_layer.generate_plates(
            hexes, config.plates, config.hex_size_km, rng,
        )

    lithosphere: LithosphereState | None = None
    elev: ElevationLayer | None = None
    sea: SeaLayer | None = None
    ocean: OceanLayer | None = None
    clim: ClimateLayer | None = None
    hydro: HydrologyLayer | None = None
    biomes: dict[Hex, str] | None = None
    hex_data: dict[Hex, HexData] | None = None

    # L0b — tectonics: time-stepped simulation that evolves plates over
    #       n_ticks × dt_myr of geological time.
    if stop_ix >= _step_index("tectonics"):
        with timed_layer(
            f"tectonics ({config.tectonics.n_ticks} ticks "
            f"× {config.tectonics.dt_myr:g} Myr)"
        ):
            lithosphere = tectonics_layer.simulate_tectonics(
                plate_field, hexes, config.tectonics, config.hex_size_km, rng,
            )

    if stop_ix >= _step_index("elevation"):
        assert lithosphere is not None
        with timed_layer("elevation"):
            elev = elevation_layer.compute(hexes, config, rng, lithosphere)

    if stop_ix >= _step_index("sea"):
        assert elev is not None
        with timed_layer("sea"):
            sea = sea_layer.compute(elev)

    if stop_ix >= _step_index("ocean"):
        assert sea is not None
        with timed_layer("ocean"):
            ocean = ocean_layer.compute(sea, hexes, config)

    if stop_ix >= _step_index("climate"):
        assert elev is not None and sea is not None and ocean is not None
        with timed_layer("climate"):
            clim = climate_layer.compute(elev, sea, ocean, config, rng)

    if stop_ix >= _step_index("hydrology"):
        assert elev is not None and sea is not None and clim is not None
        with timed_layer("hydrology"):
            hydro = hydrology_layer.compute(elev, sea, clim.precipitation_mm, config)

    if stop_ix >= _step_index("biome"):
        assert (
            elev is not None and sea is not None
            and clim is not None and hydro is not None
        )
        with timed_layer("biome"):
            biomes = biome_layer.assign(elev, sea, clim, hydro, config)

    # Per-hex HexData assembly happens only when the full pipeline ran.
    if stop_ix == _step_index("biome"):
        assert (
            lithosphere is not None and elev is not None and sea is not None
            and ocean is not None and clim is not None and hydro is not None
            and biomes is not None
        )
        plate_type_lookup = {p.id: p.type for p in plate_field.plates}
        hex_data = {}
        for h in hexes:
            pid = lithosphere.plate_id[h]
            col = lithosphere.columns[h]
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
                plate_id=pid,
                # Plate id comes from the *final* tectonic state; boundary
                # classification is inherited from the t=0 PlateField snapshot
                # (recomputing it from the simulated final state is a v2 task).
                plate_type=plate_type_lookup.get(pid),
                nearest_boundary_type=plate_field.boundary_type[h],
                distance_to_boundary_km=plate_field.distance_to_boundary_km[h],
                crust_thickness_km=col.thickness_km,
                crust_type=col.crust_type,
                crust_age_myr=col.age_myr,
                distance_to_ocean_km=ocean.distance_to_ocean_km.get(h, 0.0),
                current_temp_anomaly_c=ocean.current_temp_anomaly.get(h, 0.0),
                coastal_temp_anomaly_c=ocean.coastal_temp_anomaly.get(h, 0.0),
                gyre_id=ocean.gyre_id.get(h),
            )

    return GeneratedWorld(
        config=config,
        plates=plate_field,
        lithosphere=lithosphere,
        elevation=elev,
        sea=sea,
        ocean=ocean,
        climate=clim,
        hydrology=hydro,
        biomes=biomes,
        hexes=hex_data,
        stop_after=stop,
    )
