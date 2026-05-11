"""Standalone loader for ``[worldgen.*]`` TOML sections.

Shared by the engine-side ``SimConfig.from_toml`` and the engine-independent
preview / test entry points. Importantly this module does **not** import the
simulation engine, so terrain-only tests and the headless preview can both
build a ``WorldgenConfig`` without dragging the full agent / resolution
stack into their import graph.
"""

from __future__ import annotations

from pathlib import Path

from sim.world.worldgen.types import (
    CropDefinition,
    ResourceDefinition,
    WorldgenConfig,
)


def load_worldgen_config(path: Path) -> WorldgenConfig:
    """Load a ``WorldgenConfig`` from a TOML file."""
    import tomllib

    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse_worldgen_config(raw["worldgen"])


def parse_worldgen_config(wg: dict[str, object]) -> WorldgenConfig:
    """Parse a pre-loaded ``[worldgen]`` table into a ``WorldgenConfig``."""
    wg_elev = wg["elevation"]  # type: ignore[index]
    wg_clim = wg["climate"]  # type: ignore[index]
    wg_hydro = wg["hydrology"]  # type: ignore[index]
    wg_biome = wg["biome"]  # type: ignore[index]
    crops = parse_crops(wg.get("crops", {}))  # type: ignore[union-attr,arg-type]
    resources = parse_resources(wg.get("resources", {}))  # type: ignore[union-attr,arg-type]

    return WorldgenConfig(
        hex_size_km=wg["hex_size_km"],  # type: ignore[index]
        land_fraction=wg_elev["land_fraction"],
        feature_wavelength_km=wg_elev["feature_wavelength_km"],
        noise_octaves=wg_elev["noise_octaves"],
        noise_lacunarity=wg_elev["noise_lacunarity"],
        noise_persistence=wg_elev["noise_persistence"],
        warp_strength_km=wg_elev["warp_strength_km"],
        warp_wavelength_km=wg_elev["warp_wavelength_km"],
        ridge_octaves=wg_elev["ridge_octaves"],
        ridge_amplitude=wg_elev["ridge_amplitude"],
        ridge_threshold=wg_elev["ridge_threshold"],
        falloff_strength=wg_elev["falloff_strength"],
        falloff_power=wg_elev["falloff_power"],
        falloff_inner_fraction=wg_elev["falloff_inner_fraction"],
        equator_temp_c=wg_clim["equator_temp_c"],
        polar_temp_c=wg_clim["polar_temp_c"],
        lapse_rate_c_per_km=wg_clim["lapse_rate_c_per_km"],
        max_elevation_km=wg_clim["max_elevation_km"],
        temp_noise_amplitude=wg_clim["temp_noise_amplitude"],
        precip_base=wg_clim["precip_base"],
        precip_pickup_per_ocean_km=wg_clim["precip_pickup_per_ocean_km"],
        precip_loss_per_km=wg_clim["precip_loss_per_km"],
        precip_orographic_coef=wg_clim["precip_orographic_coef"],
        precip_noise_amplitude=wg_clim["precip_noise_amplitude"],
        wind_reach_km=wg_clim["wind_reach_km"],
        river_drainage_threshold_km2=wg_hydro["river_drainage_threshold_km2"],
        lake_min_depth=wg_hydro["lake_min_depth"],
        river_carve_amount=wg_hydro["river_carve_amount"],
        elevation_hills_threshold=wg_biome["elevation_hills_threshold"],
        elevation_mountain_threshold=wg_biome["elevation_mountain_threshold"],
        elevation_snow_threshold=wg_biome["elevation_snow_threshold"],
        tundra_max_temp_c=wg_biome["tundra_max_temp_c"],
        taiga_max_temp_c=wg_biome["taiga_max_temp_c"],
        temperate_max_temp_c=wg_biome["temperate_max_temp_c"],
        desert_max_precip=wg_biome["desert_max_precip"],
        grassland_max_precip=wg_biome["grassland_max_precip"],
        forest_max_precip=wg_biome["forest_max_precip"],
        cool_band_dry_threshold=wg_biome["cool_band_dry_threshold"],
        crops=crops,
        resources=resources,
    )


def parse_crops(raw: dict[str, dict[str, object]]) -> tuple[CropDefinition, ...]:
    """Parse a ``[worldgen.crops.*]`` table into ``CropDefinition`` objects.

    Crops are returned in name-sorted order so iteration is deterministic.
    """
    crops: list[CropDefinition] = []
    for name in sorted(raw.keys()):
        props = raw[name]
        biome_compat = props.get("biome_compatibility", {})
        crops.append(CropDefinition(
            name=name,
            temp_abs_min=float(props["temp_abs_min"]),  # type: ignore[arg-type]
            temp_opt_min=float(props["temp_opt_min"]),  # type: ignore[arg-type]
            temp_opt_max=float(props["temp_opt_max"]),  # type: ignore[arg-type]
            temp_abs_max=float(props["temp_abs_max"]),  # type: ignore[arg-type]
            precip_abs_min=float(props["precip_abs_min"]),  # type: ignore[arg-type]
            precip_opt_min=float(props["precip_opt_min"]),  # type: ignore[arg-type]
            precip_opt_max=float(props["precip_opt_max"]),  # type: ignore[arg-type]
            precip_abs_max=float(props["precip_abs_max"]),  # type: ignore[arg-type]
            elev_max=float(props["elev_max"]),  # type: ignore[arg-type]
            biome_compatibility={k: float(v) for k, v in biome_compat.items()},  # type: ignore[union-attr]
            river_bonus=float(props.get("river_bonus", 0.0)),  # type: ignore[arg-type]
            river_adjacent_bonus=float(props.get("river_adjacent_bonus", 0.0)),  # type: ignore[arg-type]
            coast_bonus=float(props.get("coast_bonus", 0.0)),  # type: ignore[arg-type]
            irrigation_replaces_rain_mm=float(props.get("irrigation_replaces_rain_mm", 0.0)),  # type: ignore[arg-type]
        ))
    return tuple(crops)


def parse_resources(raw: dict[str, dict[str, object]]) -> tuple[ResourceDefinition, ...]:
    """Parse a ``[worldgen.resources.*]`` table into ``ResourceDefinition``s."""
    resources: list[ResourceDefinition] = []
    for name in sorted(raw.keys()):
        props = raw[name]
        host_biomes = tuple(props.get("host_biomes", ()))  # type: ignore[arg-type]
        resources.append(ResourceDefinition(
            name=name,
            host_biomes=host_biomes,
            min_elevation=float(props.get("min_elevation", 0.0)),  # type: ignore[arg-type]
            max_elevation=float(props.get("max_elevation", 1.0)),  # type: ignore[arg-type]
            min_temperature_c=float(props.get("min_temperature_c", -1e6)),  # type: ignore[arg-type]
            max_temperature_c=float(props.get("max_temperature_c", 1e6)),  # type: ignore[arg-type]
            min_precipitation_mm=float(props.get("min_precipitation_mm", 0.0)),  # type: ignore[arg-type]
            max_precipitation_mm=float(props.get("max_precipitation_mm", 1e6)),  # type: ignore[arg-type]
            feature_wavelength_km=float(props["feature_wavelength_km"]),  # type: ignore[arg-type]
            abundance=float(props["abundance"]),  # type: ignore[arg-type]
            mean_quantity=float(props["mean_quantity"]),  # type: ignore[arg-type]
            elevation_quantity_bonus=float(props.get("elevation_quantity_bonus", 0.0)),  # type: ignore[arg-type]
            category=str(props.get("category", "")),
        ))
    return tuple(resources)
