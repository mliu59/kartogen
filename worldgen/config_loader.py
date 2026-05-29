"""Standalone loader for ``[worldgen.*]`` TOML sections.

Shared by the engine-side ``SimConfig.from_toml`` and the engine-independent
preview / test entry points. Importantly this module does **not** import the
simulation engine, so terrain-only tests and the headless preview can both
build a ``WorldgenConfig`` without dragging the full agent / resolution
stack into their import graph.
"""

from __future__ import annotations

from pathlib import Path

from worldgen.types import (
    OceanConfig,
    PlateConfig,
    TectonicsConfig,
    WorldgenConfig,
    WorldShape,
)


def load_worldgen_config(path: Path) -> WorldgenConfig:
    """Load a ``WorldgenConfig`` from a TOML file."""
    import tomllib

    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse_worldgen_config(raw["worldgen"])


def parse_worldgen_config(wg: dict[str, object]) -> WorldgenConfig:
    """Parse a pre-loaded ``[worldgen]`` table into a ``WorldgenConfig``.

    Reads ``[worldgen.elevation]`` (with a nested ``plates`` sub-table) plus
    the ``[worldgen.{climate,hydrology,biome,tectonics,ocean}]`` sections.
    Every required field must be present — missing fields raise rather than
    silently defaulting.
    """
    wg_elev = dict(wg.get("elevation", {}))  # type: ignore[arg-type]
    wg_clim = wg["climate"]  # type: ignore[index]
    wg_hydro = wg["hydrology"]  # type: ignore[index]
    wg_biome = wg["biome"]  # type: ignore[index]

    plates_cfg = _parse_plate_config(wg_elev)
    tect_cfg = _parse_tectonics_config(wg.get("tectonics"))  # type: ignore[arg-type]
    ocean_cfg = _parse_ocean_config(wg.get("ocean"))  # type: ignore[arg-type]
    world_cfg = _parse_world_shape(wg.get("world"))  # type: ignore[arg-type]
    if (
        plates_cfg is None or tect_cfg is None
        or ocean_cfg is None or world_cfg is None
    ):
        raise ValueError(
            "WorldgenConfig requires [worldgen.world], "
            "[worldgen.elevation.plates], [worldgen.tectonics], and "
            "[worldgen.ocean] tables."
        )

    return WorldgenConfig(
        hex_size_km=wg["hex_size_km"],  # type: ignore[index]
        world=world_cfg,
        feature_wavelength_km=wg_elev["feature_wavelength_km"],
        noise_octaves=wg_elev["noise_octaves"],
        noise_lacunarity=wg_elev["noise_lacunarity"],
        noise_persistence=wg_elev["noise_persistence"],
        warp_strength_km=wg_elev["warp_strength_km"],
        warp_wavelength_km=wg_elev["warp_wavelength_km"],
        ridge_octaves=wg_elev["ridge_octaves"],
        ridge_amplitude=wg_elev["ridge_amplitude"],
        ridge_threshold=wg_elev["ridge_threshold"],
        tectonic_blend_weight=wg_elev["tectonic_blend_weight"],
        plates=plates_cfg,
        tectonics=tect_cfg,
        ocean=ocean_cfg,
        map_lat_min=float(wg_clim["map_lat_min"]),  # type: ignore[arg-type]
        map_lat_max=float(wg_clim["map_lat_max"]),  # type: ignore[arg-type]
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
        wind_jitter_amplitude_deg=wg_clim["wind_jitter_amplitude_deg"],
        wind_jitter_wavelength_km=wg_clim["wind_jitter_wavelength_km"],
        sea_breeze_strength=wg_clim["sea_breeze_strength"],
        sea_breeze_reach_km=wg_clim["sea_breeze_reach_km"],
        wind_path_samples=int(wg_clim["wind_path_samples"]),
        wind_path_spread_deg=wg_clim["wind_path_spread_deg"],
        precip_smoothing_passes=int(wg_clim["precip_smoothing_passes"]),
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
    )


def _parse_world_shape(raw: dict[str, object] | None) -> WorldShape | None:
    """Parse the ``[worldgen.world]`` table into a ``WorldShape``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"`world` must be a TOML table, got {type(raw).__name__}"
        )
    return WorldShape(
        width_km=float(raw["width_km"]),  # type: ignore[arg-type]
        height_km=float(raw["height_km"]),  # type: ignore[arg-type]
    )


def _parse_plate_config(wg_elev: dict[str, object]) -> PlateConfig | None:
    """Parse the ``plates`` sub-table of the elevation section.

    Returns ``None`` when no ``[worldgen.elevation.plates]`` sub-table is
    present; ``parse_worldgen_config`` then raises with a clear message.
    """
    raw = wg_elev.get("plates")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"`plates` must be a TOML table, got {type(raw).__name__}"
        )
    return PlateConfig(
        count=int(raw["count"]),  # type: ignore[arg-type]
        continental_fraction=float(raw["continental_fraction"]),  # type: ignore[arg-type]
        min_separation_km=float(raw["min_separation_km"]),  # type: ignore[arg-type]
        seed_radial_bias=float(raw["seed_radial_bias"]),  # type: ignore[arg-type]
        boundary_warp_strength_km=float(raw["boundary_warp_strength_km"]),  # type: ignore[arg-type]
        boundary_warp_wavelength_km=float(raw["boundary_warp_wavelength_km"]),  # type: ignore[arg-type]
        motion_speed=float(raw["motion_speed"]),  # type: ignore[arg-type]
        convergence_threshold=float(raw["convergence_threshold"]),  # type: ignore[arg-type]
    )


def _parse_ocean_config(raw: dict[str, object] | None) -> OceanConfig | None:
    """Parse the ``[worldgen.ocean]`` table. Required for v1."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"`ocean` must be a TOML table, got {type(raw).__name__}"
        )
    return OceanConfig(
        current_persistence_km=float(raw["current_persistence_km"]),  # type: ignore[arg-type]
        current_anomaly_strength=float(raw["current_anomaly_strength"]),  # type: ignore[arg-type]
        max_current_anomaly_c=float(raw["max_current_anomaly_c"]),  # type: ignore[arg-type]
        coastal_pickup_fraction=float(raw["coastal_pickup_fraction"]),  # type: ignore[arg-type]
        coastal_decay_km=float(raw["coastal_decay_km"]),  # type: ignore[arg-type]
        continentality_dry_scale_km=float(raw["continentality_dry_scale_km"]),  # type: ignore[arg-type]
    )


def _parse_tectonics_config(raw: dict[str, object] | None) -> TectonicsConfig | None:
    """Parse the ``[worldgen.tectonics]`` table.

    Returns ``None`` if absent; ``parse_worldgen_config`` then raises.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"`tectonics` must be a TOML table, got {type(raw).__name__}"
        )
    return TectonicsConfig(
        n_ticks=int(raw["n_ticks"]),  # type: ignore[arg-type]
        dt_myr=float(raw["dt_myr"]),  # type: ignore[arg-type]
        sea_level_km=float(raw["sea_level_km"]),  # type: ignore[arg-type]
        plate_speed_kmpy=float(raw["plate_speed_kmpy"]),  # type: ignore[arg-type]
        continental_thickness_km=float(raw["continental_thickness_km"]),  # type: ignore[arg-type]
        oceanic_thickness_km=float(raw["oceanic_thickness_km"]),  # type: ignore[arg-type]
        rift_thickness_km=float(raw["rift_thickness_km"]),  # type: ignore[arg-type]
        ridge_depth_km=float(raw["ridge_depth_km"]),  # type: ignore[arg-type]
        ridge_subsidence_rate=float(raw["ridge_subsidence_rate"]),  # type: ignore[arg-type]
        max_ocean_depth_km=float(raw["max_ocean_depth_km"]),  # type: ignore[arg-type]
        continental_reference_thickness_km=float(raw["continental_reference_thickness_km"]),  # type: ignore[arg-type]
        continental_isostasy_factor=float(raw["continental_isostasy_factor"]),  # type: ignore[arg-type]
        orogeny_uplift_per_overlap_km=float(raw["orogeny_uplift_per_overlap_km"]),  # type: ignore[arg-type]
        folding_ratio=float(raw["folding_ratio"]),  # type: ignore[arg-type]
        subduction_arc_uplift_km=float(raw["subduction_arc_uplift_km"]),  # type: ignore[arg-type]
        erosion_period=int(raw["erosion_period"]),  # type: ignore[arg-type]
        erosion_strength=float(raw["erosion_strength"]),  # type: ignore[arg-type]
        boundary_warp_strength=float(raw["boundary_warp_strength"]),  # type: ignore[arg-type]
        boundary_warp_wavelength_km=float(raw["boundary_warp_wavelength_km"]),  # type: ignore[arg-type]
        snapshot_period_ticks=int(raw["snapshot_period_ticks"]),  # type: ignore[arg-type]
    )
