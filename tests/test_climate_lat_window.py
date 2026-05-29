"""Tests for the configurable latitude window in the climate layer.

The map's pixel-y axis covers a slice of the planet's latitude range, set
by ``map_lat_min`` and ``map_lat_max``. The slice is independent of
``hex_size_km`` and of the world's km dimensions — the same physical
height can span different latitude ranges, and vice versa.
"""

from __future__ import annotations

import statistics
from dataclasses import replace

from worldgen import generate
from worldgen.climate import (
    _WIND_BAND_HI_DEG,
    _WIND_BAND_LO_DEG,
    _zonal_wind_sign,
    hex_latitude_deg,
)
from worldgen.hex import Hex
from worldgen.types import WorldgenConfig, WorldShape


def _shape(width_km: float, height_km: float) -> WorldShape:
    return WorldShape(width_km=width_km, height_km=height_km)


def test_hex_latitude_endpoints_match_window(default_worldgen_config: WorldgenConfig) -> None:
    """A hex past +half_height_km in pixel-y clamps to map_lat_min;
    past -half_height_km clamps to map_lat_max."""
    cfg = replace(
        default_worldgen_config,
        map_lat_min=30.0, map_lat_max=60.0,
    )
    # Hard-pick half_height_km = 50 km. With 5 km/hex and r=±10 at q=0:
    # |pixel_y| = sqrt(3) * 10 * 5 ≈ 86.6 km > half_height, so the
    # endpoints clamp to the lat extremes.
    half_h = 50.0
    north_edge = Hex(0, -10)
    south_edge = Hex(0, 10)
    centre = Hex(0, 0)
    assert hex_latitude_deg(north_edge, half_h, cfg) == 60.0
    assert hex_latitude_deg(south_edge, half_h, cfg) == 30.0
    assert hex_latitude_deg(centre, half_h, cfg) == 45.0


def test_shifting_window_north_cools_the_map(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """A map at lat 50–70 averages colder than the same map at lat 0–20."""
    base = replace(default_worldgen_config, world=_shape(160.0, 160.0))
    tropical = replace(base, map_lat_min=0.0, map_lat_max=20.0)
    polar = replace(base, map_lat_min=50.0, map_lat_max=70.0)
    tropical_world = generate(config=tropical, seed=7)
    polar_world = generate(config=polar, seed=7)

    tropical_land = [
        d.temperature_c for d in tropical_world.hexes.values() if not d.is_ocean
    ]
    polar_land = [
        d.temperature_c for d in polar_world.hexes.values() if not d.is_ocean
    ]
    # Both windows produce land hexes (the tectonic state is identical because
    # the seed and climate-independent fields are the same).
    assert tropical_land and polar_land
    assert statistics.mean(polar_land) < statistics.mean(tropical_land) - 10.0


def test_narrowing_window_flattens_temperature_gradient(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """A 2°-wide map has a much smaller north-south temperature spread than a
    full pole-to-pole map of the same physical dimensions."""
    base = replace(default_worldgen_config, world=_shape(140.0, 140.0))
    full = replace(base, map_lat_min=-90.0, map_lat_max=90.0)
    narrow = replace(base, map_lat_min=39.0, map_lat_max=41.0)
    full_world = generate(config=full, seed=11)
    narrow_world = generate(config=narrow, seed=11)

    def temp_spread(world: object) -> float:
        # All-hex spread, not land-only: the latitudinal gradient applies to
        # ocean hexes the same way, so we don't need to restrict to land
        # (which can be empty on tiny test worlds where every hex ends up
        # below sea level).
        temps = [d.temperature_c for d in world.hexes.values()]  # type: ignore[attr-defined]
        return max(temps) - min(temps)

    assert temp_spread(full_world) > temp_spread(narrow_world) + 10.0


def test_equator_window_keeps_centre_warm(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """A symmetric window centred on the equator gives the warmest temps
    at the centre (lat 0) and cooler at the edges.

    Looks at *ocean* hexes only so the lapse-rate cooling from random
    mountain belts doesn't contaminate the latitudinal signal we're after.
    """
    cfg = replace(
        default_worldgen_config,
        map_lat_min=-30.0, map_lat_max=30.0,
        world=_shape(120.0, 120.0),
    )
    world = generate(config=cfg, seed=3)

    by_bucket: dict[int, list[float]] = {}
    for h, d in world.hexes.items():
        if not d.is_ocean:
            continue
        by_bucket.setdefault(abs(h.r) // 3, []).append(d.temperature_c)

    means = sorted(
        (b, statistics.mean(ts))
        for b, ts in by_bucket.items()
        if len(ts) >= 5
    )
    assert len(means) >= 2
    # Lowest |r| bucket (centre, equator) is warmest; higher |r| are cooler.
    assert means[0][1] > means[-1][1]


def test_wind_band_sign_matches_earth_zones() -> None:
    """The zonal wind sign convention encodes Earth's three-cell pattern."""
    # Trade easterlies inside 0–30°.
    assert _zonal_wind_sign(0.0) == -1.0
    assert _zonal_wind_sign(_WIND_BAND_LO_DEG - 10.0) == -1.0
    # Westerlies inside 30–60°.
    assert _zonal_wind_sign(45.0) == 1.0
    # Polar easterlies above 60°.
    assert _zonal_wind_sign(_WIND_BAND_HI_DEG + 10.0) == -1.0
    assert _zonal_wind_sign(89.0) == -1.0


def test_lat_window_is_deterministic(default_worldgen_config: WorldgenConfig) -> None:
    """The same lat window + seed produces identical temperature output."""
    cfg = replace(
        default_worldgen_config,
        map_lat_min=20.0, map_lat_max=50.0,
        world=_shape(80.0, 80.0),
    )
    a = generate(config=cfg, seed=42)
    b = generate(config=cfg, seed=42)
    for h in a.hexes:
        assert a.hexes[h].temperature_c == b.hexes[h].temperature_c
