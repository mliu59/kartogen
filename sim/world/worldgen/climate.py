"""Climate layer: temperature (latitude + lapse rate) and precipitation
(prevailing-wind moisture sweep with orographic uplift)."""

from __future__ import annotations

import math

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.noise import PerlinNoise2D
from sim.world.worldgen.types import (
    ClimateLayer,
    ElevationLayer,
    SeaLayer,
    WorldgenConfig,
)


def _latitude_fraction(h: Hex, radius: int) -> float:
    """Normalized latitude in [-1, 1]: equator = 0, poles = ±1.

    Uses the ``r`` axial coordinate as latitude axis (north → r negative).
    """
    if radius == 0:
        return 0.0
    return max(-1.0, min(1.0, h.r / radius))


def compute_temperature(
    elevation: ElevationLayer,
    radius: int,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, float]:
    """Per-hex annual mean temperature in °C.

    T = base_by_latitude − elevation_above_sea_in_km · lapse_rate + small_noise
    Sub-sea elevations are clamped to 0 km for the lapse term so ocean stays
    close to its latitudinal baseline.
    """
    temp_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "temp_noise"))
    eq = config.equator_temp_c
    pole = config.polar_temp_c

    temperatures: dict[Hex, float] = {}
    for h, elev in elevation.elevation.items():
        lat_abs = abs(_latitude_fraction(h, radius))
        # cosine-like latitudinal gradient (slightly fatter equatorial band).
        latitudinal = eq + (pole - eq) * (lat_abs**1.3)

        elev_above_sea = max(0.0, elev - elevation.sea_level)
        # elev_above_sea is in normalized [0,1] units; multiply by max_elevation_km.
        km = elev_above_sea * config.max_elevation_km
        lapse = km * config.lapse_rate_c_per_km

        # Small regional variation.
        n = temp_noise.sample(h.q * 0.04, h.r * 0.04)

        temperatures[h] = latitudinal - lapse + n * config.temp_noise_amplitude
    return temperatures


def _wind_direction(latitude: float) -> tuple[float, float]:
    """Prevailing wind unit vector by latitude band, in axial (q,r) space.

    Latitude is normalized in [-1, 1]. Bands (Earth-like):
      |lat| in [0.0, 0.33]   → trade easterlies (wind blows east→west)
      |lat| in [0.33, 0.66]  → westerlies (west→east)
      |lat| in [0.66, 1.00]  → polar easterlies (east→west)
    """
    a = abs(latitude)
    if a < 0.33:
        sign = -1.0  # east→west
    elif a < 0.66:
        sign = 1.0   # west→east
    else:
        sign = -1.0
    # East unit vector in axial space (q increases east) — flat-top hexes.
    return (sign, 0.0)


def compute_precipitation(
    elevation: ElevationLayer,
    sea: SeaLayer,
    temperatures: dict[Hex, float],
    radius: int,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, float]:
    """Per-hex annual precipitation in mm using a prevailing-wind moisture sweep.

    For each hex, march upwind one step at a time. Each ocean step adds moisture
    (scaled by warmth), each land step deposits a fraction of moisture, and
    uphill steps deposit additional moisture (orographic uplift / rain shadow).
    """
    precip_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "precip_noise"))

    # Maximum march distance comes from the configured physical wind reach,
    # converted to hex steps based on hex_size_km. Capped to the world diameter
    # so we never iterate further than the map extends.
    max_steps = min(config.wind_reach_hexes, max(40, radius * 2))

    precipitation: dict[Hex, float] = {}
    for h in elevation.elevation:
        lat = _latitude_fraction(h, radius)
        wq, wr = _wind_direction(lat)

        # Simulate air parcels travelling along the wind direction toward h.
        # At parameter ``step`` ∈ [max_steps, 0], the air is at position
        # ``h - step * wind`` (step=max_steps is far upwind; step=0 is h).
        moisture = 0.0
        deposited_here = 0.0

        for step in range(max_steps, -1, -1):
            qh = h.q - int(round(wq * step))
            rh = h.r - int(round(wr * step))
            cur_hex = Hex(qh, rh)
            if cur_hex not in elevation.elevation:
                continue
            if sea.is_ocean[cur_hex]:
                # Warm oceans evaporate more moisture into the air parcel.
                t = temperatures[cur_hex]
                warmth = max(0.05, min(1.0, (t + 5.0) / 35.0))
                moisture += warmth * config.precip_max_ocean_pickup
            else:
                # Fractional deposit + orographic uplift (deposit on uphill).
                deposit = moisture * config.precip_loss_per_land
                prev_q = h.q - int(round(wq * (step + 1)))
                prev_r = h.r - int(round(wr * (step + 1)))
                prev_hex = Hex(prev_q, prev_r)
                prev_elev = elevation.elevation.get(prev_hex, elevation.sea_level)
                dh = elevation.elevation[cur_hex] - prev_elev
                if dh > 0:
                    deposit += dh * config.precip_orographic_coef
                deposit = min(deposit, moisture)
                moisture -= deposit
                if cur_hex == h:
                    deposited_here = deposit
                    break

        # Combine target-hex deposit with a baseline + small fBm noise. The
        # baseline keeps deep interiors above zero; rain-shadow dryness still
        # emerges from the moisture budget being depleted upwind.
        n = precip_noise.sample(h.q * 0.03, h.r * 0.03)
        if sea.is_ocean[h]:
            value = 0.0
        else:
            base = config.precip_base * 0.15  # soft floor over land
            value = base + deposited_here + n * config.precip_noise_amplitude
        precipitation[h] = max(0.0, value)

    return precipitation


def compute(
    elevation: ElevationLayer,
    sea: SeaLayer,
    radius: int,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> ClimateLayer:
    """Compute temperature then precipitation. Precipitation depends on temperature."""
    temperatures = compute_temperature(elevation, radius, config, rng)
    precipitation = compute_precipitation(elevation, sea, temperatures, radius, config, rng)
    return ClimateLayer(temperature_c=temperatures, precipitation_mm=precipitation)
