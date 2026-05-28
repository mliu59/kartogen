"""Climate layer: temperature (latitude + lapse rate) and precipitation
(prevailing-wind moisture sweep with orographic uplift).

The precipitation model uses a per-hex *cartesian* wind direction. The base
direction is set by latitude band (the three-cell zonal pattern), then
perturbed by an annual-mean **sea-breeze** onshore component near coasts
and a spatially-coherent **Perlin jitter**. Per target hex, the moisture
sweep is averaged over **multiple sample paths** spread in a cone around
the base direction — together these break the pure zonal "stripes" pattern
without simulating seasonal physics.
"""

from __future__ import annotations

import math
from collections import deque

from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from worldgen.noise import PerlinNoise2D
from worldgen.types import (
    ClimateLayer,
    ElevationLayer,
    SeaLayer,
    WorldgenConfig,
)

_SQRT3 = math.sqrt(3.0)


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


def _zonal_wind_cartesian(latitude: float) -> tuple[float, float]:
    """Base prevailing wind unit vector by latitude band, in *cartesian*
    screen space (``+x`` = right, ``+y`` = down — matches the renderer).

    Latitude is normalized in [-1, 1]. Bands (Earth-like):
      |lat| in [0.0, 0.33]   → trade easterlies (wind blows east → west)
      |lat| in [0.33, 0.66]  → westerlies (west → east)
      |lat| in [0.66, 1.00]  → polar easterlies (east → west)

    The east/west sign transitions smoothly across a narrow band around the
    boundary latitudes (smoothstep over ±_TRANSITION_HALF_WIDTH). A hard
    step at the boundaries — which is what the prior version had — produced
    visible sharp lines in the precipitation map where adjacent hexes
    suddenly had opposite winds; on the real Earth the transition zone is
    a few degrees of latitude wide (the polar front, the subtropical jet).
    """
    a = abs(latitude)
    sign = _zonal_wind_sign(a)
    return (sign, 0.0)


# Half-width of the smoothstep transition zone at each band boundary, in
# units of normalized latitude. 0.04 ≈ 4° on Earth's 90° pole-to-equator
# span — wider than the real polar front but enough to remove the artifact.
_TRANSITION_HALF_WIDTH = 0.04


def _zonal_wind_sign(lat_abs: float) -> float:
    """E-W wind sign as a continuous function of |latitude|.

    Returns −1 (easterly) for |lat| < 0.33 and |lat| > 0.66, +1 (westerly)
    in between, with smoothstep transitions at the two band boundaries.
    """
    def smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    w = _TRANSITION_HALF_WIDTH
    # Below the first transition zone — pure trade-easterlies.
    if lat_abs <= 0.33 - w:
        return -1.0
    # Inside the lower transition: smoothly flip from −1 to +1.
    if lat_abs < 0.33 + w:
        t = (lat_abs - (0.33 - w)) / (2.0 * w)
        return -1.0 + 2.0 * smoothstep(t)
    # Inside the westerlies band proper.
    if lat_abs <= 0.66 - w:
        return 1.0
    # Inside the upper transition: smoothly flip from +1 back to −1.
    if lat_abs < 0.66 + w:
        t = (lat_abs - (0.66 - w)) / (2.0 * w)
        return 1.0 - 2.0 * smoothstep(t)
    # Polar easterlies.
    return -1.0


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Same flat-top projection the renderer uses (1 unit = one hex-step)."""
    return 1.5 * h.q, _SQRT3 * (h.r + h.q / 2.0)


def _hex_round(q_frac: float, r_frac: float) -> Hex:
    """Round fractional axial coords to the nearest hex (Patel's cube round)."""
    s_frac = -q_frac - r_frac
    qi = round(q_frac)
    ri = round(r_frac)
    si = round(s_frac)
    dq = abs(qi - q_frac)
    dr = abs(ri - r_frac)
    ds = abs(si - s_frac)
    if dq > dr and dq > ds:
        qi = -ri - si
    elif dr > ds:
        ri = -qi - si
    # else implicit: si = -qi - ri
    return Hex(qi, ri)


def _axial_step_for_theta(theta: float) -> tuple[float, float]:
    """Axial offset that moves one hex of *physical* distance in cart direction θ.

    One adjacent-hex spacing is √3 in our cartesian projection, so a unit
    cartesian step ``(cos θ, sin θ)`` scaled by √3 corresponds to one hex of
    travel. The cart→axial transform then gives the fractional ``(dq, dr)``
    to add per upwind step.
    """
    cx = math.cos(theta) * _SQRT3
    cy = math.sin(theta) * _SQRT3
    dq = (2.0 / 3.0) * cx
    dr = -cx / 3.0 + cy / _SQRT3
    return dq, dr


def _compute_sea_breeze_field(
    elevation: ElevationLayer,
    sea: SeaLayer,
    config: WorldgenConfig,
) -> dict[Hex, tuple[float, float, float]]:
    """For each land hex, return (onshore_x, onshore_y, strength).

    The vector points FROM the nearest sea hex TO this hex in cartesian
    space — that is the *inland* direction, which is also the direction the
    onshore wind blows (sea → land). Strength decays linearly from 1.0 at
    the coast to 0.0 at ``sea_breeze_reach_km``.

    Implemented as a multi-source BFS from every sea hex; cheap (single
    pass) and gives both distance and nearest-source in one go.
    """
    hexes = list(elevation.elevation.keys())
    hex_set = set(hexes)
    nearest_sea: dict[Hex, Hex] = {}
    distance_km: dict[Hex, float] = {h: math.inf for h in hexes}
    queue: deque[Hex] = deque()
    step_km = config.hex_size_km * _SQRT3
    for h in hexes:
        if sea.is_ocean[h]:
            nearest_sea[h] = h
            distance_km[h] = 0.0
            queue.append(h)
    while queue:
        h = queue.popleft()
        d = distance_km[h]
        src = nearest_sea[h]
        for nb in h.neighbors():
            if nb not in hex_set:
                continue
            new_d = d + step_km
            if new_d < distance_km[nb]:
                distance_km[nb] = new_d
                nearest_sea[nb] = src
                queue.append(nb)

    reach = config.sea_breeze_reach_km
    field: dict[Hex, tuple[float, float, float]] = {}
    for h in hexes:
        if sea.is_ocean[h] or h not in nearest_sea:
            field[h] = (0.0, 0.0, 0.0)
            continue
        d = distance_km[h]
        if d >= reach:
            field[h] = (0.0, 0.0, 0.0)
            continue
        sea_h = nearest_sea[h]
        hx, hy = _hex_to_xy(h)
        sx, sy = _hex_to_xy(sea_h)
        dx, dy = hx - sx, hy - sy
        norm = math.hypot(dx, dy)
        if norm == 0:  # shouldn't happen for non-ocean hex, but be safe
            field[h] = (0.0, 0.0, 0.0)
            continue
        # Smoothstep decay: strongest at the coast, zero at reach, with
        # zero derivative at both endpoints so the field tapers gracefully
        # into the inland baseline instead of cutting off at a sharp ring.
        t = 1.0 - d / reach
        strength = t * t * (3.0 - 2.0 * t)
        field[h] = (dx / norm, dy / norm, strength)
    return field


def _per_hex_wind_theta(
    h: Hex,
    radius: int,
    sea_breeze_field: dict[Hex, tuple[float, float, float]],
    jitter_noise: PerlinNoise2D,
    config: WorldgenConfig,
) -> float:
    """Cartesian wind direction (radians, 0 = +x) for one hex.

    = normalize(zonal + sea_breeze_strength · onshore_unit · sea_breeze_falloff)
      rotated by Perlin jitter ∈ ±wind_jitter_amplitude_deg.
    """
    lat = _latitude_fraction(h, radius)
    base_x, base_y = _zonal_wind_cartesian(lat)
    sb_x, sb_y, sb_strength = sea_breeze_field.get(h, (0.0, 0.0, 0.0))
    wx = base_x + sb_x * sb_strength * config.sea_breeze_strength
    wy = base_y + sb_y * sb_strength * config.sea_breeze_strength
    cx, cy = _hex_to_xy(h)
    freq = config.hex_size_km / config.wind_jitter_wavelength_km
    jitter_sample = jitter_noise.sample(cx * freq, cy * freq)
    jitter_rad = jitter_sample * math.radians(config.wind_jitter_amplitude_deg)
    cos_j, sin_j = math.cos(jitter_rad), math.sin(jitter_rad)
    rx = cos_j * wx - sin_j * wy
    ry = sin_j * wx + cos_j * wy
    return math.atan2(ry, rx)


def _moisture_along_path(
    target: Hex,
    theta: float,
    elevation: ElevationLayer,
    sea: SeaLayer,
    temperatures: dict[Hex, float],
    max_steps: int,
    config: WorldgenConfig,
) -> float:
    """Walk upwind from ``target`` along cartesian direction ``theta`` and
    return the moisture deposited at ``target``."""
    dq_step, dr_step = _axial_step_for_theta(theta)
    moisture = 0.0
    deposited_here = 0.0
    prev_elev = elevation.sea_level
    prev_hex: Hex | None = None
    for step in range(max_steps, -1, -1):
        qf = target.q - dq_step * step
        rf = target.r - dr_step * step
        cur_hex = _hex_round(qf, rf)
        if cur_hex not in elevation.elevation:
            continue
        if sea.is_ocean[cur_hex]:
            t = temperatures[cur_hex]
            warmth = max(0.05, min(1.0, (t + 5.0) / 35.0))
            moisture += warmth * config.precip_max_ocean_pickup
            prev_elev = elevation.sea_level
        else:
            deposit = moisture * config.precip_loss_per_land
            cur_elev = elevation.elevation[cur_hex]
            dh = cur_elev - prev_elev
            if dh > 0:
                deposit += dh * config.precip_orographic_coef
            deposit = min(deposit, moisture)
            moisture -= deposit
            prev_elev = cur_elev
            if cur_hex == target:
                deposited_here = deposit
                break
        prev_hex = cur_hex
    _ = prev_hex  # quiet linters: kept for future debugging
    return deposited_here


def compute_precipitation(
    elevation: ElevationLayer,
    sea: SeaLayer,
    temperatures: dict[Hex, float],
    radius: int,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, float]:
    """Per-hex annual precipitation in mm.

    For each target hex, build a per-hex wind direction (zonal + sea-breeze
    + Perlin jitter), then average the moisture-sweep deposit over
    ``wind_path_samples`` paths spread evenly across ±``wind_path_spread_deg``
    around that base direction. Each individual sweep walks upwind in
    fractional axial space (one hex of physical distance per step),
    accumulates moisture over ocean and deposits with orographic uplift
    over land.
    """
    precip_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "precip_noise"))
    jitter_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "wind_jitter"))

    max_steps = min(config.wind_reach_hexes, max(40, radius * 2))

    sea_breeze_field = _compute_sea_breeze_field(elevation, sea, config)

    n_paths = max(1, config.wind_path_samples)
    spread_rad = math.radians(config.wind_path_spread_deg)

    precipitation: dict[Hex, float] = {}
    for h in elevation.elevation:
        if sea.is_ocean[h]:
            precipitation[h] = 0.0
            continue
        base_theta = _per_hex_wind_theta(
            h, radius, sea_breeze_field, jitter_noise, config,
        )
        # Sample N angles evenly spread across the cone, average their deposits.
        total = 0.0
        for i in range(n_paths):
            if n_paths == 1:
                offset = 0.0
            else:
                offset = (i / (n_paths - 1) - 0.5) * 2.0 * spread_rad
            theta = base_theta + offset
            total += _moisture_along_path(
                h, theta, elevation, sea, temperatures, max_steps, config,
            )
        deposited = total / n_paths

        n = precip_noise.sample(h.q * 0.03, h.r * 0.03)
        base = config.precip_base * 0.15  # soft floor over land
        value = base + deposited + n * config.precip_noise_amplitude
        precipitation[h] = max(0.0, value)

    # Final spatial smoothing: each pass averages land hexes with their
    # land neighbors (oceans untouched). Cleans up hex-scale roughness
    # introduced by the upwind hex-rounding and the discrete band model
    # without erasing the rain-shadow / coastal-gradient signal.
    for _ in range(max(0, config.precip_smoothing_passes)):
        precipitation = _smooth_precipitation(precipitation, sea)

    return precipitation


def _smooth_precipitation(
    precipitation: dict[Hex, float],
    sea: SeaLayer,
) -> dict[Hex, float]:
    """One uniform smoothing pass: each land hex → mean(self, land neighbors).

    Oceans are passed through unchanged. Edge hexes with fewer in-bounds
    land neighbors still produce a valid mean over whatever neighbors they
    do have, so the boundary isn't distorted.
    """
    out: dict[Hex, float] = {}
    for h, p in precipitation.items():
        if sea.is_ocean[h]:
            out[h] = p
            continue
        total = p
        count = 1
        for nb in h.neighbors():
            if nb in precipitation and not sea.is_ocean[nb]:
                total += precipitation[nb]
                count += 1
        out[h] = total / count
    return out


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
