"""Resources layer (L7): per-hex agricultural crop suitability and natural
resource deposits.

Two independent sub-systems:

1. **Crop suitability** is a deterministic function of the climate and biome
   fields already computed by earlier layers. Each crop carries an
   environmental envelope (trapezoidal membership over temperature and
   precipitation) plus per-biome compatibility multipliers and optional
   river/coast bonuses. Output is a per-hex map of ``{crop: suitability ∈ [0, 1]}``.

2. **Resource deposits** are seeded by a per-resource Perlin noise field
   gated by biome / elevation / climate eligibility. Each resource has a
   feature wavelength (in km, so deposit district size is independent of
   ``hex_size_km``) and an abundance level (fraction of eligible tiles).
   Output is a per-hex map of ``{resource: quantity > 0}`` (absent if no
   deposit).

Both sub-systems are pure functions of the upstream layer outputs and a
seeded child-RNG. New resources or crops can be added via config without
touching existing ones because each resource derives its noise field from
``rng.child("worldgen", "resources", resource.name)``.
"""

from __future__ import annotations

import math

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.noise import PerlinNoise2D, fbm
from sim.world.worldgen.types import (
    ClimateLayer,
    CropDefinition,
    ElevationLayer,
    HydrologyLayer,
    ResourceDefinition,
    SeaLayer,
    WorldgenConfig,
)


# ---------------------------------------------------------------------------
# Crop suitability
# ---------------------------------------------------------------------------


def _trapezoid(x: float, lo: float, opt_lo: float, opt_hi: float, hi: float) -> float:
    """Trapezoidal membership: 0 outside [lo, hi], 1 within [opt_lo, opt_hi],
    linear ramps between. Assumes ``lo ≤ opt_lo ≤ opt_hi ≤ hi``.

    If ``opt_lo == lo`` the rising ramp is replaced by a step at lo (and
    similarly on the falling side). NaN-safe within reasonable inputs.
    """
    if x <= lo or x >= hi:
        return 0.0
    if x < opt_lo:
        return 0.0 if opt_lo == lo else (x - lo) / (opt_lo - lo)
    if x > opt_hi:
        return 0.0 if hi == opt_hi else (hi - x) / (hi - opt_hi)
    return 1.0


def crop_suitability(
    crop: CropDefinition,
    *,
    biome: str,
    temperature_c: float,
    precipitation_mm: float,
    elevation: float,
    is_river: bool,
    is_coast: bool,
    has_water_neighbor: bool,
) -> float:
    """Compute the suitability ∈ [0, 1] of a single crop at one hex.

    Hard exclusions (return 0):
      - biome not listed in ``crop.biome_compatibility``
      - elevation above ``crop.elev_max``
      - temperature or (effective) precipitation outside the crop's absolute bounds

    Otherwise: multiplicative combination of temperature score × precipitation
    score × biome multiplier × (1 + river/coast bonuses, clamped to [0, 1]).
    """
    biome_mult = crop.biome_compatibility.get(biome, 0.0)
    if biome_mult <= 0.0:
        return 0.0
    if elevation > crop.elev_max:
        return 0.0

    # Effective precipitation: optionally add an "irrigation equivalent" if a
    # river/lake neighbor (or self) provides it. This is what lets rice grow
    # in paddies fed by a river even when local rainfall is low.
    effective_precip = precipitation_mm
    if crop.irrigation_replaces_rain_mm > 0.0 and (is_river or has_water_neighbor):
        effective_precip += crop.irrigation_replaces_rain_mm

    t_score = _trapezoid(
        temperature_c, crop.temp_abs_min, crop.temp_opt_min,
        crop.temp_opt_max, crop.temp_abs_max,
    )
    if t_score == 0.0:
        return 0.0
    p_score = _trapezoid(
        effective_precip, crop.precip_abs_min, crop.precip_opt_min,
        crop.precip_opt_max, crop.precip_abs_max,
    )
    if p_score == 0.0:
        return 0.0

    bonus = 1.0
    if is_river:
        bonus += crop.river_bonus
    elif has_water_neighbor:
        # Mild bonus for tiles adjacent to a river/lake (irrigation potential).
        bonus += crop.river_adjacent_bonus
    if is_coast:
        bonus += crop.coast_bonus

    return min(1.0, t_score * p_score * biome_mult * bonus)


def compute_crop_suitability(
    hexes: list[Hex],
    elevation: ElevationLayer,
    sea: SeaLayer,
    climate: ClimateLayer,
    hydrology: HydrologyLayer,
    biomes: dict[Hex, str],
    crops: tuple[CropDefinition, ...],
) -> dict[Hex, dict[str, float]]:
    """Compute per-hex suitability for every configured crop.

    Returns an empty inner dict for ocean / lake hexes (no agriculture on water).
    """
    out: dict[Hex, dict[str, float]] = {}
    for h in hexes:
        if sea.is_ocean[h] or hydrology.is_lake[h]:
            out[h] = {}
            continue
        elev_above = elevation.elevation[h] - elevation.sea_level
        # "Has water neighbor" includes river or lake on any adjacent hex.
        has_water = any(
            (n in hydrology.is_lake and (hydrology.is_lake[n] or hydrology.is_river[n]))
            for n in h.neighbors()
        )
        scores: dict[str, float] = {}
        for crop in crops:
            s = crop_suitability(
                crop,
                biome=biomes[h],
                temperature_c=climate.temperature_c[h],
                precipitation_mm=climate.precipitation_mm[h],
                elevation=elev_above,
                is_river=hydrology.is_river[h],
                is_coast=sea.is_coast.get(h, False),
                has_water_neighbor=has_water,
            )
            if s > 0.0:
                scores[crop.name] = s
        out[h] = scores
    return out


# ---------------------------------------------------------------------------
# Resource deposits
# ---------------------------------------------------------------------------


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Axial → flat cartesian (same mapping as the elevation layer uses)."""
    x = 1.5 * h.q
    y = math.sqrt(3.0) * (h.r + h.q / 2.0)
    return x, y


def _quantile(values: list[float], q: float) -> float:
    """Return the ``q``-th quantile (0 ≤ q ≤ 1) of a sorted-or-unsorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(q * (len(s) - 1))
    return s[idx]


def compute_resource_deposits(
    hexes: list[Hex],
    elevation: ElevationLayer,
    sea: SeaLayer,
    climate: ClimateLayer,
    hydrology: HydrologyLayer,
    biomes: dict[Hex, str],
    resources: tuple[ResourceDefinition, ...],
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, dict[str, float]]:
    """Compute per-resource deposit fields and combine into per-hex deposits.

    Algorithm per resource:
      1. Build the eligibility set: hexes whose biome ∈ host_biomes (or
         host_biomes is empty, meaning any land) and elevation / climate
         within the resource's allowed range.
      2. Sample a Perlin field (per-resource child-RNG) at each eligible hex.
      3. Threshold the field by the top ``abundance`` quantile of eligible
         hexes → deposit present where noise exceeds threshold.
      4. Quantity = mean_quantity × (noise_value − threshold) /
         (1 − threshold) × (1 + elev_bonus × elevation_above_sea).
    """
    out: dict[Hex, dict[str, float]] = {h: {} for h in hexes}
    for resource in resources:
        wavelength_hex = max(1e-6, resource.feature_wavelength_km / config.hex_size_km)
        freq = 1.0 / wavelength_hex
        noise = PerlinNoise2D.from_rng(
            rng.child("worldgen", "resources", resource.name)
        )

        # 1. Eligibility — accumulate (hex, noise_value) pairs.
        eligible: list[tuple[Hex, float]] = []
        for h in hexes:
            if sea.is_ocean[h]:
                continue
            if hydrology.is_lake[h] and "lake" not in resource.host_biomes:
                continue
            if resource.host_biomes and biomes[h] not in resource.host_biomes:
                continue
            elev_above = elevation.elevation[h] - elevation.sea_level
            if elev_above < resource.min_elevation or elev_above > resource.max_elevation:
                continue
            t = climate.temperature_c[h]
            if t < resource.min_temperature_c or t > resource.max_temperature_c:
                continue
            p = climate.precipitation_mm[h]
            if p < resource.min_precipitation_mm or p > resource.max_precipitation_mm:
                continue

            x, y = _hex_to_xy(h)
            # 3-octave fBm gives smooth districts with a bit of internal variation.
            v = fbm(noise, x, y,
                    octaves=3, lacunarity=2.0, persistence=0.5,
                    base_frequency=freq)
            # Map [-1, 1] → [0, 1].
            v = 0.5 * (v + 1.0)
            eligible.append((h, v))

        if not eligible:
            continue

        # 2-3. Compute the threshold so that ~``abundance`` fraction of eligible
        # hexes pass.
        values = [v for _, v in eligible]
        threshold = _quantile(values, 1.0 - resource.abundance)
        denom = max(1e-6, 1.0 - threshold)

        # 4. Emit deposits.
        for h, v in eligible:
            if v < threshold:
                continue
            strength = (v - threshold) / denom  # [0, 1]
            elev_above = elevation.elevation[h] - elevation.sea_level
            quantity = (
                resource.mean_quantity
                * (0.5 + 0.5 * strength)  # 50–100 % of mean per cell
                * (1.0 + resource.elevation_quantity_bonus * max(0.0, elev_above))
            )
            out[h][resource.name] = quantity

    return out
