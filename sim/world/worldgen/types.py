"""Intermediate data structures for the terrain generation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sim.world.hex import Hex


@dataclass(frozen=True, slots=True)
class HexData:
    """Per-hex data produced by the generation pipeline.

    ``elevation`` is in normalized units: 0.0 = sea level, 1.0 ≈ max terrestrial
    elevation in this world (typically ~4.5 km in real-world calibration).
    Negative values are below sea level (ocean floor).

    ``crop_suitability`` maps crop name → suitability in [0, 1] (0 = cannot grow,
    1 = ideal). Empty over water tiles.

    ``deposits`` maps resource name → deposit quantity (abstract units; relative
    only). Empty if no deposit at this hex.
    """

    elevation: float
    is_ocean: bool
    is_coast: bool
    is_lake: bool
    is_river: bool
    temperature_c: float
    precipitation_mm: float
    flow_accumulation: int  # upstream hex count, 1 = headwater
    biome: str
    crop_suitability: dict[str, float]
    deposits: dict[str, float]


@dataclass(frozen=True, slots=True)
class CropDefinition:
    """Environmental envelope for a crop, plus per-biome compatibility multipliers.

    Suitability is computed as the product of trapezoidal membership scores
    for temperature and precipitation, multiplied by a per-biome compatibility
    factor, with optional river / coast / irrigation bonuses.

    Each trapezoid has four points: ``abs_min ≤ opt_min ≤ opt_max ≤ abs_max``.
    Score is 0 outside ``[abs_min, abs_max]``, 1 within ``[opt_min, opt_max]``,
    and linearly interpolated on the ramps.
    """

    name: str
    # Temperature trapezoid (°C)
    temp_abs_min: float
    temp_opt_min: float
    temp_opt_max: float
    temp_abs_max: float
    # Precipitation trapezoid (mm/yr); if the crop is irrigated, the effective
    # precipitation includes the river bonus below.
    precip_abs_min: float
    precip_opt_min: float
    precip_opt_max: float
    precip_abs_max: float
    # Max normalized elevation a crop can grow at (above sea level).
    elev_max: float
    # Per-biome multiplier; biomes not listed default to 0 (crop cannot grow).
    biome_compatibility: dict[str, float]
    # Bonus suitability multiplier if the tile is itself a river hex.
    river_bonus: float
    # Multiplier added if any neighbor is a river/lake (irrigation access).
    river_adjacent_bonus: float
    # Bonus multiplier if the tile is a coast hex.
    coast_bonus: float
    # If True, treat the river/irrigation bonus as effective rainfall —
    # i.e., apply it to the precipitation trapezoid as a fixed addition
    # (useful for crops like rice that can be paddy-irrigated).
    irrigation_replaces_rain_mm: float


@dataclass(frozen=True, slots=True)
class ResourceDefinition:
    """A natural resource deposit type and its distribution parameters.

    Distribution model: a per-resource Perlin field (seeded child-RNG by name)
    sampled at each candidate hex; deposits exist where the field exceeds the
    rarity threshold AND the hex matches the eligibility rules below. Deposit
    quantity is proportional to (noise_value − rarity_threshold).
    """

    name: str
    # Biomes that can host this deposit. Empty tuple means "any land biome".
    host_biomes: tuple[str, ...]
    # Hard elevation bounds (normalized). e.g., copper porphyries above 0.05;
    # coal must be below 0.4. Values outside [min, max] are excluded.
    min_elevation: float
    max_elevation: float
    # Climate gating: deposits only where temperature ∈ [min, max] and
    # precipitation ∈ [min, max]. Defaults span all conditions (-∞..∞).
    min_temperature_c: float
    max_temperature_c: float
    min_precipitation_mm: float
    max_precipitation_mm: float
    # Spatial pattern: feature wavelength in km. Larger = bigger contiguous
    # districts (coal basin ~250 km). Smaller = more scattered (salt ~50 km).
    feature_wavelength_km: float
    # Top fraction of eligible hexes to seed deposits in (after biome/elev
    # gating). Higher = more abundant. ~0.05 = rare, ~0.25 = common.
    abundance: float
    # Mean deposit quantity multiplier. Final quantity =
    # mean_quantity × (1 + elevation_bonus × elevation_above_sea).
    mean_quantity: float
    # If positive, deposits at higher elevation are richer (relevant for
    # iron/copper in mountains).
    elevation_quantity_bonus: float
    # Tags this resource for special rendering / domain logic.
    # Currently informational only; can be one of:
    # "ore" | "fuel" | "evaporite" | "building" | "sedimentary" | "timber".
    category: str


@dataclass(frozen=True)
class WorldgenConfig:
    """Parameters for the world generation pipeline.

    All fields come from the ``[worldgen.*]`` config sections. Scale-dependent
    parameters are stored in **physical units** (km, km², mm per km of land
    fetch, etc.) and converted to hex-based units via ``hex_size_km`` at use
    time. This means changing ``hex_size_km`` automatically rescales noise
    frequencies, wind reach, river drainage thresholds and precipitation
    rates so the generated world looks the same at the chosen resolution.
    """

    hex_size_km: float

    # Elevation — physical-unit parameters
    land_fraction: float
    # Dominant feature wavelength (km) for the base fBm — controls how
    # frequently mountain ranges and basins recur.
    feature_wavelength_km: float
    noise_octaves: int
    noise_lacunarity: float
    noise_persistence: float
    # Domain-warp parameters: max warp magnitude in km, and the wavelength
    # of the warp field itself in km.
    warp_strength_km: float
    warp_wavelength_km: float
    ridge_octaves: int
    ridge_amplitude: float
    ridge_threshold: float
    falloff_strength: float
    falloff_power: float
    falloff_inner_fraction: float

    # Climate
    equator_temp_c: float
    polar_temp_c: float
    lapse_rate_c_per_km: float
    max_elevation_km: float
    temp_noise_amplitude: float
    precip_base: float
    # Max moisture an air parcel can absorb per km of warm ocean fetch (mm).
    precip_pickup_per_ocean_km: float
    # Fraction of moisture deposited per km of land traversal (continuous-rate
    # form; per-hex deposition is computed by integrating over hex_size_km).
    precip_loss_per_km: float
    precip_orographic_coef: float
    precip_noise_amplitude: float
    # Wind reach (km of upwind path traversed). Caps how far moisture can
    # travel before stopping.
    wind_reach_km: float

    # Hydrology
    # Minimum upstream-drainage area (km²) for a hex to be marked as a river.
    river_drainage_threshold_km2: float
    lake_min_depth: float
    river_carve_amount: float

    # Biome
    elevation_hills_threshold: float
    elevation_mountain_threshold: float
    elevation_snow_threshold: float
    tundra_max_temp_c: float
    taiga_max_temp_c: float
    temperate_max_temp_c: float
    desert_max_precip: float
    grassland_max_precip: float
    forest_max_precip: float
    cool_band_dry_threshold: float

    # Crops & resources — loaded from [worldgen.crops.*] and [worldgen.resources.*]
    crops: tuple[CropDefinition, ...] = ()
    resources: tuple[ResourceDefinition, ...] = ()

    # --- Derived properties (computed from hex_size_km) ---

    @property
    def hex_area_km2(self) -> float:
        """Area of one hex in km².

        With ``hex_size_km`` interpreted as the flat-to-flat (short) diameter
        of a regular flat-top hex, the area is ``(√3/2) · hex_size_km²``.
        At 5 km/hex this evaluates to ~21.65 km².
        """
        return 0.5 * math.sqrt(3.0) * self.hex_size_km * self.hex_size_km

    @property
    def noise_base_frequency(self) -> float:
        """Base noise frequency in cycles/hex.

        Scales inversely with hex size so that ``feature_wavelength_km`` of
        physical-world wavelength stays constant across resolutions.
        """
        return self.hex_size_km / self.feature_wavelength_km

    @property
    def warp_frequency(self) -> float:
        """Domain-warp noise frequency in cycles/hex."""
        return self.hex_size_km / self.warp_wavelength_km

    @property
    def warp_strength(self) -> float:
        """Domain-warp magnitude in hex coordinates."""
        return self.warp_strength_km / self.hex_size_km

    @property
    def wind_reach_hexes(self) -> int:
        """Wind walk distance in hex steps. Minimum 40 to guarantee enough
        ocean fetch in small worlds."""
        return max(40, int(round(self.wind_reach_km / self.hex_size_km)))

    @property
    def precip_max_ocean_pickup(self) -> float:
        """Moisture pickup per ocean *hex* (mm). Derived from per-km rate."""
        return self.precip_pickup_per_ocean_km * self.hex_size_km

    @property
    def precip_loss_per_land(self) -> float:
        """Moisture deposition fraction per land *hex*. Integrates the
        per-km loss rate over the length of one hex.

        Uses an exponential model: at per-km rate ``r``, the fraction lost
        crossing distance d is ``1 - exp(-r d)``.
        """
        return 1.0 - math.exp(-self.precip_loss_per_km * self.hex_size_km)

    @property
    def river_drainage_threshold(self) -> int:
        """River threshold expressed as a count of upstream hexes."""
        return max(1, int(round(self.river_drainage_threshold_km2 / self.hex_area_km2)))


@dataclass(frozen=True)
class ElevationLayer:
    """Output of the elevation layer.

    ``raw`` holds the un-normalized noise output. ``elevation`` is the normalized
    height field after falloff is applied (range approx [-1, 1]).
    """

    elevation: dict[Hex, float]
    sea_level: float  # threshold below which is ocean

    def is_ocean(self, hex: Hex) -> bool:
        return self.elevation[hex] < self.sea_level


@dataclass(frozen=True)
class SeaLayer:
    """Ocean/coast mask derived from elevation."""

    is_ocean: dict[Hex, bool]
    is_coast: dict[Hex, bool]


@dataclass(frozen=True)
class ClimateLayer:
    """Temperature (°C) and precipitation (mm/yr) per hex."""

    temperature_c: dict[Hex, float]
    precipitation_mm: dict[Hex, float]


@dataclass(frozen=True)
class HydrologyLayer:
    """Water flow results: filled elevation, lake/river masks, flow accumulation."""

    filled_elevation: dict[Hex, float]
    is_lake: dict[Hex, bool]
    is_river: dict[Hex, bool]
    flow_accumulation: dict[Hex, int]
    downstream: dict[Hex, Hex | None]  # None = sink (ocean or terminal lake)
