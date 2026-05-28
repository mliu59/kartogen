"""Intermediate data structures for the terrain generation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from worldgen.hex import Hex


@dataclass(frozen=True, slots=True)
class HexData:
    """Per-hex data produced by the generation pipeline.

    ``elevation`` is in normalized units: 0.0 = sea level, 1.0 ≈ max terrestrial
    elevation in this world (typically ~4.5 km in real-world calibration).
    Negative values are below sea level (ocean floor).

    Plate fields (``plate_id``, ``plate_type``, ``nearest_boundary_type``,
    ``distance_to_boundary_km``) are populated when ``mask_mode == "plates"``
    and ``None`` otherwise. ``nearest_boundary_type`` is one of
    ``"cc_convergent"``, ``"oc_convergent"``, ``"oo_convergent"``,
    ``"divergent"``, ``"transform"`` — or ``None`` if no boundary within the
    falloff radius.
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
    plate_id: int | None
    plate_type: str | None
    nearest_boundary_type: str | None
    distance_to_boundary_km: float | None


@dataclass(frozen=True)
class PlateConfig:
    """Parameters for the plate-tectonics continent generator.

    Static Voronoi plates (no time-simulated motion): seeds are placed by
    rejection sampling, each plate is classified continental or oceanic, and
    each gets a randomly-drawn motion vector. Boundary effects (mountains,
    rifts) are applied as a function of distance to the nearest boundary and
    the boundary's classification.
    """

    # Macro structure
    count: int                              # number of plates
    continental_fraction: float             # share of plates that are continental
    min_separation_km: float                # rejection-sampling distance for seeds
    # Plate seed positions can be biased toward the world center (>0) or pushed
    # toward the edge (<0). 0 = uniform across all hexes. Used by presets like
    # `island_continent` (centered) and `continental_coast` (edge-biased).
    seed_radial_bias: float

    # Boundary geometry — irregular plate edges via warped Voronoi
    boundary_warp_strength_km: float
    boundary_warp_wavelength_km: float

    # Motion
    motion_speed: float                     # scale on random unit motion vectors

    # Elevation contributions
    continental_baseline: float             # added to fBm everywhere on a continental plate
    oceanic_baseline: float                 # added to fBm everywhere on an oceanic plate
    mountain_amplitude: float               # peak boundary uplift, cc_convergent
    coastal_range_amplitude: float          # oc_convergent
    island_arc_amplitude: float             # oo_convergent
    rift_depth: float                       # depression, divergent boundaries
    boundary_falloff_km: float              # decay length of boundary effects inland
    # Width of the soft-Voronoi blend used to compute the per-hex baseline:
    # at a hex whose distance to its 2nd-nearest plate seed is within this
    # many km of its distance to the nearest, baselines blend with smoothstep
    # weights. 0 = hard step (old behavior); 100–300 km = continental-shelf
    # style smooth transition between plates of different type.
    baseline_blend_km: float
    # Dot-product of relative motion onto inter-plate normal must exceed this
    # for a boundary to be classified convergent/divergent; below = transform.
    convergence_threshold: float


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
    # Continent mask: shapes the macro landmass layout before quantile sea level.
    # Modes:
    #   "none"   — no shaping; sea level alone separates land/ocean. Combined
    #              with a low land_fraction this produces an archipelago; with
    #              a high land_fraction it produces a continuous landmass.
    #   "radial" — concentric falloff toward the map edge (island continent).
    #   "axial"  — one-sided ramp along a seed-chosen direction; the "ocean
    #              side" is pulled down (continental coast).
    #   "dual"   — two anchor points along a seed-chosen axis; mid-map and
    #              edges pulled down (two-continent worlds).
    mask_mode: str
    mask_strength: float
    mask_power: float
    # Fraction-from-center (radial) or fraction-from-anchor (axial/dual) inside
    # which no mask pull is applied.
    mask_inner_fraction: float
    # For mode="dual": distance from world center to each anchor, as a fraction
    # of the cartesian world radius. Ignored by other modes.
    mask_anchor_fraction: float
    # Required when mask_mode == "plates"; may be None for the analytic modes.
    plates: PlateConfig | None

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
    # Per-hex wind direction is base zonal (latitude band) + sea-breeze
    # onshore component (annual mean) + Perlin jitter, all summed and
    # renormalized. These knobs shape the deviation from pure zonal flow.
    # Set everything to 0 to recover the original axis-aligned model.
    wind_jitter_amplitude_deg: float       # max ± angular perturbation per hex
    wind_jitter_wavelength_km: float       # Perlin wavelength of the jitter field
    sea_breeze_strength: float             # weight on onshore component (0..1)
    sea_breeze_reach_km: float             # onshore strength falls linearly to 0 here
    # Multiple-path sampling: per target hex, run the moisture sweep
    # `wind_path_samples` times with angles spread ±wind_path_spread_deg
    # around the base wind direction; average the deposits.
    wind_path_samples: int
    wind_path_spread_deg: float
    # Spatial smoothing of the final precipitation field. Each pass replaces
    # each land hex's value with a weighted average of itself and its land
    # neighbors. 0 = off; 1–3 = increasing smoothness. Helps clean up the
    # hex-scale granularity from the moisture sweep / hex-rounding without
    # changing the model's physical assumptions.
    precip_smoothing_passes: int

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
