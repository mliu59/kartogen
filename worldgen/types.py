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

    ``plate_id`` is the id of the tectonic-sim plate owning this hex in the
    final simulated state.
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
    plate_id: int  # final tectonic-sim plate owning this hex
    # Lithosphere column produced by the dynamic tectonics simulation.
    # ``crust_type`` is "continental" or "oceanic"; ``crust_age_myr`` is time
    # since formation at a ridge (meaningful for oceanic crust — continental
    # crust ages slowly and the value just accumulates).
    crust_thickness_km: float
    crust_type: str
    crust_age_myr: float
    # Ocean layer outputs. ``distance_to_ocean_km`` is 0 for ocean hexes and
    # the BFS distance to the nearest ocean hex for land hexes.
    # ``coastal_temp_anomaly_c`` is the temperature anomaly inherited from
    # nearby ocean currents (0 for ocean hexes themselves; instead they carry
    # ``current_temp_anomaly_c``).
    # ``gyre_id`` is set only on ocean hexes (None on land).
    distance_to_ocean_km: float
    current_temp_anomaly_c: float
    coastal_temp_anomaly_c: float
    gyre_id: int | None


# TectonicsConfig used to be a worldgen-side dataclass that duplicated
# the physics tunables from ``tectonic_sim.SimConfig``. Post-refactor,
# the tectonic_sim module owns its own TOML (``config/tectonic_sim.toml``)
# and SimConfig dataclass, and worldgen loads it directly. We keep the
# name as a transparent alias so existing worldgen callsites that read
# ``config.tectonics.X`` continue to work unchanged.
from tectonic_sim.types import SimConfig as TectonicsConfig  # noqa: F401


@dataclass(frozen=True)
class WorldShape:
    """Rectangular world footprint, in physical units.

    The world hex set is every hex whose flat-top pixel centre falls
    within ``[-width_km/2, width_km/2] × [-height_km/2, height_km/2]``.
    The map renders as a rectangle of these km dimensions; the hex grid
    fills it as tightly as the integer axial lattice allows.

    Configured via ``[worldgen.world] width_km, height_km``.
    """

    width_km: float
    height_km: float

    @property
    def half_width_km(self) -> float:
        return self.width_km / 2.0

    @property
    def half_height_km(self) -> float:
        return self.height_km / 2.0


@dataclass(frozen=True)
class OceanConfig:
    """Ocean-currents + continentality parameters (Tier 2 climate).

    Each connected ocean basin is split by hemisphere into one or two gyres,
    each rotating clockwise in the northern hemisphere and counter-clockwise
    in the southern (the Coriolis-driven sign). Per ocean hex, the current
    direction is the tangent of (hex − gyre_center); the temperature anomaly
    is derived from the planetary latitudinal temperature differential
    between the hex's location and a sample point ``current_persistence_km``
    upstream. This single-pass formula gives warm western-boundary currents
    and cold eastern-boundary currents without an explicit advection solver.
    """

    # How far upstream we sample for the source latitude. Real Earth: the
    # Gulf Stream traverses ~4000 km from Florida to Iceland; 2000 km is a
    # reasonable mid-strength default.
    current_persistence_km: float
    # Scale on the (upstream_temp − local_temp) differential. 1.0 = current
    # carries the full source-temp anomaly; 0.5 = half.
    current_anomaly_strength: float
    # Cap on absolute temperature anomaly (°C). Real ocean current anomalies
    # typically peak at ±5–10 °C.
    max_current_anomaly_c: float

    # Coastal pickup: each coastal land hex inherits a fraction of the
    # adjacent ocean's current anomaly, decaying exponentially with distance
    # inland. ``pickup_fraction`` is the value at the coast (0 km inland);
    # ``decay_km`` is the e-folding distance.
    coastal_pickup_fraction: float
    coastal_decay_km: float

    # Continentality: precip floor is multiplied by exp(−dist_km / scale_km).
    # At the coast, factor = 1.0; ~1500 km inland (with scale=1500), ~0.37.
    continentality_dry_scale_km: float


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

    # World footprint (width × height in km). See ``WorldShape``.
    world: WorldShape

    # Hyperparameter controlling the magnitude of random exploration
    # applied to the tectonics physics config before each run.
    #   0 (default) → deterministic run from the configured TectonicsConfig.
    #   > 0         → each tectonic_sim field is drawn from a Normal around
    #                 its configured value, with std × param_temperature.
    # The same hyperparameter is intended to extend to other worldgen
    # subsystems (climate priors, ocean coefficients, etc.) in the future;
    # for now it only affects the tectonics simulation. See
    # ``tectonic_sim.randomize_sim_config``.
    param_temperature: float

    # Elevation — physical-unit parameters
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
    # Blend weight between the tectonic baseline (1.0 = pure tectonics) and
    # the fBm/ridged noise field (0.0 = pure noise). The noise contribution
    # is scaled by the complement so the two together stay in roughly
    # [-1, 1].
    tectonic_blend_weight: float
    # Time-stepped tectonics simulation parameters.
    tectonics: TectonicsConfig
    # Ocean currents + continentality (Tier 2 climate enhancements).
    ocean: OceanConfig

    # Climate
    # Latitude window the hex grid covers, in degrees. The map's r-axis is
    # interpreted as a slice of latitude: r = -radius is at ``map_lat_max``
    # (north edge), r = +radius is at ``map_lat_min`` (south edge). The
    # latitude window is *independent* of ``hex_size_km`` — you can simulate
    # a fantasy-proportioned world where 1000 km spans 20° of latitude.
    # Defaults of (-90, 90) reproduce the original pole-to-pole behaviour.
    map_lat_min: float
    map_lat_max: float
    # Planet-wide climate anchors: ``equator_temp_c`` is the temperature at
    # geographic latitude 0°, ``polar_temp_c`` at ±90°. The map samples a
    # slice of that gradient via its lat window.
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
    """Per-hex climate fields.

    ``wind_direction`` is a unit vector in the same screen-cartesian space
    the renderer and ocean layer use (+x = east, +y = south). Computed once
    by ``compute_wind_directions`` and reused by both precipitation and the
    wind-preview render.
    """

    temperature_c: dict[Hex, float]
    precipitation_mm: dict[Hex, float]
    wind_direction: dict[Hex, tuple[float, float]]


@dataclass(frozen=True)
class HydrologyLayer:
    """Water flow results: filled elevation, lake/river masks, flow accumulation."""

    filled_elevation: dict[Hex, float]
    is_lake: dict[Hex, bool]
    is_river: dict[Hex, bool]
    flow_accumulation: dict[Hex, int]
    downstream: dict[Hex, Hex | None]  # None = sink (ocean or terminal lake)
