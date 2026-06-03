"""Shared data types for ``tectonic_sim``.

Public surface:

  - ``WorldRect`` — toroidal simulation domain in km
  - ``SimConfig`` — physics tunables (read by polygon_sim + kartogen bridge)
  - ``CRUST_CONTINENTAL`` / ``CRUST_OCEANIC`` — int8 codes
  - ``crust_type_code`` / ``crust_type_name`` — string ↔ int

Crust type encoding: integer codes ``CRUST_CONTINENTAL = 0`` and
``CRUST_OCEANIC = 1`` rather than strings, so the per-cell field can
live in an ``int8`` array. Helpers convert to/from strings at the
public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

# Type alias for "scalar or numpy array of floats" — used in WorldRect's
# wrap helpers which work on both single coordinates and bulk arrays.
FloatLike = Union[float, np.ndarray]


# Crust type encoding for integer arrays.
CRUST_CONTINENTAL: int = 0
CRUST_OCEANIC: int = 1

_CRUST_TYPE_NAMES = ("continental", "oceanic")


def crust_type_name(code: int) -> str:
    """Map an integer crust code to its string name."""
    return _CRUST_TYPE_NAMES[code]


def crust_type_code(name: str) -> int:
    """Map a crust type name to its integer code. Raises on unknown."""
    if name == "continental":
        return CRUST_CONTINENTAL
    if name == "oceanic":
        return CRUST_OCEANIC
    raise ValueError(f"unknown crust_type {name!r}")


@dataclass(frozen=True)
class WorldRect:
    """Simulation domain in km, centred on (0, 0).

    The domain is **always** a torus — particles/cells that drift past
    one edge re-enter from the opposite edge, and cross-edge distance
    queries use the toroidal shortest-path metric. There is no other
    boundary mode.
    """

    width_km: float
    height_km: float

    @property
    def half_width_km(self) -> float:
        return self.width_km / 2.0

    @property
    def half_height_km(self) -> float:
        return self.height_km / 2.0

    @property
    def area_km2(self) -> float:
        return self.width_km * self.height_km

    # --- Toroidal geometry helpers ---

    def wrapped_delta_xy(
        self, dx: FloatLike, dy: FloatLike,
    ) -> tuple[FloatLike, FloatLike]:
        """Toroidal shortest-path delta for one or many ``(dx, dy)``."""
        wx = (dx + self.half_width_km) % self.width_km - self.half_width_km
        wy = (dy + self.half_height_km) % self.height_km - self.half_height_km
        return wx, wy


@dataclass(frozen=True)
class SimConfig:
    """Physics tunables for the rigid-polygon simulator.

    Loaded by ``config_loader.load_sim_config`` from a TOML table.
    Every field is required (no defaults at construction time so
    missing-config bugs surface at load, not as silent zeros
    downstream). All fields are threaded through the per-tick polygon-
    sim modules — there is no shadow set of module-level constants any
    more.

    Groups:

      - plate population (count, fraction, motion cap, seed bias)
      - sim duration (n_ticks, dt_myr)
      - crust thicknesses (continental, oceanic, rift)
      - half-space cooling (ridge_depth, subsidence_rate, max_ocean_depth)
      - continental isostasy (reference_thickness, factor) + sea level
      - collision (orogeny, folding_ratio, folding_displacement,
        subduction_arc)
      - velocity damping, erosion, snapshot capture
      - Voronoi seeding (warp_*, weight_*, init_thickness_*)
      - per-cell physics (init_angular_velocity, momentum_*, fusion_*,
        accretion_*, hotspot_*, rift_*, buoyancy_*, alpha_factor,
        min_continental_thickness, fragment_spawn_threshold)
    """

    # --- Plate population ---
    plate_count: int
    continental_fraction: float
    motion_speed_kmpy: float
    seed_radial_bias: float                       # 0 = uniform, >0 = centre, <0 = edge

    # --- Sim duration ---
    n_ticks: int
    dt_myr: float

    # --- Crust thicknesses ---
    continental_thickness_km: float
    oceanic_thickness_km: float
    rift_thickness_km: float

    # --- Half-space cooling (oceanic floor depth) ---
    ridge_depth_km: float
    ridge_subsidence_rate: float
    max_ocean_depth_km: float

    # --- Continental isostasy ---
    continental_reference_thickness_km: float
    continental_isostasy_factor: float
    sea_level_km: float

    # --- Collision ---
    folding_ratio: float
    # Continental-continental fold-and-thrust belt (over-rider side).
    # Each tick, the contested-cell fold mass is distributed inland
    # (opposite the over-rider's velocity) across a band of depth
    # ``folding_belt_depth_km`` using a **hybrid plateau profile**:
    # low at the suture, a smooth half-cosine RAMP up over
    # ``folding_belt_ramp_km``, a broad flat PLATEAU across the middle,
    # then a cosine TAPER to zero over ``folding_belt_taper_km`` at the
    # far inland edge. This matches a mature double-vergent orogen
    # (the suture is a structural low; relief sits inland on a broad
    # high plateau — Tibet/Altiplano). Weights are normalised to sum to
    # 1 so the total deposited mass is exactly ``folding_ratio · loser
    # thickness`` regardless of profile shape. Degenerate cases:
    # ramp+taper ≥ depth collapses the plateau to zero → a smooth
    # inland-peaked bump; ramp = taper = 0 → a flat top-hat.
    folding_belt_depth_km: float
    folding_belt_ramp_km: float
    folding_belt_taper_km: float
    # Loser-side fold belt — narrower, sharper inland deposit on the
    # *down-going* plate's near-suture interior. Models the Himalayan
    # foothill / Lesser-Himalaya pattern: slices of the underthrusting
    # plate get scraped off and stacked along the suture on its own
    # side. ``folding_loser_side_ratio`` is the fraction of the loser's
    # cell thickness redeposited back onto the loser (in addition to
    # ``folding_ratio`` going to the over-rider). Sum of the two ratios
    # should be ≤ 1 to avoid creating mass; the remainder represents
    # crust "subducted to mantle". Belt starts one cell into the loser's
    # interior (the suture itself now belongs to the over-rider).
    folding_loser_side_ratio: float
    folding_belt_loser_depth_km: float
    folding_belt_loser_decay_km: float

    # --- Velocity damping ---
    velocity_damping_strength: float
    # Extra damping multiplier applied to the share of a plate's
    # contested cells that are in continental–continental contention
    # (this plate continental AND another continental plate claims the
    # same cell). C-C collisions should bleed off convergence faster
    # than C-O / O-O contacts: two buoyant continents lock and the
    # kinetic energy goes into crustal thickening rather than continued
    # convergence. Higher → the smaller continent is overridden less
    # before the pair arrests, so less of its column is carved and
    # dumped as fold mass (lower runaway peaks). 1.0 = no extra C-C
    # damping (uniform with other contacts).
    cc_velocity_damping_multiplier: float

    # --- Erosion / snapshot capture ---
    erosion_period: int
    erosion_strength: float
    snapshot_period_ticks: int
    # How often (in ticks) to re-snap each plate's rotation pivot
    # (``body_pivot_km``) onto its current body centroid. The pivot keeps
    # the plate spinning about its own centre of area; as the plate
    # deforms the centroid migrates, so the pivot is periodically
    # re-anchored (world-preserving — only the pivot label moves). 1 =
    # every tick (pivot always tracks the centroid); larger = cheaper but
    # the pivot lags the centroid between recenters. 0 disables (pivot
    # stays where seeding put it).
    recenter_period_ticks: int

    # =====================================================================
    # Polygon-sim per-cell physics fields. All read by the matching
    # polygon_sim phase module and perturbable by
    # ``randomize_sim_config``.
    # =====================================================================
    init_speed_min_ratio: float
    plate_area_per_plate_km2: float
    init_angular_velocity_max_rad_per_myr: float
    angular_damping_multiplier: float
    momentum_restitution: float
    momentum_contact_boost: float
    fusion_overlap_threshold: float
    fusion_both_continental_only: bool
    # Suturing gate: a candidate pair only welds into one plate once
    # momentum exchange has arrested their convergence — i.e. their
    # contact-normal relative velocity has decayed to at or below this
    # (km/Myr). Fusion is then the natural endpoint of momentum
    # convergence rather than a parallel mechanic that fires the instant
    # masks overlap. Large value → fuses on overlap alone (legacy);
    # small value → only fully-locked pairs weld.
    fusion_max_relative_velocity_kmpy: float
    accretion_prob_per_boundary_per_tick: float
    accretion_cells_per_event: int
    accretion_inland_offset_min_km: float
    accretion_inland_offset_max_km: float
    rift_prob_per_tick: float
    rift_min_plate_cells: int
    rift_divergence_ratio: float
    hotspot_density_per_km2: float
    hotspot_erupt_prob_per_tick: float
    hotspot_thickness_bump_km: float
    hotspot_island_radius_km: float
    hotspot_birth_stagger_ticks: int
    hotspot_lifespan_mean_ticks: float
    hotspot_lifespan_std_ticks: float

    # =====================================================================
    # Polygon-sim physics tunables that USED to live as module-level
    # constants in ``tectonic_sim.polygon_sim.types``. Moved here so
    # there's a single source of truth for everything tunable. The
    # polygon_sim physics functions read these via the SimConfig that
    # gets threaded through every per-tick phase.
    # =====================================================================

    # Cell-grid resolution. Polygon sim derives (gy, gx) from
    # (domain, target_cell_km). Smaller cell → finer grid + higher cost.
    target_cell_km: float
    # Scales the maximum plate translation speed relative to the
    # geometric cap (min(sim_w, sim_h) / total_time). 0.5 = at most
    # half the world over the run.
    translation_speed_ratio: float

    # Released-component handling: components ≥ threshold spawn as new
    # plates; smaller redistribute to neighbours.
    fragment_spawn_threshold: int

    # Alpha-complex circumradius cutoff (× cell_km). Used by polygon
    # extraction.
    alpha_factor: float

    # Continental priority multiplier in per-cell contention. At 50, a
    # 1-cell continental plate has the same priority as a 50-cell
    # oceanic plate.
    crust_continental_weight: float

    # Pyplatec buoyancy bonus on young oceanic crust. Decays linearly
    # to zero at ``max_buoyancy_age_myr``.
    buoyancy_bonus_frac: float
    max_buoyancy_age_myr: float

    # Continental cells thinned below this threshold (km) are absorbed
    # by their over-rider via thickness transfer to a neighbour, and
    # the cell reverts to oceanic. 0 disables.
    min_continental_thickness_km: float

    # --- Divergent-gap fill: continental-basin override ---
    #
    # When the trailing-edge fill encounters a connected gap whose
    # immediate surround is overwhelmingly continental, stamp the gap
    # as a *continental basin* instead of fresh oceanic ridge. This
    # captures the geology of foreland basins / inland troughs that
    # form between converging continental blocks (C-C suture interior)
    # — material plastered into the gap is continental, not oceanic.
    #
    #   divergent_fill_continental_threshold — surround continental
    #     fraction at or above which the component fills as continental.
    #     Set to 1.0 to disable (always-oceanic, legacy behaviour).
    #
    #   divergent_fill_basin_depth_km — how much thinner the basin
    #     floor is than the mean continental thickness of its surround.
    #     Per-component scalar; same thickness across the whole basin.
    #     Clamped at min_continental_thickness_km so the cell isn't
    #     reabsorbed on the next tick.
    divergent_fill_continental_threshold: float
    divergent_fill_basin_depth_km: float

    # Initial plate-shape naturalisation (Methods 1+2: domain warp +
    # power weights). All values per tick or per draw.
    voronoi_warp_amplitude_km: float
    voronoi_warp_sigma_cells: float
    voronoi_warp_jaggedness: float
    voronoi_warp_jagged_sigma_cells: float
    voronoi_weight_sigma: float
    voronoi_weight_scale_km: float

    # Initial thickness variation overlays.
    #
    #   init_thickness_per_plate_sigma — per-plate scalar multiplier
    #     drawn log-normally. 0 disables. Knob for size of plate-to-plate
    #     average-thickness variability.
    #
    #   continental_relief_* — per-cell Perlin fBm thickness perturbation
    #     applied to continental cells only, zero-mean per plate. This is
    #     "ancient basement topography": noise wavelengths in the hundreds
    #     of km produce shelves, inland basins, straits, and continental
    #     islands after sea-level sampling. Wavelength controls the
    #     scale of features (200 km → small archipelagos, 1500 km → broad
    #     basins). Amplitude is in physical km — typical: 4–8 km against
    #     the ~35 km continental baseline. 0 → flat continental interiors.
    init_thickness_per_plate_sigma: float
    continental_relief_amplitude_km: float
    continental_relief_wavelength_km: float
    continental_relief_octaves: int
    continental_relief_persistence: float

    # --- Edge smoothing (non-physics) ---
    #
    # A Gaussian-blur pass applied to crust thickness, weighted per-cell
    # by a Perlin alpha field. Runs at exactly two points in the sim:
    #
    #   - t=0 (after seeding, post continental_relief)
    #   - t=final (after the last tick, before polygon construction)
    #
    # It is NOT a physics process — it is an algorithmic edge-smoothing
    # pass that softens sharp thickness boundaries (and therefore the
    # rendered topography) where the Perlin field is high, while leaving
    # them sharp where it is low. Erosion stays in ``aging.py``.
    #
    # Per-cell blend: ``out = (1 − α) * thickness + α * gaussian(thickness)``.
    # α is the Perlin field normalised into ``[alpha_min, alpha_max]``,
    # so ``alpha_min = alpha_max = 0`` disables the pass entirely and
    # ``alpha_min = alpha_max = 1`` is uniform full-strength smoothing.
    # Gaussian σ in km is converted to cells via ``kernel_km / cell_km``.
    edge_smoothing_apply_t0: bool
    edge_smoothing_apply_tfinal: bool
    edge_smoothing_kernel_km: float
    edge_smoothing_alpha_min: float
    edge_smoothing_alpha_max: float
    edge_smoothing_noise_wavelength_km: float
    edge_smoothing_noise_octaves: int
    edge_smoothing_noise_persistence: float
    # Plate-boundary boost: an EXTRA alpha contribution that decays
    # exponentially with distance from the nearest plate boundary.
    # ``α = clip(α_perlin + peak * exp(-d_km / falloff_km), 0, 1)``.
    # ``peak = 0`` reduces to the pure Perlin pass; ``peak = 1`` slams
    # alpha to its ceiling exactly at every plate suture.
    edge_smoothing_boundary_boost_peak: float
    edge_smoothing_boundary_falloff_km: float
