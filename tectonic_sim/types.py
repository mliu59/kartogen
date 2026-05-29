"""Data types for ``tectonic_sim``.

Everything is in continuous km space. No hex anywhere. Particles are
stored as parallel numpy arrays inside ``Snapshot`` and ``Frame`` for
cheap vectorised access; the rest of the data model is plain
``@dataclass(frozen=True)`` so user code can pattern-match and compare.

Crust type encoding: integer codes ``CRUST_CONTINENTAL = 0`` and
``CRUST_OCEANIC = 1`` rather than strings, so the per-particle field can
live in an ``int8`` array (smaller, vectorised comparisons). Helpers
convert to/from strings at the public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

# Type alias for "scalar or numpy array of floats" — used in WorldRect's
# wrap helpers which work on both per-particle vectors and bulk arrays.
FloatLike = Union[float, np.ndarray]


# Crust type encoding for the integer arrays.
CRUST_CONTINENTAL: int = 0
CRUST_OCEANIC: int = 1

_CRUST_TYPE_NAMES = ("continental", "oceanic")


# Collision-detection radius is *defined* as a multiple of the
# particle spacing: Bridson Poisson-disc guarantees a minimum
# pairwise distance of ``particle_spacing_km``, so the overlap
# radius must be strictly larger or no cross-plate pair ever
# fires. 1.5× catches the immediate-neighbour rank along each
# Voronoi boundary (one collision event per ~spacing of boundary
# length) without double-triggering deeper ranks.
OVERLAP_RADIUS_MULTIPLIER: float = 1.5


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

    The sim treats this as a hard bounding box: with ``boundary_mode =
    "open"`` (the only mode for now), particles that drift past these
    bounds are deleted at the next step.
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
    # These are used wherever the simulation needs to respect the wrap-
    # around domain boundary (``boundary_mode = "wrap"``). The maths is
    # the standard "shortest signed distance modulo period" form. When
    # ``boundary_mode = "open"`` is configured, callers should *not*
    # call these — the open-boundary cull at the kinematics step keeps
    # positions inside the rectangle, and distances are taken directly.

    def wrap_positions(self, positions_km: np.ndarray) -> np.ndarray:
        """Wrap an ``(N, 2)`` array of positions onto the toroidal domain.

        Coordinates are remapped to ``[-half_width, +half_width)`` ×
        ``[-half_height, +half_height)``. Returns a new array; the input
        is not mutated.
        """
        out = np.empty_like(positions_km)
        out[:, 0] = (
            (positions_km[:, 0] + self.half_width_km) % self.width_km
            - self.half_width_km
        )
        out[:, 1] = (
            (positions_km[:, 1] + self.half_height_km) % self.height_km
            - self.half_height_km
        )
        return out

    def wrapped_delta_xy(
        self, dx: FloatLike, dy: FloatLike,
    ) -> tuple[FloatLike, FloatLike]:
        """Return the toroidal shortest-path delta for one or many ``(dx, dy)``.

        For each component, the result lies in ``[-half_period, +half_period)``
        so its magnitude is the shortest distance around the torus.
        Scalars and numpy arrays both work — the formula is identical.
        """
        wx = (dx + self.half_width_km) % self.width_km - self.half_width_km
        wy = (dy + self.half_height_km) % self.height_km - self.half_height_km
        return wx, wy

    def wrapped_distance_km(
        self,
        a_xy_km: np.ndarray,
        b_xy_km: np.ndarray,
    ) -> np.ndarray:
        """Toroidal Euclidean distance between paired points.

        ``a_xy_km`` and ``b_xy_km`` are either ``(2,)`` arrays (one pair)
        or ``(N, 2)`` arrays (N parallel pairs). Returns a scalar or
        ``(N,)`` array accordingly.
        """
        diff = a_xy_km - b_xy_km
        if diff.ndim == 1:
            dx, dy = self.wrapped_delta_xy(diff[0], diff[1])
            return float(np.hypot(dx, dy))
        wx, wy = self.wrapped_delta_xy(diff[:, 0], diff[:, 1])
        return np.hypot(wx, wy)


@dataclass(frozen=True)
class SimConfig:
    """All physics knobs. No grid, no climate, no map_lat — physics only.

    Loaded by ``config_loader.load_sim_config`` from a TOML table. Every
    field is required (no defaults at construction time so missing-config
    bugs surface at load, not as silent zeros downstream).
    """

    # --- Plate population ---
    plate_count: int
    continental_fraction: float
    motion_speed_kmpy: float
    seed_radial_bias: float                       # 0 = uniform, >0 = centre, <0 = edge

    # --- Initial particle layout ---
    # Bridson Poisson-disc target spacing; particle density ~ 1/(π·s²/4).
    particle_spacing_km: float

    # --- Sim duration ---
    n_ticks: int
    dt_myr: float

    # --- Crust thicknesses ---
    continental_thickness_km: float               # initial continental column
    oceanic_thickness_km: float                   # initial oceanic column
    rift_thickness_km: float                      # thinned continental crust spawned
                                                  # at a continental-plate divergent gap

    # --- Half-space cooling (oceanic floor depth) ---
    ridge_depth_km: float                         # depth at age 0
    ridge_subsidence_rate: float                  # km per √Myr
    max_ocean_depth_km: float                     # cap on subsidence

    # --- Continental isostasy ---
    continental_reference_thickness_km: float
    continental_isostasy_factor: float
    sea_level_km: float                           # signed km — particles below = ocean

    # --- Collision ---
    # NOTE: ``overlap_radius_km`` is *not* a config field. It is hardcoded
    # to ``OVERLAP_RADIUS_MULTIPLIER × particle_spacing_km`` and exposed
    # as a derived property — see the property definition below.
    orogeny_uplift_per_overlap_km: float          # per cc-overlap tick
    folding_ratio: float                          # fraction of smaller column folded over
    folding_displacement_km: float                # how much the lower particle moves
    subduction_arc_uplift_km: float               # per oc/oo-overlap tick on the survivor
    # A continental particle whose thickness drops below this threshold is
    # considered "absorbed" by the over-riding plate. The depleted particle
    # is removed and its remaining thickness is added to the nearest
    # cross-plate continental neighbour. Geologically: the underthruster's
    # leading-edge crust gets fully incorporated into the over-rider over
    # tens of Myr (lighter end of the Wilson cycle).
    min_continental_thickness_km: float

    # --- Contact constraints + velocity damping ---
    # Number of PBD relaxation passes per tick. Each pass detects cross-
    # plate pairs within overlap_radius and pushes them apart by half the
    # overlap depth, enforcing geometric rigid-plate behaviour.
    contact_iterations: int
    # Velocity damping strength: per-tick fraction of plate velocity lost
    # when 100 % of the plate's particles are in contact. Energy goes
    # implicitly into thickening (orogeny). Typical: 0.03–0.10.
    velocity_damping_strength: float
    # Intra-plate spacing constraint: a *second* PBD pass each iteration
    # pushes apart same-plate particles closer than
    # ``intra_plate_min_distance_factor × particle_spacing_km``. Models
    # crust incompressibility — without it, the cross-plate constraint
    # shoves same-plate neighbours into stripes (force-chain artefact).
    # 0 disables. Typical: 0.5 (fires only on severe collapse, well below
    # the Bridson Poisson-disc invariant of 1.0).
    intra_plate_min_distance_factor: float

    # --- Erosion ---
    erosion_period: int                           # apply every N ticks; 0 disables
    erosion_strength: float                       # blend fraction toward neighbour mean

    # --- Boundaries ---
    boundary_mode: str                            # "open" only for now

    # --- Snapshots ---
    snapshot_period_ticks: int                    # 0 disables Frame capture

    @property
    def overlap_radius_km(self) -> float:
        """Cross-plate collision-detection radius, derived from particle spacing.

        Always ``OVERLAP_RADIUS_MULTIPLIER × particle_spacing_km``. Not
        a configurable field because Bridson Poisson-disc guarantees a
        minimum particle separation of ``particle_spacing_km``; any
        ``overlap_radius`` below that would detect zero pairs by
        construction, and any value above ~2× would multi-rank
        collisions. The multiplier sits at 1.5 for both reasons.
        """
        return OVERLAP_RADIUS_MULTIPLIER * self.particle_spacing_km

    @property
    def intra_plate_min_distance_km(self) -> float:
        """Minimum allowed separation between same-plate particles.

        Derived as ``intra_plate_min_distance_factor × particle_spacing_km``
        so the constraint auto-scales with the chosen particle resolution.
        Returns 0 when the factor is 0 (constraint disabled).
        """
        return self.intra_plate_min_distance_factor * self.particle_spacing_km


@dataclass(frozen=True)
class Plate:
    """One plate's identity + bulk kinematics.

    The plate itself owns no particles directly; per-particle data lives
    in the ``Snapshot`` / ``Frame`` arrays, keyed by ``plate_id``. This
    keeps subduction (removing particles) and divergent fill (adding
    particles) cheap and vectorisable.
    """

    id: int
    type: str                                     # initial type — "continental" / "oceanic"
    seed_position_km: tuple[float, float]
    velocity_kmpy: tuple[float, float]


@dataclass(frozen=True)
class Frame:
    """One captured slice of particle state mid-sim, used for animations.

    Frames carry only what the renderer needs (positions, plate ids,
    crust types) — not the full thickness/age arrays — so a long history
    stays small in memory and on disk.
    """

    tick: int
    time_myr: float
    plate_centers_km: np.ndarray                  # (P, 2) float64
    particle_position_km: np.ndarray              # (N_t, 2) float64
    particle_plate_id: np.ndarray                 # (N_t,) int32
    particle_crust_type: np.ndarray               # (N_t,) int8


@dataclass(frozen=True)
class Snapshot:
    """Final output of one ``simulate()`` call.

    Particle arrays are parallel; index i in any one of them refers to
    the same particle in all the others. The arrays are ``np.ndarray``
    rather than tuple-of-dataclass for ~100× faster sampling.
    """

    domain: WorldRect
    plates: tuple[Plate, ...]

    # Particle arrays. Shape: (N,)
    particle_position_km: np.ndarray              # (N, 2) float64
    particle_plate_id: np.ndarray                 # (N,) int32
    particle_thickness_km: np.ndarray             # (N,) float64
    particle_age_myr: np.ndarray                  # (N,) float64
    particle_crust_type: np.ndarray               # (N,) int8

    # Captured drift history (empty tuple if snapshot_period_ticks == 0).
    frames: tuple[Frame, ...]

    final_tick: int
    final_time_myr: float

    @property
    def particle_count(self) -> int:
        return int(self.particle_position_km.shape[0])
