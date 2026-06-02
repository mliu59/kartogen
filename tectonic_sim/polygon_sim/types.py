"""Polygon-sim data types and deterministic RNG tags.

This module used to carry the full set of physics tunables as
module-level ``_UPPERCASE`` constants. After the SimConfig migration
they all live in ``tectonic_sim.SimConfig`` (loaded from
``config/tectonic_sim.toml``). Every per-tick physics function reads
its tunables via the ``sim_config`` argument that gets threaded
through the per-tick pipeline.

What stays here:
  - The data types: ``Hotspot``, ``AlphaComplex``, ``PolygonPlate``.
  - The deterministic RNG seed-XOR tags. These are NOT tunables — they
    are arbitrary magic words used to derive independent RNG streams
    for each phase (so toggling one mechanic doesn't reshuffle the
    others' random draws).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Deterministic per-phase RNG seed-XOR tags. Magic words used to derive
# independent ``np.random.Generator(PCG64(seed ^ TAG))`` streams. Each
# tag picks a distinct XOR-displaced seed so toggling one phase doesn't
# shift the others' random draws.
# ---------------------------------------------------------------------------

_RIFT_RNG_TAG: int = 0x21F7
_ROT_RNG_TAG: int = 0xA001
_VELOCITY_RNG_TAG: int = 0xB002
_SPAWN_RNG_TAG: int = 0xC003
_VORONOI_RNG_TAG: int = 0xE005
_ACCRETION_RNG_TAG: int = 0xD004
_HOTSPOT_RNG_TAG: int = 0xF006
_CONTINENTAL_RELIEF_RNG_TAG: int = 0x1007
_EDGE_SMOOTHING_RNG_TAG: int = 0x1008


# ---------------------------------------------------------------------------
# Data types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hotspot:
    """A mantle-frame volcanic hotspot (mantle plume).

    Position is fixed in the centred mantle-frame km coordinate system
    (0,0 at sim centre). Plates drift across it; eruptions stamp the
    cell currently above it.
    """
    position_xy_km: tuple[float, float]
    birth_tick: int
    lifespan_ticks: int

    def is_active(self, tick: int) -> bool:
        return self.birth_tick <= tick < self.birth_tick + self.lifespan_ticks


# (Delaunay triangulation in local frame, kept-triangle mask, ref point)
AlphaComplex = tuple[Delaunay, np.ndarray, np.ndarray]


@dataclass
class PolygonPlate:
    """Rigid-polygon plate state.

    The plate is modelled as **continuous polygons in km-space**: each
    plate has a body frame (centred at the plate's body origin, axis-
    aligned to the body) plus a world-frame pose (``position_km``,
    ``orientation_rad``). The body-frame paint arrays (``body_mask`` /
    ``body_crust`` / ``body_age`` / ``body_thickness``) are the canonical
    primary state. World-frame views (``cell_mask`` / ``crust`` /
    ``age`` / ``thickness``) are CACHED — regenerated each tick by
    rasterising body-frame state through the current pose.

    Kinematics are continuous: each tick ``position_km`` and
    ``orientation_rad`` get incremented by ``velocity_kmpy * dt`` and
    ``angular_velocity_rad_per_myr * dt`` respectively. No per-tick
    NN/bilinear resampling of the paint arrays — that previously caused
    rotation to be invisible because per-tick angles were sub-cell at
    typical plate sizes.

    Per-tick modules read the world-frame views as before. Changes they
    make to those views (loser-clears, accretion, hotspots, aging,
    erosion, …) are propagated back into the body frame by an inverse
    rasterisation pass at the end of each tick, so the body snapshot
    stays current.
    """
    pid: int
    velocity_kmpy: np.ndarray            # (2,) float64
    angular_velocity_rad_per_myr: float = 0.0

    # --- Continuous pose in world km-space ---
    # World-frame position of the body origin. **Snapped to an integer
    # multiple of cell_km each tick** — this is essential to make the
    # body↔world NN sampling round-trip exact. Non-integer-cell offsets
    # cause floor() to give different cell indices on the forward and
    # inverse passes (off-by-1 in some cells), which leaks mass and
    # produces fragment artefacts at the seams. Sub-cell translation
    # remainder accumulates in ``position_carry_km`` until it crosses
    # the half-cell threshold.
    position_km: np.ndarray = None  # type: ignore[assignment]
    # Sub-cell translation accumulator (km). Each tick, kinematics adds
    # ``velocity_kmpy * dt`` to the carry; when the carry magnitude
    # exceeds half a cell, the carry is flushed into ``position_km``
    # (which is then re-wrapped to the torus).
    position_carry_km: np.ndarray = None  # type: ignore[assignment]
    # World-frame orientation of the body axes (radians, counter-clockwise).
    orientation_rad: float = 0.0
    # Body-frame rotation pivot (body-km). The plate rotates about THIS
    # body point, which maps to ``position_km`` in the world. It is kept
    # on the plate's body-frame centroid (snapped to integer cells) by
    # the periodic recenter step in ``kinematics._recenter_pivots`` — so
    # a plate spins about its own centre of area rather than orbiting a
    # distant fixed origin, and the world→body wrap discontinuity sits at
    # the antipode of the centroid (far outside the plate) instead of
    # cutting through it. Maps: world = R(θ)·(body − pivot) + position;
    # body = R(−θ)·wrap(world − position) + pivot.
    body_pivot_km: np.ndarray = None  # type: ignore[assignment]

    # --- Body-frame canonical state (primary) ---
    # All four arrays share the same (gy, gx) shape as the world grid,
    # but they live in BODY coordinates. They DO NOT move per tick —
    # only the (position_km, orientation_rad) transform changes.
    body_mask: np.ndarray = None       # type: ignore[assignment]
    body_crust: np.ndarray = None      # type: ignore[assignment]
    body_age: np.ndarray = None        # type: ignore[assignment]
    body_thickness: np.ndarray = None  # type: ignore[assignment]

    # --- World-frame views (CACHED — regenerated each tick) ---
    # Per-tick modules read these as if they were the primary state.
    # They get rebuilt at the top of each tick by rasterise(), and
    # any per-tick mutations are flushed back to the body-frame state
    # by derasterise() at the end of the tick.
    cell_mask: np.ndarray = None    # type: ignore[assignment]
    crust: np.ndarray = None        # type: ignore[assignment]
    age: np.ndarray = None          # type: ignore[assignment]
    thickness: np.ndarray = None    # type: ignore[assignment]

    polygon: AlphaComplex | None = None  # derived
    alive: bool = True
