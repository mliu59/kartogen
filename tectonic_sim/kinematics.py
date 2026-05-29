"""Per-tick particle kinematics: drift + boundary handling.

Operations:

  - ``drift_positions(...)``        — pure translation by ``v·dt``.
  - ``wrap_positions(...)``         — toroidal wrap (``boundary_mode = "wrap"``).
  - ``cull_outside_domain(...)``    — open-boundary deletion (``boundary_mode = "open"``).
  - ``step_drift_and_apply_boundary(...)`` — convenience wrapper that
    drifts one tick and then applies the configured boundary handling.

Conventions:

  - Velocities are in km/Myr, stored on ``Plate``.
  - All arrays are parallel: index ``i`` in any of ``positions_km``,
    ``plate_id``, ``crust_type``, ``thickness_km``, ``age_myr`` refers
    to the same particle.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tectonic_sim.types import Plate, WorldRect


# -----------------------------------------------------------------------------
# Drift
# -----------------------------------------------------------------------------

def drift_positions(
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    plates: Sequence[Plate],
    dt_myr: float,
) -> np.ndarray:
    """Translate each particle by its owning plate's velocity × ``dt_myr``.

    Vectorised: builds a ``(P, 2)`` velocity table indexed by plate id and
    fancy-indexes per particle. Returns a *new* ``(N, 2)`` float64 array;
    the input is not mutated.

    Empty input is safe — an empty positions array yields an empty result.
    """
    if positions_km.shape[0] == 0:
        return positions_km.copy()

    # Velocity table. Plate ids may not be densely packed (e.g. after
    # subductions), so allocate room for max(plate_id) + 1, indexed
    # straight from the plate objects.
    max_id = max(p.id for p in plates)
    plate_vel = np.zeros((max_id + 1, 2), dtype=np.float64)
    for plate in plates:
        plate_vel[plate.id] = plate.velocity_kmpy

    per_particle_vel = plate_vel[plate_id]
    return positions_km + per_particle_vel * dt_myr


# -----------------------------------------------------------------------------
# Wrap (toroidal boundary)
# -----------------------------------------------------------------------------

def wrap_positions(
    domain: WorldRect,
    positions_km: np.ndarray,
) -> np.ndarray:
    """Wrap particle positions onto the toroidal domain.

    Thin convenience wrapper around ``WorldRect.wrap_positions`` so the
    kinematics-API is symmetric: ``drift_positions`` + ``wrap_positions``
    is the per-tick path under ``boundary_mode = "wrap"``.
    """
    if positions_km.shape[0] == 0:
        return positions_km.copy()
    return domain.wrap_positions(positions_km)


# -----------------------------------------------------------------------------
# Open-boundary cull
# -----------------------------------------------------------------------------

def cull_outside_domain(
    domain: WorldRect,
    positions_km: np.ndarray,
    *parallel_arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Filter parallel particle arrays to keep only those inside the rectangle.

    Returns a tuple ``(positions, *parallel_arrays)`` with each array's
    rows indexed by the same boolean mask, so they stay parallel. The
    bounds test is inclusive at the rectangle edge — a particle exactly
    on the boundary is kept.

    With ``boundary_mode = "open"`` this is the entire boundary handling:
    no reflection, no wrap, just deletion. Particles deleted here are
    gone for good; replacement (if any) is handled by the divergent-fill
    step.
    """
    if positions_km.shape[0] == 0:
        return (positions_km, *parallel_arrays)

    hw = domain.half_width_km
    hh = domain.half_height_km
    mask = (
        (positions_km[:, 0] >= -hw)
        & (positions_km[:, 0] <= hw)
        & (positions_km[:, 1] >= -hh)
        & (positions_km[:, 1] <= hh)
    )
    if mask.all():
        return (positions_km, *parallel_arrays)
    return (positions_km[mask], *(arr[mask] for arr in parallel_arrays))


# -----------------------------------------------------------------------------
# Per-tick step convenience
# -----------------------------------------------------------------------------

def step_drift_and_apply_boundary(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    age_myr: np.ndarray,
    plates: Sequence[Plate],
    dt_myr: float,
    *,
    boundary_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drift positions one tick, then apply the configured boundary handling.

    Returns ``(positions, plate_id, crust_type, thickness, age)`` — all
    parallel. ``boundary_mode`` controls what happens to particles whose
    drift carries them past the rectangle:

      - ``"wrap"`` — wrap modulo the domain (torus). No particles lost.
      - ``"open"`` — delete particles outside the rectangle (open boundary).
    """
    moved = drift_positions(positions_km, plate_id, plates, dt_myr)
    if boundary_mode == "wrap":
        return (
            wrap_positions(domain, moved),
            plate_id, crust_type, thickness_km, age_myr,
        )
    if boundary_mode == "open":
        return cull_outside_domain(  # type: ignore[return-value]
            domain, moved, plate_id, crust_type, thickness_km, age_myr,
        )
    raise ValueError(f"unknown boundary_mode {boundary_mode!r}")


# Legacy alias — older call sites used this name when "open" was the only
# mode. Equivalent to ``step_drift_and_apply_boundary(..., boundary_mode="open")``.
def step_drift_and_cull(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    age_myr: np.ndarray,
    plates: Sequence[Plate],
    dt_myr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Open-boundary specialisation of ``step_drift_and_apply_boundary``."""
    return step_drift_and_apply_boundary(
        domain, positions_km, plate_id, crust_type, thickness_km, age_myr,
        plates, dt_myr, boundary_mode="open",
    )
