"""Probabilistic plate rifting (splits)."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import (
    PolygonPlate)




def _rift_plate(
    plates: list[PolygonPlate], domain: WorldRect,
    divergence_kmpy: float, cell_km: float, gy: int, gx: int, rng,
    sim_config,
) -> bool:
    """Split an area-weighted alive plate by a random line. The two
    halves get diverging velocities; each half inherits the parent's
    per-cell paint where it owned the cell."""
    alive = [p for p in plates if p.alive]
    if not alive:
        return False
    counts = np.array(
        [int(p.cell_mask.sum()) for p in alive], dtype=np.int64)
    eligible = counts >= sim_config.rift_min_plate_cells
    if not eligible.any():
        return False
    weights = (counts * eligible).astype(np.float64)
    probs = weights / weights.sum()
    parent = alive[int(rng.choice(len(alive), p=probs))]

    iy, ix = np.where(parent.cell_mask)
    k = int(rng.choice(len(iy)))
    oy, ox = iy[k], ix[k]
    theta = float(rng.uniform(0.0, np.pi))
    nx, ny = np.cos(theta), np.sin(theta)

    dx = (ix.astype(np.float64) - ox) * cell_km
    dy = (iy.astype(np.float64) - oy) * cell_km
    dx, dy = domain.wrapped_delta_xy(dx, dy)
    side = dx * nx + dy * ny
    new_side = side < 0.0
    if new_side.all() or (~new_side).all():
        return False

    new_pid = max(p.pid for p in plates) + 1
    new_mask = np.zeros_like(parent.cell_mask)
    new_mask[iy[new_side], ix[new_side]] = True
    # Parent loses those cells; new plate gains them.
    parent_mask = parent.cell_mask & ~new_mask

    # Build the child plate with the parent's paint on its side.
    # Child plate inherits a perturbed angular velocity: roughly the
    # parent's, with a small jitter so the two halves rotate slightly
    # differently. Jitter sign-flipped relative to parent so the two
    # halves drift apart in rotation as well as translation.
    child_omega = (
        -parent.angular_velocity_rad_per_myr
        + float(rng.uniform(-sim_config.init_angular_velocity_max_rad_per_myr,
                            sim_config.init_angular_velocity_max_rad_per_myr) * 0.5)
    )
    child_mask_arr = new_mask
    child_crust_arr = np.where(new_mask, parent.crust, np.int8(0)).astype(np.int8)
    child_age_arr = np.where(new_mask, parent.age, 0.0)
    child_thick_arr = np.where(new_mask, parent.thickness, 0.0)
    new_plate = PolygonPlate(
        pid=new_pid,
        velocity_kmpy=np.array([
            parent.velocity_kmpy[0] - divergence_kmpy * nx,
            parent.velocity_kmpy[1] - divergence_kmpy * ny,
        ], dtype=np.float64),
        angular_velocity_rad_per_myr=child_omega,
        # The child inherits the parent's current world-frame pose: its
        # body frame is the parent's body frame at this instant (we're
        # snapshotting world arrays as body arrays). Future ticks
        # advance the child's pose independently.
        position_km=parent.position_km.copy(),
        orientation_rad=float(parent.orientation_rad),
        body_mask=child_mask_arr.copy(),
        body_crust=child_crust_arr.copy(),
        body_age=child_age_arr.copy(),
        body_thickness=child_thick_arr.copy(),
        cell_mask=child_mask_arr,
        crust=child_crust_arr,
        age=child_age_arr,
        thickness=child_thick_arr,
        polygon=None,
        alive=True)
    plates.append(new_plate)

    # Mutate parent in place — world view first, body will catch up at
    # the next derasterise.
    parent.cell_mask = parent_mask
    parent.crust = np.where(parent_mask, parent.crust, np.int8(0)).astype(np.int8)
    parent.age = np.where(parent_mask, parent.age, 0.0)
    parent.thickness = np.where(parent_mask, parent.thickness, 0.0)
    parent.velocity_kmpy = np.array([
        parent.velocity_kmpy[0] + divergence_kmpy * nx,
        parent.velocity_kmpy[1] + divergence_kmpy * ny,
    ], dtype=np.float64)
    return True


