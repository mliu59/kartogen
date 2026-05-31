"""Connected-component culling + fragment redistribution."""

from __future__ import annotations

import numpy as np

from tectonic_sim.polygon_sim.topology import _torus_components
from tectonic_sim.polygon_sim.types import (
    PolygonPlate)


def _cull_disconnected(
    plates: list[PolygonPlate], spawn_rng, sim_config,
) -> tuple[int, int, int]:
    """Keep one component per plate (torus topology); preserve mass by
    either transferring or spawning every released component.

    **Mass preservation invariant**: a released component (one of the
    plate's sub-components that isn't the keep-component) MUST either
    transfer to another plate or spawn as a new plate. It never just
    disappears.

    Rules:
      - **Keep-component** per plate: ALWAYS the largest by cells.
        (Earlier this was biased toward continental content, but that
        produced the bug where a tiny continental sliver was kept and
        a much larger oceanic body was released. The mass-conservation
        path below ensures continents in a released component aren't
        lost — they spawn as their own plate or weld to a neighbour.)
      - **Released components** are clustered globally (torus-aware
        connected-component label of the union of all released cells)
        so a single break-off blob is one unit:
          * **Large** (≥ ``sim_config.fragment_spawn_threshold`` cells) → spawn
            as a new plate. Paint inherits cell-by-cell. Velocity =
            dominant-parent velocity rotated + scaled by random
            perturbation; angular velocity = parent ± small jitter.
            Determinism via ``spawn_rng``.
          * **Small with neighbour** → redistribute (transfer): the
            cluster is welded onto the neighbouring plate that owns
            the most adjacent cells. Paint preserved.
          * **Small with no neighbour** (orphan microplate) → spawn
            anyway. Preserves mass even for stragglers.

    Returns ``(total_released, redistributed_cells, spawned_plates)``.
    """
    plate_by_pid = {p.pid: p for p in plates if p.alive}
    alive_plates = [p for p in plates if p.alive and p.cell_mask.any()]
    if not alive_plates:
        return 0, 0, 0

    # ----- Phase A: identify keep-mask and released cells per plate.
    # We also remember each released cell's PARENT plate so a spawned
    # microplate can inherit velocity from the plate it broke off of.
    released_paint: list[tuple[int, int, int, float, float, int]] = []
    # tuples: (y, x, crust, age, thickness, parent_pid)
    pending_apply: list[tuple[PolygonPlate, np.ndarray]] = []
    # (plate, keep_mask)

    for p in alive_plates:
        lbl = _torus_components(p.cell_mask)
        ids = lbl[lbl > 0]
        if ids.size == 0:
            continue
        unique, counts = np.unique(ids, return_counts=True)
        if unique.size <= 1:
            continue
        # Always keep the LARGEST component. Continents in smaller
        # released components don't get lost — they spawn as new plates
        # or weld to neighbours via the redistribute path below.
        keep_id = unique[int(counts.argmax())]
        keep_mask = lbl == keep_id
        released = p.cell_mask & ~keep_mask
        if not released.any():
            continue
        # Record paint at released cells before clearing.
        iy, ix = np.where(released)
        for y, x in zip(iy.tolist(), ix.tolist()):
            released_paint.append((
                y, x,
                int(p.crust[y, x]),
                float(p.age[y, x]),
                float(p.thickness[y, x]),
                int(p.pid)))
        pending_apply.append((p, keep_mask))

    if not released_paint:
        return 0, 0, 0

    # ----- Phase B: apply keep-masks (clear paint at released cells).
    for p, keep_mask in pending_apply:
        released = p.cell_mask & ~keep_mask
        p.cell_mask = keep_mask
        p.crust = np.where(released, np.int8(0), p.crust).astype(np.int8)
        p.age = np.where(released, 0.0, p.age)
        p.thickness = np.where(released, 0.0, p.thickness)

    total_released = len(released_paint)

    # ----- Phase C: cluster the released cells into components.
    gy, gx = alive_plates[0].cell_mask.shape
    released_mask = np.zeros((gy, gx), dtype=bool)
    paint_by_cell: dict[tuple[int, int], tuple[int, float, float]] = {}
    parent_by_cell: dict[tuple[int, int], int] = {}
    for y, x, c, a, t, parent_pid in released_paint:
        released_mask[y, x] = True
        paint_by_cell[(y, x)] = (c, a, t)
        parent_by_cell[(y, x)] = parent_pid

    rel_lbl = _torus_components(released_mask)
    n_comp = int(rel_lbl.max())
    if n_comp == 0:
        return total_released, 0, 0

    # Build the post-Phase-B global owner map — votes read against this.
    owner_after_apply = np.full((gy, gx), -1, dtype=np.int64)
    for p in alive_plates:
        owner_after_apply[p.cell_mask] = p.pid

    next_pid = max(p.pid for p in plates) + 1
    n_redistributed = 0
    n_spawned = 0

    for comp_id in range(1, n_comp + 1):
        comp_mask = rel_lbl == comp_id
        iy, ix = np.where(comp_mask)
        n_cells = int(iy.size)
        if n_cells == 0:
            continue

        # ----- Branch 1: large component → spawn as new plate.
        if n_cells >= sim_config.fragment_spawn_threshold:
            _spawn_from_component(
                plates, plate_by_pid, comp_mask, iy, ix,
                paint_by_cell, parent_by_cell,
                next_pid, gy, gx, spawn_rng, sim_config)
            next_pid += 1
            n_spawned += 1
            n_redistributed += n_cells   # mass moved, just into a new pid
            continue

        # ----- Branch 2: small component → vote on neighbouring plates.
        votes: dict[int, int] = {}
        for y, x in zip(iy.tolist(), ix.tolist()):
            for ny, nx in (
                (y, (x + 1) % gx),
                (y, (x - 1) % gx),
                ((y + 1) % gy, x),
                ((y - 1) % gy, x)):
                if comp_mask[ny, nx]:
                    continue
                o = int(owner_after_apply[ny, nx])
                if o < 0:
                    continue
                votes[o] = votes.get(o, 0) + 1

        if votes:
            best_pid = sorted(
                votes.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0]
            target = plate_by_pid.get(best_pid)
            if target is not None:
                for y, x in zip(iy.tolist(), ix.tolist()):
                    c, a, t = paint_by_cell[(y, x)]
                    target.cell_mask[y, x] = True
                    target.crust[y, x] = np.int8(c)
                    target.age[y, x] = a
                    target.thickness[y, x] = t
                    n_redistributed += 1
                continue   # successful transfer; next component

        # ----- Branch 3: small orphan → spawn anyway (preserve mass).
        _spawn_from_component(
            plates, plate_by_pid, comp_mask, iy, ix,
            paint_by_cell, parent_by_cell,
            next_pid, gy, gx, spawn_rng)
        next_pid += 1
        n_spawned += 1
        n_redistributed += n_cells

    return total_released, n_redistributed, n_spawned


def _spawn_from_component(
    plates: list[PolygonPlate],
    plate_by_pid: dict[int, PolygonPlate],
    comp_mask: np.ndarray, iy: np.ndarray, ix: np.ndarray,
    paint_by_cell: dict[tuple[int, int], tuple[int, float, float]],
    parent_by_cell: dict[tuple[int, int], int],
    new_pid: int, gy: int, gx: int, spawn_rng, sim_config,
) -> None:
    """Materialise a new ``PolygonPlate`` from a released component.

    Paint is inherited cell-by-cell. Velocity = the dominant parent's
    velocity rotated by a random small angle and scaled by ±30%
    multiplicative jitter — the fragment broke off the parent, so its
    motion should be similar but not identical. Angular velocity =
    parent's plus a small random jitter.
    """
    # Find dominant parent (most cells came from this plate).
    parent_counts: dict[int, int] = {}
    for y, x in zip(iy.tolist(), ix.tolist()):
        pp = parent_by_cell[(int(y), int(x))]
        parent_counts[pp] = parent_counts.get(pp, 0) + 1
    dominant_pid = max(
        parent_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    parent = plate_by_pid.get(dominant_pid)
    if parent is None:
        # Parent vanished — use zero velocity baseline.
        parent_vx = 0.0
        parent_vy = 0.0
        parent_omega = 0.0
    else:
        parent_vx = float(parent.velocity_kmpy[0])
        parent_vy = float(parent.velocity_kmpy[1])
        parent_omega = float(parent.angular_velocity_rad_per_myr)

    # Build the new plate's grids — only the component's cells are True.
    new_mask = np.zeros((gy, gx), dtype=bool)
    new_crust = np.zeros((gy, gx), dtype=np.int8)
    new_age = np.zeros((gy, gx), dtype=np.float64)
    new_thick = np.zeros((gy, gx), dtype=np.float64)
    for y, x in zip(iy.tolist(), ix.tolist()):
        c, a, t = paint_by_cell[(int(y), int(x))]
        new_mask[y, x] = True
        new_crust[y, x] = np.int8(c)
        new_age[y, x] = a
        new_thick[y, x] = t

    # Velocity perturbation: rotate by ±45° and scale by ±30%.
    angle = float(spawn_rng.uniform(-np.pi * 0.25, np.pi * 0.25))
    scale = float(spawn_rng.uniform(0.7, 1.3))
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    new_vx = (parent_vx * cos_a - parent_vy * sin_a) * scale
    new_vy = (parent_vx * sin_a + parent_vy * cos_a) * scale
    new_omega = parent_omega + float(spawn_rng.uniform(
        -sim_config.init_angular_velocity_max_rad_per_myr * 0.5,
        sim_config.init_angular_velocity_max_rad_per_myr * 0.5))

    new_plate = PolygonPlate(
        pid=new_pid,
        velocity_kmpy=np.array([new_vx, new_vy], dtype=np.float64),
        accum=np.zeros(2, dtype=np.float64),
        cell_mask=new_mask,
        crust=new_crust,
        age=new_age,
        thickness=new_thick,
        polygon=None,   # rebuilt next tick
        alive=True,
        angular_velocity_rad_per_myr=new_omega)
    plates.append(new_plate)
    plate_by_pid[new_pid] = new_plate

