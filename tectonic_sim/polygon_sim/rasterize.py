"""Body-frame ↔ world-frame rasterisation bridge.

The polygon sim's primary state is **body-frame**: each plate carries
``body_mask`` / ``body_crust`` / ``body_age`` / ``body_thickness``
arrays in its own local frame, plus a continuous world-frame pose
(``position_km``, ``orientation_rad``). Body arrays don't move per
tick — only the pose does.

Per-tick modules (contention, fusion, momentum, accretion, aging,
hotspots, …) operate on **world-frame** stacked arrays. Two bridge
functions span the gap:

  - :func:`rasterise` — sample each plate's body arrays at every world
    cell through the plate's pose, populate the plate's cached
    world-frame view (``cell_mask`` / ``crust`` / ``age`` / ``thickness``).
    Also snapshots the just-rasterised state into a hidden baseline so
    :func:`derasterise` can detect changes.

  - :func:`derasterise` — diff the current world view against the
    baseline saved at the top of the tick. Only **changed** cells get
    propagated to the body frame (via NN inverse transform). Unchanged
    cells leave the body untouched, so the body↔world round-trip is
    mass-preserving for cells that no per-tick op modified.

Both directions use **NN** sampling. Bilinear was tried earlier but
the 0.5-threshold mask reconstruction eroded ~1 cell of plate boundary
per tick, compounding to catastrophic mass leak (60-70 % of continental
crust lost in 100 ticks). NN-in-both-directions is exact at integer
poses and produces only ≤1-cell discretisation noise at fractional
poses, which the diff-based propagation cleanly absorbs.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import PolygonPlate


def _world_to_body(
    pose_position_km: np.ndarray, pose_orientation_rad: float,
    world_x: np.ndarray, world_y: np.ndarray,
    domain: WorldRect,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the plate's world→body transform: untranslate then unrotate.

    **Toroidal-wrap-aware.** The raw delta ``(world − position_km)`` is
    wrapped to the toroidal shortest path BEFORE rotation. Without this,
    a plate near the seam whose world cells require wrap to reach it
    (e.g. plate at +800 km looking at a world cell at −800 km on a
    2000-km-wide torus) would compute ``dx = −1600`` instead of the
    correct ``+400``, and the subsequent rotation would land the cell
    in a completely wrong body location — effectively scrambling the
    plate's content into random body cells. The post-rotation modulo
    wrap absorbs the error only when the orientation aligns with the
    domain axes (mult of π/2); for any other rotation the cells
    mis-sample, which manifests as mid-sim plate decimation + fragment
    bursts whenever a sufficiently rotated plate drifts near the seam.
    """
    cos_a = float(np.cos(-pose_orientation_rad))
    sin_a = float(np.sin(-pose_orientation_rad))
    dx = world_x - float(pose_position_km[0])
    dy = world_y - float(pose_position_km[1])
    # Toroidal shortest path on the (centred) torus.
    dx, dy = domain.wrapped_delta_xy(dx, dy)
    bx = cos_a * dx - sin_a * dy
    by = sin_a * dx + cos_a * dy
    return bx, by


def rasterise(
    plates: list[PolygonPlate], domain: WorldRect,
    gy: int, gx: int, cell_km: float,
) -> None:
    """For every plate, sample its body arrays at each world cell and
    populate the plate's world-frame cached arrays. Save a baseline
    snapshot on the plate so :func:`derasterise` can compute the
    per-tick diff.
    """
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km

    yy, xx = np.indices((gy, gx))
    world_kx = (xx + 0.5) * cell_km - half_w
    world_ky = (yy + 0.5) * cell_km - half_h

    for p in plates:
        if not p.alive:
            p.cell_mask = np.zeros((gy, gx), dtype=bool)
            p.crust = np.zeros((gy, gx), dtype=np.int8)
            p.age = np.zeros((gy, gx), dtype=np.float64)
            p.thickness = np.zeros((gy, gx), dtype=np.float64)
            p._world_baseline_mask = p.cell_mask.copy()  # type: ignore[attr-defined]
            p._world_baseline_crust = p.crust.copy()  # type: ignore[attr-defined]
            p._world_baseline_age = p.age.copy()  # type: ignore[attr-defined]
            p._world_baseline_thickness = p.thickness.copy()  # type: ignore[attr-defined]
            continue

        bx, by = _world_to_body(
            p.position_km, p.orientation_rad, world_kx, world_ky, domain,
        )
        bx = (bx + half_w) % domain.width_km - half_w
        by = (by + half_h) % domain.height_km - half_h

        # NN body cell index. Cells indexed at centres (i + 0.5)*cell_km
        # → floor((coord + half) / cell_km) is the cell whose interval
        # contains the position == nearest cell centre.
        bi = np.floor((bx + half_w) / cell_km).astype(np.int64) % gx
        bj = np.floor((by + half_h) / cell_km).astype(np.int64) % gy

        new_mask = p.body_mask[bj, bi].copy()
        new_crust = p.body_crust[bj, bi].astype(np.int8)
        new_age = p.body_age[bj, bi].copy()
        new_thick = p.body_thickness[bj, bi].copy()

        p.cell_mask = new_mask
        p.crust = np.where(new_mask, new_crust, np.int8(0)).astype(np.int8)
        p.age = np.where(new_mask, new_age, 0.0)
        p.thickness = np.where(new_mask, new_thick, 0.0)

        # Snapshot: derasterise compares against this to find per-tick
        # mutations. Storing on the dataclass via _underscore attrs to
        # avoid expanding the public surface.
        p._world_baseline_mask = p.cell_mask.copy()  # type: ignore[attr-defined]
        p._world_baseline_crust = p.crust.copy()  # type: ignore[attr-defined]
        p._world_baseline_age = p.age.copy()  # type: ignore[attr-defined]
        p._world_baseline_thickness = p.thickness.copy()  # type: ignore[attr-defined]


def derasterise(
    plates: list[PolygonPlate], domain: WorldRect,
    gy: int, gx: int, cell_km: float,
) -> None:
    """Diff each plate's world view against the baseline saved at the
    top of the tick, propagate only the changes back to the body
    frame. Unchanged cells leave the body untouched.

    The diff has four kinds of cells per plate:

      - **Added** (was False, now True) — per-tick op gave this plate a
        new cell. Map back to the corresponding body cell (NN inverse)
        and write the current world paint into the body arrays.
      - **Removed** (was True, now False) — per-tick op cleared this
        cell from the plate. Clear the corresponding body cell.
      - **Paint-changed** (was True, still True, paint differs) — op
        modified the cell in place (aging, accretion, fold belt,
        hotspot, erosion). Write current world paint into body.
      - **Unchanged** — body stays put. This is the common case for
        most cells most ticks, which is why diff-based derasterise
        avoids the per-tick mass leak of overwrite-style derasterise.
    """
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km

    yy, xx = np.indices((gy, gx))
    world_kx = (xx + 0.5) * cell_km - half_w
    world_ky = (yy + 0.5) * cell_km - half_h

    for p in plates:
        if not p.alive:
            continue

        # Plates spawned mid-tick (culling / rifting) won't have a
        # baseline yet — their entire world view is "new" content that
        # should be written into the body wholesale. Use a zero
        # baseline so the diff classifies everything as "added".
        bm = getattr(p, "_world_baseline_mask", None)
        if bm is None:
            bm = np.zeros_like(p.cell_mask)
            bc = np.zeros_like(p.crust)
            ba = np.zeros_like(p.age)
            bt = np.zeros_like(p.thickness)
        else:
            bc = p._world_baseline_crust  # type: ignore[attr-defined]
            ba = p._world_baseline_age  # type: ignore[attr-defined]
            bt = p._world_baseline_thickness  # type: ignore[attr-defined]

        cm = p.cell_mask
        cc = p.crust
        ca = p.age
        ct = p.thickness

        added = cm & ~bm
        removed = ~cm & bm
        paint_changed = (
            cm & bm & (
                (cc != bc) | (ca != ba) | (ct != bt)
            )
        )

        # Combine all cells we need to write back: added + paint_changed
        write_cells = added | paint_changed
        clear_cells = removed

        # For each "write" world cell, find its NN body cell and stamp
        # the current paint there.
        if write_cells.any():
            ys, xs = np.where(write_cells)
            wk_x = (xs + 0.5) * cell_km - half_w
            wk_y = (ys + 0.5) * cell_km - half_h
            bx_w, by_w = _world_to_body(
                p.position_km, p.orientation_rad, wk_x, wk_y, domain,
            )
            bx_w = (bx_w + half_w) % domain.width_km - half_w
            by_w = (by_w + half_h) % domain.height_km - half_h
            bi = np.floor((bx_w + half_w) / cell_km).astype(np.int64) % gx
            bj = np.floor((by_w + half_h) / cell_km).astype(np.int64) % gy
            p.body_mask[bj, bi] = True
            p.body_crust[bj, bi] = cc[ys, xs]
            p.body_age[bj, bi] = ca[ys, xs]
            p.body_thickness[bj, bi] = ct[ys, xs]

        # For each "clear" world cell, find its NN body cell and clear.
        if clear_cells.any():
            ys, xs = np.where(clear_cells)
            wk_x = (xs + 0.5) * cell_km - half_w
            wk_y = (ys + 0.5) * cell_km - half_h
            bx_c, by_c = _world_to_body(
                p.position_km, p.orientation_rad, wk_x, wk_y, domain,
            )
            bx_c = (bx_c + half_w) % domain.width_km - half_w
            by_c = (by_c + half_h) % domain.height_km - half_h
            bi = np.floor((bx_c + half_w) / cell_km).astype(np.int64) % gx
            bj = np.floor((by_c + half_h) / cell_km).astype(np.int64) % gy
            p.body_mask[bj, bi] = False
            p.body_crust[bj, bi] = np.int8(0)
            p.body_age[bj, bi] = 0.0
            p.body_thickness[bj, bi] = 0.0


def init_body_from_world(plate: PolygonPlate) -> None:
    """One-shot initialisation: copy the plate's existing world-frame
    arrays into the body frame at the identity pose (position=0,
    orientation=0). Used in ``_initial_state`` after Voronoi seeding,
    when body and world coincide by construction.
    """
    plate.body_mask = plate.cell_mask.copy()
    plate.body_crust = plate.crust.copy()
    plate.body_age = plate.age.copy()
    plate.body_thickness = plate.thickness.copy()
