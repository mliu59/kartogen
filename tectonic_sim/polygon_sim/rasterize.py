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

Sampling is **NN for mask + crust** (categorical) and **mask-weighted
bilinear for age + thickness** (the continuous paint). NN on the mask
is exact at integer poses and produces only ≤1-cell discretisation
noise at fractional poses, which the diff-based propagation cleanly
absorbs; reconstructing the *mask* from bilinear (a 0.5 threshold)
was tried earlier and eroded ~1 cell of boundary per tick, so the mask
stays NN. The bilinear pass on age/thickness only smooths paint values
inside the already-NN mask, avoiding the diagonal moiré that NN paint
produces under rotation.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import PolygonPlate


def _world_to_body(
    pose_position_km: np.ndarray, pose_orientation_rad: float,
    body_pivot_km: np.ndarray,
    world_x: np.ndarray, world_y: np.ndarray,
    domain: WorldRect,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the plate's world→body transform: untranslate, unrotate,
    then re-add the body pivot.

    The plate rotates about ``body_pivot_km`` (a body-frame point kept on
    the plate's centroid), which maps to ``position_km`` in the world.
    So the inverse is ``body = R(−θ)·wrap(world − position) + pivot``.

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
    bx = cos_a * dx - sin_a * dy + float(body_pivot_km[0])
    by = sin_a * dx + cos_a * dy + float(body_pivot_km[1])
    return bx, by


def rasterise(
    plates: list[PolygonPlate], domain: WorldRect,
    gy: int, gx: int, cell_km: float,
) -> None:
    """For every plate, sample its body arrays into the world view.

    Two-pass sampling — needed because **NN inverse-sampling alone
    misses body cells under rotation**. Concretely: at small rotation
    angles, the inverse from a world cell (j_w, i_w) may map to a body
    cell shifted by 1 from the body cell that forward-projects to
    (j_w, i_w). The missed body cells become "ghost" content — present
    in body, invisible in world. Trail-fill then assigns those world
    cells to neighbouring plates, producing the long perpendicular
    straight-line artefacts that align with the plate's projected
    body axes.

    Pass 1: inverse-NN sample. For each world cell, transform to body
    via the inverse pose, NN-sample the body arrays. Handles the bulk
    of the plate's content.

    Pass 2: forward-scatter fill. For every TRUE body cell, project to
    its target world cell via the forward pose. If that world cell is
    not already True from pass 1, mark it True and copy paint. This
    guarantees every body cell appears at *some* world cell, which
    closes the off-by-1 gap without disturbing the cells inverse
    already got right.

    Combined cost is ~2× pass-1, still cheap.
    """
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km

    yy, xx = np.indices((gy, gx))
    world_kx = (xx + 0.5) * cell_km - half_w
    world_ky = (yy + 0.5) * cell_km - half_h

    # Body-cell centres in body km coords, computed once.
    bj_all, bi_all = np.indices((gy, gx))
    body_kx = (bi_all + 0.5) * cell_km - half_w
    body_ky = (bj_all + 0.5) * cell_km - half_h

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

        # ----- Pass 1: inverse-NN sample (mask + crust) + mask-weighted
        # bilinear (age + thickness).
        bx, by = _world_to_body(
            p.position_km, p.orientation_rad, p.body_pivot_km,
            world_kx, world_ky, domain,
        )
        bx = (bx + half_w) % domain.width_km - half_w
        by = (by + half_h) % domain.height_km - half_h

        # NN cell index for mask + crust (categorical / boolean).
        bi = np.floor((bx + half_w) / cell_km).astype(np.int64) % gx
        bj = np.floor((by + half_h) / cell_km).astype(np.int64) % gy

        new_mask = p.body_mask[bj, bi].copy()
        new_crust = p.body_crust[bj, bi].astype(np.int8)

        # Bilinear sample for age + thickness. NN sampling produces
        # visible diagonal moiré stripes in the rendered paint under
        # rotation because adjacent body cells (with slightly different
        # age/thickness from per-cell aging history) project to a
        # tilted grid in world. Bilinear blends across the 4 body-cell
        # neighbours of the inverse-projected position, smoothing the
        # stripes without affecting mask boundaries.
        #
        # Mask-weighted: only count contributions from body cells that
        # are True in the body mask, so cells just outside the plate
        # don't bleed zero values into the plate's interior paint.
        bi_f = (bx + half_w) / cell_km - 0.5
        bj_f = (by + half_h) / cell_km - 0.5
        i0 = np.floor(bi_f).astype(np.int64) % gx
        i1 = (i0 + 1) % gx
        j0 = np.floor(bj_f).astype(np.int64) % gy
        j1 = (j0 + 1) % gy
        fx = bi_f - np.floor(bi_f)
        fy = bj_f - np.floor(bj_f)
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        m00 = p.body_mask[j0, i0].astype(np.float64)
        m10 = p.body_mask[j0, i1].astype(np.float64)
        m01 = p.body_mask[j1, i0].astype(np.float64)
        m11 = p.body_mask[j1, i1].astype(np.float64)
        w00m = w00 * m00
        w10m = w10 * m10
        w01m = w01 * m01
        w11m = w11 * m11
        W = w00m + w10m + w01m + w11m
        denom = np.maximum(W, 1e-12)
        new_age = (
            p.body_age[j0, i0] * w00m + p.body_age[j0, i1] * w10m
            + p.body_age[j1, i0] * w01m + p.body_age[j1, i1] * w11m
        ) / denom
        new_thick = (
            p.body_thickness[j0, i0] * w00m + p.body_thickness[j0, i1] * w10m
            + p.body_thickness[j1, i0] * w01m + p.body_thickness[j1, i1] * w11m
        ) / denom
        # Where mask is True (per NN) but no in-mask source for bilinear
        # (shouldn't happen normally — defensive fallback): use NN.
        no_blend = (W <= 0.0) & new_mask
        if no_blend.any():
            new_age[no_blend] = p.body_age[bj[no_blend], bi[no_blend]]
            new_thick[no_blend] = p.body_thickness[bj[no_blend], bi[no_blend]]

        # Track the body-cell SOURCE for each True world cell so that
        # derasterise can flush per-tick clears back to the exact body
        # cell that produced the world cell — closing the inverse-NN
        # off-by-1 in the world→body direction.
        source_bj = np.where(new_mask, bj, np.int64(-1))
        source_bi = np.where(new_mask, bi, np.int64(-1))

        # ----- Pass 2: forward-scatter to fill cells the inverse missed.
        if p.body_mask.any():
            src_bj, src_bi = np.where(p.body_mask)
            # Offset body coords by the rotation pivot before rotating —
            # the plate rotates about body_pivot_km (its centroid), which
            # maps to position_km in the world. **Wrap the (body − pivot)
            # offset toroidally** before rotating: the body frame is a
            # torus, so a cell on the far side of the body seam from the
            # pivot is actually CLOSE to it across the wrap. Without the
            # wrap the euclidean offset is ~a full domain, and rotating
            # it flings that half of the plate across the world. The
            # rotation must act on the toroidal-shortest body→pivot
            # vector (the same convention the inverse map uses on the
            # world side).
            src_bx = body_kx[src_bj, src_bi] - float(p.body_pivot_km[0])
            src_by = body_ky[src_bj, src_bi] - float(p.body_pivot_km[1])
            src_bx, src_by = domain.wrapped_delta_xy(src_bx, src_by)
            cos_a = float(np.cos(p.orientation_rad))
            sin_a = float(np.sin(p.orientation_rad))
            wx = cos_a * src_bx - sin_a * src_by + float(p.position_km[0])
            wy = sin_a * src_bx + cos_a * src_by + float(p.position_km[1])
            wx = (wx + half_w) % domain.width_km - half_w
            wy = (wy + half_h) % domain.height_km - half_h
            wi = np.floor((wx + half_w) / cell_km).astype(np.int64) % gx
            wj = np.floor((wy + half_h) / cell_km).astype(np.int64) % gy

            # Only fill cells the inverse didn't already mark True.
            missing = ~new_mask[wj, wi]
            if missing.any():
                add_j = wj[missing]
                add_i = wi[missing]
                add_src_j = src_bj[missing]
                add_src_i = src_bi[missing]
                new_mask[add_j, add_i] = True
                new_crust[add_j, add_i] = p.body_crust[add_src_j, add_src_i]
                new_age[add_j, add_i] = p.body_age[add_src_j, add_src_i]
                new_thick[add_j, add_i] = p.body_thickness[add_src_j, add_src_i]
                # Record forward-scatter sources.
                source_bj[add_j, add_i] = add_src_j
                source_bi[add_j, add_i] = add_src_i

        p.cell_mask = new_mask
        p.crust = np.where(new_mask, new_crust, np.int8(0)).astype(np.int8)
        p.age = np.where(new_mask, new_age, 0.0)
        p.thickness = np.where(new_mask, new_thick, 0.0)

        p._world_baseline_mask = p.cell_mask.copy()  # type: ignore[attr-defined]
        p._world_baseline_crust = p.crust.copy()  # type: ignore[attr-defined]
        p._world_baseline_age = p.age.copy()  # type: ignore[attr-defined]
        p._world_baseline_thickness = p.thickness.copy()  # type: ignore[attr-defined]
        p._source_bj = source_bj  # type: ignore[attr-defined]
        p._source_bi = source_bi  # type: ignore[attr-defined]


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

        clear_cells = removed
        source_bj = getattr(p, "_source_bj", None)
        source_bi = getattr(p, "_source_bi", None)

        # PAINT_CHANGED writes: cell was True in baseline AND current,
        # only paint differs. The source mapping is valid (records which
        # body cell produced this world cell). Use it to write the
        # changed paint back to the EXACT body cell — avoids the off-
        # by-1 NN inverse that would otherwise write to a neighbour
        # under rotation, slowly eroding the plate along a body axis.
        if paint_changed.any() and source_bj is not None and source_bi is not None:
            ys, xs = np.where(paint_changed)
            pcj = source_bj[ys, xs]
            pci = source_bi[ys, xs]
            valid = pcj >= 0
            if valid.any():
                vj = pcj[valid]
                vi = pci[valid]
                vys = ys[valid]
                vxs = xs[valid]
                p.body_crust[vj, vi] = cc[vys, vxs]
                p.body_age[vj, vi] = ca[vys, vxs]
                p.body_thickness[vj, vi] = ct[vys, vxs]
                # body_mask already True (paint_changed is in the
                # mask=True ∩ mask=True intersection); don't need to
                # set it again.

        # ADDED writes: cell was False before, now True. No source
        # mapping exists yet (it was False). Use NN inverse to pick the
        # body cell to add. The NN-inverse can be off by 1 under
        # rotation but for genuinely new cells this only shifts the
        # added body cell by 1 — bounded error that doesn't compound.
        if added.any():
            ys, xs = np.where(added)
            wk_x = (xs + 0.5) * cell_km - half_w
            wk_y = (ys + 0.5) * cell_km - half_h
            bx_w, by_w = _world_to_body(
                p.position_km, p.orientation_rad, p.body_pivot_km,
                wk_x, wk_y, domain,
            )
            bx_w = (bx_w + half_w) % domain.width_km - half_w
            by_w = (by_w + half_h) % domain.height_km - half_h
            bi = np.floor((bx_w + half_w) / cell_km).astype(np.int64) % gx
            bj = np.floor((by_w + half_h) / cell_km).astype(np.int64) % gy
            p.body_mask[bj, bi] = True
            p.body_crust[bj, bi] = cc[ys, xs]
            p.body_age[bj, bi] = ca[ys, xs]
            p.body_thickness[bj, bi] = ct[ys, xs]

        # For each "clear" world cell, use the **stored source mapping**
        # from rasterise's baseline to clear the exact body cell that
        # produced this world cell. This closes the inverse-NN off-by-1
        # in the world→body direction — without it, derasterise clears
        # neighbours of the right body cell, slowly eroding the plate
        # along a body axis and producing the long straight-line
        # boundary artefacts.
        if clear_cells.any():
            ys, xs = np.where(clear_cells)
            source_bj = getattr(p, "_source_bj", None)
            source_bi = getattr(p, "_source_bi", None)
            if source_bj is not None and source_bi is not None:
                clear_bj = source_bj[ys, xs]
                clear_bi = source_bi[ys, xs]
                valid = clear_bj >= 0
                if valid.any():
                    cj = clear_bj[valid]
                    ci = clear_bi[valid]
                    p.body_mask[cj, ci] = False
                    p.body_crust[cj, ci] = np.int8(0)
                    p.body_age[cj, ci] = 0.0
                    p.body_thickness[cj, ci] = 0.0
            else:
                # Fallback to NN inverse if no source mapping (newly
                # spawned plate before its first rasterise).
                wk_x = (xs + 0.5) * cell_km - half_w
                wk_y = (ys + 0.5) * cell_km - half_h
                bx_c, by_c = _world_to_body(
                    p.position_km, p.orientation_rad, p.body_pivot_km,
                    wk_x, wk_y, domain,
                )
                bx_c = (bx_c + half_w) % domain.width_km - half_w
                by_c = (by_c + half_h) % domain.height_km - half_h
                bi = np.floor((bx_c + half_w) / cell_km).astype(np.int64) % gx
                bj = np.floor((by_c + half_h) / cell_km).astype(np.int64) % gy
                p.body_mask[bj, bi] = False
                p.body_crust[bj, bi] = np.int8(0)
                p.body_age[bj, bi] = 0.0
                p.body_thickness[bj, bi] = 0.0
