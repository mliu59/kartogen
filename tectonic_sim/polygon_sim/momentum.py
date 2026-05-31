"""Per-pair inelastic momentum exchange + plate centroid."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import (
    PolygonPlate)


def _plate_centroid_km(
    plate: PolygonPlate, domain: WorldRect, gy: int, gx: int, cell_km: float) -> tuple[float, float] | None:
    """Wrap-aware circular-mean centroid of the plate's owned cells.

    Picks the first owned cell as reference, takes wrapped deltas to
    every other owned cell, averages, and re-wraps. The result is the
    geometric centroid of the cell-mask on the torus.
    """
    ys, xs = np.where(plate.cell_mask)
    if ys.size == 0:
        return None
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km
    c_kx = (xs + 0.5) * cell_km - half_w
    c_ky = (ys + 0.5) * cell_km - half_h
    ref_x, ref_y = float(c_kx[0]), float(c_ky[0])
    dx, dy = domain.wrapped_delta_xy(c_kx - ref_x, c_ky - ref_y)
    cent_x = ref_x + float(dx.mean())
    cent_y = ref_y + float(dy.mean())
    cent_x = ((cent_x + domain.half_width_km) % domain.width_km
              ) - domain.half_width_km
    cent_y = ((cent_y + domain.half_height_km) % domain.height_km
              ) - domain.half_height_km
    return cent_x, cent_y


def _apply_momentum_exchange(
    plates: list[PolygonPlate], domain: WorldRect,
    gy: int, gx: int, cell_km: float,
    sim_config,
) -> int:
    """Pairwise inelastic momentum exchange between contesting plates.

    For each pair (A, B) of alive plates whose ``cell_mask``s overlap
    (i.e. they're in active contention), apply an impulse along the
    centroid-to-centroid contact normal. Mass-weighted: smaller plates
    move toward the COM normal velocity more than larger plates do.

    Restitution ``e``:
      - ``e = 0``: perfectly inelastic, normal velocities equalise
        exactly at full contact (plates lock).
      - ``e = 1``: perfectly elastic, plates bounce off with reversed
        normal velocities.
      Tectonic plates are dominantly inelastic; ``e = 0.1`` is the
      default (almost-lock with slight spring-back).

    Impulse is scaled by ``contact_fraction = overlap_cells /
    min(mass_A, mass_B)`` so glancing brushes do little, full
    embedding fully locks the pair.

    Only the *normal* component of velocity is exchanged. Tangential
    components are preserved (rely on velocity damping for tangential
    friction).

    Returns the number of pair-exchanges applied this call.
    """
    alive = [p for p in plates if p.alive and p.cell_mask.any()]
    n = len(alive)
    if n < 2:
        return 0
    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contended = masks.sum(axis=0) > 1
    if not contended.any():
        return 0

    # Count contact cells per pair by walking contested cells.
    pair_counts: dict[tuple[int, int], int] = {}
    iy, ix = np.where(contended)
    for y, x in zip(iy.tolist(), ix.tolist()):
        cell_idxs = [k for k in range(n) if masks[k, y, x]]
        for ii in range(len(cell_idxs)):
            for jj in range(ii + 1, len(cell_idxs)):
                key = (cell_idxs[ii], cell_idxs[jj])
                pair_counts[key] = pair_counts.get(key, 0) + 1
    if not pair_counts:
        return 0

    # Centroids (wrap-aware) — one per alive plate, cached for the loop.
    centroids: list[tuple[float, float] | None] = [
        _plate_centroid_km(p, domain, gy, gx, cell_km) for p in alive
    ]
    masses = np.array(
        [int(p.cell_mask.sum()) for p in alive], dtype=np.float64)

    n_exchanges = 0
    for (a_idx, b_idx), n_overlap in pair_counts.items():
        a = alive[a_idx]
        b = alive[b_idx]
        m_a = float(masses[a_idx])
        m_b = float(masses[b_idx])
        if m_a <= 0 or m_b <= 0:
            continue
        ca = centroids[a_idx]
        cb = centroids[b_idx]
        if ca is None or cb is None:
            continue
        # Wrap-aware normal from A's centroid to B's centroid.
        dx, dy = domain.wrapped_delta_xy(
            np.array([cb[0] - ca[0]]), np.array([cb[1] - ca[1]]))
        dx, dy = float(dx[0]), float(dy[0])
        norm_mag = (dx * dx + dy * dy) ** 0.5
        if norm_mag < 1e-6:
            continue   # centroids coincide (rare); no defined normal.
        nx, ny = dx / norm_mag, dy / norm_mag
        # Relative normal velocity. v_rel_n > 0 ⇔ A approaching B.
        v_rel_x = float(a.velocity_kmpy[0] - b.velocity_kmpy[0])
        v_rel_y = float(a.velocity_kmpy[1] - b.velocity_kmpy[1])
        v_rel_n = v_rel_x * nx + v_rel_y * ny
        if v_rel_n <= 0:
            continue   # plates separating; no impulse.
        reduced_mass = m_a * m_b / (m_a + m_b)
        # Contact-fraction boost-and-cap: physically, real plate contact
        # transmits force across the whole suture, not just at overlap
        # cells. Boost the raw overlap fraction by ``contact_boost``
        # before capping at 1.0 — so even a modest 20% overlap (at
        # boost=5) gives full normal-velocity equalisation in one tick.
        # 1-cell grazes are still mostly ignored (1 cell / 100-cell
        # plate × 5 = 0.05 → 5% impulse only).
        contact_frac = min(
            1.0,
            sim_config.momentum_contact_boost * n_overlap / max(1.0, min(m_a, m_b)))
        impulse_n = (1.0 + sim_config.momentum_restitution) * v_rel_n * reduced_mass * contact_frac
        # Apply impulse along the contact normal, opposite on each plate.
        dv_a_n = -impulse_n / m_a
        dv_b_n = impulse_n / m_b
        a.velocity_kmpy[0] += dv_a_n * nx
        a.velocity_kmpy[1] += dv_a_n * ny
        b.velocity_kmpy[0] += dv_b_n * nx
        b.velocity_kmpy[1] += dv_b_n * ny
        n_exchanges += 1

    return n_exchanges

