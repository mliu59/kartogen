"""Plate fusion (small-into-big merge).

Fusion is a **structural** change — it welds one plate's entire body
into another. Under the continuous-pose architecture the canonical
state is the body frame; the per-tick world view is a cached
rasterisation of (body, pose). A world-frame-only merge would force
``derasterise`` to re-derive the whole transferred block through its
lossy NN-inverse ADD path every fusion, and the next tick's two-pass
forward-scatter rasterise disagrees with that NN-inverse placement —
so the welded region visibly snaps in and out for several ticks.

To avoid that, fusion transfers ``small``'s cells into ``big``'s
**body frame** directly (a body→world→body rigid map through the two
plates' poses), then:

  - rebuilds ``big``'s world view for the transferred cells so the
    remaining same-tick ops (contention, aging, …) see the merge;
  - resets ``big``'s saved baseline to the post-fusion world view so
    ``derasterise`` classifies the fused cells as *unchanged* (already
    in body) rather than *added*;
  - points ``big``'s source mapping for the transferred world cells at
    the body cells just written, so any later same-tick paint change /
    clear flushes to the right body cell;
  - zeroes ``small``'s body so it can't resurrect.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, SimConfig, WorldRect

from tectonic_sim.polygon_sim.types import PolygonPlate
from tectonic_sim.polygon_sim.rasterize import _world_to_body
from tectonic_sim.polygon_sim.momentum import _plate_centroid_km


def _transfer_body(
    small: PolygonPlate, big: PolygonPlate,
    domain: WorldRect, gy: int, gx: int, cell_km: float,
) -> None:
    """Weld ``small``'s footprint into ``big``'s body + world frames.

    Transfers from ``small``'s **world** arrays (the authoritative
    footprint rasterise's validated two-pass already produced) rather
    than re-deriving small's world position from its body — that saves
    a redundant body→world resample layer whose NN collisions would
    mottle the welded crust. For each world cell ``small`` owns that
    ``big`` doesn't:

      1. inverse-project the world cell into ``big``'s body cell via
         ``big``'s pose (single NN map),
      2. stamp ``small``'s world paint into that body cell,
      3. mirror it into ``big``'s world view (same world grid cell).

    Big keeps its own paint wherever it already owns the cell (it was
    the priority winner). Also keeps ``big``'s saved baseline + source
    mapping in step so the end-of-tick derasterise diff treats the
    fused cells as *unchanged* (already in body), not as a fresh
    NN-inverse ADD block.
    """
    # Cells small owns in world but big doesn't — these are the ones
    # that actually move ownership.
    transfer = small.cell_mask & ~big.cell_mask
    if not transfer.any():
        return

    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km

    wj, wi = np.where(transfer)
    # World km of each transferred cell (same grid for small and big).
    wx = (wi + 0.5) * cell_km - half_w
    wy = (wj + 0.5) * cell_km - half_h

    # World km → big's body cell via big's inverse pose (single NN).
    bbx, bby = _world_to_body(
        big.position_km, big.orientation_rad, big.body_pivot_km,
        wx, wy, domain,
    )
    bbx = (bbx + half_w) % domain.width_km - half_w
    bby = (bby + half_h) % domain.height_km - half_h
    bi = np.floor((bbx + half_w) / cell_km).astype(np.int64) % gx
    bj = np.floor((bby + half_h) / cell_km).astype(np.int64) % gy

    # Paint carried across, read from small's WORLD arrays.
    s_crust = small.crust[wj, wi]
    s_age = small.age[wj, wi]
    s_thick = small.thickness[wj, wi]

    # Write into big's body (big keeps its own paint where its body
    # already owns the cell — relevant only on NN collisions).
    fresh = ~big.body_mask[bj, bi]
    big.body_mask[bj[fresh], bi[fresh]] = True
    big.body_crust[bj[fresh], bi[fresh]] = s_crust[fresh]
    big.body_age[bj[fresh], bi[fresh]] = s_age[fresh]
    big.body_thickness[bj[fresh], bi[fresh]] = s_thick[fresh]

    # Mirror into big's world view for the rest of this tick.
    big.cell_mask[wj, wi] = True
    big.crust[wj, wi] = s_crust
    big.age[wj, wi] = s_age
    big.thickness[wj, wi] = s_thick

    # Keep big's saved baseline + source mapping consistent so
    # derasterise sees these cells as "unchanged" (already in body).
    if hasattr(big, "_world_baseline_mask"):
        big._world_baseline_mask[wj, wi] = True  # type: ignore[attr-defined]
        big._world_baseline_crust[wj, wi] = s_crust  # type: ignore[attr-defined]
        big._world_baseline_age[wj, wi] = s_age  # type: ignore[attr-defined]
        big._world_baseline_thickness[wj, wi] = s_thick  # type: ignore[attr-defined]
    if hasattr(big, "_source_bj") and hasattr(big, "_source_bi"):
        big._source_bj[wj, wi] = bj  # type: ignore[attr-defined]
        big._source_bi[wj, wi] = bi  # type: ignore[attr-defined]


def _apply_fusion(
    plates: list[PolygonPlate], sim_config: SimConfig,
    domain: WorldRect, gy: int, gx: int, cell_km: float,
) -> int:
    """Merge a small plate into a larger one when they're heavily
    overlapping. Models suture-zone welding: when two plates lock
    together at the boundary and the smaller is mostly embedded in the
    larger, they become a single plate (the larger's identity wins).

    Per pair (A_small, B_big) where A's cell_mask is mostly inside
    B's cell_mask (overlap_fraction = overlap_cells / mass_A >
    threshold), A's body is transferred into B's body frame (see
    :func:`_transfer_body`) and A is marked dead.

    ``both_continental_only``: when True, fusion only triggers if
    BOTH plates are predominantly continental (≥50% continental cells).
    Oceanic plates should subduct, not weld, so this gate prevents
    accidentally absorbing an oceanic plate into a continent en masse.

    Returns number of fusion events.
    """
    overlap_threshold = sim_config.fusion_overlap_threshold
    both_continental_only = sim_config.fusion_both_continental_only
    alive = [p for p in plates if p.alive and p.cell_mask.any()]
    n = len(alive)
    if n < 2:
        return 0
    masks = np.stack([p.cell_mask for p in alive], axis=0)
    masses = masks.sum(axis=(1, 2))
    contended = masks.sum(axis=0) > 1
    if not contended.any():
        return 0

    # Pair overlap counts via contested-cell iteration.
    pair_overlap: dict[tuple[int, int], int] = {}
    iy, ix = np.where(contended)
    for y, x in zip(iy.tolist(), ix.tolist()):
        idxs = [k for k in range(n) if masks[k, y, x]]
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                key = (idxs[ii], idxs[jj])
                pair_overlap[key] = pair_overlap.get(key, 0) + 1
    if not pair_overlap:
        return 0

    # Per plate: predominantly-continental?
    is_cont_plate = np.array([
        bool(
            (p.cell_mask & (p.crust == CRUST_CONTINENTAL)).sum() * 2
            >= int(p.cell_mask.sum())
        )
        for p in alive
    ], dtype=bool)

    # Wrap-aware centroids (one per alive plate) for the suturing
    # velocity gate: a pair only welds once its contact-normal relative
    # velocity has decayed to <= fusion_max_relative_velocity_kmpy.
    max_rel_v = sim_config.fusion_max_relative_velocity_kmpy
    centroids: list[tuple[float, float] | None] = [
        _plate_centroid_km(p, domain, gy, gx, cell_km) for p in alive
    ]

    def _converging_too_fast(a_idx: int, b_idx: int) -> bool:
        """True if the pair is still ramming together faster than the
        suturing gate allows. Contact-normal relative velocity along the
        wrap-aware centroid-to-centroid axis; matches the convention in
        ``_apply_momentum_exchange`` (v_rel_n > 0 ⇔ approaching). Pairs
        separating or already arrested (v_rel_n <= max_rel_v) pass."""
        ca = centroids[a_idx]
        cb = centroids[b_idx]
        if ca is None or cb is None:
            return False
        dx, dy = domain.wrapped_delta_xy(
            np.array([cb[0] - ca[0]]), np.array([cb[1] - ca[1]]))
        dx, dy = float(dx[0]), float(dy[0])
        norm_mag = (dx * dx + dy * dy) ** 0.5
        if norm_mag < 1e-6:
            return False  # coincident centroids — no defined normal.
        nx, ny = dx / norm_mag, dy / norm_mag
        a = alive[a_idx]
        b = alive[b_idx]
        v_rel_x = float(a.velocity_kmpy[0] - b.velocity_kmpy[0])
        v_rel_y = float(a.velocity_kmpy[1] - b.velocity_kmpy[1])
        v_rel_n = v_rel_x * nx + v_rel_y * ny
        return v_rel_n > max_rel_v

    # For each plate, find its dominant overlap partner that is BIGGER
    # than it. Merge the small one into the big one. Order: smallest
    # first, so we cascade fusions correctly within a single tick.
    merged_in: set[int] = set()
    fusion_jobs: list[tuple[int, int]] = []
    for p_idx in np.argsort(masses):
        if p_idx in merged_in:
            continue
        p_mass = masses[p_idx]
        if p_mass == 0:
            continue
        best_partner = -1
        best_overlap = 0
        for q_idx in range(n):
            if q_idx == p_idx or q_idx in merged_in:
                continue
            if masses[q_idx] <= p_mass:
                continue
            if both_continental_only and not (
                is_cont_plate[p_idx] and is_cont_plate[q_idx]
            ):
                continue
            key = (min(p_idx, q_idx), max(p_idx, q_idx))
            overlap = pair_overlap.get(key, 0)
            if overlap > best_overlap:
                best_overlap = overlap
                best_partner = q_idx
        if best_partner < 0:
            continue
        if best_overlap / p_mass < overlap_threshold:
            continue
        # Suturing velocity gate: only weld once momentum has arrested
        # the convergence. Still ramming together → skip this tick (the
        # pair keeps converging via momentum/contention and may weld a
        # later tick once relative velocity has decayed).
        if _converging_too_fast(int(p_idx), best_partner):
            continue
        fusion_jobs.append((p_idx, best_partner))
        merged_in.add(p_idx)

    n_fusions = 0
    for small_idx, big_idx in fusion_jobs:
        small = alive[small_idx]
        big = alive[big_idx]
        if not small.alive or not big.alive:
            continue
        # Body-frame weld: transfer small's body into big's body +
        # world (and keep big's baseline / source mapping consistent).
        _transfer_body(small, big, domain, gy, gx, cell_km)
        # Mark small dead and zero out both frames (world view AND the
        # canonical body) so it can't resurrect on a later rasterise.
        small.alive = False
        small.cell_mask = np.zeros_like(small.cell_mask)
        small.crust = np.zeros_like(small.crust)
        small.age = np.zeros_like(small.age)
        small.thickness = np.zeros_like(small.thickness)
        small.body_mask = np.zeros_like(small.body_mask)
        small.body_crust = np.zeros_like(small.body_crust)
        small.body_age = np.zeros_like(small.body_age)
        small.body_thickness = np.zeros_like(small.body_thickness)
        n_fusions += 1

    return n_fusions
