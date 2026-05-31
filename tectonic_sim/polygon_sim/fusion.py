"""Plate fusion (small-into-big merge)."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, SimConfig

from tectonic_sim.polygon_sim.types import PolygonPlate


def _apply_fusion(
    plates: list[PolygonPlate], sim_config: SimConfig,
) -> int:
    """Merge a small plate into a larger one when they're heavily
    overlapping. Models suture-zone welding: when two plates lock
    together at the boundary and the smaller is mostly embedded in the
    larger, they become a single plate (the larger's identity wins).

    Per pair (A_small, B_big) where A's cell_mask is mostly inside
    B's cell_mask (overlap_fraction = overlap_cells / mass_A >
    threshold):
      - Transfer all of A's cells (mask + paint) to B. Where they
        already coincide, B keeps its paint (it was the priority
        winner anyway). Where only A had the cell, B inherits A's
        paint — so a microcontinent's continental crust is preserved
        as continental on B.
      - Mark A as dead.

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
        fusion_jobs.append((p_idx, best_partner))
        merged_in.add(p_idx)

    n_fusions = 0
    for small_idx, big_idx in fusion_jobs:
        small = alive[small_idx]
        big = alive[big_idx]
        if not small.alive or not big.alive:
            continue
        # Cells where small has the mask but big doesn't: big inherits.
        # Cells where both have mask: big keeps its paint (priority winner).
        transfer = small.cell_mask & ~big.cell_mask
        if transfer.any():
            big.cell_mask = big.cell_mask | small.cell_mask
            big.crust = np.where(
                transfer, small.crust, big.crust).astype(np.int8)
            big.age = np.where(transfer, small.age, big.age)
            big.thickness = np.where(
                transfer, small.thickness, big.thickness)
        # Mark small dead and zero out (will be skipped by all future passes).
        small.alive = False
        small.cell_mask = np.zeros_like(small.cell_mask)
        small.crust = np.zeros_like(small.crust)
        small.age = np.zeros_like(small.age)
        small.thickness = np.zeros_like(small.thickness)
        n_fusions += 1

    return n_fusions

