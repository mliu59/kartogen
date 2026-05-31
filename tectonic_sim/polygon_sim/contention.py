"""Per-cell ownership resolution + C-C folding."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL

from tectonic_sim.polygon_sim.types import (
    PolygonPlate)


def _global_owner(plates: list[PolygonPlate], gy: int, gx: int) -> np.ndarray:
    """Snapshot the unique-owner-per-cell map (pid or -1)."""
    owner = np.full((gy, gx), -1, dtype=np.int64)
    for p in plates:
        if not p.alive:
            continue
        # If a cell is claimed by multiple plates at this snapshot (which
        # shouldn't happen at steady state), the later plate overwrites —
        # that's fine for "previous owner" purposes, this map only feeds
        # the trailing-edge gap fill.
        owner[p.cell_mask] = p.pid
    return owner



def _resolve_contention(
    plates: list[PolygonPlate], gy: int, gx: int, sim_config,
) -> None:
    """At cells claimed by >1 plate after stamping, pick a per-cell
    winner and clear the loser masks (and apply C-C fold).

    Priority is per CELL, using the cell-level crust each plate stamped:
      * continental beats oceanic;
      * among continental, lower plate id wins (pyplatec uses smaller
        segment area; this is the simpler deterministic proxy);
      * among oceanic, younger cell age wins; ties by lower pid.
    """
    alive = [p for p in plates if p.alive]
    n = len(alive)
    if n == 0:
        return

    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contend = masks.sum(axis=0)
    contested = contend > 1
    if not contested.any():
        return

    crusts = np.stack([p.crust for p in alive], axis=0)
    ages = np.stack([p.age for p in alive], axis=0)
    thicks = np.stack([p.thickness for p in alive], axis=0)
    pids = np.array([p.pid for p in alive], dtype=np.float64)

    # Per-plate-per-cell priority score, only where masked.
    # Mass-weighted priority: each plate's score at a contested cell is
    # its own total cell count, weighted by the cell's crust type at
    # that plate. Continental cells get a strong (sim_config.crust_continental_weight×)
    # boost so normal-sized continents dominate oceanic plates, but a
    # truly tiny continental island (mass << contesting plate / weight)
    # loses to a vastly larger oceanic neighbour — fixing the "small
    # island carves through a huge plate" artefact where a 5-cell
    # continental fragment used to win at every cell it stamped.
    cont_per_cell = crusts == CRUST_CONTINENTAL
    mass = masks.sum(axis=(1, 2)).astype(np.float64)
    crust_factor = np.where(cont_per_cell, sim_config.crust_continental_weight, 1.0)
    mass_score = mass[:, None, None] * crust_factor
    # Oceanic age tie-breaker (younger oceanic wins all else equal).
    # Continental cells get age_penalty=0 so age doesn't shift their
    # score; only their mass + crust weight count.
    age_penalty = np.where(cont_per_cell, 0.0, ages)
    score = mass_score - age_penalty - 1e-6 * pids[:, None, None]
    score = np.where(masks, score, -np.inf)

    winner = np.argmax(score, axis=0)        # (gy, gx)
    plate_idx = np.arange(n)[:, None, None]
    is_winner = plate_idx == winner[None, :, :]
    loser_mask = masks & ~is_winner & contested[None, :, :]

    # C-C fold: at contested cells where the winner is continental, sum
    # contributions from continental losers into the winner's thickness.
    winner_crust = np.take_along_axis(crusts, winner[None], axis=0)[0]
    cc_loser_thick = np.where(
        loser_mask & (crusts == CRUST_CONTINENTAL), thicks, 0.0).sum(axis=0)
    fold_cell = contested & (winner_crust == CRUST_CONTINENTAL) & (cc_loser_thick > 0)

    # Clear loser masks (and paint at those cells).
    cleared_masks = masks & ~loser_mask
    cleared_thicks = np.where(loser_mask, 0.0, thicks)
    cleared_ages = np.where(loser_mask, 0.0, ages)
    cleared_crust = np.where(loser_mask, np.int8(0), crusts)

    # Apply fold to winner cells: thick[winner, fold_cell] += folding_ratio * cc_loser_thick.
    fold_add = sim_config.folding_ratio * cc_loser_thick * fold_cell
    if fold_add.any():
        # Scatter-add to the winner plane at each cell.
        flat_winner = winner.ravel()
        flat_add = fold_add.ravel()
        gy_gx = gy * gx
        # Build a (n, gy*gx) addend, then sum back.
        add = np.zeros((n, gy_gx), dtype=np.float64)
        idx = np.arange(gy_gx)
        add[flat_winner, idx] = flat_add
        cleared_thicks = cleared_thicks + add.reshape(n, gy, gx)

    # Write back.
    for k, p in enumerate(alive):
        p.cell_mask = cleared_masks[k]
        p.crust = cleared_crust[k].astype(np.int8)
        p.age = cleared_ages[k]
        p.thickness = cleared_thicks[k]

