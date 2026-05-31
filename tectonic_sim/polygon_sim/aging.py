"""Aging, erosion, thinned-continental absorption, buoyancy.

All tunables read from ``sim_config``:
  - ``erosion_period`` / ``erosion_strength`` — Laplacian thickness
    smoothing on continental cells (fires every N ticks).
  - ``min_continental_thickness_km`` — continental cells thinner than
    this are absorbed by a neighbour and revert to oceanic.
  - ``buoyancy_bonus_frac`` / ``max_buoyancy_age_myr`` — young oceanic
    ridge bonus, decays linearly to 0 at max age.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, CRUST_OCEANIC, SimConfig

from tectonic_sim.polygon_sim.types import PolygonPlate


def _apply_erosion(
    plates: list[PolygonPlate], sim_config: SimConfig,
) -> None:
    """Per-plate continental thickness Laplacian smoothing.

    For each plate, compute the 4-neighbour mean thickness over the
    plate's continental cells (wrap-aware via ``np.roll``) and blend
    each cell toward that mean by ``sim_config.erosion_strength``.
    """
    strength = sim_config.erosion_strength
    if strength <= 0.0:
        return
    for p in plates:
        if not p.alive:
            continue
        mask = p.cell_mask & (p.crust == CRUST_CONTINENTAL)
        if not mask.any():
            continue
        th = p.thickness
        n_sum = np.zeros_like(th)
        n_count = np.zeros_like(th, dtype=np.float64)
        for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            roll_mask = np.roll(mask, shift=(sy, sx), axis=(0, 1))
            roll_th = np.roll(th, shift=(sy, sx), axis=(0, 1))
            n_sum = n_sum + np.where(roll_mask, roll_th, 0.0)
            n_count = n_count + roll_mask.astype(np.float64)
        mean = np.where(n_count > 0, n_sum / np.maximum(n_count, 1.0), th)
        new_th = (1.0 - strength) * th + strength * mean
        p.thickness = np.where(mask, new_th, p.thickness)


def _apply_thinned_continental_absorption(
    plates: list[PolygonPlate], sim_config: SimConfig,
) -> int:
    """Continental cells thinned below
    ``sim_config.min_continental_thickness_km`` are absorbed.

    Their remaining thickness is transferred to a 4-neighbour same-plate
    continental cell (preferred — preserves continental mass), or to a
    same-plate oceanic neighbour (fallback). The thinned cell itself is
    converted to oceanic at ``sim_config.oceanic_thickness_km``, age 0.

    Returns the number of cells absorbed.
    """
    min_thickness_km = sim_config.min_continental_thickness_km
    if min_thickness_km <= 0.0:
        return 0
    n_absorbed = 0
    for p in plates:
        if not p.alive:
            continue
        thinned = (
            p.cell_mask
            & (p.crust == CRUST_CONTINENTAL)
            & (p.thickness < min_thickness_km)
            & (p.thickness > 0.0)
        )
        if not thinned.any():
            continue
        gy, gx = p.cell_mask.shape
        ys, xs = np.where(thinned)
        cont_mask = p.cell_mask & (p.crust == CRUST_CONTINENTAL) & (~thinned)
        for y, x in zip(ys.tolist(), xs.tolist()):
            transferred = False
            for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = (y + sy) % gy
                nx = (x + sx) % gx
                if cont_mask[ny, nx]:
                    p.thickness[ny, nx] += p.thickness[y, x]
                    transferred = True
                    break
            if not transferred:
                for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = (y + sy) % gy
                    nx = (x + sx) % gx
                    if p.cell_mask[ny, nx]:
                        p.thickness[ny, nx] += p.thickness[y, x]
                        transferred = True
                        break
            p.crust[y, x] = np.int8(CRUST_OCEANIC)
            p.thickness[y, x] = sim_config.oceanic_thickness_km
            p.age[y, x] = 0.0
            n_absorbed += 1
    return n_absorbed


def _apply_aging(plates: list[PolygonPlate], dt: float) -> None:
    """Increment every owned cell's age by ``dt``."""
    for p in plates:
        if not p.alive:
            continue
        p.age = np.where(p.cell_mask, p.age + dt, p.age)


def _apply_buoyancy_to_thickness(
    plates: list[PolygonPlate], sim_config: SimConfig,
) -> None:
    """Render-time adjustment: young oceanic gets a ridge bonus
    decaying linearly to 0 at ``sim_config.max_buoyancy_age_myr``.
    """
    max_age = sim_config.max_buoyancy_age_myr
    if max_age <= 0.0 or sim_config.buoyancy_bonus_frac <= 0.0:
        return
    for p in plates:
        if not p.alive:
            continue
        oc = p.cell_mask & (p.crust == CRUST_OCEANIC)
        if not oc.any():
            continue
        frac = np.clip(1.0 - p.age / max_age, 0.0, 1.0)
        bonus = (
            sim_config.buoyancy_bonus_frac
            * sim_config.oceanic_thickness_km * frac
        )
        p.thickness = np.where(oc, p.thickness + bonus, p.thickness)
