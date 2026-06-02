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


# 4-neighbour offsets (toroidal np.roll shifts) shared by the smoothing /
# neighbour-scan passes in this module.
NEIGHBORS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _masked_laplacian_smooth(
    thickness: np.ndarray, mask: np.ndarray, strength: float,
) -> np.ndarray:
    """Wrap-aware 4-neighbour Laplacian smoothing of ``thickness`` over the
    cells in ``mask``. Each masked cell is blended toward the mean of its
    in-mask 4-neighbours by ``strength``; cells outside ``mask`` are
    unchanged. Returns a new array.
    """
    n_sum = np.zeros_like(thickness)
    n_count = np.zeros_like(thickness, dtype=np.float64)
    for sy, sx in NEIGHBORS_4:
        roll_mask = np.roll(mask, shift=(sy, sx), axis=(0, 1))
        roll_th = np.roll(thickness, shift=(sy, sx), axis=(0, 1))
        n_sum = n_sum + np.where(roll_mask, roll_th, 0.0)
        n_count = n_count + roll_mask.astype(np.float64)
    mean = np.where(n_count > 0, n_sum / np.maximum(n_count, 1.0), thickness)
    blended = (1.0 - strength) * thickness + strength * mean
    return np.where(mask, blended, thickness)


def _apply_erosion(
    plates: list[PolygonPlate], sim_config: SimConfig,
) -> None:
    """Per-plate continental thickness Laplacian smoothing, operating
    on the BODY frame.

    Erosion is a physical process that smooths thickness within the
    plate's own (body) frame — the same wrap-aware neighbour Laplacian
    that the world-frame version used, just applied to ``body_*``.
    Applying it here (rather than to the world view) sidesteps the
    inverse-NN-off-by-1 issue that derasterise has under rotation:
    erosion fires on every continental cell every ``erosion_period``
    ticks, and pumping every one of those changes through a lossy
    derasterise leaks mass at the boundary. The world view also gets
    the update for consistency within the tick.
    """
    strength = sim_config.erosion_strength
    if strength <= 0.0:
        return
    for p in plates:
        if not p.alive:
            continue
        # Body-frame Laplacian.
        body_cont = p.body_mask & (p.body_crust == CRUST_CONTINENTAL)
        if body_cont.any():
            p.body_thickness = _masked_laplacian_smooth(
                p.body_thickness, body_cont, strength)
        # World-frame Laplacian (for current-tick consumers + baseline
        # consistency so the diff doesn't try to flush it again).
        world_cont = p.cell_mask & (p.crust == CRUST_CONTINENTAL)
        if world_cont.any():
            p.thickness = _masked_laplacian_smooth(
                p.thickness, world_cont, strength)
            if hasattr(p, "_world_baseline_thickness"):
                p._world_baseline_thickness = p.thickness.copy()  # type: ignore[attr-defined]


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
            for sy, sx in NEIGHBORS_4:
                ny = (y + sy) % gy
                nx = (x + sx) % gx
                if cont_mask[ny, nx]:
                    p.thickness[ny, nx] += p.thickness[y, x]
                    transferred = True
                    break
            if not transferred:
                for sy, sx in NEIGHBORS_4:
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
    """Increment every owned cell's age by ``dt``.

    Applies directly to the **body frame** (``body_age``) rather than
    the world view. This sidesteps a fundamental issue with the
    body↔world NN sampling at rotated poses: NN forward and NN inverse
    are off by 1 in the cell index at non-axis-aligned rotations, so
    every paint_change diff propagated by ``derasterise`` slowly leaks
    mass at the plate's trailing rotational edge. Aging would cause
    that diff to fire on every cell every tick — catastrophic mass
    leak. By updating ``body_age`` in place we keep the diff sparse
    (driven only by contention / accretion / hotspot / fold-belt
    changes, which are bounded).

    The world view is updated too so any subsequent per-tick op that
    reads ``p.age`` sees the new values within this tick.
    """
    for p in plates:
        if not p.alive:
            continue
        p.body_age = np.where(p.body_mask, p.body_age + dt, p.body_age)
        p.age = np.where(p.cell_mask, p.age + dt, p.age)
        # Bring the saved baseline up to date so the end-of-tick diff
        # doesn't see this as a paint_change to flush again.
        if hasattr(p, "_world_baseline_age"):
            p._world_baseline_age = np.where(  # type: ignore[attr-defined]
                p.cell_mask, p._world_baseline_age + dt, p._world_baseline_age,
            )


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
