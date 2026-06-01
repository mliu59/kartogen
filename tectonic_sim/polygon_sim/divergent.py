"""Trailing-edge divergent fill — nearest-plate gap assignment.

With the continuous-pose architecture, plates don't leave discrete
``np.roll`` trails behind. Instead, as plates drift apart their
world-frame coverage shrinks at the trailing edges and gaps open
between plates. Those gaps are where mid-ocean ridges spawn new
oceanic crust.

This module fills gap cells by assigning each one to the **nearest
plate** in world-km space (i.e. nearest ``position_km``), with fresh
age-0 oceanic crust. The plate inherits the new cell into its
world-frame view; the end-of-tick ``derasterise`` pass folds it into
the body frame.

Plates that are far from the gap (gap is closer to another plate)
don't claim — that other plate does. So a gap opening on the trailing
side of plate A and the leading side of plate B is more likely to be
near plate A's centroid → A claims, with fresh oceanic. This matches
the physical picture of an oceanic ridge belonging to the receding
plate.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_OCEANIC, WorldRect

from tectonic_sim.polygon_sim.types import PolygonPlate


def _trailing_edge_fill(
    plates: list[PolygonPlate],
    sim_config, gy: int, gx: int,
    cell_km: float = 0.0,
    domain: WorldRect | None = None,
) -> None:
    """Assign each gap cell to the nearest plate (by world-km) and
    stamp fresh age-0 oceanic crust there.

    ``cell_km`` and ``domain`` are passed by ``simulate.py``. They're
    nominally optional only so this stays loosely coupled to the call
    site; the function does nothing useful without them. (Keeping the
    older two-argument signature would require positional-only
    plumbing; this is simpler.)
    """
    alive = [p for p in plates if p.alive]
    if not alive or domain is None or cell_km <= 0.0:
        return
    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contend = masks.sum(axis=0)
    gaps = contend == 0
    if not gaps.any():
        return

    # Gap-cell world coordinates.
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km
    gy_idx, gx_idx = np.where(gaps)
    gap_x = (gx_idx + 0.5) * cell_km - half_w
    gap_y = (gy_idx + 0.5) * cell_km - half_h

    # For each plate, wrap-aware distance from the plate's position_km
    # to every gap cell. Take per-cell argmin to pick the assigned plate.
    n_alive = len(alive)
    n_gaps = gx_idx.size
    sq_dist = np.full((n_alive, n_gaps), np.inf, dtype=np.float64)
    for k, p in enumerate(alive):
        dx = gap_x - float(p.position_km[0])
        dy = gap_y - float(p.position_km[1])
        wx, wy = domain.wrapped_delta_xy(dx, dy)
        sq_dist[k] = wx * wx + wy * wy

    assign = np.argmin(sq_dist, axis=0)

    # Apply: each assigned plate gains the gap cell as fresh oceanic.
    for k, p in enumerate(alive):
        sel_idx = np.where(assign == k)[0]
        if sel_idx.size == 0:
            continue
        ys = gy_idx[sel_idx]
        xs = gx_idx[sel_idx]
        p.cell_mask[ys, xs] = True
        p.crust[ys, xs] = np.int8(CRUST_OCEANIC)
        p.age[ys, xs] = 0.0
        p.thickness[ys, xs] = sim_config.oceanic_thickness_km
