"""Alpha-complex polygons: construction + per-tick re-extraction.

Two responsibilities, kept in one module because they're inseparable
in practice (every consumer that re-extracts also needs the build /
drift helpers, and vice versa):

  - ``_circumradius`` / ``_build_alpha_complex`` / ``_drift_polygon``
    — primitives for building an alpha-complex polygon from a point
    cloud in the plate's local (wrap-aware) frame, and advancing the
    polygon's reference point by a drift vector.

  - ``_re_extract_polygons`` — per-tick driver that rebuilds every
    alive plate's polygon from its current ``cell_mask`` and marks
    plates with too few cells (or degenerate point clouds) as not-alive.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import (
    AlphaComplex,
    PolygonPlate)


# ---------------------------------------------------------------------------
# Alpha-complex primitives.
# ---------------------------------------------------------------------------


def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    out = np.full(area.shape, np.inf)
    nz = area > 1e-9
    out[nz] = (la[nz] * lb[nz] * lc[nz]) / (4.0 * area[nz])
    return out


def _build_alpha_complex(
    points: np.ndarray, domain: WorldRect, alpha: float) -> AlphaComplex | None:
    if points.shape[0] < 4:
        return None
    ref = points[0].copy()
    dx, dy = domain.wrapped_delta_xy(
        points[:, 0] - ref[0], points[:, 1] - ref[1])
    local = np.column_stack([dx, dy])
    try:
        tri = Delaunay(local)
    except Exception:
        return None
    simp = tri.simplices
    circ = _circumradius(
        local[simp[:, 0]], local[simp[:, 1]], local[simp[:, 2]])
    keep = circ < alpha
    if not keep.any():
        return None
    return tri, keep, ref


def _drift_polygon(complex_: AlphaComplex, dx: float, dy: float) -> AlphaComplex:
    tri, keep, ref = complex_
    return tri, keep, ref + np.array([dx, dy], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-tick re-extraction.
# ---------------------------------------------------------------------------


def _re_extract_polygons(
    plates: list[PolygonPlate], domain: WorldRect,
    cell_xy: np.ndarray, cell_km: float, sim_config,
) -> None:
    """Rebuild each plate's alpha-complex from its owned-cell centres.
    Marks plates with too few cells as not-alive."""
    alpha = sim_config.alpha_factor * cell_km
    for p in plates:
        if not p.alive:
            continue
        sel = p.cell_mask.ravel()
        if int(sel.sum()) < 4:
            p.alive = False
            p.polygon = None
            continue
        pts = cell_xy[sel]
        cx = _build_alpha_complex(pts, domain, alpha)
        if cx is None:
            p.alive = False
            p.polygon = None
            continue
        p.polygon = cx
