"""Trailing-edge divergent fill — nearest-plate gap assignment with
surround-aware continental-basin override.

With the continuous-pose architecture, plates don't leave discrete
``np.roll`` trails behind. Instead, as plates drift apart their
world-frame coverage shrinks at the trailing edges and gaps open
between plates. Those gaps are where mid-ocean ridges spawn new
oceanic crust.

This module fills gap cells by assigning each one to the **nearest
plate** in world-km space (i.e. nearest ``position_km``). Each gap is
then stamped with fresh age-0 crust — type and thickness chosen by a
*per-component surround vote*:

  - **Oceanic (default)** — fresh oceanic ridge spawn. Thickness =
    ``sim_config.oceanic_thickness_km``.

  - **Continental basin** — triggered when a connected gap component's
    immediate surround is ≥ ``sim_config.divergent_fill_continental_threshold``
    continental by classified-fraction. Models foreland basins /
    inland troughs between converging continental blocks (C-C suture
    interior). Thickness = ``mean(continental surround) − basin_depth``,
    clamped at ``min_continental_thickness_km`` so the basin floor
    isn't reabsorbed on the next tick.

Connected-component grouping uses 4-connectivity with toroidal wrap.
Per-component vote means a long open-ocean ridge stays oceanic even if
one end of it touches continental coast — the global continental
fraction across the whole component dominates.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
)

from tectonic_sim.polygon_sim.types import PolygonPlate
from tectonic_sim.polygon_sim.topology import _torus_components


def _labelled_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Torus-aware 4-connectivity components, relabelled to a contiguous
    ``1..N`` (0 = outside mask). Thin wrapper over
    ``topology._torus_components`` (scipy ``label`` + seam union-find),
    remapping its arbitrary root labels to a dense range so callers can
    index per-component arrays by label directly.
    """
    labels = _torus_components(mask)
    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    if uniq.size == 0:
        return labels.astype(np.int64), 0
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    remap[uniq] = np.arange(1, uniq.size + 1)
    return remap[labels], int(uniq.size)


def _trailing_edge_fill(
    plates: list[PolygonPlate],
    sim_config: SimConfig, gy: int, gx: int,
    cell_km: float = 0.0,
    domain: WorldRect | None = None,
) -> None:
    """Assign each gap cell to the nearest plate (by world-km) and
    stamp fresh crust whose **type** is decided by a per-component
    surround vote (continental basin vs oceanic ridge).

    ``cell_km`` and ``domain`` are passed by ``simulate.py``. They're
    nominally optional only so this stays loosely coupled to the call
    site; the function does nothing useful without them.
    """
    alive = [p for p in plates if p.alive]
    if not alive or domain is None or cell_km <= 0.0:
        return
    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contend = masks.sum(axis=0)
    gaps = contend == 0
    if not gaps.any():
        return

    # --- Per-component surround vote ---
    # Build a global per-cell view of (crust, owned) sampled across all
    # alive plates. A cell may be owned by multiple plates this tick
    # because contention hasn't resolved-then-fixed yet; for the surround
    # vote we count any owned cell with its primary plate's crust code.
    # In practice the cell_mask sum > 0 set is contention-resolved by the
    # time _trailing_edge_fill runs, so each cell has a single owner
    # (or 0 = gap).
    crust_map = np.full((gy, gx), -1, dtype=np.int8)  # -1 = unowned/gap
    for p in alive:
        owned = p.cell_mask & (crust_map < 0)
        crust_map[owned] = p.crust[owned]

    labels, n_components = _labelled_components(gaps)

    # Per-component decision: crust type code + fill thickness scalar.
    threshold = float(sim_config.divergent_fill_continental_threshold)
    basin_depth_km = float(sim_config.divergent_fill_basin_depth_km)
    min_cont_km = float(sim_config.min_continental_thickness_km)
    oceanic_th = float(sim_config.oceanic_thickness_km)

    fill_type = np.full(n_components + 1, CRUST_OCEANIC, dtype=np.int8)
    fill_thickness = np.full(n_components + 1, oceanic_th, dtype=np.float64)

    # 4-neighbour shifts with wrap.
    shifts = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for comp_id in range(1, n_components + 1):
        comp_mask = labels == comp_id
        # Collect surround: any cell that is a 4-neighbour of a comp
        # cell but is NOT in the component itself (could be in a
        # different gap component or owned by a plate).
        surround = np.zeros_like(comp_mask)
        for sy, sx in shifts:
            surround |= np.roll(comp_mask, shift=(sy, sx), axis=(0, 1))
        surround &= ~comp_mask

        # Classify surround cells by the owner's crust code. Other gap
        # cells (crust_map == -1) contribute nothing to the tally.
        surround_crust = crust_map[surround]
        n_cont = int(np.count_nonzero(surround_crust == CRUST_CONTINENTAL))
        n_ocean = int(np.count_nonzero(surround_crust == CRUST_OCEANIC))
        n_classified = n_cont + n_ocean
        if n_classified == 0:
            continue  # default oceanic + oceanic_thickness_km

        cont_frac = n_cont / n_classified
        if cont_frac < threshold:
            continue  # default oceanic

        # Continental basin: mean continental thickness in surround −
        # basin_depth_km, clamped at min_continental_thickness_km.
        cont_cells = surround & (crust_map == CRUST_CONTINENTAL)
        cont_thickness_sample: list[float] = []
        for p in alive:
            owned_cont = cont_cells & p.cell_mask & (p.crust == CRUST_CONTINENTAL)
            if owned_cont.any():
                cont_thickness_sample.extend(p.thickness[owned_cont].tolist())
        if not cont_thickness_sample:
            continue
        mean_th = float(np.mean(cont_thickness_sample))
        basin_th = max(mean_th - basin_depth_km, min_cont_km)
        fill_type[comp_id] = np.int8(CRUST_CONTINENTAL)
        fill_thickness[comp_id] = basin_th

    # --- Nearest-plate assignment (km-space) ---
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km
    gy_idx, gx_idx = np.where(gaps)
    gap_x = (gx_idx + 0.5) * cell_km - half_w
    gap_y = (gy_idx + 0.5) * cell_km - half_h

    n_alive = len(alive)
    n_gaps = gx_idx.size
    sq_dist = np.full((n_alive, n_gaps), np.inf, dtype=np.float64)
    for k, p in enumerate(alive):
        dx = gap_x - float(p.position_km[0])
        dy = gap_y - float(p.position_km[1])
        wx, wy = domain.wrapped_delta_xy(dx, dy)
        sq_dist[k] = wx * wx + wy * wy

    assign = np.argmin(sq_dist, axis=0)

    # --- Stamp ---
    # Each gap cell takes the type / thickness decided for its component,
    # and is owned by its nearest plate. Age = 0 in both cases (freshly
    # accreted, mid-ocean-ridge or suture-basin material).
    gap_comp_id = labels[gy_idx, gx_idx]
    gap_fill_type = fill_type[gap_comp_id]
    gap_fill_th = fill_thickness[gap_comp_id]

    for k, p in enumerate(alive):
        sel_idx = np.where(assign == k)[0]
        if sel_idx.size == 0:
            continue
        ys = gy_idx[sel_idx]
        xs = gx_idx[sel_idx]
        p.cell_mask[ys, xs] = True
        p.crust[ys, xs] = gap_fill_type[sel_idx]
        p.age[ys, xs] = 0.0
        p.thickness[ys, xs] = gap_fill_th[sel_idx]
