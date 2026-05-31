"""Volcanic hotspots (Wilson-Morgan mantle plumes)."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, WorldRect

from tectonic_sim.polygon_sim.types import (
    Hotspot,
    PolygonPlate,
    _HOTSPOT_RNG_TAG)


def _initialize_hotspots(
    domain: WorldRect, sim_config, seed: int) -> list[Hotspot]:
    """Place ``sim_config.hotspot_density_per_km2 * area`` hotspots uniformly over
    the mantle-frame sim domain. Each gets a staggered birth tick and a
    Normal-distributed lifespan.

    Hotspots live in the centred frame ([-half_w, +half_w], [-half_h,
    +half_h]) — same convention as ``_cell_centres``. The list is sorted
    by birth_tick for deterministic iteration order.
    """
    area_km2 = domain.width_km * domain.height_km
    n = int(round(sim_config.hotspot_density_per_km2 * area_km2))
    if n <= 0:
        return []
    rng = np.random.Generator(np.random.PCG64(seed ^ _HOTSPOT_RNG_TAG))
    half_w = 0.5 * domain.width_km
    half_h = 0.5 * domain.height_km
    xs = rng.uniform(-half_w, +half_w, size=n)
    ys = rng.uniform(-half_h, +half_h, size=n)
    if sim_config.hotspot_birth_stagger_ticks > 0:
        births = rng.integers(
            -sim_config.hotspot_birth_stagger_ticks,
            +sim_config.hotspot_birth_stagger_ticks + 1,
            size=n)
    else:
        births = np.zeros(n, dtype=np.int64)
    lifespans = rng.normal(
        sim_config.hotspot_lifespan_mean_ticks,
        sim_config.hotspot_lifespan_std_ticks,
        size=n)
    lifespans = np.maximum(1, np.round(lifespans).astype(np.int64))
    hotspots = [
        Hotspot(
            position_xy_km=(float(xs[i]), float(ys[i])),
            birth_tick=int(births[i]),
            lifespan_ticks=int(lifespans[i]))
        for i in range(n)
    ]
    hotspots.sort(key=lambda h: (h.birth_tick, h.position_xy_km))
    return hotspots


def _hotspot_cell(
    h: Hotspot, cell_km: float, gy: int, gx: int) -> tuple[int, int]:
    """Return the (cy, cx) cell index currently above hotspot ``h``.

    Converts from the centred [-half_w, +half_w] km frame to the
    [0, gx) grid frame. Wraps with modulo so off-edge hotspots project
    back into the torus (defensive — initial placement is in-range).
    """
    cx = int((h.position_xy_km[0] + 0.5 * gx * cell_km) / cell_km) % gx
    cy = int((h.position_xy_km[1] + 0.5 * gy * cell_km) / cell_km) % gy
    return cy, cx


def _stamp_hotspot_eruption(
    plates: list[PolygonPlate], cy: int, cx: int,
    *, sim_config, age_myr: float = 0.0,
    cell_km: float, gy: int, gx: int,
) -> int:
    """Stamp a disk of cells (centred on cy, cx, radius ``radius_km``)
    as continental, each on whichever live plate owns the cell.
    Returns the number of cells actually stamped — some cells in the
    disk may be unowned (between plates) and get skipped.

    Wrap-aware: the disk wraps across the toroidal sim boundary.
    Lowest-pid-wins per cell when multiple plates transiently claim
    it, matching ``_flatten_state``.

    A cell can be re-stamped (re-eruption on the same trail location):
    crust → continental, thickness → continental_thickness + bump,
    age reset to 0. The "fresh volcanism resurfaces the cell" behaviour
    is intentional.
    """
    radius_km = sim_config.hotspot_island_radius_km
    new_thickness = (
        sim_config.continental_thickness_km
        + sim_config.hotspot_thickness_bump_km
    )
    if radius_km <= 0.0:
        # Single-cell stamp (legacy behaviour).
        for p in sorted(plates, key=lambda q: q.pid):
            if p.alive and p.cell_mask[cy, cx]:
                p.crust[cy, cx] = np.int8(CRUST_CONTINENTAL)
                p.thickness[cy, cx] = new_thickness
                p.age[cy, cx] = age_myr
                return 1
        return 0
    r_cells = int(np.ceil(radius_km / cell_km))
    r_cells_sq = (radius_km / cell_km) ** 2
    alive = sorted([p for p in plates if p.alive], key=lambda q: q.pid)
    n_stamped = 0
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dx * dx + dy * dy > r_cells_sq:
                continue
            sy = (cy + dy) % gy
            sx = (cx + dx) % gx
            for p in alive:
                if p.cell_mask[sy, sx]:
                    p.crust[sy, sx] = np.int8(CRUST_CONTINENTAL)
                    p.thickness[sy, sx] = new_thickness
                    p.age[sy, sx] = age_myr
                    n_stamped += 1
                    break
    return n_stamped


def _apply_hotspot_prehistory(
    plates: list[PolygonPlate], hotspots: list[Hotspot],
    *, sim_config, cell_km: float, gy: int, gx: int, rng) -> int:
    """Replay pre-history for every hotspot with birth_tick < 0.

    Walks each negative-birth tick forward from birth to min(0, death),
    back-projecting the cell that WAS above the hotspot at that past
    tick to the cell that's there NOW (using the velocity of whichever
    plate currently covers the hotspot, since plate velocities are
    constant for the sim). With probability
    ``sim_config.hotspot_erupt_prob_per_tick`` per past tick, stamp that cell as
    continental.

    Approximation: assumes the current plate has covered the hotspot
    for the entire pre-history. Fine for the typical case where plates
    don't traverse the full sim during pre-history.
    """
    n_stamped = 0
    for h in hotspots:
        if h.birth_tick >= 0:
            continue
        cy_now, cx_now = _hotspot_cell(h, cell_km, gy, gx)
        current_owner = None
        for p in sorted(plates, key=lambda q: q.pid):
            if p.alive and p.cell_mask[cy_now, cx_now]:
                current_owner = p
                break
        if current_owner is None:
            continue
        vx = float(current_owner.velocity_kmpy[0])
        vy = float(current_owner.velocity_kmpy[1])
        end_tick = min(0, h.birth_tick + h.lifespan_ticks)
        for t in range(h.birth_tick, end_tick):
            if rng.random() >= sim_config.hotspot_erupt_prob_per_tick:
                continue
            # At past tick t (negative), the cell currently above H was
            # at H at that time. That cell has since drifted with the
            # plate by v*(0-t)*dt = -v*t*dt. So it is NOW at H - v*t*dt
            # (along +v direction for negative t).
            past_dt = (-t) * sim_config.dt_myr
            stamp_x_km = h.position_xy_km[0] + vx * past_dt
            stamp_y_km = h.position_xy_km[1] + vy * past_dt
            sx = int((stamp_x_km + 0.5 * gx * cell_km) / cell_km) % gx
            sy = int((stamp_y_km + 0.5 * gy * cell_km) / cell_km) % gy
            n_stamped += _stamp_hotspot_eruption(
                plates, sy, sx,
                sim_config=sim_config, age_myr=past_dt,
                cell_km=cell_km, gy=gy, gx=gx)
    return n_stamped


def _apply_hotspot_eruptions(
    plates: list[PolygonPlate], hotspots: list[Hotspot], tick: int,
    *, sim_config, cell_km: float, gy: int, gx: int, rng) -> int:
    """Per-tick: for each currently-active hotspot, with probability
    ``sim_config.hotspot_erupt_prob_per_tick`` stamp the cell above it as
    continental on whichever plate currently owns the cell. Returns
    the number of cells stamped this tick.
    """
    n = 0
    for h in hotspots:
        if not h.is_active(tick):
            continue
        if rng.random() >= sim_config.hotspot_erupt_prob_per_tick:
            continue
        cy, cx = _hotspot_cell(h, cell_km, gy, gx)
        n += _stamp_hotspot_eruption(
            plates, cy, cx,
            sim_config=sim_config, age_myr=0.0,
            cell_km=cell_km, gy=gy, gx=gx)
    return n

