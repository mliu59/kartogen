"""C-O arc accretion (Andean-style magmatism).

Reads its tunables from ``sim_config`` directly:
  - ``accretion_prob_per_boundary_per_tick`` — firing probability
  - ``accretion_cells_per_event`` — cells stamped per fire
  - ``accretion_inland_offset_min_km`` / ``_max_km`` — inland offset
    range (sampled uniform per event in km; converted to cells via
    ``cell_km``)
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, CRUST_OCEANIC, SimConfig

from tectonic_sim.polygon_sim.types import PolygonPlate


def _apply_co_accretion(
    plates: list[PolygonPlate], gy: int, gx: int,
    sim_config: SimConfig,
    cell_km: float,
    rng,
) -> int:
    """Continental-arc accretion: convert oceanic cells inland of C-O
    boundaries to continental, on the overriding plate.

    Returns number of cells accreted.
    """
    alive = [p for p in plates if p.alive and p.cell_mask.any()]
    if not alive:
        return 0

    accretion_prob = sim_config.accretion_prob_per_boundary_per_tick
    if accretion_prob <= 0.0:
        return 0
    offset_min_cells = max(
        1, int(round(sim_config.accretion_inland_offset_min_km / cell_km)),
    )
    offset_max_cells = max(
        offset_min_cells,
        int(round(sim_config.accretion_inland_offset_max_km / cell_km)),
    )

    # Build global owner + per-cell crust maps.
    global_owner = np.full((gy, gx), -1, dtype=np.int64)
    global_crust = np.full((gy, gx), -1, dtype=np.int8)
    for p in alive:
        global_owner[p.cell_mask] = p.pid
        global_crust[p.cell_mask] = p.crust[p.cell_mask]

    cont_global = global_crust == CRUST_CONTINENTAL

    # Boundary mask: continental cell that has any 4-neighbour which is
    # oceanic of a *different* plate. Wrap-aware via np.roll.
    co_boundary = np.zeros_like(cont_global)
    for shift_axis_dy, shift_axis_dx in (
        (-1, 0), (1, 0), (0, -1), (0, 1),
    ):
        n_owner = np.roll(
            global_owner, shift=(shift_axis_dy, shift_axis_dx), axis=(0, 1),
        )
        n_crust = np.roll(
            global_crust, shift=(shift_axis_dy, shift_axis_dx), axis=(0, 1),
        )
        co_boundary |= cont_global & (n_owner != global_owner) & (
            n_crust == CRUST_OCEANIC
        )
    if not co_boundary.any():
        return 0

    iy, ix = np.where(co_boundary)
    n_boundary = iy.size
    sample = rng.random(n_boundary) < accretion_prob
    if not sample.any():
        return 0
    offsets = rng.integers(
        offset_min_cells, offset_max_cells + 1, size=n_boundary,
    )

    plate_by_pid = {p.pid: p for p in alive}
    n_accreted = 0
    for k in range(n_boundary):
        if not sample[k]:
            continue
        y, x = int(iy[k]), int(ix[k])
        owner_pid = int(global_owner[y, x])
        p = plate_by_pid.get(owner_pid)
        if p is None:
            continue
        vx, vy = float(p.velocity_kmpy[0]), float(p.velocity_kmpy[1])
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < 1e-6:
            continue
        offset_cells = int(offsets[k])
        nx_dir, ny_dir = vx / speed, vy / speed
        dx_cells = int(round(-nx_dir * offset_cells))
        dy_cells = int(round(-ny_dir * offset_cells))
        target_y = (y + dy_cells) % gy
        target_x = (x + dx_cells) % gx
        if not p.cell_mask[target_y, target_x]:
            continue
        if p.crust[target_y, target_x] != CRUST_OCEANIC:
            continue
        cluster = [(target_y, target_x)]
        cluster.extend([
            (target_y, (target_x + 1) % gx),
            (target_y, (target_x - 1) % gx),
            ((target_y + 1) % gy, target_x),
            ((target_y - 1) % gy, target_x),
        ])
        for cy, cx in cluster[:sim_config.accretion_cells_per_event]:
            if not p.cell_mask[cy, cx]:
                continue
            if p.crust[cy, cx] != CRUST_OCEANIC:
                continue
            p.crust[cy, cx] = np.int8(CRUST_CONTINENTAL)
            p.thickness[cy, cx] = sim_config.continental_thickness_km
            p.age[cy, cx] = 0.0
            n_accreted += 1

    return n_accreted
