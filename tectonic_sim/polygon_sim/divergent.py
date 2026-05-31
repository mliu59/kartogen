"""Trailing-edge divergent fill (new ocean in plate wakes)."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_OCEANIC

from tectonic_sim.polygon_sim.types import PolygonPlate


def _trailing_edge_fill(
    plates: list[PolygonPlate], prev_owner: np.ndarray,
    sim_config, gy: int, gx: int) -> None:
    """Cells nobody claims after stamping + resolution are gaps. Give
    each gap to its previous owner (the plate whose trailing edge just
    departed) with age-0 oceanic crust — pyplatec ``lithosphere.cpp:616``
    style.
    """
    alive = [p for p in plates if p.alive]
    if not alive:
        return
    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contend = masks.sum(axis=0)
    gaps = contend == 0
    if not gaps.any():
        return
    for p in alive:
        sel = gaps & (prev_owner == p.pid)
        if not sel.any():
            continue
        p.cell_mask = p.cell_mask | sel
        p.crust = np.where(sel, np.int8(CRUST_OCEANIC), p.crust)
        p.age = np.where(sel, 0.0, p.age)
        p.thickness = np.where(
            sel, sim_config.oceanic_thickness_km, p.thickness)

