"""Per-plate velocity + angular damping."""

from __future__ import annotations

import numpy as np

from tectonic_sim.polygon_sim.types import (
    PolygonPlate)


def _apply_velocity_damping(
    plates: list[PolygonPlate], sim_config,
) -> None:
    """Reduce each plate's bulk velocity by its collision exposure.

    Mirrors ``tectonic_sim.constraints.apply_velocity_damping`` for the
    particle sim, just substituting cell-mask overlap for particle-pair
    overlap. For every plate, ``fraction`` = (cells in cross-plate
    contention) / (total cells); the new velocity is::

        v_new = v_old × (1 − damping_strength × fraction)

    Plates not in contention with anything keep their velocity. Plates
    heavily in contention slow down — and energy dissipated this way is
    the budget that fold thickening already accumulated as orogeny.

    Call after stamping, before contention resolution (resolution clears
    the loser masks at contested cells, so the overlap count has to be
    captured first).
    """
    damping_strength = sim_config.velocity_damping_strength
    if damping_strength <= 0:
        return
    alive = [p for p in plates if p.alive and p.cell_mask.any()]
    if len(alive) < 2:
        return
    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contend = masks.sum(axis=0)
    contested = contend > 1
    if not contested.any():
        return
    for p in alive:
        n_total = int(p.cell_mask.sum())
        if n_total == 0:
            continue
        n_contested = int((p.cell_mask & contested).sum())
        if n_contested == 0:
            continue
        frac = n_contested / n_total
        scale = max(0.0, 1.0 - damping_strength * frac)
        p.velocity_kmpy = p.velocity_kmpy * scale
        # Also damp angular velocity — without it, a plate locked
        # against a neighbour can be stationary in translation but still
        # spinning at its initial rate, which keeps grinding cells off
        # at the boundary and never settles. Damp it MORE aggressively
        # than translation (multiplier ~3) so rotation comes to rest
        # faster than translation under sustained contact.
        ang_scale = max(
            0.0,
            1.0 - damping_strength * sim_config.angular_damping_multiplier * frac)
        p.angular_velocity_rad_per_myr = float(
            p.angular_velocity_rad_per_myr * ang_scale
        )

