"""Per-plate velocity + angular damping."""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL

from tectonic_sim.polygon_sim.types import (
    PolygonPlate)


def _apply_velocity_damping(
    plates: list[PolygonPlate], sim_config,
) -> None:
    """Reduce each plate's bulk velocity by its collision exposure.

    For every plate, the contested cells are split into two classes and
    weighted differently into an *effective* contested fraction::

        eff_frac = (n_other + cc_mult · n_cc) / n_total
        v_new    = v_old × (1 − damping_strength · eff_frac)

    where ``n_cc`` counts this plate's contested cells that are in
    **continental–continental** contention (this plate continental AND
    at least one other continental plate claims the same cell) and
    ``n_other`` is the rest (C-O / O-O). ``cc_mult =
    sim_config.cc_velocity_damping_multiplier`` makes C-C collisions
    bleed off convergence faster — two buoyant continents lock and
    dissipate their kinetic energy into crustal thickening, so the
    larger plate stops over-riding (and annihilating) the smaller one
    before its whole column gets carved and dumped as runaway fold mass.

    Plates not in contention with anything keep their velocity. Call
    after stamping, before contention resolution (resolution clears the
    loser masks at contested cells, so the overlap count has to be
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
    # Per-cell count of continental claimants. A cell with >= 2 is a
    # C-C contention site; any plate continental there is in a C-C
    # collision at that cell.
    crusts = np.stack([p.crust for p in alive], axis=0)
    cont = masks & (crusts == CRUST_CONTINENTAL)
    cont_count = cont.sum(axis=0)
    cc_site = cont_count >= 2
    cc_mult = sim_config.cc_velocity_damping_multiplier
    for k, p in enumerate(alive):
        n_total = int(p.cell_mask.sum())
        if n_total == 0:
            continue
        contested_p = p.cell_mask & contested
        n_contested = int(contested_p.sum())
        if n_contested == 0:
            continue
        # C-C share: cells this plate owns as continental that are C-C
        # contention sites. The rest (C-O from the continental side,
        # oceanic contention) gets the base damping.
        n_cc = int((cont[k] & cc_site).sum())
        n_other = n_contested - n_cc
        eff_frac = (n_other + cc_mult * n_cc) / n_total
        scale = max(0.0, 1.0 - damping_strength * eff_frac)
        p.velocity_kmpy = p.velocity_kmpy * scale
        # Also damp angular velocity — without it, a plate locked
        # against a neighbour can be stationary in translation but still
        # spinning at its initial rate, which keeps grinding cells off
        # at the boundary and never settles. Damp it MORE aggressively
        # than translation (by ``angular_damping_multiplier``) so
        # rotation comes to rest faster than translation under contact.
        # Uses the same effective fraction, so C-C contacts also kill
        # spin faster than C-O/O-O.
        ang_scale = max(
            0.0,
            1.0 - damping_strength * sim_config.angular_damping_multiplier
            * eff_frac)
        p.angular_velocity_rad_per_myr = float(
            p.angular_velocity_rad_per_myr * ang_scale
        )

