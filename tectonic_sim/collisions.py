"""Collision resolution for detected cross-plate particle pairs.

One public function, ``apply_collisions``, takes the current particle
state plus a set of overlap pairs and returns the new state after
resolving each pair. Three pair-resolution rules plus one whole-state
clean-up pass:

  - **Continental + continental.** Both particles thicken by
    ``orogeny_uplift_per_overlap_km``. The lower-thickness one moves a
    small distance toward the higher (folding) and transfers a small
    fraction of its column thickness to the higher one (mass folding).

  - **Oceanic + continental.** The oceanic particle is deleted
    (subducted). The continental survivor thickens by
    ``subduction_arc_uplift_km`` (volcanic-arc analogue: Andes).

  - **Oceanic + oceanic.** The older particle is deleted (denser, sinks).
    The younger survivor thickens by ``subduction_arc_uplift_km``
    (island-arc analogue: Japan, Marianas).

  - **Continental absorption (post-pass).** After the per-pair deltas
    are applied, any continental particle whose new thickness has
    dropped below ``min_continental_thickness_km`` is treated as fully
    absorbed by its over-rider: it is removed from the simulation and
    its remaining mass is added to the nearest cross-plate continental
    neighbour within ``2 × overlap_radius_km``. Geologically: the
    underthruster's leading edge gets fully incorporated into the
    over-rider over many ticks of folding. Requires ``domain`` so the
    spatial-index lookup can run.

Each pair contributes one "overlap event" to the relevant deltas. Per-
particle effects accumulate via ``np.add.at`` for unbuffered scatter so
a particle on a busy boundary correctly receives contributions from
every pair it's in.

The collision constants are **per-overlap-event**, not per-Myr. ``dt_myr``
does not appear here; it enters the sim only via drift (positions move
by ``v·dt``) and aging (``age += dt``). Tuning a faster simulation means
adjusting these constants, not the time step.

Ties: continental folding picks the higher-index particle as the survivor
when thicknesses are exactly equal; oceanic-oceanic subduction picks the
higher-index particle as the survivor when ages are exactly equal. Both
choices are deterministic and matter only on hand-built test inputs (in
production, exact float equality is vanishingly rare).
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.spatial import BucketGrid
from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
)


# Radius (in multiples of overlap_radius_km) within which we search for a
# cross-plate continental over-rider when a thinned particle is being
# absorbed. The over-rider was at exactly overlap_radius_km after the
# contact-constraint pass that produced the thinning; allowing 2× gives
# headroom for it to have drifted slightly.
_ABSORPTION_SEARCH_MULTIPLIER: float = 2.0


def apply_collisions(
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    age_myr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    sim_config: SimConfig,
    *,
    domain: "WorldRect | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve every cross-plate pair and return the new particle state.

    Returns parallel arrays ``(positions, plate_id, crust_type,
    thickness, age)`` after thickness/position deltas have been applied
    and subducted particles have been dropped. Index alignment with the
    input is *not* preserved — the survivor set is a strict subset whose
    indices renumber.

    Empty pair input is still processed by the continental-absorption
    post-pass — a particle thinned by *previous* ticks' folding may now
    be below threshold even if no new pair fires this tick.
    """
    n = positions_km.shape[0]
    # Accumulators — applied at the end so multiple pairs touching the
    # same particle compose cleanly.
    thickness_delta = np.zeros(n, dtype=np.float64)
    position_delta = np.zeros_like(positions_km)
    remove_mask = np.zeros(n, dtype=bool)

    has_pairs = pair_i.shape[0] > 0
    if has_pairs:
        type_i = crust_type[pair_i]
        type_j = crust_type[pair_j]
        cc_mask = (type_i == CRUST_CONTINENTAL) & (type_j == CRUST_CONTINENTAL)
        oo_mask = (type_i == CRUST_OCEANIC) & (type_j == CRUST_OCEANIC)
        oc_mask = ~(cc_mask | oo_mask)   # exactly one oceanic, one continental
    else:
        cc_mask = np.zeros(0, dtype=bool)
        oc_mask = np.zeros(0, dtype=bool)
        oo_mask = np.zeros(0, dtype=bool)

    # ----- CC: orogeny + folding -----
    if cc_mask.any():
        cc_i = pair_i[cc_mask]
        cc_j = pair_j[cc_mask]

        # Both sides thicken by the orogeny constant per pair.
        orogeny = sim_config.orogeny_uplift_per_overlap_km
        np.add.at(thickness_delta, cc_i, orogeny)
        np.add.at(thickness_delta, cc_j, orogeny)

        # Folding: the lower-thickness particle moves toward the higher
        # and transfers folding_ratio of its current thickness to the
        # higher. Ties → keep the higher-index particle as "higher".
        thick_i = thickness_km[cc_i]
        thick_j = thickness_km[cc_j]
        i_is_lower = thick_i < thick_j
        lower = np.where(i_is_lower, cc_i, cc_j)
        higher = np.where(i_is_lower, cc_j, cc_i)

        # Direction: from lower toward higher. Normalised in place;
        # exactly-coincident pairs (vanishingly rare in float) contribute
        # zero displacement. Under a toroidal domain the delta is the
        # wrapped shortest path so folding stays correct across the
        # boundary seam.
        direction = positions_km[higher] - positions_km[lower]
        if domain is not None and sim_config.boundary_mode == "wrap":
            wx, wy = domain.wrapped_delta_xy(direction[:, 0], direction[:, 1])
            direction = np.column_stack([wx, wy])
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        safe = norm > 1e-9
        unit = np.where(safe, direction / np.where(safe, norm, 1.0),
                        np.zeros_like(direction))
        displacement = sim_config.folding_displacement_km
        np.add.at(position_delta, lower, unit * displacement)

        # Mass transfer: lower loses, higher gains the same amount.
        # Capped at the lower's actual thickness so it can't go negative
        # in a single tick.
        fold_mass = np.minimum(
            sim_config.folding_ratio * thickness_km[lower],
            thickness_km[lower],
        )
        np.subtract.at(thickness_delta, lower, fold_mass)
        np.add.at(thickness_delta, higher, fold_mass)

    # ----- OC: subduction (oceanic disappears, continental thickens) -----
    if oc_mask.any():
        oc_i = pair_i[oc_mask]
        oc_j = pair_j[oc_mask]
        i_is_oceanic = (crust_type[oc_i] == CRUST_OCEANIC)
        loser = np.where(i_is_oceanic, oc_i, oc_j)
        survivor = np.where(i_is_oceanic, oc_j, oc_i)

        remove_mask[loser] = True
        arc_uplift = sim_config.subduction_arc_uplift_km
        np.add.at(thickness_delta, survivor, arc_uplift)

    # ----- OO: subduction (older oceanic disappears) -----
    if oo_mask.any():
        oo_i = pair_i[oo_mask]
        oo_j = pair_j[oo_mask]
        age_pair_i = age_myr[oo_i]
        age_pair_j = age_myr[oo_j]
        # Older = colder = denser → subducts.
        # Ties (rare in floats) → drop the higher-index one for determinism.
        i_is_older = age_pair_i > age_pair_j
        loser = np.where(i_is_older, oo_i, oo_j)
        survivor = np.where(i_is_older, oo_j, oo_i)

        remove_mask[loser] = True
        arc_uplift = sim_config.subduction_arc_uplift_km
        np.add.at(thickness_delta, survivor, arc_uplift)

    # Apply accumulators.
    new_positions = positions_km + position_delta
    # Under wrap mode, the folding displacement could push a particle
    # across the rectangle boundary — re-wrap so positions stay canonical.
    if domain is not None and sim_config.boundary_mode == "wrap":
        new_positions = domain.wrap_positions(new_positions)
    new_thickness = thickness_km + thickness_delta

    # Floor thickness so erosion/folding can't send a survivor below zero —
    # biologically these would re-rift, but tracking that is the divergent-
    # fill step's problem; here we just keep the field physical.
    new_thickness = np.maximum(new_thickness, 0.0)

    # ----- Continental absorption -----
    # Any continental particle thinned below the configured minimum is
    # consumed by the nearest cross-plate continental neighbour. Skip when
    # ``domain`` is unavailable (test entry points without spatial context).
    if domain is not None:
        _absorb_thinned_continentals(
            new_positions, plate_id, crust_type, new_thickness,
            remove_mask, sim_config, domain,
        )

    if not remove_mask.any():
        return new_positions, plate_id, crust_type, new_thickness, age_myr

    keep = ~remove_mask
    return (
        new_positions[keep],
        plate_id[keep],
        crust_type[keep],
        new_thickness[keep],
        age_myr[keep],
    )


def _absorb_thinned_continentals(
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    remove_mask: np.ndarray,
    sim_config: SimConfig,
    domain: WorldRect,
) -> None:
    """In-place: for each continental particle below
    ``min_continental_thickness_km``, transfer its remaining mass to the
    nearest cross-plate continental neighbour within
    ``_ABSORPTION_SEARCH_MULTIPLIER × overlap_radius_km`` and mark it for
    removal in ``remove_mask``.

    Particles with no eligible recipient in range are still removed —
    geologically the crust got incorporated *somewhere* (deep mantle
    delamination, lateral extrusion); the model abstracts that as a
    plain deletion when no over-rider is reachable.
    """
    min_thickness = sim_config.min_continental_thickness_km
    continental = (crust_type == CRUST_CONTINENTAL)
    thinned = continental & ~remove_mask & (thickness_km < min_thickness)
    if not thinned.any():
        return

    # Recipients: continental, still alive, *not* already thinned themselves.
    survivor_mask = continental & ~remove_mask & ~thinned
    survivor_idx_global = np.where(survivor_mask)[0]

    if survivor_idx_global.size == 0:
        # No recipients anywhere — just remove the thinned particles.
        remove_mask |= thinned
        return

    survivor_pos = positions_km[survivor_mask]
    survivor_plate = plate_id[survivor_mask]
    search_radius = _ABSORPTION_SEARCH_MULTIPLIER * sim_config.overlap_radius_km
    wrap = (sim_config.boundary_mode == "wrap")
    grid = BucketGrid.build(survivor_pos, domain, search_radius, wrap=wrap)

    for i in np.where(thinned)[0]:
        nearby_local = grid.neighbors_within(
            (float(positions_km[i, 0]), float(positions_km[i, 1])),
            search_radius,
        )
        if nearby_local.size == 0:
            continue
        # Cross-plate filter.
        cross_mask = survivor_plate[nearby_local] != plate_id[i]
        if not cross_mask.any():
            continue
        cross_local = nearby_local[cross_mask]
        # Nearest of the cross-plate survivors (wrap-aware).
        dx = survivor_pos[cross_local, 0] - positions_km[i, 0]
        dy = survivor_pos[cross_local, 1] - positions_km[i, 1]
        if wrap:
            dx, dy = domain.wrapped_delta_xy(dx, dy)
        d2 = dx * dx + dy * dy
        nearest_local = cross_local[int(np.argmin(d2))]
        nearest_global = int(survivor_idx_global[nearest_local])
        # Transfer remaining mass to the over-rider.
        thickness_km[nearest_global] += thickness_km[i]

    # Whether or not transfer happened, every thinned particle is removed.
    remove_mask |= thinned
