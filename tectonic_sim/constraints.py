"""Position-based contact constraints + plate-level velocity damping.

These run *after* ``apply_collisions`` each tick. Collisions handle the
type-dependent resolution (subduction, orogeny thickening, folding mass
transfer); these enforce the geometric rigid-plate behaviour that stops
plates phasing through each other.

Two pieces:

  - ``apply_contact_constraints`` is a position-based-dynamics (PBD)
    relaxation with two passes per iteration:

      1. **Cross-plate separation.** Detect cross-plate pairs within
         ``overlap_radius_km`` and push each pair apart by half the
         overlap depth along the connecting axis. This is the
         no-interpenetration constraint.

      2. **Intra-plate spacing.** Detect *same-plate* pairs closer than
         ``intra_plate_min_distance_km`` and push them apart the same
         way. Models crust incompressibility. Without this pass, the
         cross-plate pass shoves same-plate neighbours arbitrarily
         close, producing visible "force-chain" stripes — a PBD artefact,
         not a tectonic signal. The threshold sits well below the
         Bridson Poisson-disc rest invariant so it only fires on severe
         compression at active collision boundaries.

    Both passes use a Jacobi-style update (deltas accumulated against
    the same input positions inside one iteration, then applied
    together). ``sim_config.contact_iterations`` outer iterations
    converge the two competing constraints. Wrap-aware.

  - ``apply_velocity_damping`` reduces each plate's velocity in
    proportion to how much of it is in collision. The "energy" lost
    this way is the budget for orogeny / mountain-building thickening
    — already accumulated in ``apply_collisions``'s thickness delta.
    Without damping, plates retain their momentum forever and never
    actually stop at contact even with the geometric constraint above.

Together these reproduce the kinematics of a continental collision:
plates resist interpenetration, the leading edges stop, the trailing
edges keep drifting (compressing the plate longitudinally) but without
collapsing into stripes, and the bulk plate velocity gradually drops
as the boundary thickens.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tectonic_sim.spatial import BucketGrid
from tectonic_sim.types import Plate, SimConfig, WorldRect


# -----------------------------------------------------------------------------
# Contact constraints
# -----------------------------------------------------------------------------

def apply_contact_constraints(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    sim_config: SimConfig,
    *,
    iterations: int | None = None,
) -> np.ndarray:
    """Run the PBD relaxation: cross-plate separation + intra-plate spacing.

    For ``sim_config.contact_iterations`` outer passes:
      1. Build a bucket grid on the current positions.
      2. Detect cross-plate pairs within ``overlap_radius_km`` and
         accumulate half-violation separation pushes.
      3. Detect same-plate pairs within ``intra_plate_min_distance_km``
         (if the factor is > 0) and accumulate half-violation pushes.
      4. Apply the combined delta; rewrap under wrap mode.
      5. Stop early if both pair lists are empty.

    Returns a new positions array; the input is not mutated. Pure function
    apart from the usual numpy temporaries — same inputs yield the same
    output, byte-identical.

    Pass count defaults to ``sim_config.contact_iterations``; override via
    the keyword for tests that want to dial in convergence behaviour.
    """
    if iterations is None:
        iterations = sim_config.contact_iterations
    if iterations <= 0 or positions_km.shape[0] == 0:
        return positions_km.copy()

    wrap = (sim_config.boundary_mode == "wrap")
    overlap_radius = sim_config.overlap_radius_km
    intra_min = sim_config.intra_plate_min_distance_km

    positions = positions_km.astype(np.float64, copy=True)

    for _ in range(iterations):
        grid = BucketGrid.build(positions, domain, overlap_radius, wrap=wrap)

        cross_i, cross_j = grid.cross_label_pairs_within(plate_id, overlap_radius)
        if intra_min > 0.0:
            intra_i, intra_j = grid.same_label_pairs_within(plate_id, intra_min)
        else:
            empty = np.zeros(0, dtype=np.int32)
            intra_i, intra_j = empty, empty

        if cross_i.shape[0] == 0 and intra_i.shape[0] == 0:
            break

        # Jacobi-style: both deltas computed against the same input
        # positions in this iteration, then summed and applied together.
        # The outer loop re-builds the grid and re-detects, which is
        # what drives convergence between the competing constraints.
        delta = np.zeros_like(positions)
        _accumulate_separation_push(
            positions, cross_i, cross_j, overlap_radius, domain, wrap, delta,
        )
        _accumulate_separation_push(
            positions, intra_i, intra_j, intra_min, domain, wrap, delta,
        )

        positions = positions + delta
        if wrap:
            positions = domain.wrap_positions(positions)

    return positions


def _accumulate_separation_push(
    positions: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    min_distance_km: float,
    domain: WorldRect,
    wrap: bool,
    delta: np.ndarray,
) -> None:
    """For every pair closer than ``min_distance_km``, accumulate a
    half-violation separation push into ``delta``.

    ``delta`` is mutated in place via ``np.add.at`` / ``np.subtract.at``
    so multiple pairs touching the same particle compose cleanly. The
    direction is the (wrap-aware) unit vector from ``i`` toward ``j``;
    ``i`` moves backward along it, ``j`` forward, each by
    ``0.5 × max(0, min_distance_km - dist)``.
    """
    if pair_i.shape[0] == 0 or min_distance_km <= 0.0:
        return
    dx = positions[pair_j, 0] - positions[pair_i, 0]
    dy = positions[pair_j, 1] - positions[pair_i, 1]
    if wrap:
        dx, dy = domain.wrapped_delta_xy(dx, dy)
    dist = np.hypot(dx, dy)
    # Guard against exact-coincidence pairs (vanishingly rare in float).
    safe = dist > 1e-9
    ux = np.where(safe, dx / np.where(safe, dist, 1.0), 0.0)
    uy = np.where(safe, dy / np.where(safe, dist, 1.0), 0.0)
    half_push = 0.5 * np.maximum(0.0, min_distance_km - dist)
    push_x = ux * half_push
    push_y = uy * half_push
    # i moves opposite the direction (away from j); j moves with it.
    np.subtract.at(delta, pair_i, np.column_stack([push_x, push_y]))
    np.add.at(delta, pair_j, np.column_stack([push_x, push_y]))


# -----------------------------------------------------------------------------
# Velocity damping
# -----------------------------------------------------------------------------

def apply_velocity_damping(
    plates: Sequence[Plate],
    plate_id: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    sim_config: SimConfig,
) -> tuple[Plate, ...]:
    """Reduce each plate's bulk velocity in proportion to its collision exposure.

    For every plate ``p``, let ``fraction`` be the share of its particles
    that appear in at least one cross-plate pair. The plate's new
    velocity is::

        new_velocity = old_velocity × (1 − damping_strength × fraction)

    Pure function. Returns a fresh ``tuple[Plate, ...]`` — the input
    plates are not mutated (``Plate`` is frozen anyway).

    If no pairs are present, the plates come back unchanged.
    """
    plates_tuple = tuple(plates)
    if not plates_tuple or pair_i.shape[0] == 0:
        return plates_tuple

    damping_strength = sim_config.velocity_damping_strength
    if damping_strength <= 0:
        return plates_tuple

    max_id = max(p.id for p in plates_tuple)
    # Particle-side: mark every particle that appears in any pair.
    in_collision = np.zeros(plate_id.shape[0], dtype=bool)
    in_collision[pair_i] = True
    in_collision[pair_j] = True

    total_per_plate = np.bincount(plate_id, minlength=max_id + 1)
    collision_per_plate = np.bincount(
        plate_id[in_collision], minlength=max_id + 1,
    )
    # Avoid divide-by-zero on plates that have lost all their particles.
    fraction = np.where(
        total_per_plate > 0,
        collision_per_plate / np.maximum(total_per_plate, 1),
        0.0,
    )

    new_plates: list[Plate] = []
    for plate in plates_tuple:
        f = float(fraction[plate.id])
        damping = max(0.0, 1.0 - damping_strength * f)
        vx, vy = plate.velocity_kmpy
        new_plates.append(Plate(
            id=plate.id,
            type=plate.type,
            seed_position_km=plate.seed_position_km,
            velocity_kmpy=(vx * damping, vy * damping),
        ))
    return tuple(new_plates)
