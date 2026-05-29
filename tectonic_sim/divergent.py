"""Divergent fill — spawn new particles in any vacant region of the domain.

Per tick, after drift / cull / collision, the domain may contain regions
with no particles inside: spreading boundaries where plates have pulled
apart, subduction wakes where one plate consumed another, and the
trailing edge of plates whose leading particles drifted off-domain. This
module re-populates those vacancies with new particles.

The new particles inherit their **plate id** from the nearest existing
particle (so the territory stays continuous), and their **crust type**
from that plate's *initial* type — not the nearest particle's current
crust type:

  - A continental plate's divergent gap is filled with **rift crust** at
    ``rift_thickness_km`` (East African Rift, Dead Sea analogue).
  - An oceanic plate's divergent gap is filled with **fresh oceanic
    crust** at ``oceanic_thickness_km`` (mid-ocean ridge analogue).

Together with drift and collisions, this closes the loop on conservation
of *territory*: a plate stays roughly the same size over time because
particles continuously replenish at its trailing edge. The bulk plate
identity moves, but no plate ever shrinks to nothing through drift alone.

Algorithm:

  1. Lay out a regular grid of candidate spawn positions at one
     particle-spacing per cell. Jitter each candidate by ±0.3 spacings to
     prevent grid-aligned spawn artifacts accumulating over many ticks.
  2. Cull candidates that landed outside the domain.
  3. Build a bucket grid over existing particles. For each candidate,
     run a single broader-radius query and decide three things from it:
       - **Vacancy.** If any existing particle is within
         ``particle_spacing_km`` (the Bridson Poisson-disc invariant),
         reject — the cell isn't actually vacant.
       - **Inheritance.** Pick the nearest particle in the broader set;
         its plate id is what the new particle inherits. Skip the
         candidate entirely if the broader set is empty (deep
         isolation — only happens on a nearly empty domain).
       - **Contact-gap gate.** Reject the candidate if any
         *foreign-plate* particle sits within ``overlap_radius_km``.
         A genuine rift has the same plate's particles within broader
         radius but no foreign-plate particles anywhere near; the
         contact-constraint gap, by contrast, has the foreign plate
         right across the boundary. Without this gate the divergent
         fill would repopulate the contact band with arbitrary-plate
         particles, perpetuating intermixing.
  4. Spawn at each accepted candidate, set crust type and thickness
     from the inherited plate's initial type, age = 0.

The per-candidate loop is pure Python over ~``(W·H / spacing²)``
candidates (~4,500 for the default 1000×1000 km at 15 km), each
back-ended by an O(1) bucket-grid query. Fast in practice.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tectonic_sim.spatial import BucketGrid
from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    Plate,
    SimConfig,
    WorldRect,
)


# Multiplier for the nearest-neighbour broader-search radius when a
# candidate's neighbourhood is empty. Need this larger than spacing so
# we can locate a plate to inherit from even in mid-rift situations
# where particles have pulled back from the gap.
_NEAREST_SEARCH_MULTIPLIER: float = 4.0


def _plate_initial_type_table(plates: Sequence[Plate]) -> np.ndarray:
    """Return an array indexed by plate id holding each plate's initial
    crust type code. Plate ids may be sparse after subductions; the
    array is sized to ``max_id + 1`` with zeros in unused slots."""
    if not plates:
        return np.zeros(0, dtype=np.int8)
    max_id = max(p.id for p in plates)
    table = np.zeros(max_id + 1, dtype=np.int8)
    for p in plates:
        table[p.id] = (
            CRUST_CONTINENTAL if p.type == "continental" else CRUST_OCEANIC
        )
    return table


def divergent_fill(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    age_myr: np.ndarray,
    plates: Sequence[Plate],
    sim_config: SimConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spawn new particles in vacant cells; inherit plate id from the
    nearest existing particle, crust type from that plate's initial type.

    Returns extended ``(positions, plate_id, crust_type, thickness,
    age)`` arrays — same dtype and shape contract as the inputs. The
    new particles are appended at the tail.

    Behaviour on empty input (no existing particles): a no-op pass-through.
    Inheriting plate id requires *some* existing particle; with none, the
    fill has nothing to extrapolate from.
    """
    if positions_km.shape[0] == 0:
        return positions_km, plate_id, crust_type, thickness_km, age_myr

    spacing = sim_config.particle_spacing_km
    hw = domain.half_width_km
    hh = domain.half_height_km
    wrap = (sim_config.boundary_mode == "wrap")

    # --- 1) Candidate grid -------------------------------------------------
    n_cols = max(1, int(np.ceil(domain.width_km / spacing)))
    n_rows = max(1, int(np.ceil(domain.height_km / spacing)))
    xs = -hw + spacing * (np.arange(n_cols) + 0.5)
    ys = -hh + spacing * (np.arange(n_rows) + 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    candidates = np.column_stack([gx.ravel(), gy.ravel()])

    # Jitter ±0.3 spacing on each axis so a sustained fill doesn't carve
    # a regular grid signature into the field.
    jitter = rng.uniform(-0.3 * spacing, 0.3 * spacing, size=candidates.shape)
    candidates = candidates + jitter

    if wrap:
        # Under torus, wrap any jittered candidate back into the domain.
        candidates = domain.wrap_positions(candidates)
    else:
        # Under open, drop candidates that jittered outside.
        inside = (
            (candidates[:, 0] >= -hw)
            & (candidates[:, 0] <= hw)
            & (candidates[:, 1] >= -hh)
            & (candidates[:, 1] <= hh)
        )
        candidates = candidates[inside]
    if candidates.shape[0] == 0:
        return positions_km, plate_id, crust_type, thickness_km, age_myr

    # --- 2) Vacancy check + nearest-neighbour inheritance + contact gate ---
    # Cell size = overlap_radius (still appropriate for spatial queries —
    # the largest query inside the candidate loop is broader_radius).
    # Wrap flag mirrors the configured boundary mode.
    grid = BucketGrid.build(
        positions_km, domain, sim_config.overlap_radius_km, wrap=wrap,
    )
    broader_radius = _NEAREST_SEARCH_MULTIPLIER * spacing

    # Three distance thresholds, decided per candidate from the same broader
    # query:
    #   - ``vacancy_threshold`` = particle_spacing_km (= Bridson invariant).
    #     A candidate is "vacant" only if no existing particle is within
    #     this distance. Matches initial-state density so divergent zones
    #     don't end up systematically sparser than the seeded cloud.
    #   - ``contact_threshold`` = overlap_radius_km. A candidate sitting
    #     inside the contact-constraint gap (the ~overlap_radius-wide band
    #     between two cross-plate fronts) has a foreign-plate particle
    #     within this radius. We refuse to spawn there — the gap is a
    #     transient constraint artefact, not a true rift.
    #   - ``broader_radius`` = 4 × spacing. Used to find the nearest
    #     existing particle for plate-id inheritance.
    vacancy_threshold = sim_config.particle_spacing_km
    contact_threshold = sim_config.overlap_radius_km
    vacancy2 = vacancy_threshold * vacancy_threshold
    contact2 = contact_threshold * contact_threshold

    accepted_positions: list[tuple[float, float]] = []
    accepted_nearest: list[int] = []
    for cand in candidates:
        cx, cy = float(cand[0]), float(cand[1])
        broader = grid.neighbors_within((cx, cy), broader_radius)
        if broader.size == 0:
            # Truly isolated — nothing close enough to inherit a plate
            # from. Skip; the divergent fill needs an anchor.
            continue
        # Compute distances once and reuse them for all three decisions.
        bx = positions_km[broader, 0] - cx
        by = positions_km[broader, 1] - cy
        if wrap:
            bx, by = domain.wrapped_delta_xy(bx, by)
        d2 = bx * bx + by * by

        # Vacancy: no existing particle within particle_spacing_km.
        if (d2 <= vacancy2).any():
            continue

        # Inheritance: nearest existing particle in the broader set.
        nearest_local = int(np.argmin(d2))
        nearest_idx = int(broader[nearest_local])
        inherited_plate = int(plate_id[nearest_idx])

        # Contact-gap gate: refuse to spawn if any *foreign-plate* particle
        # sits within overlap_radius. A genuine rift has the same plate's
        # particles within broader_radius but no foreign-plate particles
        # in close range; the contact gap, by contrast, has the foreign
        # plate exactly overlap_radius away.
        in_contact = d2 <= contact2
        if in_contact.any():
            contact_plates = plate_id[broader[in_contact]]
            if (contact_plates != inherited_plate).any():
                continue

        accepted_positions.append((cx, cy))
        accepted_nearest.append(nearest_idx)

    if not accepted_positions:
        return positions_km, plate_id, crust_type, thickness_km, age_myr

    new_positions = np.asarray(accepted_positions, dtype=np.float64)
    nearest_arr = np.asarray(accepted_nearest, dtype=np.int32)

    # --- 3) Inherit plate id, derive crust type / thickness ---------------
    new_plate_id = plate_id[nearest_arr].astype(np.int32)

    initial_type = _plate_initial_type_table(plates)
    new_crust_type = initial_type[new_plate_id]

    # Rift crust for continental-plate divergent fill; fresh oceanic for
    # oceanic-plate divergent fill.
    new_thickness = np.where(
        new_crust_type == CRUST_CONTINENTAL,
        sim_config.rift_thickness_km,
        sim_config.oceanic_thickness_km,
    ).astype(np.float64)

    new_age = np.zeros(new_positions.shape[0], dtype=np.float64)

    return (
        np.vstack([positions_km, new_positions]),
        np.concatenate([plate_id, new_plate_id]),
        np.concatenate([crust_type, new_crust_type]),
        np.concatenate([thickness_km, new_thickness]),
        np.concatenate([age_myr, new_age]),
    )
