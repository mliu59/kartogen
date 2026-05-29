"""Initial particle layout for the tectonic sim.

Three steps, each its own pure function:

  1. ``poisson_disc_sample`` — fills the world rectangle with particle
     positions at a target minimum spacing (Bridson 2007).
  2. ``place_plate_seeds`` — picks the K plate seed positions with optional
     radial bias toward centre or edge, plus a minimum separation so two
     plates can't start on top of each other.
  3. ``assign_particles_to_plates`` — Voronoi assignment of each particle
     to the nearest plate seed.

``build_initial_state`` glues them together and returns the parallel
arrays that the rest of the sim consumes: positions, plate ids, crust
types, thicknesses, ages. Pure function of ``(domain, sim_config, seed)``.
"""

from __future__ import annotations

import math

import numpy as np

from tectonic_sim._log import get_logger
from tectonic_sim.rng import RngStream
from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    Plate,
    SimConfig,
    WorldRect,
    crust_type_name,
)

_log = get_logger("seeding")


# -----------------------------------------------------------------------------
# Poisson-disc sampling (Bridson 2007)
# -----------------------------------------------------------------------------

# A Bridson background grid stores at most one point per cell of side
# ``r / sqrt(2)``. Each candidate's neighbourhood check is then O(1) — we
# only need to inspect the surrounding ``ceil(2)`` cells in each axis,
# which is at most a 5×5 window of cells.
_BRIDSON_GRID_SCALE = math.sqrt(2.0)

# Bridson recommends k=30 candidates per active point.
_BRIDSON_CANDIDATES_PER_ACTIVE = 30


def poisson_disc_sample(
    domain: WorldRect,
    spacing_km: float,
    rng: np.random.Generator,
    *,
    wrap: bool = False,
) -> np.ndarray:
    """Bridson Poisson-disc sampling of a centred rectangle.

    Returns an ``(N, 2)`` float64 array of positions in km, where every
    pair of points is at least ``spacing_km`` apart. Density works out
    to ~one point per ``(π · spacing_km² / 4)`` km², so a 1000×1000 km
    world at 15 km spacing yields ~5,700 points.

    ``wrap=True`` makes both axes wrap as a torus: candidate placements
    near the boundary check distance to existing points modulo the
    domain period, so the resulting field has no spacing-gap seam at
    the rectangle edges.
    """
    if spacing_km <= 0:
        raise ValueError(f"spacing_km must be > 0, got {spacing_km}")

    hw = domain.half_width_km
    hh = domain.half_height_km
    cell = spacing_km / _BRIDSON_GRID_SCALE
    cols = int(math.ceil(domain.width_km / cell))
    rows = int(math.ceil(domain.height_km / cell))

    # Background grid: -1 = empty, else index into ``points``.
    grid = np.full((cols, rows), -1, dtype=np.int32)

    def _grid_index(x: float, y: float) -> tuple[int, int]:
        ci = int((x + hw) / cell)
        ri = int((y + hh) / cell)
        if wrap:
            return ci % cols, ri % rows
        return min(max(ci, 0), cols - 1), min(max(ri, 0), rows - 1)

    def _in_bounds(x: float, y: float) -> bool:
        # Wrap mode keeps every candidate by remapping it back into the
        # rectangle; non-wrap clips to the rectangle.
        return -hw <= x <= hw and -hh <= y <= hh

    def _too_close(x: float, y: float) -> bool:
        ci, ri = _grid_index(x, y)
        for dci in range(-2, 3):
            for dri in range(-2, 3):
                if wrap:
                    cc = (ci + dci) % cols
                    rr = (ri + dri) % rows
                else:
                    cc = ci + dci
                    rr = ri + dri
                    if not (0 <= cc < cols and 0 <= rr < rows):
                        continue
                idx = grid[cc, rr]
                if idx < 0:
                    continue
                px, py = points[idx]
                dx = px - x
                dy = py - y
                if wrap:
                    dx, dy = domain.wrapped_delta_xy(dx, dy)
                if dx * dx + dy * dy < spacing_km * spacing_km:
                    return True
        return False

    points: list[tuple[float, float]] = []
    active: list[int] = []

    x0 = float(rng.uniform(-hw, hw))
    y0 = float(rng.uniform(-hh, hh))
    points.append((x0, y0))
    ci, ri = _grid_index(x0, y0)
    grid[ci, ri] = 0
    active.append(0)

    while active:
        ai = int(rng.integers(0, len(active)))
        parent_idx = active[ai]
        px, py = points[parent_idx]

        radii = rng.uniform(spacing_km, 2.0 * spacing_km, _BRIDSON_CANDIDATES_PER_ACTIVE)
        thetas = rng.uniform(0.0, 2.0 * math.pi, _BRIDSON_CANDIDATES_PER_ACTIVE)
        accepted = False
        for r, th in zip(radii, thetas):
            cx = px + r * math.cos(th)
            cy = py + r * math.sin(th)
            if wrap:
                # Wrap back into the rectangle.
                cx = (cx + hw) % domain.width_km - hw
                cy = (cy + hh) % domain.height_km - hh
            elif not _in_bounds(cx, cy):
                continue
            if _too_close(cx, cy):
                continue
            new_idx = len(points)
            points.append((cx, cy))
            ci, ri = _grid_index(cx, cy)
            grid[ci, ri] = new_idx
            active.append(new_idx)
            accepted = True
            break
        if not accepted:
            active[ai] = active[-1]
            active.pop()

    return np.asarray(points, dtype=np.float64)


# -----------------------------------------------------------------------------
# Plate seed placement
# -----------------------------------------------------------------------------

def _plate_seed_min_separation_km(
    domain: WorldRect, plate_count: int, particle_spacing_km: float,
) -> float:
    """Derived minimum separation between plate seeds.

    Heuristic: scale with ``min(domain dimensions) / (sqrt(N) + 1)`` so
    plates spread out as ``N`` grows, with a floor of two particle
    spacings so adjacent plates don't seed inside each other's Voronoi
    cells.
    """
    base = min(domain.width_km, domain.height_km) / (math.sqrt(plate_count) + 1.0)
    floor = 2.0 * particle_spacing_km
    return max(base, floor)


def place_plate_seeds(
    domain: WorldRect,
    sim_config: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place ``plate_count`` seed positions in the domain.

    Returns an ``(N, 2)`` array of (x, y) km. Uses rejection sampling
    with a derived minimum separation and an optional radial bias:

      - ``seed_radial_bias > 0`` → prefer points near the world centre
      - ``seed_radial_bias < 0`` → prefer points near any edge
      - ``seed_radial_bias == 0`` → uniform

    Under ``boundary_mode = "wrap"`` the min-separation check uses the
    toroidal shortest-path distance, so two seeds that land on opposite
    edges (e.g. x ≈ -hw and x ≈ +hw) are correctly recognised as wrap-
    neighbours and one is rejected. Without this, plates would routinely
    spawn within a few km of each other across the seam.

    The radial bias still uses rectangle-coordinate distance to a labelled
    "centre" point — on a torus every point is equivalent, but biasing
    toward the world-coordinate origin is what users configure via
    ``seed_radial_bias`` and we keep that semantic.

    If the rejection sampler can't place all seeds, the separation is
    relaxed and we retry — keeps the function total under user-tuned
    densities. Falls back to the closest ``N`` candidates if even relaxed.
    """
    n = sim_config.plate_count
    if n <= 0:
        raise ValueError(f"plate_count must be > 0, got {n}")

    hw = domain.half_width_km
    hh = domain.half_height_km
    min_sep = _plate_seed_min_separation_km(
        domain, n, sim_config.particle_spacing_km,
    )
    bias = sim_config.seed_radial_bias
    wrap = (sim_config.boundary_mode == "wrap")

    # Oversample candidates and pick by bias.
    pool_size = max(8 * n, 128)
    xs = rng.uniform(-hw, hw, pool_size)
    ys = rng.uniform(-hh, hh, pool_size)

    if bias != 0.0:
        # Normalised distance to nearest edge, in [0, 1] (0 = at edge,
        # 1 = at centre). Symmetric in x and y.
        d_to_edge = np.minimum(hw - np.abs(xs), hh - np.abs(ys))
        d_max = min(hw, hh)
        centre_score = np.clip(d_to_edge / d_max, 0.0, 1.0)
        # Sort key: bias × centre_score + jitter. Larger key = preferred.
        jitter = rng.uniform(0.0, 0.15, pool_size)
        key = bias * centre_score + jitter
        order = np.argsort(-key)  # largest first
    else:
        order = rng.permutation(pool_size)

    accepted: list[tuple[float, float]] = []
    separation = min_sep
    for _ in range(6):  # at most a few relaxation passes
        accepted.clear()
        sep2 = separation * separation
        for i in order:
            x, y = float(xs[i]), float(ys[i])
            ok = True
            for ax, ay in accepted:
                dx, dy = ax - x, ay - y
                if wrap:
                    dx, dy = domain.wrapped_delta_xy(dx, dy)
                if dx * dx + dy * dy < sep2:
                    ok = False
                    break
            if ok:
                accepted.append((x, y))
                if len(accepted) == n:
                    return np.asarray(accepted, dtype=np.float64)
        separation *= 0.7

    # Fallback: take the first N in `order`, separation be damned. This
    # only triggers on extreme configs (e.g. plate_count > pool_size).
    _log.warning(
        "place_plate_seeds: %d plates didn't fit even after relaxation; "
        "falling back to unrestricted top-N", n,
    )
    chosen = order[:n]
    return np.stack([xs[chosen], ys[chosen]], axis=1)


def _random_unit_vector(rng: np.random.Generator) -> tuple[float, float]:
    """A single unit vector with uniformly-distributed direction."""
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    return math.cos(theta), math.sin(theta)


def build_plates(
    seed_positions: np.ndarray,
    sim_config: SimConfig,
    rng: RngStream,
) -> tuple[Plate, ...]:
    """Build the ``Plate`` dataclass tuple from seed positions.

    Each plate gets an independent type-draw and motion-direction draw
    (own child RNG, keyed on plate id), so adding or reordering plates
    later doesn't reshuffle the others' velocities or types.
    """
    plates: list[Plate] = []
    for i, (sx, sy) in enumerate(seed_positions):
        type_rng = rng.child("seeding", "plate", i, "type")
        ptype = (
            "continental"
            if float(type_rng.uniform(0.0, 1.0)) < sim_config.continental_fraction
            else "oceanic"
        )

        motion_rng = rng.child("seeding", "plate", i, "motion")
        ux, uy = _random_unit_vector(motion_rng)
        speed = sim_config.motion_speed_kmpy
        plates.append(Plate(
            id=i,
            type=ptype,
            seed_position_km=(float(sx), float(sy)),
            velocity_kmpy=(ux * speed, uy * speed),
        ))
    return tuple(plates)


# -----------------------------------------------------------------------------
# Particle → plate Voronoi assignment
# -----------------------------------------------------------------------------

def assign_particles_to_plates(
    particle_positions_km: np.ndarray,
    plate_seed_positions_km: np.ndarray,
    *,
    domain: WorldRect | None = None,
    wrap: bool = False,
) -> np.ndarray:
    """Assign each particle to the nearest plate seed (Voronoi).

    Vectorised: builds the full ``(N, P)`` squared-distance matrix and
    takes argmin per row. For our scales (N ≤ ~50k, P ≤ ~20), this is
    fine and ~10× simpler than a KDtree call.

    When ``wrap=True`` (and ``domain`` is provided), distances use the
    toroidal shortest-path metric, so a particle near one edge can be
    assigned to a plate whose seed sits across the seam. Without wrap-
    aware Voronoi, plates near the boundary get hard-clipped by the
    rectangle, producing perpendicular plate edges along the world
    border — an obvious artefact since the rest of the sim (collisions,
    drift, divergent fill) is already toroidal.

    Returns an ``(N,)`` int32 array of plate ids.
    """
    # (N, 1, 2) - (1, P, 2) → (N, P, 2)
    diff = particle_positions_km[:, None, :] - plate_seed_positions_km[None, :, :]
    if wrap:
        if domain is None:
            raise ValueError("wrap=True requires domain to be provided")
        wx, wy = domain.wrapped_delta_xy(diff[..., 0], diff[..., 1])
        sqd = wx * wx + wy * wy
    else:
        sqd = np.einsum("npc,npc->np", diff, diff)
    return np.argmin(sqd, axis=1).astype(np.int32)


# -----------------------------------------------------------------------------
# Top-level: build the initial state
# -----------------------------------------------------------------------------

def build_initial_state(
    domain: WorldRect,
    sim_config: SimConfig,
    seed: int,
) -> tuple[
    tuple[Plate, ...],   # plates
    np.ndarray,          # particle_position_km   (N, 2) float64
    np.ndarray,          # particle_plate_id      (N,)   int32
    np.ndarray,          # particle_crust_type    (N,)   int8
    np.ndarray,          # particle_thickness_km  (N,)   float64
    np.ndarray,          # particle_age_myr       (N,)   float64
]:
    """Build everything ``simulate()`` needs to start at t=0.

    Pure function of ``(domain, sim_config, seed)``.
    """
    rng = RngStream(seed)
    wrap = (sim_config.boundary_mode == "wrap")

    # 1) Particle positions via Poisson-disc.
    positions = poisson_disc_sample(
        domain,
        sim_config.particle_spacing_km,
        rng.child("seeding", "poisson"),
        wrap=wrap,
    )

    # 2) Plate seeds.
    plate_seeds = place_plate_seeds(
        domain,
        sim_config,
        rng.child("seeding", "plate_seeds"),
    )

    # 3) Build Plate objects.
    plates = build_plates(plate_seeds, sim_config, rng)

    # 4) Voronoi-assign particles to plates. Under wrap, use the toroidal
    # shortest-path metric so plates near the rectangle boundary inherit
    # natural curved edges across the seam instead of being clipped to
    # perpendicular straight lines along the world border.
    plate_id = assign_particles_to_plates(
        positions, plate_seeds, domain=domain, wrap=wrap,
    )

    # 5) Per-particle initial fields, derived from plate type.
    plate_type_codes = np.array(
        [CRUST_CONTINENTAL if p.type == "continental" else CRUST_OCEANIC
         for p in plates],
        dtype=np.int8,
    )
    crust_type = plate_type_codes[plate_id]

    cont_thick = sim_config.continental_thickness_km
    ocn_thick = sim_config.oceanic_thickness_km
    thickness = np.where(
        crust_type == CRUST_CONTINENTAL, cont_thick, ocn_thick,
    ).astype(np.float64)

    age = np.zeros(positions.shape[0], dtype=np.float64)

    _log.info(
        "initial state: %d particles across %d plates (%s)",
        positions.shape[0],
        len(plates),
        ", ".join(f"#{p.id}:{p.type[:4]}" for p in plates),
    )

    return plates, positions, plate_id, crust_type, thickness, age
