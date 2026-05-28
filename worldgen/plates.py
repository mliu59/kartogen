"""Plate-tectonics layer: static Voronoi plates with boundary classification.

This is the macro layer that determines where continents and oceans live, and
where mountain ranges and rifts run. It runs once per world (deterministic in
the seed) and feeds two things into the rest of the pipeline:

1. A per-hex elevation bias (continental/oceanic baseline + boundary uplift /
   depression) that the elevation layer adds to its fBm detail noise.
2. Per-hex plate metadata (plate_id, plate_type, nearest_boundary_type,
   distance_to_boundary_km) surfaced on HexData for the inspector and any
   downstream consumer that cares about geology.

The model is intentionally static — there is no time-simulated plate motion.
Each plate gets a random motion vector at generation time, and the relative
motion of two plates is used purely to *classify* the boundary between them
(convergent / divergent / transform). No iteration loop, no accumulated
deformation. This trades realism for clarity and reproducibility.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from worldgen.noise import PerlinNoise2D
from worldgen.types import PlateConfig

PLATE_TYPE_CONTINENTAL = "continental"
PLATE_TYPE_OCEANIC = "oceanic"

BOUNDARY_CC_CONVERGENT = "cc_convergent"
BOUNDARY_OC_CONVERGENT = "oc_convergent"
BOUNDARY_OO_CONVERGENT = "oo_convergent"
BOUNDARY_DIVERGENT = "divergent"
BOUNDARY_TRANSFORM = "transform"


@dataclass(frozen=True, slots=True)
class Plate:
    """One tectonic plate."""

    id: int
    seed_hex: Hex
    type: str  # PLATE_TYPE_CONTINENTAL | PLATE_TYPE_OCEANIC
    motion: tuple[float, float]  # cartesian (dx, dy), unit-ish (scaled by motion_speed)
    baseline_elevation: float


@dataclass(frozen=True)
class PlateField:
    """All output of the plates layer.

    ``hex_to_plate`` maps every hex in the world to its plate id.
    ``distance_by_type`` is the primary distance structure: for each
    boundary type, a per-hex BFS distance (km) to the nearest boundary
    of that type. This lets ``plate_elevation_bias`` *sum* contributions
    from every nearby boundary instead of inheriting a single type via
    BFS — eliminating the discontinuities you'd otherwise get where two
    different boundary types meet (e.g. a cc_convergent +0.65 contribution
    abruptly switching to a divergent −0.30 contribution one hex away).

    ``boundary_type`` and ``distance_to_boundary_km`` are the per-hex
    "nearest single boundary" view, derived from ``distance_by_type``,
    kept for ``HexData`` interpretability and the plates renderer.
    """

    plates: tuple[Plate, ...]
    hex_to_plate: dict[Hex, int]
    # Smoothly blended baseline per hex. Far from any plate boundary this
    # equals the hex's plate's ``baseline_elevation``; near a boundary it
    # transitions toward the neighbor plate's baseline over a smoothstep
    # of width ``baseline_blend_km``. Replaces the hard-step "every hex
    # gets its plate's baseline" approach, which produced visible elevation
    # discontinuities across plate boundaries (the "stitched plates" look).
    hex_baseline: dict[Hex, float]
    distance_by_type: dict[str, dict[Hex, float]]
    boundary_type: dict[Hex, str | None]
    distance_to_boundary_km: dict[Hex, float]


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Same projection the elevation layer uses; kept in sync intentionally."""
    return 1.5 * h.q, math.sqrt(3.0) * (h.r + h.q / 2.0)


def _hex_distance_km(a: Hex, b: Hex, hex_size_km: float) -> float:
    """Approximate cartesian distance between two hex centers, in km."""
    ax, ay = _hex_to_xy(a)
    bx, by = _hex_to_xy(b)
    return math.hypot(ax - bx, ay - by) * hex_size_km


def _place_seeds(
    hexes: list[Hex],
    radius: int,
    config: PlateConfig,
    hex_size_km: float,
    rng,
) -> list[Hex]:
    """Place plate seeds via rejection sampling.

    Iterates a shuffled hex list; accepts a hex as a seed if it is at least
    ``min_separation_km`` from every previously-accepted seed. If we cannot
    place ``count`` seeds before exhausting the hex list, the separation is
    relaxed and we try again — this keeps the function total rather than
    raising on a config the user can't easily diagnose.
    """
    # Sort hexes by radial bias: positive bias prefers center, negative prefers edge.
    # We do this by giving each hex a sort key = bias * normalized_radial_distance
    # then choosing seeds in that order (with a random jitter).
    candidates = list(hexes)
    if config.seed_radial_bias != 0.0 and radius > 0:
        # Smaller key = chosen first
        def radial_key(h: Hex) -> float:
            d = max(abs(h.q), abs(h.r), abs(h.s)) / radius  # 0 = center, 1 = edge
            jitter = rng.random() * 0.15  # small noise so ties break randomly
            # Positive bias → center first → key ~ d; negative bias → edge first → key ~ -d
            return config.seed_radial_bias * d + jitter
        candidates.sort(key=radial_key)
    else:
        rng.shuffle(candidates)

    separation = config.min_separation_km
    for _ in range(8):  # at most a few relaxations
        seeds: list[Hex] = []
        for h in candidates:
            if all(_hex_distance_km(h, s, hex_size_km) >= separation for s in seeds):
                seeds.append(h)
                if len(seeds) == config.count:
                    return seeds
        separation *= 0.7  # relax and try again
    # Final fallback: just take the first `count` shuffled candidates regardless.
    return candidates[: config.count]


def _classify_plate_type(continental_fraction: float, rng) -> str:
    return PLATE_TYPE_CONTINENTAL if rng.random() < continental_fraction else PLATE_TYPE_OCEANIC


def _random_unit_vector(rng) -> tuple[float, float]:
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return math.cos(theta), math.sin(theta)


def _assign_hex_to_plate_and_baseline(
    h: Hex,
    seeds_xy: list[tuple[int, float, float]],
    plates_by_id: dict[int, Plate],
    warp_x: PerlinNoise2D,
    warp_y: PerlinNoise2D,
    warp_freq: float,
    warp_strength_hex: float,
    blend_cart: float,
) -> tuple[int, float]:
    """Return ``(plate_id, blended_baseline)`` for hex ``h``.

    Plate assignment is nearest-warped-seed (same as before). The baseline
    is a soft-Voronoi weighted average over every plate whose seed lies
    within ``blend_cart`` units of the nearest seed — at the seed itself
    only one plate contributes, at a plate boundary two plates blend 50/50,
    and at a triple junction all three blend ~33/33/33. Smoothstep over
    the margin (= distance_to_plate − distance_to_nearest) gives zero
    derivative at both endpoints, so the field is C¹-smooth.
    """
    x, y = _hex_to_xy(h)
    wx = warp_x.sample(x * warp_freq, y * warp_freq) * warp_strength_hex
    wy = warp_y.sample(x * warp_freq + 11.1, y * warp_freq + 5.3) * warp_strength_hex
    qx, qy = x + wx, y + wy

    # Single pass: track nearest plate + all squared distances for the blend.
    best_id = seeds_xy[0][0]
    best_d2 = (qx - seeds_xy[0][1]) ** 2 + (qy - seeds_xy[0][2]) ** 2
    dists_sq: list[tuple[int, float]] = [(seeds_xy[0][0], best_d2)]
    for pid, sx, sy in seeds_xy[1:]:
        d2 = (qx - sx) ** 2 + (qy - sy) ** 2
        dists_sq.append((pid, d2))
        if d2 < best_d2:
            best_d2 = d2
            best_id = pid

    # Soft-Voronoi blend. Convert to linear distances; weight = smoothstep on
    # (1 − margin/blend_cart), so weight=1 at margin=0 and weight=0 at margin≥blend.
    if blend_cart <= 0.0:
        # Disabled → hard step, equivalent to the old behavior.
        return best_id, plates_by_id[best_id].baseline_elevation
    d_min = math.sqrt(best_d2)
    weight_sum = 0.0
    baseline_sum = 0.0
    for pid, d2 in dists_sq:
        d = math.sqrt(d2)
        margin = d - d_min
        if margin >= blend_cart:
            continue
        t = margin / blend_cart  # 0 at nearest, 1 at far edge of blend
        # Smoothstep: 1 at t=0, 0 at t=1, zero derivative at both ends.
        w = 1.0 - t * t * (3.0 - 2.0 * t)
        weight_sum += w
        baseline_sum += w * plates_by_id[pid].baseline_elevation
    baseline = (
        baseline_sum / weight_sum
        if weight_sum > 0.0
        else plates_by_id[best_id].baseline_elevation
    )
    return best_id, baseline


def _classify_boundary(
    plate_a: Plate,
    plate_b: Plate,
    threshold: float,
) -> str:
    """Classify the boundary between two adjacent plates.

    The inter-plate normal is the unit vector from A's seed to B's seed. The
    sign of (B.motion - A.motion) · normal tells us whether the plates are
    closing (negative — convergent) or opening (positive — divergent) along
    their shared boundary. Magnitude below ``threshold`` is transform.
    """
    ax, ay = _hex_to_xy(plate_a.seed_hex)
    bx, by = _hex_to_xy(plate_b.seed_hex)
    nx, ny = bx - ax, by - ay
    n_norm = math.hypot(nx, ny)
    if n_norm == 0:
        return BOUNDARY_TRANSFORM
    nx, ny = nx / n_norm, ny / n_norm
    rel_x = plate_b.motion[0] - plate_a.motion[0]
    rel_y = plate_b.motion[1] - plate_a.motion[1]
    projection = rel_x * nx + rel_y * ny
    if projection > threshold:
        return BOUNDARY_DIVERGENT
    if projection < -threshold:
        # Convergent — refine by plate-type pair
        a_cont = plate_a.type == PLATE_TYPE_CONTINENTAL
        b_cont = plate_b.type == PLATE_TYPE_CONTINENTAL
        if a_cont and b_cont:
            return BOUNDARY_CC_CONVERGENT
        if not a_cont and not b_cont:
            return BOUNDARY_OO_CONVERGENT
        return BOUNDARY_OC_CONVERGENT
    return BOUNDARY_TRANSFORM


_BOUNDARY_TYPES: tuple[str, ...] = (
    BOUNDARY_CC_CONVERGENT,
    BOUNDARY_OC_CONVERGENT,
    BOUNDARY_OO_CONVERGENT,
    BOUNDARY_DIVERGENT,
    BOUNDARY_TRANSFORM,
)


def _bfs_distance_by_type(
    hexes: list[Hex],
    hex_to_plate: dict[Hex, int],
    plates: tuple[Plate, ...],
    threshold: float,
    hex_size_km: float,
) -> tuple[
    dict[str, dict[Hex, float]],
    dict[Hex, str | None],
    dict[Hex, float],
]:
    """Compute per-boundary-type distance maps and the per-hex nearest-type.

    For each boundary type, runs a multi-source BFS from all boundary hexes
    classified as that type. Returns:

    - ``distance_by_type[type][hex]`` — BFS distance in km to the nearest
      boundary of ``type``; ``math.inf`` if none exists in the world or none
      reachable.
    - ``boundary_type[hex]`` — the single type whose distance is smallest at
      this hex (the "dominant" type used for inspector / renderer).
    - ``distance_to_boundary_km[hex]`` — the minimum over all types.

    The previous design propagated a single boundary type via BFS, so a hex
    deep in a plate inherited *one* type's amplitude. At Y-junctions where
    boundaries of different types meet, adjacent hexes could inherit
    different types, producing a visible step in elevation (e.g. cc_conv
    +0.65 next to divergent −0.30). Computing per-type distance maps lets
    ``plate_elevation_bias`` sum smooth, decayed contributions from *all*
    nearby boundary types, removing those steps.
    """
    hex_set = set(hexes)
    plate_by_id = {p.id: p for p in plates}

    # 1. Classify every boundary hex by all boundary types it participates
    # in (a single hex on a triple-junction may sit on more than one type;
    # we record it as a source for each).
    sources_by_type: dict[str, list[Hex]] = {t: [] for t in _BOUNDARY_TYPES}
    dominant_type: dict[Hex, str | None] = {h: None for h in hexes}

    for h in hexes:
        pid = hex_to_plate[h]
        plate_a = plate_by_id[pid]
        types_here: set[str] = set()
        best_type: str | None = None
        best_priority = -1
        for nb in h.neighbors():
            if nb not in hex_set:
                continue
            nb_pid = hex_to_plate[nb]
            if nb_pid == pid:
                continue
            plate_b = plate_by_id[nb_pid]
            btype = _classify_boundary(plate_a, plate_b, threshold)
            types_here.add(btype)
            # Track the dominant type for interpretability.
            priority = _boundary_priority(btype)
            if priority > best_priority:
                best_priority = priority
                best_type = btype
        if types_here:
            dominant_type[h] = best_type
            for t in types_here:
                sources_by_type[t].append(h)

    # 2. Per-type BFS from each boundary set.
    step_km = hex_size_km * math.sqrt(3.0)
    distance_by_type: dict[str, dict[Hex, float]] = {}
    for t, sources in sources_by_type.items():
        d: dict[Hex, float] = {h: math.inf for h in hexes}
        if not sources:
            distance_by_type[t] = d
            continue
        queue: deque[Hex] = deque()
        for s in sources:
            d[s] = 0.0
            queue.append(s)
        while queue:
            h = queue.popleft()
            current = d[h]
            for nb in h.neighbors():
                if nb not in hex_set:
                    continue
                new_d = current + step_km
                if new_d < d[nb]:
                    d[nb] = new_d
                    queue.append(nb)
        distance_by_type[t] = d

    # 3. Per-hex "nearest boundary" view derived from the type maps.
    distance_to_boundary_km: dict[Hex, float] = {}
    boundary_type: dict[Hex, str | None] = {}
    for h in hexes:
        # Boundary hexes themselves keep their dominant classification; for
        # them the minimum-distance type is whichever they sit on, and the
        # priority rule above already picked the most expressive one.
        if dominant_type[h] is not None:
            boundary_type[h] = dominant_type[h]
            distance_to_boundary_km[h] = 0.0
            continue
        best_t: str | None = None
        best_d = math.inf
        for t in _BOUNDARY_TYPES:
            dt = distance_by_type[t][h]
            if dt < best_d:
                best_d = dt
                best_t = t
        boundary_type[h] = best_t
        distance_to_boundary_km[h] = best_d

    return distance_by_type, boundary_type, distance_to_boundary_km


def _boundary_priority(btype: str) -> int:
    """Tie-break for T-junctions: pick the more interpretable boundary."""
    return {
        BOUNDARY_CC_CONVERGENT: 5,
        BOUNDARY_OC_CONVERGENT: 4,
        BOUNDARY_OO_CONVERGENT: 3,
        BOUNDARY_DIVERGENT: 2,
        BOUNDARY_TRANSFORM: 1,
    }.get(btype, 0)


def generate_plates(
    hexes: Iterable[Hex],
    radius: int,
    plate_config: PlateConfig,
    hex_size_km: float,
    rng: RngHierarchy,
) -> PlateField:
    """Build the full PlateField for the world.

    Pure function of (hex set, radius, config, hex_size_km, seed). Each
    sub-step gets its own child RNG so reordering or adding plates later
    doesn't reshuffle earlier random draws.
    """
    hex_list = list(hexes)

    seeds_rng = rng.child("worldgen", "plates", "seeds")
    seeds = _place_seeds(hex_list, radius, plate_config, hex_size_km, seeds_rng)

    plates: list[Plate] = []
    for i, seed_hex in enumerate(seeds):
        ptype_rng = rng.child("worldgen", "plates", "plate", i, "type")
        ptype = _classify_plate_type(plate_config.continental_fraction, ptype_rng)
        motion_rng = rng.child("worldgen", "plates", "plate", i, "motion")
        mx, my = _random_unit_vector(motion_rng)
        baseline = (
            plate_config.continental_baseline
            if ptype == PLATE_TYPE_CONTINENTAL
            else plate_config.oceanic_baseline
        )
        plates.append(Plate(
            id=i,
            seed_hex=seed_hex,
            type=ptype,
            motion=(mx * plate_config.motion_speed, my * plate_config.motion_speed),
            baseline_elevation=baseline,
        ))

    # Domain-warp noise for boundary irregularity.
    warp_x = PerlinNoise2D.from_rng(rng.child("worldgen", "plates", "warp_x"))
    warp_y = PerlinNoise2D.from_rng(rng.child("worldgen", "plates", "warp_y"))
    warp_freq = hex_size_km / plate_config.boundary_warp_wavelength_km
    warp_strength_hex = plate_config.boundary_warp_strength_km / hex_size_km

    # Cache seeds in cartesian for the nearest-neighbor loop.
    seeds_xy: list[tuple[int, float, float]] = [
        (p.id, *_hex_to_xy(p.seed_hex)) for p in plates
    ]
    plates_by_id: dict[int, Plate] = {p.id: p for p in plates}

    # Convert the baseline-blend distance from km to cartesian units. One hex
    # of physical distance = √3 cart units in this projection.
    blend_cart = plate_config.baseline_blend_km * math.sqrt(3.0) / hex_size_km

    hex_to_plate: dict[Hex, int] = {}
    hex_baseline: dict[Hex, float] = {}
    for h in hex_list:
        pid, baseline = _assign_hex_to_plate_and_baseline(
            h, seeds_xy, plates_by_id,
            warp_x, warp_y, warp_freq, warp_strength_hex,
            blend_cart,
        )
        hex_to_plate[h] = pid
        hex_baseline[h] = baseline

    distance_by_type, boundary_type, distance_km = _bfs_distance_by_type(
        hex_list, hex_to_plate, tuple(plates),
        plate_config.convergence_threshold, hex_size_km,
    )

    return PlateField(
        plates=tuple(plates),
        hex_to_plate=hex_to_plate,
        hex_baseline=hex_baseline,
        distance_by_type=distance_by_type,
        boundary_type=boundary_type,
        distance_to_boundary_km=distance_km,
    )


def plate_elevation_bias(
    h: Hex,
    field: PlateField,
    plate_config: PlateConfig,
) -> float:
    """Signed elevation contribution from plates for a single hex.

    = baseline of the hex's plate
      + decayed boundary contribution if within ``boundary_falloff_km``
    """
    # Baseline is the soft-Voronoi blend computed during plate generation —
    # equals the hex's plate.baseline_elevation deep in a plate, transitions
    # smoothly toward the neighbor plate's baseline across the boundary.
    bias = field.hex_baseline[h]

    falloff = plate_config.boundary_falloff_km
    for btype, distances in field.distance_by_type.items():
        distance = distances[h]
        if distance >= falloff:
            continue
        amplitude = _boundary_amplitude(btype, plate_config)
        if amplitude == 0.0:  # transform contributes nothing
            continue
        t = 1.0 - distance / falloff
        # Smoothstep (3t² − 2t³): zero derivative at both endpoints, steepest
        # in the middle. Rounded peak at the boundary (no knife edge) and a
        # gentle taper into the plate interior. Summing contributions across
        # every nearby boundary *type* (rather than inheriting one via BFS)
        # keeps the field smooth at triple junctions: cc-convergent uplift
        # fades out and divergent depression fades in over the same falloff
        # radius, instead of stepping abruptly at the BFS Voronoi seam.
        factor = t * t * (3.0 - 2.0 * t)
        bias += amplitude * factor
    return bias


def _boundary_amplitude(btype: str, c: PlateConfig) -> float:
    if btype == BOUNDARY_CC_CONVERGENT:
        return c.mountain_amplitude
    if btype == BOUNDARY_OC_CONVERGENT:
        return c.coastal_range_amplitude
    if btype == BOUNDARY_OO_CONVERGENT:
        return c.island_arc_amplitude
    if btype == BOUNDARY_DIVERGENT:
        return -c.rift_depth
    return 0.0  # transform
