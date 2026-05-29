"""Time-stepped plate-tectonics simulation (PlaTec-style, hex-grid).

Each plate owns a *plate-local* dictionary of lithosphere columns
(``LithosphereColumn``). The plate's centre drifts through continuous km
coordinates each tick; the crust attached to it drifts along with it. We
re-evaluate which world hex each plate-local column maps to on every tick.

Overlap (two plates claiming the same world hex) is a collision: the denser
column subducts, transferring some mass to the overriding plate (volcanic
arc). Continental-on-continental overlaps fold instead — mass transfers from
the smaller plate's column to the larger's. World hexes that no plate claims
are filled with fresh oceanic crust attached to the nearest plate (divergent
fill). Oceanic crust ages and sinks via half-space cooling.

The final lithosphere is converted to an elevation field via two formulas:

- Continental:  e_km = (thickness − reference) × continental_isostasy_factor
- Oceanic:      e_km = −ridge_depth_km − ridge_subsidence_rate × √age_myr

Deterministic in (world_seed, sim seed, config). Pure function of input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from worldgen._log import progress
from worldgen.hex import Hex
from worldgen.noise import PerlinNoise2D
from worldgen.plates import (
    PLATE_TYPE_CONTINENTAL,
    PLATE_TYPE_OCEANIC,
    PlateField,
)
from worldgen.types import TectonicsConfig

if TYPE_CHECKING:
    from worldgen.rng import RngHierarchy


# --- Data structures -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LithosphereColumn:
    """One hex's crust column. Carried by a plate."""

    crust_type: str        # "continental" | "oceanic"
    thickness_km: float
    age_myr: float


@dataclass
class TectonicPlate:
    """A plate with its own crust map keyed by plate-local hex coords.

    ``center_km`` is the world-km position the plate's local (0, 0) currently
    sits at. Plate motion is integration of ``velocity_kmpy`` over the sim's
    ``dt_myr`` ticks.
    """

    id: int
    initial_type: str
    center_km: tuple[float, float]
    velocity_kmpy: tuple[float, float]
    crust: dict[Hex, LithosphereColumn] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TectonicFrame:
    """One snapshot of the tectonic simulation mid-flight.

    Captured periodically during ``simulate_tectonics`` so the drift can
    be played back as a per-tick animation. Carries enough state to render
    the world's plate ownership and continental/oceanic mask at that tick
    — but not full column thickness/age (kept smaller, animation-friendly).
    """

    tick: int
    time_myr: float
    # World-hex → final-tick-equivalent (plate_id, "continental" | "oceanic").
    # Hexes that no plate claims at this tick get plate_id = -1. Has had the
    # boundary warp applied so it can be rendered as a hex-grid image
    # directly.
    plate_id: dict[Hex, int]
    crust_type: dict[Hex, str]
    # Each plate's current centre position in world-km cartesian space.
    plate_centers_km: dict[int, tuple[float, float]]
    # For the continuous-field drift render: each plate's currently-owned
    # *plate-local* hex set, plus matching crust types. Combined with
    # ``plate_centers_km`` the renderer reconstructs every particle's
    # continuous world-km position (= plate_local_xy + plate.center_km) and
    # draws at sub-pixel float coords, so the visible motion between frames
    # is smooth instead of teleporting one integer hex at a time.
    plate_owned_local_hexes: dict[int, tuple[Hex, ...]]
    plate_owned_crust_type: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class LithosphereState:
    """Final, world-hex-indexed result of the simulation.

    Consumed by the elevation layer (uses ``elevation_km`` as the baseline)
    and by the snapshot serializer (the column fields flow into ``HexData``).

    ``history`` carries the per-snapshot ``TectonicFrame`` captures (empty
    when ``TectonicsConfig.snapshot_period_ticks == 0``). The drift-GIF
    renderer reads this.
    """

    columns: dict[Hex, LithosphereColumn]
    plate_id: dict[Hex, int]
    elevation_km: dict[Hex, float]
    sea_level_km: float
    n_ticks_simulated: int
    plates: tuple[TectonicPlate, ...]
    history: tuple[TectonicFrame, ...]


# --- Hex ↔ km conversions --------------------------------------------------


def _hex_to_xy_units(h: Hex) -> tuple[float, float]:
    """Axial hex → flat cartesian units (matches the elevation projection)."""
    return 1.5 * h.q, math.sqrt(3.0) * (h.r + h.q / 2.0)


def _hex_to_xy_km(h: Hex, hex_size_km: float) -> tuple[float, float]:
    x, y = _hex_to_xy_units(h)
    return x * hex_size_km, y * hex_size_km


def _axial_round(qf: float, rf: float) -> Hex:
    """Round fractional axial coords to the nearest valid hex (cube round)."""
    sf = -qf - rf
    rq = round(qf)
    rr = round(rf)
    rs = round(sf)
    q_diff = abs(rq - qf)
    r_diff = abs(rr - rf)
    s_diff = abs(rs - sf)
    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs
    return Hex(int(rq), int(rr))


def _xy_km_to_hex(x_km: float, y_km: float, hex_size_km: float) -> Hex:
    """Inverse of ``_hex_to_xy_km`` — nearest integer hex."""
    x = x_km / hex_size_km
    y = y_km / hex_size_km
    qf = x / 1.5
    rf = y / math.sqrt(3.0) - qf / 2.0
    return _axial_round(qf, rf)


def _local_for_world(world_hex: Hex, plate: TectonicPlate, hex_size_km: float) -> Hex:
    """World hex → plate-local hex, given the plate's current centre."""
    wx, wy = _hex_to_xy_km(world_hex, hex_size_km)
    lx = wx - plate.center_km[0]
    ly = wy - plate.center_km[1]
    return _xy_km_to_hex(lx, ly, hex_size_km)


def _world_for_local(local_hex: Hex, plate: TectonicPlate, hex_size_km: float) -> Hex:
    """Plate-local hex → world hex, given the plate's current centre."""
    lx, ly = _hex_to_xy_km(local_hex, hex_size_km)
    wx = lx + plate.center_km[0]
    wy = ly + plate.center_km[1]
    return _xy_km_to_hex(wx, wy, hex_size_km)


# --- Initial-condition seeding ---------------------------------------------


def _seed_plates_from_field(
    initial: PlateField,
    config: TectonicsConfig,
    hex_size_km: float,
) -> list[TectonicPlate]:
    """Build TectonicPlate list from the static PlateField at t=0.

    Each plate's centre starts at the world position of its seed hex; each
    owned world hex becomes a plate-local column whose initial thickness
    depends on the plate's type.
    """
    out: list[TectonicPlate] = []
    plates_by_id = {p.id: p for p in initial.plates}
    # Group owned world hexes by plate id.
    owned: dict[int, list[Hex]] = {p.id: [] for p in initial.plates}
    for world_hex, pid in initial.hex_to_plate.items():
        owned[pid].append(world_hex)

    for plate_id in sorted(plates_by_id.keys()):
        src = plates_by_id[plate_id]
        center_km = _hex_to_xy_km(src.seed_hex, hex_size_km)
        # PlateConfig.motion_speed already scales the unit vector at plate
        # construction time; treat it as km/Myr directly. The tectonics
        # config's plate_speed_kmpy is an additional global multiplier so
        # the same plate config can drive both static (slow) and dynamic
        # (faster) layouts.
        vx = src.motion[0] * config.plate_speed_kmpy
        vy = src.motion[1] * config.plate_speed_kmpy
        plate = TectonicPlate(
            id=src.id,
            initial_type=src.type,
            center_km=center_km,
            velocity_kmpy=(vx, vy),
            crust={},
        )
        initial_thickness = (
            config.continental_thickness_km
            if src.type == PLATE_TYPE_CONTINENTAL
            else config.oceanic_thickness_km
        )
        crust_type = src.type
        for world_hex in owned[plate_id]:
            local_hex = _local_for_world(world_hex, plate, hex_size_km)
            plate.crust[local_hex] = LithosphereColumn(
                crust_type=crust_type,
                thickness_km=initial_thickness,
                age_myr=0.0,
            )
        out.append(plate)
    return out


# --- Per-tick steps --------------------------------------------------------


def _advance_positions(
    plates: list[TectonicPlate],
    dt_myr: float,
    world_radius_km: float,
) -> None:
    """Integrate plate centres by velocity × dt, bouncing off the world edge.

    The hex disc isn't a torus, so plates that drift past the edge would have
    all their crust mapped off-world (silently lost). Bouncing keeps the
    simulation interesting on bounded worlds: when a plate's centre would
    cross the circumscribed circle of the disc, its velocity component along
    the outward normal flips. This is a cheap stand-in for the wrap-around
    PlaTec gets on its rectangular torus.
    """
    r2_max = world_radius_km * world_radius_km
    for p in plates:
        x, y = p.center_km
        vx, vy = p.velocity_kmpy
        nx = x + vx * dt_myr
        ny = y + vy * dt_myr
        r2 = nx * nx + ny * ny
        if r2 > r2_max:
            r = math.sqrt(r2)
            ux, uy = nx / r, ny / r
            v_dot_n = vx * ux + vy * uy
            if v_dot_n > 0:
                vx -= 2.0 * v_dot_n * ux
                vy -= 2.0 * v_dot_n * uy
            nx = ux * world_radius_km
            ny = uy * world_radius_km
        p.center_km = (nx, ny)
        p.velocity_kmpy = (vx, vy)


def _compute_overlap(
    plates: list[TectonicPlate],
    world_hexes: set[Hex],
    hex_size_km: float,
) -> dict[Hex, list[tuple[TectonicPlate, Hex]]]:
    """For each world hex, list every (plate, local_hex) currently claiming it.

    Plate-local crust whose world projection falls outside the world disc is
    silently dropped (the crust has drifted off the world).
    """
    overlap: dict[Hex, list[tuple[TectonicPlate, Hex]]] = {}
    for p in plates:
        for local_hex in p.crust:
            world_hex = _world_for_local(local_hex, p, hex_size_km)
            if world_hex not in world_hexes:
                continue
            overlap.setdefault(world_hex, []).append((p, local_hex))
    return overlap


def _resolve_collisions(
    plates: list[TectonicPlate],
    overlap: dict[Hex, list[tuple[TectonicPlate, Hex]]],
    config: TectonicsConfig,
) -> None:
    """Process every world hex claimed by ≥2 plates.

    Subduction rule: of the contenders, the column with the lowest *buoyancy*
    sinks. Buoyancy = continental(2) > young oceanic(1) > old oceanic(0). The
    sinking column is removed from its plate; the surviving column on the
    overriding plate is uplifted by ``subduction_arc_uplift_km``.

    Continental-on-continental: the smaller plate's column transfers
    ``folding_ratio`` of its thickness to the larger plate's column at that
    location, and both plates' columns gain ``orogeny_uplift_per_overlap_km``
    (the canonical PlaTec mountain-building term).
    """

    def buoyancy(col: LithosphereColumn) -> float:
        # Continental crust is always more buoyant than any oceanic crust.
        if col.crust_type == PLATE_TYPE_CONTINENTAL:
            return 1e6
        # Younger oceanic crust is more buoyant than older.
        return -col.age_myr

    for world_hex, claimants in overlap.items():
        if len(claimants) < 2:
            continue
        # Sort claimants by buoyancy descending: highest stays, lowest subducts.
        claimants.sort(
            key=lambda c: (buoyancy(c[0].crust[c[1]]), c[0].id),
            reverse=True,
        )
        upper_plate, upper_local = claimants[0]
        upper_col = upper_plate.crust[upper_local]

        for lower_plate, lower_local in claimants[1:]:
            lower_col = lower_plate.crust[lower_local]

            if (
                upper_col.crust_type == PLATE_TYPE_CONTINENTAL
                and lower_col.crust_type == PLATE_TYPE_CONTINENTAL
            ):
                # Continental folding. Transfer mass from smaller plate (fewer
                # cells) to larger; both columns thicken.
                small, small_local = (
                    (lower_plate, lower_local)
                    if len(lower_plate.crust) <= len(upper_plate.crust)
                    else (upper_plate, upper_local)
                )
                large, large_local = (
                    (upper_plate, upper_local)
                    if small is lower_plate
                    else (lower_plate, lower_local)
                )
                small_col = small.crust[small_local]
                transfer = small_col.thickness_km * config.folding_ratio
                large.crust[large_local] = LithosphereColumn(
                    crust_type=PLATE_TYPE_CONTINENTAL,
                    thickness_km=(
                        large.crust[large_local].thickness_km
                        + transfer
                        + config.orogeny_uplift_per_overlap_km
                    ),
                    age_myr=large.crust[large_local].age_myr,
                )
                small.crust[small_local] = LithosphereColumn(
                    crust_type=PLATE_TYPE_CONTINENTAL,
                    thickness_km=max(
                        0.0,
                        small_col.thickness_km
                        - transfer
                        + config.orogeny_uplift_per_overlap_km,
                    ),
                    age_myr=small_col.age_myr,
                )
                # Recompute upper_col for any subsequent claimant in this hex.
                upper_col = upper_plate.crust[upper_local]
            else:
                # Subduction: the lower-buoyancy column sinks and is removed.
                # Arc-uplift the survivor.
                del lower_plate.crust[lower_local]
                upper_plate.crust[upper_local] = LithosphereColumn(
                    crust_type=upper_col.crust_type,
                    thickness_km=upper_col.thickness_km + config.subduction_arc_uplift_km,
                    age_myr=upper_col.age_myr,
                )
                upper_col = upper_plate.crust[upper_local]


def _seed_new_crust_in_gaps(
    plates: list[TectonicPlate],
    world_hexes: set[Hex],
    overlap: dict[Hex, list[tuple[TectonicPlate, Hex]]],
    config: TectonicsConfig,
    hex_size_km: float,
) -> None:
    """Empty world hexes get new crust attached to the nearest plate.

    Branches on the nearest plate's *initial* type:

    - **Continental plate** → spawn thinned continental crust at
      ``rift_thickness_km``. Models continental rifting (East African
      Rift, Dead Sea graben): the crust thins but doesn't become true
      oceanic. Isostasy puts the new column below sea level for a typical
      rift_thickness, so it reads as a low-lying basin / inland sea rather
      than a deep ocean trench.

    - **Oceanic plate** → spawn fresh oceanic crust at ``oceanic_thickness_km``.
      Classic divergent-boundary sea-floor spreading.

    The previous version always spawned oceanic crust regardless of nearest-
    plate type, which made all-continental worlds (``continental_fraction =
    1.0``) become majority-ocean by the end of the sim — and produced a
    sharp linear "drift wake" coastline behind each plate from the head/tail
    rigid-stamp boundary.
    """
    for world_hex in world_hexes:
        if world_hex in overlap:
            continue
        wx, wy = _hex_to_xy_km(world_hex, hex_size_km)
        nearest_plate = min(
            plates,
            key=lambda p: (wx - p.center_km[0]) ** 2 + (wy - p.center_km[1]) ** 2,
        )
        local_hex = _local_for_world(world_hex, nearest_plate, hex_size_km)
        if nearest_plate.initial_type == PLATE_TYPE_CONTINENTAL:
            nearest_plate.crust[local_hex] = LithosphereColumn(
                crust_type=PLATE_TYPE_CONTINENTAL,
                thickness_km=config.rift_thickness_km,
                age_myr=0.0,
            )
        else:
            nearest_plate.crust[local_hex] = LithosphereColumn(
                crust_type=PLATE_TYPE_OCEANIC,
                thickness_km=config.oceanic_thickness_km,
                age_myr=0.0,
            )


def _age_all_crust(plates: list[TectonicPlate], dt_myr: float) -> None:
    for p in plates:
        for local_hex, col in p.crust.items():
            p.crust[local_hex] = LithosphereColumn(
                crust_type=col.crust_type,
                thickness_km=col.thickness_km,
                age_myr=col.age_myr + dt_myr,
            )


def _erode(plates: list[TectonicPlate], config: TectonicsConfig) -> None:
    """Simple PlaTec-style blur: blend each column's thickness toward its
    plate-local neighbors' mean. Only applied to continental crust above
    the sea-level threshold — submerged crust isn't ground down by wind/rain.
    """
    strength = config.erosion_strength
    if strength <= 0.0:
        return
    for p in plates:
        crust = p.crust
        new_thickness: dict[Hex, float] = {}
        for local_hex, col in crust.items():
            if col.crust_type != PLATE_TYPE_CONTINENTAL:
                continue
            neighbor_thicknesses = [
                crust[nb].thickness_km
                for nb in local_hex.neighbors()
                if nb in crust
            ]
            if not neighbor_thicknesses:
                continue
            mean = sum(neighbor_thicknesses) / len(neighbor_thicknesses)
            new_thickness[local_hex] = (
                col.thickness_km * (1.0 - strength) + mean * strength
            )
        for local_hex, t in new_thickness.items():
            col = crust[local_hex]
            crust[local_hex] = LithosphereColumn(
                crust_type=col.crust_type,
                thickness_km=t,
                age_myr=col.age_myr,
            )


# --- Elevation model -------------------------------------------------------


def column_to_elevation_km(col: LithosphereColumn, config: TectonicsConfig) -> float:
    """Convert a lithosphere column to a signed elevation in km."""
    if col.crust_type == PLATE_TYPE_CONTINENTAL:
        return (
            (col.thickness_km - config.continental_reference_thickness_km)
            * config.continental_isostasy_factor
        )
    # Oceanic: half-space cooling. Older crust sits deeper.
    depth = config.ridge_depth_km + config.ridge_subsidence_rate * math.sqrt(
        max(0.0, col.age_myr)
    )
    return -min(depth, config.max_ocean_depth_km)


# --- Finalization ----------------------------------------------------------


def _warp_crust_boundaries(
    columns: dict[Hex, LithosphereColumn],
    config: TectonicsConfig,
    hex_size_km: float,
    rng: RngHierarchy,
) -> dict[Hex, LithosphereColumn]:
    """Perlin-noise warp of coastline boundaries.

    The PlaTec-style rigid-stamp plate translation produces coastlines that
    lie exactly along the integer-hex grid (sharp horizontal / diagonal
    lines in the elevation render). To break that signature we look at the
    **elevation-sign boundary** — every hex whose elevation places it on
    the opposite side of sea level from some neighbour — and (with
    probability tied to a deterministic Perlin field) overwrite the hex's
    column with that neighbour's column. The result is a wavy,
    bay-and-headland coastline at boundary-warp-wavelength scale.

    Using elevation-sign rather than crust-type as the boundary makes the
    warp fire correctly on continental-rift worlds too (where coastlines
    separate original continental crust from thinned rift crust, both
    nominally continental).

    ``boundary_warp_strength = 0`` disables the pass.
    """
    if config.boundary_warp_strength <= 0.0:
        return columns
    noise = PerlinNoise2D.from_rng(
        rng.child("worldgen", "tectonics", "boundary_warp"),
    )
    freq = hex_size_km / max(1.0, config.boundary_warp_wavelength_km)
    threshold = 1.0 - config.boundary_warp_strength
    sea_km = config.sea_level_km

    # Precompute the above-sea-level boolean for every hex.
    above_sea = {
        h: column_to_elevation_km(col, config) > sea_km
        for h, col in columns.items()
    }

    swaps: dict[Hex, LithosphereColumn] = {}
    for h, col in columns.items():
        my_side = above_sea[h]
        diff_neighbour: LithosphereColumn | None = None
        for nb in h.neighbors():
            nb_col = columns.get(nb)
            if nb_col is None:
                continue
            if above_sea[nb] != my_side:
                diff_neighbour = nb_col
                break
        if diff_neighbour is None:
            continue
        x, y = _hex_to_xy_units(h)
        n01 = 0.5 * (noise.sample(x * freq, y * freq) + 1.0)
        if n01 > threshold:
            swaps[h] = diff_neighbour
    if not swaps:
        return columns
    new_columns = dict(columns)
    new_columns.update(swaps)
    return new_columns


def _warp_frame_plate_ids(
    plate_id: dict[Hex, int],
    crust_type: dict[Hex, str],
    config: TectonicsConfig,
    hex_size_km: float,
    rng: RngHierarchy,
) -> tuple[dict[Hex, int], dict[Hex, str]]:
    """Apply the Perlin boundary warp to a per-frame (plate_id, crust_type)
    pair. Same deterministic Perlin field used by ``_warp_crust_boundaries``
    in finalize, so the warp pattern is consistent across frames (the
    coastline at world_xy stays at the same warpy shape over time, just
    translated as plates drift).
    """
    if config.boundary_warp_strength <= 0.0:
        return plate_id, crust_type
    noise = PerlinNoise2D.from_rng(
        rng.child("worldgen", "tectonics", "boundary_warp"),
    )
    freq = hex_size_km / max(1.0, config.boundary_warp_wavelength_km)
    threshold = 1.0 - config.boundary_warp_strength

    new_plate_id = dict(plate_id)
    new_crust_type = dict(crust_type)
    for h, pid in plate_id.items():
        target_pid: int | None = None
        target_type: str | None = None
        for nb in h.neighbors():
            if nb in plate_id and plate_id[nb] != pid:
                target_pid = plate_id[nb]
                target_type = crust_type[nb]
                break
        if target_pid is None:
            continue
        x, y = _hex_to_xy_units(h)
        n01 = 0.5 * (noise.sample(x * freq, y * freq) + 1.0)
        if n01 > threshold:
            new_plate_id[h] = target_pid
            new_crust_type[h] = target_type  # type: ignore[assignment]
    return new_plate_id, new_crust_type


def _capture_frame(
    plates: list[TectonicPlate],
    world_hexes: set[Hex],
    config: TectonicsConfig,
    hex_size_km: float,
    tick: int,
    time_myr: float,
    rng: RngHierarchy,
) -> TectonicFrame:
    """Snapshot the world at this tick.

    Records two views of the same state:

    - **Hex-grid view** (``plate_id``, ``crust_type``): per-world-hex
      assignment using the same "highest-buoyancy claimant" rule as
      ``_finalize``, with the boundary warp applied (same Perlin field used
      at finalize, so the warpy pattern is temporally coherent).

    - **Continuous-field view** (``plate_owned_local_hexes`` +
      ``plate_centers_km``): the plate-local hex set each plate currently
      owns, plus the plate's continuous km centre. The drift renderer uses
      this pair to draw plates at sub-pixel pixel positions, eliminating
      the integer-hex teleport-between-frames artifact.
    """
    overlap = _compute_overlap(plates, world_hexes, hex_size_km)

    def buoyancy(c: tuple[TectonicPlate, Hex]) -> float:
        col = c[0].crust[c[1]]
        if col.crust_type == PLATE_TYPE_CONTINENTAL:
            return 1e6
        return -col.age_myr

    plate_id_map: dict[Hex, int] = {}
    crust_type_map: dict[Hex, str] = {}
    for h in world_hexes:
        claimants = overlap.get(h)
        if not claimants:
            plate_id_map[h] = -1
            crust_type_map[h] = PLATE_TYPE_OCEANIC
            continue
        best = max(claimants, key=buoyancy)
        plate_id_map[h] = best[0].id
        crust_type_map[h] = best[0].crust[best[1]].crust_type

    plate_id_map, crust_type_map = _warp_frame_plate_ids(
        plate_id_map, crust_type_map, config, hex_size_km, rng,
    )

    centers = {p.id: p.center_km for p in plates}
    owned_local: dict[int, tuple[Hex, ...]] = {
        p.id: tuple(p.crust.keys()) for p in plates
    }
    owned_crust_type: dict[int, tuple[str, ...]] = {
        p.id: tuple(p.crust[h].crust_type for h in owned_local[p.id])
        for p in plates
    }
    return TectonicFrame(
        tick=tick,
        time_myr=time_myr,
        plate_id=plate_id_map,
        crust_type=crust_type_map,
        plate_centers_km=centers,
        plate_owned_local_hexes=owned_local,
        plate_owned_crust_type=owned_crust_type,
    )


def _finalize(
    plates: list[TectonicPlate],
    world_hexes: set[Hex],
    config: TectonicsConfig,
    hex_size_km: float,
    n_ticks: int,
    rng: RngHierarchy,
    history: tuple[TectonicFrame, ...],
) -> LithosphereState:
    """Build the final world-hex-indexed lithosphere state.

    After the last tick, the same overlap procedure is used to determine
    which plate "wins" each world hex (highest-buoyancy column survives;
    others subduct silently). Any remaining empty hex is filled with new
    crust on the nearest plate (continental rift if that plate is itself
    continental, oceanic otherwise). Finally a Perlin-noise pass warps the
    crust-type boundaries so the result doesn't lie exactly along the
    integer-hex grid.
    """
    overlap = _compute_overlap(plates, world_hexes, hex_size_km)
    _resolve_collisions(plates, overlap, config)
    _seed_new_crust_in_gaps(plates, world_hexes, overlap, config, hex_size_km)
    # Recompute overlap now that gaps are filled.
    overlap = _compute_overlap(plates, world_hexes, hex_size_km)

    columns: dict[Hex, LithosphereColumn] = {}
    plate_id: dict[Hex, int] = {}

    for world_hex in world_hexes:
        claimants = overlap.get(world_hex)
        if not claimants:
            # Shouldn't happen after the fill pass, but be defensive: a
            # bare-ocean column with no plate ownership.
            columns[world_hex] = LithosphereColumn(
                crust_type=PLATE_TYPE_OCEANIC,
                thickness_km=config.oceanic_thickness_km,
                age_myr=0.0,
            )
            plate_id[world_hex] = -1
        else:
            # Pick the highest-buoyancy claimant (same rule as collisions).
            def buoyancy(c: tuple[TectonicPlate, Hex]) -> float:
                col = c[0].crust[c[1]]
                if col.crust_type == PLATE_TYPE_CONTINENTAL:
                    return 1e6
                return -col.age_myr

            best = max(claimants, key=buoyancy)
            columns[world_hex] = best[0].crust[best[1]]
            plate_id[world_hex] = best[0].id

    # Boundary warp — break the grid-aligned coastline signature.
    columns = _warp_crust_boundaries(columns, config, hex_size_km, rng)

    elevation_km: dict[Hex, float] = {
        h: column_to_elevation_km(columns[h], config) for h in world_hexes
    }

    return LithosphereState(
        columns=columns,
        plate_id=plate_id,
        elevation_km=elevation_km,
        sea_level_km=config.sea_level_km,
        n_ticks_simulated=n_ticks,
        plates=tuple(plates),
        history=history,
    )


# --- Main entry point ------------------------------------------------------


def _world_radius_km(world_hexes: set[Hex], hex_size_km: float) -> float:
    """Effective bounce-boundary radius for the hex disc."""
    return max(
        math.hypot(*_hex_to_xy_km(h, hex_size_km)) for h in world_hexes
    )


def simulate_tectonics(
    initial: PlateField,
    world_hexes_iter: list[Hex],
    config: TectonicsConfig,
    hex_size_km: float,
    rng: RngHierarchy,
) -> LithosphereState:
    """Run ``config.n_ticks`` of time-stepped tectonics from the initial state.

    Pure function of (initial, world_hexes, config, hex_size_km, rng). The RNG
    is currently unused — collision resolution is deterministic given the
    initial state — but plumbed through so future stochastic effects
    (hotspots, transform earthquakes) can hook in without changing the
    signature.
    """
    world_hexes = set(world_hexes_iter)
    world_radius_km = _world_radius_km(world_hexes, hex_size_km)
    plates = _seed_plates_from_field(initial, config, hex_size_km)
    dt = config.dt_myr

    snap_period = config.snapshot_period_ticks
    history: list[TectonicFrame] = []
    if snap_period > 0:
        history.append(
            _capture_frame(plates, world_hexes, config, hex_size_km, 0, 0.0, rng),
        )

    for tick in progress(
        range(config.n_ticks), desc="tectonics", total=config.n_ticks,
    ):
        _advance_positions(plates, dt, world_radius_km)
        overlap = _compute_overlap(plates, world_hexes, hex_size_km)
        _resolve_collisions(plates, overlap, config)
        _seed_new_crust_in_gaps(plates, world_hexes, overlap, config, hex_size_km)
        _age_all_crust(plates, dt)
        if config.erosion_period > 0 and (tick + 1) % config.erosion_period == 0:
            _erode(plates, config)
        if snap_period > 0 and (tick + 1) % snap_period == 0:
            history.append(
                _capture_frame(
                    plates, world_hexes, config, hex_size_km,
                    tick + 1, (tick + 1) * dt, rng,
                )
            )

    return _finalize(
        plates, world_hexes, config, hex_size_km, config.n_ticks, rng,
        tuple(history),
    )
