"""L0b tectonics layer — thin adaptor over ``tectonic_sim.polygon_sim``.

``simulate_tectonics`` delegates unconditionally to
``kartogen.tectonics_cast.simulate_tectonics_via_continuous_sim``. This
module owns the per-hex result types the downstream layers consume —
``LithosphereColumn``, ``LithosphereState``, ``TectonicPlate`` — plus
the closed-form elevation map ``column_to_elevation_km(col, cfg)`` and
a fallback ``WorldShape`` derivation for callers that don't pass one
in.

Deterministic in ``(config, seed)`` — the only entropy source is the
RNG's hash-seed handed to the polygon sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kartogen.hex import Hex
from kartogen.types import TectonicsConfig

if TYPE_CHECKING:
    from kartogen.rng import RngHierarchy

# Crust-type tags carried on every ``LithosphereColumn`` (and the polygon
# sim's per-cell crust). The strings are the canonical kartogen values.
PLATE_TYPE_CONTINENTAL = "continental"
PLATE_TYPE_OCEANIC = "oceanic"


# --- Data structures -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LithosphereColumn:
    """One hex's crust column."""

    crust_type: str        # "continental" | "oceanic"
    thickness_km: float
    age_myr: float


@dataclass
class TectonicPlate:
    """Per-plate handle attached to the final ``LithosphereState``.

    ``center_km`` is the plate's anchor position at t=0 (wrap-aware
    centroid of its initial cell footprint). ``velocity_kmpy`` is the
    plate's translational velocity. ``crust`` maps plate-local hex
    coordinates to ``LithosphereColumn`` — populated by the bridge from
    the final cell-mask state.
    """

    id: int
    initial_type: str
    center_km: tuple[float, float]
    velocity_kmpy: tuple[float, float]
    crust: dict[Hex, LithosphereColumn] = field(default_factory=dict)


@dataclass(frozen=True)
class LithosphereState:
    """Final, world-hex-indexed result of the simulation.

    Consumed by the elevation layer (uses ``elevation_km`` as the
    baseline) and by the snapshot serializer (the column fields flow
    into ``HexData``). The pre-rendered drift / thickness / topography
    GIF frames live on ``raw_snapshot`` so the export-time renderers
    can emit ``tectonic_sim_views/*.gif`` without re-running the sim.
    """

    columns: dict[Hex, LithosphereColumn]
    plate_id: dict[Hex, int]
    elevation_km: dict[Hex, float]
    sea_level_km: float
    n_ticks_simulated: int
    plates: tuple[TectonicPlate, ...]
    # Raw polygon-sim output, preserved for export-time renderers
    # (tectonic_sim_views/). Typed as ``object`` to avoid a circular
    # import of ``polygon_sim.PolygonPlate`` at kartogen-init time. The
    # value is a dict with keys ``plates``, ``owner``/``crust``/``age``/
    # ``thickness`` ((gy, gx) ndarrays), ``cell_km``, ``hotspots``,
    # ``frames``/``frames_thickness``/``frames_topography`` (GIF panels),
    # ``timeline``, ``sim_config``.
    raw_snapshot: object | None = None


# --- Elevation map ---------------------------------------------------------


def column_to_elevation_km(col: LithosphereColumn, config: TectonicsConfig) -> float:
    """Convert a lithosphere column to a signed elevation in km.

    Continental: isostatic excess from a reference thickness.
    Oceanic: half-space cooling depth from crust age, capped at
    ``max_ocean_depth_km``.
    """
    if col.crust_type == PLATE_TYPE_CONTINENTAL:
        return (
            (col.thickness_km - config.continental_reference_thickness_km)
            * config.continental_isostasy_factor
        )
    depth = config.ridge_depth_km + config.ridge_subsidence_rate * math.sqrt(
        max(0.0, col.age_myr)
    )
    return -min(depth, config.max_ocean_depth_km)


# --- Entry point -----------------------------------------------------------


def simulate_tectonics(
    world_hexes_iter: list[Hex],
    config: TectonicsConfig,
    hex_size_km: float,
    rng: "RngHierarchy",
    *,
    world_shape=None,                   # type: ignore[no-untyped-def]
    param_temperature: float = 0.0,
    render_visuals: bool = False,
) -> LithosphereState:
    """Run the tectonics simulation and return a ``LithosphereState``.

    Thin adaptor: delegates to
    ``kartogen.tectonics_cast.simulate_tectonics_via_continuous_sim``. The
    polygon sim seeds its own plates from ``tectonic_sim.toml``.
    ``world_shape`` is the simulation footprint; if ``None``, it's derived
    from the world-hex set's bounding box. ``param_temperature`` > 0
    randomizes the loaded ``SimConfig`` before the run. ``render_visuals``
    populates ``LithosphereState.raw_snapshot`` with the export-time
    rendering payload (GIF frames + outline polygons); leave it ``False``
    for plain generation.

    Pure function of all inputs.
    """
    from kartogen.tectonics_cast import simulate_tectonics_via_continuous_sim

    if world_shape is None:
        world_shape = _world_shape_from_hexes(world_hexes_iter, hex_size_km)

    seed = int(rng.child("tectonics").randrange(1 << 31))

    return simulate_tectonics_via_continuous_sim(
        list(world_hexes_iter),
        config,
        world_shape,
        hex_size_km,
        seed,
        param_temperature=param_temperature,
        render_visuals=render_visuals,
    )


def _world_shape_from_hexes(
    world_hexes: list[Hex], hex_size_km: float,
):                                      # type: ignore[no-untyped-def]
    """Derive a ``WorldShape`` whose rectangle encloses every input hex.

    Used by ``simulate_tectonics`` when the caller didn't pass an
    explicit ``WorldShape`` — e.g. tests that drive the function
    directly with just hexes + config.
    """
    from kartogen.types import WorldShape

    if not world_hexes:
        return WorldShape(width_km=100.0, height_km=100.0)
    xs: list[float] = []
    ys: list[float] = []
    for h in world_hexes:
        x = 1.5 * h.q * hex_size_km
        y = math.sqrt(3.0) * (h.r + h.q / 2.0) * hex_size_km
        xs.append(x)
        ys.append(y)
    return WorldShape(
        width_km=max(2.0 * max(xs), -2.0 * min(xs), 100.0),
        height_km=max(2.0 * max(ys), -2.0 * min(ys), 100.0),
    )
