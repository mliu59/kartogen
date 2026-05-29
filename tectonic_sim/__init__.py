"""tectonic_sim — particle-cloud plate-tectonics simulator in continuous 2D.

Pure-physics module. Knows nothing about hex grids; everything happens in
floating-point km space. Worldgen (or any other consumer) calls
``simulate(...)`` to get a ``Snapshot`` and then samples it at points of
its own choice (e.g. hex centres) via the sampling API.

Public surface:

  - ``WorldRect, SimConfig, Plate, Snapshot, Frame``  — data types
  - ``load_sim_config``                               — TOML → SimConfig
  - ``simulate``                                      — run the sim
  - ``sample_lithosphere_columns_at``, plus per-field helpers  — query API
  - ``render_particles_png, render_voronoi_png,
     render_drift_gif, render_single_plate_png``      — viz utils
"""

from __future__ import annotations

from tectonic_sim.collisions import apply_collisions
from tectonic_sim.constraints import (
    apply_contact_constraints,
    apply_velocity_damping,
)
from tectonic_sim.divergent import divergent_fill
from tectonic_sim.isostasy import particle_elevation_km
from tectonic_sim.config_loader import (
    load_sim_config,
    load_sim_config_from_path,
)
from tectonic_sim.kinematics import (
    cull_outside_domain,
    drift_positions,
    step_drift_and_apply_boundary,
    step_drift_and_cull,
    wrap_positions,
)
from tectonic_sim.overlap import detect_overlaps
from tectonic_sim.seeding import build_initial_state
from tectonic_sim.viz import (
    PLATE_PALETTE,
    render_elevation_png,
    render_initial_state,
    render_particles_png,
    render_single_plate_png,
    render_snapshot_particles,
    render_snapshot_single_plate,
    render_snapshot_voronoi,
    render_voronoi_png,
)
from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    Frame,
    Plate,
    SimConfig,
    Snapshot,
    WorldRect,
    crust_type_code,
    crust_type_name,
)

__all__ = [
    "CRUST_CONTINENTAL",
    "CRUST_OCEANIC",
    "Frame",
    "PLATE_PALETTE",
    "Plate",
    "SimConfig",
    "Snapshot",
    "WorldRect",
    "apply_collisions",
    "apply_contact_constraints",
    "apply_velocity_damping",
    "build_initial_state",
    "crust_type_code",
    "crust_type_name",
    "cull_outside_domain",
    "detect_overlaps",
    "divergent_fill",
    "drift_positions",
    "load_sim_config",
    "load_sim_config_from_path",
    "particle_elevation_km",
    "render_elevation_png",
    "render_initial_state",
    "render_particles_png",
    "render_single_plate_png",
    "render_snapshot_particles",
    "render_snapshot_single_plate",
    "render_snapshot_voronoi",
    "render_voronoi_png",
    "step_drift_and_apply_boundary",
    "step_drift_and_cull",
    "wrap_positions",
]
