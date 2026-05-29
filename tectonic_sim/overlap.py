"""Cross-plate proximity detection.

Thin wrapper that builds a ``BucketGrid`` sized for the collision query
and returns the pair indices the collision step will iterate. Kept as
its own module so the wiring (build grid → query) is in one place and
not duplicated across simulate.py / collisions.py / demos.

The query radius is taken from ``SimConfig.overlap_radius_km`` (derived
as 1.5 × particle_spacing_km — see ``types.OVERLAP_RADIUS_MULTIPLIER``).
The grid's cell size is set equal to the query radius so the spatial
index's 3×3 cell halo covers the full query range.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.spatial import BucketGrid
from tectonic_sim.types import SimConfig, WorldRect


def detect_overlaps(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    sim_config: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """All unordered cross-plate particle pairs within ``overlap_radius_km``.

    Builds a bucket grid sized to ``sim_config.overlap_radius_km`` and
    runs ``cross_label_pairs_within(plate_id, overlap_radius_km)``.

    Returns parallel int32 arrays ``(i, j)`` with ``plate_id[i] !=
    plate_id[j]``. Each unordered pair appears exactly once but the
    index order within a pair is not guaranteed. Empty arrays if no
    pairs are detected.

    Pure function of (domain, positions, plate_id, sim_config) — same
    inputs yield the same pair set, byte-identical.
    """
    radius = sim_config.overlap_radius_km
    wrap = (sim_config.boundary_mode == "wrap")
    grid = BucketGrid.build(positions_km, domain, radius, wrap=wrap)
    return grid.cross_label_pairs_within(plate_id, radius)
