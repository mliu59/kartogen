"""Hydrology layer: sink-fill (priority-flood), D6 flow accumulation,
rivers, and lakes."""

from __future__ import annotations

import heapq

from worldgen.hex import Hex
from worldgen.types import (
    ElevationLayer,
    HydrologyLayer,
    SeaLayer,
    WorldgenConfig,
)


_FILL_EPSILON = 1e-6


def _priority_flood(
    elevation: dict[Hex, float],
    is_ocean: dict[Hex, bool],
    sea_level: float,
) -> tuple[dict[Hex, float], dict[Hex, bool]]:
    """Fill depressions using priority-flood + epsilon (Barnes et al. 2014).

    Each filled cell is raised by at least ``_FILL_EPSILON`` above its parent
    so that flat regions retain a monotonic descent path back to the ocean.
    Without epsilon, lake interiors and plateaus are perfectly flat and
    downhill flow stalls — producing zero rivers.

    Returns:
        filled: elevation with sinks filled to spill height (+ epsilon tilt).
        is_lake: True for land hexes whose elevation was raised by filling.
    """
    filled: dict[Hex, float] = dict(elevation)
    closed: set[Hex] = set()
    is_lake: dict[Hex, bool] = {h: False for h in elevation}

    # Tiebreakers in the heap key ensure deterministic order.
    pq: list[tuple[float, int, int, Hex]] = []

    # Seed the queue with every ocean hex (they are the natural outlets).
    for h in elevation:
        if is_ocean[h]:
            heapq.heappush(pq, (elevation[h], h.q, h.r, h))
            closed.add(h)

    while pq:
        e, _, _, h = heapq.heappop(pq)
        for n in h.neighbors():
            if n in closed or n not in elevation:
                continue
            # Epsilon tilt: neighbor's fill elevation must exceed parent's,
            # creating a guaranteed downhill path back through the flood front.
            tilted = e + _FILL_EPSILON
            new_e = max(elevation[n], tilted)
            if new_e > elevation[n]:
                is_lake[n] = True
            filled[n] = new_e
            closed.add(n)
            heapq.heappush(pq, (new_e, n.q, n.r, n))

    return filled, is_lake


def _flow_direction(
    h: Hex,
    filled: dict[Hex, float],
) -> Hex | None:
    """Return the steepest downhill neighbor, or None if no neighbor is lower
    or equal. Ties broken by neighbor canonical order."""
    elev = filled[h]
    best: Hex | None = None
    best_drop = 0.0
    for n in h.neighbors():
        if n not in filled:
            continue
        drop = elev - filled[n]
        if drop > best_drop:
            best_drop = drop
            best = n
    return best


def compute(
    elevation: ElevationLayer,
    sea: SeaLayer,
    precipitation: dict[Hex, float],
    config: WorldgenConfig,
) -> HydrologyLayer:
    """Compute flow directions, flow accumulation, rivers and lakes.

    Lake hexes are tagged where the priority-flood raised the surface above
    the original elevation by at least ``lake_min_depth``. River hexes are
    tagged where accumulated upstream-hex count exceeds ``river_drainage_threshold``.
    """
    filled, is_lake_raw = _priority_flood(
        elevation.elevation, sea.is_ocean, elevation.sea_level
    )

    # Apply lake minimum depth: only count hexes raised by at least lake_min_depth.
    is_lake: dict[Hex, bool] = {}
    for h, raised in is_lake_raw.items():
        if not raised or sea.is_ocean[h]:
            is_lake[h] = False
            continue
        depth = filled[h] - elevation.elevation[h]
        is_lake[h] = depth >= config.lake_min_depth

    # Flow direction
    downstream: dict[Hex, Hex | None] = {}
    for h in filled:
        if sea.is_ocean[h]:
            downstream[h] = None
            continue
        downstream[h] = _flow_direction(h, filled)

    # Flow accumulation: topological sort by filled elevation descending,
    # then push each cell's accumulation into its downhill neighbor.
    sorted_hexes = sorted(filled.keys(), key=lambda h: (-filled[h], h.q, h.r))
    flow: dict[Hex, int] = {h: 1 for h in filled}  # each hex contributes 1 unit
    for h in sorted_hexes:
        if sea.is_ocean[h]:
            continue
        down = downstream[h]
        if down is not None:
            flow[down] = flow.get(down, 0) + flow[h]

    # River mask: above threshold and not ocean / lake.
    is_river: dict[Hex, bool] = {}
    for h in filled:
        if sea.is_ocean[h] or is_lake[h]:
            is_river[h] = False
            continue
        is_river[h] = flow[h] >= config.river_drainage_threshold

    return HydrologyLayer(
        filled_elevation=filled,
        is_lake=is_lake,
        is_river=is_river,
        flow_accumulation=flow,
        downstream=downstream,
    )
