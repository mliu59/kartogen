"""Ocean layer: gyre-based current direction, temperature anomaly,
continentality (Tier 2 climate).

Each connected ocean basin is split by hemisphere into one or two gyres,
each rotating CW in the northern hemisphere and CCW in the southern (the
Coriolis-driven sign on Earth). Per ocean hex, the current direction is the
tangent of (hex − gyre_centre) × rotation sign.

The temperature anomaly carried by each current is computed by sampling
the planet's latitudinal temperature ``current_persistence_km`` upstream
along the current direction. If the upstream sample is warmer (closer to
the equator), the current is a *warm current* and the hex picks up a
positive anomaly. If cooler, it's a *cold current* and picks up a
negative anomaly. This single-pass formula reproduces the major Earth
pattern: western boundaries of subtropical gyres are warm (Gulf Stream,
Kuroshio); eastern boundaries are cold (California, Humboldt, Canary).

Coastal land hexes inherit a fraction of the nearest ocean hex's anomaly,
decaying exponentially with distance inland. Continentality is exposed
as ``distance_to_ocean_km`` so the climate layer can dry the precipitation
floor far from the sea.

Annual-mean snapshot only — no seasonality.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from kartogen.hex import Hex
from kartogen.types import SeaLayer, KartogenConfig
from kartogen.world import hex_to_xy_km, map_half_extents_km


_SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class OceanLayer:
    """Output of the ocean currents layer.

    ``current_direction[h]`` is the unit current vector at ocean hex h in
    cartesian space (same projection the climate / preview layers use).
    Land hexes are not present in this dict.

    ``current_temp_anomaly[h]`` is the °C anomaly added on top of the
    latitudinal baseline temperature at ocean hex h. Land hexes are not
    present.

    ``coastal_temp_anomaly[h]`` is the °C anomaly transferred to land hex h
    from the nearest ocean hex, decayed by distance. 0 for ocean hexes.

    ``distance_to_ocean_km[h]`` is the BFS distance (km) from hex h to the
    nearest ocean hex. 0 for ocean hexes.

    ``gyre_id[h]`` is the gyre index for ocean hex h. Not present for land.
    """

    current_direction: dict[Hex, tuple[float, float]]
    current_temp_anomaly: dict[Hex, float]
    coastal_temp_anomaly: dict[Hex, float]
    distance_to_ocean_km: dict[Hex, float]
    gyre_id: dict[Hex, int]
    n_gyres: int


def _hex_latitude_deg(
    h: Hex, half_height_km: float, config: KartogenConfig,
) -> float:
    """Same convention as ``climate.hex_latitude_deg`` — kept duplicated to
    avoid pulling the climate module into this layer's import graph.
    ``half_height_km`` is derived by ``compute`` from the input hex set."""
    if half_height_km <= 0:
        return 0.5 * (config.map_lat_min + config.map_lat_max)
    pixel_y_km = _SQRT3 * (h.r + h.q / 2.0) * config.hex_size_km
    fraction_south = (pixel_y_km + half_height_km) / (2.0 * half_height_km)
    fraction_south = max(0.0, min(1.0, fraction_south))
    return (
        config.map_lat_max
        - fraction_south * (config.map_lat_max - config.map_lat_min)
    )


def _lat_baseline_temp(lat_deg: float, config: KartogenConfig) -> float:
    """Planet's annual-mean temperature at the given latitude.

    Mirrors ``climate.compute_temperature``'s latitudinal formula so the
    ocean layer reasons about the same gradient.
    """
    polar_fraction = min(1.0, abs(lat_deg) / 90.0)
    return (
        config.equator_temp_c
        + (config.polar_temp_c - config.equator_temp_c) * (polar_fraction**1.3)
    )


def _connected_ocean_basins(
    sea: SeaLayer, all_hexes: set[Hex],
) -> list[list[Hex]]:
    """Find connected components of ocean hexes (4 / 6-neighbour BFS)."""
    visited: set[Hex] = set()
    basins: list[list[Hex]] = []
    for start in all_hexes:
        if start in visited or not sea.is_ocean.get(start, False):
            continue
        basin: list[Hex] = []
        queue: deque[Hex] = deque([start])
        visited.add(start)
        while queue:
            h = queue.popleft()
            basin.append(h)
            for nb in h.neighbors():
                if (
                    nb in all_hexes
                    and nb not in visited
                    and sea.is_ocean.get(nb, False)
                ):
                    visited.add(nb)
                    queue.append(nb)
        basins.append(basin)
    return basins


def _split_basins_into_gyres(
    basins: list[list[Hex]],
    half_height_km: float,
    config: KartogenConfig,
    hex_size_km: float,
) -> list[tuple[list[Hex], tuple[float, float], int]]:
    """For each basin, split by hemisphere (lat 0°) into 1–2 gyres.

    Returns a list of ``(hexes, centre_xy_km, rotation_sign)`` tuples, where
    rotation_sign is +1 for CW (NH) and −1 for CCW (SH).
    """
    gyres: list[tuple[list[Hex], tuple[float, float], int]] = []
    for basin in basins:
        northern: list[Hex] = []
        southern: list[Hex] = []
        for h in basin:
            lat = _hex_latitude_deg(h, half_height_km, config)
            if lat >= 0.0:
                northern.append(h)
            else:
                southern.append(h)
        for sub_hexes, rotation in ((northern, +1), (southern, -1)):
            if not sub_hexes:
                continue
            # Centroid in cartesian km.
            sx = sum(hex_to_xy_km(h, hex_size_km)[0] for h in sub_hexes)
            sy = sum(hex_to_xy_km(h, hex_size_km)[1] for h in sub_hexes)
            cx = sx / len(sub_hexes)
            cy = sy / len(sub_hexes)
            gyres.append((sub_hexes, (cx, cy), rotation))
    return gyres


def _current_direction_at(
    h: Hex,
    centre_xy: tuple[float, float],
    rotation: int,
    hex_size_km: float,
) -> tuple[float, float]:
    """Unit tangent vector around the gyre centre.

    NH gyres rotate **visually clockwise** when seen on a north-up map; SH
    gyres rotate visually counter-clockwise. Our screen projection uses +y
    = south (y points down in the image). In that frame, visual CW
    corresponds to the math transform (rx, ry) → (−ry, rx) (i.e. *math
    CCW*, because flipping y flips the rotation sense).

    Verification at the east side of a NH gyre (rx > 0, ry = 0):
        direction = (0, rx) = (0, positive) = SOUTH
    Going south, upstream is to the north (lower r, higher latitude),
    which is colder than local. anomaly = upstream − local < 0 ⇒ COLD.
    That's the canonical NH east-of-gyre = west-coast-of-the-continent-to-
    the-east = Canary / California / Humboldt cold-current pattern.

    At the west side of the same gyre (rx < 0, ry = 0):
        direction = (0, rx) = (0, negative) = NORTH
    Upstream south, lower lat, warmer ⇒ WARM. That's the Gulf-Stream /
    Kuroshio / Brazil-Current warm-boundary pattern.
    """
    hx, hy = hex_to_xy_km(h, hex_size_km)
    rx = hx - centre_xy[0]
    ry = hy - centre_xy[1]
    norm = math.hypot(rx, ry)
    if norm == 0.0:
        return (0.0, 0.0)
    if rotation > 0:
        # NH, visual CW in y-down screen coords = math CCW: (x, y) → (−y, x).
        tx, ty = -ry, rx
    else:
        # SH, visual CCW = math CW: (x, y) → (y, −x).
        tx, ty = ry, -rx
    tnorm = math.hypot(tx, ty)
    return (tx / tnorm, ty / tnorm)


def _nearesthex_to_xy_km(
    x_km: float, y_km: float, hex_size_km: float,
) -> Hex:
    """Approximate inverse of ``hex_to_xy_km`` — cube round."""
    x = x_km / hex_size_km
    y = y_km / hex_size_km
    qf = x / 1.5
    rf = y / _SQRT3 - qf / 2.0
    sf = -qf - rf
    qi = round(qf)
    ri = round(rf)
    si = round(sf)
    dq = abs(qi - qf)
    dr = abs(ri - rf)
    ds = abs(si - sf)
    if dq > dr and dq > ds:
        qi = -ri - si
    elif dr > ds:
        ri = -qi - si
    return Hex(int(qi), int(ri))


def _compute_current_anomaly(
    h: Hex,
    direction: tuple[float, float],
    half_height_km: float,
    config: KartogenConfig,
    hex_size_km: float,
    persistence_km: float,
    strength: float,
    cap: float,
) -> float:
    """Sample upstream latitude temperature; anomaly = (upstream − local) × strength."""
    if direction == (0.0, 0.0):
        return 0.0
    hx, hy = hex_to_xy_km(h, hex_size_km)
    ux = hx - direction[0] * persistence_km
    uy = hy - direction[1] * persistence_km
    upstream_hex = _nearesthex_to_xy_km(ux, uy, hex_size_km)
    upstream_lat = _hex_latitude_deg(upstream_hex, half_height_km, config)
    local_lat = _hex_latitude_deg(h, half_height_km, config)
    upstream_t = _lat_baseline_temp(upstream_lat, config)
    local_t = _lat_baseline_temp(local_lat, config)
    anomaly = (upstream_t - local_t) * strength
    if anomaly > cap:
        anomaly = cap
    elif anomaly < -cap:
        anomaly = -cap
    return anomaly


def _bfs_distance_to_ocean(
    sea: SeaLayer,
    all_hexes: set[Hex],
    hex_size_km: float,
) -> tuple[dict[Hex, float], dict[Hex, Hex]]:
    """Multi-source BFS from every ocean hex.

    Returns (distance_km_per_hex, nearest_ocean_hex_per_hex). Distance is
    measured along the hex grid in physical km (per-step = hex_size_km × √3).
    Ocean hexes themselves get distance 0 and "nearest ocean" = themselves.
    """
    distance: dict[Hex, float] = {h: math.inf for h in all_hexes}
    nearest: dict[Hex, Hex] = {}
    queue: deque[Hex] = deque()
    step_km = hex_size_km * _SQRT3
    for h in all_hexes:
        if sea.is_ocean.get(h, False):
            distance[h] = 0.0
            nearest[h] = h
            queue.append(h)
    while queue:
        h = queue.popleft()
        d = distance[h]
        n = nearest[h]
        for nb in h.neighbors():
            if nb not in all_hexes:
                continue
            new_d = d + step_km
            if new_d < distance[nb]:
                distance[nb] = new_d
                nearest[nb] = n
                queue.append(nb)
    return distance, nearest


def compute(
    sea: SeaLayer,
    hexes: list[Hex],
    config: KartogenConfig,
) -> OceanLayer:
    """Build the OceanLayer.

    Pure function of (sea, hexes, config). No RNG used — the gyre geometry
    is determined by the connected-component layout, which is in turn
    deterministic given the upstream world state. Map dimensions are
    derived from the input hex set; no reference to ``config.world``.
    """
    all_hexes = set(hexes)
    cfg = config.ocean
    half_w_km, half_h_km = map_half_extents_km(hexes, config.hex_size_km)

    basins = _connected_ocean_basins(sea, all_hexes)
    gyres = _split_basins_into_gyres(
        basins, half_h_km, config, config.hex_size_km,
    )

    # Cap upstream-sample persistence at a fraction of the world's smaller
    # half-dimension. The configured default (~2000 km) is
    # Earth-Gulf-Stream-scale; on tiny test worlds it would walk off-grid
    # and clamp at the wrong pole, flipping the anomaly sign. Capping at
    # 80 % of the smaller half-dimension keeps the sample inside the
    # simulated lat range.
    half_extent_km = max(1.0, min(half_w_km, half_h_km))
    effective_persistence = min(
        cfg.current_persistence_km, half_extent_km * 0.8,
    )

    current_direction: dict[Hex, tuple[float, float]] = {}
    current_temp_anomaly: dict[Hex, float] = {}
    gyre_id: dict[Hex, int] = {}
    for idx, (sub_hexes, centre_xy, rotation) in enumerate(gyres):
        for h in sub_hexes:
            direction = _current_direction_at(
                h, centre_xy, rotation, config.hex_size_km,
            )
            current_direction[h] = direction
            current_temp_anomaly[h] = _compute_current_anomaly(
                h, direction, half_h_km, config, config.hex_size_km,
                effective_persistence,
                cfg.current_anomaly_strength,
                cfg.max_current_anomaly_c,
            )
            gyre_id[h] = idx

    distance_to_ocean_km, nearest_ocean = _bfs_distance_to_ocean(
        sea, all_hexes, config.hex_size_km,
    )

    coastal_temp_anomaly: dict[Hex, float] = {}
    pickup = cfg.coastal_pickup_fraction
    decay = cfg.coastal_decay_km
    for h in hexes:
        if sea.is_ocean.get(h, False):
            coastal_temp_anomaly[h] = 0.0
            continue
        if h not in nearest_ocean:
            coastal_temp_anomaly[h] = 0.0
            continue
        src = nearest_ocean[h]
        src_anomaly = current_temp_anomaly.get(src, 0.0)
        d = distance_to_ocean_km[h]
        coastal_temp_anomaly[h] = src_anomaly * pickup * math.exp(-d / decay)

    return OceanLayer(
        current_direction=current_direction,
        current_temp_anomaly=current_temp_anomaly,
        coastal_temp_anomaly=coastal_temp_anomaly,
        distance_to_ocean_km=distance_to_ocean_km,
        gyre_id=gyre_id,
        n_gyres=len(gyres),
    )
