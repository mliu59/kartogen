"""World footprint: rectangular hex-set construction and geometric helpers.

Coordinate convention (matches the rest of the package):

  pixel_x(h) = 1.5 * h.q * hex_size_km        (km)
  pixel_y(h) = sqrt(3) * (h.r + h.q/2) * hex_size_km   (km)

The world is the set of hexes whose pixel centre falls in
``[-width_km/2, +width_km/2] × [-height_km/2, +height_km/2]``. Hex (0, 0)
is always inside any positive-area world.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from kartogen.hex import Hex
from kartogen.types import WorldShape

_SQRT3 = math.sqrt(3.0)


def map_half_extents_km(
    hexes: Iterable[Hex], hex_size_km: float,
) -> tuple[float, float]:
    """Half-width and half-height (km) of the bounding box of ``hexes``.

    Used by layers downstream of canvas selection so they don't need a
    separate world-shape parameter — the input hex set *is* the map.
    Assumes the world is centred at the origin (which it is by
    construction); the extents are ``max(|pixel_x|)`` and ``max(|pixel_y|)``
    across the set.

    Returns ``(0.0, 0.0)`` for an empty hex set so callers can short-circuit.
    """
    max_abs_x = 0.0
    max_abs_y = 0.0
    for h in hexes:
        x = 1.5 * h.q * hex_size_km
        y = _SQRT3 * (h.r + h.q / 2.0) * hex_size_km
        if abs(x) > max_abs_x:
            max_abs_x = abs(x)
        if abs(y) > max_abs_y:
            max_abs_y = abs(y)
    return max_abs_x, max_abs_y


def hex_to_xy_km(h: Hex, hex_size_km: float) -> tuple[float, float]:
    """Flat-top hex axial coord → cartesian pixel centre in km.

    Canonical projection used everywhere kartogen converts a ``Hex`` to
    physical (x, y) km.
    """
    return 1.5 * h.q * hex_size_km, _SQRT3 * (h.r + h.q / 2.0) * hex_size_km


def rect_world_hexes(shape: WorldShape, hex_size_km: float) -> list[Hex]:
    """Return all hexes whose pixel centre falls inside ``shape``'s rectangle.

    Iterates a generous axial bounding box and filters by exact pixel
    bounds, so the returned set follows the rectangle precisely (no
    diagonal staircase artifacts beyond the lattice's inherent granularity).
    """
    hw = shape.half_width_km
    hh = shape.half_height_km
    # Outer q range: pixel_x = 1.5 * q * hex_size_km, so q ≤ hw/(1.5*hs).
    q_max = int(math.floor(hw / (1.5 * hex_size_km))) + 1
    # For each q, the r range is determined by pixel_y ≤ hh.
    # pixel_y = sqrt(3) * (r + q/2) * hex_size_km ≤ hh
    # → r ≤ hh / (sqrt(3) * hex_size_km) - q/2
    r_span = hh / (_SQRT3 * hex_size_km)

    out: list[Hex] = []
    for q in range(-q_max, q_max + 1):
        px = 1.5 * q * hex_size_km
        if abs(px) > hw:
            continue
        r_min = int(math.ceil(-r_span - q / 2.0))
        r_max = int(math.floor(r_span - q / 2.0))
        for r in range(r_min, r_max + 1):
            py = _SQRT3 * (r + q / 2.0) * hex_size_km
            if abs(py) <= hh:
                out.append(Hex(q, r))
    return out


def world_pixel_bounds(
    shape: WorldShape, hex_px: float, hex_size_km: float,
) -> tuple[int, int]:
    """Renderer canvas size (pixels) for a world of this shape.

    ``hex_px`` is the pixel scale per unit hex (the renderer's outer-radius
    convention); ``hex_size_km`` converts the world's km dimensions to the
    same units. A small margin of one hex on every side keeps the
    rectangle's edge hexes from being clipped by the canvas border.
    """
    px_per_km = hex_px / hex_size_km
    w = int(round(shape.width_km * px_per_km + 4 * hex_px))
    h = int(round(shape.height_km * px_per_km + 4 * hex_px))
    return w, h
