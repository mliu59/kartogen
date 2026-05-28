"""Sea-level mask: ocean / coast tagging from the elevation layer."""

from __future__ import annotations

from worldgen.hex import Hex
from worldgen.types import ElevationLayer, SeaLayer


def compute(elevation: ElevationLayer) -> SeaLayer:
    """Compute ocean mask and coast mask.

    A hex is ocean iff its elevation is below sea level.
    A hex is coast iff it is land and at least one neighbor is ocean.
    """
    is_ocean: dict[Hex, bool] = {h: elevation.is_ocean(h) for h in elevation.elevation}
    is_coast: dict[Hex, bool] = {}
    for h, ocean in is_ocean.items():
        if ocean:
            is_coast[h] = False
            continue
        on_coast = any(is_ocean.get(n, False) for n in h.neighbors())
        is_coast[h] = on_coast
    return SeaLayer(is_ocean=is_ocean, is_coast=is_coast)
