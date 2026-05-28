"""Layered, deterministic terrain generation pipeline.

Each layer is a pure function of (prior_state, seeded_rng, config). Layers:

    0. plates (optional)  — Voronoi plate field with motion + boundary tags
    1. elevation          — fBm + ridged + domain warp + radial falloff
    2. sea level          — quantile threshold, ocean / coast tagging
    3. temperature        — latitude gradient + elevation lapse
    4. precipitation      — prevailing winds, moisture sweep, orographic uplift
    5. hydrology          — sink-fill (priority-flood) + D6 flow accum + rivers + lakes
    6. biome              — Whittaker(T, P) lookup with overrides → terrain

The entry point is ``pipeline.generate(config, seed) -> GeneratedWorld``.
"""

from worldgen.pipeline import GeneratedWorld, generate
from worldgen.types import HexData, WorldgenConfig

__all__ = ["GeneratedWorld", "HexData", "WorldgenConfig", "generate"]
