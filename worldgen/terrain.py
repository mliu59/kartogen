"""Terrain types and their properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerrainType:
    """A terrain type with its properties.

    Per-good yields are NOT a property of the terrain type — they are computed
    per hex by ``goods.yield_model.compute_yields`` from the worldgen
    pipeline's outputs (crop suitability, deposits, water flags) plus the
    configured ``[goods.extraction.*]`` rules.
    """

    name: str
    movement_cost: float


# Canonical terrain names produced by the world generator. Configuration
# (``default.toml``) must define a ``[terrain.types.X]`` entry for each.
# Generated worlds use these names; engine code keys off them.
TERRAIN_NAMES: tuple[str, ...] = (
    # Water
    "deep_ocean",
    "ocean",
    "coast",
    "lake",
    "river",
    # Lowland biomes
    "plains",
    "grassland",
    "savanna",
    "desert",
    "tundra",
    # Forested
    "temperate_forest",
    "taiga",
    "jungle",
    # Elevated
    "hills",
    "mountain",
    "snow_peak",
)
