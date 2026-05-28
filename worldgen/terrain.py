"""Terrain types and their properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerrainType:
    """A terrain type with its properties."""

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
