"""Terrain types and their properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerrainType:
    """A terrain type with its properties."""

    name: str
    movement_cost: float
    yields: dict[str, float]  # good_name → base yield per population-unit per tick
