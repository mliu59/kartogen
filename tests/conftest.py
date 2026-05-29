"""Test fixtures for the terrain generation suite.

Loads the default world-gen parameters from ``config/worldgen.toml`` so tests
track real config changes. Test worlds override ``WorldgenConfig.world``
with a smaller ``WorldShape`` via ``dataclasses.replace`` so we don't pay
for the full default world dimensions in every test.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from worldgen import generate
from worldgen.config_loader import load_worldgen_config
from worldgen.pipeline import GeneratedWorld
from worldgen.types import WorldgenConfig, WorldShape


def _with_shape(
    cfg: WorldgenConfig, width_km: float, height_km: float,
) -> WorldgenConfig:
    """Return a copy of ``cfg`` with its ``world`` shape replaced."""
    return dataclasses.replace(
        cfg, world=WorldShape(width_km=width_km, height_km=height_km),
    )


@pytest.fixture(scope="session")
def default_worldgen_config() -> WorldgenConfig:
    """Worldgen parameters loaded from ``config/worldgen.toml``."""
    return load_worldgen_config(
        Path(__file__).parent.parent / "config" / "worldgen.toml"
    )


@pytest.fixture(scope="session")
def small_world_config(default_worldgen_config: WorldgenConfig) -> WorldgenConfig:
    """Small (120×120 km) world config for fast tests. ~400-500 hexes at the
    default 5 km hex size — comparable to the previous radius=12 hex world."""
    return _with_shape(default_worldgen_config, width_km=120.0, height_km=120.0)


@pytest.fixture(scope="session")
def medium_world_config(default_worldgen_config: WorldgenConfig) -> WorldgenConfig:
    """Medium (300×300 km) world config. ~2800 hexes at 5 km/hex —
    comparable to the previous radius=30 hex world."""
    return _with_shape(default_worldgen_config, width_km=300.0, height_km=300.0)


@pytest.fixture(scope="session")
def small_world(small_world_config: WorldgenConfig) -> GeneratedWorld:
    """A small (120×120 km) world generated with seed 42.

    Session-scoped so we only pay the generation cost once. Small enough
    for fast tests but large enough that every layer has interesting
    structure (continent, mountains, rivers, climate gradient).
    """
    return generate(config=small_world_config, seed=42)


@pytest.fixture(scope="session")
def medium_world(medium_world_config: WorldgenConfig) -> GeneratedWorld:
    """A medium (300×300 km) world. Use sparingly — slower."""
    return generate(config=medium_world_config, seed=42)
