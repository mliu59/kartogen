"""Test fixtures for the terrain generation suite.

Loads the default world-gen parameters from ``config/worldgen.toml`` so tests
track real config changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worldgen import generate
from worldgen.config_loader import load_worldgen_config
from worldgen.pipeline import GeneratedWorld
from worldgen.types import WorldgenConfig


@pytest.fixture(scope="session")
def default_worldgen_config() -> WorldgenConfig:
    """Worldgen parameters loaded from ``config/worldgen.toml``."""
    return load_worldgen_config(
        Path(__file__).parent.parent / "config" / "worldgen.toml"
    )


@pytest.fixture(scope="session")
def small_world(default_worldgen_config: WorldgenConfig) -> GeneratedWorld:
    """A small (radius 12) world generated with seed 42.

    Session-scoped so we only pay the generation cost once. Radius 12 is small
    enough for fast tests (~430 hexes) but large enough that every layer has
    interesting structure (continent, mountains, rivers, climate gradient).
    """
    return generate(radius=12, config=default_worldgen_config, seed=42)


@pytest.fixture(scope="session")
def medium_world(default_worldgen_config: WorldgenConfig) -> GeneratedWorld:
    """A radius-30 world (~2800 hexes). Use sparingly — slower."""
    return generate(radius=30, config=default_worldgen_config, seed=42)
