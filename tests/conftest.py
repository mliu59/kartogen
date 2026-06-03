"""Test fixtures for the terrain generation suite.

Loads the default world-gen parameters from ``config/kartogen.toml`` so tests
track real config changes. Test worlds override ``KartogenConfig.world``
with a smaller ``WorldShape`` via ``dataclasses.replace`` so we don't pay
for the full default world dimensions in every test.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from kartogen import generate
from kartogen.config_loader import load_kartogen_config
from kartogen.pipeline import GeneratedWorld
from kartogen.types import KartogenConfig, WorldShape


def _with_shape(
    cfg: KartogenConfig, width_km: float, height_km: float,
) -> KartogenConfig:
    """Return a copy of ``cfg`` with its ``world`` shape replaced.

    The edge-smoothing override lives on ``default_kartogen_config`` so
    every downstream fixture and every test that builds a world from
    that fixture inherits it — no per-fixture handling needed here.
    """
    return dataclasses.replace(
        cfg, world=WorldShape(width_km=width_km, height_km=height_km),
    )


@pytest.fixture(scope="session")
def default_kartogen_config() -> KartogenConfig:
    """Kartogen parameters loaded from ``config/kartogen.toml`` — with
    the non-physics edge-smoothing pass DISABLED.

    Edge smoothing is a Perlin-modulated Gaussian blur on crust
    thickness whose kernel is sized in physical km. Production-tuned
    defaults (kernel ≈25 km) are correct for production-sized worlds
    (≥1000 km) but become catastrophic on the tiny worlds these tests
    use (~100-300 km) — sigma in cells gets large enough relative to
    the grid to flatten land/ocean diversity. Every other kartogen
    test treats thickness as the un-smoothed simulation output; we
    keep that contract by turning smoothing off here.

    The smoothing's own correctness lives in
    ``tectonic_sim/tests/test_edge_smoothing.py`` (unit tests of the
    operator). End-to-end smoothing behaviour is exercised by the
    ``python -m kartogen`` smoke runs on the production-sized config.
    """
    cfg = load_kartogen_config(
        Path(__file__).parent.parent / "config" / "kartogen.toml"
    )
    return dataclasses.replace(
        cfg,
        tectonics=dataclasses.replace(
            cfg.tectonics,
            edge_smoothing_apply_t0=False,
            edge_smoothing_apply_tfinal=False,
        ),
    )


@pytest.fixture(scope="session")
def small_world_config(default_kartogen_config: KartogenConfig) -> KartogenConfig:
    """Small (120×120 km) world config for fast tests. ~400-500 hexes at the
    default 5 km hex size — comparable to the previous radius=12 hex world."""
    return _with_shape(default_kartogen_config, width_km=120.0, height_km=120.0)


@pytest.fixture(scope="session")
def medium_world_config(default_kartogen_config: KartogenConfig) -> KartogenConfig:
    """Medium (300×300 km) world config. ~2800 hexes at 5 km/hex —
    comparable to the previous radius=30 hex world."""
    return _with_shape(default_kartogen_config, width_km=300.0, height_km=300.0)


@pytest.fixture(scope="session")
def small_world(small_world_config: KartogenConfig) -> GeneratedWorld:
    """A small (120×120 km) world generated with seed 42.

    Session-scoped so we only pay the generation cost once. Small enough
    for fast tests but large enough that every layer has interesting
    structure (continent, mountains, rivers, climate gradient).
    """
    return generate(config=small_world_config, seed=42)


@pytest.fixture(scope="session")
def medium_world(medium_world_config: KartogenConfig) -> GeneratedWorld:
    """A medium (300×300 km) world. Use sparingly — slower."""
    return generate(config=medium_world_config, seed=42)
