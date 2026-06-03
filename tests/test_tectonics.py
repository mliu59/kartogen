"""Tests for the time-stepped plate-tectonics simulation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import pytest
from worldgen import generate
from worldgen.tectonics import (
    PLATE_TYPE_CONTINENTAL,
    PLATE_TYPE_OCEANIC,
    LithosphereColumn,
    LithosphereState,
    column_to_elevation_km,
)
from worldgen.types import TectonicsConfig, WorldgenConfig, WorldShape

# ---------------------------------------------------------------------------
# Closed-form elevation map (``column_to_elevation_km``) — exercised against
# the live tectonics config so the constants stay in sync with the loaded
# TOML rather than a hand-built fixture.
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.slow  # full generate()/sim per test
def _tectonics(default_worldgen_config: WorldgenConfig) -> TectonicsConfig:
    return cast(TectonicsConfig, default_worldgen_config.tectonics)


def test_old_oceanic_crust_subsides_below_young(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Half-space cooling: older oceanic crust must sit deeper than fresh ridge crust."""
    cfg = _tectonics(default_worldgen_config)
    young = LithosphereColumn(
        crust_type=PLATE_TYPE_OCEANIC, thickness_km=cfg.oceanic_thickness_km,
        age_myr=0.0,
    )
    old = LithosphereColumn(
        crust_type=PLATE_TYPE_OCEANIC, thickness_km=cfg.oceanic_thickness_km,
        age_myr=80.0,
    )
    young_elev = column_to_elevation_km(young, cfg)
    old_elev = column_to_elevation_km(old, cfg)
    assert young_elev == pytest.approx(-cfg.ridge_depth_km)
    expected_depth = cfg.ridge_depth_km + cfg.ridge_subsidence_rate * math.sqrt(80.0)
    assert old_elev == pytest.approx(-min(expected_depth, cfg.max_ocean_depth_km))
    assert old_elev < young_elev


def test_ocean_floor_depth_caps_at_max(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Subsidence is capped at ``max_ocean_depth_km`` — abyssal plains don't sink forever."""
    cfg = _tectonics(default_worldgen_config)
    ancient = LithosphereColumn(
        crust_type=PLATE_TYPE_OCEANIC, thickness_km=cfg.oceanic_thickness_km,
        age_myr=10_000.0,  # absurdly old
    )
    assert column_to_elevation_km(ancient, cfg) == pytest.approx(-cfg.max_ocean_depth_km)


def test_continental_isostasy_thicker_means_higher(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Thicker continental crust → higher elevation; equal to reference → sea level."""
    cfg = _tectonics(default_worldgen_config)
    at_ref = LithosphereColumn(
        crust_type=PLATE_TYPE_CONTINENTAL,
        thickness_km=cfg.continental_reference_thickness_km,
        age_myr=0.0,
    )
    assert column_to_elevation_km(at_ref, cfg) == pytest.approx(0.0)

    thick = LithosphereColumn(
        crust_type=PLATE_TYPE_CONTINENTAL,
        thickness_km=cfg.continental_reference_thickness_km + 10.0,
        age_myr=0.0,
    )
    expected = 10.0 * cfg.continental_isostasy_factor
    assert column_to_elevation_km(thick, cfg) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# End-to-end pipeline checks via ``generate`` — these exercise the polygon-
# sim bridge, not the closed-form formulas.
# ---------------------------------------------------------------------------


def test_simulation_deterministic_under_fixed_seed(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Identical (config, seed) → identical lithosphere state."""
    a = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=80.0, height_km=80.0)), seed=42)
    b = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=80.0, height_km=80.0)), seed=42)
    assert a.lithosphere is not None and b.lithosphere is not None
    for h in a.hexes:
        assert a.lithosphere.elevation_km[h] == b.lithosphere.elevation_km[h]
        assert a.lithosphere.columns[h] == b.lithosphere.columns[h]
        assert a.lithosphere.plate_id[h] == b.lithosphere.plate_id[h]


def test_sea_level_threshold_classifies_consistently(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Hexes above sea_level_km in the lithosphere should map to is_ocean=False
    (and the converse). The normalized elevation field puts sea level at 0."""
    world = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=100.0, height_km=100.0)), seed=42)
    lith = cast(LithosphereState, world.lithosphere)
    sea_km = _tectonics(default_worldgen_config).sea_level_km

    above = below = 0
    for h, d in world.hexes.items():
        if lith.elevation_km[h] > sea_km:
            assert not d.is_ocean, (
                f"hex {h} elev_km={lith.elevation_km[h]:.3f} > sea {sea_km} but marked ocean"
            )
            above += 1
        elif lith.elevation_km[h] < sea_km:
            assert d.is_ocean, (
                f"hex {h} elev_km={lith.elevation_km[h]:.3f} < sea {sea_km} but marked land"
            )
            below += 1
    # Sanity: both sides have meaningful population (the threshold actually splits).
    assert above > 0 and below > 0


def test_changing_sea_level_threshold_shifts_land_fraction(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Raising sea_level_km drowns land; lowering it exposes ocean floor."""
    base_world = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    base_land = sum(1 for d in base_world.hexes.values() if not d.is_ocean)

    high_sea = replace(
        _tectonics(default_worldgen_config),
        sea_level_km=2.0,
    )
    high_cfg = replace(default_worldgen_config, tectonics=high_sea)
    high_world = generate(config=replace(high_cfg, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    high_land = sum(1 for d in high_world.hexes.values() if not d.is_ocean)

    assert high_land < base_land
