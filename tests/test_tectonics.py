"""Tests for the time-stepped plate-tectonics simulation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import pytest

from worldgen import generate
from worldgen.hex import Hex
from worldgen.plates import (
    PLATE_TYPE_CONTINENTAL,
    PLATE_TYPE_OCEANIC,
    Plate,
    PlateField,
)
from worldgen.rng import RngHierarchy
from worldgen.tectonics import (
    LithosphereColumn,
    LithosphereState,
    column_to_elevation_km,
    simulate_tectonics,
)
from worldgen.types import WorldShape, TectonicsConfig, WorldgenConfig


def _tectonics_cfg(**overrides: object) -> TectonicsConfig:
    base = TectonicsConfig(
        n_ticks=20,
        dt_myr=2.0,
        sea_level_km=0.0,
        plate_speed_kmpy=2.0,
        continental_thickness_km=35.0,
        oceanic_thickness_km=7.0,
        ridge_depth_km=2.5,
        ridge_subsidence_rate=0.35,
        max_ocean_depth_km=6.0,
        continental_reference_thickness_km=35.0,
        continental_isostasy_factor=0.15,
        orogeny_uplift_per_overlap_km=0.08,
        folding_ratio=0.05,
        subduction_arc_uplift_km=0.04,
        erosion_period=0,
        erosion_strength=0.0,
        rift_thickness_km=28.0,
        boundary_warp_strength=0.0,    # tests want deterministic mass; warp off
        boundary_warp_wavelength_km=80.0,
        snapshot_period_ticks=0,       # no history capture for unit tests
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _two_plate_field(
    motion_a: tuple[float, float],
    motion_b: tuple[float, float],
    type_a: str = PLATE_TYPE_CONTINENTAL,
    type_b: str = PLATE_TYPE_CONTINENTAL,
    radius: int = 6,
) -> tuple[PlateField, list[Hex]]:
    """Hand-built two-plate field for unit testing.

    Plate A's seed sits west of origin, plate B's east. Owned hexes are split
    by the q-axis sign (q < 0 → A, q ≥ 0 → B). Motion vectors are caller-
    chosen so we can exercise convergent, divergent, and isolated behaviour.
    """
    hexes = Hex(0, 0).spiral(radius)
    plate_a = Plate(id=0, seed_hex=Hex(-radius // 2, 0), type=type_a, motion=motion_a)
    plate_b = Plate(id=1, seed_hex=Hex(radius // 2, 0), type=type_b, motion=motion_b)
    hex_to_plate = {h: (0 if h.q < 0 else 1) for h in hexes}
    field = PlateField(
        plates=(plate_a, plate_b),
        hex_to_plate=hex_to_plate,
        distance_by_type={},
        boundary_type={h: None for h in hexes},
        distance_to_boundary_km={h: 0.0 for h in hexes},
    )
    return field, hexes


def test_isolated_plate_no_motion_preserves_continental_mass(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """A single non-moving plate keeps the same continental column at every hex."""
    cfg = _tectonics_cfg(plate_speed_kmpy=0.0, n_ticks=10)
    hexes = Hex(0, 0).spiral(4)
    only_plate = Plate(
        id=0, seed_hex=Hex(0, 0),
        type=PLATE_TYPE_CONTINENTAL, motion=(0.0, 0.0),
    )
    field = PlateField(
        plates=(only_plate,),
        hex_to_plate={h: 0 for h in hexes},
        distance_by_type={},
        boundary_type={h: None for h in hexes},
        distance_to_boundary_km={h: 0.0 for h in hexes},
    )
    state = simulate_tectonics(
        field, hexes, cfg, default_worldgen_config.hex_size_km, RngHierarchy(42),
    )
    for h in hexes:
        col = state.columns[h]
        assert col.crust_type == PLATE_TYPE_CONTINENTAL
        # Thickness stays at the initial value (no collisions, no erosion).
        assert col.thickness_km == pytest.approx(cfg.continental_thickness_km)


def test_convergent_continents_thicken_at_boundary(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Two continents driven into each other should thicken at the contact zone."""
    cfg = _tectonics_cfg(plate_speed_kmpy=3.0, n_ticks=20, folding_ratio=0.10)
    field, hexes = _two_plate_field(
        motion_a=(1.0, 0.0),    # A moves east
        motion_b=(-1.0, 0.0),   # B moves west — head-on collision
        type_a=PLATE_TYPE_CONTINENTAL,
        type_b=PLATE_TYPE_CONTINENTAL,
        radius=6,
    )
    state = simulate_tectonics(
        field, hexes, cfg, default_worldgen_config.hex_size_km, RngHierarchy(0),
    )
    # The collision zone is around q = 0 — pick a few hexes near the original
    # boundary and verify at least one has thickened past the initial value.
    boundary_thickened = any(
        state.columns[h].thickness_km > cfg.continental_thickness_km + 0.01
        for h in hexes
        if abs(h.q) <= 2
    )
    assert boundary_thickened, (
        "Convergent continent-on-continent boundary failed to thicken any hex"
    )


def test_old_oceanic_crust_subsides_below_young(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Half-space cooling: older oceanic crust must sit deeper than fresh ridge crust."""
    cfg = _tectonics_cfg()
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
    cfg = _tectonics_cfg()
    ancient = LithosphereColumn(
        crust_type=PLATE_TYPE_OCEANIC, thickness_km=cfg.oceanic_thickness_km,
        age_myr=10_000.0,  # absurdly old
    )
    assert column_to_elevation_km(ancient, cfg) == pytest.approx(-cfg.max_ocean_depth_km)


def test_continental_isostasy_thicker_means_higher(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Thicker continental crust → higher elevation; equal to reference → sea level."""
    cfg = _tectonics_cfg()
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
    sea_km = default_worldgen_config.tectonics.sea_level_km  # type: ignore[union-attr]

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
        cast(TectonicsConfig, default_worldgen_config.tectonics),
        sea_level_km=2.0,
    )
    high_cfg = replace(default_worldgen_config, tectonics=high_sea)
    high_world = generate(config=replace(high_cfg, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    high_land = sum(1 for d in high_world.hexes.values() if not d.is_ocean)

    assert high_land < base_land
