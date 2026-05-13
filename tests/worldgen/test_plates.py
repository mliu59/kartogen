"""Tests for the plate-tectonics layer (`sim.world.worldgen.plates`)."""

from __future__ import annotations

from dataclasses import replace

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.worldgen import generate
from sim.world.worldgen import plates as plates_layer
from sim.world.worldgen.plates import (
    BOUNDARY_CC_CONVERGENT,
    BOUNDARY_DIVERGENT,
    BOUNDARY_OC_CONVERGENT,
    BOUNDARY_OO_CONVERGENT,
    BOUNDARY_TRANSFORM,
    PLATE_TYPE_CONTINENTAL,
    PLATE_TYPE_OCEANIC,
)
from sim.world.worldgen.types import PlateConfig, WorldgenConfig

VALID_BOUNDARIES = {
    BOUNDARY_CC_CONVERGENT,
    BOUNDARY_OC_CONVERGENT,
    BOUNDARY_OO_CONVERGENT,
    BOUNDARY_DIVERGENT,
    BOUNDARY_TRANSFORM,
}


def _plate_config(**overrides: object) -> PlateConfig:
    base = PlateConfig(
        count=6,
        continental_fraction=0.5,
        min_separation_km=400.0,
        seed_radial_bias=0.0,
        boundary_warp_strength_km=50.0,
        boundary_warp_wavelength_km=250.0,
        motion_speed=1.0,
        continental_baseline=0.25,
        oceanic_baseline=-0.40,
        mountain_amplitude=0.55,
        coastal_range_amplitude=0.45,
        island_arc_amplitude=0.35,
        rift_depth=0.25,
        boundary_falloff_km=150.0,
        baseline_blend_km=200.0,
        convergence_threshold=0.10,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_plate_generation_deterministic() -> None:
    """Same seed + config produces identical plate fields."""
    hexes = Hex(0, 0).spiral(12)
    cfg = _plate_config()
    a = plates_layer.generate_plates(hexes, 12, cfg, hex_size_km=5.0, rng=RngHierarchy(42))
    b = plates_layer.generate_plates(hexes, 12, cfg, hex_size_km=5.0, rng=RngHierarchy(42))
    assert [(p.id, p.seed_hex, p.type, p.motion) for p in a.plates] == \
           [(p.id, p.seed_hex, p.type, p.motion) for p in b.plates]
    for h in hexes:
        assert a.hex_to_plate[h] == b.hex_to_plate[h]
        assert a.boundary_type[h] == b.boundary_type[h]
        assert a.distance_to_boundary_km[h] == b.distance_to_boundary_km[h]


def test_every_hex_assigned_to_a_plate() -> None:
    hexes = Hex(0, 0).spiral(15)
    cfg = _plate_config(count=8)
    field = plates_layer.generate_plates(hexes, 15, cfg, hex_size_km=5.0, rng=RngHierarchy(1))
    assert set(field.hex_to_plate.keys()) == set(hexes)
    plate_ids = {p.id for p in field.plates}
    for h in hexes:
        assert field.hex_to_plate[h] in plate_ids


def test_plate_count_matches_config() -> None:
    hexes = Hex(0, 0).spiral(15)
    cfg = _plate_config(count=8)
    field = plates_layer.generate_plates(hexes, 15, cfg, hex_size_km=5.0, rng=RngHierarchy(1))
    assert len(field.plates) == 8


def test_plate_types_respect_continental_fraction() -> None:
    """With continental_fraction=1.0, every plate is continental."""
    hexes = Hex(0, 0).spiral(12)
    cfg = _plate_config(continental_fraction=1.0)
    field = plates_layer.generate_plates(hexes, 12, cfg, hex_size_km=5.0, rng=RngHierarchy(0))
    assert all(p.type == PLATE_TYPE_CONTINENTAL for p in field.plates)

    cfg2 = _plate_config(continental_fraction=0.0)
    field2 = plates_layer.generate_plates(hexes, 12, cfg2, hex_size_km=5.0, rng=RngHierarchy(0))
    assert all(p.type == PLATE_TYPE_OCEANIC for p in field2.plates)


def test_boundary_types_are_valid() -> None:
    hexes = Hex(0, 0).spiral(15)
    cfg = _plate_config(count=7)
    field = plates_layer.generate_plates(hexes, 15, cfg, hex_size_km=5.0, rng=RngHierarchy(3))
    boundary_count = 0
    for h in hexes:
        btype = field.boundary_type[h]
        if btype is not None:
            assert btype in VALID_BOUNDARIES
            boundary_count += 1
    # Most hexes inherit some boundary type via BFS as long as there's at least
    # one boundary in the world (which there always is with >1 plate).
    assert boundary_count > 0


def test_boundary_hexes_have_zero_distance() -> None:
    hexes = Hex(0, 0).spiral(15)
    cfg = _plate_config(count=7)
    field = plates_layer.generate_plates(hexes, 15, cfg, hex_size_km=5.0, rng=RngHierarchy(3))
    hex_set = set(hexes)
    for h in hexes:
        pid = field.hex_to_plate[h]
        # A hex is on a boundary iff any in-bounds neighbor is on a different plate.
        on_boundary = any(
            nb in hex_set and field.hex_to_plate[nb] != pid
            for nb in h.neighbors()
        )
        if on_boundary:
            assert field.distance_to_boundary_km[h] == 0.0


def test_all_plates_use_preset_in_full_pipeline(default_worldgen_config: WorldgenConfig) -> None:
    """The default preset (whichever it currently is) drives mask_mode='plates'
    via the bundled preset configuration; pipeline.generate must produce a
    populated PlateField and HexData with plate metadata."""
    world = generate(radius=10, config=default_worldgen_config, seed=0)
    assert world.plates is not None
    assert len(world.plates.plates) == default_worldgen_config.plates.count  # type: ignore[union-attr]
    for h, data in world.hexes.items():
        assert data.plate_id is not None
        assert data.plate_type in (PLATE_TYPE_CONTINENTAL, PLATE_TYPE_OCEANIC)
        assert data.distance_to_boundary_km is not None
        assert data.distance_to_boundary_km >= 0.0
