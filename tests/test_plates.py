"""Tests for the plate-tectonics layer (`worldgen.plates`)."""

from __future__ import annotations

from dataclasses import replace

from worldgen import plates as plates_layer
from worldgen.hex import Hex
from worldgen.plates import (
    BOUNDARY_CC_CONVERGENT,
    BOUNDARY_DIVERGENT,
    BOUNDARY_OC_CONVERGENT,
    BOUNDARY_OO_CONVERGENT,
    BOUNDARY_TRANSFORM,
    PLATE_TYPE_CONTINENTAL,
    PLATE_TYPE_OCEANIC,
)
from worldgen.rng import RngHierarchy
from worldgen.types import PlateConfig, WorldShape
from worldgen.world import rect_world_hexes

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
        convergence_threshold=0.10,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


_HEX_SIZE_KM = 5.0


def _shape(side_km: float) -> WorldShape:
    return WorldShape(width_km=side_km, height_km=side_km)


def _hexes(side_km: float) -> list[Hex]:
    return rect_world_hexes(_shape(side_km), _HEX_SIZE_KM)


def test_plate_generation_deterministic() -> None:
    """Same seed + config produces identical plate fields."""
    hexes = _hexes(120.0)
    cfg = _plate_config()
    a = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(42))
    b = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(42))
    assert [(p.id, p.seed_hex, p.type, p.motion) for p in a.plates] == \
           [(p.id, p.seed_hex, p.type, p.motion) for p in b.plates]
    for h in hexes:
        assert a.hex_to_plate[h] == b.hex_to_plate[h]
        assert a.boundary_type[h] == b.boundary_type[h]
        assert a.distance_to_boundary_km[h] == b.distance_to_boundary_km[h]


def test_every_hex_assigned_to_a_plate() -> None:
    hexes = _hexes(150.0)
    cfg = _plate_config(count=8)
    field = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(1))
    assert set(field.hex_to_plate.keys()) == set(hexes)
    plate_ids = {p.id for p in field.plates}
    for h in hexes:
        assert field.hex_to_plate[h] in plate_ids


def test_plate_count_matches_config() -> None:
    hexes = _hexes(150.0)
    cfg = _plate_config(count=8)
    field = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(1))
    assert len(field.plates) == 8


def test_plate_types_respect_continental_fraction() -> None:
    """With continental_fraction=1.0, every plate is continental."""
    hexes = _hexes(120.0)
    cfg = _plate_config(continental_fraction=1.0)
    field = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(0))
    assert all(p.type == PLATE_TYPE_CONTINENTAL for p in field.plates)

    cfg2 = _plate_config(continental_fraction=0.0)
    field2 = plates_layer.generate_plates(hexes, cfg2, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(0))
    assert all(p.type == PLATE_TYPE_OCEANIC for p in field2.plates)


def test_boundary_types_are_valid() -> None:
    hexes = _hexes(150.0)
    cfg = _plate_config(count=7)
    field = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(3))
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
    hexes = _hexes(150.0)
    cfg = _plate_config(count=7)
    field = plates_layer.generate_plates(hexes, cfg, hex_size_km=_HEX_SIZE_KM, rng=RngHierarchy(3))
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


