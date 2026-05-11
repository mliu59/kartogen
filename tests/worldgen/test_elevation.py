"""Tests for the elevation layer."""

from __future__ import annotations

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.worldgen import elevation as elevation_layer
from sim.world.worldgen.types import WorldgenConfig


def test_elevation_deterministic(default_worldgen_config: WorldgenConfig) -> None:
    """Same seed + config produces byte-identical elevation."""
    hexes = Hex(0, 0).spiral(8)
    rng1 = RngHierarchy(42)
    rng2 = RngHierarchy(42)
    a = elevation_layer.compute(hexes, 8, default_worldgen_config, rng1)
    b = elevation_layer.compute(hexes, 8, default_worldgen_config, rng2)
    for h in hexes:
        assert a.elevation[h] == b.elevation[h]
    assert a.sea_level == b.sea_level


def test_elevation_different_seeds_diverge(default_worldgen_config: WorldgenConfig) -> None:
    hexes = Hex(0, 0).spiral(8)
    a = elevation_layer.compute(hexes, 8, default_worldgen_config, RngHierarchy(42))
    b = elevation_layer.compute(hexes, 8, default_worldgen_config, RngHierarchy(7))
    diffs = sum(1 for h in hexes if a.elevation[h] != b.elevation[h])
    assert diffs > len(hexes) * 0.95  # essentially all hexes differ


def test_sea_level_meets_target_land_fraction(default_worldgen_config: WorldgenConfig) -> None:
    """The quantile threshold should produce close to the configured land fraction."""
    hexes = Hex(0, 0).spiral(20)
    layer = elevation_layer.compute(hexes, 20, default_worldgen_config, RngHierarchy(42))
    land = sum(1 for h in hexes if not layer.is_ocean(h))
    target = default_worldgen_config.land_fraction
    actual = land / len(hexes)
    assert abs(actual - target) < 0.02


def test_radial_falloff_makes_edges_ocean(default_worldgen_config: WorldgenConfig) -> None:
    """With radial falloff enabled, hexes near the world edge should usually be ocean."""
    radius = 20
    hexes = Hex(0, 0).spiral(radius)
    layer = elevation_layer.compute(hexes, radius, default_worldgen_config, RngHierarchy(42))

    edge_hexes = [h for h in hexes if max(abs(h.q), abs(h.r), abs(h.s)) >= radius - 1]
    ocean_edge_fraction = sum(1 for h in edge_hexes if layer.is_ocean(h)) / len(edge_hexes)
    assert ocean_edge_fraction > 0.8
