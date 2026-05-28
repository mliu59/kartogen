"""Tests for the elevation layer.

Most invariants (determinism, land-fraction targeting) are now exercised
through ``pipeline.generate`` so that ``mask_mode="plates"`` configs — where
elevation depends on a separately-generated PlateField — work without
duplicating wiring in the test. A dedicated radial-mask test still drops
into ``elevation_layer.compute`` directly with a synthesized analytic-mode
config.
"""

from __future__ import annotations

from dataclasses import replace

from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from worldgen import elevation as elevation_layer
from worldgen import generate
from worldgen.types import WorldgenConfig


def test_elevation_deterministic(default_worldgen_config: WorldgenConfig) -> None:
    """Same seed + config produces byte-identical elevation."""
    a = generate(radius=8, config=default_worldgen_config, seed=42)
    b = generate(radius=8, config=default_worldgen_config, seed=42)
    for h in a.hexes:
        assert a.elevation.elevation[h] == b.elevation.elevation[h]
    assert a.elevation.sea_level == b.elevation.sea_level


def test_elevation_different_seeds_diverge(default_worldgen_config: WorldgenConfig) -> None:
    a = generate(radius=8, config=default_worldgen_config, seed=42)
    b = generate(radius=8, config=default_worldgen_config, seed=7)
    diffs = sum(
        1 for h in a.hexes
        if a.elevation.elevation[h] != b.elevation.elevation[h]
    )
    assert diffs > len(a.hexes) * 0.95


def test_sea_level_meets_target_land_fraction(default_worldgen_config: WorldgenConfig) -> None:
    """Quantile threshold should produce close to the configured land fraction."""
    world = generate(radius=20, config=default_worldgen_config, seed=42)
    land = sum(1 for h in world.hexes.values() if not h.is_ocean)
    target = default_worldgen_config.land_fraction
    actual = land / len(world.hexes)
    assert abs(actual - target) < 0.02


def test_radial_mask_makes_edges_ocean(default_worldgen_config: WorldgenConfig) -> None:
    """With the radial continent mask enabled, hexes near the world edge should usually be ocean."""
    radius = 20
    hexes = Hex(0, 0).spiral(radius)
    # Override mask params to force the radial-only path; bypasses plates.
    # ``land_fraction`` is set explicitly so the test is independent of the
    # default preset (which may be tuned for nearly-all-land worlds where
    # even the edges stay above the quantile sea level).
    radial_config = replace(
        default_worldgen_config,
        land_fraction=0.55,
        mask_mode="radial",
        mask_strength=0.55,
        mask_power=2.2,
        mask_inner_fraction=0.45,
    )
    layer = elevation_layer.compute(hexes, radius, radial_config, RngHierarchy(42))

    edge_hexes = [h for h in hexes if max(abs(h.q), abs(h.r), abs(h.s)) >= radius - 1]
    ocean_edge_fraction = sum(1 for h in edge_hexes if layer.is_ocean(h)) / len(edge_hexes)
    assert ocean_edge_fraction > 0.8
