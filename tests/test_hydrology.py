"""Tests for the hydrology layer (sink-fill, flow accumulation, rivers, lakes)."""

from __future__ import annotations

import pytest
from worldgen.pipeline import GeneratedWorld

pytestmark = pytest.mark.slow  # full generate()/sim per test
def test_filled_elevation_is_at_least_natural(medium_world: GeneratedWorld) -> None:
    """Priority-flood can only RAISE elevations (fill sinks), never lower them."""
    hydro = medium_world.hydrology
    for h in medium_world.elevation.elevation:
        assert hydro.filled_elevation[h] >= medium_world.elevation.elevation[h] - 1e-9


def test_downstream_is_lower_or_ocean(medium_world: GeneratedWorld) -> None:
    """Every land cell's downstream neighbor must have lower filled elevation
    (or be an ocean cell — which acts as the global sink)."""
    hydro = medium_world.hydrology
    sea = medium_world.sea
    for h, down in hydro.downstream.items():
        if sea.is_ocean[h] or down is None:
            continue
        e_here = hydro.filled_elevation[h]
        e_down = hydro.filled_elevation[down]
        # Epsilon-tilt ensures strict descent.
        assert e_down < e_here + 1e-12


def test_flow_accumulation_conservation(medium_world: GeneratedWorld) -> None:
    """Sum of all land cells' base contribution equals the total flow that
    reaches the ocean (each land cell contributes exactly 1, and water can
    only exit through ocean cells)."""
    sea = medium_world.sea
    hydro = medium_world.hydrology
    land_count = sum(1 for h, is_o in sea.is_ocean.items() if not is_o)
    # Flow into ocean: sum of flow at land cells whose downstream is ocean.
    flow_to_ocean = 0
    for h, down in hydro.downstream.items():
        if sea.is_ocean[h]:
            continue
        if down is not None and sea.is_ocean[down]:
            flow_to_ocean += hydro.flow_accumulation[h]
    # Every land cell's water must eventually reach the ocean.
    assert flow_to_ocean == land_count


def test_rivers_only_on_land(medium_world: GeneratedWorld) -> None:
    for h, d in medium_world.hexes.items():
        if d.is_ocean:
            assert not d.is_river


def test_lakes_only_on_land(medium_world: GeneratedWorld) -> None:
    for h, d in medium_world.hexes.items():
        if d.is_ocean:
            assert not d.is_lake


def test_rivers_form_connected_chains(medium_world: GeneratedWorld) -> None:
    """A river hex's downstream chain should reach ocean (no orphan rivers)."""
    hydro = medium_world.hydrology
    sea = medium_world.sea

    for h, d in medium_world.hexes.items():
        if not d.is_river:
            continue
        # Walk downstream until we hit ocean. Bound the walk for safety.
        cur = h
        for _ in range(10_000):
            down = hydro.downstream[cur]
            if down is None:
                # Terminal sink — should be ocean.
                assert sea.is_ocean[cur], f"river hex {h} terminates at non-ocean {cur}"
                break
            if sea.is_ocean[down]:
                break
            cur = down
        else:
            raise AssertionError(f"river chain from {h} did not terminate")
