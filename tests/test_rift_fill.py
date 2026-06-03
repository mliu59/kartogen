"""Tests for the rift-vs-oceanic divergent fill: when the polygon sim
opens a gap, it inherits crust from the surrounding context — for
kartogen seeds with ``continental_fraction = 1.0`` that produces some
thinned (rift) continental columns rather than uniformly thick crust.

The PlaTec-style finalize-step boundary warp that used to live here was
deleted in the Phase 12 cleanup; the polygon sim produces organic
boundaries by construction.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from kartogen import generate
from kartogen.types import KartogenConfig, WorldShape

pytestmark = pytest.mark.slow  # full generate()/sim per test
def test_rift_thickness_produces_below_sea_level_continental(
    default_kartogen_config: KartogenConfig,
) -> None:
    """Some continental columns end up thinner than the unaltered reference
    thickness — those are the rift-fill columns that bring inland basins
    below sea level. (After collisions/erosion the thickness may drift away
    from the configured ``rift_thickness_km`` exactly, so we just check it
    sits below the unaltered continental thickness.)"""
    world = generate(config=replace(default_kartogen_config, world=WorldShape(width_km=300.0, height_km=300.0)), seed=42)
    cont_thick = default_kartogen_config.tectonics.continental_thickness_km
    n_thinned = sum(
        1 for c in world.lithosphere.columns.values()
        if c.crust_type == "continental" and c.thickness_km < cont_thick - 1.0
    )
    assert n_thinned > 0, "expected at least some thinned (rift) columns"


def test_generate_is_deterministic_under_fixed_seed(
    default_kartogen_config: KartogenConfig,
) -> None:
    """Same ``(config, seed)`` produces byte-identical column maps.

    Subsumes the old ``test_warp_disabled_is_deterministic_passthrough``
    and ``test_warp_enabled_is_deterministic`` — with the boundary warp
    gone, there's only one deterministic path through ``generate``.
    """
    a = generate(config=replace(default_kartogen_config, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    b = generate(config=replace(default_kartogen_config, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    for h in a.hexes:
        assert a.lithosphere.columns[h] == b.lithosphere.columns[h]
