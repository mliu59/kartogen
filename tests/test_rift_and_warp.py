"""Tests for the rift-vs-oceanic divergent fill fix + the Perlin boundary warp.

Both address the "rectangular elevation edges" artifact that came from
``_seed_new_oceanic_crust`` over-eagerly creating oceanic basins around
every drifting plate, regardless of whether the plate was continental.
"""

from __future__ import annotations

from dataclasses import replace

from worldgen import generate
from worldgen.types import WorldShape, WorldgenConfig


def test_all_continental_world_stays_all_continental(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """The bundled config seeds every plate as continental
    (``continental_fraction = 1.0``). With the rift fix, divergent gaps
    must be filled with continental rift crust, not oceanic — so the final
    composition stays 100 % continental.
    """
    world = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=300.0, height_km=300.0)), seed=42)
    cont = sum(
        1 for c in world.lithosphere.columns.values()
        if c.crust_type == "continental"
    )
    assert cont == len(world.lithosphere.columns), (
        f"expected 100% continental crust, got "
        f"{cont}/{len(world.lithosphere.columns)}"
    )


def test_rift_thickness_produces_below_sea_level_continental(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Some continental columns end up thinner than the unaltered reference
    thickness — those are the rift-fill columns that bring inland basins
    below sea level. (After collisions/erosion the thickness may drift away
    from the configured ``rift_thickness_km`` exactly, so we just check it
    sits below the unaltered continental thickness.)"""
    world = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=300.0, height_km=300.0)), seed=42)
    cont_thick = default_worldgen_config.tectonics.continental_thickness_km
    n_thinned = sum(
        1 for c in world.lithosphere.columns.values()
        if c.crust_type == "continental" and c.thickness_km < cont_thick - 1.0
    )
    assert n_thinned > 0, "expected at least some thinned (rift) columns"


def test_boundary_warp_perturbs_coastline(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Turning the boundary warp on changes which specific hexes are ocean
    vs land, vs the warp-off baseline. Total land fraction should stay
    roughly comparable (the warp is symmetric — it flips both ways)."""
    off = replace(
        default_worldgen_config,
        tectonics=replace(
            default_worldgen_config.tectonics,
            boundary_warp_strength=0.0,
        ),
    )
    on = replace(
        default_worldgen_config,
        tectonics=replace(
            default_worldgen_config.tectonics,
            boundary_warp_strength=0.4,
        ),
    )
    w_off = generate(config=replace(off, world=WorldShape(width_km=300.0, height_km=300.0)), seed=42)
    w_on = generate(config=replace(on, world=WorldShape(width_km=300.0, height_km=300.0)), seed=42)

    diffs = sum(
        1 for h in w_off.hexes
        if w_off.hexes[h].is_ocean != w_on.hexes[h].is_ocean
    )
    assert diffs > 0, (
        "boundary warp = 0.4 should change at least one hex's land/ocean tag"
    )

    land_off = sum(1 for d in w_off.hexes.values() if not d.is_ocean)
    land_on = sum(1 for d in w_on.hexes.values() if not d.is_ocean)
    # Macro land fraction shouldn't shift by more than ~10 %.
    assert abs(land_on - land_off) / max(1, land_off) < 0.1


def test_warp_disabled_is_deterministic_passthrough(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """With warp strength = 0, the column dict is the raw simulation output —
    the same two runs must produce byte-identical columns."""
    cfg = replace(
        default_worldgen_config,
        tectonics=replace(
            default_worldgen_config.tectonics,
            boundary_warp_strength=0.0,
        ),
    )
    a = generate(config=replace(cfg, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    b = generate(config=replace(cfg, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    for h in a.hexes:
        assert a.lithosphere.columns[h] == b.lithosphere.columns[h]


def test_warp_enabled_is_deterministic(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Even with the Perlin warp on, the same (seed, config) → same world."""
    a = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    b = generate(config=replace(default_worldgen_config, world=WorldShape(width_km=120.0, height_km=120.0)), seed=42)
    for h in a.hexes:
        assert a.lithosphere.columns[h] == b.lithosphere.columns[h]
