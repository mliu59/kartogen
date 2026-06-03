"""Smoke tests for ``tectonic_sim`` data types + TOML loader."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tectonic_sim import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
    crust_type_code,
    crust_type_name,
    load_sim_config,
)
# OVERLAP_RADIUS_MULTIPLIER was a particle-sim concept; the polygon
# sim has no per-pair overlap radius (mass intersection is direct).
# Kept the tests below trimmed accordingly.


def test_world_rect_derived_properties() -> None:
    rect = WorldRect(width_km=1000.0, height_km=500.0)
    assert rect.half_width_km == 500.0
    assert rect.half_height_km == 250.0
    assert rect.area_km2 == 500_000.0


def test_crust_type_encoding_round_trip() -> None:
    assert crust_type_code("continental") == CRUST_CONTINENTAL
    assert crust_type_code("oceanic") == CRUST_OCEANIC
    assert crust_type_name(CRUST_CONTINENTAL) == "continental"
    assert crust_type_name(CRUST_OCEANIC) == "oceanic"


def test_crust_type_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown crust_type"):
        crust_type_code("granite")


def test_default_config_loads_from_disk(default_sim_config: SimConfig) -> None:
    """The bundled ``config/tectonic_sim.toml`` parses into a SimConfig
    with every required field populated."""
    cfg = default_sim_config
    # Spot-check a few fields rather than enumerating all of them — the
    # loader's required-keys check covers the rest at construction time.
    assert cfg.plate_count > 0
    assert cfg.n_ticks > 0
    assert cfg.snapshot_period_ticks >= 0


def test_load_sim_config_missing_keys_raises() -> None:
    """A table missing any required key fails loudly with the offenders
    listed — no silent defaults."""
    partial: dict[str, object] = {"plate_count": 5}
    with pytest.raises(KeyError, match="missing required keys"):
        load_sim_config(partial)


def test_sim_config_is_frozen(default_sim_config: SimConfig) -> None:
    """``SimConfig`` is frozen — mutation raises rather than silently
    breaking determinism."""
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        default_sim_config.plate_count = 99  # type: ignore[misc]
    # And ``dataclasses.replace`` is the supported way to override:
    nudged = replace(default_sim_config, plate_count=7)
    assert nudged.plate_count == 7
    assert default_sim_config.plate_count != 7


def test_sim_config_drops_particle_only_fields(
    default_sim_config: SimConfig,
) -> None:
    """After the polygon refactor, SimConfig should no longer carry
    particle-only fields. Sanity-check that they're gone."""
    cfg = default_sim_config
    for field in (
        "contact_iterations",
        "intra_plate_min_distance_factor",
        "surface_tension_strength",
        "surface_tension_radius_factor",
        "density_relief_strength",
        "density_relief_radius_factor",
        "density_relief_noise_factor",
        "overlap_radius_km",
    ):
        # min_continental_thickness_km was particle-only at one point but
        # is needed again by polygon_sim's absorption phase — it's back
        # in SimConfig as a polygon-sim tunable.
        assert not hasattr(cfg, field), (
            f"SimConfig still has particle-only field {field!r}"
        )


