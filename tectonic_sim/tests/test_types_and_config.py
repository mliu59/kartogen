"""Smoke tests for ``tectonic_sim`` data types + TOML loader."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tectonic_sim import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
    crust_type_code,
    crust_type_name,
    load_sim_config,
    load_sim_config_from_path,
)
from tectonic_sim.types import OVERLAP_RADIUS_MULTIPLIER


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
    # Spot-check a few fields rather than enumerating all 25 — the loader's
    # required-keys check covers the rest at construction time.
    assert cfg.plate_count > 0
    assert cfg.particle_spacing_km > 0
    assert cfg.n_ticks > 0
    assert cfg.boundary_mode in {"open", "wrap"}
    assert cfg.snapshot_period_ticks >= 0


def test_load_sim_config_missing_keys_raises() -> None:
    """A table missing any required key fails loudly with the offenders
    listed — no silent defaults."""
    partial: dict[str, object] = {"plate_count": 5}
    with pytest.raises(KeyError, match="missing required keys"):
        load_sim_config(partial)


def test_load_sim_config_unsupported_boundary_raises(
    default_sim_config: SimConfig,
) -> None:
    """Only ``boundary_mode in {"open", "wrap"}`` is supported."""
    cfg_dict = {
        # Build a complete dict from the default and override boundary_mode.
        **_config_to_dict(default_sim_config),
        "boundary_mode": "reflect",
    }
    with pytest.raises(ValueError, match="boundary_mode"):
        load_sim_config(cfg_dict)


def test_load_sim_config_path_matches_dict_load(
    default_sim_config: SimConfig,
) -> None:
    """``load_sim_config_from_path`` and ``load_sim_config(dict)`` produce
    the same object for the same TOML."""
    direct = load_sim_config_from_path(
        Path(__file__).resolve().parents[2] / "config" / "tectonic_sim.toml"
    )
    assert direct == default_sim_config


def test_sim_config_is_frozen(default_sim_config: SimConfig) -> None:
    """``SimConfig`` is frozen — mutation raises rather than silently
    breaking determinism."""
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        default_sim_config.plate_count = 99  # type: ignore[misc]
    # And ``dataclasses.replace`` is the supported way to override:
    nudged = replace(default_sim_config, plate_count=7)
    assert nudged.plate_count == 7
    assert default_sim_config.plate_count != 7


def test_overlap_radius_derives_from_particle_spacing(
    default_sim_config: SimConfig,
) -> None:
    """``overlap_radius_km`` is *not* a config knob — it's a derived property
    equal to ``OVERLAP_RADIUS_MULTIPLIER × particle_spacing_km``. The value
    must move with the spacing under ``dataclasses.replace``."""
    cfg = default_sim_config
    assert cfg.overlap_radius_km == OVERLAP_RADIUS_MULTIPLIER * cfg.particle_spacing_km

    nudged = replace(cfg, particle_spacing_km=20.0)
    assert nudged.overlap_radius_km == OVERLAP_RADIUS_MULTIPLIER * 20.0


def test_overlap_radius_km_not_in_toml_keys() -> None:
    """Setting ``overlap_radius_km`` in the TOML must not silently
    override the derived value — the loader ignores unknown keys, but
    the field itself doesn't exist on the dataclass, so ``replace`` with
    that kwarg should raise."""
    with pytest.raises(TypeError, match="overlap_radius_km"):
        replace(
            load_sim_config_from_path(
                Path(__file__).resolve().parents[2] / "config" / "tectonic_sim.toml"
            ),
            overlap_radius_km=99.0,  # type: ignore[call-arg]
        )


def _config_to_dict(cfg: SimConfig) -> dict[str, object]:
    """Convert a SimConfig back into its TOML-table form (for tests that
    need to load a tweaked version)."""
    from dataclasses import asdict
    return asdict(cfg)
