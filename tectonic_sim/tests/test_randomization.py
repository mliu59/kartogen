"""Tests for ``tectonic_sim.randomization``."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pytest

from tectonic_sim import (
    FieldRandomizer,
    SimConfig,
    randomize_dataclass_fields,
    randomize_sim_config,
)
from tectonic_sim.randomization import _SIM_CONFIG_RANDOMIZERS


# -----------------------------------------------------------------------------
# Layer 1: generic helper
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _Toy:
    """Minimal dataclass for direct generic-helper tests."""
    x: float
    y: float
    n: int
    label: str


def test_zero_temperature_is_identity() -> None:
    """``T = 0`` returns the input unchanged."""
    base = _Toy(x=1.0, y=2.0, n=5, label="hi")
    specs = (
        FieldRandomizer("x", std=10.0, minimum=None, maximum=None),
        FieldRandomizer("y", std=10.0, minimum=None, maximum=None),
        FieldRandomizer("n", std=2.0, minimum=0, maximum=100, is_integer=True),
    )
    rng = np.random.Generator(np.random.PCG64(0))
    out = randomize_dataclass_fields(base, specs, 0.0, rng)
    assert out is base or out == base


def test_negative_temperature_raises() -> None:
    base = _Toy(x=1.0, y=2.0, n=5, label="hi")
    rng = np.random.Generator(np.random.PCG64(0))
    with pytest.raises(ValueError, match="param_temperature"):
        randomize_dataclass_fields(base, (), -0.1, rng)


def test_unknown_field_in_spec_raises() -> None:
    base = _Toy(x=1.0, y=2.0, n=5, label="hi")
    specs = (FieldRandomizer("bogus", std=1.0, minimum=None, maximum=None),)
    rng = np.random.Generator(np.random.PCG64(0))
    with pytest.raises(ValueError, match=r"unknown field"):
        randomize_dataclass_fields(base, specs, 1.0, rng)


def test_fields_not_in_spec_pass_through_unchanged() -> None:
    """A field omitted from the spec tuple is untouched."""
    base = _Toy(x=1.0, y=2.0, n=5, label="hi")
    specs = (FieldRandomizer("x", std=10.0, minimum=None, maximum=None),)
    rng = np.random.Generator(np.random.PCG64(0))
    out = randomize_dataclass_fields(base, specs, 1.0, rng)
    # y, n, label all unchanged.
    assert out.y == base.y
    assert out.n == base.n
    assert out.label == base.label
    # x was drawn — almost certainly changed.
    assert out.x != base.x


def test_clip_bounds_enforced() -> None:
    """Even at huge temperature, drawn values respect the clip range."""
    base = _Toy(x=5.0, y=0.0, n=0, label="hi")
    specs = (
        FieldRandomizer("x", std=10.0, minimum=2.0, maximum=8.0),
    )
    for seed in range(20):
        rng = np.random.Generator(np.random.PCG64(seed))
        out = randomize_dataclass_fields(base, specs, 10.0, rng)
        assert 2.0 <= out.x <= 8.0


def test_integer_field_returns_int() -> None:
    base = _Toy(x=0.0, y=0.0, n=5, label="hi")
    specs = (
        FieldRandomizer("n", std=3.0, minimum=0, maximum=100, is_integer=True),
    )
    rng = np.random.Generator(np.random.PCG64(0))
    out = randomize_dataclass_fields(base, specs, 1.0, rng)
    assert isinstance(out.n, int)


def test_integer_field_honours_bounds_after_rounding() -> None:
    """Rounding can pull a draw below the float minimum; we re-clip
    after rounding so the integer result still respects bounds."""
    base = _Toy(x=0.0, y=0.0, n=5, label="hi")
    specs = (
        FieldRandomizer("n", std=100.0, minimum=2, maximum=10, is_integer=True),
    )
    for seed in range(30):
        rng = np.random.Generator(np.random.PCG64(seed))
        out = randomize_dataclass_fields(base, specs, 5.0, rng)
        assert 2 <= out.n <= 10
        assert isinstance(out.n, int)


def test_determinism_per_seed() -> None:
    """Same seed → same output across calls."""
    base = _Toy(x=1.0, y=2.0, n=5, label="hi")
    specs = (
        FieldRandomizer("x", std=10.0, minimum=None, maximum=None),
        FieldRandomizer("y", std=10.0, minimum=None, maximum=None),
    )
    a = randomize_dataclass_fields(
        base, specs, 1.0, np.random.Generator(np.random.PCG64(42)),
    )
    b = randomize_dataclass_fields(
        base, specs, 1.0, np.random.Generator(np.random.PCG64(42)),
    )
    assert a == b


def test_temperature_scales_std() -> None:
    """At larger temperature, drawn values spread further from base.

    Statistical check: stddev of draws at T=2 should be ~2× that at T=1
    over a few hundred draws."""
    base = _Toy(x=0.0, y=0.0, n=0, label="hi")
    specs = (
        FieldRandomizer("x", std=1.0, minimum=None, maximum=None),
    )
    draws_t1 = np.array([
        randomize_dataclass_fields(
            base, specs, 1.0, np.random.Generator(np.random.PCG64(s)),
        ).x
        for s in range(500)
    ])
    draws_t2 = np.array([
        randomize_dataclass_fields(
            base, specs, 2.0, np.random.Generator(np.random.PCG64(s + 10_000)),
        ).x
        for s in range(500)
    ])
    # Empirical stds should be approximately 1.0 and 2.0; allow 25 %
    # tolerance for 500-draw sampling noise.
    std1 = float(draws_t1.std())
    std2 = float(draws_t2.std())
    assert 0.75 <= std1 <= 1.25, f"std1 = {std1}"
    assert 1.5 <= std2 <= 2.5, f"std2 = {std2}"
    assert std2 > std1


# -----------------------------------------------------------------------------
# Layer 2: SimConfig wrapper
# -----------------------------------------------------------------------------

def test_randomize_sim_config_default_temperature_is_zero(
    default_sim_config: SimConfig,
) -> None:
    """The default ``param_temperature = 0`` means ``randomize_sim_config(
    base)`` is a no-op identity. Callers who forget to pass a temperature
    get the safe deterministic config back, not an accidentally-perturbed
    one. (Also implicitly proves ``T = 0`` identity, which the generic
    helper test covers explicitly.)"""
    out = randomize_sim_config(default_sim_config)
    assert out == default_sim_config


def test_randomize_sim_config_different_seeds_differ(
    default_sim_config: SimConfig,
) -> None:
    """Different seeds at the same T produce different draws."""
    a = randomize_sim_config(default_sim_config, 1.0, seed=0)
    b = randomize_sim_config(default_sim_config, 1.0, seed=1)
    assert a != b


def test_randomize_sim_config_returns_valid_loaded_state(
    default_sim_config: SimConfig,
) -> None:
    """A draw should produce a config that's still type-correct and
    coherent: ints stay ints, floats stay floats, derived properties
    still compute."""
    out = randomize_sim_config(default_sim_config, 1.0, seed=7)
    assert isinstance(out.plate_count, int)
    assert isinstance(out.continental_thickness_km, float)
    # Polygon sim no longer carries particle-only derived properties
    # (overlap_radius_km, intra_plate_min_distance_km, etc.). Just
    # check the field types stay coherent.
    assert isinstance(out.particle_spacing_km, float)


def test_randomize_sim_config_excluded_fields_pass_through(
    default_sim_config: SimConfig,
) -> None:
    """Fields explicitly excluded (time step, output cadence, iteration
    counts) must not be touched by the randomizer."""
    out = randomize_sim_config(default_sim_config, 10.0, seed=0)
    assert out.dt_myr == default_sim_config.dt_myr
    assert out.n_ticks == default_sim_config.n_ticks
    assert out.snapshot_period_ticks == default_sim_config.snapshot_period_ticks
    assert out.erosion_period == default_sim_config.erosion_period


def test_randomize_sim_config_respects_bounds_under_extreme_temperature(
    default_sim_config: SimConfig,
) -> None:
    """At very high T, draws hit the clip range often; all fields must
    still be in valid bounds after every draw across many seeds."""
    for seed in range(30):
        out = randomize_sim_config(default_sim_config, 5.0, seed=seed)
        # Spot-check the physically critical bounds.
        assert 2 <= out.plate_count <= 20
        assert 0.0 <= out.continental_fraction <= 1.0
        assert out.motion_speed_kmpy > 0
        assert out.continental_thickness_km > 0
        assert out.oceanic_thickness_km > 0
        assert out.particle_spacing_km > 0


def test_randomize_sim_config_actually_perturbs_each_field(
    default_sim_config: SimConfig,
) -> None:
    """Across 100 seeds at T=1, every randomized field's draws have
    nonzero variance — a regression catch for "I forgot to add the
    spec" / "the spec's std is zero by accident."""
    randomized_field_names = {
        spec.field_name for spec in _SIM_CONFIG_RANDOMIZERS
    }
    draws = [
        randomize_sim_config(default_sim_config, 1.0, seed=s)
        for s in range(100)
    ]
    for field_name in randomized_field_names:
        values = [getattr(d, field_name) for d in draws]
        unique_count = len(set(values))
        assert unique_count > 1, (
            f"field {field_name!r} produced only {unique_count} unique value(s) "
            f"across 100 draws — its FieldRandomizer std may be 0"
        )


def test_randomize_sim_config_spec_field_names_exist_on_simconfig(
    default_sim_config: SimConfig,
) -> None:
    """The spec tuple's field names must all exist on SimConfig — guards
    against typos when adding new entries."""
    valid = {f.name for f in fields(SimConfig)}
    for spec in _SIM_CONFIG_RANDOMIZERS:
        assert spec.field_name in valid, (
            f"FieldRandomizer references nonexistent SimConfig field "
            f"{spec.field_name!r}"
        )
