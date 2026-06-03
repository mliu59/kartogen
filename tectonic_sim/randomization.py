"""Hyperparameter-driven randomization of physics configs.

One knob — ``param_temperature`` — controls the magnitude of random
exploration around a base config. At ``T = 0`` the returned config
equals the base byte-for-byte; at ``T > 0`` each numeric field is drawn
from ``Normal(base_value, T × hardcoded_std)``, clipped to a safe range
and rounded to int if applicable.

The design is two-layered so the same machinery can be reused for other
configs later (the obvious next case is ``KartogenConfig``):

  - **Layer 1: a generic helper.** ``randomize_dataclass_fields(base,
    randomizers, param_temperature, rng)`` walks a tuple of
    ``FieldRandomizer`` specs and produces a new instance of the same
    dataclass with the listed fields perturbed.

  - **Layer 2: per-config public wrapper.** ``randomize_sim_config(
    base, param_temperature, seed)`` hides the generic helper and
    binds the ``SimConfig``-specific spec tuple. Kartogen would add
    its own ``randomize_kartogen_config`` later that delegates to the
    same Layer 1.

**Std + clamp are relative to the config value.** The user only
controls ``param_temperature``. The per-field ``std`` at ``T = 1`` is
the "reasonable exploration unit" for that field — usually ~10–25 % of
its central value. Each draw is then clamped to a band **around the
loaded config value**: ``base ± clamp_sigmas · (T · std)`` (default
3σ). So the exploration window *tracks whatever base value the config
sets* — raise ``max_ocean_depth_km`` in the TOML and its randomized
range rises with it; there is no hardcoded absolute ceiling to silently
cap it. Optional ``hard_min`` / ``hard_max`` add ABSOLUTE physical
invariants on top (a fraction can't exceed 1, a thickness can't go
negative), applied after the relative clamp.

**What's excluded.** World size is the user's explicit exclusion (it
sets the simulation domain, not a physics magnitude). Two other
categories are also left alone:

  - Temporal / output knobs: ``dt_myr``, ``n_ticks``,
    ``snapshot_period_ticks``.
  - Numerical / cadence knobs: ``contact_iterations``, ``erosion_period``.

These would all reshape the *run* rather than the *world* and are
better held fixed when sweeping ``param_temperature``.

**Independent draws — known limitation.** Fields are randomized
independently in v0. Some pairs have natural coupling (rift crust is
*usually* thinner than established continental crust; reference
thickness usually equals starting thickness). The clip bounds chosen
below prevent the most pathological cross-field combinations, but they
don't strictly enforce coupling. If you draw at ``T = 2`` you may get
configs that are physically possible but oddly tuned (e.g. rift crust
slightly thicker than continental). Future work: a constraint post-pass
or a derived-field option on ``FieldRandomizer``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from tectonic_sim.types import SimConfig


@dataclass(frozen=True)
class FieldRandomizer:
    """Specification for randomizing one numeric field on a dataclass.

    Attributes:
        field_name: the dataclass attribute to perturb.
        std: standard deviation at ``param_temperature = 1.0``. The
            effective std at temperature ``T`` is ``T × std``.
        clamp_sigmas: the draw is clamped to a band **around the base
            config value**: ``base ± clamp_sigmas × (T × std)``. This is
            a RELATIVE window that tracks the config value (and scales
            with temperature), not a fixed absolute range — change the
            base and the clamp moves with it. ``0`` disables the
            relative clamp (rely on ``hard_min`` / ``hard_max`` only).
        hard_min: optional ABSOLUTE physical floor, applied after the
            relative clamp. ``None`` = no floor.
        hard_max: optional ABSOLUTE physical ceiling. ``None`` = no
            ceiling.
        is_integer: if True, the drawn value is rounded to the nearest
            integer (after clamping). Use for fields typed ``int``.
    """

    field_name: str
    std: float
    clamp_sigmas: float = 3.0
    hard_min: float | None = None
    hard_max: float | None = None
    is_integer: bool = False


# ----------------------------------------------------------------------------
# Generic Layer 1
# ----------------------------------------------------------------------------

def randomize_dataclass_fields(
    base: Any,
    randomizers: tuple[FieldRandomizer, ...],
    param_temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Any:
    """Return a new instance of ``base``'s dataclass with the listed
    fields perturbed by Normal noise scaled by ``param_temperature``.

    Fields not listed in ``randomizers`` pass through unchanged.
    Validates that every ``randomizer.field_name`` exists on the base
    dataclass — typos are bugs and should fail loud.

    Returns a new frozen-dataclass instance (uses ``dataclasses.replace``).
    """
    if param_temperature < 0:
        raise ValueError(
            f"param_temperature must be >= 0, got {param_temperature}",
        )
    if param_temperature == 0.0:
        # Identity path — return the input unchanged. Byte-identical
        # because no draws are made. The default value of 0 means the
        # safe / no-randomization path is what you get if you forget
        # to set the temperature.
        return base

    if rng is None:
        raise ValueError(
            "rng is required when param_temperature > 0",
        )

    known_fields = {f.name for f in fields(base)}
    overrides: dict[str, Any] = {}
    for spec in randomizers:
        if spec.field_name not in known_fields:
            raise ValueError(
                f"FieldRandomizer references unknown field "
                f"{spec.field_name!r} on {type(base).__name__}",
            )
        base_value = float(getattr(base, spec.field_name))
        sigma = spec.std * param_temperature
        drawn = float(rng.normal(loc=base_value, scale=sigma))
        # Relative clamp: a band AROUND THE CONFIG VALUE, sized in
        # std-units scaled by temperature. Tracks the base — raise the
        # config value and the window moves with it (no fixed ceiling).
        if spec.clamp_sigmas > 0.0 and sigma > 0.0:
            band = spec.clamp_sigmas * sigma
            drawn = min(max(drawn, base_value - band), base_value + band)
        # Absolute physical invariants (optional), applied after.
        if spec.hard_min is not None:
            drawn = max(drawn, spec.hard_min)
        if spec.hard_max is not None:
            drawn = min(drawn, spec.hard_max)
        if spec.is_integer:
            drawn_int = int(round(drawn))
            # Re-honour the hard bounds after rounding (rounding can push
            # a draw just below a float hard_min / above a float hard_max).
            if spec.hard_min is not None:
                drawn_int = max(drawn_int, int(np.ceil(spec.hard_min)))
            if spec.hard_max is not None:
                drawn_int = min(drawn_int, int(np.floor(spec.hard_max)))
            overrides[spec.field_name] = drawn_int
        else:
            overrides[spec.field_name] = drawn

    return replace(base, **overrides)


# ----------------------------------------------------------------------------
# SimConfig-specific Layer 2
# ----------------------------------------------------------------------------

# Per-field ``std`` for ``SimConfig``. The std is the T=1 exploration
# unit (~15–25 % of the typical central value); it ALSO sizes the
# relative clamp band (``base ± clamp_sigmas · T · std``, default 3σ),
# so the exploration window tracks whatever the config sets. ``hard_min``
# / ``hard_max`` are added ONLY where a true physical invariant exists
# (a fraction can't exceed 1, a thickness/length/probability can't go
# negative, ≥2 plates are needed) — they are NOT tuning windows, just
# safety rails. Everything else is bounded purely by the relative band.
_SIM_CONFIG_RANDOMIZERS: tuple[FieldRandomizer, ...] = (
    # --- Plate population ---
    FieldRandomizer("plate_count", std=2.0, hard_min=2, is_integer=True),
    FieldRandomizer("continental_fraction",
                    std=0.15, hard_min=0.0, hard_max=1.0),
    FieldRandomizer("motion_speed_kmpy", std=20.0, hard_min=1.0),
    FieldRandomizer("seed_radial_bias",
                    std=0.3, hard_min=-1.0, hard_max=1.0),

    # --- Crust thicknesses ---
    FieldRandomizer("continental_thickness_km", std=6.0, hard_min=1.0),
    FieldRandomizer("oceanic_thickness_km", std=1.5, hard_min=1.0),
    FieldRandomizer("rift_thickness_km", std=5.0, hard_min=1.0),

    # --- Half-space cooling ---
    FieldRandomizer("ridge_depth_km", std=0.5, hard_min=0.0),
    FieldRandomizer("ridge_subsidence_rate", std=0.1, hard_min=0.0),
    # No hard ceiling — the band tracks the config value, so raising
    # max_ocean_depth_km in the TOML raises its randomized range too.
    FieldRandomizer("max_ocean_depth_km", std=1.0, hard_min=0.0),

    # --- Continental isostasy ---
    FieldRandomizer("continental_reference_thickness_km",
                    std=5.0, hard_min=1.0),
    FieldRandomizer("continental_isostasy_factor", std=0.04, hard_min=0.0),
    # Sea level is signed (a sampling threshold), so no non-negativity
    # floor — only the relative band bounds it.
    FieldRandomizer("sea_level_km", std=1.0),

    # --- Collision constants ---
    FieldRandomizer("folding_ratio", std=0.15, hard_min=0.0, hard_max=1.0),
    # Fold-belt geometry (over-rider, hybrid plateau profile). Depth =
    # total inland breadth; ramp = suture→plateau rise; taper = far-edge
    # falloff. Plateau is the remainder.
    FieldRandomizer("folding_belt_depth_km", std=50.0, hard_min=0.0),
    FieldRandomizer("folding_belt_ramp_km", std=12.0, hard_min=0.0),
    FieldRandomizer("folding_belt_taper_km", std=18.0, hard_min=0.0),
    # Loser-side belt. Ratio capped at 1 (a fraction); depths/decays
    # non-negative.
    FieldRandomizer("folding_loser_side_ratio",
                    std=0.08, hard_min=0.0, hard_max=1.0),
    FieldRandomizer("folding_belt_loser_depth_km", std=15.0, hard_min=0.0),
    FieldRandomizer("folding_belt_loser_decay_km", std=5.0, hard_min=1.0),

    # --- Continental relief (Perlin "ancient basement topography") ---
    FieldRandomizer("continental_relief_amplitude_km", std=2.0, hard_min=0.0),
    FieldRandomizer("continental_relief_wavelength_km",
                    std=400.0, hard_min=1.0),
    FieldRandomizer("continental_relief_octaves",
                    std=1.0, hard_min=1, is_integer=True),
    FieldRandomizer("continental_relief_persistence",
                    std=0.1, hard_min=0.0, hard_max=1.0),

    # --- Edge smoothing (non-physics) ---
    FieldRandomizer("edge_smoothing_kernel_km", std=15.0, hard_min=0.0),
    FieldRandomizer("edge_smoothing_alpha_max",
                    std=0.15, hard_min=0.0, hard_max=1.0),
    FieldRandomizer("edge_smoothing_noise_wavelength_km",
                    std=200.0, hard_min=1.0),
    FieldRandomizer("edge_smoothing_boundary_boost_peak",
                    std=0.15, hard_min=0.0, hard_max=1.0),
    FieldRandomizer("edge_smoothing_boundary_falloff_km",
                    std=15.0, hard_min=1.0),

    # --- Velocity damping ---
    FieldRandomizer("velocity_damping_strength",
                    std=0.03, hard_min=0.0, hard_max=1.0),

    # --- Erosion ---
    FieldRandomizer("erosion_strength", std=0.03, hard_min=0.0, hard_max=1.0),

    # Excluded from randomization, recorded here for documentation:
    #   n_ticks, dt_myr     — temporal knobs
    #   snapshot_period_ticks — output knob
    #   erosion_period      — numerical cadence
    # World size (WorldRect) is not a SimConfig field at all; the
    # caller passes it to the polygon sim separately.
)


def randomize_sim_config(
    base: SimConfig,
    param_temperature: float = 0.0,
    seed: int = 0,
) -> SimConfig:
    """Return a randomized ``SimConfig`` derived from ``base``.

    ``param_temperature`` controls the breadth of the exploration:

      - ``0.0`` (default) returns ``base`` unchanged (byte-identical,
        no draws). Default 0 makes the no-randomization path opt-in
        elsewhere: callers that don't set a temperature get the safe,
        deterministic config back.
      - ``1.0`` is the "natural" exploration breadth — each field is
        drawn from a Normal around ``base`` with hardcoded std.
      - ``> 1.0`` widens the draws (e.g. 2.0 doubles every std).

    ``seed`` controls determinism: same ``(base, temperature, seed)``
    yields the same output. Pass a different seed for a different draw
    at the same temperature. Defaults to 0; only consulted when
    ``param_temperature > 0`` (the identity path makes no draws).

    Fields not in ``_SIM_CONFIG_RANDOMIZERS`` (boundary mode, time step,
    output cadence, etc.) pass through unchanged — see this module's
    docstring for the exclusion rationale.

    Each draw is clamped to a band around its base config value
    (``base ± clamp_sigmas · T · std``) plus any physical hard bounds —
    clipped, not rejected, so the function always returns a usable
    config. Because the band is relative, the exploration window tracks
    the loaded config (e.g. raising ``max_ocean_depth_km`` raises its
    randomized range). Independent per-field draws mean some pairs may
    sit in unusual relative positions at high temperatures (e.g. rift
    crust slightly thicker than continental); future work can add a
    coupling layer if that becomes a problem.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    return randomize_dataclass_fields(
        base, _SIM_CONFIG_RANDOMIZERS, param_temperature, rng,
    )
