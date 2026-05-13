"""Elevation layer: fBm + ridged multifractal + domain warp + continent mask."""

from __future__ import annotations

import math
from collections.abc import Iterable

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.noise import PerlinNoise2D, fbm, ridged_fbm
from sim.world.worldgen.plates import PlateField, plate_elevation_bias
from sim.world.worldgen.types import ElevationLayer, WorldgenConfig


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Axial → flat cartesian for noise sampling.

    Uses the same flat-top-hex pixel mapping the renderer uses, normalized so
    units are roughly "hexes" rather than pixels.
    """
    x = 1.5 * h.q
    y = math.sqrt(3.0) * (h.r + h.q / 2.0)
    return x, y


def _ramp(t: float, power: float) -> float:
    """Standard outward ramp: 0 below 0, t**power in [0, 1], clamped to 1."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t**power


def _radial_mask(h: Hex, radius: int, inner: float, power: float) -> float:
    """Concentric falloff from center to edge — island continent shape."""
    if radius == 0:
        return 0.0
    d = max(abs(h.q), abs(h.r), abs(h.s)) / radius
    if d <= inner:
        return 0.0
    return _ramp((d - inner) / (1.0 - inner), power)


def _axial_mask(
    h: Hex, radius: int, theta: float, inner: float, power: float
) -> float:
    """One-sided ramp along ``theta``: hexes on the negative-projection side
    are pulled toward ocean, hexes on the positive side are untouched.

    Produces a continental-coast world: the map represents a coastal strip
    of a continent extending beyond the map edge in the +theta direction.
    """
    if radius == 0:
        return 0.0
    cx, cy = _hex_to_xy(h)
    max_r = radius * math.sqrt(3.0)
    proj = (math.cos(theta) * cx + math.sin(theta) * cy) / max_r
    d = -proj  # how far into the "ocean side" we are, signed
    if d <= inner:
        return 0.0
    return _ramp((d - inner) / (1.0 - inner), power)


def _dual_mask(
    h: Hex,
    radius: int,
    theta: float,
    inner: float,
    power: float,
    anchor_fraction: float,
) -> float:
    """Two anchor points at ``±anchor_fraction · max_r`` along ``theta``;
    pull-down ramps with distance to the *nearest* anchor.

    Produces a two-continent world separated by an ocean channel along the
    axis perpendicular to ``theta``.
    """
    if radius == 0:
        return 0.0
    cx, cy = _hex_to_xy(h)
    max_r = radius * math.sqrt(3.0)
    ax_x = math.cos(theta) * anchor_fraction * max_r
    ax_y = math.sin(theta) * anchor_fraction * max_r
    d1 = math.hypot(cx - ax_x, cy - ax_y) / max_r
    d2 = math.hypot(cx + ax_x, cy + ax_y) / max_r
    d = min(d1, d2)
    if d <= inner:
        return 0.0
    return _ramp((d - inner) / (1.0 - inner), power)


def _continent_mask_value(
    h: Hex,
    radius: int,
    config: WorldgenConfig,
    theta: float,
    plate_field: PlateField | None,
) -> float:
    """Dispatch to the configured continent-mask mode.

    Returns a *signed* adjustment to be added to fBm elevation. The analytic
    modes (radial / axial / dual) only push down (negative return), while
    the plates mode returns a signed bias (positive on continental plates and
    near convergent boundaries, negative on oceanic plates and near rifts).
    The single ``mask_strength`` knob scales the result uniformly.
    """
    mode = config.mask_mode
    if mode == "none":
        return 0.0
    if mode == "radial":
        return -_radial_mask(
            h, radius, config.mask_inner_fraction, config.mask_power
        )
    if mode == "axial":
        return -_axial_mask(
            h, radius, theta, config.mask_inner_fraction, config.mask_power
        )
    if mode == "dual":
        return -_dual_mask(
            h,
            radius,
            theta,
            config.mask_inner_fraction,
            config.mask_power,
            config.mask_anchor_fraction,
        )
    if mode == "plates":
        if plate_field is None or config.plates is None:
            raise ValueError(
                "mask_mode='plates' requires both a PlateField and "
                "[worldgen.elevation.plates] config."
            )
        return plate_elevation_bias(h, plate_field, config.plates)
    raise ValueError(
        f"Unknown continent mask_mode={mode!r}. "
        "Expected one of: none, radial, axial, dual, plates."
    )


def compute(
    hexes: Iterable[Hex],
    radius: int,
    config: WorldgenConfig,
    rng: RngHierarchy,
    plate_field: PlateField | None = None,
) -> ElevationLayer:
    """Compute the elevation field and a sea-level threshold."""
    noise_base = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "base"))
    noise_ridge = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "ridge"))
    noise_warp_x = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "warp_x"))
    noise_warp_y = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "warp_y"))

    # Orientation for axial/dual masks. Drawn from a dedicated child RNG so
    # adding/removing the mask doesn't reshuffle other layers' seeds.
    orientation_rng = rng.child("worldgen", "elevation", "mask_orientation")
    theta = orientation_rng.uniform(0.0, 2.0 * math.pi)

    elevation: dict[Hex, float] = {}
    hex_list = list(hexes)

    for h in hex_list:
        x, y = _hex_to_xy(h)

        # Domain warp: bend coordinates by a low-frequency noise field.
        wx = noise_warp_x.sample(x * config.warp_frequency, y * config.warp_frequency)
        wy = noise_warp_y.sample(x * config.warp_frequency + 13.7,
                                 y * config.warp_frequency + 7.3)
        x_w = x + wx * config.warp_strength
        y_w = y + wy * config.warp_strength

        # Base elevation: fBm in [-1, 1] approx.
        base = fbm(
            noise_base, x_w, y_w,
            octaves=config.noise_octaves,
            lacunarity=config.noise_lacunarity,
            persistence=config.noise_persistence,
            base_frequency=config.noise_base_frequency,
        )

        # Ridge contribution: high-frequency multifractal ridges, gated by
        # base elevation so ridges only appear in already-high terrain.
        if base > config.ridge_threshold:
            gate = min(1.0, (base - config.ridge_threshold) / (1.0 - config.ridge_threshold))
            ridge = ridged_fbm(
                noise_ridge, x_w, y_w,
                octaves=config.ridge_octaves,
                lacunarity=config.noise_lacunarity,
                persistence=config.noise_persistence,
                base_frequency=config.noise_base_frequency * 1.5,
            )
            base = base + gate * ridge * config.ridge_amplitude

        # Continent mask: signed adjustment shaping the macro layout. Analytic
        # modes (radial/axial/dual) return negative pull-down; plates returns
        # signed bias from baseline + boundary effects. Added directly to fBm.
        adjustment = _continent_mask_value(h, radius, config, theta, plate_field)
        base = base + adjustment * config.mask_strength

        elevation[h] = base

    # Sea level by quantile so target land fraction is met exactly.
    values = sorted(elevation.values())
    n = len(values)
    ocean_count = int(round((1.0 - config.land_fraction) * n))
    ocean_count = max(0, min(n - 1, ocean_count))
    sea_level = values[ocean_count]

    # Normalize peak land elevation to ≤ 1.0. The rest of the pipeline (lapse
    # rate, biome elevation thresholds, crop ``elev_max`` gates) is calibrated
    # assuming ``elevation - sea_level`` lies in roughly [0, 1] for land, with
    # 1.0 meaning ``max_elevation_km`` (~4.5 km, a tall mountain peak). Plates
    # add a continental/oceanic baseline plus boundary uplift on top of the
    # fBm field, which can push peaks well past 1.0 (1.2–1.4 with the default
    # presets) — that would make every mountain register as 6+ km tall and
    # 40 °C colder than its latitude. Rescaling here keeps the field's shape
    # intact while restoring the downstream calibration. No-op when the peak
    # is already ≤ 1.0 (analytic mask modes, archipelago, etc.).
    max_above = max(0.0, values[-1] - sea_level)
    if max_above > 1.0:
        scale = 1.0 / max_above
        for h, v in elevation.items():
            delta = v - sea_level
            if delta > 0:
                elevation[h] = sea_level + delta * scale

    return ElevationLayer(elevation=elevation, sea_level=sea_level)
