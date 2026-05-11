"""Elevation layer: fBm + ridged multifractal + domain warp + radial falloff."""

from __future__ import annotations

import math
from collections.abc import Iterable

from sim.engine.rng import RngHierarchy
from sim.world.hex import Hex
from sim.world.noise import PerlinNoise2D, fbm, ridged_fbm
from sim.world.worldgen.types import ElevationLayer, WorldgenConfig


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Axial → flat cartesian for noise sampling.

    Uses the same flat-top-hex pixel mapping the renderer uses, normalized so
    units are roughly "hexes" rather than pixels.
    """
    x = 1.5 * h.q
    y = math.sqrt(3.0) * (h.r + h.q / 2.0)
    return x, y


def _radial_falloff(h: Hex, radius: int, inner_fraction: float, power: float) -> float:
    """Smooth radial falloff: 0 at the inner radius, 1 at the world edge.

    Encourages the map edge to be ocean, producing a continent / island world
    rather than wraparound noise.
    """
    if radius == 0:
        return 0.0
    d = max(abs(h.q), abs(h.r), abs(h.s)) / radius  # hex-distance-from-center
    inner = inner_fraction
    if d <= inner:
        return 0.0
    t = min(1.0, (d - inner) / (1.0 - inner))
    return t**power


def compute(
    hexes: Iterable[Hex],
    radius: int,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> ElevationLayer:
    """Compute the elevation field and a sea-level threshold."""
    noise_base = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "base"))
    noise_ridge = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "ridge"))
    noise_warp_x = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "warp_x"))
    noise_warp_y = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "warp_y"))

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

        # Radial falloff toward map edge.
        f = _radial_falloff(h, radius, config.falloff_inner_fraction, config.falloff_power)
        base = base - f * config.falloff_strength

        elevation[h] = base

    # Sea level by quantile so target land fraction is met exactly.
    values = sorted(elevation.values())
    n = len(values)
    ocean_count = int(round((1.0 - config.land_fraction) * n))
    ocean_count = max(0, min(n - 1, ocean_count))
    sea_level = values[ocean_count]

    return ElevationLayer(elevation=elevation, sea_level=sea_level)
