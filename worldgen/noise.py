"""Deterministic 2D Perlin noise.

Seeded from a ``random.Random`` instance produced by ``RngHierarchy``.
No global state. Used for fBm and ridged-multifractal elevation noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _fade(t: float) -> float:
    """Quintic fade curve (Perlin's improved interpolant)."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# Unit gradient vectors for the 8 octants — classic Perlin 2D grads.
_GRADS_2D: tuple[tuple[float, float], ...] = (
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (math.sqrt(0.5), math.sqrt(0.5)),
    (-math.sqrt(0.5), math.sqrt(0.5)),
    (math.sqrt(0.5), -math.sqrt(0.5)),
    (-math.sqrt(0.5), -math.sqrt(0.5)),
)


@dataclass(frozen=True)
class PerlinNoise2D:
    """2D Perlin noise with a seeded permutation table.

    Output range is approximately ``[-1, 1]``.
    """

    perm: tuple[int, ...]  # length 512 (doubled to avoid modulo on lookup)

    @staticmethod
    def from_rng(rng: random.Random) -> PerlinNoise2D:
        table = list(range(256))
        rng.shuffle(table)
        return PerlinNoise2D(perm=tuple(table + table))

    def _grad(self, ix: int, iy: int, dx: float, dy: float) -> float:
        h = self.perm[(ix + self.perm[iy & 255]) & 255] & 7
        gx, gy = _GRADS_2D[h]
        return gx * dx + gy * dy

    def sample(self, x: float, y: float) -> float:
        x0 = math.floor(x)
        y0 = math.floor(y)
        xf = x - x0
        yf = y - y0
        u = _fade(xf)
        v = _fade(yf)

        n00 = self._grad(x0, y0, xf, yf)
        n10 = self._grad(x0 + 1, y0, xf - 1.0, yf)
        n01 = self._grad(x0, y0 + 1, xf, yf - 1.0)
        n11 = self._grad(x0 + 1, y0 + 1, xf - 1.0, yf - 1.0)

        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        return _lerp(nx0, nx1, v)


def fbm(
    noise: PerlinNoise2D,
    x: float,
    y: float,
    *,
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    base_frequency: float = 1.0,
) -> float:
    """Fractal Brownian motion: sum of noise octaves with rising frequency
    and falling amplitude. Output approximately ``[-1, 1]``.
    """
    total = 0.0
    amplitude = 1.0
    frequency = base_frequency
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * noise.sample(x * frequency, y * frequency)
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / norm if norm > 0 else 0.0


def ridged_fbm(
    noise: PerlinNoise2D,
    x: float,
    y: float,
    *,
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    base_frequency: float = 1.0,
) -> float:
    """Ridged multifractal: ``1 - |noise|`` per octave, with each octave
    modulated by the previous. Output approximately ``[0, 1]`` with sharp
    ridge-lines at zero-crossings of the underlying noise.
    """
    total = 0.0
    amplitude = 1.0
    frequency = base_frequency
    weight = 1.0
    norm = 0.0
    for _ in range(octaves):
        n = noise.sample(x * frequency, y * frequency)
        n = 1.0 - abs(n)
        n *= n  # sharpen
        n *= weight
        weight = min(1.0, n * 2.0)  # gate higher octaves by lower ones
        total += amplitude * n
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / norm if norm > 0 else 0.0
