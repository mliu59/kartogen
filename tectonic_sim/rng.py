"""Deterministic numpy-RNG hierarchy for ``tectonic_sim``.

Mirrors the shape of ``worldgen.rng.RngHierarchy`` but produces
``np.random.Generator`` instances rather than ``random.Random``, so the
sim's vectorised draws (positions, velocities, type assignments) stay on
numpy throughout.

Child generators are derived by hashing ``(root_seed, *path)`` into a
fresh 64-bit seed. Same root + same path → same generator, always. Adding
a new sub-stream later won't reshuffle existing ones.
"""

from __future__ import annotations

import hashlib

import numpy as np


class RngStream:
    """Hierarchical RNG factory keyed on a root seed.

    Usage::

        rng = RngStream(42)
        seeds_gen = rng.child("seeding", "plate_seeds")
        positions = seeds_gen.uniform(-1.0, 1.0, size=(N, 2))
    """

    __slots__ = ("_seed",)

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def child(self, *path: str | int) -> np.random.Generator:
        """Build a fresh ``np.random.Generator`` from a path under this root.

        The path is hashed into a 64-bit seed; adding a new path doesn't
        reshuffle generators bound to other paths.
        """
        key_str = ":".join(str(k) for k in path)
        h = hashlib.sha256(f"{self._seed}:{key_str}".encode()).digest()
        child_seed = int.from_bytes(h[:8], "big")
        return np.random.Generator(np.random.PCG64(child_seed))
