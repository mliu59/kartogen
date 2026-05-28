"""Seeded RNG hierarchy for deterministic simulation."""

from __future__ import annotations

import hashlib
import random


class RngHierarchy:
    """Manages a hierarchy of seeded RNGs for deterministic simulation.

    The root seed produces child RNGs for each subsystem by hashing
    (seed, subsystem_name, tick, agent_id). This ensures:
    - Full determinism given (seed, config)
    - Each subsystem/agent gets independent random streams
    - Partial replay is possible (can recreate any sub-RNG without running the full sim)
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    def child(self, *keys: str | int) -> random.Random:
        """Create a child RNG from the root seed and a sequence of keys.

        Usage:
            rng.child("worldgen")
            rng.child("decision", tick, agent_id)
            rng.child("demographics", tick)
        """
        key_str = ":".join(str(k) for k in keys)
        h = hashlib.sha256(f"{self._seed}:{key_str}".encode()).digest()
        child_seed = int.from_bytes(h[:8], "big")
        return random.Random(child_seed)
