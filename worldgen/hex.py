"""Hexagonal grid coordinates using axial (q, r) system with flat-top hexes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Hex:
    """Axial hex coordinate."""

    q: int
    r: int

    @property
    def s(self) -> int:
        """Cube coordinate s (derived: q + r + s = 0)."""
        return -self.q - self.r

    def distance(self, other: Hex) -> int:
        """Hex distance (number of steps between two hexes)."""
        return (abs(self.q - other.q) + abs(self.r - other.r) + abs(self.s - other.s)) // 2

    def neighbors(self) -> tuple[Hex, ...]:
        """Return the 6 adjacent hexes in canonical order."""
        return tuple(
            Hex(self.q + dq, self.r + dr)
            for dq, dr in _AXIAL_DIRECTIONS
        )

    def ring(self, radius: int) -> list[Hex]:
        """Return all hexes at exactly `radius` distance, in order."""
        if radius == 0:
            return [self]
        results: list[Hex] = []
        # Start at the "bottom-left" of the ring
        h = Hex(self.q - radius, self.r + radius)
        for direction in range(6):
            for _ in range(radius):
                results.append(h)
                dq, dr = _AXIAL_DIRECTIONS[direction]
                h = Hex(h.q + dq, h.r + dr)
        return results

    def spiral(self, max_radius: int) -> list[Hex]:
        """Return hexes in spiral order from center out to max_radius (inclusive)."""
        results: list[Hex] = []
        for r in range(max_radius + 1):
            results.extend(self.ring(r))
        return results

    def __repr__(self) -> str:
        return f"Hex({self.q}, {self.r})"


# Axial direction vectors for the 6 neighbors (flat-top hex)
_AXIAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),   # east
    (1, -1),  # northeast
    (0, -1),  # northwest
    (-1, 0),  # west
    (-1, 1),  # southwest
    (0, 1),   # southeast
)
