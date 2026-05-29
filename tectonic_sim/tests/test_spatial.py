"""Tests for ``tectonic_sim.spatial.BucketGrid``."""

from __future__ import annotations

import numpy as np
import pytest

from tectonic_sim import WorldRect
from tectonic_sim.spatial import BucketGrid


def _brute_force_neighbors(
    positions: np.ndarray, point: tuple[float, float], radius: float,
) -> set[int]:
    px, py = point
    d2 = (positions[:, 0] - px) ** 2 + (positions[:, 1] - py) ** 2
    return set(int(i) for i in np.where(d2 <= radius * radius)[0])


def _brute_force_cross_pairs(
    positions: np.ndarray, labels: np.ndarray, radius: float,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    n = positions.shape[0]
    r2 = radius * radius
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            if dx * dx + dy * dy <= r2:
                pairs.add((i, j))
    return pairs


def _brute_force_same_pairs(
    positions: np.ndarray, labels: np.ndarray, radius: float,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    n = positions.shape[0]
    r2 = radius * radius
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            if dx * dx + dy * dy <= r2:
                pairs.add((i, j))
    return pairs


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

def test_build_invalid_cell_size_raises() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    with pytest.raises(ValueError, match="cell_size_km"):
        BucketGrid.build(np.zeros((0, 2)), domain, cell_size_km=0.0)


def test_build_empty_positions() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    g = BucketGrid.build(np.zeros((0, 2)), domain, cell_size_km=10.0)
    assert g.particle_order.shape == (0,)
    assert g.cell_start.shape[0] >= 1


def test_build_groups_particles_by_cell() -> None:
    """Particles in the same grid cell appear contiguously in ``particle_order``."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([
        [-40.0, -40.0],   # cell (1, 1)
        [-42.0, -38.0],   # cell (0, 1) — neighbour, separate cell
        [40.0, 40.0],     # far away
        [-39.0, -39.0],   # back to cell (1, 1)
    ])
    g = BucketGrid.build(positions, domain, cell_size_km=10.0)
    # Every particle index appears exactly once.
    assert sorted(g.particle_order.tolist()) == [0, 1, 2, 3]
    # cell_start is monotone non-decreasing.
    diffs = np.diff(g.cell_start)
    assert (diffs >= 0).all()


# -----------------------------------------------------------------------------
# neighbors_within
# -----------------------------------------------------------------------------

def test_neighbors_within_matches_brute_force_random() -> None:
    """For a random cloud, the grid's answer equals the O(N²) brute-force
    answer for every query."""
    rng = np.random.Generator(np.random.PCG64(0))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(120, 2),
    )
    g = BucketGrid.build(positions, domain, cell_size_km=20.0)
    queries = rng.uniform(-100.0, 100.0, size=(15, 2))
    for q in queries:
        grid_neighbors = set(int(i) for i in g.neighbors_within(tuple(q), 30.0))
        truth = _brute_force_neighbors(positions, tuple(q), 30.0)
        assert grid_neighbors == truth


def test_neighbors_within_zero_radius_returns_empty() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    g = BucketGrid.build(positions, domain, cell_size_km=10.0)
    assert g.neighbors_within((0.0, 0.0), 0.0).shape == (0,)


def test_neighbors_within_handles_point_outside_domain() -> None:
    """Queries from outside the rectangle still return correct neighbours
    inside the rectangle within radius."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([[40.0, 40.0]])
    g = BucketGrid.build(positions, domain, cell_size_km=10.0)
    # Query from outside the rectangle, but inside the radius.
    got = g.neighbors_within((60.0, 50.0), radius_km=30.0)
    assert got.tolist() == [0]


def test_neighbors_within_dtype() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([[0.0, 0.0]])
    g = BucketGrid.build(positions, domain, cell_size_km=10.0)
    assert g.neighbors_within((0.0, 0.0), 5.0).dtype == np.int32


# -----------------------------------------------------------------------------
# cross_label_pairs_within
# -----------------------------------------------------------------------------

def test_cross_label_pairs_matches_brute_force() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(150, 2),
    )
    labels = rng.integers(0, 4, size=150).astype(np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=20.0)

    radius = 12.0
    gi, gj = g.cross_label_pairs_within(labels, radius_km=radius)
    grid_pairs = {(int(a), int(b)) if a < b else (int(b), int(a))
                  for a, b in zip(gi, gj)}
    truth = _brute_force_cross_pairs(positions, labels, radius)
    assert grid_pairs == truth


def test_cross_label_pairs_same_label_yields_nothing() -> None:
    """If every particle has the same label, there are no cross-label pairs."""
    rng = np.random.Generator(np.random.PCG64(2))
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = rng.uniform(-50.0, 50.0, size=(40, 2))
    labels = np.zeros(40, dtype=np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=10.0)
    gi, gj = g.cross_label_pairs_within(labels, radius_km=20.0)
    assert gi.shape == (0,)
    assert gj.shape == (0,)


def test_cross_label_pairs_radius_zero() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 1], dtype=np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=5.0)
    gi, gj = g.cross_label_pairs_within(labels, radius_km=0.0)
    assert gi.shape == (0,)


# -----------------------------------------------------------------------------
# Toroidal (wrap=True) behaviour
# -----------------------------------------------------------------------------

def _toroidal_brute_force_neighbors(
    positions: np.ndarray,
    domain: WorldRect,
    point: tuple[float, float],
    radius: float,
) -> set[int]:
    """Brute-force wrap-aware neighbor enumeration for cross-checks."""
    out: set[int] = set()
    for i, (x, y) in enumerate(positions):
        dx = x - point[0]
        dy = y - point[1]
        dx, dy = domain.wrapped_delta_xy(dx, dy)
        if dx * dx + dy * dy <= radius * radius:
            out.add(int(i))
    return out


def _toroidal_brute_force_cross_pairs(
    positions: np.ndarray,
    domain: WorldRect,
    labels: np.ndarray,
    radius: float,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    n = positions.shape[0]
    r2 = radius * radius
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx, dy = domain.wrapped_delta_xy(dx, dy)
            if dx * dx + dy * dy <= r2:
                pairs.add((i, j))
    return pairs


def test_neighbors_within_wrap_finds_cross_boundary_particles() -> None:
    """Two particles on opposite east-west edges should be neighbours on a
    torus even though their direct distance is nearly the world width."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([
        [-49.0, 0.0],   # near west edge
        [49.0, 0.0],    # near east edge — direct dist 98 km, wrap 2 km
    ])
    g = BucketGrid.build(positions, domain, cell_size_km=10.0, wrap=True)
    # Query near west edge with small radius — only particle 0 directly,
    # but particle 1 is 2 km away across the wrap.
    got = g.neighbors_within((-49.0, 0.0), radius_km=5.0)
    assert set(got.tolist()) == {0, 1}


def test_neighbors_within_wrap_matches_brute_force() -> None:
    """Random cloud + multiple queries: grid result equals wrap-aware
    brute force for every query."""
    rng = np.random.Generator(np.random.PCG64(0))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(100, 2),
    )
    g = BucketGrid.build(positions, domain, cell_size_km=20.0, wrap=True)
    # Queries near the boundary — most interesting case for wrap.
    queries = [
        (95.0, 0.0), (-95.0, 0.0),    # east/west edges
        (0.0, 95.0), (0.0, -95.0),    # north/south edges
        (95.0, 95.0), (-95.0, -95.0), # corners
        (0.0, 0.0),                    # centre (should match non-wrap here)
    ]
    for q in queries:
        grid_n = set(int(i) for i in g.neighbors_within(q, 30.0))
        truth = _toroidal_brute_force_neighbors(positions, domain, q, 30.0)
        assert grid_n == truth, f"mismatch at {q}: grid={grid_n} truth={truth}"


def test_cross_label_pairs_wrap_includes_boundary_pairs() -> None:
    """Two cross-label particles on opposite edges form a pair under
    wrap, but not under no-wrap."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([
        [-49.0, 0.0],
        [49.0, 0.0],
    ])
    labels = np.array([0, 1], dtype=np.int32)

    # No-wrap: too far → no pair.
    g_nowrap = BucketGrid.build(positions, domain, cell_size_km=10.0, wrap=False)
    gi, gj = g_nowrap.cross_label_pairs_within(labels, radius_km=10.0)
    assert gi.shape == (0,)

    # Wrap: 2 km apart → one pair.
    g_wrap = BucketGrid.build(positions, domain, cell_size_km=10.0, wrap=True)
    gi, gj = g_wrap.cross_label_pairs_within(labels, radius_km=10.0)
    assert gi.shape == (1,)
    assert {int(gi[0]), int(gj[0])} == {0, 1}


def test_cross_label_pairs_wrap_matches_brute_force() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(120, 2),
    )
    labels = rng.integers(0, 4, size=120).astype(np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=20.0, wrap=True)

    radius = 18.0
    gi, gj = g.cross_label_pairs_within(labels, radius_km=radius)
    grid_pairs = {
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in zip(gi, gj)
    }
    truth = _toroidal_brute_force_cross_pairs(positions, domain, labels, radius)
    assert grid_pairs == truth


# -----------------------------------------------------------------------------
# same_label_pairs_within
# -----------------------------------------------------------------------------

def test_same_label_pairs_matches_brute_force() -> None:
    """Random cloud: grid same-label result matches O(N²) brute force."""
    rng = np.random.Generator(np.random.PCG64(4))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(150, 2),
    )
    labels = rng.integers(0, 4, size=150).astype(np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=20.0)

    radius = 12.0
    gi, gj = g.same_label_pairs_within(labels, radius_km=radius)
    grid_pairs = {(int(a), int(b)) if a < b else (int(b), int(a))
                  for a, b in zip(gi, gj)}
    truth = _brute_force_same_pairs(positions, labels, radius)
    assert grid_pairs == truth


def test_same_label_pairs_and_cross_pairs_are_disjoint() -> None:
    """Cross-label and same-label queries on the same data partition the
    set of distance-eligible pairs — no pair appears in both."""
    rng = np.random.Generator(np.random.PCG64(5))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(-90.0, 90.0, size=(80, 2))
    labels = rng.integers(0, 3, size=80).astype(np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=20.0)

    ci, cj = g.cross_label_pairs_within(labels, radius_km=15.0)
    si, sj = g.same_label_pairs_within(labels, radius_km=15.0)
    cross = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in zip(ci, cj)}
    same = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in zip(si, sj)}
    assert cross.isdisjoint(same)


def test_same_label_pairs_cross_label_yields_nothing() -> None:
    """If every particle has a distinct label, there are no same-label pairs."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 2], dtype=np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=10.0)
    gi, gj = g.same_label_pairs_within(labels, radius_km=5.0)
    assert gi.shape == (0,)
    assert gj.shape == (0,)


def test_same_label_pairs_radius_zero() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 0], dtype=np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=5.0)
    gi, gj = g.same_label_pairs_within(labels, radius_km=0.0)
    assert gi.shape == (0,)


def test_same_label_pairs_wrap_matches_brute_force() -> None:
    """Wrap mode: same-label query matches toroidal brute force, including
    cross-seam pairs."""
    rng = np.random.Generator(np.random.PCG64(6))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(120, 2),
    )
    labels = rng.integers(0, 3, size=120).astype(np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=20.0, wrap=True)

    radius = 18.0
    gi, gj = g.same_label_pairs_within(labels, radius_km=radius)
    grid_pairs = {
        (int(a), int(b)) if a < b else (int(b), int(a))
        for a, b in zip(gi, gj)
    }
    # Brute force, wrap-aware, same-label only.
    truth: set[tuple[int, int]] = set()
    r2 = radius * radius
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            if labels[i] != labels[j]:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx, dy = domain.wrapped_delta_xy(dx, dy)
            if dx * dx + dy * dy <= r2:
                truth.add((i, j))
    assert grid_pairs == truth


def test_cross_label_pairs_handles_high_density() -> None:
    """Dense cluster: 50 particles in two intermixed groups within one cell.
    The grid mustn't miss any cross-pairs even when many fall into the
    same cell (within-cell ``triu_indices`` path)."""
    rng = np.random.Generator(np.random.PCG64(3))
    domain = WorldRect(width_km=100.0, height_km=100.0)
    positions = rng.uniform(-5.0, 5.0, size=(50, 2))
    labels = rng.integers(0, 2, size=50).astype(np.int32)
    g = BucketGrid.build(positions, domain, cell_size_km=30.0)

    gi, gj = g.cross_label_pairs_within(labels, radius_km=4.0)
    grid_pairs = {(int(a), int(b)) if a < b else (int(b), int(a))
                  for a, b in zip(gi, gj)}
    truth = _brute_force_cross_pairs(positions, labels, 4.0)
    assert grid_pairs == truth
