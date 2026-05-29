"""Tests for ``tectonic_sim.overlap.detect_overlaps``.

``detect_overlaps`` is a thin wrapper around ``BucketGrid``; the heavy
correctness checks live in ``test_spatial`` (cross-label pair correctness
against O(N²) brute force). Here we just confirm the wrapper:

  - passes the configured radius through correctly,
  - returns the documented dtypes / shapes,
  - is deterministic,
  - is safe on empty input.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tectonic_sim import SimConfig, WorldRect, build_initial_state, detect_overlaps


def test_detect_overlaps_uses_configured_radius(
    default_sim_config: SimConfig,
) -> None:
    """The pair set must match the bucket grid's ``cross_label_pairs_within``
    answer at exactly ``sim_config.overlap_radius_km``, under the same
    wrap setting the config drives."""
    from tectonic_sim.spatial import BucketGrid

    domain = WorldRect(width_km=500.0, height_km=500.0)
    _, pos, pid, _, _, _ = build_initial_state(domain, default_sim_config, seed=0)

    gi, gj = detect_overlaps(domain, pos, pid, default_sim_config)
    # Independent re-query at the same radius must match. detect_overlaps
    # honours boundary_mode → pass the same wrap flag to the grid here.
    wrap = (default_sim_config.boundary_mode == "wrap")
    grid = BucketGrid.build(
        pos, domain, default_sim_config.overlap_radius_km, wrap=wrap,
    )
    expect_i, expect_j = grid.cross_label_pairs_within(
        pid, default_sim_config.overlap_radius_km,
    )
    np.testing.assert_array_equal(gi, expect_i)
    np.testing.assert_array_equal(gj, expect_j)


def test_detect_overlaps_dtypes(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=200.0, height_km=200.0)
    _, pos, pid, _, _, _ = build_initial_state(domain, default_sim_config, seed=0)
    gi, gj = detect_overlaps(domain, pos, pid, default_sim_config)
    assert gi.dtype == np.int32
    assert gj.dtype == np.int32
    assert gi.shape == gj.shape


def test_detect_overlaps_pair_indices_distinct(default_sim_config: SimConfig) -> None:
    """Every returned pair references two distinct particles.

    The bucket-grid contract is *unordered* pairs — neither ``i < j`` nor
    ``i > j`` is guaranteed across cells — but each pair must reference
    two different particles (no self-pairs)."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    _, pos, pid, _, _, _ = build_initial_state(domain, default_sim_config, seed=0)
    gi, gj = detect_overlaps(domain, pos, pid, default_sim_config)
    assert (gi != gj).all()


def test_detect_overlaps_only_cross_plate_pairs(
    default_sim_config: SimConfig,
) -> None:
    """Every detected pair must be cross-plate by construction."""
    domain = WorldRect(width_km=300.0, height_km=300.0)
    _, pos, pid, _, _, _ = build_initial_state(domain, default_sim_config, seed=0)
    gi, gj = detect_overlaps(domain, pos, pid, default_sim_config)
    assert (pid[gi] != pid[gj]).all()


def test_detect_overlaps_determinism(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=300.0, height_km=300.0)
    _, pos, pid, _, _, _ = build_initial_state(domain, default_sim_config, seed=0)
    a = detect_overlaps(domain, pos.copy(), pid.copy(), default_sim_config)
    b = detect_overlaps(domain, pos.copy(), pid.copy(), default_sim_config)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_detect_overlaps_empty_state(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    pos = np.zeros((0, 2))
    pid = np.zeros(0, dtype=np.int32)
    gi, gj = detect_overlaps(domain, pos, pid, default_sim_config)
    assert gi.shape == (0,)
    assert gj.shape == (0,)


def test_detect_overlaps_pair_count_grows_with_radius(
    default_sim_config: SimConfig,
) -> None:
    """Doubling ``particle_spacing_km`` (which doubles the overlap radius)
    should produce strictly more pairs on the same hex set — boundaries
    haven't moved, but the detection ring is bigger.

    NB: we vary spacing rather than radius directly because radius is
    derived from spacing (1.5×).
    """
    domain = WorldRect(width_km=400.0, height_km=400.0)
    _, pos, pid, _, _, _ = build_initial_state(domain, default_sim_config, seed=0)

    tight_cfg = replace(default_sim_config, particle_spacing_km=10.0)
    wide_cfg = replace(default_sim_config, particle_spacing_km=30.0)
    n_tight = detect_overlaps(domain, pos, pid, tight_cfg)[0].shape[0]
    n_wide = detect_overlaps(domain, pos, pid, wide_cfg)[0].shape[0]
    assert n_wide > n_tight
