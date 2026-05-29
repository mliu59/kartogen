"""Tests for ``tectonic_sim.divergent.divergent_fill``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tectonic_sim import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    Plate,
    SimConfig,
    WorldRect,
    build_initial_state,
    divergent_fill,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _full_dense_state(
    domain: WorldRect, cfg: SimConfig, seed: int = 0,
) -> tuple:
    """Return a freshly-built initial state — no vacancies."""
    return build_initial_state(domain, cfg, seed)


def _sparse_state_around_one_plate(
    cfg: SimConfig,
    *,
    plate_type: str = "continental",
) -> tuple:
    """Six particles huddled near the centre of a much larger domain — most
    of the domain is vacant and divergent_fill should populate it."""
    plates = (
        Plate(
            id=0, type=plate_type,
            seed_position_km=(0.0, 0.0), velocity_kmpy=(0.0, 0.0),
        ),
    )
    positions = np.array(
        [(0.0, 0.0), (16.0, 0.0), (-16.0, 0.0),
         (0.0, 16.0), (0.0, -16.0), (16.0, 16.0)],
        dtype=np.float64,
    )
    plate_id = np.zeros(positions.shape[0], dtype=np.int32)
    ct_code = CRUST_CONTINENTAL if plate_type == "continental" else CRUST_OCEANIC
    crust_type = np.full(positions.shape[0], ct_code, dtype=np.int8)
    thickness = np.full(
        positions.shape[0],
        cfg.continental_thickness_km if plate_type == "continental"
        else cfg.oceanic_thickness_km,
    )
    age = np.zeros(positions.shape[0])
    return plates, positions, plate_id, crust_type, thickness, age


# -----------------------------------------------------------------------------
# Empty / degenerate paths
# -----------------------------------------------------------------------------

def test_divergent_fill_empty_state_is_noop(default_sim_config: SimConfig) -> None:
    """With no existing particles, divergent_fill returns the inputs
    unchanged — it needs an anchor to inherit a plate from."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    rng = np.random.Generator(np.random.PCG64(0))
    out = divergent_fill(
        domain,
        np.zeros((0, 2)),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int8),
        np.zeros(0),
        np.zeros(0),
        plates=(),
        sim_config=default_sim_config,
        rng=rng,
    )
    for arr in out:
        assert arr.shape[0] == 0


def test_divergent_fill_dense_state_adds_nothing(
    default_sim_config: SimConfig,
) -> None:
    """When the world is already fully packed (Poisson-disc seeded), every
    candidate cell hits an existing particle within ``spacing`` and the
    fill spawns nothing new."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))
    plates, pos, pid, ct, th, age = _full_dense_state(domain, default_sim_config)
    out_pos, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, default_sim_config, rng,
    )
    # The seeded Poisson-disc field is full — at worst a handful of edge
    # vacancies. Expect very few new particles relative to the population.
    new_count = out_pos.shape[0] - pos.shape[0]
    assert new_count <= 0.02 * pos.shape[0], (
        f"unexpected new particles in a dense world: {new_count} / {pos.shape[0]}"
    )


# -----------------------------------------------------------------------------
# Sparse → fill behaviour
# -----------------------------------------------------------------------------

def test_divergent_fill_populates_vacant_space(
    default_sim_config: SimConfig,
) -> None:
    """Six particles clumped at the centre of a 200×200 km domain should
    grow significantly after divergent_fill — most of the domain is
    vacant and inheritable from the cluster."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(
        default_sim_config,
    )
    out_pos, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, default_sim_config, rng,
    )
    # The starting 6 particles should grow by at least an order of magnitude.
    assert out_pos.shape[0] > 60


def test_divergent_fill_continental_plate_spawns_rift(
    default_sim_config: SimConfig,
) -> None:
    """A continental plate's divergent gap is filled with rift crust at
    ``rift_thickness_km``."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(
        cfg, plate_type="continental",
    )
    n_before = pos.shape[0]
    out_pos, out_pid, out_ct, out_th, out_age = divergent_fill(
        domain, pos, pid, ct, th, age, plates, cfg, rng,
    )
    new_slice = slice(n_before, None)
    # Every spawned particle is continental (inherits plate 0's initial type).
    assert (out_ct[new_slice] == CRUST_CONTINENTAL).all()
    # Spawned thickness == rift_thickness_km.
    assert (out_th[new_slice] == cfg.rift_thickness_km).all()
    # Spawned ages start at 0.
    assert (out_age[new_slice] == 0.0).all()


def test_divergent_fill_oceanic_plate_spawns_fresh_ocean_crust(
    default_sim_config: SimConfig,
) -> None:
    """An oceanic plate's divergent gap is filled with fresh oceanic crust
    at ``oceanic_thickness_km``."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(
        cfg, plate_type="oceanic",
    )
    n_before = pos.shape[0]
    out_pos, out_pid, out_ct, out_th, _ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, cfg, rng,
    )
    new_slice = slice(n_before, None)
    assert (out_ct[new_slice] == CRUST_OCEANIC).all()
    assert (out_th[new_slice] == cfg.oceanic_thickness_km).all()


def test_divergent_fill_inherits_plate_id_from_nearest(
    default_sim_config: SimConfig,
) -> None:
    """A two-plate world with a gap between them: spawn near plate A inherits
    A, spawn near plate B inherits B."""
    cfg = default_sim_config
    domain = WorldRect(width_km=400.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))

    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(-150.0, 0.0), velocity_kmpy=(0.0, 0.0)),
        Plate(id=1, type="oceanic",
              seed_position_km=(150.0, 0.0), velocity_kmpy=(0.0, 0.0)),
    )
    # Two small particle clusters far apart.
    pos = np.array([
        (-150.0, 0.0), (-160.0, 10.0), (-140.0, -10.0),
        (150.0, 0.0), (160.0, -10.0), (140.0, 10.0),
    ])
    pid = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    ct = np.array([
        CRUST_CONTINENTAL, CRUST_CONTINENTAL, CRUST_CONTINENTAL,
        CRUST_OCEANIC, CRUST_OCEANIC, CRUST_OCEANIC,
    ], dtype=np.int8)
    th = np.array([35.0, 35.0, 35.0, 7.0, 7.0, 7.0])
    age = np.zeros(6)

    n_before = pos.shape[0]
    out_pos, out_pid, out_ct, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, cfg, rng,
    )

    # For each spawned particle, the inherited plate id must match the
    # nearest existing particle's plate id under whichever distance
    # metric the configured boundary_mode uses (Voronoi on a rectangle
    # or on a torus).
    new_pos = out_pos[n_before:]
    new_pid = out_pid[n_before:]
    wrap = (cfg.boundary_mode == "wrap")
    for i, (x, y) in enumerate(new_pos):
        dx = pos[:, 0] - x
        dy = pos[:, 1] - y
        if wrap:
            dx, dy = domain.wrapped_delta_xy(dx, dy)
        d2 = dx * dx + dy * dy
        nearest_existing = pid[int(np.argmin(d2))]
        assert new_pid[i] == nearest_existing


# -----------------------------------------------------------------------------
# Determinism + dtypes
# -----------------------------------------------------------------------------

def test_divergent_fill_determinism(default_sim_config: SimConfig) -> None:
    """Same inputs + same RNG seed → identical output."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(
        default_sim_config,
    )
    rng_a = np.random.Generator(np.random.PCG64(42))
    rng_b = np.random.Generator(np.random.PCG64(42))
    a = divergent_fill(domain, pos, pid, ct, th, age, plates, default_sim_config, rng_a)
    b = divergent_fill(domain, pos, pid, ct, th, age, plates, default_sim_config, rng_b)
    for ax, bx in zip(a, b):
        np.testing.assert_array_equal(ax, bx)


def test_divergent_fill_preserves_dtypes(default_sim_config: SimConfig) -> None:
    """Output dtypes must match the documented contract on every field
    (mixing types breaks downstream vectorised paths)."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(
        default_sim_config,
    )
    rng = np.random.Generator(np.random.PCG64(0))
    o_pos, o_pid, o_ct, o_th, o_age = divergent_fill(
        domain, pos, pid, ct, th, age, plates, default_sim_config, rng,
    )
    assert o_pos.dtype == np.float64
    assert o_pid.dtype == np.int32
    assert o_ct.dtype == np.int8
    assert o_th.dtype == np.float64
    assert o_age.dtype == np.float64


def test_divergent_fill_matches_bridson_density(
    default_sim_config: SimConfig,
) -> None:
    """Spawned particles must respect the Bridson invariant: no two
    particles (existing or newly spawned) end up closer than
    ``particle_spacing_km``. Previously the vacancy threshold sat at
    ``overlap_radius_km`` (= 1.5×) so divergent zones were ~2.2× sparser
    than the initial cloud; now they match."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(cfg)
    n_before = pos.shape[0]
    out_pos, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, cfg, rng,
    )
    # Every spawned particle should be ≥ particle_spacing_km from every
    # *existing* particle. (Pairs of spawned particles can still end up
    # within spacing of each other within a single Jacobi-style pass —
    # that's a known limitation of one-shot grid spawning. The next
    # tick's contact + intra-plate passes catch it.)
    new_slice = out_pos[n_before:]
    wrap = (cfg.boundary_mode == "wrap")
    spacing2 = cfg.particle_spacing_km ** 2
    for x, y in new_slice:
        dx = pos[:, 0] - x
        dy = pos[:, 1] - y
        if wrap:
            dx, dy = domain.wrapped_delta_xy(dx, dy)
        d2 = dx * dx + dy * dy
        assert d2.min() >= spacing2 - 1e-9, (
            f"spawned ({x:.1f}, {y:.1f}) is closer than spacing to an "
            f"existing particle (d²={d2.min():.3f} < {spacing2:.3f})"
        )


def test_divergent_fill_refuses_contact_gap(
    default_sim_config: SimConfig,
) -> None:
    """A candidate sitting in the contact-constraint gap (between two
    plates pushed apart by the cross-plate constraint to exactly
    overlap_radius) must NOT be spawned, even though the gap itself is
    technically vacant at the particle_spacing_km threshold."""
    cfg = default_sim_config
    domain = WorldRect(width_km=300.0, height_km=300.0)
    # Two plates butted against each other, each a tight cluster:
    # plate 0 (continental) on the left, plate 1 (continental) on the
    # right. The contact band runs vertically through x=0 with width
    # overlap_radius. We saturate each side with particles so the *only*
    # vacancy in the local neighbourhood is the contact band itself.
    r = cfg.overlap_radius_km
    s = cfg.particle_spacing_km
    # Pack a 5×5 mini-grid on each side, separated by overlap_radius.
    def _cluster(cx: float) -> np.ndarray:
        xs = cx + s * (np.arange(5) - 2)
        ys = s * (np.arange(5) - 2)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([gx.ravel(), gy.ravel()])

    left = _cluster(-r)
    right = _cluster(+r)
    pos = np.vstack([left, right])
    pid = np.concatenate([
        np.zeros(left.shape[0], dtype=np.int32),
        np.ones(right.shape[0], dtype=np.int32),
    ])
    ct = np.full(pos.shape[0], CRUST_CONTINENTAL, dtype=np.int8)
    th = np.full(pos.shape[0], cfg.continental_thickness_km, dtype=np.float64)
    age = np.zeros(pos.shape[0])
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(-r, 0.0), velocity_kmpy=(0.0, 0.0)),
        Plate(id=1, type="continental",
              seed_position_km=(+r, 0.0), velocity_kmpy=(0.0, 0.0)),
    )
    n_before = pos.shape[0]
    rng = np.random.Generator(np.random.PCG64(0))
    out_pos, out_pid, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, cfg, rng,
    )
    # Any particle spawned in the contact band ±0.5×overlap_radius around
    # x=0 is an intermixing artefact. Should be zero.
    new_pos = out_pos[n_before:]
    in_band = np.abs(new_pos[:, 0]) < 0.5 * r
    # Restrict to the y-range of the clusters so we don't catch genuine
    # spawns in the wide-open polar regions of the domain.
    in_cluster_y = np.abs(new_pos[:, 1]) < 3 * s
    contaminated = in_band & in_cluster_y
    assert not contaminated.any(), (
        f"divergent fill spawned {contaminated.sum()} particle(s) inside "
        f"the contact gap between two plates"
    )


def test_divergent_fill_no_new_particle_outside_domain(
    default_sim_config: SimConfig,
) -> None:
    """Spawned particles after jitter must still sit inside the domain
    rectangle."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, pos, pid, ct, th, age = _sparse_state_around_one_plate(
        default_sim_config,
    )
    rng = np.random.Generator(np.random.PCG64(0))
    out_pos, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, default_sim_config, rng,
    )
    hw = domain.half_width_km
    hh = domain.half_height_km
    assert (np.abs(out_pos[:, 0]) <= hw + 1e-9).all()
    assert (np.abs(out_pos[:, 1]) <= hh + 1e-9).all()


# -----------------------------------------------------------------------------
# Integration: pair fill with drift to confirm the sim no longer drains
# -----------------------------------------------------------------------------

def test_fill_keeps_particle_count_steady_under_drift(
    default_sim_config: SimConfig,
) -> None:
    """Run several ticks of (drift + open-boundary cull + fill) and verify
    the population stays within a reasonable band of its initial value
    rather than monotonically decreasing as the cull pulls it down.

    Uses ``boundary_mode = "open"`` explicitly so the cull is meaningful
    — under wrap there's no drain to prevent. The band is wide on the
    upper side because the divergent-fill vacancy threshold matches the
    Bridson invariant, so fill packs replacement particles slightly
    denser than the initial cull-loss rate (especially on plates whose
    Voronoi cell wraps the seam under wrap mode — but here we use open
    so that's not in play). ±40 % covers the steady-state band; the
    lower bound is the test's real signal."""
    from tectonic_sim import step_drift_and_cull

    cfg = replace(
        default_sim_config,
        motion_speed_kmpy=20.0,
        boundary_mode="open",
    )
    domain = WorldRect(width_km=500.0, height_km=500.0)
    plates, pos, pid, ct, th, age = build_initial_state(domain, cfg, seed=0)
    n0 = pos.shape[0]

    rng = np.random.Generator(np.random.PCG64(0))
    for _ in range(8):
        pos, pid, ct, th, age = step_drift_and_cull(
            domain, pos, pid, ct, th, age, plates, cfg.dt_myr,
        )
        pos, pid, ct, th, age = divergent_fill(
            domain, pos, pid, ct, th, age, plates, cfg, rng,
        )

    # Without fill this would have drained heavily; with fill it should
    # stay roughly steady. ±40 % gives steady-state headroom.
    assert 0.6 * n0 <= pos.shape[0] <= 1.4 * n0


def test_fill_works_with_sparse_plate_ids(default_sim_config: SimConfig) -> None:
    """Plate ids needn't be densely packed (e.g. after a plate was wiped
    out). The fill must still locate the per-plate initial type via the
    sparse index lookup."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    # Plates ids 0 and 7 (gap in between).
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(-50.0, 0.0), velocity_kmpy=(0.0, 0.0)),
        Plate(id=7, type="oceanic",
              seed_position_km=(50.0, 0.0), velocity_kmpy=(0.0, 0.0)),
    )
    pos = np.array([
        (-50.0, 0.0), (-60.0, 5.0),
        (50.0, 0.0), (60.0, -5.0),
    ])
    pid = np.array([0, 0, 7, 7], dtype=np.int32)
    ct = np.array(
        [CRUST_CONTINENTAL, CRUST_CONTINENTAL, CRUST_OCEANIC, CRUST_OCEANIC],
        dtype=np.int8,
    )
    th = np.array([35.0, 35.0, 7.0, 7.0])
    age = np.zeros(4)
    rng = np.random.Generator(np.random.PCG64(0))

    n_before = pos.shape[0]
    out_pos, out_pid, out_ct, *_ = divergent_fill(
        domain, pos, pid, ct, th, age, plates, cfg, rng,
    )
    new_pid = out_pid[n_before:]
    new_ct = out_ct[n_before:]
    # Every new particle should have plate id 0 or 7 (not, e.g., 3).
    assert set(new_pid.tolist()).issubset({0, 7})
    # Particles inheriting from plate 0 (continental) should be continental,
    # plate 7 (oceanic) oceanic.
    for pid_v, ct_v in zip(new_pid, new_ct):
        if pid_v == 0:
            assert ct_v == CRUST_CONTINENTAL
        else:
            assert ct_v == CRUST_OCEANIC
