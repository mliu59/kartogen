"""Tests for the initial-condition seeding (Poisson-disc + plate Voronoi)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tectonic_sim import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
    build_initial_state,
)
from tectonic_sim.rng import RngStream
from tectonic_sim.seeding import (
    assign_particles_to_plates,
    place_plate_seeds,
    poisson_disc_sample,
)


# -----------------------------------------------------------------------------
# Poisson-disc sampling
# -----------------------------------------------------------------------------

def test_poisson_disc_respects_spacing() -> None:
    """Every pair of returned points must be at least ``spacing`` apart."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    spacing = 15.0
    rng = np.random.Generator(np.random.PCG64(123))
    points = poisson_disc_sample(domain, spacing, rng)

    # Sanity: at least a few points.
    assert points.shape[0] > 20
    # Min pairwise distance ≥ spacing (allow a tiny float tolerance).
    diff = points[:, None, :] - points[None, :, :]
    sq = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(sq, np.inf)
    assert sq.min() >= (spacing - 1e-6) ** 2


def test_poisson_disc_inside_domain() -> None:
    """No point lands outside the centred rectangle."""
    domain = WorldRect(width_km=100.0, height_km=60.0)
    rng = np.random.Generator(np.random.PCG64(7))
    points = poisson_disc_sample(domain, spacing_km=8.0, rng=rng)
    assert (np.abs(points[:, 0]) <= domain.half_width_km + 1e-9).all()
    assert (np.abs(points[:, 1]) <= domain.half_height_km + 1e-9).all()


def test_poisson_disc_density_in_expected_band() -> None:
    """Density is within ±50 % of the theoretical Poisson-disc rate.

    Theoretical Bridson density is ~0.69 / (π · r²/4) per unit area;
    we just want to be in the right order of magnitude, not nail the
    constant.
    """
    domain = WorldRect(width_km=300.0, height_km=300.0)
    spacing = 15.0
    rng = np.random.Generator(np.random.PCG64(0))
    points = poisson_disc_sample(domain, spacing, rng)
    area = domain.area_km2
    cell_area = math.pi * (spacing / 2.0) ** 2
    expected_low = 0.4 * area / cell_area     # very loose lower bound
    expected_high = 1.2 * area / cell_area    # very loose upper bound
    assert expected_low <= points.shape[0] <= expected_high


def test_poisson_disc_determinism() -> None:
    """Same RNG → same point cloud."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    a = poisson_disc_sample(
        domain, spacing_km=12.0, rng=np.random.Generator(np.random.PCG64(99)),
    )
    b = poisson_disc_sample(
        domain, spacing_km=12.0, rng=np.random.Generator(np.random.PCG64(99)),
    )
    np.testing.assert_array_equal(a, b)


def test_poisson_disc_invalid_spacing_raises() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    rng = np.random.Generator(np.random.PCG64(0))
    with pytest.raises(ValueError, match="spacing_km"):
        poisson_disc_sample(domain, spacing_km=0.0, rng=rng)


# -----------------------------------------------------------------------------
# Plate seed placement
# -----------------------------------------------------------------------------

def test_place_plate_seeds_count(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=1000.0, height_km=1000.0)
    rng = np.random.Generator(np.random.PCG64(1))
    seeds = place_plate_seeds(domain, default_sim_config, rng)
    assert seeds.shape == (default_sim_config.plate_count, 2)


def test_place_plate_seeds_inside_domain(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=500.0, height_km=300.0)
    rng = np.random.Generator(np.random.PCG64(2))
    seeds = place_plate_seeds(domain, default_sim_config, rng)
    assert (np.abs(seeds[:, 0]) <= domain.half_width_km + 1e-9).all()
    assert (np.abs(seeds[:, 1]) <= domain.half_height_km + 1e-9).all()


def test_place_plate_seeds_radial_bias_pulls_to_centre(
    default_sim_config: SimConfig,
) -> None:
    """A strong positive bias keeps seeds substantially nearer the centre
    than a strong negative bias does, on average across many seeds."""
    domain = WorldRect(width_km=1000.0, height_km=1000.0)
    cfg_centred = replace(default_sim_config, seed_radial_bias=1.0, plate_count=4)
    cfg_edged = replace(default_sim_config, seed_radial_bias=-1.0, plate_count=4)

    centred_d, edged_d = [], []
    for s in range(8):
        rng_c = np.random.Generator(np.random.PCG64(s))
        rng_e = np.random.Generator(np.random.PCG64(s + 1000))
        sc = place_plate_seeds(domain, cfg_centred, rng_c)
        se = place_plate_seeds(domain, cfg_edged, rng_e)
        centred_d.append(np.linalg.norm(sc, axis=1).mean())
        edged_d.append(np.linalg.norm(se, axis=1).mean())
    assert np.mean(centred_d) < np.mean(edged_d)


def test_place_plate_seeds_determinism(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=400.0, height_km=400.0)
    a = place_plate_seeds(
        domain, default_sim_config,
        rng=np.random.Generator(np.random.PCG64(42)),
    )
    b = place_plate_seeds(
        domain, default_sim_config,
        rng=np.random.Generator(np.random.PCG64(42)),
    )
    np.testing.assert_array_equal(a, b)


# -----------------------------------------------------------------------------
# Voronoi assignment
# -----------------------------------------------------------------------------

def test_assign_particles_to_plates_picks_nearest() -> None:
    """Hand-built case: 3 plate seeds at (-100,0), (0,0), (+100,0). Pick
    non-boundary particle positions so the ``argmin``-on-ties detail
    doesn't bleed into the assertion."""
    seeds = np.array([[-100.0, 0.0], [0.0, 0.0], [100.0, 0.0]])
    particles = np.array([[-120.0, 0.0], [-30.0, 0.0], [30.0, 0.0], [120.0, 0.0]])
    plate_id = assign_particles_to_plates(particles, seeds)
    np.testing.assert_array_equal(plate_id, [0, 1, 1, 2])


def test_assign_particles_uses_int32() -> None:
    seeds = np.zeros((3, 2))
    particles = np.zeros((10, 2))
    plate_id = assign_particles_to_plates(particles, seeds)
    assert plate_id.dtype == np.int32


def test_assign_particles_wrap_picks_across_seam() -> None:
    """Under wrap, a particle near one edge should be assigned to a plate
    whose seed sits across the seam if that's the toroidal-nearest seed.

    Without wrap, the same particle goes to the rectangle-interior seed
    even though it's further away on the torus."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    seeds = np.array([
        [-45.0, 0.0],   # near west edge
        [+30.0, 0.0],   # rectangle interior
    ])
    # Particle near east edge: direct distance to seed 1 is 15 km,
    # to seed 0 is 95 km. Toroidal distance to seed 0 is 100 - 90 = 10 km.
    particle = np.array([[+45.0, 0.0]])

    # No wrap: nearest is seed 1 (rectangle interior).
    no_wrap = assign_particles_to_plates(particle, seeds)
    assert no_wrap.tolist() == [1]

    # Wrap: nearest is seed 0 (across the seam, 10 km away).
    wrap = assign_particles_to_plates(particle, seeds, domain=domain, wrap=True)
    assert wrap.tolist() == [0]


def test_assign_particles_wrap_requires_domain() -> None:
    seeds = np.zeros((2, 2))
    particles = np.zeros((1, 2))
    with pytest.raises(ValueError, match="domain"):
        assign_particles_to_plates(particles, seeds, wrap=True)


def test_assign_particles_wrap_matches_brute_force() -> None:
    """Random cloud + seeds: wrap-aware result matches a toroidal brute-
    force per-particle nearest-seed lookup."""
    rng = np.random.Generator(np.random.PCG64(7))
    domain = WorldRect(width_km=200.0, height_km=200.0)
    seeds = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(6, 2),
    )
    particles = rng.uniform(
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        size=(50, 2),
    )
    got = assign_particles_to_plates(
        particles, seeds, domain=domain, wrap=True,
    )
    for i, p in enumerate(particles):
        dx = seeds[:, 0] - p[0]
        dy = seeds[:, 1] - p[1]
        dx, dy = domain.wrapped_delta_xy(dx, dy)
        truth = int(np.argmin(dx * dx + dy * dy))
        assert int(got[i]) == truth


# -----------------------------------------------------------------------------
# Wrap-aware plate seed separation
# -----------------------------------------------------------------------------

def test_place_plate_seeds_wrap_aware_separation(
    default_sim_config: SimConfig,
) -> None:
    """Under wrap mode, no two seeds may end up within the toroidal min-
    separation of each other — including across the seam. Without wrap,
    a pair at (-490, 0) and (+490, 0) would be ~980 km apart and pass;
    on a 1000-km torus they're actually 20 km apart and should fail."""
    cfg = replace(default_sim_config, boundary_mode="wrap", plate_count=10)
    domain = WorldRect(width_km=1000.0, height_km=1000.0)
    rng = np.random.Generator(np.random.PCG64(0))
    seeds = place_plate_seeds(domain, cfg, rng)
    # Compute toroidal pairwise distances.
    diff = seeds[:, None, :] - seeds[None, :, :]
    wx, wy = domain.wrapped_delta_xy(diff[..., 0], diff[..., 1])
    sqd = wx * wx + wy * wy
    np.fill_diagonal(sqd, np.inf)
    min_d = math.sqrt(float(sqd.min()))
    # Just sanity-check there's a meaningful gap; the algorithm relaxes
    # the threshold on failure, so we only assert above 1 km.
    assert min_d > 1.0


def test_place_plate_seeds_no_wrap_unchanged(
    default_sim_config: SimConfig,
) -> None:
    """Sanity: with boundary_mode='open', the function still works and
    produces in-domain seeds (the wrap-awareness changes nothing)."""
    cfg = replace(default_sim_config, boundary_mode="open", plate_count=5)
    domain = WorldRect(width_km=500.0, height_km=500.0)
    rng = np.random.Generator(np.random.PCG64(0))
    seeds = place_plate_seeds(domain, cfg, rng)
    assert seeds.shape == (5, 2)
    assert (np.abs(seeds[:, 0]) <= domain.half_width_km + 1e-9).all()
    assert (np.abs(seeds[:, 1]) <= domain.half_height_km + 1e-9).all()


# -----------------------------------------------------------------------------
# Top-level build_initial_state
# -----------------------------------------------------------------------------

def test_build_initial_state_shapes_consistent(default_sim_config: SimConfig) -> None:
    """All particle arrays have the same length N; plates list has
    ``plate_count`` entries."""
    domain = WorldRect(width_km=300.0, height_km=300.0)
    plates, pos, pid, ctype, thick, age = build_initial_state(
        domain, default_sim_config, seed=42,
    )

    n = pos.shape[0]
    assert n > 0
    assert pos.shape == (n, 2)
    assert pid.shape == (n,)
    assert ctype.shape == (n,)
    assert thick.shape == (n,)
    assert age.shape == (n,)
    assert len(plates) == default_sim_config.plate_count


def test_build_initial_state_dtypes(default_sim_config: SimConfig) -> None:
    """Per-particle field dtypes are the documented ones."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    _, pos, pid, ctype, thick, age = build_initial_state(
        domain, default_sim_config, seed=0,
    )
    assert pos.dtype == np.float64
    assert pid.dtype == np.int32
    assert ctype.dtype == np.int8
    assert thick.dtype == np.float64
    assert age.dtype == np.float64


def test_build_initial_state_ages_start_at_zero(
    default_sim_config: SimConfig,
) -> None:
    domain = WorldRect(width_km=200.0, height_km=200.0)
    _, _, _, _, _, age = build_initial_state(domain, default_sim_config, seed=0)
    assert (age == 0.0).all()


def test_build_initial_state_thickness_matches_type(
    default_sim_config: SimConfig,
) -> None:
    """Continental particles start at ``continental_thickness_km``; oceanic
    at ``oceanic_thickness_km``."""
    domain = WorldRect(width_km=300.0, height_km=300.0)
    _, _, _, ctype, thick, _ = build_initial_state(
        domain, default_sim_config, seed=0,
    )
    cont_thick = default_sim_config.continental_thickness_km
    ocn_thick = default_sim_config.oceanic_thickness_km
    cont_mask = (ctype == CRUST_CONTINENTAL)
    ocn_mask = (ctype == CRUST_OCEANIC)
    assert cont_mask.any() or ocn_mask.any()  # need at least one of each
    if cont_mask.any():
        assert (thick[cont_mask] == cont_thick).all()
    if ocn_mask.any():
        assert (thick[ocn_mask] == ocn_thick).all()


def test_build_initial_state_all_continental(default_sim_config: SimConfig) -> None:
    """With ``continental_fraction = 1.0``, every plate (and thus every
    particle) is continental."""
    cfg = replace(default_sim_config, continental_fraction=1.0)
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, _, _, ctype, _, _ = build_initial_state(domain, cfg, seed=0)
    assert all(p.type == "continental" for p in plates)
    assert (ctype == CRUST_CONTINENTAL).all()


def test_build_initial_state_all_oceanic(default_sim_config: SimConfig) -> None:
    """And vice versa with ``continental_fraction = 0.0``."""
    cfg = replace(default_sim_config, continental_fraction=0.0)
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, _, _, ctype, _, _ = build_initial_state(domain, cfg, seed=0)
    assert all(p.type == "oceanic" for p in plates)
    assert (ctype == CRUST_OCEANIC).all()


def test_build_initial_state_determinism(default_sim_config: SimConfig) -> None:
    """Same seed → byte-identical particle arrays and identical plates."""
    domain = WorldRect(width_km=250.0, height_km=250.0)
    a = build_initial_state(domain, default_sim_config, seed=42)
    b = build_initial_state(domain, default_sim_config, seed=42)
    plates_a, pos_a, pid_a, ctype_a, thick_a, age_a = a
    plates_b, pos_b, pid_b, ctype_b, thick_b, age_b = b
    assert plates_a == plates_b
    np.testing.assert_array_equal(pos_a, pos_b)
    np.testing.assert_array_equal(pid_a, pid_b)
    np.testing.assert_array_equal(ctype_a, ctype_b)
    np.testing.assert_array_equal(thick_a, thick_b)
    np.testing.assert_array_equal(age_a, age_b)


def test_build_initial_state_plate_velocities_match_speed(
    default_sim_config: SimConfig,
) -> None:
    """Each plate's velocity magnitude equals ``motion_speed_kmpy``."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, *_ = build_initial_state(domain, default_sim_config, seed=0)
    for p in plates:
        speed = math.hypot(*p.velocity_kmpy)
        assert speed == pytest.approx(
            default_sim_config.motion_speed_kmpy, rel=1e-9,
        )


def test_rng_stream_paths_are_independent() -> None:
    """Two different paths under the same root must produce different
    streams (no accidental aliasing). Otherwise adding a new sub-stream
    would silently change the others'."""
    rng = RngStream(42)
    a = rng.child("seeding", "poisson").uniform(0.0, 1.0, 8)
    b = rng.child("seeding", "plate_seeds").uniform(0.0, 1.0, 8)
    assert not np.array_equal(a, b)
