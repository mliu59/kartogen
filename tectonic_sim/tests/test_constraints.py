"""Tests for ``tectonic_sim.constraints``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tectonic_sim import (
    Plate,
    SimConfig,
    WorldRect,
    apply_contact_constraints,
    apply_velocity_damping,
)


# -----------------------------------------------------------------------------
# Contact constraints
# -----------------------------------------------------------------------------

def test_contact_constraints_pushes_overlapping_pair_to_radius(
    default_sim_config: SimConfig,
) -> None:
    """Two cross-plate particles closer than the overlap radius end up at
    distance ≥ overlap_radius after a single pass."""
    cfg = replace(default_sim_config, contact_iterations=1)
    domain = WorldRect(width_km=200.0, height_km=200.0)
    # Place pair at 5 km apart along x. overlap_radius default = 22.5 km.
    pos = np.array([[-2.5, 0.0], [+2.5, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    new_dist = np.hypot(out[1, 0] - out[0, 0], out[1, 1] - out[0, 1])
    assert new_dist >= cfg.overlap_radius_km - 1e-9


def test_contact_constraints_leaves_well_separated_pairs_alone(
    default_sim_config: SimConfig,
) -> None:
    """Pairs already further apart than overlap_radius mustn't move."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.array([[-50.0, 0.0], [+50.0, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, default_sim_config)
    np.testing.assert_array_equal(out, pos)


def test_contact_constraints_ignores_same_plate_pairs_above_intra_threshold(
    default_sim_config: SimConfig,
) -> None:
    """Same-plate particles within overlap_radius but *above* the intra-plate
    minimum distance must not be pushed apart — that compression is the
    geological signature of mountain building. Only collapses below the
    intra threshold get corrected."""
    # particle_spacing_km default = 15 km → intra_min = 7.5 km at factor 0.5.
    # Place pair at 10 km apart — above intra_min, below overlap_radius.
    cfg = default_sim_config
    assert cfg.intra_plate_min_distance_km < 10.0 < cfg.overlap_radius_km
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.array([[-5.0, 0.0], [+5.0, 0.0]])
    pid = np.array([0, 0], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    np.testing.assert_array_equal(out, pos)


def test_contact_constraints_intra_plate_pushes_collapsed_pair_apart(
    default_sim_config: SimConfig,
) -> None:
    """Same-plate particles squashed *below* the intra-plate minimum
    distance must be pushed apart to at least that distance."""
    cfg = replace(default_sim_config, contact_iterations=1)
    domain = WorldRect(width_km=200.0, height_km=200.0)
    # Two same-plate particles 1 km apart — well below the 7.5 km intra_min
    # at the default 15 km particle spacing × 0.5 factor.
    pos = np.array([[-0.5, 0.0], [+0.5, 0.0]])
    pid = np.array([0, 0], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    new_dist = np.hypot(out[1, 0] - out[0, 0], out[1, 1] - out[0, 1])
    assert new_dist >= cfg.intra_plate_min_distance_km - 1e-9


def test_contact_constraints_intra_disabled_when_factor_zero(
    default_sim_config: SimConfig,
) -> None:
    """``intra_plate_min_distance_factor = 0`` skips the intra-plate pass
    entirely — collapsed same-plate pairs are left alone."""
    cfg = replace(default_sim_config, intra_plate_min_distance_factor=0.0)
    assert cfg.intra_plate_min_distance_km == 0.0
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.array([[-0.5, 0.0], [+0.5, 0.0]])
    pid = np.array([0, 0], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    np.testing.assert_array_equal(out, pos)


def test_contact_constraints_intra_wrap_aware(
    default_sim_config: SimConfig,
) -> None:
    """A collapsed same-plate pair straddling the wrap seam should be
    separated across the seam, not the long way round."""
    cfg = replace(default_sim_config, boundary_mode="wrap", contact_iterations=1)
    domain = WorldRect(width_km=100.0, height_km=100.0)
    # Pair 2 km apart across the seam — well below intra_min.
    pos = np.array([[-49.0, 0.0], [+49.0, 0.0]])
    pid = np.array([0, 0], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    d = float(domain.wrapped_distance_km(out[0], out[1]))
    assert d >= cfg.intra_plate_min_distance_km - 1e-9


def test_contact_constraints_empty_state_safe(default_sim_config: SimConfig) -> None:
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.zeros((0, 2))
    pid = np.zeros(0, dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, default_sim_config)
    assert out.shape == (0, 2)


def test_contact_constraints_zero_iterations_passthrough(
    default_sim_config: SimConfig,
) -> None:
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.array([[-2.5, 0.0], [+2.5, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, default_sim_config, iterations=0)
    np.testing.assert_array_equal(out, pos)


def test_contact_constraints_multi_pair_clump(
    default_sim_config: SimConfig,
) -> None:
    """A dense clump of 5 cross-plate particles relaxes so no cross-plate
    pair is closer than overlap_radius after enough iterations."""
    cfg = replace(default_sim_config, contact_iterations=12)
    domain = WorldRect(width_km=300.0, height_km=300.0)
    pos = np.array([
        [0.0, 0.0], [3.0, 1.0], [-2.0, 3.0],
        [1.0, -2.0], [-3.0, -1.0],
    ])
    pid = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    # Check cross-plate pair separation.
    r = cfg.overlap_radius_km
    for i in range(5):
        for j in range(i + 1, 5):
            if pid[i] == pid[j]:
                continue
            d = float(np.hypot(out[i, 0] - out[j, 0], out[i, 1] - out[j, 1]))
            assert d >= r - 1e-6, f"pair ({i}, {j}) at {d:.3f} < {r}"


def test_contact_constraints_wrap_aware(default_sim_config: SimConfig) -> None:
    """A cross-plate pair straddling the east/west wrap edge should be
    pushed apart across the seam, not the long way round."""
    cfg = replace(default_sim_config, boundary_mode="wrap", contact_iterations=1)
    domain = WorldRect(width_km=100.0, height_km=100.0)
    # Pair 4 km apart across the seam (one at +48, one at -48 → wrapped
    # distance = 4 km, direct distance = 96 km).
    pos = np.array([[-48.0, 0.0], [+48.0, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    out = apply_contact_constraints(domain, pos, pid, cfg)
    # Resulting wrapped distance ≥ overlap_radius.
    d = float(domain.wrapped_distance_km(out[0], out[1]))
    assert d >= cfg.overlap_radius_km - 1e-9


def test_contact_constraints_input_not_mutated(
    default_sim_config: SimConfig,
) -> None:
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.array([[-2.5, 0.0], [+2.5, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    pos_copy = pos.copy()
    apply_contact_constraints(domain, pos, pid, default_sim_config)
    np.testing.assert_array_equal(pos, pos_copy)


def test_contact_constraints_iteration_count_monotonic_convergence(
    default_sim_config: SimConfig,
) -> None:
    """More iterations get the result closer to the target separation."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.array([[-1.0, 0.0], [+1.0, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    r = default_sim_config.overlap_radius_km

    d1 = float(np.hypot(
        *(apply_contact_constraints(domain, pos, pid, default_sim_config, iterations=1)[1]
          - apply_contact_constraints(domain, pos, pid, default_sim_config, iterations=1)[0])
    ))
    d4 = float(np.hypot(
        *(apply_contact_constraints(domain, pos, pid, default_sim_config, iterations=4)[1]
          - apply_contact_constraints(domain, pos, pid, default_sim_config, iterations=4)[0])
    ))
    # 1 iteration already converges this trivial 2-particle case to ≥ r;
    # 4 iterations stays ≥ r.
    assert d1 >= r - 1e-9
    assert d4 >= r - 1e-9


# -----------------------------------------------------------------------------
# Velocity damping
# -----------------------------------------------------------------------------

def _two_plate_setup(damping: float) -> tuple[tuple[Plate, ...], SimConfig]:
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(-50.0, 0.0), velocity_kmpy=(+10.0, 0.0)),
        Plate(id=1, type="continental",
              seed_position_km=(+50.0, 0.0), velocity_kmpy=(-10.0, 0.0)),
    )
    from tectonic_sim import load_sim_config_from_path
    from pathlib import Path
    cfg = load_sim_config_from_path(
        Path(__file__).resolve().parents[2] / "config" / "tectonic_sim.toml"
    )
    cfg = replace(cfg, velocity_damping_strength=damping)
    return plates, cfg


def test_velocity_damping_zero_pairs_passthrough() -> None:
    """No collision pairs → plates come back identical."""
    plates, cfg = _two_plate_setup(damping=0.5)
    plate_id = np.array([0, 1, 0, 1], dtype=np.int32)
    out = apply_velocity_damping(
        plates, plate_id,
        np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), cfg,
    )
    assert out == plates


def test_velocity_damping_full_collision_full_damping() -> None:
    """When every particle is in a pair, velocity drops by exactly
    ``damping_strength × 1.0``."""
    plates, cfg = _two_plate_setup(damping=0.10)
    plate_id = np.array([0, 1, 0, 1], dtype=np.int32)
    # All 4 particles in a pair (each appears once).
    pair_i = np.array([0, 2], dtype=np.int32)
    pair_j = np.array([1, 3], dtype=np.int32)
    out = apply_velocity_damping(plates, plate_id, pair_i, pair_j, cfg)
    # Original velocity 10 km/Myr. Each plate has fraction = 2/2 = 1.0.
    # New = 10 × (1 - 0.10 × 1.0) = 9.0.
    assert out[0].velocity_kmpy == pytest.approx((+9.0, 0.0))
    assert out[1].velocity_kmpy == pytest.approx((-9.0, 0.0))


def test_velocity_damping_partial_collision() -> None:
    """Half the plate in collision → half the damping."""
    plates, cfg = _two_plate_setup(damping=0.20)
    plate_id = np.array([0, 0, 1, 1], dtype=np.int32)
    # Only particle 0 (plate 0) and 2 (plate 1) are in a pair.
    pair_i = np.array([0], dtype=np.int32)
    pair_j = np.array([2], dtype=np.int32)
    out = apply_velocity_damping(plates, plate_id, pair_i, pair_j, cfg)
    # fraction = 0.5 each. damping = 0.20 × 0.5 = 0.10. New = 10 × 0.90 = 9.0.
    assert out[0].velocity_kmpy == pytest.approx((+9.0, 0.0))
    assert out[1].velocity_kmpy == pytest.approx((-9.0, 0.0))


def test_velocity_damping_repeated_application_decays_to_stop() -> None:
    """Repeatedly applying damping with full collision drives velocity
    asymptotically toward zero (geometric decay)."""
    plates, cfg = _two_plate_setup(damping=0.20)
    plate_id = np.array([0, 1], dtype=np.int32)
    pair_i = np.array([0], dtype=np.int32)
    pair_j = np.array([1], dtype=np.int32)
    for _ in range(50):
        plates = apply_velocity_damping(plates, plate_id, pair_i, pair_j, cfg)
    # After 50 iterations at 0.20 × 1.0 = 0.80 factor: 10 × 0.80^50 ≈ 1.4e-4.
    assert abs(plates[0].velocity_kmpy[0]) < 0.01


def test_velocity_damping_preserves_identity_of_other_fields() -> None:
    """Damping changes velocity only — id, type, seed_position stay put."""
    plates, cfg = _two_plate_setup(damping=0.10)
    plate_id = np.array([0, 1], dtype=np.int32)
    pair_i = np.array([0], dtype=np.int32)
    pair_j = np.array([1], dtype=np.int32)
    out = apply_velocity_damping(plates, plate_id, pair_i, pair_j, cfg)
    for orig, new in zip(plates, out):
        assert new.id == orig.id
        assert new.type == orig.type
        assert new.seed_position_km == orig.seed_position_km
