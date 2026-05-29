"""Tests for ``tectonic_sim.kinematics``."""

from __future__ import annotations

import numpy as np
import pytest

from tectonic_sim import Plate, SimConfig, WorldRect, build_initial_state
from tectonic_sim.kinematics import (
    cull_outside_domain,
    drift_positions,
    step_drift_and_apply_boundary,
    step_drift_and_cull,
    wrap_positions,
)
from tectonic_sim.types import CRUST_CONTINENTAL, CRUST_OCEANIC


# -----------------------------------------------------------------------------
# drift_positions
# -----------------------------------------------------------------------------

def test_drift_moves_by_velocity_times_dt() -> None:
    """Particles translate by exactly ``v · dt`` in km."""
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(10.0, -5.0)),
    )
    pos = np.array([[0.0, 0.0], [50.0, 50.0], [-100.0, 25.0]])
    pid = np.zeros(3, dtype=np.int32)
    new_pos = drift_positions(pos, pid, plates, dt_myr=2.0)
    expected = pos + np.array([20.0, -10.0])
    np.testing.assert_array_equal(new_pos, expected)


def test_drift_input_is_not_mutated() -> None:
    """``drift_positions`` returns a new array; the input stays intact."""
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(1.0, 0.0)),
    )
    pos = np.array([[0.0, 0.0]])
    pid = np.zeros(1, dtype=np.int32)
    drift_positions(pos, pid, plates, dt_myr=1.0)
    np.testing.assert_array_equal(pos, np.array([[0.0, 0.0]]))


def test_drift_different_plates_use_their_own_velocity() -> None:
    """Each particle moves by its plate's velocity, not a global one."""
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(10.0, 0.0)),
        Plate(id=1, type="oceanic",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(0.0, -20.0)),
    )
    pos = np.array([[0.0, 0.0], [0.0, 0.0]])
    pid = np.array([0, 1], dtype=np.int32)
    new_pos = drift_positions(pos, pid, plates, dt_myr=1.0)
    np.testing.assert_array_equal(new_pos[0], [10.0, 0.0])
    np.testing.assert_array_equal(new_pos[1], [0.0, -20.0])


def test_drift_zero_dt_no_change() -> None:
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(50.0, 50.0)),
    )
    pos = np.array([[1.0, 2.0]])
    pid = np.zeros(1, dtype=np.int32)
    new_pos = drift_positions(pos, pid, plates, dt_myr=0.0)
    np.testing.assert_array_equal(new_pos, pos)


def test_drift_handles_empty_state() -> None:
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(1.0, 1.0)),
    )
    pos = np.zeros((0, 2))
    pid = np.zeros(0, dtype=np.int32)
    new_pos = drift_positions(pos, pid, plates, dt_myr=5.0)
    assert new_pos.shape == (0, 2)


def test_drift_handles_sparse_plate_ids() -> None:
    """Plate ids aren't necessarily 0..P-1 after subductions — drift must
    still look up the right velocity for whichever ids survive."""
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(1.0, 0.0)),
        Plate(id=5, type="oceanic",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(0.0, 1.0)),
    )
    pos = np.array([[0.0, 0.0], [0.0, 0.0]])
    pid = np.array([0, 5], dtype=np.int32)
    new_pos = drift_positions(pos, pid, plates, dt_myr=3.0)
    np.testing.assert_array_equal(new_pos[0], [3.0, 0.0])
    np.testing.assert_array_equal(new_pos[1], [0.0, 3.0])


# -----------------------------------------------------------------------------
# cull_outside_domain
# -----------------------------------------------------------------------------

def test_cull_keeps_only_in_domain() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    pos = np.array([
        [0.0, 0.0],         # inside
        [50.0, 50.0],       # on the corner — kept (inclusive)
        [60.0, 0.0],        # outside east
        [0.0, -120.0],      # outside south
        [-50.0, 50.0],      # on the corner — kept
    ])
    pid = np.arange(5, dtype=np.int32)
    (pos_out, pid_out) = cull_outside_domain(domain, pos, pid)
    np.testing.assert_array_equal(pid_out, [0, 1, 4])
    assert pos_out.shape == (3, 2)


def test_cull_filters_all_parallel_arrays() -> None:
    """Every passed array is filtered with the same mask."""
    domain = WorldRect(width_km=50.0, height_km=50.0)
    pos = np.array([[10.0, 10.0], [100.0, 0.0], [-10.0, 10.0]])
    pid = np.array([7, 8, 9], dtype=np.int32)
    thickness = np.array([35.0, 7.0, 33.0])
    ct = np.array([CRUST_CONTINENTAL, CRUST_OCEANIC, CRUST_CONTINENTAL],
                  dtype=np.int8)
    pos_out, pid_out, thk_out, ct_out = cull_outside_domain(
        domain, pos, pid, thickness, ct,
    )
    np.testing.assert_array_equal(pid_out, [7, 9])
    np.testing.assert_array_equal(thk_out, [35.0, 33.0])
    np.testing.assert_array_equal(ct_out, [CRUST_CONTINENTAL, CRUST_CONTINENTAL])


def test_cull_inside_only_is_passthrough() -> None:
    """When everything is in-domain, the same arrays come back unchanged."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    pos = np.array([[0.0, 0.0], [10.0, 10.0]])
    pid = np.array([0, 1], dtype=np.int32)
    pos_out, pid_out = cull_outside_domain(domain, pos, pid)
    np.testing.assert_array_equal(pos_out, pos)
    np.testing.assert_array_equal(pid_out, pid)


def test_cull_empty_state_is_safe() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    pos = np.zeros((0, 2))
    pid = np.zeros(0, dtype=np.int32)
    pos_out, pid_out = cull_outside_domain(domain, pos, pid)
    assert pos_out.shape == (0, 2)
    assert pid_out.shape == (0,)


# -----------------------------------------------------------------------------
# step_drift_and_cull
# -----------------------------------------------------------------------------

def test_step_drift_and_cull_removes_particles_that_leave() -> None:
    """A particle whose drift carries it past the boundary is gone after
    one step."""
    domain = WorldRect(width_km=200.0, height_km=200.0)  # half_width = 100
    plates = (
        # Fast east-moving plate.
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(60.0, 0.0)),
    )
    # Two particles: one near east edge, one in the centre.
    pos = np.array([[45.0, 0.0], [-30.0, 0.0]])
    pid = np.zeros(2, dtype=np.int32)
    ct = np.array([CRUST_CONTINENTAL] * 2, dtype=np.int8)
    th = np.array([35.0, 35.0])
    age = np.zeros(2)

    pos_out, pid_out, ct_out, th_out, age_out = step_drift_and_cull(
        domain, pos, pid, ct, th, age, plates, dt_myr=2.0,
    )
    # 45 + 60*2 = 165 → outside (>100). -30 + 120 = 90 → inside (≤100).
    assert pos_out.shape == (1, 2)
    np.testing.assert_array_equal(pos_out, [[90.0, 0.0]])
    np.testing.assert_array_equal(pid_out, [0])
    np.testing.assert_array_equal(th_out, [35.0])


def test_step_drift_and_cull_preserves_arrays_consistently(
    default_sim_config: SimConfig,
) -> None:
    """Running drift+cull on the seeded initial state leaves all parallel
    arrays the same length."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, pos, pid, ct, th, age = build_initial_state(
        domain, default_sim_config, seed=0,
    )
    pos_out, pid_out, ct_out, th_out, age_out = step_drift_and_cull(
        domain, pos, pid, ct, th, age, plates,
        dt_myr=default_sim_config.dt_myr,
    )
    n = pos_out.shape[0]
    assert pid_out.shape == (n,)
    assert ct_out.shape == (n,)
    assert th_out.shape == (n,)
    assert age_out.shape == (n,)


def test_step_drift_and_cull_determinism(
    default_sim_config: SimConfig,
) -> None:
    """Same inputs → same outputs, byte-identical."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates, pos, pid, ct, th, age = build_initial_state(
        domain, default_sim_config, seed=0,
    )
    a = step_drift_and_cull(
        domain, pos.copy(), pid.copy(), ct.copy(), th.copy(), age.copy(),
        plates, dt_myr=2.0,
    )
    b = step_drift_and_cull(
        domain, pos.copy(), pid.copy(), ct.copy(), th.copy(), age.copy(),
        plates, dt_myr=2.0,
    )
    for ax, bx in zip(a, b):
        np.testing.assert_array_equal(ax, bx)


# -----------------------------------------------------------------------------
# Wrap (toroidal) boundary
# -----------------------------------------------------------------------------

def test_wrap_positions_wraps_off_world_to_inside() -> None:
    """A particle that drifted past the east edge re-enters from the west."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    pos = np.array([
        [60.0, 0.0],         # 10 km past east edge → -40 km
        [-75.0, 80.0],       # 25 km past west, 30 km past north → +25, -20
        [0.0, 0.0],           # centre — unchanged
        [50.0, 50.0],        # exactly on corner — wraps to (-50, -50)
    ])
    out = wrap_positions(domain, pos)
    np.testing.assert_allclose(out[0], [-40.0, 0.0])
    np.testing.assert_allclose(out[1], [25.0, -20.0])
    np.testing.assert_allclose(out[2], [0.0, 0.0])
    np.testing.assert_allclose(out[3], [-50.0, -50.0])


def test_wrap_positions_empty_state_safe() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    out = wrap_positions(domain, np.zeros((0, 2)))
    assert out.shape == (0, 2)


def test_step_drift_and_apply_boundary_wrap_no_loss() -> None:
    """Under wrap mode, particle count is conserved no matter how fast
    plates drift."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(500.0, 200.0)),
    )
    pos = np.array([[0.0, 0.0], [50.0, -50.0], [-90.0, 90.0]])
    pid = np.zeros(3, dtype=np.int32)
    ct = np.array([0, 0, 0], dtype=np.int8)
    th = np.array([35.0, 35.0, 35.0])
    age = np.zeros(3)

    pos_out, *_ = step_drift_and_apply_boundary(
        domain, pos, pid, ct, th, age, plates, dt_myr=1.0,
        boundary_mode="wrap",
    )
    assert pos_out.shape == (3, 2)
    # Every output is inside the domain.
    assert (np.abs(pos_out[:, 0]) <= domain.half_width_km + 1e-9).all()
    assert (np.abs(pos_out[:, 1]) <= domain.half_height_km + 1e-9).all()


def test_step_drift_and_apply_boundary_open_drops_outside() -> None:
    """Under open mode, the previous cull behaviour is preserved."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(80.0, 0.0)),
    )
    pos = np.array([[40.0, 0.0], [-30.0, 0.0]])
    pid = np.zeros(2, dtype=np.int32)
    ct = np.array([0, 0], dtype=np.int8)
    th = np.array([35.0, 35.0])
    age = np.zeros(2)

    pos_out, *_ = step_drift_and_apply_boundary(
        domain, pos, pid, ct, th, age, plates, dt_myr=1.0,
        boundary_mode="open",
    )
    # 40 + 80 = 120 → outside (>50). -30 + 80 = 50 → just on the edge,
    # kept (inclusive).
    assert pos_out.shape == (1, 2)
    np.testing.assert_array_equal(pos_out, [[50.0, 0.0]])


def test_step_drift_and_apply_boundary_invalid_mode_raises() -> None:
    domain = WorldRect(width_km=100.0, height_km=100.0)
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(1.0, 0.0)),
    )
    pos = np.array([[0.0, 0.0]])
    pid = np.zeros(1, dtype=np.int32)
    ct = np.array([0], dtype=np.int8)
    th = np.array([35.0])
    age = np.zeros(1)
    with pytest.raises(ValueError, match="boundary_mode"):
        step_drift_and_apply_boundary(
            domain, pos, pid, ct, th, age, plates, dt_myr=1.0,
            boundary_mode="reflect",
        )


def test_drift_multi_tick_accumulates_correctly() -> None:
    """N drift steps of dt move a particle by N · v · dt."""
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(5.0, 0.0)),
    )
    pos = np.array([[0.0, 0.0]])
    pid = np.zeros(1, dtype=np.int32)
    for _ in range(10):
        pos = drift_positions(pos, pid, plates, dt_myr=1.0)
    assert pos[0, 0] == pytest.approx(50.0)
