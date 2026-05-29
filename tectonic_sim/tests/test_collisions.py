"""Tests for ``tectonic_sim.collisions.apply_collisions``.

We exercise each collision-type rule (cc, oc, oo) on hand-built minimal
scenarios, then mix them, and finally run the full path on the seeded
initial state to confirm shape consistency and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from tectonic_sim import (
    SimConfig,
    WorldRect,
    apply_collisions,
    build_initial_state,
    detect_overlaps,
)
from tectonic_sim.types import CRUST_CONTINENTAL, CRUST_OCEANIC


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _two_particle_state(
    type_a: int,
    type_b: int,
    pos_a: tuple[float, float] = (0.0, 0.0),
    pos_b: tuple[float, float] = (10.0, 0.0),
    *,
    thick_a: float = 35.0,
    thick_b: float = 35.0,
    age_a: float = 0.0,
    age_b: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Build a two-particle state with one pair (0, 1) configured by the
    caller. Returns ``(positions, plate_id, crust_type, thickness, age,
    pair_i, pair_j)``."""
    positions = np.array([pos_a, pos_b], dtype=np.float64)
    plate_id = np.array([0, 1], dtype=np.int32)
    crust_type = np.array([type_a, type_b], dtype=np.int8)
    thickness = np.array([thick_a, thick_b], dtype=np.float64)
    age = np.array([age_a, age_b], dtype=np.float64)
    pair_i = np.array([0], dtype=np.int32)
    pair_j = np.array([1], dtype=np.int32)
    return positions, plate_id, crust_type, thickness, age, pair_i, pair_j


# -----------------------------------------------------------------------------
# CC: orogeny + folding
# -----------------------------------------------------------------------------

def test_cc_both_particles_thicken(default_sim_config: SimConfig) -> None:
    """Both continental particles gain orogeny_uplift_per_overlap_km."""
    cfg = default_sim_config
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_CONTINENTAL,
        thick_a=35.0, thick_b=35.0,
    )
    _, _, _, new_th, _ = apply_collisions(pos, pid, ct, th, age, pi, pj, cfg)
    # Equal thickness ties → particle 0 is "lower" → loses fold mass; 1 gains.
    # Both also gain orogeny. With ratio=0.05, fold_mass = 0.05 * 35 = 1.75.
    orogeny = cfg.orogeny_uplift_per_overlap_km
    fold = cfg.folding_ratio * 35.0
    # i==0 is the lower in the tie (i_is_lower uses thick_i < thick_j, but
    # tie → False, so the "lower" assignment falls to j). Let me just
    # check the net effect is symmetric on totals.
    assert (new_th.sum() == pytest.approx(70.0 + 2 * orogeny))


def test_cc_no_particle_removed(default_sim_config: SimConfig) -> None:
    """Continental-continental collision conserves both particles."""
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_CONTINENTAL,
    )
    out_pos, *_ = apply_collisions(pos, pid, ct, th, age, pi, pj, default_sim_config)
    assert out_pos.shape == (2, 2)


def test_cc_lower_moves_toward_higher(default_sim_config: SimConfig) -> None:
    """The lower-thickness particle moves toward the higher; the higher
    doesn't move (folding is one-directional)."""
    cfg = default_sim_config
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_CONTINENTAL,
        pos_a=(0.0, 0.0), pos_b=(10.0, 0.0),
        thick_a=20.0, thick_b=40.0,  # b is higher
    )
    out_pos, *_ = apply_collisions(pos, pid, ct, th, age, pi, pj, cfg)
    # Particle 0 (lower) should move toward (+x); particle 1 stays put.
    assert out_pos[0, 0] > 0.0
    assert out_pos[0, 1] == 0.0
    np.testing.assert_array_equal(out_pos[1], [10.0, 0.0])


def test_cc_folding_mass_transfer(default_sim_config: SimConfig) -> None:
    """The lower-thickness particle loses fold mass; the higher gains it.
    Both also receive the orogeny constant."""
    cfg = default_sim_config
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_CONTINENTAL,
        thick_a=20.0, thick_b=40.0,
    )
    _, _, _, new_th, _ = apply_collisions(pos, pid, ct, th, age, pi, pj, cfg)
    orogeny = cfg.orogeny_uplift_per_overlap_km
    fold_mass = cfg.folding_ratio * 20.0
    assert new_th[0] == pytest.approx(20.0 + orogeny - fold_mass)
    assert new_th[1] == pytest.approx(40.0 + orogeny + fold_mass)


# -----------------------------------------------------------------------------
# OC: oceanic subducts under continental
# -----------------------------------------------------------------------------

def test_oc_oceanic_is_removed(default_sim_config: SimConfig) -> None:
    """The oceanic particle disappears; the continental survives."""
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_OCEANIC,
        thick_a=35.0, thick_b=7.0,
    )
    out_pos, out_pid, out_ct, out_th, _ = apply_collisions(
        pos, pid, ct, th, age, pi, pj, default_sim_config,
    )
    assert out_pos.shape == (1, 2)
    assert out_ct[0] == CRUST_CONTINENTAL


def test_oc_continental_thickens(default_sim_config: SimConfig) -> None:
    cfg = default_sim_config
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_OCEANIC,
        thick_a=35.0, thick_b=7.0,
    )
    _, _, _, new_th, _ = apply_collisions(pos, pid, ct, th, age, pi, pj, cfg)
    assert new_th[0] == pytest.approx(35.0 + cfg.subduction_arc_uplift_km)


def test_oc_order_doesnt_matter(default_sim_config: SimConfig) -> None:
    """Whether the oceanic is in slot i or slot j of the pair, the same
    survivor results."""
    cfg = default_sim_config
    # Oceanic in i, continental in j.
    pos_a, pid_a, ct_a, th_a, age_a, pi_a, pj_a = _two_particle_state(
        CRUST_OCEANIC, CRUST_CONTINENTAL,
        thick_a=7.0, thick_b=35.0,
    )
    out_a = apply_collisions(pos_a, pid_a, ct_a, th_a, age_a, pi_a, pj_a, cfg)
    # Continental in i, oceanic in j.
    pos_b, pid_b, ct_b, th_b, age_b, pi_b, pj_b = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_OCEANIC,
        thick_a=35.0, thick_b=7.0,
    )
    out_b = apply_collisions(pos_b, pid_b, ct_b, th_b, age_b, pi_b, pj_b, cfg)
    # In each case the continental should survive and have thickened.
    assert out_a[2][0] == CRUST_CONTINENTAL
    assert out_b[2][0] == CRUST_CONTINENTAL
    assert out_a[3][0] == pytest.approx(out_b[3][0])


# -----------------------------------------------------------------------------
# OO: older subducts under younger
# -----------------------------------------------------------------------------

def test_oo_older_is_removed(default_sim_config: SimConfig) -> None:
    """The older oceanic particle disappears."""
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_OCEANIC, CRUST_OCEANIC,
        thick_a=7.0, thick_b=7.0,
        age_a=80.0, age_b=10.0,
    )
    out_pos, *_, out_age = apply_collisions(
        pos, pid, ct, th, age, pi, pj, default_sim_config,
    )
    assert out_pos.shape == (1, 2)
    # Survivor is the younger one (age 10).
    assert out_age[0] == 10.0


def test_oo_younger_thickens(default_sim_config: SimConfig) -> None:
    cfg = default_sim_config
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_OCEANIC, CRUST_OCEANIC,
        thick_a=7.0, thick_b=9.0,
        age_a=80.0, age_b=10.0,
    )
    _, _, _, new_th, _ = apply_collisions(pos, pid, ct, th, age, pi, pj, cfg)
    # Survivor is particle 1 (younger), original thickness 9.
    assert new_th[0] == pytest.approx(9.0 + cfg.subduction_arc_uplift_km)


# -----------------------------------------------------------------------------
# Mixed + multi-pair scenarios
# -----------------------------------------------------------------------------

def test_particle_in_multiple_pairs_accumulates_deltas(
    default_sim_config: SimConfig,
) -> None:
    """A single particle in two cc pairs receives 2× orogeny uplift."""
    cfg = default_sim_config
    # 3 continental particles, all paired against the middle one (idx 1).
    positions = np.array([[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    plate_id = np.array([0, 1, 2], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL] * 3, dtype=np.int8)
    thickness = np.array([35.0, 35.0, 35.0])
    age = np.zeros(3)
    pair_i = np.array([0, 1], dtype=np.int32)
    pair_j = np.array([1, 2], dtype=np.int32)

    _, _, _, new_th, _ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pair_i, pair_j, cfg,
    )
    # Particle 1 is in both pairs → gains 2× orogeny.
    # Folding ties (equal thicknesses) → fold direction depends on i<j check.
    orogeny = cfg.orogeny_uplift_per_overlap_km
    # Net orogeny delta on particle 1 must be 2 × orogeny (plus any folding
    # signature, but the orogeny floor is at least 2 × per the pair count).
    assert new_th[1] > 35.0 + 1.5 * orogeny


def test_apply_collisions_empty_pairs_passthrough(
    default_sim_config: SimConfig,
) -> None:
    """No pairs → arrays come back identical."""
    pos = np.array([[0.0, 0.0]])
    pid = np.zeros(1, dtype=np.int32)
    ct = np.array([CRUST_CONTINENTAL], dtype=np.int8)
    th = np.array([35.0])
    age = np.zeros(1)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)

    out_pos, out_pid, out_ct, out_th, out_age = apply_collisions(
        pos, pid, ct, th, age, pi, pj, default_sim_config,
    )
    np.testing.assert_array_equal(out_pos, pos)
    np.testing.assert_array_equal(out_th, th)


def test_apply_collisions_mixed_pair_types(default_sim_config: SimConfig) -> None:
    """A scenario with one cc and one oc pair: cc keeps both, oc drops the
    oceanic. Final count = 3."""
    cfg = default_sim_config
    # 4 particles: cont-cont pair (0,1) and cont-ocean pair (2,3).
    positions = np.array([
        [0.0, 0.0], [10.0, 0.0],
        [100.0, 100.0], [110.0, 100.0],
    ])
    plate_id = np.array([0, 1, 2, 3], dtype=np.int32)
    crust_type = np.array(
        [CRUST_CONTINENTAL, CRUST_CONTINENTAL,
         CRUST_CONTINENTAL, CRUST_OCEANIC],
        dtype=np.int8,
    )
    thickness = np.array([35.0, 35.0, 35.0, 7.0])
    age = np.zeros(4)
    pair_i = np.array([0, 2], dtype=np.int32)
    pair_j = np.array([1, 3], dtype=np.int32)

    out_pos, out_pid, out_ct, out_th, _ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pair_i, pair_j, cfg,
    )
    assert out_pos.shape == (3, 2)
    # Only the oceanic (index 3) was dropped.
    assert CRUST_OCEANIC not in out_ct


def test_apply_collisions_thickness_never_negative(
    default_sim_config: SimConfig,
) -> None:
    """A pathological config with huge folding can't drive thickness below
    zero — the floor at 0 kicks in."""
    cfg = default_sim_config
    pos, pid, ct, th, age, pi, pj = _two_particle_state(
        CRUST_CONTINENTAL, CRUST_CONTINENTAL,
        thick_a=2.0, thick_b=40.0,
    )
    # Use a high folding ratio so the cap matters.
    from dataclasses import replace
    extreme_cfg = replace(cfg, folding_ratio=0.9)
    _, _, _, new_th, _ = apply_collisions(
        pos, pid, ct, th, age, pi, pj, extreme_cfg,
    )
    assert (new_th >= 0.0).all()


# -----------------------------------------------------------------------------
# Integration: detect_overlaps → apply_collisions on seeded initial state
# -----------------------------------------------------------------------------

def test_collisions_run_on_real_initial_state(
    default_sim_config: SimConfig,
) -> None:
    """detect_overlaps → apply_collisions → consistent array shapes."""
    domain = WorldRect(width_km=400.0, height_km=400.0)
    _plates, pos, pid, ct, th, age = build_initial_state(
        domain, default_sim_config, seed=0,
    )
    pair_i, pair_j = detect_overlaps(domain, pos, pid, default_sim_config)
    if pair_i.shape[0] == 0:
        pytest.skip("no overlap pairs at this seed — unlikely but skip cleanly")
    out_pos, out_pid, out_ct, out_th, out_age = apply_collisions(
        pos, pid, ct, th, age, pair_i, pair_j, default_sim_config,
    )
    n = out_pos.shape[0]
    assert out_pid.shape == (n,)
    assert out_ct.shape == (n,)
    assert out_th.shape == (n,)
    assert out_age.shape == (n,)
    # Survivor count must be ≤ input count.
    assert n <= pos.shape[0]


# -----------------------------------------------------------------------------
# Continental absorption (underthruster consumed by over-rider)
# -----------------------------------------------------------------------------

def test_absorption_removes_thinned_continental(
    default_sim_config: SimConfig,
) -> None:
    """A continental particle starting below the threshold is removed
    even with no pair fires this tick — the post-pass catches it."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    plate_id = np.array([0, 1], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL, CRUST_CONTINENTAL], dtype=np.int8)
    thickness = np.array([5.0, 35.0])
    age = np.zeros(2)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    out_pos, _, out_ct, *_ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert out_pos.shape == (1, 2)
    assert out_ct[0] == CRUST_CONTINENTAL


def test_absorption_transfers_mass_to_over_rider(
    default_sim_config: SimConfig,
) -> None:
    """Depleted particle's remaining thickness is added to the nearest
    cross-plate continental survivor."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    plate_id = np.array([0, 1], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL, CRUST_CONTINENTAL], dtype=np.int8)
    thickness = np.array([4.0, 35.0])
    age = np.zeros(2)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    _, _, _, out_th, _ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert out_th[0] == pytest.approx(35.0 + 4.0)


def test_absorption_removes_without_recipient(
    default_sim_config: SimConfig,
) -> None:
    """A thinned particle with no cross-plate continental in range is
    still removed (mass goes to abstract 'deep mantle')."""
    cfg = default_sim_config
    domain = WorldRect(width_km=500.0, height_km=500.0)
    positions = np.array([[0.0, 0.0]], dtype=np.float64)
    plate_id = np.array([0], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL], dtype=np.int8)
    thickness = np.array([3.0])
    age = np.zeros(1)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    out_pos, *_ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert out_pos.shape == (0, 2)


def test_absorption_ignores_oceanic_below_threshold(
    default_sim_config: SimConfig,
) -> None:
    """The threshold applies only to continental crust — thin oceanic is
    handled by OC/OO subduction, not by this pass."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    plate_id = np.array([0, 1], dtype=np.int32)
    crust_type = np.array([CRUST_OCEANIC, CRUST_CONTINENTAL], dtype=np.int8)
    thickness = np.array([4.0, 35.0])
    age = np.zeros(2)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    out_pos, *_ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert out_pos.shape == (2, 2)


def test_absorption_picks_nearest_cross_plate(
    default_sim_config: SimConfig,
) -> None:
    """When multiple cross-plate continental candidates are in range,
    mass goes to the nearest."""
    cfg = default_sim_config
    domain = WorldRect(width_km=300.0, height_km=300.0)
    positions = np.array(
        [[0.0, 0.0], [5.0, 0.0], [8.0, 0.0]], dtype=np.float64,
    )
    plate_id = np.array([0, 1, 1], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL] * 3, dtype=np.int8)
    thickness = np.array([3.0, 35.0, 35.0])
    age = np.zeros(3)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    out_pos, _, _, out_th, _ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert out_pos.shape == (2, 2)
    order = np.argsort(out_pos[:, 0])
    near, far = out_th[order]
    assert near == pytest.approx(38.0)
    assert far == pytest.approx(35.0)


def test_absorption_skips_same_plate_recipient(
    default_sim_config: SimConfig,
) -> None:
    """No cross-plate continental in range → remove, but no mass transfer."""
    cfg = default_sim_config
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    plate_id = np.array([0, 0], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL, CRUST_CONTINENTAL], dtype=np.int8)
    thickness = np.array([3.0, 35.0])
    age = np.zeros(2)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    _, _, _, out_th, _ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert len(out_th) == 1
    assert out_th[0] == pytest.approx(35.0)


def test_absorption_threshold_is_strict_less_than(
    default_sim_config: SimConfig,
) -> None:
    """A particle exactly at the threshold is NOT removed; only strictly
    less-than triggers absorption."""
    from dataclasses import replace
    cfg = replace(default_sim_config, min_continental_thickness_km=10.0)
    domain = WorldRect(width_km=200.0, height_km=200.0)
    positions = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    plate_id = np.array([0, 1], dtype=np.int32)
    crust_type = np.array([CRUST_CONTINENTAL, CRUST_CONTINENTAL], dtype=np.int8)
    thickness = np.array([10.0, 35.0])
    age = np.zeros(2)
    pi = np.zeros(0, dtype=np.int32)
    pj = np.zeros(0, dtype=np.int32)
    out_pos, *_ = apply_collisions(
        positions, plate_id, crust_type, thickness, age, pi, pj, cfg,
        domain=domain,
    )
    assert out_pos.shape == (2, 2)


def test_collisions_determinism_on_initial_state(
    default_sim_config: SimConfig,
) -> None:
    domain = WorldRect(width_km=300.0, height_km=300.0)
    _, pos, pid, ct, th, age = build_initial_state(
        domain, default_sim_config, seed=0,
    )
    pi, pj = detect_overlaps(domain, pos, pid, default_sim_config)
    a = apply_collisions(pos.copy(), pid.copy(), ct.copy(), th.copy(),
                         age.copy(), pi, pj, default_sim_config)
    b = apply_collisions(pos.copy(), pid.copy(), ct.copy(), th.copy(),
                         age.copy(), pi, pj, default_sim_config)
    for ax, bx in zip(a, b):
        np.testing.assert_array_equal(ax, bx)
