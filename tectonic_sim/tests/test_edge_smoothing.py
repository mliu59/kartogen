"""Tests for the non-physics Perlin-modulated edge-smoothing pass.

Validates the smoothing operator itself (input shape preservation,
alpha-bound clamping, determinism, off-mode no-op, blur direction)
without spinning up the full polygon sim.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tectonic_sim import SimConfig, WorldRect
from tectonic_sim.polygon_sim.edge_smoothing import apply_edge_smoothing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_domain_grid(width_km: float = 400.0, height_km: float = 400.0,
                     cell_km: float = 10.0):
    domain = WorldRect(width_km=width_km, height_km=height_km)
    gy = int(round(height_km / cell_km))
    gx = int(round(width_km / cell_km))
    return domain, gy, gx, cell_km


def _step_thickness(gy: int, gx: int, lo: float = 7.0, hi: float = 50.0) -> np.ndarray:
    """Build a thickness array with a sharp vertical step at the
    horizontal midpoint — easy to verify a blur "smears" the step into
    a gradient. lo and hi are the two flat-region values in km.
    """
    arr = np.full((gy, gx), lo, dtype=np.float64)
    arr[:, gx // 2:] = hi
    return arr


def _uniform_owner(gy: int, gx: int) -> np.ndarray:
    """All cells in one plate → no boundaries → boundary boost field is
    identically zero. Useful for isolating Perlin-modulation tests from
    the boundary-boost mechanic.
    """
    return np.zeros((gy, gx), dtype=np.int64)


def _split_owner(gy: int, gx: int) -> np.ndarray:
    """Two plates split vertically at the midpoint — produces a clear
    boundary along ``x = gx // 2``. Useful for boundary-boost tests.
    """
    owner = np.zeros((gy, gx), dtype=np.int64)
    owner[:, gx // 2:] = 1
    return owner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shape_and_dtype_preserved(default_sim_config: SimConfig) -> None:
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    out, alpha = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        default_sim_config, domain, cell_km, rng_seed=42)
    assert out.shape == thick.shape
    assert alpha.shape == thick.shape
    assert out.dtype == np.float64
    assert alpha.dtype == np.float64


def test_off_mode_is_identity(default_sim_config: SimConfig) -> None:
    """alpha range zero AND no boundary boost → output identical to input
    AND alpha field all zeros. Uniform-owner array so no boundaries
    exist either.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.0,
                  edge_smoothing_alpha_max=0.0,
                  edge_smoothing_boundary_boost_peak=0.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    out, alpha = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    np.testing.assert_array_equal(out, thick)
    assert float(alpha.min()) == 0.0
    assert float(alpha.max()) == 0.0


def test_alpha_field_bounded(default_sim_config: SimConfig) -> None:
    """With uniform owner (no boundary boost), the alpha field is just
    the Perlin field, which must lie within ``[alpha_min, alpha_max]``.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.1,
                  edge_smoothing_alpha_max=0.6,
                  edge_smoothing_boundary_boost_peak=0.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    _, alpha = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=7)
    assert float(alpha.min()) == pytest.approx(0.1)
    assert float(alpha.max()) == pytest.approx(0.6)


def test_blur_smears_sharp_step(default_sim_config: SimConfig) -> None:
    """The blurred output must show intermediate values between the two
    flat regions on either side of the step. The original step has only
    two distinct values; after smoothing there should be many.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_apply_t0=True,
                  edge_smoothing_alpha_min=0.5,
                  edge_smoothing_alpha_max=1.0,
                  edge_smoothing_kernel_km=30.0,
                  edge_smoothing_boundary_boost_peak=0.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx, lo=7.0, hi=50.0)
    out, _ = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    row = out[gy // 2]
    n_distinct = len(np.unique(np.round(row, 3)))
    assert n_distinct > 10, (
        f"row has {n_distinct} distinct values; blur didn't smear the step")
    sigma_cells = int(cfg.edge_smoothing_kernel_km / cell_km)
    near = row[gx // 2 - sigma_cells : gx // 2 + sigma_cells]
    assert float(near.min()) > 7.0 + 1e-3
    assert float(near.max()) < 50.0 - 1e-3


def test_determinism_same_seed(default_sim_config: SimConfig) -> None:
    cfg = default_sim_config
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    owner = _uniform_owner(gy, gx)
    out_a, alpha_a = apply_edge_smoothing(
        thick, owner, cfg, domain, cell_km, rng_seed=42)
    out_b, alpha_b = apply_edge_smoothing(
        thick, owner, cfg, domain, cell_km, rng_seed=42)
    np.testing.assert_array_equal(out_a, out_b)
    np.testing.assert_array_equal(alpha_a, alpha_b)


def test_different_seeds_diverge(default_sim_config: SimConfig) -> None:
    cfg = default_sim_config
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    owner = _uniform_owner(gy, gx)
    out_a, alpha_a = apply_edge_smoothing(
        thick, owner, cfg, domain, cell_km, rng_seed=42)
    out_b, alpha_b = apply_edge_smoothing(
        thick, owner, cfg, domain, cell_km, rng_seed=123)
    assert not np.array_equal(alpha_a, alpha_b)
    assert not np.array_equal(out_a, out_b)


def test_input_not_mutated(default_sim_config: SimConfig) -> None:
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    snapshot = thick.copy()
    apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        default_sim_config, domain, cell_km, rng_seed=42)
    np.testing.assert_array_equal(thick, snapshot)


def test_uniform_alpha_uniform_blur(default_sim_config: SimConfig) -> None:
    """alpha_min == alpha_max AND no boundary boost → flat alpha field.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.5,
                  edge_smoothing_alpha_max=0.5,
                  edge_smoothing_boundary_boost_peak=0.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    _, alpha = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    np.testing.assert_allclose(alpha, 0.5)


def test_seam_wrap_in_blur(default_sim_config: SimConfig) -> None:
    """The Gaussian filter uses wrap mode — a cell at column 0 should be
    influenced by column gx-1.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=1.0,
                  edge_smoothing_alpha_max=1.0,
                  edge_smoothing_kernel_km=20.0,
                  edge_smoothing_boundary_boost_peak=0.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = np.full((gy, gx), 10.0, dtype=np.float64)
    thick[:, 0] = 100.0
    out, _ = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    far_right = float(out[gy // 2, gx - 1])
    centre = float(out[gy // 2, gx // 2])
    source = float(out[gy // 2, 0])
    assert source > far_right > centre, (
        f"wrap blur failed: source={source:.2f} > "
        f"far_right={far_right:.2f} > centre={centre:.2f} expected; "
        f"got {source:.2f}, {far_right:.2f}, {centre:.2f}")


# ---------------------------------------------------------------------------
# Boundary-boost tests
# ---------------------------------------------------------------------------


def test_boundary_boost_disabled_when_peak_zero(
    default_sim_config: SimConfig,
) -> None:
    """``boundary_boost_peak = 0`` AND ``alpha_min = alpha_max = 0`` →
    output identical to input even with a split-owner grid.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.0,
                  edge_smoothing_alpha_max=0.0,
                  edge_smoothing_boundary_boost_peak=0.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    out, alpha = apply_edge_smoothing(
        thick, _split_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    np.testing.assert_array_equal(out, thick)
    np.testing.assert_array_equal(alpha, np.zeros_like(alpha))


def test_boundary_boost_active_at_boundary_and_decays_inland(
    default_sim_config: SimConfig,
) -> None:
    """Disable Perlin (alpha_min = alpha_max = 0), keep only the boundary
    boost. The alpha field should peak at the boundary cells and decay
    exponentially with distance.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.0,
                  edge_smoothing_alpha_max=0.0,
                  edge_smoothing_boundary_boost_peak=0.6,
                  edge_smoothing_boundary_falloff_km=20.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    _, alpha = apply_edge_smoothing(
        thick, _split_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    # Cells right at the suture (column gx//2 and gx//2 - 1) sit at distance
    # 0 from the boundary → α = peak * exp(0) = peak.
    boundary_alpha = float(alpha[gy // 2, gx // 2])
    assert boundary_alpha == pytest.approx(0.6, abs=1e-9)
    # Cell one falloff length away should drop to peak / e ≈ 0.22.
    # The split is at gx//2, so a cell ``falloff_km / cell_km`` cells
    # inland (column gx//2 - sigma) sits at d ≈ falloff_km.
    sigma_cells = int(round(cfg.edge_smoothing_boundary_falloff_km / cell_km))
    one_falloff = float(alpha[gy // 2, gx // 2 - 1 - sigma_cells])
    assert one_falloff == pytest.approx(0.6 / np.e, rel=0.20)
    # A cell far from BOTH boundaries should be near zero. On a torus
    # split at column gx//2, there are TWO boundaries: one at gx//2 and
    # one at the wrap point (column 0 ↔ gx-1). The interior midpoint of
    # plate 0 is column gx//4. At cell_km=10, falloff=20 km, that's
    # ~10 cells = 100 km from the suture → α ≈ 0.6 * exp(-5) ≈ 0.004.
    far = float(alpha[gy // 2, gx // 4])
    assert far < 0.05, f"far-from-boundary alpha {far:.3f} should be near zero"


def test_boundary_boost_clipped_at_one(
    default_sim_config: SimConfig,
) -> None:
    """``α_perlin + boost`` is clipped to [0, 1]. With max Perlin (1.0)
    and a non-zero boost, the resulting alpha never exceeds 1.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.9,
                  edge_smoothing_alpha_max=0.9,
                  edge_smoothing_boundary_boost_peak=0.4,
                  edge_smoothing_boundary_falloff_km=20.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    _, alpha = apply_edge_smoothing(
        thick, _split_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    assert float(alpha.max()) <= 1.0 + 1e-12
    # Boundary cells should be exactly clipped to 1.0 (0.9 + 0.4 = 1.3
    # before clip).
    assert float(alpha[gy // 2, gx // 2]) == pytest.approx(1.0)


def test_boundary_boost_no_effect_with_single_plate(
    default_sim_config: SimConfig,
) -> None:
    """With a single plate everywhere there are no boundaries, so the
    boost field is identically zero — even when peak > 0.
    """
    cfg = replace(default_sim_config,
                  edge_smoothing_alpha_min=0.0,
                  edge_smoothing_alpha_max=0.0,
                  edge_smoothing_boundary_boost_peak=0.6,
                  edge_smoothing_boundary_falloff_km=20.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx)
    out, alpha = apply_edge_smoothing(
        thick, _uniform_owner(gy, gx),
        cfg, domain, cell_km, rng_seed=42)
    np.testing.assert_array_equal(out, thick)
    np.testing.assert_array_equal(alpha, np.zeros_like(alpha))


def test_boundary_boost_makes_seam_smoother(
    default_sim_config: SimConfig,
) -> None:
    """Compare two runs with the same Perlin α range but different
    boundary boosts. The high-boost run must have a smaller maximum
    thickness gradient across the suture.
    """
    base = replace(default_sim_config,
                   edge_smoothing_alpha_min=0.0,
                   edge_smoothing_alpha_max=0.0,
                   edge_smoothing_kernel_km=40.0)
    cfg_off = replace(base, edge_smoothing_boundary_boost_peak=0.0)
    cfg_on  = replace(base,
                      edge_smoothing_boundary_boost_peak=0.9,
                      edge_smoothing_boundary_falloff_km=40.0)
    domain, gy, gx, cell_km = _make_domain_grid()
    thick = _step_thickness(gy, gx, lo=7.0, hi=50.0)
    owner = _split_owner(gy, gx)
    out_off, _ = apply_edge_smoothing(
        thick, owner, cfg_off, domain, cell_km, rng_seed=42)
    out_on, _ = apply_edge_smoothing(
        thick, owner, cfg_on, domain, cell_km, rng_seed=42)
    # Gradient across the seam should be smaller in the boost-on run.
    row_off = out_off[gy // 2]
    row_on = out_on[gy // 2]
    grad_off = float(np.abs(np.diff(row_off)).max())
    grad_on = float(np.abs(np.diff(row_on)).max())
    assert grad_on < grad_off, (
        f"boundary boost did not flatten the seam: "
        f"grad_off={grad_off:.2f} vs grad_on={grad_on:.2f}")
