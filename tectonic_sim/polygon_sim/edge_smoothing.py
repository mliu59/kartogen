"""Non-physics edge smoothing — Perlin-weighted Gaussian blur of crust
thickness, with an optional plate-boundary boost.

This pass softens sharp thickness boundaries (and the rendered
topography derived from them via isostasy) without modelling any
geological process. It exists alongside ``aging.py`` (where physics-
grounded erosion lives) but is deliberately kept separate: this is a
purely algorithmic UI/visualisation concern, controlled by a Perlin
alpha field so some edges stay sharp while others smear.

The pass runs at exactly two sim instants — t=0 (after seeding) and
t=final (after the last tick) — never per-tick. Both passes use
independent Perlin draws (separate RNG tags) so the t=final pass
smooths the *evolved* sim rather than reapplying the t=0 pattern.

Mechanics per pass:

  1. ``blur(x, y) = gaussian_filter(thickness, sigma = kernel_km / cell_km)``.
     Gaussian is applied with wrap mode (mode='wrap') because the sim
     domain is toroidal — a blur at the seam reads neighbours from the
     opposite edge.
  2. ``α_perlin(x, y) = perlin_fbm(x, y)`` normalised from its observed
     min/max into ``[alpha_min, alpha_max]``. (Normalising on observed
     bounds keeps the modulation predictable for any wavelength /
     octaves combo.)
  3. ``boost(x, y) = peak * exp(-d_km(x, y) / falloff_km)`` where
     ``d_km`` is the toroidal Euclidean distance from each cell to its
     nearest plate-boundary cell (a cell adjacent to a different plate
     id). ``peak = 0`` disables the boost.
  4. ``α(x, y) = clip(α_perlin + boost, 0, 1)``.
  5. ``thickness(x, y) = (1 - α) * thickness + α * blur``.

Returns the blended thickness AND the final α field (Perlin + boost,
clipped) so visualisation / analysis tools can show where smoothing
was concentrated.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from tectonic_sim.noise import PerlinNoise2D, fbm_grid
from tectonic_sim.types import SimConfig, WorldRect


def _build_alpha_field(
    sim_config: SimConfig,
    domain: WorldRect,
    gy: int,
    gx: int,
    cell_km: float,
    rng_seed: int,
) -> np.ndarray:
    """Generate a Perlin alpha field on the cell grid, normalised to
    ``[alpha_min, alpha_max]``. Same shape as the thickness array.
    """
    alpha_min = float(sim_config.edge_smoothing_alpha_min)
    alpha_max = float(sim_config.edge_smoothing_alpha_max)

    # Cell-centre coordinates in km in the sim's centred frame.
    half_w = domain.half_width_km
    half_h = domain.half_height_km
    xs = (np.arange(gx) + 0.5) * cell_km - half_w
    ys = (np.arange(gy) + 0.5) * cell_km - half_h
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")

    noise = PerlinNoise2D.from_rng(np.random.Generator(np.random.PCG64(rng_seed)))
    raw = fbm_grid(
        noise, x_grid, y_grid,
        octaves=int(sim_config.edge_smoothing_noise_octaves),
        persistence=float(sim_config.edge_smoothing_noise_persistence),
        base_frequency=1.0 / float(sim_config.edge_smoothing_noise_wavelength_km),
    )

    # Normalise observed range to [0, 1] then map to [alpha_min, alpha_max].
    raw_lo = float(raw.min())
    raw_hi = float(raw.max())
    span = max(raw_hi - raw_lo, 1e-9)
    norm = (raw - raw_lo) / span
    return alpha_min + (alpha_max - alpha_min) * norm


def _build_boundary_boost(
    owner: np.ndarray,
    cell_km: float,
    peak: float,
    falloff_km: float,
) -> np.ndarray:
    """Compute an exponential boundary-boost field over the cell grid.

    A cell is a "plate-boundary cell" if any of its 4-neighbours has a
    different ``owner`` id. We compute the **toroidal** distance (km)
    from each cell to its nearest boundary cell, then return
    ``peak * exp(-d_km / falloff_km)``.

    The result is the EXTRA alpha to add on top of the Perlin field.
    Returned shape matches ``owner``.

    Returns a zeros array when ``peak <= 0`` (the cheap short-circuit).
    """
    if peak <= 0.0:
        return np.zeros_like(owner, dtype=np.float64)

    # Boundary mask: a cell is on the boundary if any 4-neighbour has a
    # different owner. np.roll wraps, which is correct for a torus.
    boundary = (
        (owner != np.roll(owner,  1, axis=0))
        | (owner != np.roll(owner, -1, axis=0))
        | (owner != np.roll(owner,  1, axis=1))
        | (owner != np.roll(owner, -1, axis=1))
    )
    if not boundary.any():
        # Single-plate world (or all-unowned) — no boundaries to boost
        # against. Return zeros.
        return np.zeros_like(owner, dtype=np.float64)

    # Toroidal distance transform: scipy's distance_transform_edt is
    # not torus-aware, so tile the boundary mask 3x3, run the transform,
    # crop the centre block. Distance from any cell to its nearest
    # boundary then reads through the wrap correctly. Cost is 9x a
    # single distance_transform_edt — negligible at sim sizes (sub-ms).
    gy, gx = owner.shape
    tiled = np.tile(~boundary, (3, 3))
    d_cells = distance_transform_edt(tiled)  # type: ignore[no-untyped-call]
    d_cells_centre = d_cells[gy:2 * gy, gx:2 * gx]
    d_km = d_cells_centre * cell_km

    return peak * np.exp(-d_km / max(falloff_km, 1e-9))


def apply_edge_smoothing(
    thickness: np.ndarray,
    owner: np.ndarray,
    sim_config: SimConfig,
    domain: WorldRect,
    cell_km: float,
    *,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Perlin-modulated Gaussian smoothing pass to ``thickness``.

    Returns ``(smoothed_thickness, alpha_field)``. Both arrays have the
    same shape as the input. The input is not mutated.

    ``owner`` (int) shares ``thickness.shape`` and carries the plate id
    at each cell. It feeds the optional plate-boundary boost — see the
    module docstring for the formula.

    ``rng_seed`` is the *already-tagged* seed for the Perlin draw —
    callers compute it as ``seed ^ _EDGE_SMOOTHING_RNG_TAG ^ pass_tag``
    where ``pass_tag`` distinguishes t=0 from t=final (so the two passes
    use independent noise fields).

    The alpha field returned is the FINAL clipped field — Perlin α plus
    the boundary boost, clipped to ``[0, 1]`` — so downstream
    visualisations see exactly the per-cell smoothing strength that
    was applied.

    A no-op short-circuit fires when there is no smoothing to perform
    at all (alpha range is zero AND no boundary boost). The alpha field
    is still returned (all zeros) for callers that always want it.
    """
    gy, gx = thickness.shape
    alpha_min = float(sim_config.edge_smoothing_alpha_min)
    alpha_max = float(sim_config.edge_smoothing_alpha_max)
    boost_peak = float(sim_config.edge_smoothing_boundary_boost_peak)
    falloff_km = float(sim_config.edge_smoothing_boundary_falloff_km)
    kernel_km = float(sim_config.edge_smoothing_kernel_km)

    # Build the Perlin alpha field unconditionally.
    perlin_alpha = _build_alpha_field(
        sim_config, domain, gy, gx, cell_km, rng_seed,
    )

    # Boundary boost: extra alpha that decays exponentially inland from
    # plate-boundary cells. Skipped (zeros) when peak <= 0.
    boost = _build_boundary_boost(owner, cell_km, boost_peak, falloff_km)

    # Final alpha = clip(perlin + boost, 0, 1). Clipping is essential so
    # the blend coefficient stays a valid linear-interpolation weight.
    alpha = np.clip(perlin_alpha + boost, 0.0, 1.0)

    # Short-circuit: nothing to blend in. Both contributions zero
    # everywhere → output identical to input.
    if alpha_min == 0.0 and alpha_max == 0.0 and boost_peak <= 0.0:
        return thickness.copy(), alpha

    # Gaussian blur in cell units. mode='wrap' respects the toroidal
    # domain — a cell near the seam blurs against the opposite edge.
    sigma_cells = max(1e-3, kernel_km / max(cell_km, 1e-6))
    blurred = gaussian_filter(
        thickness.astype(np.float64, copy=False),
        sigma=sigma_cells, mode="wrap",
    )

    # Per-cell linear blend.
    return (1.0 - alpha) * thickness + alpha * blurred, alpha
