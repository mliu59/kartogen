"""Per-particle elevation derived from crust thickness + age.

Two physical models, one per crust type:

  - **Continental:** isostatic equilibrium against a mantle column. The
    elevation above the reference thickness is linear in the excess::

          elevation_km = (thickness − reference_thickness) × isostasy_factor

    Continental thickening produces elevation gain — this is what
    visualises mountain belts at the boundaries between converging
    plates.

  - **Oceanic:** half-space cooling. The ocean floor sinks with the
    square root of age as the lithosphere cools and contracts::

          depth_km = ridge_depth + ridge_subsidence_rate × √age,
                     capped at max_ocean_depth_km

This is a small piece of what Phase 7 (full aging + erosion + isostasy)
will own; we land it now because the visualisation in Phase 5/6 needs
to colour particles by elevation to make the orogeny effect visible.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, SimConfig


def particle_elevation_km(
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    age_myr: np.ndarray,
    sim_config: SimConfig,
) -> np.ndarray:
    """Compute signed elevation (km) per particle.

    Negative = below sea level (ocean floor), zero ≈ sea level, positive
    = above sea level (land / mountains). All inputs are ``(N,)`` arrays
    parallel to the simulation state.
    """
    n = thickness_km.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    cont_mask = (crust_type == CRUST_CONTINENTAL)
    ocn_mask = ~cont_mask

    # Continental: isostatic equilibrium.
    if cont_mask.any():
        excess = (
            thickness_km[cont_mask]
            - sim_config.continental_reference_thickness_km
        )
        out[cont_mask] = excess * sim_config.continental_isostasy_factor

    # Oceanic: half-space cooling depth (signed negative).
    if ocn_mask.any():
        age_clamped = np.maximum(age_myr[ocn_mask], 0.0)
        depth = (
            sim_config.ridge_depth_km
            + sim_config.ridge_subsidence_rate * np.sqrt(age_clamped)
        )
        depth = np.minimum(depth, sim_config.max_ocean_depth_km)
        out[ocn_mask] = -depth

    return out
