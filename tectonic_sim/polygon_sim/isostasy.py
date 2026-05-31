"""Per-cell elevation derived from crust thickness + age.

Two physical models, one per crust type:

  - **Continental:** isostatic equilibrium against a mantle column::

          elevation_km = (thickness − reference_thickness) × isostasy_factor

  - **Oceanic:** half-space cooling — ocean floor sinks with √(age)::

          depth_km = ridge_depth + ridge_subsidence_rate × √age,
                     capped at max_ocean_depth_km

"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, SimConfig


def particle_elevation_km(
    crust_type: np.ndarray,
    thickness_km: np.ndarray,
    age_myr: np.ndarray,
    sim_config: SimConfig) -> np.ndarray:
    """Compute signed elevation (km) for each input cell.

    Negative = below sea level (ocean floor), zero ≈ sea level, positive
    = above sea level. All inputs are array-shape-agnostic — flat ``(N)``,
    2D ``(gy, gx)``, or anything else as long as shapes match.
    """
    out = np.zeros_like(thickness_km, dtype=np.float64)
    if thickness_km.size == 0:
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
