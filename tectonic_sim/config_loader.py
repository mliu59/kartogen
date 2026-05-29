"""Load a ``SimConfig`` from a TOML file or a pre-parsed table.

The module deliberately has *no* knowledge of where its config lives —
worldgen (or any other caller) is free to keep ``tectonic_sim.toml`` as a
standalone file or to inline the keys inside its own config. Two
entry points cover both cases:

  - ``load_sim_config_from_path(path)`` — read a TOML file, parse it.
  - ``load_sim_config(table)`` — parse a pre-loaded dict (the body of a
    TOML table the caller has already extracted).

Every field listed on ``SimConfig`` is required; missing keys raise at
parse time rather than defaulting silently downstream.
"""

from __future__ import annotations

from pathlib import Path

from tectonic_sim.types import SimConfig


_REQUIRED_KEYS: tuple[str, ...] = (
    "plate_count",
    "continental_fraction",
    "motion_speed_kmpy",
    "seed_radial_bias",
    "particle_spacing_km",
    "n_ticks",
    "dt_myr",
    "continental_thickness_km",
    "oceanic_thickness_km",
    "rift_thickness_km",
    "ridge_depth_km",
    "ridge_subsidence_rate",
    "max_ocean_depth_km",
    "continental_reference_thickness_km",
    "continental_isostasy_factor",
    "sea_level_km",
    # overlap_radius_km is *not* required — it's derived from
    # particle_spacing_km via SimConfig.overlap_radius_km.
    "orogeny_uplift_per_overlap_km",
    "folding_ratio",
    "folding_displacement_km",
    "subduction_arc_uplift_km",
    "min_continental_thickness_km",
    "contact_iterations",
    "velocity_damping_strength",
    "intra_plate_min_distance_factor",
    "erosion_period",
    "erosion_strength",
    "boundary_mode",
    "snapshot_period_ticks",
)

_VALID_BOUNDARY_MODES: frozenset[str] = frozenset({"open", "wrap"})


def load_sim_config_from_path(path: Path) -> SimConfig:
    """Read a TOML file and parse it into a ``SimConfig``."""
    import tomllib

    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return load_sim_config(raw)


def load_sim_config(table: dict[str, object]) -> SimConfig:
    """Parse a TOML table into a ``SimConfig``.

    Validates that every required key is present, that values have the
    expected types, and that ``boundary_mode`` is one of the supported
    modes ("open" only, for now).
    """
    missing = [k for k in _REQUIRED_KEYS if k not in table]
    if missing:
        raise KeyError(
            f"tectonic_sim config missing required keys: {sorted(missing)}"
        )

    boundary_mode = str(table["boundary_mode"])
    if boundary_mode not in _VALID_BOUNDARY_MODES:
        raise ValueError(
            f"tectonic_sim boundary_mode={boundary_mode!r} unsupported; "
            f"valid options: {sorted(_VALID_BOUNDARY_MODES)}"
        )

    return SimConfig(
        plate_count=int(table["plate_count"]),  # type: ignore[arg-type]
        continental_fraction=float(table["continental_fraction"]),  # type: ignore[arg-type]
        motion_speed_kmpy=float(table["motion_speed_kmpy"]),  # type: ignore[arg-type]
        seed_radial_bias=float(table["seed_radial_bias"]),  # type: ignore[arg-type]
        particle_spacing_km=float(table["particle_spacing_km"]),  # type: ignore[arg-type]
        n_ticks=int(table["n_ticks"]),  # type: ignore[arg-type]
        dt_myr=float(table["dt_myr"]),  # type: ignore[arg-type]
        continental_thickness_km=float(table["continental_thickness_km"]),  # type: ignore[arg-type]
        oceanic_thickness_km=float(table["oceanic_thickness_km"]),  # type: ignore[arg-type]
        rift_thickness_km=float(table["rift_thickness_km"]),  # type: ignore[arg-type]
        ridge_depth_km=float(table["ridge_depth_km"]),  # type: ignore[arg-type]
        ridge_subsidence_rate=float(table["ridge_subsidence_rate"]),  # type: ignore[arg-type]
        max_ocean_depth_km=float(table["max_ocean_depth_km"]),  # type: ignore[arg-type]
        continental_reference_thickness_km=float(
            table["continental_reference_thickness_km"]  # type: ignore[arg-type]
        ),
        continental_isostasy_factor=float(table["continental_isostasy_factor"]),  # type: ignore[arg-type]
        sea_level_km=float(table["sea_level_km"]),  # type: ignore[arg-type]
        orogeny_uplift_per_overlap_km=float(
            table["orogeny_uplift_per_overlap_km"]  # type: ignore[arg-type]
        ),
        folding_ratio=float(table["folding_ratio"]),  # type: ignore[arg-type]
        folding_displacement_km=float(table["folding_displacement_km"]),  # type: ignore[arg-type]
        subduction_arc_uplift_km=float(table["subduction_arc_uplift_km"]),  # type: ignore[arg-type]
        min_continental_thickness_km=float(table["min_continental_thickness_km"]),  # type: ignore[arg-type]
        contact_iterations=int(table["contact_iterations"]),  # type: ignore[arg-type]
        velocity_damping_strength=float(table["velocity_damping_strength"]),  # type: ignore[arg-type]
        intra_plate_min_distance_factor=float(
            table["intra_plate_min_distance_factor"]  # type: ignore[arg-type]
        ),
        erosion_period=int(table["erosion_period"]),  # type: ignore[arg-type]
        erosion_strength=float(table["erosion_strength"]),  # type: ignore[arg-type]
        boundary_mode=boundary_mode,
        snapshot_period_ticks=int(table["snapshot_period_ticks"]),  # type: ignore[arg-type]
    )
