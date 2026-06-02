"""Tests for the tectonic drift GIF emission.

The polygon sim captures per-tick drift / thickness / topography
animations directly (pre-rendered PIL frames). ``export_world`` writes
those to ``<export>/tectonic_sim_views/*.gif`` when
``snapshot_period_ticks > 0`` and omits them otherwise.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from worldgen import export_world
from worldgen.types import WorldgenConfig, WorldShape

pytestmark = pytest.mark.slow  # full generate()/sim per test
def _cfg_with_snapshots(
    base: WorldgenConfig, period: int, side_km: float = 100.0,
) -> WorldgenConfig:
    """Shrink the world to a ``side_km × side_km`` square for fast drift
    tests and apply the requested snapshot period."""
    return replace(
        base,
        world=WorldShape(width_km=side_km, height_km=side_km),
        tectonics=replace(base.tectonics, snapshot_period_ticks=period),
    )


def test_export_world_writes_drift_gif(
    default_worldgen_config: WorldgenConfig,
    tmp_path: Path,
) -> None:
    """When snapshots are captured, export_world emits drift.gif inside
    ``tectonic_sim_views/`` alongside the static layer PNGs."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=25)
    folder = export_world(
        config=cfg, seed=42, output_root=tmp_path,
    )
    drift = folder / "tectonic_sim_views" / "drift.gif"
    assert drift.exists()
    assert drift.stat().st_size > 0


def test_export_world_skips_drift_when_disabled(
    default_worldgen_config: WorldgenConfig,
    tmp_path: Path,
) -> None:
    """No snapshots → no drift.gif emitted."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=0)
    folder = export_world(
        config=cfg, seed=42, output_root=tmp_path,
    )
    assert not (folder / "tectonic_sim_views" / "drift.gif").exists()
    assert not (folder / "drift.gif").exists()
