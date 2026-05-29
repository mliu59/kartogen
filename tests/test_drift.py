"""Tests for the tectonic drift snapshot capture + GIF rendering."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from worldgen import export_world, generate
from worldgen.types import WorldgenConfig, WorldShape


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


def test_history_length_matches_snapshot_period(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """For n_ticks ticks captured every k, history length = n_ticks // k + 1
    (the +1 accounts for the explicit t=0 frame)."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=10)
    n_ticks = cfg.tectonics.n_ticks
    world = generate(config=cfg, seed=42)
    expected = n_ticks // 10 + 1
    assert len(world.lithosphere.history) == expected


def test_history_empty_when_period_zero(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Disable capture entirely with period=0; history must be empty."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=0)
    world = generate(config=cfg, seed=42)
    assert world.lithosphere.history == ()


def test_history_is_deterministic(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """Same (seed, config) → same captured history frames."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=20)
    a = generate(config=cfg, seed=42)
    b = generate(config=cfg, seed=42)
    assert len(a.lithosphere.history) == len(b.lithosphere.history)
    for fa, fb in zip(a.lithosphere.history, b.lithosphere.history):
        assert fa.tick == fb.tick
        assert fa.time_myr == fb.time_myr
        assert fa.plate_id == fb.plate_id
        assert fa.crust_type == fb.crust_type
        assert fa.plate_centers_km == fb.plate_centers_km


def test_first_frame_is_t0(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """The first captured frame is at tick 0, time = 0 Myr."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=25)
    world = generate(config=cfg, seed=42)
    assert world.lithosphere.history[0].tick == 0
    assert world.lithosphere.history[0].time_myr == 0.0


def test_last_frame_advanced(
    default_worldgen_config: WorldgenConfig,
) -> None:
    """The last captured frame is at tick = n_ticks (the simulation's end)."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=20)
    world = generate(config=cfg, seed=42)
    n_ticks = cfg.tectonics.n_ticks
    dt = cfg.tectonics.dt_myr
    last = world.lithosphere.history[-1]
    assert last.tick == n_ticks
    assert last.time_myr == n_ticks * dt


def test_export_world_writes_drift_gif(
    default_worldgen_config: WorldgenConfig,
    tmp_path: Path,
) -> None:
    """When snapshots are captured, export_world emits drift.gif alongside
    the static layer PNGs."""
    cfg = _cfg_with_snapshots(default_worldgen_config, period=25)
    folder = export_world(
        config=cfg, seed=42, output_root=tmp_path,
    )
    drift = folder / "drift.gif"
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
    assert not (folder / "drift.gif").exists()
