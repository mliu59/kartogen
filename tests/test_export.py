"""Tests for the export endpoints: serialize_world, save/load_snapshot,
export_world."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kartogen import (
    GeneratedWorld,
    KartogenConfig,
    WorldSnapshot,
    export_world,
    load_snapshot,
    save_snapshot,
    serialize_world,
)

pytestmark = pytest.mark.slow  # full generate()/sim per test
def test_serialize_returns_generic_container(small_world: GeneratedWorld) -> None:
    """All WorldSnapshot fields are open-ended dicts / list of dicts so new
    per-hex fields and new layers can be added without changing the container.
    """
    snap = serialize_world(small_world)

    assert isinstance(snap, WorldSnapshot)
    assert isinstance(snap.metadata, dict)
    assert isinstance(snap.hexes, list)
    assert isinstance(snap.layers, dict)
    assert all(isinstance(rec, dict) for rec in snap.hexes)
    assert all(isinstance(v, dict) for v in snap.layers.values())


def test_serialize_hex_records_include_coords_and_hexdata_fields(
    small_world: GeneratedWorld,
) -> None:
    snap = serialize_world(small_world)
    sample = snap.hexes[0]

    # Coordinates are stamped on every record.
    assert {"q", "r"}.issubset(sample.keys())

    # Every HexData field is present and is a primitive (no nested dataclass).
    expected = {
        "elevation", "is_ocean", "is_coast", "is_lake", "is_river",
        "temperature_c", "precipitation_mm", "flow_accumulation", "biome",
        "plate_id",
    }
    assert expected.issubset(sample.keys())


def test_serialize_metadata_carries_run_params(small_world: GeneratedWorld) -> None:
    snap = serialize_world(small_world)
    assert snap.metadata["world_width_km"] == small_world.config.world.width_km
    assert snap.metadata["world_height_km"] == small_world.config.world.height_km
    assert snap.metadata["hex_count"] == len(small_world.hexes)
    assert snap.metadata["hex_size_km"] == small_world.config.hex_size_km
    assert "schema_version" in snap.metadata


def test_serialize_layers_carry_sea_level(
    small_world: GeneratedWorld,
) -> None:
    snap = serialize_world(small_world)
    assert snap.layers["elevation"]["sea_level"] == small_world.elevation.sea_level


def test_serialize_is_pure(small_world: GeneratedWorld) -> None:
    """No wall-clock timestamp / seed is injected by ``serialize_world`` —
    two calls on the same world produce equal snapshots."""
    a = serialize_world(small_world)
    b = serialize_world(small_world)
    assert a.to_dict() == b.to_dict()


def test_snapshot_roundtrip_via_disk(
    small_world: GeneratedWorld, tmp_path: Path,
) -> None:
    snap = serialize_world(small_world)
    save_snapshot(snap, tmp_path / "snap.json")
    loaded = load_snapshot(tmp_path / "snap.json")
    assert loaded.to_dict() == snap.to_dict()


def test_export_world_creates_folder_with_snapshot_and_layer_pngs(
    small_world_config: KartogenConfig, tmp_path: Path,
) -> None:
    folder = export_world(
        config=small_world_config,
        seed=42,
        output_root=tmp_path,
    )

    assert folder.exists() and folder.is_dir()
    assert folder.parent == tmp_path
    w_km = int(round(small_world_config.world.width_km))
    h_km = int(round(small_world_config.world.height_km))
    assert folder.name.startswith(f"seed42_{w_km}x{h_km}km_")

    # Snapshot file is valid JSON.
    snapshot_path = folder / "snapshot.json"
    assert snapshot_path.exists()
    parsed = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert {"metadata", "hexes", "layers"} == set(parsed.keys())
    assert parsed["metadata"]["seed"] == 42
    assert "timestamp" in parsed["metadata"]

    # One PNG per layer.
    layers_dir = folder / "layers"
    assert layers_dir.is_dir()
    expected_layers = {
        "elevation.png", "temperature.png", "precipitation.png",
        "flow.png", "biome.png", "composite.png",
    }
    assert expected_layers.issubset({p.name for p in layers_dir.iterdir()})
