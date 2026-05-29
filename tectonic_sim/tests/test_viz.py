"""Tests for ``tectonic_sim.viz``.

We don't pixel-compare against fixtures — that's too brittle for a
single-developer codebase. Instead we check structural properties:
output dimensions, non-empty pixels in the right regions, distinct plate
colours, palette consistency, deterministic output from determinist
input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tectonic_sim import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    Plate,
    PLATE_PALETTE,
    SimConfig,
    WorldRect,
    build_initial_state,
    render_initial_state,
    render_particles_png,
    render_single_plate_png,
    render_voronoi_png,
)


def _toy_state() -> tuple[WorldRect, np.ndarray, np.ndarray, np.ndarray, tuple[Plate, ...]]:
    """A small hand-built state with 3 plates and 30 particles."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    rng = np.random.Generator(np.random.PCG64(0))
    positions = rng.uniform(-90.0, 90.0, size=(30, 2))
    plate_id = (positions[:, 0] // 60 + 1).astype(np.int32).clip(0, 2)
    crust_type = np.where(
        plate_id == 1, CRUST_OCEANIC, CRUST_CONTINENTAL,
    ).astype(np.int8)
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(-80.0, 0.0), velocity_kmpy=(10.0, 0.0)),
        Plate(id=1, type="oceanic",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(0.0, 10.0)),
        Plate(id=2, type="continental",
              seed_position_km=(80.0, 0.0), velocity_kmpy=(-5.0, -5.0)),
    )
    return domain, positions, plate_id, crust_type, plates


# -----------------------------------------------------------------------------
# render_particles_png
# -----------------------------------------------------------------------------

def test_render_particles_returns_image_of_expected_size() -> None:
    domain, pos, pid, ct, plates = _toy_state()
    img = render_particles_png(
        domain, pos, pid, plates, crust_type=ct, px_per_km=2.0,
    )
    # Domain 200×200 km @ 2 px/km plus margins + caption band.
    assert img.size[0] >= 400
    assert img.size[1] >= 400 + 22  # caption band
    assert img.mode == "RGB"


def test_render_particles_uses_plate_colors() -> None:
    """Plate-coloured particles dominate the central region — i.e. the
    image is not all background colour."""
    domain, pos, pid, ct, plates = _toy_state()
    img = render_particles_png(
        domain, pos, pid, plates, crust_type=ct, px_per_km=4.0,
    )
    arr = np.array(img)
    # Strip caption band.
    arr = arr[22:]
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
    # At least: background + border + 3 plate palette colours.
    assert unique_colors.shape[0] >= 4


def test_render_particles_handles_no_crust_type() -> None:
    """Passing ``crust_type=None`` falls back to the continental colour."""
    domain, pos, pid, _ct, plates = _toy_state()
    img = render_particles_png(
        domain, pos, pid, plates, crust_type=None, px_per_km=2.0,
    )
    assert img.size[0] > 0


def test_render_particles_deterministic() -> None:
    """Same inputs → byte-identical PNG bytes."""
    domain, pos, pid, ct, plates = _toy_state()
    img1 = render_particles_png(
        domain, pos, pid, plates, crust_type=ct, px_per_km=1.5,
    )
    img2 = render_particles_png(
        domain, pos, pid, plates, crust_type=ct, px_per_km=1.5,
    )
    assert np.array_equal(np.array(img1), np.array(img2))


def test_render_particles_velocity_arrows_off_doesnt_crash() -> None:
    domain, pos, pid, ct, plates = _toy_state()
    img = render_particles_png(
        domain, pos, pid, plates, crust_type=ct, show_velocities=False,
    )
    assert img.size[0] > 0


# -----------------------------------------------------------------------------
# render_voronoi_png
# -----------------------------------------------------------------------------

def test_render_voronoi_returns_image() -> None:
    domain, pos, pid, ct, plates = _toy_state()
    img = render_voronoi_png(
        domain, pos, pid, plates, crust_type=ct, px_per_km=0.6,
    )
    assert img.size[0] > 0
    assert img.size[1] > 0
    assert img.mode == "RGB"


def test_render_voronoi_pixels_match_nearest_particle() -> None:
    """Voronoi field at a sample point should be the colour of the
    nearest particle's plate. We sample interior pixels (away from the
    border / caption band) and check the nearest-particle invariant."""
    domain, pos, pid, _ct, plates = _toy_state()
    img = render_voronoi_png(
        domain, pos, pid, plates, crust_type=None, px_per_km=2.0,
    )
    arr = np.array(img)
    # The image has a caption band + margin. Pick an interior pixel.
    h, w, _ = arr.shape
    sample_pixel = (h // 2, w // 2)
    sample_color = tuple(arr[sample_pixel])

    # Predict: at canvas centre, the corresponding km coordinate is (0, 0).
    # Find the nearest particle to (0, 0).
    d2 = (pos[:, 0]) ** 2 + (pos[:, 1]) ** 2
    nearest_idx = int(np.argmin(d2))
    expected_plate = int(pid[nearest_idx])

    # The colour at the centre should match the plate's palette colour
    # (modulo the small hue shift _plate_color applies). We compare
    # against the palette family rather than an exact RGB.
    pal = PLATE_PALETTE[expected_plate % len(PLATE_PALETTE)]
    # Cast to int to avoid uint8 wrap; allow generous L1 distance because
    # the per-plate hue shift can be ±30 per channel.
    delta = sum(abs(int(s) - int(p)) for s, p in zip(sample_color, pal))
    assert delta < 240, f"colour {sample_color} too far from palette {pal} (Δ={delta})"


def test_render_voronoi_empty_particles_produces_background() -> None:
    """Zero particles → all-background canvas (no crash)."""
    domain = WorldRect(width_km=100.0, height_km=100.0)
    plates: tuple[Plate, ...] = ()
    img = render_voronoi_png(
        domain,
        positions_km=np.zeros((0, 2)),
        plate_id=np.zeros(0, dtype=np.int32),
        plates=plates,
        crust_type=None,
    )
    assert img.size[0] > 0


# -----------------------------------------------------------------------------
# render_single_plate_png
# -----------------------------------------------------------------------------

def test_render_single_plate_returns_image() -> None:
    domain, pos, pid, ct, plates = _toy_state()
    img = render_single_plate_png(
        domain, pos, pid, plates, target_plate_id=0,
        crust_type=ct, px_per_km=1.5,
    )
    assert img.size[0] > 0
    assert img.mode == "RGB"


def test_render_single_plate_unknown_plate_raises() -> None:
    domain, pos, pid, _ct, plates = _toy_state()
    with pytest.raises(ValueError, match="not in this snapshot"):
        render_single_plate_png(
            domain, pos, pid, plates, target_plate_id=999,
        )


def test_render_single_plate_empty_plate_produces_placeholder() -> None:
    """A plate with no owned particles produces a small placeholder image
    rather than crashing."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    pos = np.zeros((0, 2))
    pid = np.zeros(0, dtype=np.int32)
    plates = (
        Plate(id=0, type="continental",
              seed_position_km=(0.0, 0.0), velocity_kmpy=(0.0, 0.0)),
    )
    img = render_single_plate_png(domain, pos, pid, plates, target_plate_id=0)
    assert img.size[0] > 0


def test_render_single_plate_canvas_smaller_than_world() -> None:
    """The per-plate render is cropped to the plate's bbox, so it should
    typically be smaller than the full-world particle render."""
    domain, pos, pid, ct, plates = _toy_state()
    world_img = render_particles_png(
        domain, pos, pid, plates, crust_type=ct, px_per_km=2.0,
    )
    plate_img = render_single_plate_png(
        domain, pos, pid, plates, target_plate_id=0,
        crust_type=ct, px_per_km=2.0,
    )
    # Plate 0 occupies only the left third roughly; per-plate canvas
    # should be substantially narrower than the full domain.
    assert plate_img.size[0] < world_img.size[0]


# -----------------------------------------------------------------------------
# Top-level render_initial_state
# -----------------------------------------------------------------------------

def test_render_initial_state_emits_expected_files(
    default_sim_config: SimConfig, tmp_path: Path,
) -> None:
    """``render_initial_state`` writes particles.png, voronoi.png, and one
    per-plate PNG under ``plates/``."""
    domain = WorldRect(width_km=200.0, height_km=200.0)
    out_dir = render_initial_state(
        domain,
        default_sim_config,
        seed=42,
        out_dir=tmp_path / "seed_view",
        px_per_km=1.0,
        voronoi_px_per_km=0.6,
    )
    assert out_dir.exists()
    assert (out_dir / "particles.png").exists()
    assert (out_dir / "voronoi.png").exists()
    plates_dir = out_dir / "plates"
    assert plates_dir.is_dir()
    plate_files = sorted(plates_dir.glob("plate_*.png"))
    assert len(plate_files) == default_sim_config.plate_count


def test_render_initial_state_pngs_are_valid_images(
    default_sim_config: SimConfig, tmp_path: Path,
) -> None:
    """Each emitted file is openable as an image (catches truncation /
    encoder bugs)."""
    domain = WorldRect(width_km=150.0, height_km=150.0)
    out_dir = render_initial_state(
        domain, default_sim_config, seed=7, out_dir=tmp_path,
    )
    for p in out_dir.rglob("*.png"):
        with Image.open(p) as img:
            img.load()
            assert img.size[0] > 0
            assert img.size[1] > 0


def test_render_initial_state_deterministic(
    default_sim_config: SimConfig, tmp_path: Path,
) -> None:
    """Same (config, domain, seed) → byte-identical particles.png."""
    domain = WorldRect(width_km=150.0, height_km=150.0)
    a_dir = render_initial_state(
        domain, default_sim_config, seed=7,
        out_dir=tmp_path / "a", px_per_km=1.0,
    )
    b_dir = render_initial_state(
        domain, default_sim_config, seed=7,
        out_dir=tmp_path / "b", px_per_km=1.0,
    )
    a_bytes = (a_dir / "particles.png").read_bytes()
    b_bytes = (b_dir / "particles.png").read_bytes()
    assert a_bytes == b_bytes


# -----------------------------------------------------------------------------
# Sanity smoke: build_initial_state + viz end-to-end with default config
# -----------------------------------------------------------------------------

def test_viz_works_end_to_end_with_default_config(
    default_sim_config: SimConfig,
) -> None:
    """Smoke: build the initial state at the default config + 1000×1000 km
    domain and check the three renderers all produce non-empty images."""
    domain = WorldRect(width_km=400.0, height_km=400.0)
    plates, pos, pid, ct, _, _ = build_initial_state(
        domain, default_sim_config, seed=42,
    )
    img1 = render_particles_png(domain, pos, pid, plates, crust_type=ct)
    img2 = render_voronoi_png(domain, pos, pid, plates, crust_type=ct)
    img3 = render_single_plate_png(
        domain, pos, pid, plates, plates[0].id, crust_type=ct,
    )
    for img in (img1, img2, img3):
        assert img.size[0] > 0
        assert img.size[1] > 0
