"""Continuous-2D visualisation utilities for ``tectonic_sim``.

All renderers work in km space and project to pixels at the end; they
never touch a hex. Functions take the array form (positions, plate_id,
crust_type, plates) rather than a ``Snapshot`` so they can be called on
intermediate state at any phase — useful for iterative debugging during
development.

Three primary renderers, each returns a PIL ``Image``:

  - ``render_particles_png`` — every particle as a small colored disc;
    plate seed positions marked; optional velocity arrows from seed
    positions; world rectangle border.
  - ``render_voronoi_png`` — every pixel coloured by the nearest
    particle's plate id (smooth tessellation that previews what a
    sampling-based cast will look like at high resolution).
  - ``render_single_plate_png`` — one plate's particles isolated and
    cropped to their bounding box (analogous to the per-plate PNG
    worldgen used to ship, but in continuous space rather than hex).

A convenience entry point ``render_initial_state(...)`` calls
``build_initial_state`` and emits all three renders to a directory; the
``python -m tectonic_sim inspect-seed`` CLI uses it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    Plate,
    SimConfig,
    Snapshot,
    WorldRect,
)


# -----------------------------------------------------------------------------
# Palette + colouring
# -----------------------------------------------------------------------------

# Perceptually-spaced colours for plate ids. Matches worldgen's palette so a
# cast-to-hex render keeps the same plate→colour mapping.
PLATE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (220,  90,  90),  # red
    ( 90, 160, 220),  # sky blue
    (220, 200,  90),  # gold
    (140, 200, 100),  # leaf green
    (200, 110, 200),  # magenta
    (110, 210, 200),  # teal
    (240, 160,  80),  # amber
    (170, 130, 220),  # lavender
    (230, 130, 160),  # pink
    ( 90, 220, 140),  # mint
    (180, 180, 200),  # cool grey
    (210, 100,  60),  # rust
    (130, 170, 240),  # cornflower
    (220, 220, 130),  # pale yellow
)

_BACKGROUND_RGB = (20, 20, 30)
_BORDER_RGB = (90, 90, 110)
_SEED_FILL_RGB = (245, 245, 245)
_SEED_OUTLINE_RGB = (15, 15, 20)
_LAND_TINT = 1.0          # multiplier for continental crust
_OCEAN_TINT = 0.55        # multiplier for oceanic crust


def _plate_color(plate_id: int, *, oceanic: bool = False) -> tuple[int, int, int]:
    """Plate's palette colour, dimmed for oceanic crust."""
    base = PLATE_PALETTE[plate_id % len(PLATE_PALETTE)]
    # Cycle through a small hue shift so plate 0 and plate 14 don't collide.
    cycle = plate_id // len(PLATE_PALETTE)
    shift = (cycle * 23) % 60 - 30
    r = max(0, min(255, base[0] + shift))
    g = max(0, min(255, base[1] - shift // 2))
    b = max(0, min(255, base[2] + shift // 2))
    if oceanic:
        r = int(r * _OCEAN_TINT)
        g = int(g * _OCEAN_TINT + 20)
        b = int(b * _OCEAN_TINT + 50)
    else:
        r = int(r * _LAND_TINT)
        g = int(g * _LAND_TINT)
        b = int(b * _LAND_TINT)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

# Pixel layout: y axis points down (PIL convention); we flip y so the
# physics' +y (north) reads as up on screen.

_CAPTION_BAND_PX = 22


def _canvas_size(domain: WorldRect, px_per_km: float) -> tuple[int, int]:
    """Total canvas size (width, height) including caption band + margin."""
    margin_px = 12
    w = int(round(domain.width_km * px_per_km)) + 2 * margin_px
    h = int(round(domain.height_km * px_per_km)) + 2 * margin_px + _CAPTION_BAND_PX
    return w, h


def _km_to_pixel(
    points_km: np.ndarray,
    domain: WorldRect,
    px_per_km: float,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Project (N, 2) km points into pixel coords inside the canvas."""
    w, _h = canvas_size
    cx = w / 2.0
    cy = _CAPTION_BAND_PX + (
        (canvas_size[1] - _CAPTION_BAND_PX) / 2.0
    )
    px = cx + points_km[:, 0] * px_per_km
    # Physics +y = north; PIL +y = south. Flip.
    py = cy - points_km[:, 1] * px_per_km
    return np.column_stack([px, py])


def _draw_rect_border(
    draw: ImageDraw.ImageDraw,
    domain: WorldRect,
    px_per_km: float,
    canvas_size: tuple[int, int],
) -> None:
    corners_km = np.array([
        [-domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, -domain.half_height_km],
        [domain.half_width_km, domain.half_height_km],
        [-domain.half_width_km, domain.half_height_km],
    ])
    corners_px = _km_to_pixel(corners_km, domain, px_per_km, canvas_size)
    xy = [tuple(p) for p in corners_px.tolist()]
    draw.line([xy[0], xy[1], xy[2], xy[3], xy[0]], fill=_BORDER_RGB, width=1)


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 12)
    except OSError:
        return ImageFont.load_default()


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    rgb: tuple[int, int, int] = (230, 230, 230),
) -> None:
    draw.text((10, 5), text, fill=rgb, font=_font())


# -----------------------------------------------------------------------------
# Renderer 1: scatter ('render_particles_png')
# -----------------------------------------------------------------------------

def render_particles_png(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    plates: Sequence[Plate],
    *,
    crust_type: np.ndarray | None = None,
    px_per_km: float = 1.0,
    show_velocities: bool = True,
    elapsed_myr: float = 0.0,
    caption: str | None = None,
) -> Image.Image:
    """Each particle as a small colored disc; plate seeds marked.

    Arguments:
        domain: the world rectangle.
        positions_km: ``(N, 2)`` particle positions in km.
        plate_id: ``(N,)`` int array of plate ids per particle.
        plates: the ``Plate`` tuple (used for type colouring + seed markers).
        crust_type: optional ``(N,)`` int8 crust type array. If provided,
            oceanic particles are drawn dimmer/bluer than continental.
        show_velocities: draw a velocity arrow at each plate's *current*
            centre (seed_position + velocity × elapsed_myr).
        elapsed_myr: time elapsed for plate-centre projection. Defaults
            to 0 (renders the t=0 state).
        caption: header text. Auto-generated if None.
    """
    canvas = _canvas_size(domain, px_per_km)
    img = Image.new("RGB", canvas, color=_BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)

    _draw_rect_border(draw, domain, px_per_km, canvas)

    # Per-particle colour. Vectorise via a (n_plate, 2) colour table:
    # one row for continental, one for oceanic, indexed by plate id.
    n_plates = len(plates)
    color_table_cont = np.array(
        [_plate_color(p.id, oceanic=False) for p in plates], dtype=np.uint8,
    )
    color_table_ocn = np.array(
        [_plate_color(p.id, oceanic=True) for p in plates], dtype=np.uint8,
    )

    if crust_type is None:
        colors = color_table_cont[plate_id]
    else:
        oceanic_mask = (crust_type == CRUST_OCEANIC)
        colors = np.where(
            oceanic_mask[:, None], color_table_ocn[plate_id], color_table_cont[plate_id],
        ).astype(np.uint8)

    px_xy = _km_to_pixel(positions_km, domain, px_per_km, canvas)
    radius = max(1.5, 0.6 * px_per_km)

    # Draw each particle. Vectorised batch isn't possible with PIL ellipse,
    # so we loop — but the loop is over a few thousand items, fast enough.
    rgb_tuples = [tuple(int(c) for c in row) for row in colors]
    for (x, y), col in zip(px_xy, rgb_tuples):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=col,
        )

    # Plate seed markers + velocity arrows.
    for plate in plates:
        sx_km, sy_km = plate.seed_position_km
        vx_kmpy, vy_kmpy = plate.velocity_kmpy
        # Current centre = seed + v·t.
        cx_km = sx_km + vx_kmpy * elapsed_myr
        cy_km = sy_km + vy_kmpy * elapsed_myr
        cx_px, cy_px = _km_to_pixel(
            np.array([[cx_km, cy_km]]), domain, px_per_km, canvas,
        )[0]
        r = max(2.5, px_per_km * 1.0)
        draw.ellipse(
            [cx_px - r, cy_px - r, cx_px + r, cy_px + r],
            fill=_SEED_FILL_RGB, outline=_SEED_OUTLINE_RGB,
        )

        if show_velocities:
            # Draw a small velocity arrow from the current centre.
            arrow_len_km = max(domain.width_km, domain.height_km) * 0.05
            speed = math.hypot(vx_kmpy, vy_kmpy)
            if speed > 0:
                ax_km = cx_km + arrow_len_km * vx_kmpy / speed
                ay_km = cy_km + arrow_len_km * vy_kmpy / speed
                ax_px, ay_px = _km_to_pixel(
                    np.array([[ax_km, ay_km]]), domain, px_per_km, canvas,
                )[0]
                draw.line(
                    [(cx_px, cy_px), (ax_px, ay_px)],
                    fill=_SEED_OUTLINE_RGB, width=1,
                )

    # Caption.
    if caption is None:
        n_cont = sum(1 for p in plates if p.type == "continental")
        n_ocn = n_plates - n_cont
        caption = (
            f"particles: {positions_km.shape[0]}   "
            f"plates: {n_plates} (cont {n_cont} / ocn {n_ocn})   "
            f"domain: {domain.width_km:g}×{domain.height_km:g} km   "
            f"t = {elapsed_myr:g} Myr"
        )
    _draw_caption(draw, caption)
    return img


# -----------------------------------------------------------------------------
# Renderer 2: voronoi ('render_voronoi_png')
# -----------------------------------------------------------------------------

def _voronoi_field(
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    domain: WorldRect,
    width_px: int,
    height_px: int,
    *,
    crust_type: np.ndarray | None,
    plates: Sequence[Plate],
    wrap: bool = False,
) -> np.ndarray:
    """Per-pixel nearest-particle colour. Returns ``(H, W, 3) uint8``."""
    if positions_km.shape[0] == 0:
        return np.full((height_px, width_px, 3), _BACKGROUND_RGB, dtype=np.uint8)

    # Pixel coordinates in km space (centre of each pixel).
    xs_km = np.linspace(
        -domain.half_width_km + 0.5 * domain.width_km / width_px,
        domain.half_width_km - 0.5 * domain.width_km / width_px,
        width_px,
    )
    ys_km = np.linspace(
        domain.half_height_km - 0.5 * domain.height_km / height_px,
        -domain.half_height_km + 0.5 * domain.height_km / height_px,
        height_px,
    )
    # Flat pixel-km array (W*H, 2). We chunk in rows to keep memory bounded.
    px_per_chunk = max(1, 4_000_000 // max(1, positions_km.shape[0]))
    rows_per_chunk = max(1, px_per_chunk // width_px)

    image = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    color_table_cont = np.array(
        [_plate_color(p.id, oceanic=False) for p in plates], dtype=np.uint8,
    )
    color_table_ocn = np.array(
        [_plate_color(p.id, oceanic=True) for p in plates], dtype=np.uint8,
    )

    for y_start in range(0, height_px, rows_per_chunk):
        y_end = min(y_start + rows_per_chunk, height_px)
        rows_y = ys_km[y_start:y_end]
        gx, gy = np.meshgrid(xs_km, rows_y)
        pixels = np.column_stack([gx.ravel(), gy.ravel()])

        # Distance²: (P, N, 2) - subtract via broadcasting on axes.
        # (chunk*W, 1, 2) - (1, N, 2) → (chunk*W, N, 2) → sum -> (chunk*W, N)
        dxy = pixels[:, None, :] - positions_km[None, :, :]
        if wrap:
            dxy_x, dxy_y = domain.wrapped_delta_xy(dxy[..., 0], dxy[..., 1])
            d2 = dxy_x * dxy_x + dxy_y * dxy_y
        else:
            d2 = np.sum(dxy * dxy, axis=-1)
        nearest = np.argmin(d2, axis=1)
        nearest_pid = plate_id[nearest]
        if crust_type is None:
            row_colors = color_table_cont[nearest_pid]
        else:
            nearest_ct = crust_type[nearest]
            ocean = (nearest_ct == CRUST_OCEANIC)
            row_colors = np.where(
                ocean[:, None],
                color_table_ocn[nearest_pid],
                color_table_cont[nearest_pid],
            ).astype(np.uint8)
        image[y_start:y_end] = row_colors.reshape(y_end - y_start, width_px, 3)
    return image


def _color_elevation(
    elev_km: float, *, low_km: float, high_km: float,
) -> tuple[int, int, int]:
    """Blue (ocean) → green → tan → white (snow) colormap."""
    if high_km == low_km:
        return (180, 180, 180)
    t = float((elev_km - low_km) / (high_km - low_km))
    t = max(0.0, min(1.0, t))
    if t < 0.45:                   # ocean: deep blue → cyan
        s = t / 0.45
        return (
            int(20 + 80 * s),
            int(60 + 130 * s),
            int(160 + 50 * s),
        )
    if t < 0.55:                   # coast: cyan → green
        s = (t - 0.45) / 0.1
        return (
            int(100 - 30 * s),
            int(190 - 20 * s),
            int(210 - 130 * s),
        )
    if t < 0.75:                   # lowland: green → tan
        s = (t - 0.55) / 0.20
        return (
            int(70 + 160 * s),
            int(170 - 30 * s),
            int(80 - 40 * s),
        )
    if t < 0.90:                   # midland: tan → brown
        s = (t - 0.75) / 0.15
        return (
            int(230 - 80 * s),
            int(140 - 70 * s),
            int(40 + 10 * s),
        )
    # high: brown → white (snowcap)
    s = (t - 0.90) / 0.10
    return (
        int(150 + 105 * s),
        int(70 + 185 * s),
        int(50 + 205 * s),
    )


def render_elevation_png(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    plates: Sequence[Plate],
    elevation_km: np.ndarray,
    *,
    px_per_km: float = 1.0,
    elev_band_km: tuple[float, float] | None = None,
    show_seeds: bool = True,
    elapsed_myr: float = 0.0,
    caption: str | None = None,
) -> Image.Image:
    """Render particles coloured by elevation.

    ``elevation_km`` is the parallel ``(N,)`` array from
    ``isostasy.particle_elevation_km``. ``elev_band_km`` sets the
    colormap range; auto-derived from the data when ``None``.
    """
    canvas = _canvas_size(domain, px_per_km)
    img = Image.new("RGB", canvas, color=_BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)
    _draw_rect_border(draw, domain, px_per_km, canvas)

    if elev_band_km is None:
        low = float(np.percentile(elevation_km, 2)) if elevation_km.size else -3.0
        high = float(np.percentile(elevation_km, 98)) if elevation_km.size else 3.0
        # Pad slightly so extremes don't saturate to the band ends.
        spread = max(high - low, 0.5)
        low -= 0.05 * spread
        high += 0.05 * spread
    else:
        low, high = elev_band_km

    px_xy = _km_to_pixel(positions_km, domain, px_per_km, canvas)
    radius = max(1.5, 0.6 * px_per_km)
    for (x, y), e in zip(px_xy, elevation_km):
        col = _color_elevation(float(e), low_km=low, high_km=high)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=col)

    if show_seeds:
        for plate in plates:
            sx_km, sy_km = plate.seed_position_km
            vx, vy = plate.velocity_kmpy
            cx_km = sx_km + vx * elapsed_myr
            cy_km = sy_km + vy * elapsed_myr
            cx_px, cy_px = _km_to_pixel(
                np.array([[cx_km, cy_km]]), domain, px_per_km, canvas,
            )[0]
            r = max(2.5, px_per_km * 1.0)
            draw.ellipse(
                [cx_px - r, cy_px - r, cx_px + r, cy_px + r],
                fill=_SEED_FILL_RGB, outline=_SEED_OUTLINE_RGB,
            )

    if caption is None:
        peak_idx = int(np.argmax(elevation_km)) if elevation_km.size else 0
        peak = float(elevation_km[peak_idx]) if elevation_km.size else 0.0
        deep = float(elevation_km.min()) if elevation_km.size else 0.0
        caption = (
            f"elevation   range [{low:+.1f}, {high:+.1f}] km   "
            f"peak {peak:+.2f}km   deepest {deep:+.2f}km   "
            f"particles: {positions_km.shape[0]}   "
            f"t = {elapsed_myr:g} Myr"
        )
    _draw_caption(draw, caption)
    return img


def render_voronoi_png(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    plates: Sequence[Plate],
    *,
    crust_type: np.ndarray | None = None,
    px_per_km: float = 0.6,
    show_seeds: bool = True,
    elapsed_myr: float = 0.0,
    caption: str | None = None,
    wrap: bool = False,
) -> Image.Image:
    """Per-pixel nearest-particle voronoi tessellation.

    Slower than ``render_particles_png`` (O(W·H·N) brute force), so the
    default ``px_per_km`` is gentler (0.6 px/km → 600×600 for a 1000 km
    world). Bump it for finer renders.
    """
    canvas = _canvas_size(domain, px_per_km)
    w_total, h_total = canvas
    margin = 12
    field_w = w_total - 2 * margin
    field_h = h_total - 2 * margin - _CAPTION_BAND_PX

    # Build the voronoi field at the inner canvas size.
    field = _voronoi_field(
        positions_km, plate_id, domain, field_w, field_h,
        crust_type=crust_type, plates=plates, wrap=wrap,
    )

    img = Image.new("RGB", canvas, color=_BACKGROUND_RGB)
    img.paste(Image.fromarray(field), (margin, _CAPTION_BAND_PX + margin))
    draw = ImageDraw.Draw(img)

    _draw_rect_border(draw, domain, px_per_km, canvas)

    if show_seeds:
        for plate in plates:
            sx_km, sy_km = plate.seed_position_km
            vx, vy = plate.velocity_kmpy
            cx_km = sx_km + vx * elapsed_myr
            cy_km = sy_km + vy * elapsed_myr
            cx_px, cy_px = _km_to_pixel(
                np.array([[cx_km, cy_km]]), domain, px_per_km, canvas,
            )[0]
            r = max(2.5, px_per_km * 1.5)
            draw.ellipse(
                [cx_px - r, cy_px - r, cx_px + r, cy_px + r],
                fill=_SEED_FILL_RGB, outline=_SEED_OUTLINE_RGB,
            )

    if caption is None:
        caption = (
            f"voronoi (nearest particle per pixel)   "
            f"particles: {positions_km.shape[0]}   "
            f"domain: {domain.width_km:g}×{domain.height_km:g} km   "
            f"t = {elapsed_myr:g} Myr"
        )
    _draw_caption(draw, caption)
    return img


# -----------------------------------------------------------------------------
# Renderer 3: single plate, cropped ('render_single_plate_png')
# -----------------------------------------------------------------------------

def render_single_plate_png(
    domain: WorldRect,
    positions_km: np.ndarray,
    plate_id: np.ndarray,
    plates: Sequence[Plate],
    target_plate_id: int,
    *,
    crust_type: np.ndarray | None = None,
    px_per_km: float = 1.5,
    margin_km: float = 30.0,
    elapsed_myr: float = 0.0,
    caption: str | None = None,
) -> Image.Image:
    """Render one plate's particles, cropped to their bounding box.

    The focal plate's particles are coloured; everything else is drawn
    dim grey so the focal plate's shape stands out. Canvas is sized to
    the particle bbox plus ``margin_km``; ``px_per_km`` controls the
    on-screen density.
    """
    plate_by_id = {p.id: p for p in plates}
    if target_plate_id not in plate_by_id:
        raise ValueError(f"plate {target_plate_id} not in this snapshot")
    plate = plate_by_id[target_plate_id]

    mask = (plate_id == target_plate_id)
    own = positions_km[mask]
    n_own = own.shape[0]
    if n_own == 0:
        # No particles. Emit a small placeholder.
        img = Image.new("RGB", (200, 56), color=_BACKGROUND_RGB)
        draw = ImageDraw.Draw(img)
        _draw_caption(draw, f"plate #{target_plate_id}: 0 particles")
        return img

    own_ct = crust_type[mask] if crust_type is not None else None
    xmin, ymin = own.min(axis=0)
    xmax, ymax = own.max(axis=0)

    # Sub-domain = bbox + margin (with parity to the world rectangle border).
    sub = WorldRect(
        width_km=(xmax - xmin) + 2 * margin_km,
        height_km=(ymax - ymin) + 2 * margin_km,
    )
    # Recenter coordinates so the bbox centre lands at sub's origin.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    def to_sub(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[:, 0] -= center_x
        out[:, 1] -= center_y
        return out

    canvas = _canvas_size(sub, px_per_km)
    img = Image.new("RGB", canvas, color=_BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)
    _draw_rect_border(draw, sub, px_per_km, canvas)

    # Draw out-of-plate particles dim grey (only those whose bbox crosses
    # the visible region; cheap to test).
    other_mask = ~mask
    others = positions_km[other_mask].copy()
    others = to_sub(others)
    others_in_view = (
        (np.abs(others[:, 0]) <= sub.half_width_km + 5)
        & (np.abs(others[:, 1]) <= sub.half_height_km + 5)
    )
    others = others[others_in_view]
    if others.size > 0:
        px_other = _km_to_pixel(others, sub, px_per_km, canvas)
        r_other = max(1.0, 0.4 * px_per_km)
        dim = (60, 60, 70)
        for x, y in px_other:
            draw.ellipse([x - r_other, y - r_other, x + r_other, y + r_other], fill=dim)

    # Draw focal plate particles.
    focal_cont = _plate_color(plate.id, oceanic=False)
    focal_ocn = _plate_color(plate.id, oceanic=True)
    own_local = to_sub(own)
    px_own = _km_to_pixel(own_local, sub, px_per_km, canvas)
    r_own = max(1.5, 0.6 * px_per_km)
    if own_ct is None:
        for x, y in px_own:
            draw.ellipse(
                [x - r_own, y - r_own, x + r_own, y + r_own], fill=focal_cont,
            )
    else:
        oceanic_own = (own_ct == CRUST_OCEANIC)
        for (x, y), is_ocean in zip(px_own, oceanic_own):
            col = focal_ocn if is_ocean else focal_cont
            draw.ellipse(
                [x - r_own, y - r_own, x + r_own, y + r_own], fill=col,
            )

    # Plate centre (white dot) at its current position in sub-coords.
    cx_km = plate.seed_position_km[0] + plate.velocity_kmpy[0] * elapsed_myr - center_x
    cy_km = plate.seed_position_km[1] + plate.velocity_kmpy[1] * elapsed_myr - center_y
    cx_px, cy_px = _km_to_pixel(
        np.array([[cx_km, cy_km]]), sub, px_per_km, canvas,
    )[0]
    r_c = max(2.5, px_per_km * 1.2)
    draw.ellipse(
        [cx_px - r_c, cy_px - r_c, cx_px + r_c, cy_px + r_c],
        fill=_SEED_FILL_RGB, outline=_SEED_OUTLINE_RGB,
    )

    if caption is None:
        if own_ct is None:
            split = f"{n_own} particles"
        else:
            n_ocn = int(np.sum(own_ct == CRUST_OCEANIC))
            split = f"{n_own} particles (cont {n_own - n_ocn} / ocn {n_ocn})"
        caption = (
            f"plate #{plate.id} ({plate.type})   {split}   "
            f"bbox: {sub.width_km - 2*margin_km:.0f}×{sub.height_km - 2*margin_km:.0f} km"
        )
    _draw_caption(draw, caption)
    return img


# -----------------------------------------------------------------------------
# Snapshot-aware wrappers (for later phases that hand us a Snapshot/Frame)
# -----------------------------------------------------------------------------

def render_snapshot_particles(snapshot: Snapshot, **kwargs) -> Image.Image:
    """Convenience: ``render_particles_png`` from a ``Snapshot``."""
    return render_particles_png(
        snapshot.domain,
        snapshot.particle_position_km,
        snapshot.particle_plate_id,
        snapshot.plates,
        crust_type=snapshot.particle_crust_type,
        elapsed_myr=snapshot.final_time_myr,
        **kwargs,
    )


def render_snapshot_voronoi(snapshot: Snapshot, **kwargs) -> Image.Image:
    """Convenience: ``render_voronoi_png`` from a ``Snapshot``."""
    return render_voronoi_png(
        snapshot.domain,
        snapshot.particle_position_km,
        snapshot.particle_plate_id,
        snapshot.plates,
        crust_type=snapshot.particle_crust_type,
        elapsed_myr=snapshot.final_time_myr,
        **kwargs,
    )


def render_snapshot_single_plate(
    snapshot: Snapshot, target_plate_id: int, **kwargs,
) -> Image.Image:
    """Convenience: ``render_single_plate_png`` from a ``Snapshot``."""
    return render_single_plate_png(
        snapshot.domain,
        snapshot.particle_position_km,
        snapshot.particle_plate_id,
        snapshot.plates,
        target_plate_id,
        crust_type=snapshot.particle_crust_type,
        elapsed_myr=snapshot.final_time_myr,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Top-level: render the seeded initial state to a directory
# -----------------------------------------------------------------------------

def render_initial_state(
    domain: WorldRect,
    sim_config: SimConfig,
    seed: int,
    out_dir: Path,
    *,
    px_per_km: float = 1.0,
    voronoi_px_per_km: float = 0.6,
) -> Path:
    """Build the t=0 state and emit particles / voronoi / per-plate PNGs.

    Returns the output directory path. Used by the ``inspect-seed`` CLI
    and useful for ad-hoc debugging during development.
    """
    from tectonic_sim.seeding import build_initial_state

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plates, positions, plate_id, crust_type, _thick, _age = build_initial_state(
        domain, sim_config, seed=seed,
    )

    # 1. Particles scatter.
    img = render_particles_png(
        domain, positions, plate_id, plates,
        crust_type=crust_type, px_per_km=px_per_km,
    )
    img.save(out_dir / "particles.png")

    # 2. Voronoi tessellation.
    img = render_voronoi_png(
        domain, positions, plate_id, plates,
        crust_type=crust_type, px_per_km=voronoi_px_per_km,
    )
    img.save(out_dir / "voronoi.png")

    # 3. Per-plate isolated views.
    plates_dir = out_dir / "plates"
    plates_dir.mkdir(exist_ok=True)
    for plate in plates:
        img = render_single_plate_png(
            domain, positions, plate_id, plates, plate.id,
            crust_type=crust_type, px_per_km=max(px_per_km, 1.2),
        )
        img.save(plates_dir / f"plate_{plate.id:02d}.png")

    return out_dir
