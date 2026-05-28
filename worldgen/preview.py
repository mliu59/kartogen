"""Headless terrain preview: renders a ``GeneratedWorld`` to a PNG.

Standalone — does not import the simulation engine. Useful for iterating on
terrain generation without a display, and for snapshot-style visual checks
in the test suite.

Usage::

    python -m worldgen.preview --seed 42 --radius 80 \\
        --layer biome --out preview.png

Layers:
    biome           Final terrain classification (default; colored by biome).
    elevation       Heightmap, blue-low-to-white-high.
    temperature     Temperature, blue-cold-to-red-hot.
    precipitation   Precipitation, tan-dry-to-darkgreen-wet.
    flow            Log-scaled flow accumulation (drainage map).
    composite       Biome with hillshade + river overlay.
    plates          Each tectonic plate colored uniquely; oceanic plates
                    drawn cooler/darker than continental ones, plate
                    boundaries outlined. Requires ``mask_mode = "plates"``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from worldgen.hex import Hex
from worldgen import generate as run_pipeline
from worldgen.config_loader import load_worldgen_config
from worldgen.types import WorldgenConfig

if TYPE_CHECKING:
    from worldgen.pipeline import GeneratedWorld


# Biome palette (final terrain). Tuned to read clearly when many hexes
# are crowded into a small image.
BIOME_COLORS: dict[str, tuple[int, int, int]] = {
    "deep_ocean":       (28, 50, 110),
    "ocean":            (44, 92, 158),
    "coast":            (90, 170, 200),
    "lake":             (74, 138, 200),
    "river":            (88, 154, 210),
    "plains":           (170, 200, 100),
    "grassland":        (190, 210, 120),
    "savanna":          (210, 200, 95),
    "desert":           (232, 215, 145),
    "tundra":           (200, 200, 195),
    "temperate_forest": (60, 130, 75),
    "taiga":            (52, 100, 80),
    "jungle":           (30, 110, 55),
    "hills":            (150, 130, 90),
    "mountain":         (120, 105, 95),
    "snow_peak":        (240, 240, 245),
}


def _hex_to_pixel(q: int, r: int, size: float, cx: float, cy: float) -> tuple[float, float]:
    """Flat-top hex → pixel."""
    x = cx + size * 1.5 * q
    y = cy + size * math.sqrt(3.0) * (r + q / 2.0)
    return x, y


def _hex_corners(px: float, py: float, size: float) -> list[tuple[float, float]]:
    return [
        (px + size * math.cos(math.radians(60 * i)),
         py + size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def _figure_size(radius: int, hex_px: float) -> tuple[int, int, float, float]:
    w = int(hex_px * 1.5 * (2 * radius) + hex_px * 2 + 40)
    h = int(hex_px * math.sqrt(3.0) * (2 * radius) + hex_px * math.sqrt(3.0) + 40)
    return w, h, w / 2.0, h / 2.0


def _color_elevation(elev: float, sea_level: float = 0.0) -> tuple[int, int, int]:
    """Blue (below sea) → green (low land) → tan → white (high)."""
    if elev < sea_level:
        # Ocean depth: deeper = darker.
        d = max(0.0, min(1.0, (sea_level - elev) / 0.5))
        b = int(110 - 80 * d)
        g = int(60 - 40 * d)
        r = int(28 - 18 * d)
        return (max(0, r), max(0, g), max(20, b))
    e = max(0.0, min(1.0, elev / 0.8))
    if e < 0.25:
        # Green to yellow
        t = e / 0.25
        return (int(80 + 130 * t), int(150 - 20 * t), int(80 - 50 * t))
    if e < 0.6:
        # Yellow-tan to brown
        t = (e - 0.25) / 0.35
        return (int(210 - 60 * t), int(130 - 50 * t), int(30 + 30 * t))
    # Brown to white (snow caps)
    t = (e - 0.6) / 0.4
    return (int(150 + 105 * t), int(80 + 160 * t), int(60 + 190 * t))


def _color_temperature(t_c: float) -> tuple[int, int, int]:
    """Blue (<-20°C) → cyan → green → yellow → red (>35°C)."""
    x = max(0.0, min(1.0, (t_c + 20.0) / 55.0))
    if x < 0.25:
        s = x / 0.25
        return (int(20 + 40 * s), int(60 + 130 * s), int(180 - 30 * s))
    if x < 0.5:
        s = (x - 0.25) / 0.25
        return (int(60 + 60 * s), int(190 - 40 * s), int(150 - 110 * s))
    if x < 0.75:
        s = (x - 0.5) / 0.25
        return (int(120 + 100 * s), int(150 + 40 * s), int(40 + 0 * s))
    s = (x - 0.75) / 0.25
    return (int(220 + 30 * s), int(190 - 100 * s), int(40 - 30 * s))


def _color_precipitation(p_mm: float) -> tuple[int, int, int]:
    """Tan (dry) → green (moderate) → dark green (wet) → blue (very wet).
    Ocean (p=0) is rendered separately by the caller; this colormap assumes land.
    """
    x = max(0.0, min(1.0, p_mm / 2200.0))
    if x < 0.15:
        s = x / 0.15
        return (int(232 - 60 * s), int(215 - 80 * s), int(145 - 50 * s))
    if x < 0.5:
        s = (x - 0.15) / 0.35
        return (int(172 - 100 * s), int(135 - 5 * s), int(95 - 70 * s))
    s = (x - 0.5) / 0.5
    return (int(72 - 40 * s), int(130 - 30 * s), int(25 + 100 * s))


def _color_flow(flow: int, max_flow: int) -> tuple[int, int, int]:
    if flow <= 1 or max_flow <= 1:
        return (45, 45, 60)
    x = math.log10(flow) / math.log10(max_flow)
    x = max(0.0, min(1.0, x))
    return (int(40 + 70 * x), int(60 + 140 * x), int(120 + 100 * x))


# Distinct, perceptually-spaced colors for plate IDs. Chosen to read well on
# the dark map background and to make adjacent plates easy to tell apart.
# Plate IDs > len(palette) cycle through with a deterministic hue shift so
# the same plate id always renders the same color across runs.
_PLATE_PALETTE: tuple[tuple[int, int, int], ...] = (
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


def _plate_color(
    plate_id: int, is_oceanic: bool,
) -> tuple[int, int, int]:
    """Color for a plate. Oceanic plates are darkened + slightly blue-shifted
    so the continental / oceanic distinction is visible at a glance."""
    base = _PLATE_PALETTE[plate_id % len(_PLATE_PALETTE)]
    # Deterministic shift per palette cycle so plates 0 and 14 don't collide.
    cycle = plate_id // len(_PLATE_PALETTE)
    shift = (cycle * 23) % 60  # within ±30 of base
    r = max(0, min(255, base[0] + shift - 30))
    g = max(0, min(255, base[1] - shift + 30))
    b = max(0, min(255, base[2] + (shift // 2)))
    if is_oceanic:
        # Dim and pull toward blue.
        r = int(r * 0.45)
        g = int(g * 0.55)
        b = int(b * 0.70 + 40)
    return (r, g, b)


def render(
    gen: GeneratedWorld,
    layer: str,
    hex_px: float = 6.0,
    show_legend: bool = True,
) -> Image.Image:
    """Render the given layer to a PIL Image."""
    w, h, cx, cy = _figure_size(gen.radius, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    max_flow = max((d.flow_accumulation for d in gen.hexes.values()), default=1)

    if layer == "plates":
        return _render_plates(gen, hex_px, show_legend)

    for hex, data in gen.hexes.items():
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)

        if layer == "biome":
            color = BIOME_COLORS.get(data.biome, (200, 0, 200))
        elif layer == "elevation":
            color = _color_elevation(data.elevation, sea_level=0.0)
        elif layer == "temperature":
            # Same colormap for land and ocean — muting ocean produced
            # visually sharp discontinuities along coastlines and rift
            # valleys that read like data errors. The latitudinal gradient
            # is already legible because ocean and land share the same
            # temperature field.
            color = _color_temperature(data.temperature_c)
        elif layer == "precipitation":
            if data.is_ocean:
                color = BIOME_COLORS["ocean"]
            else:
                color = _color_precipitation(data.precipitation_mm)
        elif layer == "flow":
            color = _color_flow(data.flow_accumulation, max_flow)
        elif layer == "composite":
            # Biome + hillshade tinting + river overlay.
            base = BIOME_COLORS.get(data.biome, (200, 0, 200))
            # Simple north-westward hillshade using the elevation field.
            tint = _hillshade_tint(hex, gen)
            color = (
                max(0, min(255, base[0] + tint)),
                max(0, min(255, base[1] + tint)),
                max(0, min(255, base[2] + tint)),
            )
        else:
            color = (0, 0, 0)

        draw.polygon(corners, fill=color)

    if layer == "composite":
        _draw_rivers(draw, gen, hex_px, cx, cy)

    if show_legend:
        _draw_legend(draw, layer, w, h, gen)

    return img


def _render_plates(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Color each hex by its plate id; oceanic plates darkened/blue-shifted.

    Plate boundary hexes get a darker outline so plate edges read clearly.
    The plate seed hex for each plate is marked with a small white dot.
    """
    w, h, cx, cy = _figure_size(gen.radius, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.plates is None:
        # Plates weren't generated for this world — produce an explanatory
        # tile rather than silently rendering an empty image.
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
        draw.text(
            (10, 10),
            "Plates layer not generated for this world.\n"
            "Set `mask_mode = \"plates\"` in [worldgen.elevation] to enable.",
            fill=(230, 200, 120), font=font,
        )
        return img

    field = gen.plates
    plate_by_id = {p.id: p for p in field.plates}

    for hex, data in gen.hexes.items():
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        pid = field.hex_to_plate[hex]
        plate = plate_by_id[pid]
        is_oceanic = plate.type == "oceanic"
        color = _plate_color(pid, is_oceanic)
        # Boundary hexes get a darker outline; interior hexes have no outline
        # so plate interiors read as solid color blocks.
        on_boundary = field.distance_to_boundary_km[hex] == 0.0
        if on_boundary:
            outline = (15, 15, 20)
            width = max(1, int(hex_px / 6))
        else:
            outline = None  # type: ignore[assignment]
            width = 0
        draw.polygon(corners, fill=color, outline=outline, width=width)

    # Mark each plate's seed hex with a small contrasting dot.
    for plate in field.plates:
        if plate.seed_hex not in gen.hexes:
            continue
        px, py = _hex_to_pixel(
            plate.seed_hex.q, plate.seed_hex.r, hex_px, cx, cy,
        )
        r = max(2.0, hex_px * 0.35)
        draw.ellipse(
            [(px - r, py - r), (px + r, py + r)],
            fill=(245, 245, 245), outline=(20, 20, 20),
        )

    if show_legend:
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        n_continental = sum(1 for p in field.plates if p.type == "continental")
        n_oceanic = sum(1 for p in field.plates if p.type == "oceanic")
        caption = (
            f"plates: {len(field.plates)}   "
            f"continental: {n_continental}   oceanic: {n_oceanic}   "
            f"(seeds = white dots, boundaries outlined)"
        )
        draw.text((10, 10), caption, fill=(230, 230, 230), font=font)

        # Per-plate id swatches stacked along the bottom-left so the user can
        # match the colors on the map to a numeric id.
        swatch = 14
        gap = 2
        col_h = (swatch + gap) * len(field.plates)
        y0 = h - 10 - col_h
        for i, plate in enumerate(field.plates):
            yy = y0 + i * (swatch + gap)
            sw_color = _plate_color(plate.id, plate.type == "oceanic")
            draw.rectangle([10, yy, 10 + swatch, yy + swatch], fill=sw_color)
            label = f"#{plate.id} {plate.type[:4]}"
            draw.text(
                (10 + swatch + 4, yy - 1), label,
                fill=(220, 220, 220), font=font,
            )

    return img


def _hillshade_tint(h: Hex, gen: GeneratedWorld) -> int:
    """Compute a -40..+40 brightness adjustment from slope toward NW light."""
    elev_here = gen.hexes[h].elevation
    if gen.hexes[h].is_ocean:
        return 0
    # NW direction in axial space ≈ (-1, 0) on flat-top; the canonical NW
    # neighbor is direction (0, -1) per the AXIAL_DIRECTIONS table.
    nw = Hex(h.q + 0, h.r - 1)
    if nw not in gen.hexes:
        return 0
    elev_nw = gen.hexes[nw].elevation
    slope = elev_nw - elev_here  # positive: NW is higher → we are in shadow
    tint = int(-slope * 600.0)
    return max(-50, min(50, tint))


def _draw_rivers(
    draw: ImageDraw.ImageDraw,
    gen: GeneratedWorld,
    hex_px: float,
    cx: float,
    cy: float,
) -> None:
    """Draw river polylines from each river hex to its downstream river hex."""
    hydro = gen.hydrology
    for h, data in gen.hexes.items():
        if not data.is_river:
            continue
        down = hydro.downstream.get(h)
        if down is None or down not in gen.hexes:
            continue
        px1, py1 = _hex_to_pixel(h.q, h.r, hex_px, cx, cy)
        px2, py2 = _hex_to_pixel(down.q, down.r, hex_px, cx, cy)
        # Width scales with log flow.
        flow = data.flow_accumulation
        width = max(1.0, min(3.5, 0.4 + 0.6 * math.log10(max(2, flow))))
        draw.line([(px1, py1), (px2, py2)], fill=(60, 130, 200), width=int(round(width)))


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    layer: str,
    w: int,
    h: int,
    gen: GeneratedWorld,
) -> None:
    """Draw a small caption and biome legend."""
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    caption = f"layer={layer}  radius={gen.radius}  hexes={len(gen.hexes)}"
    draw.text((10, 10), caption, fill=(220, 220, 220), font=font)

    if layer in ("biome", "composite"):
        # Bottom-left biome key.
        x0, y0 = 10, h - 10 - 16 * len(BIOME_COLORS)
        for i, (name, color) in enumerate(BIOME_COLORS.items()):
            yy = y0 + i * 16
            draw.rectangle([x0, yy, x0 + 12, yy + 12], fill=color)
            count = sum(1 for d in gen.hexes.values() if d.biome == name)
            label = f"{name} ({count})"
            draw.text((x0 + 18, yy - 1), label, fill=(220, 220, 220), font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a generated world to PNG.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=80)
    parser.add_argument("--config", type=Path, default=Path("config/worldgen.toml"))
    parser.add_argument("--layer", default="biome",
                        help="Layer name: biome, elevation, temperature, "
                             "precipitation, flow, composite, plates")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hex-px", type=float, default=6.0)
    parser.add_argument("--all", action="store_true",
                        help="Render every layer; --out is treated as a directory.")
    args = parser.parse_args()

    cfg = _load_worldgen_config(args.config)
    gen = run_pipeline(args.radius, cfg, args.seed)

    if args.all:
        args.out.mkdir(parents=True, exist_ok=True)
        n = 0
        for layer in ("elevation", "temperature", "precipitation",
                      "flow", "biome", "composite"):
            img = render(gen, layer, hex_px=args.hex_px)
            img.save(args.out / f"seed{args.seed}_r{args.radius}_{layer}.png")
            n += 1
        print(f"Wrote {n} PNGs to {args.out}")
    else:
        img = render(gen, args.layer, hex_px=args.hex_px)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        img.save(args.out)
        print(f"Wrote {args.out}")


def _load_worldgen_config(path: Path) -> WorldgenConfig:
    """Standalone WorldgenConfig loader — does not pull in the simulation engine."""
    return load_worldgen_config(path)


if __name__ == "__main__":
    main()
