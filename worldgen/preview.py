"""Headless terrain preview: rendering library for a ``GeneratedWorld``.

Pure library — no CLI; ``python -m worldgen`` (see ``export.py``) is the
canonical entry point for producing PNGs. Each layer is rendered from
intermediate state on ``GeneratedWorld`` so partial pipelines (set via
``stop_after``) still produce the layers that can be made.

Layers:
    plates_t0       t=0 Voronoi assignment (per-hex initial plate id).
    plates          Final post-warp plate ownership (per-hex final plate id,
                    drift-translated seed dots).
    elevation       Heightmap, blue-low-to-white-high.
    temperature     Temperature, blue-cold-to-red-hot.
    precipitation   Precipitation, tan-dry-to-darkgreen-wet.
    flow            Log-scaled flow accumulation (drainage map).
    biome           Final terrain classification.
    composite       Biome with hillshade + river overlay.
    currents        Ocean current temperature anomaly + direction arrows.
    wind            Per-hex wind direction over a zonal-band background.
    continentality  Distance to nearest ocean (coastal blue → inland tan).
    gyres           Categorical colouring per gyre id.
    ocean_depth     Bathymetry from the lithosphere column elevation_km.

``render_single_plate`` produces an isolated per-plate footprint; the export
hook writes one such PNG per plate into the ``plates/`` subfolder.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from worldgen.hex import Hex
from worldgen.world import rect_world_hexes, world_pixel_bounds

if TYPE_CHECKING:
    from worldgen.pipeline import GeneratedWorld


# Minimum pipeline step (from ``pipeline.PIPELINE_STEPS``) required for a
# render layer to have its source data populated. Used by ``export_world``
# to filter the render set when the pipeline stopped early. Layers not in
# this map are considered always-renderable (none currently).
LAYER_REQUIRES: dict[str, str] = {
    "plates_t0": "plates",
    "plates": "tectonics",
    "elevation": "elevation",
    "ocean_depth": "sea",
    "currents": "ocean",
    "continentality": "ocean",
    "gyres": "ocean",
    "temperature": "climate",
    "precipitation": "climate",
    "wind": "climate",
    "flow": "hydrology",
    "biome": "biome",
    "composite": "biome",
}


def _world_hexes(gen: "GeneratedWorld") -> list[Hex]:
    """All hexes in the rectangular world footprint, independent of per-hex
    assembly. Renderers iterate this instead of ``gen.hexes`` so they keep
    working when the pipeline stops short of the per-hex assembly step.
    """
    return rect_world_hexes(gen.config.world, gen.config.hex_size_km)


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


def _figure_size(gen: "GeneratedWorld", hex_px: float) -> tuple[int, int, float, float]:
    """Renderer canvas size (w, h) and centre (cx, cy) for the world's
    rectangular footprint. ``hex_px`` is the per-hex pixel scale; the
    canvas adds a margin so edge hexes aren't clipped by the border."""
    w, h = world_pixel_bounds(gen.config.world, hex_px, gen.config.hex_size_km)
    # Header band for the caption line, matched to the old layout.
    w += 40
    h += 40
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
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if layer == "plates":
        return _render_plates(gen, hex_px, show_legend)
    if layer == "plates_t0":
        return _render_plates_t0(gen, hex_px, show_legend)
    if layer == "currents":
        return _render_currents(gen, hex_px, show_legend)
    if layer == "wind":
        return _render_wind(gen, hex_px, show_legend)
    if layer == "continentality":
        return _render_continentality(gen, hex_px, show_legend)
    if layer == "gyres":
        return _render_gyres(gen, hex_px, show_legend)
    if layer == "ocean_depth":
        return _render_ocean_depth(gen, hex_px, show_legend)

    # Per-layer source data lookup. We read from intermediate state rather
    # than ``gen.hexes`` so renders work for partial pipelines (e.g.
    # ``stop_after="elevation"`` still produces the elevation PNG even
    # though no per-hex HexData was assembled).
    elev = gen.elevation
    sea = gen.sea
    clim = gen.climate
    hydro = gen.hydrology
    biomes = gen.biomes

    if layer == "flow":
        if hydro is None:
            raise ValueError("flow layer requires hydrology step")
        max_flow = max(hydro.flow_accumulation.values(), default=1)
    else:
        max_flow = 1

    for hex in _world_hexes(gen):
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)

        if layer == "biome":
            if biomes is None:
                raise ValueError("biome layer requires biome step")
            color = BIOME_COLORS.get(biomes[hex], (200, 0, 200))
        elif layer == "elevation":
            if elev is None:
                raise ValueError("elevation layer requires elevation step")
            color = _color_elevation(
                elev.elevation[hex] - elev.sea_level, sea_level=0.0,
            )
        elif layer == "temperature":
            if clim is None:
                raise ValueError("temperature layer requires climate step")
            # Same colormap for land and ocean — muting ocean produced
            # visually sharp discontinuities along coastlines and rift
            # valleys that read like data errors. The latitudinal gradient
            # is already legible because ocean and land share the same
            # temperature field.
            color = _color_temperature(clim.temperature_c[hex])
        elif layer == "precipitation":
            if clim is None or sea is None:
                raise ValueError("precipitation layer requires climate step")
            if sea.is_ocean[hex]:
                color = BIOME_COLORS["ocean"]
            else:
                color = _color_precipitation(clim.precipitation_mm[hex])
        elif layer == "flow":
            assert hydro is not None
            color = _color_flow(hydro.flow_accumulation[hex], max_flow)
        elif layer == "composite":
            if biomes is None:
                raise ValueError("composite layer requires biome step")
            # Biome + hillshade tinting + river overlay.
            base = BIOME_COLORS.get(biomes[hex], (200, 0, 200))
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
    """Color each hex by its **final** (post-tectonic-simulation) plate id.

    The colors come from the lithosphere state — i.e. which plate currently
    owns each world hex after ``n_ticks × dt_myr`` of simulated drift. The
    t=0 Voronoi seeding (which is what ``PlateField`` records) is **not**
    what's drawn here; rendering the t=0 state would not match the elevation
    field downstream of the tectonics simulation.

    Boundary hexes (between two different final plates) get a darker outline.
    Each plate's *current* seed-hex position (the initial seed translated by
    the simulated drift) is marked with a small white dot.
    """
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.lithosphere is None:
        raise ValueError("plates layer requires tectonics step")
    # Use the SIMULATED plates as source of truth — their ids match
    # ``final_pid`` (both come from the same tectonic_sim run). After
    # param-temperature randomization the simulated plate_count may
    # differ from worldgen's t=0 ``PlateField``, so using
    # ``gen.plates.plates`` would KeyError on extra ids.
    plate_by_id = {p.id: p for p in gen.lithosphere.plates}
    final_pid = gen.lithosphere.plate_id
    world_hexes = _world_hexes(gen)
    world_hex_set = set(world_hexes)

    for hex in world_hexes:
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        pid = final_pid.get(hex, -1)
        if pid < 0:
            color = (20, 20, 30)
        else:
            plate = plate_by_id.get(pid)
            # If the plate id was sampled from a particle that's since
            # been subducted between the snapshot and the sample (very
            # rare; defensive), fall back to continental colouring.
            is_oceanic = plate is not None and plate.initial_type == "oceanic"
            color = _plate_color(pid, is_oceanic)
        # Boundary = any in-bounds neighbor has a different final pid.
        on_boundary = any(
            nb in world_hex_set and final_pid.get(nb, pid) != pid
            for nb in hex.neighbors()
        )
        if on_boundary:
            outline = (15, 15, 20)
            width = max(1, int(hex_px / 6))
        else:
            outline = None  # type: ignore[assignment]
            width = 0
        draw.polygon(corners, fill=color, outline=outline, width=width)

    # Mark each plate's **current** centre (km-coords drifted from the t=0
    # seed hex over the geological simulation). Useful for confirming how
    # far each plate has moved.
    hex_size = gen.config.hex_size_km
    for tplate in gen.lithosphere.plates:
        px = cx + hex_px * tplate.center_km[0] / hex_size
        py = cy + hex_px * tplate.center_km[1] / hex_size
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
        # Iterate the simulated plate set — `TectonicPlate.initial_type`
        # rather than worldgen `Plate.type`.
        sim_plates = gen.lithosphere.plates
        n_continental = sum(1 for p in sim_plates if p.initial_type == "continental")
        n_oceanic = sum(1 for p in sim_plates if p.initial_type == "oceanic")
        caption = (
            f"plates (final state): {len(sim_plates)}   "
            f"continental: {n_continental}   oceanic: {n_oceanic}   "
            f"(white dots = current plate centres after drift)"
        )
        draw.text((10, 10), caption, fill=(230, 230, 230), font=font)

        # Per-plate id swatches stacked along the bottom-left so the user can
        # match the colors on the map to a numeric id.
        swatch = 14
        gap = 2
        col_h = (swatch + gap) * len(sim_plates)
        y0 = h - 10 - col_h
        for i, plate in enumerate(sim_plates):
            yy = y0 + i * (swatch + gap)
            sw_color = _plate_color(plate.id, plate.initial_type == "oceanic")
            draw.rectangle([10, yy, 10 + swatch, yy + swatch], fill=sw_color)
            label = f"#{plate.id} {plate.initial_type[:4]}"
            draw.text(
                (10 + swatch + 4, yy - 1), label,
                fill=(220, 220, 220), font=font,
            )

    return img


def _render_plates_t0(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Color each hex by its **initial** (t=0 Voronoi) plate id.

    The colors come from ``gen.plates.hex_to_plate`` — the Perlin-domain-warped
    Voronoi assignment seeded before any tectonic drift. Useful as a
    before/after pair with the final-state ``plates`` render: same seeds,
    same warp field, but no drift, no collisions, no rift fill.
    """
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    field = gen.plates
    plate_by_id = {p.id: p for p in field.plates}
    init_pid = field.hex_to_plate
    world_hexes = _world_hexes(gen)
    world_hex_set = set(world_hexes)

    for hex in world_hexes:
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        pid = init_pid.get(hex, -1)
        if pid < 0:
            color = (20, 20, 30)
        else:
            plate = plate_by_id[pid]
            color = _plate_color(pid, plate.type == "oceanic")
        on_boundary = any(
            nb in world_hex_set and init_pid.get(nb, pid) != pid
            for nb in hex.neighbors()
        )
        if on_boundary:
            outline = (15, 15, 20)
            width = max(1, int(hex_px / 6))
        else:
            outline = None  # type: ignore[assignment]
            width = 0
        draw.polygon(corners, fill=color, outline=outline, width=width)

    # Mark each plate's t=0 seed hex.
    hex_size = gen.config.hex_size_km
    for plate in field.plates:
        sx, sy = _hex_to_pixel(plate.seed_hex.q, plate.seed_hex.r, hex_px, cx, cy)
        r = max(2.0, hex_px * 0.35)
        draw.ellipse(
            [(sx - r, sy - r), (sx + r, sy + r)],
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
            f"plates (t=0 Voronoi): {len(field.plates)}   "
            f"continental: {n_continental}   oceanic: {n_oceanic}   "
            f"(white dots = seed hexes)"
        )
        draw.text((10, 10), caption, fill=(230, 230, 230), font=font)
    return img


def render_single_plate(
    gen: GeneratedWorld,
    plate_id: int,
    hex_px: float = 6.0,
    margin_hexes: float = 1.5,
) -> Image.Image:
    """Render one plate's intrinsic geometry, cropped to its own bounding box.

    The output canvas is sized to fit just this plate's hexes (with
    ``margin_hexes`` of padding on every side); plate position in the world
    is irrelevant. ``hex_px`` is the per-hex pixel scale, held constant
    across plates so side-by-side comparison reveals true relative size.

    The plate's footprint is the set of final-state world hexes assigned
    to ``plate_id``; each hex carries the crust type it ended up with
    after the sim. Continental hexes are coloured in the plate's palette
    colour; oceanic hexes (rift / fresh oceanic crust acquired during
    the sim) use the darker oceanic variant.
    """
    if gen.lithosphere is None:
        raise ValueError("render_single_plate requires the tectonics step")

    # Source plate metadata from the *simulated* plate set — under
    # param_temperature randomization the simulated plate count can
    # diverge from worldgen's t=0 ``PlateField``.
    plate_by_id = {p.id: p for p in gen.lithosphere.plates}
    if plate_id not in plate_by_id:
        raise ValueError(f"plate {plate_id} not in this world")
    plate = plate_by_id[plate_id]
    cont_color = _plate_color(plate_id, False)
    ocn_color = _plate_color(plate_id, True)

    # Collect (axial q, r, crust_type) tuples for the focal plate: world
    # hexes whose final assignment is this plate, recentered so the
    # bounding-box centre lands at the canvas centre.
    cells: list[tuple[int, int, str]] = [
        (h.q, h.r, gen.lithosphere.columns[h].crust_type)
        for h in _world_hexes(gen)
        if gen.lithosphere.plate_id.get(h) == plate_id
    ]
    source_note = "final world-state"

    if not cells:
        # Edge case: a plate with no owned hexes (fully consumed by
        # subduction, or never assigned). Emit a small placeholder so
        # downstream code doesn't crash on a zero-size image.
        img = Image.new("RGB", (120, 40), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        draw.text((6, 12), f"plate #{plate_id}: 0 hexes",
                  fill=(220, 220, 220), font=font)
        return img

    # Project axial → flat cartesian (in "hex units": 1 hex ≈ 1 unit wide).
    def to_xy(q: int, r: int) -> tuple[float, float]:
        return 1.5 * q, math.sqrt(3.0) * (r + q / 2.0)

    xs_ys = [to_xy(q, r) for q, r, _ in cells]
    xs = [xy[0] for xy in xs_ys]
    ys = [xy[1] for xy in xs_ys]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Canvas: bbox plus margin, in pixels.
    w = int(round((max_x - min_x) * hex_px + 2 * margin_hexes * hex_px + 2 * hex_px))
    h = int(round((max_y - min_y) * hex_px + 2 * margin_hexes * hex_px + 2 * hex_px))
    # Leave headroom for the caption strip.
    caption_h = 18
    img = Image.new("RGB", (w, h + caption_h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Shift origin so the plate sits with margin_hexes padding on the left
    # and below the caption strip on the top.
    ox = -min_x * hex_px + margin_hexes * hex_px + hex_px
    oy = -min_y * hex_px + margin_hexes * hex_px + hex_px + caption_h

    cont_count = 0
    ocn_count = 0
    for (q, r, ct), (lx, ly) in zip(cells, xs_ys):
        px = ox + lx * hex_px
        py = oy + ly * hex_px
        if ct == "oceanic":
            color = ocn_color
            ocn_count += 1
        else:
            color = cont_color
            cont_count += 1
        draw.polygon(_hex_corners(px, py, hex_px), fill=color)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    # Bounding-box dimensions in km, for size context.
    hex_size = gen.config.hex_size_km
    bbox_w_km = (max_x - min_x + 1.0) * hex_size
    bbox_h_km = (max_y - min_y + 1.0) * hex_size
    caption = (
        f"plate #{plate_id} {plate.initial_type}   "
        f"hexes: {len(cells)} (cont {cont_count} / ocn {ocn_count})   "
        f"bbox: {bbox_w_km:.0f}×{bbox_h_km:.0f} km   "
        f"[{source_note}]"
    )
    draw.text((6, 3), caption, fill=(230, 230, 230), font=font)
    return img


def _hillshade_tint(h: Hex, gen: GeneratedWorld) -> int:
    """Compute a -40..+40 brightness adjustment from slope toward NW light."""
    if gen.elevation is None or gen.sea is None:
        return 0
    if h not in gen.elevation.elevation:
        return 0
    elev_here = gen.elevation.elevation[h] - gen.elevation.sea_level
    if gen.sea.is_ocean.get(h, False):
        return 0
    # NW direction in axial space ≈ (-1, 0) on flat-top; the canonical NW
    # neighbor is direction (0, -1) per the AXIAL_DIRECTIONS table.
    nw = Hex(h.q + 0, h.r - 1)
    if nw not in gen.elevation.elevation:
        return 0
    elev_nw = gen.elevation.elevation[nw] - gen.elevation.sea_level
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
    if hydro is None:
        return
    world_hex_set = set(_world_hexes(gen))
    for h in _world_hexes(gen):
        if not hydro.is_river.get(h, False):
            continue
        down = hydro.downstream.get(h)
        if down is None or down not in world_hex_set:
            continue
        px1, py1 = _hex_to_pixel(h.q, h.r, hex_px, cx, cy)
        px2, py2 = _hex_to_pixel(down.q, down.r, hex_px, cx, cy)
        # Width scales with log flow.
        flow = hydro.flow_accumulation.get(h, 0)
        width = max(1.0, min(3.5, 0.4 + 0.6 * math.log10(max(2, flow))))
        draw.line([(px1, py1), (px2, py2)], fill=(60, 130, 200), width=int(round(width)))


def _color_ocean_anomaly(anomaly_c: float, max_abs: float) -> tuple[int, int, int]:
    """Blue = cold (negative), neutral = 0, red = warm (positive)."""
    x = max(-1.0, min(1.0, anomaly_c / max_abs)) if max_abs > 0 else 0.0
    if x >= 0:
        # Warm — interpolate from a neutral teal to a warm orange-red.
        return (int(60 + 195 * x), int(110 - 50 * x), int(140 - 110 * x))
    x = -x
    return (int(50 + 30 * x), int(110 + 90 * x), int(180 + 50 * x))


def _render_currents(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Color ocean hexes by current temperature anomaly + draw direction arrows.

    Land hexes are drawn as a faded biome background. Ocean hexes are tinted
    by the warm/cold current anomaly; a short arrow shows the current's
    cartesian direction at each ocean hex.
    """
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.ocean is None or gen.sea is None:
        raise ValueError("currents layer requires ocean step")
    sea = gen.sea
    ocean = gen.ocean
    biomes = gen.biomes

    anomalies = [
        ocean.current_temp_anomaly.get(hh, 0.0)
        for hh in _world_hexes(gen)
        if sea.is_ocean.get(hh, False)
    ]
    max_abs = max((abs(a) for a in anomalies), default=1.0)
    if max_abs == 0:
        max_abs = 1.0

    for hex in _world_hexes(gen):
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        if sea.is_ocean.get(hex, False):
            color = _color_ocean_anomaly(
                ocean.current_temp_anomaly.get(hex, 0.0), max_abs,
            )
        else:
            # Faded biome background so land context is legible. Falls back
            # to a neutral grey when the biome layer hasn't run yet.
            if biomes is not None:
                base = BIOME_COLORS.get(biomes[hex], (180, 180, 180))
            else:
                base = (180, 180, 180)
            color = tuple(int(c * 0.40 + 35) for c in base)  # type: ignore[assignment]
        draw.polygon(corners, fill=color)

    # Draw current direction arrows on a sparse sample of ocean hexes.
    arrow_len = hex_px * 1.6
    for hex in _world_hexes(gen):
        if not sea.is_ocean.get(hex, False):
            continue
        # Render arrows on roughly every Nth hex to keep the field readable.
        if (hex.q + hex.r) % 3 != 0:
            continue
        direction = ocean.current_direction.get(hex, (0.0, 0.0))
        dx, dy = direction
        if dx == 0.0 and dy == 0.0:
            continue
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        # Cartesian projection matches the rest of the pipeline (+y = down).
        x2 = px + dx * arrow_len
        y2 = py + dy * arrow_len
        draw.line([(px, py), (x2, y2)], fill=(245, 245, 245), width=1)
        # Tiny arrow head: dot at the tip.
        draw.ellipse(
            [x2 - 1.2, y2 - 1.2, x2 + 1.2, y2 + 1.2],
            fill=(245, 245, 245),
        )

    if show_legend:
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        caption = (
            f"currents: {ocean.n_gyres} gyres   "
            f"|anomaly| peak: {max_abs:.1f} °C   "
            f"red=warm cold=blue"
        )
        draw.text((10, 10), caption, fill=(240, 240, 240), font=font)
    return img


def _render_wind(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Per-hex wind direction overlaid on a zonal-band background.

    Background colour encodes the east-west sign of the latitudinal wind
    band — orange for easterlies (trade easterlies in the tropics, polar
    easterlies above 60°), blue for westerlies (mid-latitude band) — so the
    Hadley/Ferrel/Polar three-cell structure is legible at a glance.

    Arrows on a sparse sample of hexes show the actual per-hex wind
    direction, including sea-breeze + Perlin jitter perturbations.
    """
    from worldgen.climate import (
        _WIND_BAND_HI_DEG,
        _WIND_BAND_LO_DEG,
        _zonal_wind_sign,
        hex_latitude_deg,
    )
    from worldgen.world import map_half_extents_km

    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.climate is None:
        raise ValueError("wind layer requires climate step")
    clim = gen.climate

    world_hexes = _world_hexes(gen)
    _half_w, half_h_km = map_half_extents_km(world_hexes, gen.config.hex_size_km)

    for hex in world_hexes:
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        lat_deg = hex_latitude_deg(hex, half_h_km, gen.config)
        sign = _zonal_wind_sign(abs(lat_deg))
        if sign > 0.5:
            # Westerlies — cool blue.
            color = (90, 150, 200)
        elif sign < -0.5:
            # Easterlies — warm tan.
            color = (210, 160, 90)
        else:
            # Transition zone — neutral grey.
            color = (170, 170, 170)
        draw.polygon(corners, fill=color)

    # Sparse arrows so the field stays legible.
    arrow_len = hex_px * 1.4
    for hex in world_hexes:
        if (hex.q + hex.r) % 3 != 0:
            continue
        dx, dy = clim.wind_direction.get(hex, (0.0, 0.0))
        if dx == 0.0 and dy == 0.0:
            continue
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        x2 = px + dx * arrow_len
        y2 = py + dy * arrow_len
        draw.line([(px, py), (x2, y2)], fill=(245, 245, 245), width=1)
        draw.ellipse(
            [x2 - 1.2, y2 - 1.2, x2 + 1.2, y2 + 1.2],
            fill=(245, 245, 245),
        )

    if show_legend:
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        caption = (
            f"wind   bands: easterlies <{_WIND_BAND_LO_DEG:.0f}° "
            f"westerlies {_WIND_BAND_LO_DEG:.0f}–{_WIND_BAND_HI_DEG:.0f}° "
            f"polar easterlies >{_WIND_BAND_HI_DEG:.0f}°"
        )
        draw.text((10, 10), caption, fill=(240, 240, 240), font=font)
    return img


def _render_continentality(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Heat map of distance to the nearest ocean. Coastal blue → inland tan."""
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.ocean is None or gen.sea is None:
        raise ValueError("continentality layer requires ocean step")
    ocean = gen.ocean
    sea = gen.sea

    max_d = max(
        ocean.distance_to_ocean_km.values(),
        default=1.0,
    )
    # Stretch the colour ramp to the actual data range so contrast within
    # the world is preserved. (On very tiny worlds with no land or a single
    # ring of coast, max_d → 0; fall back to 1 km to avoid div-by-zero.)
    ramp_scale = max(max_d, 1.0)

    for hex in _world_hexes(gen):
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        t = min(1.0, ocean.distance_to_ocean_km.get(hex, 0.0) / ramp_scale)
        if t < 0.5:
            # Sea-blue → green
            s = t / 0.5
            color = (
                int(70 + 100 * s), int(140 + 50 * s), int(180 - 80 * s),
            )
        else:
            # Green → tan → brown (drier)
            s = (t - 0.5) / 0.5
            color = (
                int(170 + 50 * s), int(190 - 90 * s), int(100 - 60 * s),
            )
        draw.polygon(corners, fill=color)

    if show_legend:
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        max_inland = max(
            (
                ocean.distance_to_ocean_km.get(hh, 0.0)
                for hh in _world_hexes(gen)
                if not sea.is_ocean.get(hh, False)
            ),
            default=0.0,
        )
        caption = (
            f"continentality   coast=blue interior=tan   "
            f"max inland: {max_inland:.0f} km   "
            f"dry-scale: {gen.config.ocean.continentality_dry_scale_km:.0f} km"
        )
        draw.text((10, 10), caption, fill=(240, 240, 240), font=font)
    return img


def _render_gyres(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Categorical colour per gyre id; land drawn as faded biome background."""
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.ocean is None or gen.sea is None:
        raise ValueError("gyres layer requires ocean step")
    ocean = gen.ocean
    sea = gen.sea
    biomes = gen.biomes

    # Reuse the plate palette; it's well-spaced and we typically have ≤10 gyres.
    for hex in _world_hexes(gen):
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        is_ocean_hex = sea.is_ocean.get(hex, False)
        gyre_id = ocean.gyre_id.get(hex)
        if is_ocean_hex and gyre_id is not None:
            color = _PLATE_PALETTE[gyre_id % len(_PLATE_PALETTE)]
        else:
            # Faded biome background (or neutral grey if biome step hasn't run).
            if biomes is not None:
                base = BIOME_COLORS.get(biomes[hex], (180, 180, 180))
            else:
                base = (180, 180, 180)
            color = tuple(int(c * 0.40 + 30) for c in base)  # type: ignore[assignment]
        draw.polygon(corners, fill=color)

    if show_legend:
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        n_gyres = ocean.n_gyres
        caption = f"gyres   total: {n_gyres}   (colour by gyre id)"
        draw.text((10, 10), caption, fill=(240, 240, 240), font=font)
    return img


def _color_ocean_depth(depth_km: float, max_depth: float) -> tuple[int, int, int]:
    """Light teal at the shelves → near-black at the abyss."""
    t = min(1.0, depth_km / max(max_depth, 0.001))
    # Light-cyan (shallow) → deep navy (deep).
    return (
        int(120 * (1 - t) + 10 * t),
        int(180 * (1 - t) + 30 * t),
        int(210 * (1 - t) + 65 * t),
    )


def _render_ocean_depth(
    gen: GeneratedWorld,
    hex_px: float,
    show_legend: bool,
) -> Image.Image:
    """Bathymetric map: ocean hexes coloured by depth below sea level.

    Uses ``lithosphere.elevation_km`` directly (signed; negative = below sea
    level). Land hexes are flat grey so the ocean structure stands out.
    """
    w, h, cx, cy = _figure_size(gen, hex_px)
    img = Image.new("RGB", (w, h), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if gen.sea is None or gen.lithosphere is None:
        raise ValueError("ocean_depth layer requires sea step")
    sea = gen.sea
    litho = gen.lithosphere
    sea_km = gen.config.tectonics.sea_level_km
    max_depth = 0.0
    for hex_ in _world_hexes(gen):
        if not sea.is_ocean.get(hex_, False):
            continue
        depth = sea_km - litho.elevation_km[hex_]
        if depth > max_depth:
            max_depth = depth
    if max_depth == 0:
        max_depth = 1.0

    for hex in _world_hexes(gen):
        px, py = _hex_to_pixel(hex.q, hex.r, hex_px, cx, cy)
        corners = _hex_corners(px, py, hex_px)
        if sea.is_ocean.get(hex, False):
            depth = sea_km - litho.elevation_km[hex]
            color = _color_ocean_depth(max(0.0, depth), max_depth)
        else:
            color = (90, 85, 80)
        draw.polygon(corners, fill=color)

    if show_legend:
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        caption = (
            f"ocean depth   max: {max_depth:.2f} km below sea level   "
            f"shallow=cyan abyss=navy"
        )
        draw.text((10, 10), caption, fill=(240, 240, 240), font=font)
    return img


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

    hex_count = len(_world_hexes(gen))
    caption = f"layer={layer}  {gen.config.world.width_km:g}x{gen.config.world.height_km:g} km  hexes={hex_count}"
    draw.text((10, 10), caption, fill=(220, 220, 220), font=font)

    if layer in ("biome", "composite") and gen.biomes is not None:
        # Bottom-left biome key.
        x0, y0 = 10, h - 10 - 16 * len(BIOME_COLORS)
        for i, (name, color) in enumerate(BIOME_COLORS.items()):
            yy = y0 + i * 16
            draw.rectangle([x0, yy, x0 + 12, yy + 12], fill=color)
            count = sum(1 for b in gen.biomes.values() if b == name)
            label = f"{name} ({count})"
            draw.text((x0 + 18, yy - 1), label, fill=(220, 220, 220), font=font)
