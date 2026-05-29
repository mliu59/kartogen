"""Public export endpoints: serialize a generated world and bundle a snapshot
plus per-layer rendered PNGs into a timestamped folder.

The serialized format is intentionally a *generic container*: every field on
``WorldSnapshot`` is an open-ended ``dict[str, Any]`` or ``list[dict[str, Any]]``
keyed by strings, with no fixed schema. Adding new per-hex fields, new
intermediate layers, or new metadata never requires changing ``WorldSnapshot``
itself — the new data slots in as another key.

Two public endpoints:

  - ``serialize_world(world) -> WorldSnapshot``
        Pure projection from ``GeneratedWorld`` into the generic container.
        No side effects, no timestamps, no I/O.

  - ``export_world(config, seed, output_root, stop_after=None) -> Path``
        Full bundle: generates the world (up to ``stop_after`` if set),
        serializes, writes JSON to disk, and renders one PNG per generation
        layer whose source data is available. World dimensions come from
        ``config.world``.

``save_snapshot`` / ``load_snapshot`` are the file I/O helpers; ``WorldSnapshot``
itself is format-agnostic (``to_dict`` / ``from_dict``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from worldgen._log import configure_logging
from worldgen.config_loader import load_worldgen_config
from worldgen.hex import Hex
from worldgen.pipeline import GeneratedWorld, PIPELINE_STEPS, generate
from worldgen.types import WorldgenConfig


@dataclass(frozen=True)
class WorldSnapshot:
    """Generic, serialization-friendly container for a generated world.

    Three open-ended bags:

      - ``metadata``: run parameters (world_width_km, world_height_km,
        hex_size_km, plus any extras layered on by the caller, e.g. seed
        and timestamp).
      - ``hexes``: list of per-hex records. Each record is a flat ``dict`` with
        ``q``, ``r``, and every ``HexData`` field as primitive JSON values.
      - ``layers``: layer-name → layer-level (non-per-hex) data dict.

    All three are ``dict[str, Any]`` / ``list[dict[str, Any]]`` so future
    additions (new per-hex fields, new intermediate layers, new metadata) do
    not require schema changes here.
    """

    metadata: dict[str, Any]
    hexes: list[dict[str, Any]]
    layers: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "hexes": self.hexes,
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldSnapshot:
        return cls(
            metadata=d["metadata"],
            hexes=d["hexes"],
            layers=d["layers"],
        )


def _to_primitive(v: Any) -> Any:
    """Recursively convert a value into JSON-friendly primitives.

    ``Hex`` becomes ``{"q": q, "r": r}``. Dataclass instances are flattened
    field-by-field. Tuples become lists. Dict keys become strings.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Hex):
        return {"q": v.q, "r": v.r}
    if is_dataclass(v) and not isinstance(v, type):
        return {f.name: _to_primitive(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, (list, tuple)):
        return [_to_primitive(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_primitive(val) for k, val in v.items()}
    return v  # leave as-is; json.dumps will raise if not encodable


def serialize_world(world: GeneratedWorld) -> WorldSnapshot:
    """Pure projection of a ``GeneratedWorld`` into a ``WorldSnapshot``.

    The result is deterministic: identical worlds serialize to identical
    snapshots. Runtime context like seed or timestamp is *not* set here —
    the caller (e.g. ``export_world``) layers that into ``metadata``.

    For partial pipelines (``world.stop_after`` < ``"biome"``), the per-hex
    record list is empty and only those ``layers`` whose source state is
    populated are emitted.
    """
    hex_records: list[dict[str, Any]] = []
    if world.hexes is not None:
        for h, data in world.hexes.items():
            rec: dict[str, Any] = {"q": h.q, "r": h.r}
            for f in fields(data):
                rec[f.name] = _to_primitive(getattr(data, f.name))
            hex_records.append(rec)

    layers: dict[str, dict[str, Any]] = {}
    if world.elevation is not None:
        layers["elevation"] = {"sea_level": world.elevation.sea_level}
    if world.plates is not None:
        layers["plates"] = {
            "count": len(world.plates.plates),
            "plates": [_to_primitive(p) for p in world.plates.plates],
        }

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "world_width_km": world.config.world.width_km,
        "world_height_km": world.config.world.height_km,
        "hex_count": len(world.hexes) if world.hexes is not None else 0,
        "hex_size_km": world.config.hex_size_km,
        "stop_after": world.stop_after,
        "tectonics": {
            "n_ticks": world.config.tectonics.n_ticks,
            "dt_myr": world.config.tectonics.dt_myr,
            "sea_level_km": world.config.tectonics.sea_level_km,
        },
    }

    return WorldSnapshot(metadata=metadata, hexes=hex_records, layers=layers)


def save_snapshot(snap: WorldSnapshot, path: Path) -> None:
    """Write a snapshot to ``path`` as JSON. Creates parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snap.to_dict(), indent=2, sort_keys=False),
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> WorldSnapshot:
    """Load a snapshot from a JSON file written by ``save_snapshot``."""
    return WorldSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))


# Standard renderable layers (in order). ``plates`` is appended automatically
# when ``world.plates is not None`` so non-plates worlds don't get a
# placeholder image.
DEFAULT_RENDER_LAYERS: tuple[str, ...] = (
    "elevation",
    "temperature",
    "precipitation",
    "flow",
    "biome",
    "composite",
    "currents",
    "wind",
    "continentality",
    "gyres",
    "ocean_depth",
    "plates_t0",
)


def _layers_available_at(stop_after: str, candidates: tuple[str, ...]) -> list[str]:
    """Filter render-layer names to those whose required pipeline step has run.

    ``LAYER_REQUIRES`` (in ``preview``) maps each layer to its minimum
    required step. A layer is kept when the index of its required step in
    ``PIPELINE_STEPS`` is ≤ the index of ``stop_after``.
    """
    from worldgen.preview import LAYER_REQUIRES  # lazy: avoids Pillow import here.

    stop_ix = PIPELINE_STEPS.index(stop_after)
    out: list[str] = []
    for layer in candidates:
        required = LAYER_REQUIRES.get(layer)
        if required is None or PIPELINE_STEPS.index(required) <= stop_ix:
            out.append(layer)
    return out


def export_world(
    config: WorldgenConfig,
    seed: int,
    output_root: Path,
    render_layers: tuple[str, ...] = DEFAULT_RENDER_LAYERS,
    hex_px: float = 6.0,
    stop_after: str | None = None,
) -> Path:
    """Generate a world, save its snapshot, render every available layer to PNG.

    Folder name is ``seed<seed>_<W>x<H>km_<YYYYMMDD-HHMMSS>`` under
    ``output_root``. World dimensions come from ``config.world``. Returns
    the path to the created folder.

    ``stop_after`` runs the pipeline up to (and including) the named step
    (see ``PIPELINE_STEPS``) and skips PNGs for layers whose source data
    isn't populated. Defaults to running the full pipeline.

    Layout::

        <folder>/
          snapshot.json
          layers/
            elevation.png
            temperature.png
            ...
          plates/
            plate_NN.png
          drift.gif

    Requires Pillow (the ``[preview]`` extra).
    """
    from worldgen import preview  # lazy: Pillow is only needed for rendering.

    world = generate(config=config, seed=seed, stop_after=stop_after)
    snap = serialize_world(world)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    snap.metadata["seed"] = seed
    snap.metadata["timestamp"] = timestamp

    w_km = int(round(config.world.width_km))
    h_km = int(round(config.world.height_km))
    folder = output_root / f"seed{seed}_{w_km}x{h_km}km_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    save_snapshot(snap, folder / "snapshot.json")

    layers_dir = folder / "layers"
    layers_dir.mkdir(exist_ok=True)

    # "plates" is auto-appended when the tectonics step ran.
    layers_to_render: list[str] = list(render_layers)
    if world.lithosphere is not None and "plates" not in layers_to_render:
        layers_to_render.append("plates")
    layers_to_render = _layers_available_at(world.stop_after, tuple(layers_to_render))

    for layer in layers_to_render:
        img = preview.render(world, layer, hex_px=hex_px)
        img.save(layers_dir / f"{layer}.png")

    # Per-plate final-state footprints (post-warp). One PNG per plate goes
    # into a ``plates/`` subfolder so the layers/ listing stays clean.
    if world.lithosphere is not None:
        plates_dir = folder / "plates"
        plates_dir.mkdir(exist_ok=True)
        for plate in world.plates.plates:
            img = preview.render_single_plate(world, plate.id, hex_px=hex_px)
            img.save(plates_dir / f"plate_{plate.id:02d}.png")

    # Drift animation: only written when the tectonic sim captured snapshots.
    if world.lithosphere is not None and world.lithosphere.history:
        n = preview.render_drift_animation(
            world, folder / "drift.gif", hex_px=max(2.0, hex_px * 0.7),
        )
        snap.metadata["drift_frames"] = n

    return folder


def main() -> None:
    """CLI entry point. Invoked by ``python -m worldgen``."""
    parser = argparse.ArgumentParser(
        description="Generate a world and export snapshot + per-layer PNGs.",
        prog="python -m worldgen",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/worldgen.toml"))
    parser.add_argument("--out", type=Path, default=Path("exports"))
    parser.add_argument("--hex-px", type=float, default=6.0)
    parser.add_argument(
        "--stop-after", choices=PIPELINE_STEPS, default=None,
        help=(
            "Stop the pipeline after this step (default: run all). "
            "Layers downstream of the stop point are skipped, and only PNGs "
            "whose source data is available are rendered."
        ),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Silence all worldgen logs and progress bars.",
    )
    args = parser.parse_args()

    if args.quiet:
        configure_logging(logging.WARNING)
    else:
        # Verbose (DEBUG) is the default — surfaces per-layer timing,
        # progress bars, and chatty per-hex details.
        configure_logging(logging.DEBUG)

    cfg = load_worldgen_config(args.config)
    folder = export_world(
        config=cfg,
        seed=args.seed,
        output_root=args.out,
        hex_px=args.hex_px,
        stop_after=args.stop_after,
    )
    print(f"Exported to {folder}")
