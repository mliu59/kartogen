# worldgen

Deterministic, layered hex-grid world generator. Produces per-hex elevation, sea/coast/lake/river flags, temperature, precipitation, and biomes — all as a pure function of `(radius, config, seed)`.

Extracted from a larger agent-based macro-history simulator; lives on its own so the terrain pipeline can be developed, tested, and previewed in isolation.

## Install

```bash
pip install -e .
pip install -e .[preview]   # adds Pillow for the PNG renderer
pip install -e .[dev]       # pytest, mypy, ruff, Pillow
```

## Usage

```python
from worldgen import generate
from worldgen.config_loader import load_worldgen_config

cfg = load_worldgen_config("config/worldgen.toml")
world = generate(radius=80, config=cfg, seed=42)

for hex_coord, data in world.hexes.items():
    print(hex_coord, data.biome, data.elevation, data.temperature_c)
```

`world` also exposes the intermediate layers (`world.elevation`, `world.sea`, `world.climate`, `world.hydrology`, `world.plates`) for inspection and rendering.

## Export

Generate + serialize + render in one go:

```bash
python -m worldgen --seed 42 --radius 80 --out exports/
python -m worldgen --seed 42 --radius 80 --out exports/ --stop-after ocean
```

Produces `exports/seed42_r80_<YYYYMMDD-HHMMSS>/` containing:

- `snapshot.json` — the serialized world
- `layers/<name>.png` — one PNG per renderable layer (biome, elevation,
  temperature, precipitation, flow, composite, currents, wind,
  continentality, gyres, ocean_depth, plates, plates_t0)
- `plates/plate_NN.png` — each plate's final-state (post-warp) footprint
- `drift.gif` — animated drift of all plates over the simulated history

`--stop-after STEP` runs the pipeline up to and including the named step
(`plates`, `tectonics`, `elevation`, `sea`, `ocean`, `climate`,
`hydrology`, `biome`) and only writes PNGs for layers whose source data
is available. `-q` / `--quiet` silences the per-layer logs and progress
bars; otherwise DEBUG-level output is on by default.

From Python:

```python
from pathlib import Path
from worldgen import export_world, serialize_world, save_snapshot

# Full bundle (snapshot + PNGs):
folder = export_world(radius=80, config=cfg, seed=42, output_root=Path("exports"))

# Or just the serialization endpoint, no rendering:
snap = serialize_world(world)
save_snapshot(snap, Path("world.json"))
```

`WorldSnapshot` is a generic container — `metadata`, `hexes`, and `layers` are
all open-ended dicts, so new per-hex fields or new layers added later don't
require schema changes.

## Pipeline

```
seed + config
  ↓ L0  plates        (optional) Voronoi plate field
  ↓ L1  elevation     fBm + ridged + domain warp + radial falloff
  ↓ L2  sea           quantile threshold, coast tag
  ↓ L3  temperature   latitude band + elevation lapse
  ↓ L4  precipitation prevailing-wind moisture sweep + orographic uplift
  ↓ L5  hydrology     priority-flood → D6 flow accumulation → rivers, lakes
  ↓ L6  biome         Whittaker(T, P) + overrides
GeneratedWorld
```

See [CLAUDE.md](CLAUDE.md) for the design contract.

## Test

```bash
pytest
```
