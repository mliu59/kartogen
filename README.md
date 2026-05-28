# worldgen

Deterministic, layered hex-grid world generator. Produces per-hex elevation, sea/coast/lake/river flags, temperature, precipitation, biomes, crop suitability, and resource deposits — all as a pure function of `(radius, config, seed)`.

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

## Preview

```bash
python -m worldgen.preview --seed 42 --radius 80 --layer biome --out world.png
python -m worldgen.preview --seed 42 --radius 80 --all --out out/
```

`--all` renders every standard layer plus one PNG per crop and per resource.

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
  ↓ L7  resources     crop suitability + clustered deposit noise
GeneratedWorld
```

See [docs/terrain/TERRAIN_GENERATION.md](docs/terrain/TERRAIN_GENERATION.md) for the methods writeup. See [CLAUDE.md](CLAUDE.md) for the design contract.

## Test

```bash
pytest
```
