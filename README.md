# kartogen

Deterministic, layered hex-grid world generator: from a `(config, seed)` pair it
produces per-hex elevation, sea/coast/lake/river flags, temperature, precipitation,
and biome. World size and every tunable live in the TOML config; output is a pure
function of `(config, seed)`.

## Setup

```bash
pip install -e .          # runtime
pip install -e .[dev]     # + pytest, mypy, ruff, py-spy
```

## Run

Generate, serialize, and render every layer to PNGs:

```bash
python -m kartogen --seed 42 --out exports/
python -m kartogen --seed 42 --out exports/ --config config/kartogen.toml
python -m kartogen --seed 42 --out exports/ --stop-after climate   # stop early
python -m kartogen --seed 42 --out exports/ -q                     # quiet
```

World dimensions come from `[kartogen.world] width_km, height_km` in the TOML —
there is no size flag; point `--config` at a different TOML to change the footprint.
Each run writes a fresh `exports/seed<seed>_<W>x<H>km_<timestamp>/` containing
`snapshot.json`, `layers/<name>.png`, and `tectonic_sim_views/` (plate
partition / crust / topography + drift GIFs).

From Python:

```python
from pathlib import Path
from kartogen import generate
from kartogen.config_loader import load_kartogen_config

cfg = load_kartogen_config(Path("config/kartogen.toml"))
world = generate(config=cfg, seed=42)
for hex_coord, data in world.hexes.items():
    print(hex_coord, data.biome, data.elevation, data.temperature_c)
```

## Test

```bash
pytest -m "not slow"   # fast unit tests (sub-second)
pytest                 # full suite, incl. end-to-end sims
```

See [CLAUDE.md](CLAUDE.md) for the design contract, pipeline, and architecture.
