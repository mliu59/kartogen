# CLAUDE.md — Worldgen

This file is the project's memory and design contract. Read it fully before making any changes. When in doubt, the principles here override your defaults.

## Project mission

`worldgen` is a deterministic, layered hex-grid world generator: from a `(radius, config, seed)` triple it produces a `GeneratedWorld` containing per-hex elevation, sea/coast/lake/river flags, temperature, precipitation, biome, crop suitability, and resource deposits — plus the intermediate layer outputs needed for testing and rendering.

It was extracted from a larger agent-based macro-history simulator so the terrain pipeline can be developed, tested, and previewed on its own. The package has no runtime dependencies; the optional `[preview]` extra adds Pillow for PNG rendering.

## Three commitments that override convenience

These are non-negotiable and drive most architectural decisions:

1. **Determinism / simulatability** — every output is a pure function of `(radius, config, seed)`. No `random.random()`, no `time.time()`, no unordered iteration over dicts that affects results. Each layer derives its child RNG by hashing the layer name into `RngHierarchy`, so adding a new layer or reordering layers within `pipeline.generate` never reshuffles existing seeds.

2. **Interpretability** — the full result is inspectable. `GeneratedWorld` exposes both per-hex `HexData` and every intermediate layer (`ElevationLayer`, `SeaLayer`, `ClimateLayer`, `HydrologyLayer`, `PlateField | None`). Anything that affects the final map can be examined directly; nothing is hidden behind aggregated outputs.

3. **Testability** — every deterministic function gets a unit test. Tests live under `tests/` and run without any heavy dependency (Pillow is only needed for `preview.py`, never imported by the layers themselves). Snapshot-style tests pin the final world hash for a fixed `(seed, radius, config)`; any change to outputs should be a conscious decision, not a surprise.

If a design choice trades against any of these three, flag it and ask before proceeding.

## Architectural principles

**Pure functions, layer by layer.** Each layer is `compute(prior_layer_outputs, config, rng_child) → LayerOutput`. Mutation is fine inside a layer; the layer's inputs are treated as immutable. `pipeline.generate` is the only orchestrator.

**Configuration over code.** Coefficients (noise frequencies, lapse rates, river thresholds, crop envelopes, deposit cluster parameters) live in `config/worldgen.toml`, not in code. The engine should be the same regardless of which world parameters are loaded. Adding a new crop or resource is a config-only change.

**No backwards compatibility.** v0 is pre-release. When changing a feature, *change it*. Don't add fallback paths, default values to paper over missing fields, optional flags toggling old vs new behavior, or compatibility shims. If old call sites break, fix them.

**Explicit data structures.** All state objects are `@dataclass(frozen=True)` in `worldgen/types.py`. No bare dicts for anything with a stable schema. No default values on dataclass fields whose meaning is non-trivial — missing-data bugs should surface at construction, not as silent zeros downstream.

**Module boundaries:**

```
worldgen/
├── hex.py             — Hex coordinate primitive (axial + spiral iterator)
├── noise.py           — PerlinNoise2D, fbm, ridged_fbm (pure, seeded)
├── terrain.py         — TERRAIN_NAMES (the canonical biome name tuple)
├── rng.py             — RngHierarchy (sha256-keyed child RNGs)
├── types.py           — WorldgenConfig + per-layer dataclasses + HexData
├── config_loader.py   — TOML → WorldgenConfig
├── pipeline.py        — orchestrator; returns GeneratedWorld
├── plates.py          — L0: optional tectonic plate field
├── elevation.py       — L1: fBm + ridged + domain warp + radial falloff
├── sea.py             — L2: quantile threshold + coast tag
├── climate.py         — L3+L4: temperature + precipitation (wind sweep, orographic)
├── hydrology.py       — L5: priority-flood + D6 flow accumulation → rivers/lakes
├── biome.py           — L6: Whittaker(T, P) + elevation/coast/water overrides
├── resources.py       — L7: crop suitability + clustered deposit noise
└── preview.py         — CLI: PNG renderer (requires Pillow)
```

Tests under `tests/` mirror the package; `conftest.py` exposes `default_worldgen_config`, `small_world` (radius 12, seed 42), and `medium_world` (radius 30, seed 42) session-scoped fixtures.

## World generation pipeline

Layer order — each layer is a pure function of all earlier layers' outputs plus its own seeded child RNG. Adding a new layer or a new entry within a layer never reshuffles existing seeds.

```
seed + config
  ↓ L0  plates        (optional) Voronoi plate field with type & boundary tags
  ↓ L1  elevation     fBm + ridged multifractal + domain warp + radial falloff
  ↓ L2  sea level     quantile threshold; coast tag
  ↓ L3  temperature   latitude band + elevation lapse + small noise
  ↓ L4  precipitation prevailing-wind moisture sweep with orographic uplift
  ↓ L5  hydrology     priority-flood + ε-tilt → D6 flow accum → rivers, lakes
  ↓ L6  biome         Whittaker(T, P) lookup + elevation/coast/water overrides
  ↓ L7  resources     crop suitability + resource deposits
GeneratedWorld
```

See [docs/terrain/TERRAIN_GENERATION.md](docs/terrain/TERRAIN_GENERATION.md) for the full methods writeup and cited sources.

**Physical-unit scaling.** Scale-dependent generator parameters are stored in physical units (km, km², mm/km of land fetch) and converted to per-hex units via `hex_size_km` at use time. Changing `hex_size_km` (default 5 km) automatically rescales noise frequency, wind reach, river thresholds, deposit feature wavelengths, and precipitation rates — the same physical world looks the same at any chosen hex resolution.

**Per-hex output (`HexData`).**

| Field | Source |
|---|---|
| `elevation`, `is_ocean`, `is_coast` | L1 + L2 |
| `temperature_c`, `precipitation_mm` | L3 + L4 |
| `is_river`, `is_lake`, `flow_accumulation` | L5 |
| `biome` | L6 (a name from `TERRAIN_NAMES`) |
| `crop_suitability: dict[crop_name, float]` | L7 (FAO-style trapezoidal envelopes) |
| `deposits: dict[resource_name, float]` | L7 (per-resource clustered Perlin) |
| `plate_id`, `plate_type`, `nearest_boundary_type`, `distance_to_boundary_km` | L0 (or `None` when plates are off) |

## Preview CLI

```
python -m worldgen.preview --seed 42 --radius 80 --layer biome --out world.png
python -m worldgen.preview --seed 42 --radius 80 --all --out out/
```

`--all` renders every standard layer plus one PNG per crop and per resource. Requires the `[preview]` extra (Pillow).

## Conventions

**Python.** 3.11+. Modern type hints (`list[int]`, `X | None`, etc.). Code should pass `mypy --strict` on `worldgen/`.

**Formatting & linting.** `ruff` for both. Config in `pyproject.toml`.

**Testing.** `pytest`. Tests in `tests/` mirror `worldgen/`. Use fixtures for common world setups.

**Naming.**
- `layer` = one pipeline stage; `LayerOutput` = its frozen dataclass result
- `hex` (lowercase) = a `Hex` coordinate; `hexes` = an iterable of them
- `gen` / `world` = a `GeneratedWorld`
- `cfg` = a `WorldgenConfig`

**RNG discipline.** Single root seed → `RngHierarchy(seed)`. Layers call `rng.child("layer_name")` (or a more specific tuple) to get a `random.Random`. Never call the root RNG directly. Never seed from `time.time()`.

**Errors.** Fail loudly. If a config is missing a required field, raise. No silent defaults.

## Anti-patterns to avoid

- Hidden randomness outside `RngHierarchy`.
- Magic numbers in code. Coefficients go in `config/worldgen.toml`.
- Backwards-compat shims when changing a feature. No fallback paths, no "if not configured, do the old thing" branches.
- Premature abstraction. Don't build a plugin system for layers; we have seven.
- Premature optimization. Clarity wins for v0. Profile before tuning.
- Layer-to-layer reaching past the explicit `pipeline.generate` wiring (e.g. importing `hydrology` from inside `climate`). New cross-layer information must flow through a typed layer output.

## Behavioral patterns to follow

### 1. Think Before Coding

Before implementing:
- State your assumptions. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so.

### 2. Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.

### 3. Surgical Changes

- Touch only what you must.
- Don't refactor adjacent code that isn't broken.
- Match existing style.

### 4. Goal-Driven Execution

- Define success criteria.
- For multi-step tasks, state the plan as steps with verification points.
