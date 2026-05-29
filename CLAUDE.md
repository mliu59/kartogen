# CLAUDE.md — Worldgen

This file is the project's memory and design contract. Read it fully before making any changes. When in doubt, the principles here override your defaults.

## Project mission

`worldgen` is a deterministic, layered hex-grid world generator: from a `(radius, config, seed)` triple it produces a `GeneratedWorld` containing per-hex elevation, sea/coast/lake/river flags, temperature, precipitation, and biome — plus the intermediate layer outputs needed for testing and rendering.

A resource-and-crops layer existed previously (FAO-style envelopes + clustered Perlin deposits) but was removed pending a better approach.

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
├── plates.py          — L0a: t=0 plate seed placement + boundary classification
├── tectonics.py       — L0b: time-stepped PlaTec-style simulation
├── elevation.py       — L1: tectonic baseline + fBm/ridged detail + analytic mask
├── sea.py             — L2: ocean/coast mask
├── climate.py         — L3+L4: temperature + precipitation (wind sweep, orographic)
├── hydrology.py       — L5: priority-flood + D6 flow accumulation → rivers/lakes
├── ocean.py           — L2.5: gyre-based currents + continentality (Tier 2)
├── biome.py           — L6: Whittaker(T, P) + elevation/coast/water overrides
├── preview.py         — Library: PNG renderer (requires Pillow), used by export
└── export.py          — Public: WorldSnapshot container, serialize/save/load,
                         and export_world (snapshot + per-layer PNGs to a
                         timestamped folder)
```

Tests under `tests/` mirror the package; `conftest.py` exposes `default_worldgen_config`, `small_world` (radius 12, seed 42), and `medium_world` (radius 30, seed 42) session-scoped fixtures.

## World generation pipeline

Layer order — each layer is a pure function of all earlier layers' outputs plus its own seeded child RNG. Adding a new layer or a new entry within a layer never reshuffles existing seeds.

```
seed + config
  ↓ L0a plates         Voronoi plate seeding
  ↓ L0b tectonics      time-stepped sim (n_ticks × dt_myr of geological time)
  ↓ L1  elevation      tectonic baseline (km) + fBm/ridged detail
  ↓ L2  sea level      ocean/coast mask (absolute sea_level_km)
  ↓ L2.5 ocean         gyre-based currents + continentality (Tier 2)
  ↓ L3  temperature    latitude band + lapse + ocean-current anomaly
  ↓ L4  precipitation  prevailing-wind sweep + orographic uplift,
                        floor damped by continentality
  ↓ L5  hydrology      priority-flood + ε-tilt → D6 flow accum → rivers, lakes
  ↓ L6  biome          Whittaker(T, P) lookup + elevation/coast/water overrides
GeneratedWorld
```

## Ocean layer (Tier 2 climate)

`worldgen/ocean.py` runs between sea and climate, producing per-hex current
directions and temperature anomalies. Annual-mean snapshot — no seasonality.

**Gyres.** Each connected ocean basin is split by hemisphere into one or two
gyres (CW rotation in NH, CCW in SH — the Coriolis sign). Per ocean hex,
the current direction is the tangent of `(hex − gyre_centre)` rotated by
the rotation sign.

**Anomaly.** Single-pass formula: sample the planet's latitudinal
temperature `current_persistence_km` upstream along the current; anomaly =
`(upstream_temp − local_temp) × strength`, capped at `max_anomaly_c`. This
reproduces warm western-boundary currents (Gulf Stream / Kuroshio) and cold
eastern-boundary currents (California / Humboldt / Canary) without an
explicit advection solver.

**Coastal pickup.** BFS distance from every hex to the nearest ocean.
Coastal land hexes inherit `ocean_anomaly × pickup_fraction × exp(-d/decay)`
from the nearest ocean hex.

**Continentality.** The precipitation floor is multiplied by
`exp(-distance_to_ocean_km / continentality_dry_scale_km)`, drying deep
continental interiors. The existing upwind moisture sweep continues to
handle the dominant wind-driven drying — this term just damps the baseline
floor so a 1500-km-inland hex doesn't get a uniform "240 mm minimum carpet"
it shouldn't have.

**Tier 2 deferrals.** No seasonal cycle (no monsoons, no winter sea-ice
expansion), no pressure-field advection, no vegetation feedback. The
single-pass anomaly model produces visible streaks where currents reverse
between adjacent gyres; smoothing the field is a v2 task.

## Tectonics layer

When `mask_mode = "plates"`, the static Voronoi plate field becomes the t=0
initial condition for a time-stepped PlaTec-style simulation. Each plate
carries its own crust dictionary in plate-local hex coords; the plate's
centre drifts in continuous km each tick; crust moves with the plate. Plates
bounce off the world's circumscribed circle so they don't drift entirely
off-disc (the hex disc isn't a torus).

Per-tick: advance plate positions → compute per-world-hex overlap →
resolve collisions (subduction or continental folding) → seed fresh oceanic
crust where plates have pulled apart → age all crust. Optional erosion
(continental thickness blur) every `erosion_period` ticks.

After `n_ticks`, every world hex carries a `LithosphereColumn`
(`crust_type`, `thickness_km`, `age_myr`). Elevation is derived:

- Continental: `e_km = (thickness - reference) × continental_isostasy_factor`
- Oceanic (half-space cooling): `e_km = -ridge_depth - subsidence_rate × √age`

The configurable `sea_level_km` threshold separates land from ocean. In
`plates` mode this absolute value is the contract, not the analytic-mode
`land_fraction` quantile.

**Sea level is decoupled from crust dynamics.** `sea_level_km` is a
passive sampling threshold + the elevation-render colormap midpoint;
nothing in the collision / drift / divergent-fill / contact-constraint
pipeline reads it. Particles know their thickness, age, type, plate, and
position — not whether they're "above water." The isostasy module returns
signed elevation in km from a mantle reference, and sea level is just the
water line on that signed axis. The payoff is that two natural sweeps
come for free:

  - **Hold the world fixed, vary sea level.** Raising `sea_level_km` by
    +0.5 km instantly converts every continental hex sitting in
    `[reference, reference + 3.3 km]` of thickness from land to
    epicontinental sea — no sim rerun. Cretaceous-style high stands and
    glacial-maximum low stands are a config edit.
  - **Hold sea level fixed, vary tectonic parameters.** Plate dynamics
    alone reshape geography; the water line stays at the same physical
    reference so before/after maps are directly comparable.

Fold sea level into dynamics only when a feedback effect needs it —
erosion (Phase 7) is the first place that becomes physically meaningful
(weathering above the water line, deposition below it). None of the
current physics phases need it.

**v1 deferrals.** No Wilson-cycle re-seeding (plates drift on their initial
velocities for the full sim). No plate rotation. No continent merging. No
proper stream-power erosion. No hotspots / transform faults. Erosion is a
PlaTec-style continental thickness blur, not a real hydraulic model.

## Map latitude window

`hex_size_km` (the physical resolution) is **independent** of where the map
sits on the planet. `[worldgen.climate]` carries `map_lat_min` /
`map_lat_max` — the geographic latitudes the map's r-axis covers. The r-axis
convention is **north = negative r** (matches the renderer's top-of-image).

The planet's overall climate is anchored by `equator_temp_c` (at lat 0°) and
`polar_temp_c` (at ±90°); the map samples a slice of that gradient through
its lat window. Wind bands are Earth-like: trade easterlies inside ±30°,
westerlies 30°–60°, polar easterlies above 60°, with smoothstep transitions.

Defaults are `(-90, 90)` for pole-to-pole behaviour. For a temperate slice
(e.g. a Europe-shaped continent) try `map_lat_min = 30, map_lat_max = 60`.
The km extent of that map is whatever `hex_size_km × (2 × radius)` works out
to — you can simulate a 1000-km map spanning 1° or 60° of latitude.

**Physical-unit scaling.** Scale-dependent generator parameters are stored in physical units (km, km², mm/km of land fetch) and converted to per-hex units via `hex_size_km` at use time. Changing `hex_size_km` (default 5 km) automatically rescales noise frequency, wind reach, river thresholds, deposit feature wavelengths, and precipitation rates — the same physical world looks the same at any chosen hex resolution.

**Per-hex output (`HexData`).**

| Field | Source |
|---|---|
| `elevation`, `is_ocean`, `is_coast` | L1 + L2 |
| `temperature_c`, `precipitation_mm` | L3 + L4 |
| `is_river`, `is_lake`, `flow_accumulation` | L5 |
| `biome` | L6 (a name from `TERRAIN_NAMES`) |
| `plate_id`, `plate_type`, `nearest_boundary_type`, `distance_to_boundary_km` | L0 (or `None` when plates are off) |

## Export

Two public endpoints in `worldgen.export` (re-exported from the package root):

- `serialize_world(world) -> WorldSnapshot` — pure projection of a
  `GeneratedWorld` into a generic, JSON-friendly container. No side effects;
  no timestamp / seed injected here. Identical worlds → equal snapshots.
- `export_world(radius, config, seed, output_root) -> Path` — generates,
  serializes to `snapshot.json`, and renders one PNG per layer under
  `<output_root>/seed<seed>_r<radius>_<YYYYMMDD-HHMMSS>/layers/`. The
  per-export folder name carries the seed + radius + timestamp; the snapshot's
  `metadata` carries seed, timestamp, schema_version, mask_mode, hex_size_km.

`save_snapshot` / `load_snapshot` are JSON file I/O helpers; `WorldSnapshot`
itself is format-agnostic via `to_dict` / `from_dict`.

**`WorldSnapshot` is deliberately a generic data container.** All three fields
are open-ended `dict[str, Any]` / `list[dict[str, Any]]`:

| Field | Shape | Holds |
|---|---|---|
| `metadata` | `dict[str, Any]` | run parameters, schema version, anything caller-injected |
| `hexes` | `list[dict[str, Any]]` | one flat record per hex (q, r + every HexData field as primitives) |
| `layers` | `dict[str, dict[str, Any]]` | layer-name → layer-level (non-per-hex) data |

Adding a new per-hex field, a new intermediate layer, or new metadata keys
does **not** require changing `WorldSnapshot`'s schema — the new data slots
into the existing dicts. This is the extension point as the simulator grows.

## Export CLI

`python -m worldgen` is the canonical entry point — it generates a world,
serializes it to `snapshot.json`, and renders every available layer as a
PNG (plus a per-plate `plates/plate_NN.png` and a drift `drift.gif`).

```
python -m worldgen --seed 42 --radius 80 --out exports/
python -m worldgen --seed 42 --radius 80 --out exports/ --stop-after climate
python -m worldgen --seed 42 --radius 80 --out exports/ -q   # silence logs
```

`--stop-after STEP` halts the pipeline after the named step (one of
`PIPELINE_STEPS`); only the layers whose source data is populated are
rendered. Default logging is DEBUG (per-layer timings + progress bars);
`-q`/`--quiet` drops to WARNING. Requires the `[preview]` extra (Pillow).
`preview.py` is a pure library now — the only CLI is `python -m worldgen`.

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
- Premature abstraction. Don't build a plugin system for layers; we have a handful.
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
