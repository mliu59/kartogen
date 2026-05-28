"""Tests for the resources layer (crops + deposits).

Covers:
 - trapezoidal envelope math
 - crop suitability gates (biome, elevation, hard climate bounds, irrigation)
 - per-resource biome / elevation / climate eligibility
 - deposit-noise determinism and approximate-quantile abundance enforcement
 - integration into the pipeline (HexData carries non-empty crop & deposit dicts)
"""

from __future__ import annotations

from worldgen.pipeline import GeneratedWorld
from worldgen.resources import _trapezoid, crop_suitability
from worldgen.types import CropDefinition


def test_trapezoid_zero_outside_bounds() -> None:
    assert _trapezoid(-1.0, 0.0, 10.0, 20.0, 30.0) == 0.0
    assert _trapezoid(31.0, 0.0, 10.0, 20.0, 30.0) == 0.0


def test_trapezoid_one_inside_plateau() -> None:
    for x in (10.0, 12.0, 15.0, 20.0):
        assert _trapezoid(x, 0.0, 10.0, 20.0, 30.0) == 1.0


def test_trapezoid_linear_ramps() -> None:
    # Rising ramp 0..10 — midpoint 5.0 → 0.5
    assert abs(_trapezoid(5.0, 0.0, 10.0, 20.0, 30.0) - 0.5) < 1e-9
    # Falling ramp 20..30 — midpoint 25.0 → 0.5
    assert abs(_trapezoid(25.0, 0.0, 10.0, 20.0, 30.0) - 0.5) < 1e-9


def _wheat() -> CropDefinition:
    return CropDefinition(
        name="wheat",
        temp_abs_min=0.0, temp_opt_min=15.0, temp_opt_max=20.0, temp_abs_max=30.0,
        precip_abs_min=250.0, precip_opt_min=500.0, precip_opt_max=900.0,
        precip_abs_max=1600.0,
        elev_max=0.5,
        biome_compatibility={"plains": 1.0, "hills": 0.8, "desert": 0.1},
        river_bonus=0.1, river_adjacent_bonus=0.05, coast_bonus=0.0,
        irrigation_replaces_rain_mm=0.0,
    )


def test_crop_suitability_excludes_unlisted_biomes() -> None:
    c = _wheat()
    s = crop_suitability(
        c, biome="jungle",
        temperature_c=18.0, precipitation_mm=700.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    assert s == 0.0


def test_crop_suitability_excludes_high_elevation() -> None:
    c = _wheat()
    s = crop_suitability(
        c, biome="hills",
        temperature_c=18.0, precipitation_mm=700.0,
        elevation=0.9, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    assert s == 0.0


def test_crop_suitability_excludes_outside_temp() -> None:
    c = _wheat()
    s_cold = crop_suitability(
        c, biome="plains",
        temperature_c=-5.0, precipitation_mm=700.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    s_hot = crop_suitability(
        c, biome="plains",
        temperature_c=35.0, precipitation_mm=700.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    assert s_cold == 0.0
    assert s_hot == 0.0


def test_crop_suitability_optimum() -> None:
    c = _wheat()
    s = crop_suitability(
        c, biome="plains",
        temperature_c=18.0, precipitation_mm=700.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    assert s == 1.0


def test_crop_suitability_river_bonus_clamps_to_one() -> None:
    c = _wheat()
    s_no_river = crop_suitability(
        c, biome="plains",
        temperature_c=18.0, precipitation_mm=700.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    s_river = crop_suitability(
        c, biome="plains",
        temperature_c=18.0, precipitation_mm=700.0,
        elevation=0.1, is_river=True, is_coast=False, has_water_neighbor=False,
    )
    assert s_river >= s_no_river
    assert s_river <= 1.0


def test_irrigation_replaces_rain() -> None:
    """A crop with irrigation_replaces_rain_mm gets effective rainfall added
    when next to a river/lake — letting it grow in otherwise-dry tropics."""
    rice = CropDefinition(
        name="rice",
        temp_abs_min=10.0, temp_opt_min=22.0, temp_opt_max=30.0, temp_abs_max=38.0,
        precip_abs_min=700.0, precip_opt_min=1200.0, precip_opt_max=2500.0,
        precip_abs_max=4500.0,
        elev_max=0.4,
        biome_compatibility={"plains": 1.0, "river": 1.0},
        river_bonus=0.5, river_adjacent_bonus=0.3, coast_bonus=0.0,
        irrigation_replaces_rain_mm=800.0,
    )
    # 500 mm of rain — too dry for rice (below precip_abs_min=700).
    dry_no_water = crop_suitability(
        rice, biome="plains", temperature_c=25.0, precipitation_mm=500.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=False,
    )
    # Same conditions but with a river neighbor — irrigation adds 800 mm effective
    # rainfall → 1300 mm → within optimum band.
    dry_with_river = crop_suitability(
        rice, biome="plains", temperature_c=25.0, precipitation_mm=500.0,
        elevation=0.1, is_river=False, is_coast=False, has_water_neighbor=True,
    )
    assert dry_no_water == 0.0
    assert dry_with_river > 0.5


def test_pipeline_emits_crop_scores_for_land_only(medium_world: GeneratedWorld) -> None:
    """Ocean and lake hexes get empty crop dicts; land hexes get some entries."""
    for h, d in medium_world.hexes.items():
        if d.is_ocean or d.is_lake:
            assert d.crop_suitability == {}
        # Land hexes may legitimately have no compatible crops (e.g. snow_peak),
        # so we only require *some* land hex globally to have entries.
    has_any = any(d.crop_suitability for d in medium_world.hexes.values()
                  if not d.is_ocean and not d.is_lake)
    assert has_any


def test_pipeline_emits_deposits_on_eligible_biomes_only(
    medium_world: GeneratedWorld,
) -> None:
    """Each deposit must occur in a biome listed in its host_biomes (if any)."""
    config = medium_world.config
    eligible_by_resource = {
        r.name: set(r.host_biomes) for r in config.resources if r.host_biomes
    }
    for h, d in medium_world.hexes.items():
        for resource_name in d.deposits:
            biomes = eligible_by_resource.get(resource_name)
            if biomes is None:
                continue
            assert d.biome in biomes, (
                f"{resource_name} deposit at biome={d.biome}, "
                f"only allowed in {biomes}"
            )


def test_pipeline_deposit_quantities_positive(medium_world: GeneratedWorld) -> None:
    for d in medium_world.hexes.values():
        for q in d.deposits.values():
            assert q > 0.0


def test_resources_deterministic(medium_world: GeneratedWorld, default_worldgen_config) -> None:  # type: ignore[no-untyped-def]
    """Two runs with the same seed produce identical deposit + crop fields."""
    from worldgen import generate

    a = medium_world
    b = generate(radius=a.radius, config=default_worldgen_config, seed=42)
    for h in a.hexes:
        assert a.hexes[h].deposits == b.hexes[h].deposits
        assert a.hexes[h].crop_suitability == b.hexes[h].crop_suitability


def test_abundance_roughly_matches_config(medium_world: GeneratedWorld) -> None:
    """The fraction of eligible hexes that host each resource should be close
    to the configured abundance (within ±50 %)."""
    config = medium_world.config
    for resource in config.resources:
        eligible_count = 0
        deposit_count = 0
        for h, d in medium_world.hexes.items():
            if d.is_ocean:
                continue
            if resource.host_biomes and d.biome not in resource.host_biomes:
                continue
            if d.elevation < resource.min_elevation or d.elevation > resource.max_elevation:
                continue
            if d.temperature_c < resource.min_temperature_c or d.temperature_c > resource.max_temperature_c:
                continue
            if d.precipitation_mm < resource.min_precipitation_mm or d.precipitation_mm > resource.max_precipitation_mm:
                continue
            eligible_count += 1
            if resource.name in d.deposits:
                deposit_count += 1
        if eligible_count < 25:
            # Skip resources too rare in the radius-30 test world to measure.
            continue
        actual = deposit_count / eligible_count
        target = resource.abundance
        # Loose bound: factor-of-2 tolerance — quantile is approximate at small N.
        assert actual >= target * 0.5
        assert actual <= target * 1.5 + 0.05
