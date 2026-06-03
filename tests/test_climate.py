"""Tests for the climate layer (temperature + precipitation)."""

from __future__ import annotations

import statistics

import pytest
from kartogen.pipeline import GeneratedWorld

pytestmark = pytest.mark.slow  # full generate()/sim per test
def test_temperature_decreases_toward_poles(medium_world: GeneratedWorld) -> None:
    """Annual mean temperature should be coldest at high latitude and warmest near the equator."""
    from kartogen.climate import hex_latitude_deg
    from kartogen.world import map_half_extents_km
    cfg = medium_world.config
    _, half_h = map_half_extents_km(medium_world.hexes.keys(), cfg.hex_size_km)
    by_lat: dict[int, list[float]] = {}
    for h, d in medium_world.hexes.items():
        if d.is_ocean:
            continue
        bucket = int(abs(hex_latitude_deg(h, half_h, cfg)) // 5)
        by_lat.setdefault(bucket, []).append(d.temperature_c)

    means = sorted(((b, statistics.mean(ts)) for b, ts in by_lat.items() if len(ts) >= 5))
    # Coldest band must be at higher |latitude| than warmest band.
    coldest_band = min(means, key=lambda x: x[1])[0]
    warmest_band = max(means, key=lambda x: x[1])[0]
    assert coldest_band > warmest_band


def test_elevation_cools_with_altitude(medium_world: GeneratedWorld) -> None:
    """Land hexes at similar latitude but higher elevation should be cooler."""
    # Bucket by latitude band; within each band, correlation between elevation and temp must be negative.
    by_lat: dict[int, list[tuple[float, float]]] = {}
    for h, d in medium_world.hexes.items():
        if d.is_ocean:
            continue
        bucket = h.r // 3
        by_lat.setdefault(bucket, []).append((d.elevation, d.temperature_c))

    # Pick bands with enough samples.
    correlations: list[float] = []
    for _, samples in by_lat.items():
        if len(samples) < 12:
            continue
        es = [s[0] for s in samples]
        ts = [s[1] for s in samples]
        mu_e = statistics.mean(es)
        mu_t = statistics.mean(ts)
        cov = sum((e - mu_e) * (t - mu_t) for e, t in samples)
        correlations.append(cov)

    assert correlations, "expected at least one latitude band with enough samples"
    # Average covariance across bands must be negative (higher elevation → cooler).
    assert statistics.mean(correlations) < 0


def test_precipitation_nonnegative(medium_world: GeneratedWorld) -> None:
    """Precipitation can never go below zero."""
    for d in medium_world.hexes.values():
        assert d.precipitation_mm >= 0


def test_precipitation_zero_over_ocean(medium_world: GeneratedWorld) -> None:
    """Ocean hexes get no precipitation in the simulation's bookkeeping."""
    for d in medium_world.hexes.values():
        if d.is_ocean:
            assert d.precipitation_mm == 0.0


