"""Shared fixtures for the ``tectonic_sim`` test suite.

The default sim config is loaded from ``config/tectonic_sim.toml`` so
tests track real config changes. Tests build small worlds inline via
``WorldRect(...)`` since the world rectangle isn't a sim-config knob —
it's passed in by the caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tectonic_sim import SimConfig, load_sim_config_from_path

# Path layout: this file lives at
#   <repo>/tectonic_sim/tests/conftest.py
# so ``parents[2]`` is the repo root, where ``config/`` lives.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def default_sim_config() -> SimConfig:
    """Sim parameters loaded from ``config/tectonic_sim.toml``."""
    return load_sim_config_from_path(
        _REPO_ROOT / "config" / "tectonic_sim.toml"
    )
