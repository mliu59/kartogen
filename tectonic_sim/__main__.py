"""CLI entry points for ``tectonic_sim``.

Subcommands:

  - ``inspect-seed``  build the initial particle layout and emit PNGs
                      (particles, voronoi, per-plate) under ``--out``.

Future phases will add ``inspect-snapshot``, ``inspect-frame``, and a
top-level ``simulate`` runner. The CLI is deliberately small for now —
it's a debugging aid for the phases as they come online, not a full UX.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tectonic_sim import (
    WorldRect,
    load_sim_config_from_path,
)
from tectonic_sim._log import configure_logging
from tectonic_sim.viz import render_initial_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tectonic_sim",
        description="Continuous-2D plate tectonics simulator (debugging CLI).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- inspect-seed ----
    p = sub.add_parser(
        "inspect-seed",
        help="Render the t=0 seeded state to PNGs.",
    )
    p.add_argument(
        "--config", type=Path, default=Path("config/tectonic_sim.toml"),
        help="Path to a tectonic_sim TOML config.",
    )
    p.add_argument("--width-km", type=float, default=1000.0)
    p.add_argument("--height-km", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output directory. Will be created if missing.",
    )
    p.add_argument(
        "--px-per-km", type=float, default=1.0,
        help="Pixel scale for the particle scatter (default 1 px/km).",
    )
    p.add_argument(
        "--voronoi-px-per-km", type=float, default=0.6,
        help="Pixel scale for the per-pixel voronoi render (default 0.6 px/km).",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    return parser


def _cmd_inspect_seed(args: argparse.Namespace) -> None:
    cfg = load_sim_config_from_path(args.config)
    domain = WorldRect(width_km=args.width_km, height_km=args.height_km)
    out_dir = render_initial_state(
        domain,
        cfg,
        seed=args.seed,
        out_dir=args.out,
        px_per_km=args.px_per_km,
        voronoi_px_per_km=args.voronoi_px_per_km,
    )
    print(f"Wrote initial-state renders to {out_dir}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(logging.WARNING if args.quiet else logging.DEBUG)

    if args.cmd == "inspect-seed":
        _cmd_inspect_seed(args)
    else:
        parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
