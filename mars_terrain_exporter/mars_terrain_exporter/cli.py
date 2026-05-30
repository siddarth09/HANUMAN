# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Adapted from jasmeet0915/artemis_mission_simulator (Apache-2.0).


"""Command-line interface for generating Mars terrain models for MuJoCo.

Subcommands
-----------
site   Export a single site (center crop, geographic crop, or full DEM).
batch  Export multiple sites listed in a YAML config file.
list   List the available Mars HiRISE sites.
"""

import argparse
import sys
from pathlib import Path

import yaml

from .mars_terrain_exporter import MarsTerrainExporter
from .utils.types import BoundingBox, ROI, MarsSite
from .utils.site_catalog import list_sites
from .raster_processors.procedural_terrain import TERRAIN_TYPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mars_terrain_exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Export MuJoCo terrain models (hfield PNG + MJCF) from NASA Mars 2020 "
            "TRN HiRISE Digital Terrain Models (1 m/pixel, Jezero Crater region)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # site subcommand
    site_parser = subparsers.add_parser(
        "site",
        help="Export model for a single site from the Mars HiRISE catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # 150 m patch from the raster center at 0.1 m/px\n"
            "  mars_terrain_exporter site jezero_c --size 150 --output-dir ./models\n\n"
            "  # Geographic crop centered on a lat/lon\n"
            "  mars_terrain_exporter site jezero_c "
            "--lat 18.44 --lon 77.45 --width 0.2 --height 0.2 --output-dir ./models"
        ),
    )
    site_parser.add_argument(
        "site_name", type=str,
        help="Site name or code from the Mars HiRISE catalog (e.g. jezero_c or Jezero_C)",
    )
    site_parser.add_argument(
        "--name", type=str, default=None,
        help="Override the output model name (folder + MJCF id + asset filenames)",
    )
    site_parser.add_argument(
        "--lat", type=float, default=None, help="Center latitude for a geographic crop",
    )
    site_parser.add_argument(
        "--lon", type=float, default=None, help="Center longitude for a geographic crop",
    )
    site_parser.add_argument(
        "--width", type=float, default=0.2, help="Crop width in km (geographic crop)",
    )
    site_parser.add_argument(
        "--height", type=float, default=0.2, help="Crop height in km (geographic crop)",
    )
    site_parser.add_argument(
        "--size", type=float, default=100.0,
        help="Center-crop size in meters (used when --lat/--lon are omitted)",
    )
    site_parser.add_argument(
        "--resolution", type=float, default=0.1, help="Target resolution in m/px",
    )
    site_parser.add_argument(
        "--full", action="store_true", help="Use the entire DEM tile (large!)",
    )
    site_parser.add_argument(
        "--orthophoto", action="store_true",
        help="Drape the REAL NASA HiRISE orthophoto over the DTM (remote read)",
    )
    site_parser.add_argument(
        "--textured", action="store_true",
        help="Bake a synthetic Mars albedo (relief + regolith + rocks) for VO/SLAM",
    )
    site_parser.add_argument(
        "--texture-size", type=int, default=2048,
        help="Baked albedo texture size in pixels (default: 2048)",
    )
    site_parser.add_argument(
        "--rocks", type=int, default=1200,
        help="Number of rock specks baked into the texture (default: 1200)",
    )
    site_parser.add_argument(
        "--boulders", type=int, default=0,
        help="Number of 3D boulders scattered as nav landmarks (default: 0)",
    )
    site_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for texture/boulders",
    )
    site_parser.add_argument(
        "--output-dir", type=str, default="./models",
        help="Output directory for generated models (default: ./models)",
    )

    # batch subcommand
    batch_parser = subparsers.add_parser(
        "batch",
        help="Export multiple sites from a YAML config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  mars_terrain_exporter batch "
            "--config config/hanuman_sites.yaml --output-dir ./models"
        ),
    )
    batch_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file listing sites to export",
    )
    batch_parser.add_argument(
        "--output-dir", type=str, default="./models",
        help="Output directory for generated models (default: ./models)",
    )

    # procedural subcommand
    proc_parser = subparsers.add_parser(
        "procedural",
        help="Generate synthetic rough/hilly/sloped terrain (no DEM download)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "terrain types:\n"
            "  rough       random uniform roughness (legged_gym random_uniform)\n"
            "  slope       square pyramid slope (peak at center)\n"
            "  slope_inv   inverted pyramid (crater / pit)\n"
            "  hills       smooth fractal rolling hills\n"
            "  mix         legged_gym-style grid mixing all of the above\n\n"
            "examples:\n"
            "  # 50 m mixed rough terrain\n"
            "  mars_terrain_exporter procedural mars_rough --type mix --size 50\n\n"
            "  # 50 m rolling hills, ~2 m relief\n"
            "  mars_terrain_exporter procedural mars_hills --type hills "
            "--size 50 --amplitude 2.0"
        ),
    )
    proc_parser.add_argument(
        "name", type=str,
        help="Output model name (becomes the <name>/ folder and MJCF model id)",
    )
    proc_parser.add_argument(
        "--type", dest="terrain_type", choices=TERRAIN_TYPES, default="mix",
        help="Terrain type to generate (default: mix)",
    )
    proc_parser.add_argument(
        "--size", type=float, default=50.0, help="Square patch size in meters",
    )
    proc_parser.add_argument(
        "--resolution", type=float, default=0.1, help="Resolution in m/px",
    )
    proc_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (for reproducibility)",
    )
    proc_parser.add_argument(
        "--slope", type=float, default=None,
        help="Incline tangent for slope/slope_inv (0.8 ~= 38 deg)",
    )
    proc_parser.add_argument(
        "--amplitude", type=float, dest="amplitude_m", default=None,
        help="Peak-to-trough height in meters (hills)",
    )
    proc_parser.add_argument(
        "--tile", type=float, dest="tile_m", default=None,
        help="Sub-terrain tile size in meters (mix)",
    )
    proc_parser.add_argument(
        "--max-slope", type=float, dest="max_slope", default=None,
        help="Max slope tangent for tiles (mix)",
    )
    proc_parser.add_argument(
        "--output-dir", type=str, default="./models",
        help="Output directory for generated models (default: ./models)",
    )

    # list subcommand
    subparsers.add_parser("list", help="List available Mars HiRISE sites")

    return parser


def _roi_from_args(args) -> ROI:
    """Build an ROI from `site` subcommand arguments."""
    if args.full:
        return ROI(use_full=True, resolution_m=args.resolution)
    if args.lat is not None and args.lon is not None:
        return ROI(
            bounding_box=BoundingBox(
                lat=args.lat, lon=args.lon,
                width_km=args.width, height_km=args.height,
            ),
            resolution_m=args.resolution,
        )
    return ROI(size_m=args.size, resolution_m=args.resolution)


def load_sites_from_yaml(config_path: Path) -> list[MarsSite]:
    """Parse a YAML config file and return a list of MarsSite objects."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    sites: list[MarsSite] = []
    for entry in data["sites"]:
        roi_raw = entry.get("roi", {})
        bounding_box = None
        bb_raw = roi_raw.get("bounding_box")
        if bb_raw is not None:
            bounding_box = BoundingBox(
                lat=float(bb_raw["lat"]),
                lon=float(bb_raw["lon"]),
                width_km=float(bb_raw.get("width_km", 0.2)),
                height_km=float(bb_raw.get("height_km", 0.2)),
            )
        roi = ROI(
            use_full=bool(roi_raw.get("use_full", False)),
            bounding_box=bounding_box,
            size_m=float(roi_raw.get("size_m", 100.0)),
            resolution_m=float(roi_raw.get("resolution_m", 0.1)),
        )
        try:
            site = MarsSite.from_catalog(entry["site"], roi=roi)
        except (KeyError, ValueError) as exc:
            print(f"Warning: skipping entry {entry!r}: {exc}", file=sys.stderr)
            continue
        sites.append(site)

    return sites


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        print("\n  HANUMAN — Available Mars Terrain Sites\n  " + "=" * 44)
        for entry in list_sites():
            print(f"  {entry['site_name']:<18s} {entry['description']}")
        print()
        return

    if args.command == "procedural":
        exporter = MarsTerrainExporter(Path(args.output_dir))
        gen_kwargs = {
            "slope": args.slope,
            "amplitude_m": args.amplitude_m,
            "tile_m": args.tile_m,
            "max_slope": args.max_slope,
        }
        exporter.export_procedural(
            name=args.name,
            terrain_type=args.terrain_type,
            size_m=args.size,
            resolution_m=args.resolution,
            seed=args.seed,
            **gen_kwargs,
        )
        print("\nDone!")
        return

    output_dir = Path(args.output_dir)
    sites: list[MarsSite] = []

    if args.command == "site":
        try:
            site = MarsSite.from_catalog(args.site_name, roi=_roi_from_args(args))
            if args.name:
                site.name = args.name
                site.validate()
        except (KeyError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        sites.append(site)

    elif args.command == "batch":
        print(f"Loading sites from {args.config}...")
        sites = load_sites_from_yaml(Path(args.config))
        print(f"Found {len(sites)} site(s) to export.")

    # Texture/boulder options are only exposed on the `site` subcommand.
    tex_kwargs = {}
    if args.command == "site":
        tex_kwargs = dict(
            bake_texture=args.textured,
            orthophoto=args.orthophoto,
            texture_size=args.texture_size,
            n_rocks=args.rocks,
            n_boulders=args.boulders,
            seed=args.seed,
        )

    exporter = MarsTerrainExporter(output_dir)
    for site in sites:
        print(f"\nExporting '{site.name}' ({site.site_code}) → {output_dir}/")
        exporter.export_model(site, **tex_kwargs)

    print("\nDone!")
