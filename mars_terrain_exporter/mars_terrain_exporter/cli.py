# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Command-line interface for generating Mars terrain models.

Subcommands
-----------
site   Export a single site (full HiRISE DTM tile or custom bounding-box crop).
batch  Export multiple sites listed in a YAML config file.

Examples
--------
Full tile export::

    mars_terrain_exporter site jezero_c --output-dir ./models

Custom 3 × 3 km crop around the delta fan::

    mars_terrain_exporter site jezero_dl \\
      --lat 18.45 --lon 77.40 --width 3 --height 3 \\
      --output-dir ./models

Batch from config file::

    mars_terrain_exporter batch \\
      --config config/jezero_sites.yaml --output-dir ./models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .mars_terrain_exporter import MarsTerrainExporter
from .utils.types import BoundingBox, ROI, MarsSite


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mars_terrain_exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Export terrain models from NASA HiRISE DTMs for Mars simulation.\n"
            "Data: ASC-PDS Services (ESRI:103885, 1 m/px HiRISE mosaics).\n"
            "Output: Gazebo Harmonic SDF models with Bullet Featherstone physics."
        ),
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---- site subcommand --------------------------------------------------
    site_parser = subparsers.add_parser(
        "site",
        help="Export model for a single site from the HiRISE catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # Full DTM tile\n"
            "  mars_terrain_exporter site jezero_c --output-dir ./models\n\n"
            "  # Custom 3 km ROI around the delta fan\n"
            "  mars_terrain_exporter site jezero_dl \\\n"
            "    --lat 18.45 --lon 77.40 --width 3 --height 3 \\\n"
            "    --output-dir ./models"
        ),
    )
    site_parser.add_argument(
        "site_name",
        type=str,
        help="Site name or site code (e.g. jezero_c)",
    )
    site_parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Center latitude (°) for custom crop (requires --lon)",
    )
    site_parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Center longitude (°) for custom crop (requires --lat)",
    )
    site_parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Crop width in km (default: 10)",
    )
    site_parser.add_argument(
        "--height",
        type=float,
        default=10.0,
        help="Crop height in km (default: 10)",
    )
    site_parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated models (default: .)",
    )

    # ---- batch subcommand -------------------------------------------------
    batch_parser = subparsers.add_parser(
        "batch",
        help="Export multiple sites from a YAML config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  mars_terrain_exporter batch \\\n"
            "    --config config/jezero_sites.yaml --output-dir ./models"
        ),
    )
    batch_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config listing sites to export",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated models (default: .)",
    )

    return parser


# ---------------------------------------------------------------------------
# YAML batch loader
# ---------------------------------------------------------------------------

def load_sites_from_yaml(config_path: Path) -> list[MarsSite]:
    """Parse a batch YAML config and return a list of ``MarsSite`` instances."""
    with open(config_path) as fh:
        data = yaml.safe_load(fh)

    sites: list[MarsSite] = []

    for entry in data["sites"]:
        roi_raw = entry.get("roi", {})
        use_full = bool(roi_raw.get("use_full", True))

        bounding_box = None
        bb_raw = roi_raw.get("bounding_box")
        if bb_raw is not None:
            bounding_box = BoundingBox(
                lat=float(bb_raw["lat"]),
                lon=float(bb_raw["lon"]),
                width_km=float(bb_raw.get("width_km", 10.0)),
                height_km=float(bb_raw.get("height_km", 10.0)),
            )

        roi = ROI(use_full=use_full, bounding_box=bounding_box)

        try:
            site = MarsSite.from_catalog(entry["site"], roi=roi)
        except (KeyError, ValueError) as exc:
            print(f"Warning: skipping entry {entry!r}: {exc}", file=sys.stderr)
            continue

        sites.append(site)

    return sites


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output_dir)
    sites: list[MarsSite] = []

    if args.command == "site":
        if args.lat is not None and args.lon is not None:
            roi = ROI(
                use_full=False,
                bounding_box=BoundingBox(
                    lat=args.lat,
                    lon=args.lon,
                    width_km=args.width,
                    height_km=args.height,
                ),
            )
        else:
            roi = ROI(use_full=True)

        try:
            site = MarsSite.from_catalog(args.site_name, roi=roi)
        except (KeyError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        sites.append(site)

    elif args.command == "batch":
        print(f"Loading sites from {args.config}…")
        sites = load_sites_from_yaml(Path(args.config))
        print(f"Found {len(sites)} site(s) to export.")

    exporter = MarsTerrainExporter(output_dir)

    for site in sites:
        print(f"\nExporting '{site.name}' ({site.site_code}) → {output_dir}/")
        exporter.export_model(site)

    print("\nDone!")
