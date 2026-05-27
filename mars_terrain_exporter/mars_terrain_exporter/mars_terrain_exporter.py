# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mars terrain generation pipeline.

Three-stage pipeline (mirrors lunar_terrain_exporter):

1. **Download** — Fetches the HiRISE DTM GeoTIFF via HTTP with local caching.
2. **DEM Processing** — Reads the GeoTIFF (ESRI:103885, metres), handles the
   projected CRS (``src.res`` already in m/px), optionally crops to a bounding
   box, and extracts float64 elevation data.
3. **Model Writing** — Generates ``heightmap.tif``, ``model.sdf``,
   ``world.sdf`` (Bullet Featherstone, 0.005 s step), ``model.config``, and
   ``metadata.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .utils.types import MarsSite
from .utils.file_downloader import FileDownloader
from .model_writers.sdf_model_writer import SDFModelWriter
from .raster_processors.dem_processor import DEMProcessor


class MarsTerrainExporter:
    """Generates Gazebo SDF terrain models from NASA HiRISE DTMs."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        workspace_dir = os.getenv("WORKSPACE_DIR", "")
        if workspace_dir:
            self._default_cache_dir = Path(workspace_dir) / ".hirise_dem_cache"
        else:
            self._default_cache_dir = Path.home() / ".cache" / "mars_terrain_exporter"
        self._downloader = FileDownloader(self._default_cache_dir)
        self._model_writer = SDFModelWriter(self._output_dir)

    def export_model(self, site: MarsSite) -> Path:
        """Export a complete Gazebo terrain model for *site*.

        Returns the path to the generated model directory.
        """
        print(f"\n=== Generating Mars terrain: {site.name} ===")

        # Stage 1 — Download
        dem_file = self._downloader.download(site.dem_url)

        # Stage 2 — DEM Processing
        # ESRI:103885 is projected (metres): src.res → (x_res_m, y_res_m).
        # No degree-to-metre conversion required.
        elevations, elev_min, elev_max, bounds, dem_profile = (
            DEMProcessor.extract_from_raw(dem_file, site.roi)
        )

        lat = bounds["center_lat"]
        lon = bounds["center_lon"]
        width_km = bounds["width_km"]
        height_km = bounds["height_km"]

        print(
            f"  ROI: center=({lat:.4f}°, {lon:.4f}°), "
            f"{width_km:.1f} × {height_km:.1f} km"
        )
        print(
            f"  Elevation range: {elev_min:.1f} … {elev_max:.1f} m "
            f"(Δ {elev_max - elev_min:.1f} m)"
        )
        print(f"  Raster size: {elevations.shape[1]} × {elevations.shape[0]} px")

        size_x_m = int(round(width_km * 1000.0))
        size_y_m = int(round(height_km * 1000.0))

        # Stage 3 — Model Writing
        self._model_writer.write(
            site_id=site.name,
            display_name=site.name.replace("_", " ").title(),
            description=site.description or (
                f"Mars HiRISE terrain at ({lat:.4f}°, {lon:.4f}°)"
            ),
            elevations=elevations,
            dem_profile=dem_profile,
            size_x_m=size_x_m,
            size_y_m=size_y_m,
            elevation_min=elev_min,
            elevation_max=elev_max,
            lat=lat,
            lon=lon,
            source="nasa_hirise_asc_pds",
        )

        return self._output_dir / site.name
