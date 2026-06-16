# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Adapted from jasmeet0915/artemis_mission_simulator (Apache-2.0).


"""Mars terrain generation pipeline."""

import os
from pathlib import Path

from .utils.types import MarsSite
from .utils.file_downloader import FileDownloader
from .model_writers.mjcf_model_writer import MJCFModelWriter
from .raster_processors.dem_processor import DEMProcessor
from .raster_processors import procedural_terrain


class MarsTerrainExporter:
    """Generates MuJoCo MJCF terrain models from Mars 2020 HiRISE DTMs."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        workspace_dir = os.getenv("WORKSPACE_DIR", str(Path.home()))
        self._default_cache_dir = Path(workspace_dir) / ".dem_cache"
        self._downloader = FileDownloader(self._default_cache_dir)
        self._model_writer = MJCFModelWriter(self._output_dir)

    def export_model(
        self,
        site: MarsSite,
        bake_texture: bool = False,
        orthophoto: bool = False,
        texture_size: int = 2048,
        n_rocks: int = 1200,
        n_boulders: int = 0,
        seed: int | None = None,
    ) -> Path:
        """Export a complete MuJoCo terrain model for a site.

        ``orthophoto`` drapes the real NASA HiRISE orthophoto over the DTM (best
        for VO/SLAM); ``bake_texture`` drapes a synthetic Mars albedo instead;
        ``n_boulders`` scatters 3D rocks as nav landmarks.
        """
        print(f"\n=== Generating: {site.name} ===")

        dem_file = self._downloader.download(site.dem_url)

        elevations, elev_min, elev_max, bounds, meta = (
            DEMProcessor.extract_from_raw(dem_file, site.roi)
        )
        lat = bounds["center_lat"]
        lon = bounds["center_lon"]
        print(f"    ROI: center=({lat:.4f}, {lon:.4f}), "
              f"{bounds['width_km']*1000:.0f}x{bounds['height_km']*1000:.0f}m, "
              f"{meta['resolution_m']:.3f} m/px")
        print(f"    Elevation: {elev_min:.1f}m to {elev_max:.1f}m "
              f"(range {elev_max - elev_min:.1f}m)")

        # Real HiRISE orthophoto draped over the DTM (windowed remote read).
        albedo_rgb = None
        if orthophoto:
            from .utils.orthophoto import fetch_window
            from .utils.types import BoundingBox
            print("    Fetching HiRISE orthophoto window (25 cm/px, remote)...")
            albedo_rgb, ortho_res = fetch_window(BoundingBox(
                lat=lat, lon=lon,
                width_km=bounds["width_km"], height_km=bounds["height_km"],
            ))
            print(f"    Orthophoto: {albedo_rgb.shape[1]}x{albedo_rgb.shape[0]} px "
                  f"at {ortho_res:.3f} m/px")

        self._model_writer.write(
            site_id=site.name,
            display_name=site.name.replace("_", " ").title(),
            description=site.description or f"Mars terrain at ({lat}, {lon})",
            elevations=elevations,
            resolution_m=meta["resolution_m"],
            elevation_min=elev_min,
            elevation_max=elev_max,
            lat=lat,
            lon=lon,
            source="nasa_hirise_ortho" if orthophoto else "nasa_hirise_mars2020",
            bake_texture=bake_texture,
            texture_size=texture_size,
            n_rocks=n_rocks,
            n_boulders=n_boulders,
            seed=seed,
            albedo_rgb=albedo_rgb,
        )
        return self._output_dir / site.name

    def export_procedural(
        self,
        name: str,
        terrain_type: str,
        size_m: float,
        resolution_m: float,
        seed: int | None = None,
        **kwargs,
    ) -> Path:
        """Export a synthetic (procedurally generated) terrain model.

        ``terrain_type`` is one of ``procedural_terrain.TERRAIN_TYPES``; extra
        ``kwargs`` (e.g. ``slope``, ``amplitude_m``, ``tile_m``, ``max_slope``)
        are forwarded to the chosen generator.
        """
        print(f"\n=== Generating: {name} (procedural: {terrain_type}) ===")

        elevations = procedural_terrain.generate(
            terrain_type, size_m, resolution_m, seed=seed, **kwargs
        )
        rows, cols = elevations.shape
        elev_max = float(elevations.max())
        print(f"    Patch: {cols * resolution_m:.0f}x{rows * resolution_m:.0f}m, "
              f"{resolution_m:.3f} m/px ({cols}x{rows} px)")
        print(f"    Elevation: 0.0m to {elev_max:.2f}m (range {elev_max:.2f}m)")

        self._model_writer.write(
            site_id=name,
            display_name=name.replace("_", " ").title(),
            description=f"Procedural Mars terrain ({terrain_type})",
            elevations=elevations,
            resolution_m=resolution_m,
            elevation_min=0.0,
            elevation_max=elev_max,
            lat=0.0,
            lon=0.0,
            source=f"procedural_{terrain_type}",
        )
        return self._output_dir / name
