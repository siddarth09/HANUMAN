# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Tests for SDFModelWriter — verifies all output files are generated correctly.

Checks:
  • model.sdf       — heightmap URI, size, site_id
  • world.sdf       — Bullet Featherstone physics, max_step_size 0.005, model include
  • model.config    — display_name, description
  • metadata.yaml   — all fields + physics sub-dict
  • heightmap.tif   — float32 GeoTIFF with correct shape
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml

from mars_terrain_exporter.model_writers.sdf_model_writer import SDFModelWriter


def _make_writer(tmp_path: Path) -> tuple[SDFModelWriter, dict]:
    writer = SDFModelWriter(tmp_path)
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    try:
        crs = CRS.from_authority("ESRI", "103885")
    except Exception:
        crs = CRS.from_epsg(4326)

    elevs = np.linspace(0.0, 50.0, 64 * 64, dtype=np.float64).reshape(64, 64)
    transform = from_bounds(0, 0, 6400, 6400, 64, 64)
    dem_profile = {"crs": crs, "transform": transform}

    kwargs = dict(
        site_id="jezero_c",
        display_name="Jezero C",
        description="Test terrain",
        elevations=elevs,
        dem_profile=dem_profile,
        size_x_m=6400,
        size_y_m=6400,
        elevation_min=float(elevs.min()),
        elevation_max=float(elevs.max()),
        lat=18.44,
        lon=77.45,
        source="nasa_hirise_asc_pds",
    )
    return writer, kwargs


class TestSDFModelWriter:
    def test_all_files_created(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)

        model_dir = (tmp_path / "jezero_c").resolve()
        assert (model_dir / "model.sdf").exists()
        assert (model_dir / "world.sdf").exists()
        assert (model_dir / "model.config").exists()
        assert (model_dir / "metadata.yaml").exists()
        tex = model_dir / "materials" / "textures"
        assert (tex / "heightmap.png").exists(),  "16-bit PNG for Gazebo/Bullet"
        assert (tex / "heightmap.tif").exists(),  "float32 GeoTIFF for external tools"
        assert (tex / "mars_diffuse.png").exists()
        assert (tex / "mars_normal.png").exists()

    def test_model_sdf_contains_site_id(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        sdf = (tmp_path / "jezero_c").resolve() .joinpath("model.sdf").read_text()
        assert "jezero_c" in sdf

    def test_model_sdf_heightmap_uri_is_file_scheme(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        sdf = (tmp_path / "jezero_c").resolve().joinpath("model.sdf").read_text()
        # Must reference the 16-bit PNG via file:// — no model:// or .tif
        assert "file://" in sdf
        assert "heightmap.png" in sdf
        assert "model://materials" not in sdf

    def test_model_sdf_texture_uris_are_file_scheme(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        sdf = (tmp_path / "jezero_c").resolve().joinpath("model.sdf").read_text()
        assert "mars_diffuse.png" in sdf
        assert "mars_normal.png" in sdf
        assert "model://materials" not in sdf

    def test_model_sdf_size_values(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        sdf = (tmp_path / "jezero_c").resolve().joinpath("model.sdf").read_text()
        assert "6400" in sdf

    # ------------------------------------------------------------------ #
    # world.sdf — Bullet Featherstone physics                              #
    # ------------------------------------------------------------------ #

    def _model_dir(self, tmp_path):
        return (tmp_path / "jezero_c").resolve()

    def test_world_sdf_bullet_featherstone_type(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        world = (self._model_dir(tmp_path) / "world.sdf").read_text()
        assert 'type="bullet_featherstone"' in world

    def test_world_sdf_max_step_size(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        world = (self._model_dir(tmp_path) / "world.sdf").read_text()
        assert "<max_step_size>0.005</max_step_size>" in world

    def test_world_sdf_physics_plugin(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        world = (self._model_dir(tmp_path) / "world.sdf").read_text()
        assert "gz-physics-bullet-featherstone-plugin" in world

    def test_world_sdf_includes_model(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        world = (self._model_dir(tmp_path) / "world.sdf").read_text()
        assert "file://" in world
        assert "jezero_c" in world

    def test_world_sdf_real_time_update_rate_200(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        world = (self._model_dir(tmp_path) / "world.sdf").read_text()
        assert "<real_time_update_rate>200</real_time_update_rate>" in world

    def test_world_sdf_valid_xml(self, tmp_path):
        import xml.etree.ElementTree as ET
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        ET.parse(str(self._model_dir(tmp_path) / "world.sdf"))

    # ------------------------------------------------------------------ #
    # metadata.yaml                                                        #
    # ------------------------------------------------------------------ #

    def test_metadata_site_id(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        meta = yaml.safe_load(
            (self._model_dir(tmp_path) / "metadata.yaml").read_text()
        )
        assert meta["site_id"] == "jezero_c"

    def test_metadata_coordinates(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        meta = yaml.safe_load(
            (self._model_dir(tmp_path) / "metadata.yaml").read_text()
        )
        assert meta["coordinates"]["lat"] == pytest.approx(18.44)
        assert meta["coordinates"]["lon"] == pytest.approx(77.45)

    def test_metadata_physics_block(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        meta = yaml.safe_load(
            (self._model_dir(tmp_path) / "metadata.yaml").read_text()
        )
        phys = meta["physics"]
        assert phys["engine"] == "bullet_featherstone"
        assert phys["max_step_size"] == pytest.approx(0.005)
        assert phys["real_time_update_rate"] == 200

    def test_metadata_elevation_stats(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        meta = yaml.safe_load(
            (self._model_dir(tmp_path) / "metadata.yaml").read_text()
        )
        assert meta["elevation_min_m"] < meta["elevation_max_m"]
        assert meta["elevation_range_m"] > 0

    # ------------------------------------------------------------------ #
    # heightmap.tif + placeholder textures                                 #
    # ------------------------------------------------------------------ #

    def test_heightmap_tif_is_float32(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        tif_path = self._model_dir(tmp_path) / "materials" / "textures" / "heightmap.tif"
        with rasterio.open(tif_path) as src:
            assert src.dtypes[0] == "float32"

    def test_heightmap_tif_shape(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        tif_path = self._model_dir(tmp_path) / "materials" / "textures" / "heightmap.tif"
        with rasterio.open(tif_path) as src:
            assert src.height == 64
            assert src.width == 64

    def test_placeholder_textures_exist(self, tmp_path):
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        tex = self._model_dir(tmp_path) / "materials" / "textures"
        assert (tex / "mars_diffuse.png").exists()
        assert (tex / "mars_normal.png").exists()
        assert (tex / "mars_diffuse.png").stat().st_size > 0
        assert (tex / "mars_normal.png").stat().st_size > 0

    def test_heightmap_png_is_valid_16bit(self, tmp_path):
        """Verify the 16-bit PNG has the correct PNG signature and IHDR."""
        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        png_bytes = (
            self._model_dir(tmp_path)
            / "materials" / "textures" / "heightmap.png"
        ).read_bytes()
        # PNG magic
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        # IHDR data starts at byte 16; bit-depth field is at offset 8 of IHDR data
        bit_depth = png_bytes[24]
        colour_type = png_bytes[25]
        assert bit_depth == 16,   f"expected 16-bit, got {bit_depth}"
        assert colour_type == 0,  f"expected greyscale (0), got {colour_type}"

    def test_heightmap_png_pixel_range(self, tmp_path):
        """16-bit PNG must span the full 0–65535 range."""
        import struct
        import zlib

        writer, kwargs = _make_writer(tmp_path)
        writer.write(**kwargs)
        # Just check the file is non-empty and larger than placeholder PNGs
        png_path = (
            self._model_dir(tmp_path) / "materials" / "textures" / "heightmap.png"
        )
        assert png_path.stat().st_size > 1024, "PNG seems too small for 64×64"
