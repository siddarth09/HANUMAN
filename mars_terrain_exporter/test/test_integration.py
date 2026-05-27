# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Integration tests for MarsTerrainExporter.

Mocks FileDownloader and DEMProcessor so no network access or real HiRISE
DTMs are required.  Verifies the full pipeline produces all expected output
files under the correct directory structure.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from mars_terrain_exporter.mars_terrain_exporter import MarsTerrainExporter
from mars_terrain_exporter.utils.types import MarsSite, ROI


def _make_fake_dem_result():
    """Return (elevations, emin, emax, bounds, dem_profile) for a 64×64 patch."""
    elevs = np.linspace(-500.0, 200.0, 64 * 64).reshape(64, 64)
    try:
        crs = CRS.from_authority("ESRI", "103885")
    except Exception:
        crs = CRS.from_proj4(
            "+proj=eqc +lat_ts=0 +lon_0=0 +a=3396190 +b=3376200 +units=m"
        )
    transform = from_bounds(0, 0, 6400, 6400, 64, 64)
    return (
        elevs,
        float(elevs.min()),
        float(elevs.max()),
        {"center_lat": 18.44, "center_lon": 77.45, "width_km": 6.4, "height_km": 6.4},
        {"crs": crs, "transform": transform},
    )


class TestMarsTerrainExporterIntegration:
    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self._output_dir = Path(self._tmp)

    def _run_export(self, site_name: str = "jezero_c") -> Path:
        site = MarsSite.from_catalog(site_name, roi=ROI(use_full=True))
        fake_dem = Path(self._tmp) / "fake_dem.tif"
        fake_dem.touch()

        with (
            patch(
                "mars_terrain_exporter.mars_terrain_exporter.FileDownloader.download",
                return_value=fake_dem,
            ),
            patch(
                "mars_terrain_exporter.mars_terrain_exporter.DEMProcessor.extract_from_raw",
                return_value=_make_fake_dem_result(),
            ),
        ):
            exporter = MarsTerrainExporter(self._output_dir)
            return exporter.export_model(site)

    def test_returns_model_dir_path(self):
        result = self._run_export("jezero_c")
        assert result == self._output_dir / "jezero_c"

    def test_model_sdf_created(self):
        self._run_export("jezero_c")
        assert (self._output_dir / "jezero_c" / "model.sdf").exists()

    def test_world_sdf_created(self):
        self._run_export("jezero_c")
        assert (self._output_dir / "jezero_c" / "world.sdf").exists()

    def test_model_config_created(self):
        self._run_export("jezero_c")
        assert (self._output_dir / "jezero_c" / "model.config").exists()

    def test_metadata_yaml_created(self):
        self._run_export("jezero_c")
        assert (self._output_dir / "jezero_c" / "metadata.yaml").exists()

    def test_heightmap_tif_created(self):
        self._run_export("jezero_c")
        assert (
            self._output_dir / "jezero_c" / "materials" / "textures" / "heightmap.tif"
        ).exists()

    def test_jezero_dl_export(self):
        self._run_export("jezero_dl")
        assert (self._output_dir / "jezero_dl" / "model.sdf").exists()
        assert (self._output_dir / "jezero_dl" / "world.sdf").exists()

    def test_world_sdf_has_bullet_featherstone(self):
        self._run_export("jezero_c")
        model_dir = (self._output_dir / "jezero_c").resolve()
        world = model_dir.with_name("jezero_c") / "world.sdf"
        # resolve() inside the writer means we need to read from the resolved path
        world_text = (self._output_dir / "jezero_c" / "world.sdf").read_text()
        assert "bullet_featherstone" in world_text
        assert "0.005" in world_text
        # URI must be file://, not model:// (no GZ_SIM_RESOURCE_PATH needed)
        assert "file://" in world_text
        assert "model://jezero_c" not in world_text

    def test_model_sdf_static(self):
        self._run_export("jezero_c")
        sdf = (self._output_dir / "jezero_c" / "model.sdf").read_text()
        assert "<static>true</static>" in sdf
