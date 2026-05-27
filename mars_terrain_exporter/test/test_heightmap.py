# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Tests for DEMProcessor — elevation read, projected-CRS bounds, ROI crop.

All tests use synthetic GeoTIFFs in ESRI:103885 (Mars Equidistant Cylindrical)
so they run without network access or a real HiRISE DTM.

ESRI:103885 is projected (metres).  ``src.res`` returns metres directly —
no degree-to-metre conversion is performed.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from mars_terrain_exporter.raster_processors.dem_processor import DEMProcessor
from mars_terrain_exporter.utils.types import BoundingBox, ROI
from mars_terrain_exporter.utils.raster_utils import normalize_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dem(
    tmp_path: Path,
    rows: int = 64,
    cols: int = 64,
    x_min: float = 0.0,
    y_min: float = 0.0,
    x_max: float = 6400.0,   # 100 m/px * 64 px → 6400 m
    y_max: float = 6400.0,
    crs_wkt: str | None = None,
    dtype: str = "float32",
    nodata: float | None = None,
) -> Path:
    """Create a minimal synthetic GeoTIFF in an approximately projected CRS."""
    # Use a simple projected CRS if ESRI:103885 is not available in the
    # test environment's pyproj/PROJ database.
    if crs_wkt is None:
        try:
            crs = CRS.from_authority("ESRI", "103885")
        except Exception:
            # Fallback to a generic equidistant CRS for unit testing
            crs = CRS.from_proj4(
                "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 "
                "+x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs"
            )
    else:
        crs = CRS.from_wkt(crs_wkt)

    data = np.linspace(100.0, 200.0, rows * cols, dtype=dtype).reshape(rows, cols)
    if nodata is not None:
        data[0, 0] = nodata

    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
    dem_path = tmp_path / "synthetic_dem.tif"

    with rasterio.open(
        dem_path, "w",
        driver="GTiff", height=rows, width=cols, count=1,
        dtype=dtype, crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)

    return dem_path


# ---------------------------------------------------------------------------
# _read_elevations
# ---------------------------------------------------------------------------

class TestReadElevations:
    def test_float_passthrough(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = DEMProcessor._read_elevations(raw, nodata=None)
        np.testing.assert_allclose(result, raw.astype(np.float64))

    def test_scale_and_offset(self):
        raw = np.array([[10, 20]], dtype=np.int16)
        result = DEMProcessor._read_elevations(raw, nodata=None, scale=0.5, offset=5.0)
        np.testing.assert_allclose(result, [[10.0, 15.0]])

    def test_nodata_becomes_nan(self):
        raw = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        result = DEMProcessor._read_elevations(raw, nodata=-9999.0)
        assert np.isnan(result[0, 1])
        assert not np.isnan(result[0, 0])

    def test_all_finite_without_nodata(self):
        raw = np.arange(9, dtype=np.float32).reshape(3, 3)
        result = DEMProcessor._read_elevations(raw, nodata=None)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# extract_from_raw — full ROI
# ---------------------------------------------------------------------------

class TestExtractFromRawFullROI:
    def test_shape_matches_raster(self, tmp_path):
        dem_path = _make_synthetic_dem(tmp_path, rows=32, cols=48)
        roi = ROI(use_full=True)
        elevs, emin, emax, bounds, profile = DEMProcessor.extract_from_raw(
            dem_path, roi
        )
        assert elevs.shape == (32, 48)

    def test_min_less_than_max(self, tmp_path):
        dem_path = _make_synthetic_dem(tmp_path)
        elevs, emin, emax, bounds, profile = DEMProcessor.extract_from_raw(
            dem_path, ROI(use_full=True)
        )
        assert emin < emax

    def test_bounds_width_km_correct(self, tmp_path):
        """6400 m extent → 6.4 km."""
        dem_path = _make_synthetic_dem(
            tmp_path, x_min=0.0, x_max=6400.0, y_min=0.0, y_max=6400.0
        )
        _, _, _, bounds, _ = DEMProcessor.extract_from_raw(
            dem_path, ROI(use_full=True)
        )
        assert bounds["width_km"] == pytest.approx(6.4, rel=1e-3)
        assert bounds["height_km"] == pytest.approx(6.4, rel=1e-3)

    def test_profile_contains_crs_and_transform(self, tmp_path):
        dem_path = _make_synthetic_dem(tmp_path)
        _, _, _, _, profile = DEMProcessor.extract_from_raw(
            dem_path, ROI(use_full=True)
        )
        assert "crs" in profile
        assert "transform" in profile

    def test_nodata_handled(self, tmp_path):
        dem_path = _make_synthetic_dem(tmp_path, nodata=-9999.0)
        elevs, emin, emax, _, _ = DEMProcessor.extract_from_raw(
            dem_path, ROI(use_full=True)
        )
        # The nodata pixel should become NaN and not affect emin/emax
        assert not np.isnan(emin)
        assert not np.isnan(emax)


# ---------------------------------------------------------------------------
# extract_from_raw — bounding-box ROI
# ---------------------------------------------------------------------------

class TestExtractFromRawBoundingBox:
    def test_cropped_shape_smaller_than_full(self, tmp_path):
        dem_path = _make_synthetic_dem(
            tmp_path, rows=128, cols=128,
            x_min=-64000.0, x_max=64000.0,
            y_min=-64000.0, y_max=64000.0,
        )
        bb = BoundingBox(lat=0.0, lon=0.0, width_km=20.0, height_km=20.0)
        roi = ROI(use_full=False, bounding_box=bb)
        elevs_full, _, _, _, _ = DEMProcessor.extract_from_raw(
            dem_path, ROI(use_full=True)
        )
        elevs_crop, _, _, bounds, _ = DEMProcessor.extract_from_raw(
            dem_path, roi
        )
        assert elevs_crop.shape[0] < elevs_full.shape[0]
        assert elevs_crop.shape[1] < elevs_full.shape[1]

    def test_cropped_bounds_approx_requested_size(self, tmp_path):
        dem_path = _make_synthetic_dem(
            tmp_path, rows=128, cols=128,
            x_min=-64000.0, x_max=64000.0,
            y_min=-64000.0, y_max=64000.0,
        )
        bb = BoundingBox(lat=0.0, lon=0.0, width_km=20.0, height_km=20.0)
        roi = ROI(use_full=False, bounding_box=bb)
        _, _, _, bounds, _ = DEMProcessor.extract_from_raw(dem_path, roi)
        # Allow ±2 px at 1000 m/px (synthetic DEM has large pixels)
        assert bounds["width_km"] == pytest.approx(20.0, abs=2.0)
        assert bounds["height_km"] == pytest.approx(20.0, abs=2.0)


# ---------------------------------------------------------------------------
# normalize_array
# ---------------------------------------------------------------------------

class TestNormalizeArray:
    def test_range_zero_to_one(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = normalize_array(data)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_uniform_array_returns_zeros(self):
        data = np.full((4, 4), 5.0)
        out = normalize_array(data)
        np.testing.assert_array_equal(out, np.zeros((4, 4)))

    def test_nan_becomes_zero(self):
        data = np.array([np.nan, 1.0, 2.0])
        out = normalize_array(data)
        assert out[0] == pytest.approx(0.0)
