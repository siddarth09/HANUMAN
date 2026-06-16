# Copyright 2026 HANUMAN — MIT License.

"""Tests for MarsSite / ROI / BoundingBox configuration types."""

import pytest

from mars_terrain_exporter.utils.types import BoundingBox, ROI, MarsSite, _BASE_URL


class TestBoundingBox:
    def test_valid(self):
        BoundingBox(lat=18.44, lon=77.45, width_km=0.2, height_km=0.2).validate()

    def test_bad_lat(self):
        with pytest.raises(ValueError, match="lat"):
            BoundingBox(lat=120.0, lon=0.0).validate()

    def test_bad_width(self):
        with pytest.raises(ValueError, match="width_km"):
            BoundingBox(lat=0.0, lon=0.0, width_km=0.0).validate()


class TestROI:
    def test_default_center_crop(self):
        roi = ROI()
        roi.validate()
        assert roi.bounding_box is None and not roi.use_full

    def test_bad_size(self):
        with pytest.raises(ValueError, match="size_m"):
            ROI(size_m=-1).validate()

    def test_bad_resolution(self):
        with pytest.raises(ValueError, match="resolution_m"):
            ROI(resolution_m=0).validate()


class TestMarsSite:
    def test_from_catalog_by_name(self):
        site = MarsSite.from_catalog("jezero_c")
        assert site.site_code == "Jezero_C"
        assert site.name == "jezero_c"

    def test_dem_url(self):
        site = MarsSite.from_catalog("jezero_c")
        assert site.dem_url.startswith(_BASE_URL)
        assert site.dem_url.endswith(site.dem_filename)

    def test_from_catalog_unknown_raises(self):
        with pytest.raises(KeyError):
            MarsSite.from_catalog("nope")

    def test_invalid_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            MarsSite(site_code="X", name="bad name!", dem_filename="x.tif").validate()
