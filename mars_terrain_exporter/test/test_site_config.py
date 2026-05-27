# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Tests for BoundingBox, ROI, and MarsSite dataclasses."""

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_terrain_exporter.utils.types import BoundingBox, ROI, MarsSite
from mars_terrain_exporter.cli import load_sites_from_yaml
from pathlib import Path
import tempfile
import yaml


class TestBoundingBox:
    def test_valid_jezero(self):
        bb = BoundingBox(lat=18.44, lon=77.45, width_km=5.0, height_km=5.0)
        bb.validate()  # should not raise

    def test_equator(self):
        bb = BoundingBox(lat=0.0, lon=180.0, width_km=1.0, height_km=1.0)
        bb.validate()

    def test_invalid_lat_too_high(self):
        with pytest.raises(ValueError, match="lat must be"):
            BoundingBox(lat=91.0, lon=0.0).validate()

    def test_invalid_lat_too_low(self):
        with pytest.raises(ValueError, match="lat must be"):
            BoundingBox(lat=-91.0, lon=0.0).validate()

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="width_km"):
            BoundingBox(lat=18.0, lon=77.0, width_km=0.0).validate()

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="height_km"):
            BoundingBox(lat=18.0, lon=77.0, height_km=-1.0).validate()

    def test_defaults(self):
        bb = BoundingBox(lat=18.44, lon=77.45)
        assert bb.width_km == 10.0
        assert bb.height_km == 10.0


class TestROI:
    def test_use_full_no_bb(self):
        roi = ROI(use_full=True)
        roi.validate()  # should not raise

    def test_use_partial_with_bb(self):
        bb = BoundingBox(lat=18.44, lon=77.45, width_km=3.0, height_km=3.0)
        roi = ROI(use_full=False, bounding_box=bb)
        roi.validate()

    def test_use_partial_no_bb_raises(self):
        roi = ROI(use_full=False, bounding_box=None)
        with pytest.raises(ValueError, match="bounding_box is required"):
            roi.validate()

    def test_default_use_full(self):
        roi = ROI()
        assert roi.use_full is True


class TestMarsSite:
    def test_from_catalog_jezero_c(self):
        site = MarsSite.from_catalog("jezero_c")
        assert site.site_code == "jezero_c"
        assert site.name == "jezero_c"
        assert "Jezero_C" in site.dem_url
        assert site.roi.use_full is True

    def test_from_catalog_jezero_dl(self):
        site = MarsSite.from_catalog("jezero_dl")
        assert "Jezero_DL" in site.dem_url

    def test_custom_roi(self):
        roi = ROI(
            use_full=False,
            bounding_box=BoundingBox(lat=18.44, lon=77.45, width_km=3.0),
        )
        site = MarsSite.from_catalog("jezero_c", roi=roi)
        assert site.roi.use_full is False
        assert site.roi.bounding_box.width_km == 3.0

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="name must be"):
            MarsSite(
                site_code="x", name="has spaces!", dem_url="https://a.tif"
            ).validate()

    def test_empty_site_code_raises(self):
        with pytest.raises(ValueError, match="site_code"):
            MarsSite(site_code="", name="valid", dem_url="https://a.tif").validate()

    def test_empty_dem_url_raises(self):
        with pytest.raises(ValueError, match="dem_url"):
            MarsSite(site_code="x", name="valid", dem_url="").validate()

    def test_unknown_catalog_entry_raises(self):
        with pytest.raises(KeyError):
            MarsSite.from_catalog("gale_crater")


class TestLoadSitesFromYaml:
    def _write_yaml(self, data: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(data, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_load_two_jezero_sites(self):
        cfg = {"sites": [{"site": "jezero_c"}, {"site": "jezero_dl"}]}
        path = self._write_yaml(cfg)
        sites = load_sites_from_yaml(path)
        assert len(sites) == 2
        assert sites[0].name == "jezero_c"
        assert sites[1].name == "jezero_dl"

    def test_unknown_site_skipped_with_warning(self, capsys):
        cfg = {"sites": [{"site": "jezero_c"}, {"site": "does_not_exist"}]}
        path = self._write_yaml(cfg)
        sites = load_sites_from_yaml(path)
        assert len(sites) == 1
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_roi_bounding_box_parsed(self):
        cfg = {
            "sites": [
                {
                    "site": "jezero_c",
                    "roi": {
                        "use_full": False,
                        "bounding_box": {
                            "lat": 18.44,
                            "lon": 77.45,
                            "width_km": 4.0,
                            "height_km": 4.0,
                        },
                    },
                }
            ]
        }
        path = self._write_yaml(cfg)
        sites = load_sites_from_yaml(path)
        assert len(sites) == 1
        assert sites[0].roi.use_full is False
        assert sites[0].roi.bounding_box.lat == pytest.approx(18.44)
