# Copyright 2026 HANUMAN — MIT License.

"""Tests for CLI argument parsing and ROI construction."""

import pytest

from mars_terrain_exporter.cli import build_parser, _roi_from_args, load_sites_from_yaml


def _parse(argv):
    return build_parser().parse_args(argv)


class TestParser:
    def test_no_command(self):
        assert _parse([]).command is None

    def test_site_defaults(self):
        args = _parse(["site", "jezero_c"])
        assert args.command == "site"
        assert args.site_name == "jezero_c"
        assert args.size == 100.0
        assert args.resolution == 0.1
        assert args.output_dir == "./models"

    def test_site_geographic(self):
        args = _parse(["site", "jezero_c", "--lat", "18.4", "--lon", "77.4"])
        assert args.lat == 18.4 and args.lon == 77.4

    def test_batch_requires_config(self):
        with pytest.raises(SystemExit):
            _parse(["batch"])


class TestRoiFromArgs:
    def test_center_crop(self):
        roi = _roi_from_args(_parse(["site", "jezero_c", "--size", "200"]))
        assert roi.size_m == 200 and roi.bounding_box is None and not roi.use_full

    def test_geographic_crop(self):
        roi = _roi_from_args(
            _parse(["site", "jezero_c", "--lat", "18.4", "--lon", "77.4",
                    "--width", "0.3", "--height", "0.3"])
        )
        assert roi.bounding_box is not None
        assert roi.bounding_box.width_km == 0.3

    def test_full(self):
        roi = _roi_from_args(_parse(["site", "jezero_c", "--full"]))
        assert roi.use_full


class TestYamlConfig:
    def test_load(self, tmp_path):
        cfg = tmp_path / "sites.yaml"
        cfg.write_text(
            "sites:\n"
            "  - site: jezero_c\n"
            "    roi:\n"
            "      size_m: 120.0\n"
            "  - site: jezero_w\n"
        )
        sites = load_sites_from_yaml(cfg)
        assert [s.name for s in sites] == ["jezero_c", "jezero_w"]
        assert sites[0].roi.size_m == 120.0

    def test_skips_bad_entries(self, tmp_path):
        cfg = tmp_path / "sites.yaml"
        cfg.write_text("sites:\n  - site: not_a_site\n  - site: jezero_c\n")
        sites = load_sites_from_yaml(cfg)
        assert [s.name for s in sites] == ["jezero_c"]
