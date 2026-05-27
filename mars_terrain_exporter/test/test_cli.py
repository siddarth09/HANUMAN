# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0

"""Tests for the CLI argument parser and subcommands."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from mars_terrain_exporter.cli import build_parser
from mars_terrain_exporter.utils.types import MarsSite


class TestBuildParser:
    def setup_method(self):
        self.parser = build_parser()

    def test_no_subcommand_returns_none(self):
        args = self.parser.parse_args([])
        assert args.command is None

    def test_site_subcommand(self):
        args = self.parser.parse_args(["site", "jezero_c"])
        assert args.command == "site"
        assert args.site_name == "jezero_c"

    def test_site_defaults(self):
        args = self.parser.parse_args(["site", "jezero_dl"])
        assert args.lat is None
        assert args.lon is None
        assert args.width == pytest.approx(10.0)
        assert args.height == pytest.approx(10.0)
        assert args.output_dir == "."

    def test_site_with_roi(self):
        args = self.parser.parse_args([
            "site", "jezero_c",
            "--lat", "18.44", "--lon", "77.45",
            "--width", "3", "--height", "3",
            "--output-dir", "/tmp/models",
        ])
        assert args.lat == pytest.approx(18.44)
        assert args.lon == pytest.approx(77.45)
        assert args.width == pytest.approx(3.0)
        assert args.height == pytest.approx(3.0)
        assert args.output_dir == "/tmp/models"

    def test_batch_subcommand(self):
        args = self.parser.parse_args([
            "batch", "--config", "config/jezero_sites.yaml",
        ])
        assert args.command == "batch"
        assert args.config == "config/jezero_sites.yaml"

    def test_batch_output_dir_default(self):
        args = self.parser.parse_args(["batch", "--config", "foo.yaml"])
        assert args.output_dir == "."


class TestFromCatalog:
    def test_from_catalog_by_name(self):
        site = MarsSite.from_catalog("jezero_c")
        assert site.name == "jezero_c"

    def test_from_catalog_by_code(self):
        site = MarsSite.from_catalog("jezero_c")
        assert site.site_code == "jezero_c"

    def test_from_catalog_jezero_dl(self):
        site = MarsSite.from_catalog("jezero_dl")
        assert "Jezero_DL" in site.dem_url

    def test_invalid_site_raises(self):
        with pytest.raises(KeyError):
            MarsSite.from_catalog("shackleton_rim")  # lunar site, not Mars
